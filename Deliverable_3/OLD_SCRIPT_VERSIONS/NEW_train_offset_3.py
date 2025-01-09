import sys
import json
import os
import shutil
import pickle
import threading
import numpy as np
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf, split, window, from_unixtime
from pyspark.sql.types import StructType, StringType, DoubleType, LongType, TimestampType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from datetime import datetime

# ==========================
# Configuration Parameters
# ==========================

# Define symbols and their corresponding feature columns
symbols_features = {
    "BP": ["volume", "volatility", "market_sentiment", "trading_activity", "price"],
    "COP": ["volume", "volatility", "market_sentiment", "trading_activity", "price"],
    "SHEL": ["volume", "volatility", "market_sentiment", "trading_activity", "price"],
    "XOM": ["volume", "volatility", "market_sentiment", "trading_activity", "price"],
    "ETHEREUM": ["bid", "ask", "spread_raw", "spread_table", "price"],
}

# ==========================
# Parameterization
# ==========================

# Set the target symbol for this instance from command-line arguments
if len(sys.argv) != 2:
    raise ValueError("Please provide the TARGET_SYMBOL as a command-line argument.")

TARGET_SYMBOL = sys.argv[1]

if TARGET_SYMBOL not in symbols_features:
    raise ValueError(f"TARGET_SYMBOL '{TARGET_SYMBOL}' is not defined in symbols_features.")

# Retrieve feature columns for the target symbol
FEATURE_COLUMNS = symbols_features[TARGET_SYMBOL]
NUM_FEATURES = len(FEATURE_COLUMNS)

# ==========================
# Model and Checkpoint Paths
# ==========================

# Model persistence path
MODEL_BASE_PATH = f"./models/{TARGET_SYMBOL}"

# Checkpoint locations for Spark Structured Streaming
CHECKPOINT_PATH_AGG = f"./checkpoint/dir/{TARGET_SYMBOL}_agg"
CHECKPOINT_PATH_PRED = f"./checkpoint/dir/{TARGET_SYMBOL}_pred"

# Ensure that the directories exist
os.makedirs(MODEL_BASE_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH_AGG, exist_ok=True)
os.makedirs(CHECKPOINT_PATH_PRED, exist_ok=True)

# ==========================
# Cassandra Configuration
# ==========================

CASSANDRA_KEYSPACE = "stream_predictions"  # Replace with your keyspace
CASSANDRA_TABLE = "model_predictions"  # Replace with your table name

# ==========================
# Kafka Configuration
# ==========================

KAFKA_BROKERS = "localhost:9092"
KAFKA_TOPIC = "model-topic"

# ==========================
# Hyperparameters for Model Training
# ==========================

MAX_ITER = 50  # Number of iterations for model fitting
REG_PARAM = 0.5
ELASTIC_NET_PARAM = 0.5

# ==========================
# Initialize Spark Session
# ==========================

spark_port_offset = {
    "ETHEREUM": 0,
    "SHEL": 1,
    "BP": 2,
    "COP": 3,
    "XOM": 4
}

spark = SparkSession.builder \
    .appName(f"ContinuousTrainingLinearRegression_{TARGET_SYMBOL}") \
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
            "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0") \
    .config("spark.cassandra.connection.host", "localhost") \
    .config("spark.ui.port", str(4050 + spark_port_offset[TARGET_SYMBOL])) \
    .getOrCreate()


# spark = SparkSession.builder \
#     .appName(f"ContinuousTrainingLinearRegression_{TARGET_SYMBOL}") \
#     .config("spark.jars.packages",
#             "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
#             "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0") \
#     .config("spark.cassandra.connection.host", "localhost") \
#     .config("spark.ui.port", str(4050 + spark_port_offset[TARGET_SYMBOL])) \
#     .config("spark.blockManager.port", str(6060 + spark_port_offset[TARGET_SYMBOL]))\
#     .config("spark.executor.memory", "2g")\
#     .config("spark.driver.memory", "1g")\
#     .config("spark.executor.cores", "1")\
#     .getOrCreate()

# spark = SparkSession.builder \
#     .appName(f"ContinuousTrainingLinearRegression_{TARGET_SYMBOL}") \
#     .config("spark.jars.packages",
#             "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
#             "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0") \
#     .config("spark.cassandra.connection.host", "localhost") \
#     .config("spark.ui.port", str(4050 + spark_port_offset[TARGET_SYMBOL])) \
#     .config("spark.executor.memory", "2g")\
#     .config("spark.driver.memory", "1g")\
#     .config("spark.executor.cores", "1")\
#     .getOrCreate()

# Set log level to ERROR to reduce verbosity
spark.sparkContext.setLogLevel("ERROR")

# ==========================
# Define Schema for Kafka Data
# ==========================

schema = StructType() \
    .add("symbol", StringType()) \
    .add("timestamp", LongType()) \
    .add("source", StringType()) \
    .add("data_type", StringType()) \
    .add("bid", DoubleType()) \
    .add("ask", DoubleType()) \
    .add("price", DoubleType()) \
    .add("volume", DoubleType()) \
    .add("spread_raw", DoubleType()) \
    .add("spread_table", DoubleType()) \
    .add("volatility", DoubleType()) \
    .add("market_sentiment", DoubleType()) \
    .add("trading_activity", DoubleType())

# ==========================
# Read Streaming Data from Kafka
# ==========================

df_kafka = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BROKERS) \
    .option("subscribe", KAFKA_TOPIC) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .option("kafkaConsumer.pollTimeoutMs", 180000) \
    .load()

# Parse the JSON data
parsed_df = df_kafka.selectExpr("CAST(value AS STRING) as json") \
    .select(from_json(col("json"), schema).alias("data")) \
    .select("data.*")

# ==========================
# Filter for Target Symbol
# ==========================

filtered_df = parsed_df.filter(col("symbol") == TARGET_SYMBOL)

# ==========================
# Feature Engineering
# ==========================

# Set the label column to "price" for all symbols
LABEL_COLUMN = "price"

# Select feature columns and label
processed_df = filtered_df.select(
    col("symbol").alias("symbol"),
    *[col(feature).cast("double").alias(f"{feature}") for feature in FEATURE_COLUMNS],
    col(LABEL_COLUMN).cast("double").alias("label"),
    col("timestamp").cast("long").alias("timestamp")  # Include timestamp for Cassandra or other sinks
)

# ==========================
# Correct Timestamp Conversion
# ==========================

# Convert the timestamp to TimestampType (from milliseconds to seconds)
processed_df = processed_df.withColumn(
    "event_time",
    (col("timestamp") / 1000).cast(TimestampType())
)

# ==========================
# AGGREGATOR STREAM
# ==========================
# This stream aggregates data in 1-minute windows and updates the model once per minute.

# Define a 1-minute window with a 2-minute watermark
windowed_df = processed_df \
    .withWatermark("event_time", "2 minutes") \
    .groupBy(
        window(col("event_time"), "1 minute"),
        col("symbol")
    ) \
    .agg(
        *[
            F.avg(col(feature)).alias(f"avg_{feature}") 
            for feature in FEATURE_COLUMNS
        ],
        F.last(col("label")).alias("label")  # Use the last price in the window as label
    )

aggregated_feature_columns = [f"avg_{feature}" for feature in FEATURE_COLUMNS]

assembler_agg = VectorAssembler(
    inputCols=aggregated_feature_columns,
    outputCol="features"
)

windowed_features_df = assembler_agg.transform(windowed_df).select(
    "symbol",
    "features",
    "label",
    col("window.start").alias("window_start"),
    col("window.end").alias("window_end")
)

# ==========================
# Initialize or Load Linear Regression Model
# ==========================

def load_model():
    """
    Loads the latest LinearRegressionModel from disk if it exists.
    """
    latest_model_path = os.path.join(MODEL_BASE_PATH, f"latest_model.txt")
    if os.path.exists(latest_model_path):
        try:
            with open(latest_model_path, "r") as f:
                model_path = f.read().strip()
            model = LinearRegressionModel.load(model_path)
            print(f"[{TARGET_SYMBOL}] Loaded existing Linear Regression model from {model_path}.")
            return model
        except Exception as e:
            print(f"[{TARGET_SYMBOL}] Error loading model: {e}")
    return None

def save_model(model):
    """
    Saves the LinearRegressionModel to disk and updates the latest model reference.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_BASE_PATH, f"model_{timestamp}")
    model.write().overwrite().save(model_path)

    latest_model_path = os.path.join(MODEL_BASE_PATH, "latest_model.txt")
    with open(latest_model_path, "w") as f:
        f.write(model_path)

    print(f"[{TARGET_SYMBOL}] Saved model to {model_path} and updated latest reference.")

# Keep a global reference to the model for aggregator
lr_model = load_model()

def update_model(batch_df: DataFrame, batch_id: int):
    """
    update_model is called for each micro-batch in the aggregator stream.
    We only expect new aggregated data about once a minute (due to trigger).
    This function re-trains or updates the model and saves it to disk.
    """
    global lr_model

    if batch_df.rdd.isEmpty():
        print(f"[{TARGET_SYMBOL}] Batch {batch_id} is empty. Skipping model update.")
        return

    batch_df.show(truncate=False)

    try:
        lr = LinearRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=MAX_ITER,
            regParam=REG_PARAM,
            elasticNetParam=ELASTIC_NET_PARAM
        )
        new_model = lr.fit(batch_df)
        lr_model = new_model

        print(f"[{TARGET_SYMBOL}] Updated Linear Regression model with batch {batch_id}")
        print(f"[{TARGET_SYMBOL}] New Coefficients: {lr_model.coefficients}")
        print(f"[{TARGET_SYMBOL}] New Intercept: {lr_model.intercept}")

        save_model(lr_model)

    except Exception as e:
        print(f"[{TARGET_SYMBOL}] Error in batch {batch_id}: {e}")
        raise e

# ==========================
# Start the AGGREGATOR STREAM (Query 1)
# ==========================

query_agg = windowed_features_df.writeStream \
    .foreachBatch(update_model) \
    .outputMode("update") \
    .option("checkpointLocation", CHECKPOINT_PATH_AGG) \
    .trigger(processingTime='1 minute') \
    .start()

print(f"[{TARGET_SYMBOL}] Started aggregator query (model update).")

# ==========================
# PREDICTOR STREAM
# ==========================

assembler_pred = VectorAssembler(
    inputCols=FEATURE_COLUMNS,
    outputCol="features"
)

processed_df_for_pred = assembler_pred.transform(
    processed_df.select("symbol", *FEATURE_COLUMNS, "label", "timestamp", "event_time")
).select(
    "symbol",
    "features",
    "label",
    "timestamp",
    "event_time"
)

def predict_incoming(batch_df: DataFrame, batch_id: int):
    """
    For each micro-batch in the predictor stream, load the current model from disk
    and apply it to new records.
    """
    if batch_df.rdd.isEmpty():
        print(f"[{TARGET_SYMBOL}] Predictor micro-batch {batch_id} is empty. Skipping predictions.")
        return

    print(f"[{TARGET_SYMBOL}] Predictor micro-batch {batch_id} received {batch_df.count()} records.")

    current_model = load_model()
    if current_model is None:
        print(f"[{TARGET_SYMBOL}] No model available yet for predictions. Skipping batch {batch_id}.")
        return

    predictions = current_model.transform(batch_df)
    print("finished predictions")

    predictions_to_save = predictions.select(
        "symbol",
        "timestamp",
        "event_time",
        "features",
        "label",
        "prediction"
    )

    @udf(StringType())
    def features_to_json(features):
        return json.dumps({
            f: float(value) for f, value in zip(FEATURE_COLUMNS, features)
        })

    predictions_to_save = predictions_to_save.withColumn("input_data", features_to_json(col("features")))

    # predictions_to_save.show(5, truncate=False)

    print("Writing to cassanrda")
    predictions_to_save.select(
        col("symbol"),
        col("timestamp"),
        col("event_time"),
        col("input_data"),
        col("prediction"),
        col("label")
    ).write \
     .format("org.apache.spark.sql.cassandra") \
     .mode("append") \
     .options(table=CASSANDRA_TABLE, keyspace=CASSANDRA_KEYSPACE) \
     .save()

    print(f"[{TARGET_SYMBOL}] Completed predictions for micro-batch {batch_id}.")

# ==========================
# Start the PREDICTOR STREAM (Query 2)
# ==========================

query_pred = processed_df_for_pred.writeStream \
    .foreachBatch(predict_incoming) \
    .outputMode("append") \
    .option("checkpointLocation", CHECKPOINT_PATH_PRED) \
    .start()

print(f"[{TARGET_SYMBOL}] Started predictor query (real-time predictions).")

# ==========================
# Await Termination
# ==========================

spark.streams.awaitAnyTermination()
