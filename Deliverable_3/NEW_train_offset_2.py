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
MODEL_PATH = f"./models/linear_regression_model_{TARGET_SYMBOL}"

# Checkpoint locations for Spark Structured Streaming
CHECKPOINT_PATH_AGG = f"./checkpoint/dir/{TARGET_SYMBOL}_agg"
CHECKPOINT_PATH_PRED = f"./checkpoint/dir/{TARGET_SYMBOL}_pred"

# Ensure that the directories exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
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
REG_PARAM = 0.1
ELASTIC_NET_PARAM = 0.5

# ==========================
# Initialize Spark Session
# ==========================



spark_port_offset = {
    "ETHEREUM" : 0,
    "SHEL" : 1,
    "BP" : 2,
    "COP" : 3,
    "XOM" : 4
}

spark = SparkSession.builder \
    .appName(f"ContinuousTrainingLinearRegression_{TARGET_SYMBOL}") \
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
            "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0") \
    .config("spark.cassandra.connection.host", "localhost") \
    .config("spark.ui.port", str(4050 + spark_port_offset[TARGET_SYMBOL])) \
    .getOrCreate()

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

def load_model(model_path: str) -> LinearRegressionModel:
    """
    Loads a LinearRegressionModel from disk if it exists, else returns None.
    """
    if os.path.exists(model_path):
        try:
            model = LinearRegressionModel.load(model_path)
            print(f"[{TARGET_SYMBOL}] Loaded existing Linear Regression model from {model_path}.")
            return model
        except Exception as e:
            print(f"[{TARGET_SYMBOL}] Failed to load model from {model_path}: {e}")
            print(f"[{TARGET_SYMBOL}] Removing corrupted model directory.")
            shutil.rmtree(model_path)
            return None
    else:
        print(f"[{TARGET_SYMBOL}] No existing model found at {model_path}. Will initialize on the first batch.")
        return None


# Keep a global reference to the model for aggregator
lr_model = load_model(MODEL_PATH)


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
        # If no model is loaded yet, create a new LR and fit
        if lr_model is None:
            lr = LinearRegression(
                featuresCol="features",
                labelCol="label",
                maxIter=MAX_ITER,
                regParam=REG_PARAM,
                elasticNetParam=ELASTIC_NET_PARAM
            )
            lr_model = lr.fit(batch_df)
            print(f"[{TARGET_SYMBOL}] Initialized Linear Regression model with batch {batch_id}")

            # Print coefficients and intercept
            print(f"[{TARGET_SYMBOL}] Coefficients after initialization: {lr_model.coefficients}")
            print(f"[{TARGET_SYMBOL}] Intercept after initialization: {lr_model.intercept}")
        else:
            # Retrain a brand new model on this batch
            lr = LinearRegression(
                featuresCol="features",
                labelCol="label",
                maxIter=MAX_ITER,
                regParam=REG_PARAM,
                elasticNetParam=ELASTIC_NET_PARAM
            )
            new_model = lr.fit(batch_df)

            # However, PySpark doesn't support setting these directly, so let's just use new_model
            # or you can adapt if you have a specific approach to incremental learning
            lr_model = new_model

            print(f"[{TARGET_SYMBOL}] Updated Linear Regression model with batch {batch_id}")
            print(f"[{TARGET_SYMBOL}] New Coefficients: {lr_model.coefficients}")
            print(f"[{TARGET_SYMBOL}] New Intercept: {lr_model.intercept}")

        # Persist the updated model to disk so the predictor stream can load it
        if lr_model is not None:
            lr_model.write().overwrite().save(MODEL_PATH)
            print(f"[{TARGET_SYMBOL}] Model saved after batch {batch_id}")

        # Optional: you could also log or store aggregator-based predictions here
        # predictions = lr_model.transform(batch_df)
        # do something with aggregator predictions if desired

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
# This stream uses the latest saved model to predict on *each* micro-batch of data,
# so you get near real-time predictions on all new events (rather than waiting for the 1-minute window).

# For real-time predictions, we do NOT aggregate. We just transform the raw features (non-aggregated).
# We'll rename some columns to keep it consistent with the aggregator approach,
# but you can just feed the raw columns to a VectorAssembler.

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
    and apply it to new records. This can happen as frequently as Spark's micro-batch allows.
    """
    if batch_df.rdd.isEmpty():
        print(f"[{TARGET_SYMBOL}] Predictor micro-batch {batch_id} is empty. Skipping predictions.")
        return

    print(f"[{TARGET_SYMBOL}] Predictor micro-batch {batch_id} received {batch_df.count()} records.")

    # Try loading the latest model from disk with retries in case the aggregator is writing
    current_model = None
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            current_model = load_model(MODEL_PATH)
            if current_model is not None:
                # Successfully loaded a valid model
                break
        except Exception as e:
            # If load_model raises an exception, catch it and retry
            print(
                f"[{TARGET_SYMBOL}] Attempt {attempt}/{max_retries} to load model failed: {str(e)}"
            )
        # If we get here, either current_model is still None or load_model raised an exception
        if attempt < max_retries:
            print(f"[{TARGET_SYMBOL}] Waiting 3 seconds before retrying...")
            time.sleep(3)

    if current_model is None:
        # If no model is available after all attempts, skip
        print(f"[{TARGET_SYMBOL}] No model available yet for predictions. Skipping batch {batch_id}.")
        return

    # Make predictions
    predictions = current_model.transform(batch_df)

    # You can select columns to store or log
    # For example, let's just print or store them:
    predictions_to_save = predictions.select(
        "symbol",
        "timestamp",
        "event_time",
        "features",
        "label",
        "prediction"
    )

    # Example: Convert features vector to JSON string for storage/logging
    @udf(StringType())
    def features_to_json(features):
        # For convenience, map each feature column name to value
        return json.dumps({
            f: float(value)
            for f, value in zip(FEATURE_COLUMNS, features)
        })

    predictions_to_save = predictions_to_save.withColumn("input_data", features_to_json(col("features")))

    # Show a few predictions for debugging (comment out in production)
    predictions_to_save.show(5, truncate=False)

    # If you want, write the predictions to Cassandra (uncomment and configure):
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
