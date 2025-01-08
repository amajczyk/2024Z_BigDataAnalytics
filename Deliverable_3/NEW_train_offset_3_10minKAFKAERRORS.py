import sys
import json
import os
import shutil
import pickle
import threading
import numpy as np
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf, split, window, from_unixtime, avg
from pyspark.sql.types import StructType, StringType, DoubleType, LongType, TimestampType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from datetime import datetime

# Configuration Parameters
PREFIX = "10m"
symbols_features = {
    "BP": ["volume", "volatility", "market_sentiment", "trading_activity", "price"],
    "COP": ["volume", "volatility", "market_sentiment", "trading_activity", "price"],
    "SHEL": ["volume", "volatility", "market_sentiment", "trading_activity", "price"],
    "XOM": ["volume", "volatility", "market_sentiment", "trading_activity", "price"],
    "ETHEREUM": ["bid", "ask", "spread_raw", "spread_table", "price"],
}

if len(sys.argv) != 2:
    raise ValueError("Please provide the TARGET_SYMBOL as a command-line argument.")

TARGET_SYMBOL = sys.argv[1]

if TARGET_SYMBOL not in symbols_features:
    raise ValueError(f"TARGET_SYMBOL '{TARGET_SYMBOL}' is not defined in symbols_features.")

FEATURE_COLUMNS = symbols_features[TARGET_SYMBOL]
NUM_FEATURES = len(FEATURE_COLUMNS)

# Model and Checkpoint Paths with PREFIX
MODEL_BASE_PATH = f"./{PREFIX}/models/{TARGET_SYMBOL}"
CHECKPOINT_PATH_AGG = f"./{PREFIX}/checkpoint/dir/{TARGET_SYMBOL}_agg"
CHECKPOINT_PATH_PRED = f"./{PREFIX}/checkpoint/dir/{TARGET_SYMBOL}_pred"

os.makedirs(MODEL_BASE_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH_AGG, exist_ok=True)
os.makedirs(CHECKPOINT_PATH_PRED, exist_ok=True)

CASSANDRA_KEYSPACE = "stream_predictions"
CASSANDRA_TABLE = "model_predictions_10m"

KAFKA_BROKERS = "localhost:9092"
KAFKA_TOPIC = "model-topic"

MAX_ITER = 50
REG_PARAM = 0.01
ELASTIC_NET_PARAM = 0.5


# Add this near the top of your configuration
MAX_OFFSETS_PER_TRIGGER = 10000
MAX_TRIGGER_DELAY = "5 minutes"

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

spark.sparkContext.setLogLevel("ERROR")

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

df_kafka = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BROKERS) \
    .option("subscribe", KAFKA_TOPIC) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .option("kafkaConsumer.pollTimeoutMs", 180000) \
    .option("kafka.request.timeout.ms", "300000") \
    .option("kafka.session.timeout.ms", "180000") \
    .option("kafka.heartbeat.interval.ms", "60000") \
    .option("maxOffsetsPerTrigger", "10000") \
    .load()

parsed_df = df_kafka.selectExpr("CAST(value AS STRING) as json") \
    .select(from_json(col("json"), schema).alias("data")) \
    .select("data.*")

filtered_df = parsed_df.filter(col("symbol") == TARGET_SYMBOL)

LABEL_COLUMN = "price"

processed_df = filtered_df.select(
    col("symbol").alias("symbol"),
    *[col(feature).cast("double").alias(f"{feature}") for feature in FEATURE_COLUMNS],
    col(LABEL_COLUMN).cast("double").alias("label"),
    col("timestamp").cast("long").alias("timestamp")
)

processed_df = processed_df.withColumn(
    "event_time",
    (col("timestamp") / 1000).cast(TimestampType())
)

# Modified to use 10-minute windows
windowed_df = processed_df \
    .withWatermark("event_time", "20 minutes") \
    .groupBy(
        window(col("event_time"), "10 minutes"),
        col("symbol")
    ) \
    .agg(
        *[
            F.avg(col(feature)).alias(f"avg_{feature}") 
            for feature in FEATURE_COLUMNS
        ],
        F.avg(col("label")).alias("label")  # Use 10-minute average as label
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

def load_model():
    latest_model_path = os.path.join(MODEL_BASE_PATH, f"latest_model.txt")
    if os.path.exists(latest_model_path):
        try:
            with open(latest_model_path, "r") as f:
                model_path = f.read().strip()
            model = LinearRegressionModel.load(model_path)
            print(f"[{TARGET_SYMBOL}] Loaded model from {model_path}")
            return model
        except Exception as e:
            print(f"[{TARGET_SYMBOL}] Error loading model: {e}")
    return None

def save_model(model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_BASE_PATH, f"model_{timestamp}")
    model.write().overwrite().save(model_path)

    latest_model_path = os.path.join(MODEL_BASE_PATH, "latest_model.txt")
    with open(latest_model_path, "w") as f:
        f.write(model_path)


def write_to_cassandra_with_retry(df, max_retries=3):
    for attempt in range(max_retries):
        try:
            df.write \
                .format("org.apache.spark.sql.cassandra") \
                .mode("append") \
                .options(table=CASSANDRA_TABLE, keyspace=CASSANDRA_KEYSPACE) \
                .save()
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"[{TARGET_SYMBOL}] Cassandra write failed, attempt {attempt + 1} of {max_retries}")
            time.sleep(5)  # Wait before retry


lr_model = load_model()

def update_model(batch_df: DataFrame, batch_id: int):
    global lr_model
    
    try:
        if batch_df.rdd.isEmpty():
            print(f"[{TARGET_SYMBOL}] Batch {batch_id} is empty. Skipping update.")
            return

        start_time = time()

        # Add checkpoint to save intermediate state
        batch_df.persist()  # Cache the DataFrame
        
        if batch_df.count() < 2:  # Minimum required for regression
            print(f"[{TARGET_SYMBOL}] Insufficient data in batch {batch_id}")
            batch_df.unpersist()
            return

        lr = LinearRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=MAX_ITER,
            regParam=REG_PARAM,
            elasticNetParam=ELASTIC_NET_PARAM
        )
        
        new_model = lr.fit(batch_df)
        lr_model = new_model
        update_time = time() - start_time

        print(f"[{TARGET_SYMBOL}] Updated model with batch {batch_id}")
        print(f"[{TARGET_SYMBOL}] Model update took {update_time:.5f}")
        print(f"[{TARGET_SYMBOL}] Coefficients: {lr_model.coefficients}")
        print(f"[{TARGET_SYMBOL}] Intercept: {lr_model.intercept}")

        save_model(lr_model)
        batch_df.unpersist()

    except Exception as e:
        print(f"[{TARGET_SYMBOL}] Error in batch {batch_id}: {str(e)}")
        if batch_df is not None:
            batch_df.unpersist()

# Start aggregator stream with 10-minute trigger
query_agg = windowed_features_df.writeStream \
    .foreachBatch(update_model) \
    .outputMode("update") \
    .option("checkpointLocation", CHECKPOINT_PATH_AGG) \
    .trigger(processingTime='10 minutes') \
    .start()

print(f"[{TARGET_SYMBOL}] Started aggregator query")

# Predictor stream setup
assembler_pred = VectorAssembler(
    inputCols=FEATURE_COLUMNS,
    outputCol="features"
)

# Process each record immediately for prediction
processed_df_for_pred = assembler_pred.transform(
    processed_df.select("symbol", *FEATURE_COLUMNS, "label", "timestamp", "event_time")
).select(
    "symbol",
    "features",
    "label",
    "timestamp",
    "event_time"
)

# def predict_incoming(batch_df: DataFrame, batch_id: int):
#     if batch_df.rdd.isEmpty():
#         print(f"[{TARGET_SYMBOL}] Predictor batch {batch_id} empty")
#         return

#     current_model = load_model()
#     if current_model is None:
#         print(f"[{TARGET_SYMBOL}] No model available for batch {batch_id}")
#         return

#     predictions = current_model.transform(batch_df)

#     @udf(StringType())
#     def features_to_json(features):
#         return json.dumps({
#             f: float(value) for f, value in zip(FEATURE_COLUMNS, features)
#         })

#     # Set label to null for immediate predictions (will be updated later)
#     predictions_to_save = predictions.withColumn(
#         "input_data", 
#         features_to_json(col("features"))
#     ).withColumn(
#         "label",
#         F.lit(None).cast(DoubleType())
#     )

#     predictions_to_save.select(
#         col("symbol"),
#         col("timestamp"),
#         col("event_time"),
#         col("input_data"),
#         col("prediction"),
#         col("label")
#     ).write \
#      .format("org.apache.spark.sql.cassandra") \
#      .mode("append") \
#      .options(table=CASSANDRA_TABLE, keyspace=CASSANDRA_KEYSPACE) \
#      .save()

#     print(f"[{TARGET_SYMBOL}] Completed predictions for batch {batch_id}")

def predict_incoming(batch_df: DataFrame, batch_id: int):
    # start_time = time()
    if batch_df.rdd.isEmpty():
        print(f"[{TARGET_SYMBOL}] Predictor batch {batch_id} empty")
        return

    try:
        current_model = load_model()
        if current_model is None:
            print(f"[{TARGET_SYMBOL}] No model available for batch {batch_id}")
            return

        predictions = current_model.transform(batch_df)

         @udf(StringType())
        def features_to_json(features):
            return json.dumps({
                f: float(value) for f, value in zip(FEATURE_COLUMNS, features)
            })

        predictions_to_save = predictions.withColumn(
            "input_data", 
            features_to_json(col("features"))
        ).withColumn(
            "label",
            F.lit(None).cast(DoubleType())
        )

        # Use the retry function here
        write_to_cassandra_with_retry(
            predictions_to_save.select(
                col("symbol"),
                col("timestamp"),
                col("event_time"),
                col("input_data"),
                col("prediction"),
                col("label")
            )
        )

        # update_time = time() - start_time
        print(f"[{TARGET_SYMBOL}] Predictions took {update_time:.8f}.")
        print(f"[{TARGET_SYMBOL}] Completed predictions for batch {batch_id}")
        
    except Exception as e:
        print(f"[{TARGET_SYMBOL}] Error in predict_incoming: {str(e)}")

# Start predictor stream with minimal trigger interval
# query_pred = processed_df_for_pred.writeStream \
#     .foreachBatch(predict_incoming) \
#     .outputMode("append") \
#     .option("checkpointLocation", CHECKPOINT_PATH_PRED) \
#     .trigger(processingTime='1 second') \
#     .start()

# Modify your streaming queries
query_pred = processed_df_for_pred.writeStream \
    .foreachBatch(predict_incoming) \
    .outputMode("append") \
    .option("checkpointLocation", CHECKPOINT_PATH_PRED) \
    .trigger(processingTime='1 second') \
    .option("maxOffsetsPerTrigger", MAX_OFFSETS_PER_TRIGGER) \
    .start()

print(f"[{TARGET_SYMBOL}] Started predictor query")

# Start label updater stream
def update_labels(batch_df: DataFrame, batch_id: int):
    if batch_df.rdd.isEmpty():
        return

    # Calculate 10-minute average prices and select window bounds
    avg_prices = batch_df.groupBy(
        window(col("event_time"), "10 minutes")
    ).agg(
        avg("label").alias("actual_price")
    ).select(
        col("window.start").alias("window_start"),
        col("window.end").alias("window_end"),
        col("actual_price")
    )

    # For each window, update records in Cassandra
    for row in avg_prices.collect():
        window_start = row['window_start']
        window_end = row['window_end']
        actual_price = row['actual_price']
        
        # Read matching records
        matching_records = spark.read \
            .format("org.apache.spark.sql.cassandra") \
            .options(table=CASSANDRA_TABLE, keyspace=CASSANDRA_KEYSPACE) \
            .load() \
            .filter(
                (col("symbol") == TARGET_SYMBOL) &
                (col("event_time") >= window_start) &
                (col("event_time") < window_end)
            )

        # Update records with actual price
        if not matching_records.rdd.isEmpty():
            # matching_records \
            #     .withColumn("label", F.lit(actual_price)) \
            #     .write \
            #     .format("org.apache.spark.sql.cassandra") \
            #     .mode("append") \
            #     .options(table=CASSANDRA_TABLE, keyspace=CASSANDRA_KEYSPACE) \
            #     .save()
            # Use the retry function here
            write_to_cassandra_with_retry(
                matching_records.withColumn("label", F.lit(actual_price))
            )

# Start label updater stream
query_labels = processed_df.writeStream \
    .foreachBatch(update_labels) \
    .outputMode("update") \
    .trigger(processingTime='10 minutes') \
    .start()

spark.streams.awaitAnyTermination()