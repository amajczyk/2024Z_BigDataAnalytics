import sys
import json
import os
import pickle
import threading
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf, split
from pyspark.sql.types import StructType, StringType, DoubleType, LongType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.sql import DataFrame

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

# Checkpoint location for Spark Structured Streaming
CHECKPOINT_PATH = f"./checkpoint/dir/{TARGET_SYMBOL}"

# Ensure that the directories exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# ==========================
# Cassandra Configuration
# ==========================

CASSANDRA_KEYSPACE = "your_keyspace"  # Replace with your keyspace
CASSANDRA_TABLE = "model_predictions"  # Replace with your table name

# ==========================
# Kafka Configuration
# ==========================

KAFKA_BROKERS = "localhost:9092"
KAFKA_TOPIC = "model-topic"

# ==========================
# Hyperparameters for Model Training
# ==========================

MAX_ITER = 100  # Number of iterations for model fitting
REG_PARAM = 10
ELASTIC_NET_PARAM = 0.5

# ==========================
# Initialize Spark Session
# ==========================

spark = SparkSession.builder \
    .appName(f"ContinuousTrainingLinearRegression_{TARGET_SYMBOL}") \
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
            "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0") \
    .config("spark.cassandra.connection.host", "localhost") \
    .config("spark.ui.port", "4050") \
    .getOrCreate()

# Set log level to WARN to reduce verbosity
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

# Determine the label column based on the target symbol
if TARGET_SYMBOL in ["BP", "COP", "SHEL", "XOM"]:
    LABEL_COLUMN = "trading_activity"
elif TARGET_SYMBOL == "ETHEREUM":
    LABEL_COLUMN = "price"  # Adjust as needed
else:
    raise ValueError(f"Unhandled TARGET_SYMBOL '{TARGET_SYMBOL}'.")

# Select feature columns and label
selected_columns = FEATURE_COLUMNS + [LABEL_COLUMN]
processed_df = filtered_df.select(
    col("symbol").alias("symbol"),
    *[col(feature).cast("double").alias(f"{feature}") for feature in FEATURE_COLUMNS],
    col(LABEL_COLUMN).cast("double").alias("label"),
    col("timestamp").cast("long").alias("timestamp")  # Include timestamp for Cassandra
)

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features")
processed_df = assembler.transform(processed_df).select("symbol", "features", "label", "timestamp")

# ==========================
# Initialize or Load Linear Regression Model
# ==========================

def load_model(model_path: str) -> LinearRegressionModel:
    if os.path.exists(model_path):
        try:
            model = LinearRegressionModel.load(model_path)
            print(f"[{TARGET_SYMBOL}] Loaded existing Linear Regression model from {model_path}.")
            return model
        except Exception as e:
            print(f"[{TARGET_SYMBOL}] Failed to load model from {model_path}: {e}")
            # Optionally, remove the corrupted model directory
            print(f"[{TARGET_SYMBOL}] Removing corrupted model directory.")
            import shutil
            shutil.rmtree(model_path)
            return None
    else:
        print(f"[{TARGET_SYMBOL}] No existing model found at {model_path}. Will initialize on the first batch.")
        return None


lr_model = load_model(MODEL_PATH)

# ==========================
# Define the foreachBatch Function
# ==========================

def update_model(batch_df: DataFrame, batch_id: int):
    global lr_model

    if batch_df.rdd.isEmpty():
        print(f"[{TARGET_SYMBOL}] Batch {batch_id} is empty. Skipping model update.")
        return

    try:
        if lr_model is None:
            # Initialize the model on the first batch
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
            # Update the model with the new batch by retraining
            lr = LinearRegression(
                featuresCol="features",
                labelCol="label",
                maxIter=MAX_ITER,
                regParam=REG_PARAM,
                elasticNetParam=ELASTIC_NET_PARAM
            )
            new_model = lr.fit(batch_df)

            # Average coefficients and intercept
            updated_coefficients = (lr_model.coefficients.toArray() + new_model.coefficients.toArray()) / 2
            updated_intercept = (lr_model.intercept + new_model.intercept) / 2

            # Since PySpark does not support directly setting coefficients and intercept,
            # we'll overwrite the existing model with the new model
            lr_model = new_model
            print(f"[{TARGET_SYMBOL}] Updated Linear Regression model with batch {batch_id}")
            
            # Print coefficients and intercept
            print(f"[{TARGET_SYMBOL}] Coefficients after update: {lr_model.coefficients}")
            print(f"[{TARGET_SYMBOL}] Intercept after update: {lr_model.intercept}")

        # Persist the updated model
        if lr_model is not None:
            lr_model.write().overwrite().save(MODEL_PATH)
            print(f"[{TARGET_SYMBOL}] Model saved after batch {batch_id}")

        # Make predictions using the current model
        predictions = lr_model.transform(batch_df)

        # Select relevant columns for Cassandra
        predictions_to_save = predictions.select(
            "symbol",
            "timestamp",
            "features",
            "prediction"
        )

        # Convert features vector to JSON string for storage
        @udf(StringType())
        def features_to_json(features):
            return json.dumps({f: float(value) for f, value in zip(FEATURE_COLUMNS, features)})

        predictions_to_save = predictions_to_save.withColumn("input_data", features_to_json(col("features")))

        # Prepare DataFrame for Cassandra
        cassandra_df = predictions_to_save.select(
            "symbol",
            "timestamp",
            "input_data",
            "prediction"
        )

        # # Write predictions to Cassandra
        # cassandra_df.write \
        #     .format("org.apache.spark.sql.cassandra") \
        #     .mode("append") \
        #     .options(table=CASSANDRA_TABLE, keyspace=CASSANDRA_KEYSPACE) \
        #     .save()

        # print(f"[{TARGET_SYMBOL}] Saved predictions of batch {batch_id} to Cassandra.")

    except Exception as e:
        print(f"[{TARGET_SYMBOL}] Error in batch {batch_id}: {e}")
        # Raising the exception will terminate the streaming query
        raise e


# ==========================
# Start Streaming Query
# ==========================

query = processed_df.writeStream \
    .foreachBatch(update_model) \
    .outputMode("append") \
    .option("checkpointLocation", CHECKPOINT_PATH) \
    .start()

print(f"[{TARGET_SYMBOL}] Started Structured Streaming query. Awaiting termination...")

# Await termination to keep the application running
query.awaitTermination()
