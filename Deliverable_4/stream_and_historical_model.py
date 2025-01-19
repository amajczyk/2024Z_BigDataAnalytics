import sys
import json
import os
import shutil
import pickle
import threading
import numpy as np
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf, split, window, from_unixtime, avg,  when, lit
from pyspark.sql.types import StructType, StringType, DoubleType, LongType, TimestampType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel, RandomForestRegressionModel, GBTRegressionModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from datetime import datetime

from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from typing import Tuple, Dict

# Configuration Parameters
PREFIX = "10m"
symbols_features = {
    "BP": ["volume", "volatility", "market_sentiment", "trading_activity", "price"],
    "COP": ["volume", "volatility", "market_sentiment", "trading_activity", "price"],
    "SHEL": ["volume", "volatility", "market_sentiment", "trading_activity", "price"],
    "XOM": ["volume", "volatility", "market_sentiment", "trading_activity", "price"],
    "ETHEREUM": ["bid", "ask", "spread_raw", "spread_table", "price"],
}

# Add at the beginning of the script, after imports
if len(sys.argv) != 3:
    raise ValueError("Please provide the TARGET_SYMBOL and TRAIN_HISTORICAL (true/false) as command-line arguments.")

TARGET_SYMBOL = sys.argv[1]
TRAIN_HISTORICAL = sys.argv[2].lower() == 'true'

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

HISTORICAL_MODEL_BASE_PATH = f"./{PREFIX}/historical_models/{TARGET_SYMBOL}"
os.makedirs(HISTORICAL_MODEL_BASE_PATH, exist_ok=True)

LABEL_COLUMN = "price"

SELECTED_MODEL_TYPE = None

# Add constant for selected model type file
SELECTED_MODEL_TYPE_FILE = "SELECTED_MODEL.txt"

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
    .master("local[2]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

def save_historical_model(model):
    """
    Save the historical model to disk
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(HISTORICAL_MODEL_BASE_PATH, f"model_{timestamp}")
    model.write().overwrite().save(model_path)

    latest_model_path = os.path.join(HISTORICAL_MODEL_BASE_PATH, "latest_model.txt")
    with open(latest_model_path, "w") as f:
        f.write(model_path)

    # Save selected model type to a separate file
    model_type_path = os.path.join(HISTORICAL_MODEL_BASE_PATH, SELECTED_MODEL_TYPE_FILE)
    with open(model_type_path, "w") as f:
        f.write(SELECTED_MODEL_TYPE)
    
    print(f"[{TARGET_SYMBOL}] Saved historical model to {model_path}")
    print(f"[{TARGET_SYMBOL}] Saved selected model type: {SELECTED_MODEL_TYPE}")

def load_historical_model():
    """
    Load the historical model from disk based on saved model type
    """
    latest_model_path = os.path.join(HISTORICAL_MODEL_BASE_PATH, "latest_model.txt")
    model_type_path = os.path.join(HISTORICAL_MODEL_BASE_PATH, SELECTED_MODEL_TYPE_FILE)
    
    if not os.path.exists(latest_model_path) or not os.path.exists(model_type_path):
        print(f"[{TARGET_SYMBOL}] Missing model files")
        return None
        
    try:
        # Read the model type first
        with open(model_type_path, "r") as f:
            model_type = f.read().strip()
            
        # Read the model path
        with open(latest_model_path, "r") as f:
            model_path = f.read().strip()
            
        # Load appropriate model based on type
        if model_type == 'LinearRegression':
            model = LinearRegressionModel.load(model_path)
        elif model_type == 'RandomForest':
            model = RandomForestRegressionModel.load(model_path)
        elif model_type == 'GradientBoosting':
            model = GBTRegressionModel.load(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        print(f"[{TARGET_SYMBOL}] Loaded historical {model_type} model from {model_path}")
        return model
        
    except Exception as e:
        print(f"[{TARGET_SYMBOL}] Error loading historical model: {e}")
        return None


def train_and_evaluate_historical_models(historical_features, test_size: float = 0.2) -> Tuple[object, str, Dict]:
    """
    Trains multiple models on historical data and evaluates their performance.
    Returns the best model, its type, and evaluation metrics.
    """
    # Split data into training and test sets
    train_data, test_data = historical_features.randomSplit([1 - test_size, test_size], seed=42)
    
    # Initialize models with their parameters
    models = {
        'LinearRegression': LinearRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=MAX_ITER,
            regParam=REG_PARAM,
            elasticNetParam=ELASTIC_NET_PARAM
        ),
        'RandomForest': RandomForestRegressor(
            featuresCol="features",
            labelCol="label",
            numTrees=100,
            maxDepth=10,
            seed=42
        ),
        'GradientBoosting': GBTRegressor(
            featuresCol="features",
            labelCol="label",
            maxIter=100,
            maxDepth=5
        )
    }
    
    # Initialize evaluator
    evaluator = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="rmse"
    )
    
    # Train and evaluate each model
    model_metrics = {}
    best_rmse = float('inf')
    best_model = None
    best_model_type = None
    
    print(f"[{TARGET_SYMBOL}] Training and evaluating models...")
    
    for model_type, model in models.items():
        try:
            # Train model
            print(f"[{TARGET_SYMBOL}] Training {model_type}...")
            trained_model = model.fit(train_data)
            
            # Make predictions on test data
            predictions = trained_model.transform(test_data)
            
            # Calculate metrics
            rmse = evaluator.evaluate(predictions)
            
            # Calculate additional metrics
            evaluator.setMetricName("mae")
            mae = evaluator.evaluate(predictions)
            evaluator.setMetricName("r2")
            r2 = evaluator.evaluate(predictions)
            
            model_metrics[model_type] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            print(f"[{TARGET_SYMBOL}] {model_type} metrics:")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    MAE: {mae:.4f}")
            print(f"    R2: {r2:.4f}")
            
            # Update best model if current model performs better
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = trained_model
                best_model_type = model_type
                
        except Exception as e:
            print(f"[{TARGET_SYMBOL}] Error training {model_type}: {str(e)}")
            model_metrics[model_type] = None
    
    if best_model is None:
        raise ValueError("No models were successfully trained")
    
    # Update global selected model type
    global SELECTED_MODEL_TYPE
    SELECTED_MODEL_TYPE = best_model_type
    
    # Save the best model
    save_historical_model(best_model)
    
    print(f"[{TARGET_SYMBOL}] Best performing model: {best_model_type}")
    return best_model, best_model_type, model_metrics


def train_on_historical_data():
    """
    Loads historical data from Cassandra and trains initial model with debug info
    """
    print(f"[{TARGET_SYMBOL}] Loading historical data from Cassandra...")
    
    # Load historical data from Cassandra
    historical_df = spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .options(table=CASSANDRA_TABLE, keyspace=CASSANDRA_KEYSPACE) \
        .load() \
        .filter(col("symbol") == TARGET_SYMBOL)

    # Debug print
    print(f"[{TARGET_SYMBOL}] Initial schema:")
    historical_df.printSchema()
    print(f"[{TARGET_SYMBOL}] Initial count: {historical_df.count()}")

    if historical_df.rdd.isEmpty():
        raise ValueError(f"No historical data found for symbol {TARGET_SYMBOL}")

    # Parse input_data JSON to get features
    for feature in FEATURE_COLUMNS:
        historical_df = historical_df.withColumn(
            feature,
            F.get_json_object(col("input_data"), f"$.{feature}").cast("double")
        )
    
    # Debug print after feature extraction
    print(f"[{TARGET_SYMBOL}] Schema after feature extraction:")
    historical_df.printSchema()
    print(f"[{TARGET_SYMBOL}] Sample data after feature extraction:")
    historical_df.select("symbol", *FEATURE_COLUMNS, "label").show(5)

    # Add event_time column if not exists
    if "event_time" not in historical_df.columns:
        historical_df = historical_df.withColumn(
            "event_time",
            (col("timestamp") / 1000).cast(TimestampType())
        )

    # Group by 10-minute windows
    historical_windowed = historical_df \
        .groupBy(
            window(col("event_time"), "10 minutes"),
            col("symbol")
        ) \
        .agg(
            *[
                F.avg(col(feature)).alias(f"avg_{feature}")
                for feature in FEATURE_COLUMNS
            ],
            F.avg(col("label")).alias("label")
        )

    # Debug print after windowing
    print(f"[{TARGET_SYMBOL}] Schema after windowing:")
    historical_windowed.printSchema()
    print(f"[{TARGET_SYMBOL}] Sample data after windowing:")
    historical_windowed.show(5)

    # Check for null values
    for feature in [f"avg_{feature}" for feature in FEATURE_COLUMNS] + ["label"]:
        null_count = historical_windowed.filter(col(feature).isNull()).count()
        if null_count > 0:
            print(f"[{TARGET_SYMBOL}] WARNING: Found {null_count} null values in {feature}")

    # Prepare features
    assembler_historical = VectorAssembler(
        inputCols=[f"avg_{feature}" for feature in FEATURE_COLUMNS],
        outputCol="features"
    )

    try:
        historical_features = assembler_historical.transform(historical_windowed)
        
        # Debug print assembled features
        print(f"[{TARGET_SYMBOL}] Schema after feature assembly:")
        historical_features.printSchema()
        print(f"[{TARGET_SYMBOL}] Sample data after feature assembly:")
        historical_features.select("features", "label").show(5, truncate=False)
        
        # Check for problematic data
        invalid_count = historical_features.filter(
            col("features").isNull() | 
            col("label").isNull() |
            F.isnan("label")
        ).count()
        
        if invalid_count > 0:
            print(f"[{TARGET_SYMBOL}] WARNING: Found {invalid_count} rows with null/NaN features or labels")
            
            # Remove problematic rows
            historical_features = historical_features.filter(
                col("features").isNotNull() & 
                col("label").isNotNull() &
                ~F.isnan("label")
            )
            print(f"[{TARGET_SYMBOL}] Cleaned data count: {historical_features.count()}")
            
        best_model, model_type, metrics = train_and_evaluate_historical_models(historical_features)
        
        print(f"[{TARGET_SYMBOL}] Selected model type: {SELECTED_MODEL_TYPE}")
        print(f"[{TARGET_SYMBOL}] Model evaluation metrics:")
        for model_type, model_metrics in metrics.items():
            if model_metrics:
                print(f"\n{model_type}:")
                for metric, value in model_metrics.items():
                    print(f"    {metric}: {value:.4f}")
        
        return best_model
        
    except Exception as e:
        print(f"[{TARGET_SYMBOL}] Error during model training and evaluation: {str(e)}")
        raise e



# Modify the historical model training section
if TRAIN_HISTORICAL:
    print(f"[{TARGET_SYMBOL}] Training new historical model...")
    historical_model = train_on_historical_data()
else:
    print(f"[{TARGET_SYMBOL}] Loading existing historical model...")
    historical_model = load_historical_model()

if historical_model is None:
    raise ValueError("Failed to obtain historical model")


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
    .load()

parsed_df = df_kafka.selectExpr("CAST(value AS STRING) as json") \
    .select(from_json(col("json"), schema).alias("data")) \
    .select("data.*")

filtered_df = parsed_df.filter(col("symbol") == TARGET_SYMBOL)

LABEL_COLUMN = "price"


# Correct the 'price' for Ethereum if it's set to -1 by computing the average of 'bid' and 'ask'
if TARGET_SYMBOL == "ETHEREUM":
    # Create a corrected 'price' column
    corrected_price = when(col("price") == -1, (col("bid") + col("ask")) / 2).otherwise(col("price"))
    
    # Apply the corrected 'price' to both 'price' feature and 'label'
    processed_df = filtered_df.withColumn(
        "price_corrected",
        corrected_price.cast("double")
    ).select(
        col("symbol").alias("symbol"),
        *[col(feature).cast("double").alias(f"{feature}") for feature in FEATURE_COLUMNS if feature != "price"],
        col("price_corrected").alias("price"),  # Updated 'price' for features
        col("price_corrected").alias("label"),  # Updated 'label'
        col("timestamp").cast("long").alias("timestamp")  # Include timestamp for Cassandra or other sinks
    )
else:
    # For other symbols, no correction needed
    processed_df = filtered_df.select(
        col("symbol").alias("symbol"),
        *[col(feature).cast("double").alias(f"{feature}") for feature in FEATURE_COLUMNS],
        col(LABEL_COLUMN).cast("double").alias("label"),  # Use the original 'price' as label
        col("timestamp").cast("long").alias("timestamp")  # Include timestamp for Cassandra or other sinks
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


def update_model(batch_df: DataFrame, batch_id: int):
    global lr_model

    if batch_df.rdd.isEmpty():
        print(f"[{TARGET_SYMBOL}] Batch {batch_id} is empty. Skipping update.")
        return

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

        print(f"[{TARGET_SYMBOL}] Updated model with batch {batch_id}")
        print(f"[{TARGET_SYMBOL}] Coefficients: {lr_model.coefficients}")
        print(f"[{TARGET_SYMBOL}] Intercept: {lr_model.intercept}")

        save_model(lr_model)

    except Exception as e:
        print(f"[{TARGET_SYMBOL}] Error in batch {batch_id}: {e}")
        raise e


lr_model = load_model()


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

def predict_incoming(batch_df: DataFrame, batch_id: int):
    global historical_model, lr_model
    
    if batch_df.rdd.isEmpty():
        print(f"[{TARGET_SYMBOL}] Predictor batch {batch_id} empty")
        return

    if historical_model is None:
        historical_model = load_historical_model()
    if lr_model is None:
        lr_model = load_model()

    if lr_model is None or historical_model is None:
        print(f"[{TARGET_SYMBOL}] One or both models not available")
        return

    try:
        # Get predictions from streaming model
        streaming_predictions = lr_model.transform(batch_df)
        
        # Get predictions from historical model separately
        historical_predictions = historical_model.transform(batch_df)
        
        # Select only necessary columns from each prediction DataFrame
        streaming_cols = streaming_predictions.select(
            "symbol",
            "timestamp",
            "event_time",
            "features",
            "label",
            col("prediction").alias("prediction_streaming")
        )
        
        historical_cols = historical_predictions.select(
            "symbol",
            col("prediction").alias("prediction_historical")
        )

        # Join the predictions using symbol as the key
        combined_predictions = streaming_cols.join(
            historical_cols,
            "symbol",
            "inner"
        )

        @udf(StringType())
        def features_to_json(features):
            return json.dumps({
                f: float(value) for f, value in zip(FEATURE_COLUMNS, features)
            })

        predictions_to_save = combined_predictions.withColumn(
            "input_data", 
            features_to_json(col("features"))
        ).withColumn(
            "prediction",
            col("prediction_streaming")  # Use streaming prediction as main prediction
        ).drop("prediction_streaming")  # Remove the temporary column

        # Select final columns for saving
        final_predictions = predictions_to_save.select(
            "symbol",
            "timestamp",
            "event_time",
            "input_data",
            "prediction",
            "prediction_historical",
            "label"
        )

        # Save to Cassandra
        final_predictions.write \
            .format("org.apache.spark.sql.cassandra") \
            .mode("append") \
            .options(table=CASSANDRA_TABLE, keyspace=CASSANDRA_KEYSPACE) \
            .save()

        print(f"[{TARGET_SYMBOL}] Completed predictions for batch {batch_id}")
        
    except Exception as e:
        print(f"[{TARGET_SYMBOL}] Error in predict_incoming for batch {batch_id}: {str(e)}")
        raise e

# Start predictor stream with minimal trigger interval
query_pred = processed_df_for_pred.writeStream \
    .foreachBatch(predict_incoming) \
    .outputMode("append") \
    .option("checkpointLocation", CHECKPOINT_PATH_PRED) \
    .trigger(processingTime='1 second') \
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
            matching_records \
                .withColumn("label", F.lit(actual_price)) \
                .write \
                .format("org.apache.spark.sql.cassandra") \
                .mode("append") \
                .options(table=CASSANDRA_TABLE, keyspace=CASSANDRA_KEYSPACE) \
                .save()

# Start label updater stream
query_labels = processed_df.writeStream \
    .foreachBatch(update_labels) \
    .outputMode("update") \
    .trigger(processingTime='10 minutes') \
    .start()

spark.streams.awaitAnyTermination()