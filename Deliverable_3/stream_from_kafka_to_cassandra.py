from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType, DoubleType, LongType
import requests

# Kafka and model endpoints
kafka_brokers = "localhost:9092"
topic = "model-topic"


# Create a Spark session
spark = SparkSession.builder \
    .appName("KafkaReadExample") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .config("spark.ui.port", "4050") \
    .getOrCreate()

# Define the schema of the JSON data
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

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_brokers) \
    .option("subscribe", topic) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load()

parsed_df = df.selectExpr("CAST(value AS STRING) as json") \
    .select(from_json(col("json"), schema).alias("data")) \
    .select("data.*")


def save_to_cassandra(symbol, payload, prediction):
    """Save the input data and prediction to a Cassandra table."""
    cassandra_df = spark.createDataFrame([{
        "symbol": symbol,
        "timestamp": payload.get("timestamp"),
        "input_data": payload,
        "prediction": prediction
    }])
    cassandra_df.write \
        .format("org.apache.spark.sql.cassandra") \
        .mode("append") \
        .options(table="model_predictions", keyspace="your_keyspace") \
        .save()

# Process each row from the streaming DataFrame
query = parsed_df.writeStream \
    .foreach(process_row) \
    .outputMode("update") \
    .start()

query.awaitTermination()
