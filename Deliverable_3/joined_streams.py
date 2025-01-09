import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime

# Spark setup
spark = (
    SparkSession.builder.appName("BP-ETH-Correlation")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
        "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0",
    )
    .config("spark.cassandra.connection.host", "localhost")
    .config("spark.streaming.backpressure.enabled", "true") # Limit input rate
    .config("spark.streaming.kafka.maxRatePerPartition", "100")  # Limit input rate
    .config("spark.streaming.kafka.minRatePerPartition", "10")  # Ensure minimum processing
    .config("spark.driver.host", "localhost")
    .config("spark.ui.port", "4047")
    .master("local[4]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# Schema definition remains the same
schema = (
    StructType()
    .add("symbol", StringType())
    .add("timestamp", LongType())
    .add("source", StringType())
    .add("data_type", StringType())
    .add("bid", DoubleType())
    .add("ask", DoubleType())
    .add("price", DoubleType())
    .add("volume", DoubleType())
    .add("spread_raw", DoubleType())
    .add("spread_table", DoubleType())
    .add("volatility", DoubleType())
    .add("market_sentiment", DoubleType())
    .add("trading_activity", DoubleType())
)

# Read BP stream with alias
df_bp = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "model-topic-bp")
    .option("startingOffsets", "earliest")
    .option("failOnDataLoss", "false")
    .load()
    .selectExpr("CAST(value AS STRING) as json")
    .select(from_json(col("json"), schema).alias("data"))
    .select(
        col("data.timestamp").alias("bp_timestamp"), col("data.price").alias("bp_price")
    )
    .withColumn("event_time_bp", (col("bp_timestamp") / 1000).cast(TimestampType()))
    .withColumn(
        "time_bucket_bp", window(col("event_time_bp"), "1 minute")
    )  # Changed column name
    .withWatermark("event_time_bp", "10 minutes")
)

# Read ETH stream with alias
df_eth = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "model-topic-eth")
    .option("startingOffsets", "earliest")
    .option("failOnDataLoss", "false")
    .load()
    .selectExpr("CAST(value AS STRING) as json")
    .select(from_json(col("json"), schema).alias("data"))
    .select(
        col("data.timestamp").alias("eth_timestamp"), col("data.ask").alias("eth_ask")
    )
    .withColumn("event_time_eth", (col("eth_timestamp") / 1000).cast(TimestampType()))
    .withColumn(
        "time_bucket_eth", window(col("event_time_eth"), "1 minute")
    )  # Changed column name
    .withWatermark("event_time_eth", "10 minutes")
)

# Join streams with unambiguous condition
joined_df = df_bp.join(
    df_eth,
    expr(
        "time_bucket_bp.start = time_bucket_eth.start AND time_bucket_bp.end = time_bucket_eth.end"
    ),
    "inner",
)


# Rest of the code remains the same
def process_window(batch_df, batch_id):
    if batch_df.rdd.isEmpty():
        print(f"Batch {batch_id} is empty")
        return

    try:
        # Group by window and calculate averages
        minute_avgs = batch_df.groupBy("time_bucket_bp").agg(  # Using time_bucket_bp
            avg("bp_price").alias("avg_bp_price"), avg("eth_ask").alias("avg_eth_ask")
        )

        # Calculate correlation
        correlation = minute_avgs.select(
            corr("avg_bp_price", "avg_eth_ask").alias("correlation")
        ).collect()[0]["correlation"]

        if correlation is not None:
            # Get window info
            window_data = batch_df.agg(
                min("bp_timestamp").alias("window_start_ts"),
                max("bp_timestamp").alias("window_end_ts"),
                min("event_time_bp").alias("event_time"),
            ).collect()[0]

            # Calculate average prices
            window_avgs = batch_df.agg(
                avg("bp_price").alias("bp_price"), avg("eth_ask").alias("eth_ask")
            ).collect()[0]

            # Create DataFrame for Cassandra
            cassandra_df = spark.createDataFrame(
                [
                    (
                        "BP-ETH",  # symbol
                        int(window_data["window_start_ts"]),  # timestamp
                        window_data["event_time"],  # event_time
                        int(window_data["window_start_ts"]),  # window_start_ts
                        int(window_data["window_end_ts"]),  # window_end_ts
                        float(correlation),  # correlation
                        float(window_avgs["bp_price"]),  # bp_price
                        float(window_avgs["eth_ask"]),  # eth_ask
                    )
                ],
                [
                    "symbol",
                    "timestamp",
                    "event_time",
                    "window_start_ts",
                    "window_end_ts",
                    "correlation",
                    "bp_price",
                    "eth_ask",
                ],
            )

            print(f"Attempting to write {cassandra_df.count()} records to Cassandra")
            # Write to Cassandra
            cassandra_df.write.format("org.apache.spark.sql.cassandra").mode(
                "append"
            ).options(table="correlations", keyspace="stream_predictions").save()

            print(
                f"Batch {batch_id}: correlation={correlation:.3f}, "
                f"avg_bp_price={window_avgs['bp_price']:.2f}, "
                f"avg_eth_ask={window_avgs['eth_ask']:.2f}"
            )

    except Exception as e:
        print(f"Error processing batch {batch_id}: {str(e)}")


# Start the streaming query with error handling
try:
    # Ensure checkpoint directory exists
    import os

    checkpoint_path = "./checkpoint/correlations"
    os.makedirs(checkpoint_path, exist_ok=True)

    query = (
        joined_df.writeStream.foreachBatch(process_window)
        .trigger(processingTime="5 minutes")
        .option("checkpointLocation", checkpoint_path)
        .start()
    )

    print("Started streaming query. Waiting for termination...")
    query.awaitTermination()
except Exception as e:
    print(f"Error in main loop: {str(e)}")
    if "query" in locals():
        query.stop()
    spark.stop()
 