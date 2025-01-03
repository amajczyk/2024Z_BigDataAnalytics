import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, date_format, explode, lit

def setup_logger():
    """Set up the logger for the script."""
    logger = logging.getLogger("SparkJobLogger")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("spark_bronze_to_silver.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger()

def main():
    logger.info("Starting Spark job")

    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("Process Bronze to Silver") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .config("spark.sql.caseSensitive", "false") \
        .getOrCreate()

    logger.info("Spark session initialized")

    # Paths
    yfinance_bronze_path = "/user/adam_majczyk2001/nifi/bronze_parquet/yfinance/"
    news_bronze_path = "/user/adam_majczyk2001/nifi/bronze_parquet/news/"

    yfinance_silver_path = "/user/adam_majczyk2001/nifi/silver_parquet/yfinance/"
    news_silver_path = "/user/adam_majczyk2001/nifi/silver_parquet/news/"

    # Read data
    logger.info("Reading Bronze data")
    yfinance_df = spark.read.parquet(yfinance_bronze_path)
    news_df = spark.read.parquet(news_bronze_path)

    # Process News Data
    logger.info("Processing News data")
    news_df_no_duplicates = news_df.dropDuplicates(["title"])
    news_df_no_duplicates = news_df_no_duplicates \
        .withColumn("date", to_timestamp(col("date"), "yyyy-MM-dd")) \
        .withColumnRenamed("date", "datetime") \
        .withColumn("datetime", date_format(col("datetime"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"))

    # Save News Data to Silver Path
    logger.info("Saving processed News data to Silver layer")
    news_df_no_duplicates.write.mode("overwrite").parquet(news_silver_path)

    # Process YFinance Data
    logger.info("Processing YFinance data")

    # Explode and transform updates for each company
    xom_updates = yfinance_df \
        .withColumn("update", explode(col("updates_XOM"))) \
        .select(
            col("timestamp").alias("record_timestamp"),
            col("update.price").alias("price"),
            col("update.volume").alias("volume"),
            col("update.volatility").alias("volatility"),
            col("update.bid_ask_spread").alias("bid_ask_spread"),
            col("update.market_sentiment").alias("market_sentiment"),
            col("update.trading_activity").alias("trading_activity"),
            col("update.timestamp").alias("update_timestamp"),
            col("update.source").alias("source"),
            lit("XOM").alias("company")
        )

    bp_updates = yfinance_df \
        .withColumn("update", explode(col("updates_BP"))) \
        .select(
            col("timestamp").alias("record_timestamp"),
            col("update.price").alias("price"),
            col("update.volume").alias("volume"),
            col("update.volatility").alias("volatility"),
            col("update.bid_ask_spread").alias("bid_ask_spread"),
            col("update.market_sentiment").alias("market_sentiment"),
            col("update.trading_activity").alias("trading_activity"),
            col("update.timestamp").alias("update_timestamp"),
            col("update.source").alias("source"),
            lit("BP").alias("company")
        )

    shell_updates = yfinance_df \
        .withColumn("update", explode(col("updates_SHEL"))) \
        .select(
            col("timestamp").alias("record_timestamp"),
            col("update.price").alias("price"),
            col("update.volume").alias("volume"),
            col("update.volatility").alias("volatility"),
            col("update.bid_ask_spread").alias("bid_ask_spread"),
            col("update.market_sentiment").alias("market_sentiment"),
            col("update.trading_activity").alias("trading_activity"),
            col("update.timestamp").alias("update_timestamp"),
            col("update.source").alias("source"),
            lit("SHEL").alias("company")
        )
    
    cop_updates = yfinance_df \
        .withColumn("update", explode(col("updates_COP"))) \
        .select(
            col("timestamp").alias("record_timestamp"),
            col("update.price").alias("price"),
            col("update.volume").alias("volume"),
            col("update.volatility").alias("volatility"),
            col("update.bid_ask_spread").alias("bid_ask_spread"),
            col("update.market_sentiment").alias("market_sentiment"),
            col("update.trading_activity").alias("trading_activity"),
            col("update.timestamp").alias("update_timestamp"),
            col("update.source").alias("source"),
            lit("COP").alias("company")
        )

    # Union all updates and remove duplicates
    updates = xom_updates.union(bp_updates).union(shell_updates).union(cop_updates)
    updates = updates.dropDuplicates(["record_timestamp", "update_timestamp", "company"])

    # Save YFinance Data to Silver Path
    logger.info("Saving processed YFinance data to Silver layer")
    updates.write.mode("overwrite").parquet(yfinance_silver_path)

    logger.info("Spark job completed successfully")

if __name__ == "__main__":
    main()
