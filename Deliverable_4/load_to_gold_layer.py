import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, to_date, count, avg, max, min
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

def setup_logger():
    """Set up the logger for the script."""
    logger = logging.getLogger("SparkJobLogger")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("spark_silver_to_gold.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def main():
    logger = setup_logger()
    logger.info("Starting Spark job for silver to gold layer aggregation.")

    try:
        # Initialize Spark Session
        spark = SparkSession.builder \
            .appName("AggregationForGoldLayer") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
            .getOrCreate()

        # Paths
        yfinance_silver_path = "/user/adam_majczyk2001/nifi/silver_parquet/yfinance/"
        news_silver_path = "/user/adam_majczyk2001/nifi/silver_parquet/news/"

        # Read Data
        yfinance_df = spark.read.parquet(yfinance_silver_path)
        news_df = spark.read.parquet(news_silver_path)

        logger.info("Data loaded from silver layer.")

        # News Aggregation
        news_df = news_df.withColumn("aggregation_date", to_date(col("datetime"), "yyyy-MM-dd"))
        aggregated_news_df = news_df.groupBy(
            col("aggregation_date"), col("source_site").alias("symbol")
        ).agg(count("title").alias("total_articles"))

        keywords_df = news_df.select(
            col("aggregation_date"),
            col("source_site").alias("symbol"),
            explode(col("keywords")).alias("keyword")
        ).groupBy("symbol", "aggregation_date", "keyword").count()

        logger.info("News data aggregated.")

        # YFinance Aggregation
        yfinance_df = yfinance_df.withColumn("aggregation_date", to_date(col("record_timestamp"), "yyyy-MM-dd"))
        aggregated_yfinance_df = yfinance_df.groupBy(
            col("company").alias("symbol"), col("aggregation_date")
        ).agg(
            avg("price").alias("avg_price"),
            max("price").alias("max_price"),
            min("price").alias("min_price"),
            avg("volume").alias("avg_volume"),
            avg("volatility").alias("avg_volatility"),
            avg("market_sentiment").alias("avg_sentiment")
        )

        logger.info("YFinance data aggregated.")

        # Cassandra Connection
        cluster = Cluster(['127.0.0.1'])
        session = cluster.connect()
        session.set_keyspace('gold_layer')

        logger.info("Connected to Cassandra.")

        # Insert into Cassandra
        for row in aggregated_news_df.collect():
            session.execute("""
                INSERT INTO aggregated_news (symbol, aggregation_date, total_articles)
                VALUES (%s, %s, %s)
            """, (row['symbol'], row['aggregation_date'], row['total_articles']))

        logger.info("Aggregated news data inserted into Cassandra.")

        for row in keywords_df.collect():
            session.execute("""
                INSERT INTO aggregated_keywords (symbol, aggregation_date, keyword, count)
                VALUES (%s, %s, %s, %s)
            """, (row['symbol'], row['aggregation_date'], row['keyword'], row['count']))

        logger.info("Aggregated keywords data inserted into Cassandra.")

        for row in aggregated_yfinance_df.collect():
            session.execute("""
                INSERT INTO aggregated_yfinance (symbol, aggregation_date, avg_stock_price, max_stock_price, min_stock_price, volume_traded, avg_volatility, avg_sentiment)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (row['symbol'], row['aggregation_date'], row['avg_price'], row['max_price'], row['min_price'], row['avg_volume'], row['avg_volatility'], row['avg_sentiment']))

        logger.info("Aggregated YFinance data inserted into Cassandra.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

    finally:
        spark.stop()
        logger.info("Spark session stopped.")

if __name__ == "__main__":
    main()
