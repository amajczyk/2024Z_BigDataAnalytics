from confluent_kafka import Consumer, TopicPartition, KafkaException, KafkaError
import pandas as pd
import json
from cassandra.cluster import Cluster
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import time
from pyspark.ml.pipeline import PipelineModel
import os
from pyspark.sql import functions as F
from datetime import datetime


# Kafka Consumer Configuration
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',  # Kafka broker address
    'group.id': 'eda-group',               # Consumer group
    'auto.offset.reset': 'earliest',       # Start from the beginning
    'enable.auto.commit': False            # Do not commit offsets automatically
})

# Subscribe to the topic
topic = 'model-topic'
consumer.subscribe([topic])

# Connect to Cassandra
def connect_to_cassandra():
    # Replace '127.0.0.1' with your Cassandra node's IP address
    cluster = Cluster(['127.0.0.1'])  
    session = cluster.connect()
    session.set_keyspace('offset_keyspace')  # Replace with your keyspace
    return session

# Read the value from the table
def read_offset(session, offset_name):
    query = "SELECT value FROM offset_table WHERE offset = %s"
    result = session.execute(query, (offset_name,))
    for row in result:
        print(f"Current value for offset '{offset_name}': {row.value}")
        return row.value
    print(f"No record found for offset '{offset_name}'")
    return None

def update_offset(session, offset_name, new_value):
    """
    Updates the value of the given offset in the Cassandra table.
    :param session: The Cassandra session object.
    :param offset_name: The name of the offset to update.
    :param new_value: The new value to set for the offset.
    """
    try:
        query = "UPDATE offset_table SET value = %s WHERE offset = %s"
        session.execute(query, (new_value, offset_name))
        print(f"Updated offset '{offset_name}' to new value: {new_value}")
    except Exception as e:
        print(f"Error updating Cassandra: {e}")


# Retrieve metadata to get the end offsets
def get_end_offsets(consumer, topic):
    metadata = consumer.list_topics(topic)
    partitions = metadata.topics[topic].partitions.keys()  # Get all partition IDs
    end_offsets = {}
    for partition in partitions:
        tp = TopicPartition(topic, partition)
        low, high = consumer.get_watermark_offsets(tp, timeout=5.0)
        if high >= 0:  # Ensure valid offsets
            end_offsets[tp] = high
        else:
            print(f"Warning: Partition {partition} has no valid end offset.")
    return end_offsets

def load_latest_model(symbol, models_dir = './models'):
    """
    Load the latest model for a given symbol.
    """
    try:
        # List all directories in the models path
        model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]

        # Filter model directories for the given symbol
        symbol_model_dirs = [d for d in model_dirs if d.startswith(f"{symbol}_model_")]

        if not symbol_model_dirs:
            print(f"No models found for symbol: {symbol}")
            return None

        # Sort directories by timestamp (assuming the format: <symbol>_model_<timestamp>)
        latest_model_dir = sorted(symbol_model_dirs, key=lambda x: x.split('_')[-1])[-1]

        # Construct the full path to the latest model
        latest_model_path = os.path.join(models_dir, latest_model_dir)

        # Load the latest model
        print(f"Loading model from: {latest_model_path}")
        return PipelineModel.load(latest_model_path)

    except Exception as e:
        print(f"Error loading the latest model: {e}")
        return None

def main():
    session = connect_to_cassandra()

    if session:
        # Define the offset name you want to read
        offset_name = 'current_offset'
        # Read the current offset value
        current_value = read_offset(session, offset_name)
        # Perform additional logic with the retrieved value if necessary
        if current_value is not None:
            print(f"The current offset value is: {current_value}")
        else:
            print("No value retrieved. Make sure the record exists in the table.")

    print("Fetching end offsets for the topic...")
    end_offsets = get_end_offsets(consumer, topic)
    print("End offsets:", end_offsets)
    # Consume messages up to the current end offsets

    messages = []
    try:
        while True:
            msg = consumer.poll(1.0)  # Poll for messages
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    print(f"Reached end of partition {msg.partition()}")
                    continue
                else:
                    raise KafkaException(msg.error())

            # Decode and store the message
            message_value = json.loads(msg.value().decode('utf-8'))
            messages.append(message_value)

            # Check if we've consumed up to the end offset
            tp = TopicPartition(msg.topic(), msg.partition())
            if tp in end_offsets and msg.offset() + 1 >= end_offsets[tp]:
                print(f"Finished consuming partition {msg.partition()}")
                del end_offsets[tp]
                if not end_offsets:
                    break  # Stop if all partitions are consumed

    finally:
        consumer.close()
            
    df = pd.DataFrame(messages)
    update_offset(session, offset_name, len(messages))

        # Display the schema
    print("\nSchema:")
    print(df.dtypes)

    # Show the first few rows
    print("\nFirst few rows of the DataFrame:")
    print(df.head())

    # Perform basic statistics
    print("\nBasic Statistics:")
    print(df.describe(include='all'))

    # # Save messages to a CSV file for future analysis
    # df.to_csv("kafka_messages.csv", index=False)
    # print("\nMessages saved to kafka_messages.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['5s_interval'] = df['timestamp'].dt.floor('5s')
    df_ylife = df[df['source'] == 'YLIFE_FEED']
    df_xtb = df[df['source'] == 'XTB_FEED']
    df_ylife = df_ylife.drop(columns = ['bid', 'ask', 'spread_raw', 'spread_table', 'data_type', 'source'])
    df_ylife = df_ylife.drop(columns = 'timestamp')
    # Group by 5-second intervals and symbol, and calculate the mean for numerical columns
    grouped_df_ylife = (
        df_ylife.groupby(['5s_interval', 'symbol'])
        .mean()
        .reset_index()
    )
    # Generate a complete range of 5-second intervals
    full_intervals = pd.date_range(
        start=df['5s_interval'].min(),
        end=df['5s_interval'].max(),
        freq='5S'
    )

    # Create a complete DataFrame with all tickers and intervals
    complete_df_ylife = pd.MultiIndex.from_product(
        [full_intervals, df['symbol'].unique()],
        names=['5s_interval', 'symbol']
    ).to_frame(index=False)


    # Merge with grouped data to find missing intervals
    merged_df = complete_df_ylife.merge(grouped_df_ylife, on=['5s_interval', 'symbol'], how='left')

    # Interpolate missing values for numerical columns
    interpolated_df_ylife = merged_df.copy()
    interpolated_df_ylife = interpolated_df_ylife.groupby('symbol').apply(lambda group: group.interpolate())
    df_xtb = df_xtb.drop(columns = ['price', 'volume', 'volatility', 'market_sentiment','trading_activity', 'data_type', 'source', 'timestamp'])
    # Group by 5-second intervals and symbol, and calculate the mean for numerical columns
    grouped_df_xtb = (
        df_xtb.groupby(['5s_interval', 'symbol'])
        .mean()
        .reset_index()
    )
    complete_df_xtb = pd.MultiIndex.from_product(
        [full_intervals, df_xtb['symbol'].unique()],
        names=['5s_interval', 'symbol']
        ).to_frame(index=False)


    # Merge with grouped data to find missing intervals
    merged_df = complete_df_xtb.merge(grouped_df_xtb, on=['5s_interval', 'symbol'], how='left')

    # Interpolate missing values for numerical columns
    interpolated_df_xtb = merged_df.copy()
    interpolated_df_xtb = interpolated_df_xtb.groupby('symbol').apply(lambda group: group.interpolate())
    interpolated_df_xtb["price"] = interpolated_df_xtb[["bid", "ask"]].mean(axis=1)

    interpolated_df_xtb = interpolated_df_xtb.drop(columns = 'symbol')
    interpolated_df_ylife = interpolated_df_ylife.drop(columns = 'symbol')

    merged_df = pd.merge(
    interpolated_df_xtb,
    interpolated_df_ylife,
    on=["5s_interval", "symbol"],
    how="outer"
    )
    
    merged_df['price'] = merged_df[['price_x', 'price_y']].bfill(axis=1).iloc[:, 0]

    # Drop the now redundant 'price_x' and 'price_y' columns
    merged_df = merged_df.drop(columns=['price_x', 'price_y'])

    symbol_first_last_intervals = merged_df.groupby("symbol").apply(
    lambda group: pd.Series({
        "first_interval": group.loc[group['price'].notna(), '5s_interval'].min(),
        "last_interval": group.loc[group['price'].notna(), '5s_interval'].max()
    })
    )
    valid_start = symbol_first_last_intervals["first_interval"].max()
    valid_end = symbol_first_last_intervals["last_interval"].min()

    # Filter the merged DataFrame to keep only rows within the valid range
    trimmed_df = merged_df[
        (merged_df["5s_interval"] >= valid_start) & 
        (merged_df["5s_interval"] <= valid_end)
    ]

    df = trimmed_df.copy()
    # Convert '5s_interval' to a datetime type
    df['5s_interval'] = pd.to_datetime(df['5s_interval'])

    # Sort by symbol and timestamp
    df = df.sort_values(by=['symbol', '5s_interval'])

    # Create the 1-minute-ahead price (12 rows ahead for 5-second intervals)
    df['price_1min_ahead'] = df.groupby('symbol')['price'].shift(-12)

    # Drop rows where the target variable (price_1min_ahead) is NaN
    df = df.dropna(subset=['price_1min_ahead'])
    # Pivot the DataFrame to create separate price columns for each symbol
    pivot_df = df.reset_index().pivot(index='5s_interval', columns='symbol', values='price')

    # Flatten the column names to include 'price_' prefix
    pivot_df.columns = [f"price_{symbol}" for symbol in pivot_df.columns]

    # Reset index to make `5s_interval` a regular column (optional)
    pivot_df.reset_index(inplace=True)
    # Merge the original DataFrame (df) with the pivoted DataFrame (pivot_df) on '5s_interval'
    merged_df = df.reset_index().merge(pivot_df, on='5s_interval', how='left')

    # Drop the original 'price' column
    merged_df = merged_df.drop(columns=['price'])

    # If you want to set 'symbol' back as the index after the merge (optional)
    merged_df.set_index('symbol', inplace=True)
        
    spark = SparkSession.builder.appName("SymbolModelTraining").getOrCreate()
    symbols_features = {
    "BP": ["volume", "volatility", "market_sentiment", "trading_activity", 
           "price_BP", "price_COP", "price_ETHEREUM", "price_SHEL", "price_XOM"],
    "COP": ["volume", "volatility", "market_sentiment", "trading_activity", 
            "price_BP", "price_COP", "price_ETHEREUM", "price_SHEL", "price_XOM"],
    "SHEL": ["volume", "volatility", "market_sentiment", "trading_activity", 
             "price_BP", "price_COP", "price_ETHEREUM", "price_SHEL", "price_XOM"],
    "XOM": ["volume", "volatility", "market_sentiment", "trading_activity", 
            "price_BP", "price_COP", "price_ETHEREUM", "price_SHEL", "price_XOM"],
    "ETHEREUM": ["bid", "ask", "spread_raw", "spread_table", 
                 "price_BP", "price_COP", "price_ETHEREUM", "price_SHEL", "price_XOM"]
    }
    # Target column
    target_col = "price_1min_ahead"

    # Generate a timestamp for this training session
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    merged_df = merged_df.reset_index()

    df = spark.createDataFrame(merged_df)

    for symbol, features in symbols_features.items():
        print(f"Training model for symbol: {symbol}")
        
        # Filter data for the symbol
        symbol_df = df.filter(df["symbol"] == symbol)  # Ensure 'symbol' is a column
        
        # Drop rows with missing values
        symbol_df = symbol_df.dropna(subset=features + [target_col])
        
        # Check if 'features' already exists and drop it to avoid conflicts
        if "features" in symbol_df.columns:
            symbol_df = symbol_df.drop("features")
        
        # Assemble features into a single vector with a unique name
        feature_col_name = f"{symbol}_features"
        assembler = VectorAssembler(inputCols=features, outputCol=feature_col_name)
        symbol_df = assembler.transform(symbol_df)
        
        # # Split data into train, validation, and test sets
        # train_df, temp_df = symbol_df.randomSplit([0.7, 0.3], seed=42)
        # val_df, test_df = temp_df.randomSplit([0.5, 0.5], seed=42)
        
        # Train a Linear Regression model
        lr = LinearRegression(featuresCol=feature_col_name, labelCol=target_col, predictionCol="prediction")
        pipeline = Pipeline(stages=[lr])  # No need to reassemble in the pipeline
        model = pipeline.fit(symbol_df)
        
        # Evaluate on validation set
        evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse")

        # comparing models on new data
        ms_timestamp = messages[current_value-1]['timestamp']
        datetime_str = datetime.fromtimestamp(ms_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')

        # Convert '5s_interval' to timestamp if not already
        symbol_df = symbol_df.withColumn("5s_interval", F.col("5s_interval").cast("timestamp"))

        # Filter rows where '5s_interval' is greater than the datetime
        test_df = symbol_df.filter(F.col("5s_interval") > F.lit(datetime_str))


        val_predictions = model.transform(test_df)
        rmse = evaluator.evaluate(val_predictions)
        print(f"Symbol: {symbol}, Validation RMSE: {rmse}")\
        
        # check if a model is already deployed
        old_model = load_latest_model(symbol)
        if old_model == None:
            pass
        else:
            # compare metrics
            old_val_predictions = old_model.transform(test_df)
            old_rmse = evaluator.evaluate(old_val_predictions)
            print(f"Old model RMSE: {old_rmse}")
            if rmse >= old_rmse:
                print("New model is worse. Keeping the old one.")
                continue
            else:
                print("New model is better. Saving it.")
        # save the new one

        # Save the model with a timestamp
        model_path = f"models/{symbol}_model_{timestamp}"
        model.save(model_path)
        print(f"Model for {symbol} saved at {model_path}")

    # Stop the Spark session
    spark.stop()
    session.shutdown()

# if __name__ == '__main__':
main()