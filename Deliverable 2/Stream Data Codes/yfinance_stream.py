import yfinance as yf
import yliveticker
import time
from datetime import datetime
import random
from collections import defaultdict
import threading
import json
from queue import Queue
from google.cloud import pubsub_v1
from google.oauth2 import service_account
import os
import requests

def write_to_hdfs_json(hdfs_host, folder_path, file_name, results, hdfs_user="adam_majczyk2001"):
    """
    Write scraper results to HDFS as a single JSON file.
    
    Args:
    - hdfs_host (str): The WebHDFS host URL.
    - folder_path (str): The folder path in HDFS where the file will be stored.
    - file_name (str): The name of the JSON file.
    - results (dict): Scraper results as a dictionary.
    - hdfs_user (str): The HDFS user to use for the operation.
    """
    # Convert results to a JSON string
    json_data = json.dumps(results, ensure_ascii=False, indent=4)
    
    # Ensure the folder exists in HDFS
    ensure_hdfs_folder(hdfs_host, folder_path, hdfs_user)
    
    # Upload the JSON file to HDFS
    upload_to_hdfs(hdfs_host, folder_path, file_name, json_data, hdfs_user)

def upload_to_hdfs(hdfs_host, folder_path, file_name, data, hdfs_user):
    """
    Upload a file to HDFS using WebHDFS.
    
    Args:
    - hdfs_host (str): The WebHDFS host URL.
    - folder_path (str): The HDFS folder path.
    - file_name (str): The name of the file to upload.
    - data (str): The content to write to the file.
    - hdfs_user (str): The HDFS user to use for the operation.
    """
    file_path = f"{folder_path}{file_name}"
    create_url = f"{hdfs_host}/webhdfs/v1{file_path}?op=CREATE&overwrite=true&user.name={hdfs_user}"
    response = requests.put(create_url, allow_redirects=False)
    if response.status_code == 307:
        # Follow the redirect URL
        redirect_url = response.headers['Location'].replace(
            "hadoop.us-central1-a.c.ccproject-419413.internal", hdfs_host
        )
        final_response = requests.put(redirect_url, data=data.encode('utf-8'))
        if final_response.status_code == 201:
            print(f"Successfully wrote JSON file to {file_path}")
        else:
            print(f"Failed to write JSON file to {file_path}. Status: {final_response.status_code}, Response: {final_response.text}")
    else:
        print(f"Failed to create file at {file_path}. Status: {response.status_code}, Response: {response.text}")

def ensure_hdfs_folder(hdfs_host, folder_path, hdfs_user):
    try:
        url = f"{hdfs_host}/webhdfs/v1{folder_path}?op=MKDIRS&user.name={hdfs_user}"
        response = requests.put(url)
        if response.status_code in [200, 201]:
            print(f"HDFS folder {folder_path} created/verified successfully (existing content preserved)")
        else:
            print(f"Failed to create HDFS folder: {response.status_code}")
    except Exception as e:
        print(f"Error ensuring HDFS folder exists: {e}")

class StockTracker:
    def __init__(self, tickers):
        self.tickers = tickers
        self.stock_data = defaultdict(list)
        self.last_prices = {}
        self.lock = threading.Lock()
        self.message_queue = Queue()
        self.project_id = "ccproject-419413"
        self.topic_id = "topic-1"
        
        # Initialize Pub/Sub (with fallback to no publishing if setup fails)
        self.publisher = None
        self.topic_path = None
        self.setup_pubsub()
        
        # Initialize with current prices and other metrics
        self.initialize_stocks()
        
    def setup_pubsub(self):
        try:
            # Load config file
            with open('config.json', 'r') as f:
                self.config = json.load(f)
            
            # Use config file directly as credentials
            credentials = service_account.Credentials.from_service_account_info(
                self.config,
                scopes=['https://www.googleapis.com/auth/pubsub']
            )
            
            # Initialize Pub/Sub publisher
            self.publisher = pubsub_v1.PublisherClient(credentials=credentials)
            self.topic_path = self.publisher.topic_path(
                self.project_id,
                self.topic_id
            )
            print("Successfully initialized Pub/Sub publisher")
            
        except FileNotFoundError as e:
            print(f"Warning: Configuration file not found: {e}")
            print("Running without Pub/Sub integration")
        except Exception as e:
            print(f"Warning: Failed to initialize Pub/Sub: {e}")
            print("Running without Pub/Sub integration")
            self.publisher = None
            self.topic_path = None

    def initialize_stocks(self):
        print("Initializing current stock data...")
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                self.last_prices[ticker] = {
                    'price': stock.info.get('currentPrice', 100),
                    'volume': stock.info.get('volume', 100000),
                    'volatility': random.uniform(0.2, 1.5),
                    'bid_ask_spread': random.uniform(0.01, 0.05),
                    'market_sentiment': random.uniform(-1, 1),
                    'trading_activity': random.uniform(0, 100)
                }
                print(f"Initial {ticker}: ${self.last_prices[ticker]['price']:.2f}")
            except Exception as e:
                print(f"Error getting initial price for {ticker}: {e}")
                self.last_prices[ticker] = {
                    'price': 100,
                    'volume': 100000,
                    'volatility': 1.0,
                    'bid_ask_spread': 0.02,
                    'market_sentiment': 0,
                    'trading_activity': 50
                }

    def publish_to_pubsub(self, ticker, data):
        if not self.publisher or not self.topic_path:
            return
            
        try:
            # Convert data to desired format
            pubsub_data = {
                'symbol': ticker,
                'timestamp': int(datetime.fromisoformat(data['timestamp']).timestamp() * 1000), # Convert to milliseconds
                'source': 'YLIFE_FEED', # Match enum value
                'data_type': 'MARKET_DATA', # Already matches enum
                'bid': -1.0,
                'ask': -1.0,
                'price': float(data['price']),
                'volume': float(data['volume']),
                'spread_raw': -1.0,
                'spread_table': -1.0,
                'volatility': float(data['volatility']),
                'market_sentiment': float(data['market_sentiment']),
                'trading_activity': float(data['trading_activity'])
            }
            
            # Convert to JSON string and encode as bytes
            message_data = json.dumps(pubsub_data).encode('utf-8')
            
            # Publish message
            print(f"Publishing message to Pub/Sub for {ticker}...")
            future = self.publisher.publish(self.topic_path, message_data)
            future.result()  # Wait for message to be published
            print(f"Successfully published message to Pub/Sub for {ticker}")
            
        except Exception as e:
            print(f"Error publishing to Pub/Sub: {e}")

    def on_ticker_update(self, ws, msg):
        self.message_queue.put(msg)

    def process_message_queue(self):
        while True:
            if not self.message_queue.empty():
                msg = self.message_queue.get()
                with self.lock:
                    ticker = msg.get('id', '')
                    if ticker in self.tickers:
                        price = msg.get('price', self.last_prices[ticker]['price'])
                        self.last_prices[ticker]['price'] = price
                        self.add_stock_update(ticker, 'real')
            time.sleep(0.01)

    def simulate_metrics(self, ticker):
        last_data = self.last_prices[ticker]
        
        # Simulate price change
        price_change = random.uniform(-0.5, 0.5) * (last_data['price'] * 0.01)
        new_price = max(0.01, last_data['price'] + price_change)
        
        # Simulate other metrics with different patterns
        new_volume = last_data['volume'] * random.uniform(0.95, 1.05)
        new_volatility = max(0.1, min(3.0, last_data['volatility'] * random.uniform(0.98, 1.02)))
        new_spread = max(0.01, min(0.2, last_data['bid_ask_spread'] * random.uniform(0.95, 1.05)))
        new_sentiment = max(-1, min(1, last_data['market_sentiment'] + random.uniform(-0.1, 0.1)))
        new_activity = max(0, min(100, last_data['trading_activity'] + random.uniform(-5, 5)))

        self.last_prices[ticker] = {
            'price': new_price,
            'volume': new_volume,
            'volatility': new_volatility,
            'bid_ask_spread': new_spread,
            'market_sentiment': new_sentiment,
            'trading_activity': new_activity
        }

    def add_stock_update(self, ticker, source):
        data = self.last_prices[ticker]
        update_data = {
            'price': round(data['price'], 2),
            'volume': int(data['volume']),
            'volatility': round(data['volatility'], 3),
            'bid_ask_spread': round(data['bid_ask_spread'], 4),
            'market_sentiment': round(data['market_sentiment'], 3),
            'trading_activity': round(data['trading_activity'], 2),
            'timestamp': datetime.now().isoformat(),
            'source': source
        }
        self.stock_data[ticker].append(update_data)
        
        # Publish to Pub/Sub
        self.publish_to_pubsub(ticker, update_data)

    def print_latest_update(self, ticker):
        if ticker in self.stock_data and self.stock_data[ticker]:
            latest = self.stock_data[ticker][-1]
            print(f"\n=== {ticker} Update ===")
            print(f"Time: {latest['timestamp']}")
            print(f"Price: ${latest['price']:.2f}")
            print(f"Volume: {latest['volume']:,}")
            print(f"Volatility: {latest['volatility']:.3f}")
            print(f"Bid-Ask Spread: {latest['bid_ask_spread']:.4f}")
            print(f"Market Sentiment: {latest['market_sentiment']:.3f}")
            print(f"Trading Activity: {latest['trading_activity']:.2f}")
            print(f"Source: {latest['source']}")
            print("==================\n")

    def clear_old_data(self, max_age_seconds=300):  # Keep 5 minutes of history
        with self.lock:
            current_time = datetime.now()
            for ticker in self.tickers:
                self.stock_data[ticker] = [
                    data for data in self.stock_data[ticker]
                    if (current_time - datetime.fromisoformat(data['timestamp'])).total_seconds() <= max_age_seconds
                ]

    def simulate_random_update(self):
        ticker = random.choice(self.tickers)
        self.simulate_metrics(ticker)
        with self.lock:
            self.add_stock_update(ticker, 'simulated')
            self.print_latest_update(ticker)

    def run(self):
        # HDFS configuration
        hdfs_host = "http://34.67.32.69:50070/"
        folder_path = "/user/adam_majczyk2001/nifi/bronze/yfinance/"
        ensure_hdfs_folder(hdfs_host=hdfs_host, folder_path=folder_path, hdfs_user="adam_majczyk2001")

        # Start websocket in a separate thread
        websocket_thread = threading.Thread(target=lambda: yliveticker.YLiveTicker(
            on_ticker=self.on_ticker_update,
            ticker_names=self.tickers
        ))
        websocket_thread.daemon = True
        websocket_thread.start()

        # Start message processing thread
        process_thread = threading.Thread(target=self.process_message_queue)
        process_thread.daemon = True
        process_thread.start()

        # Initialize timer for HDFS updates
        last_hdfs_update = datetime.now()

        # Main loop for simulating random updates
        try:
            while True:
                # Random interval between 0.1 and 2 seconds
                time.sleep(random.uniform(0.1, 2.0))
                self.simulate_random_update()
                self.clear_old_data()

                # Check if 10 minutes have passed
                current_time = datetime.now()
                if (current_time - last_hdfs_update).total_seconds() >= 600:  # 10 minutes
                    # Prepare data for HDFS
                    hdfs_data = {
                        'timestamp': current_time.isoformat(),
                        'updates': {ticker: self.stock_data[ticker] for ticker in self.tickers}
                    }
                    
                    # Create filename with timestamp
                    filename = f"stock_updates_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
                    
                    try:
                        # Write to HDFS using the provided function
                        write_to_hdfs_json(hdfs_host, folder_path, filename, hdfs_data, hdfs_user="adam_majczyk2001")
                        print(f"Successfully wrote updates to HDFS: {filename}")
                        # Reinitialize stock prices after successful HDFS update
                        print("Reinitializing stock prices...")
                        self.initialize_stocks()
                    except Exception as e:
                        print(f"Error writing to HDFS: {e}")
                    
                    # Reset timer
                    last_hdfs_update = current_time

        except KeyboardInterrupt:
            print("\nStopping stock tracker...")

if __name__ == "__main__":
    tickers = ["XOM", "BP", "SHEL", "COP"]
    tracker = StockTracker(tickers)
    tracker.run()
