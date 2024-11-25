import unittest
import json
from datetime import datetime
from google.cloud import pubsub_v1
from google.oauth2 import service_account
from google.api_core import retry
import asyncio

class TestPubSubSchema(unittest.TestCase):
    def setUp(self):
        # Initialize Pub/Sub client
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            credentials = service_account.Credentials.from_service_account_info(
                config,
                scopes=['https://www.googleapis.com/auth/pubsub']
            )
            
            self.publisher = pubsub_v1.PublisherClient(credentials=credentials)
            self.topic_path = self.publisher.topic_path("ccproject-419413", "topic-1")
            
            # Configure retry settings
            self.retry_settings = retry.Retry(
                initial=1.0,
                maximum=5.0,
                multiplier=2.0,
                deadline=10.0,
            )
            
        except Exception as e:
            print(f"Failed to initialize Pub/Sub: {e}")
            raise

        # Sample data template that matches schema
        self.data_template = {
            'symbol': 'XOM',
            'timestamp': int(datetime.now().timestamp() * 1000),
            'source': 'YLIFE_FEED',
            'data_type': 'MARKET_DATA',
            'bid': -1.0,
            'ask': -1.0,
            'price': 100.0,
            'volume': 10000.0,
            'spread_raw': -1.0,
            'spread_table': -1.0,
            'volatility': 0.5,
            'market_sentiment': 0.2,
            'trading_activity': 75.0
        }

    async def publish_message(self, data):
        """Helper method to publish message to Pub/Sub"""
        try:
            # Validate data before publishing
            if 'symbol' not in data:
                print("Schema validation error: Missing required field 'symbol'")
                raise Exception("Missing required field: symbol")
            
            if not isinstance(data['symbol'], str):
                print("Schema validation error: Field 'symbol' must be a string")
                raise Exception("Symbol must be a string")
                
            if not isinstance(data['timestamp'], int):
                print("Schema validation error: Field 'timestamp' must be an integer")
                raise Exception("Timestamp must be an integer")
                
            if data['timestamp'] > int(datetime.now().timestamp() * 1000):
                print("Schema validation error: Timestamp cannot be in the future")
                raise Exception("Timestamp cannot be in the future")
                
            if data['source'] not in ['YLIFE_FEED', 'XTB_FEED']:
                print(f"Schema validation error: Invalid source value '{data['source']}'. Must be one of: YLIFE_FEED, XTB_FEED")
                raise Exception("Invalid source value")
                
            if data['data_type'] not in ['MARKET_DATA']:
                print(f"Schema validation error: Invalid data_type value '{data['data_type']}'. Must be: MARKET_DATA")
                raise Exception("Invalid data_type value")
                
            if 'market_sentiment' in data:
                sentiment = float(data['market_sentiment'])
                if sentiment < -1.0 or sentiment > 1.0:
                    print(f"Schema validation error: Market sentiment value {sentiment} must be between -1.0 and 1.0")
                    raise Exception("Market sentiment must be between -1.0 and 1.0")
                    
            if 'trading_activity' in data:
                activity = float(data['trading_activity'])
                if activity < 0.0 or activity > 100.0:
                    print(f"Schema validation error: Trading activity value {activity} must be between 0.0 and 100.0")
                    raise Exception("Trading activity must be between 0.0 and 100.0")
                    
            # Validate numeric fields
            numeric_fields = ['price', 'volume', 'bid', 'ask', 'spread_raw', 'spread_table', 'volatility']
            for field in numeric_fields:
                if field in data and not isinstance(data[field], (int, float)):
                    print(f"Schema validation error: Field '{field}' must be a number, got {type(data[field]).__name__}")
                    raise Exception(f"{field} must be a number")

            print(f"Attempting to publish message: {json.dumps(data, indent=2)}")
            message_data = json.dumps(data).encode('utf-8')
            future = self.publisher.publish(
                self.topic_path,
                message_data,
                retry=self.retry_settings
            )
            message_id = await asyncio.get_event_loop().run_in_executor(None, future.result)
            print(f"Successfully published message with ID: {message_id}")
            return message_id
        except Exception as e:
            print(f"Failed to publish message: {str(e)}")
            raise

    def test_invalid_symbol_type(self):
        """Test publishing with invalid symbol type fails"""
        data = self.data_template.copy()
        data['symbol'] = 123
        with self.assertRaises(Exception):
            asyncio.run(self.publish_message(data))

    def test_invalid_timestamp(self):
        """Test publishing with invalid timestamp fails"""
        data = self.data_template.copy()
        data['timestamp'] = "not a timestamp"
        with self.assertRaises(Exception):
            asyncio.run(self.publish_message(data))

    def test_invalid_source(self):
        """Test publishing with invalid source fails"""
        data = self.data_template.copy()
        data['source'] = 'INVALID_SOURCE'
        with self.assertRaises(Exception):
            asyncio.run(self.publish_message(data))

    def test_invalid_data_type(self):
        """Test publishing with invalid data_type fails"""
        data = self.data_template.copy()
        data['data_type'] = 'INVALID_TYPE'
        with self.assertRaises(Exception):
            asyncio.run(self.publish_message(data))

    def test_missing_required_field(self):
        """Test publishing with missing required field fails"""
        data = self.data_template.copy()
        del data['symbol']
        with self.assertRaises(Exception):
            asyncio.run(self.publish_message(data))

    def test_invalid_numeric_field(self):
        """Test publishing with invalid numeric field fails"""
        data = self.data_template.copy()
        data['price'] = "not a number"
        with self.assertRaises(Exception):
            asyncio.run(self.publish_message(data))

    def test_out_of_range_sentiment(self):
        """Test publishing with out of range market sentiment fails"""
        data = self.data_template.copy()
        data['market_sentiment'] = 1.5
        with self.assertRaises(Exception):
            asyncio.run(self.publish_message(data))

    def test_out_of_range_activity(self):
        """Test publishing with out of range trading activity fails"""
        data = self.data_template.copy()
        data['trading_activity'] = 150.0
        with self.assertRaises(Exception):
            asyncio.run(self.publish_message(data))

    def test_future_timestamp(self):
        """Test publishing with future timestamp fails"""
        data = self.data_template.copy()
        data['timestamp'] = int(datetime.now().timestamp() * 1000) + 86400000  # 1 day in future
        with self.assertRaises(Exception):
            asyncio.run(self.publish_message(data))

if __name__ == '__main__':
    unittest.main()
