import websockets
import json
import asyncio
import ssl
import logging
import random
from datetime import datetime
from asyncio import Queue
from google.cloud import pubsub_v1
from google.api_core import retry
from google.oauth2 import service_account

class XTBWebSocket:
    def __init__(self, userId, password, project_id, topic_id, credentials_path, app_name="Python_XTB_API"):
        self.userId = userId
        self.password = password
        self.app_name = app_name
        self.websocket = None
        self.session_id = None
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        # Using demo server by default
        self.ws_url = "wss://ws.xtb.com/demo"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Flag to control the continuous symbol requests
        self.running = False
        
        # Message queues for different types of responses
        self.symbol_queue = Queue()
        self.price_queue = Queue()

        # Initialize credentials
        try:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/pubsub"]
            )
            
            # Initialize Pub/Sub publisher with explicit credentials
            self.publisher = pubsub_v1.PublisherClient(credentials=credentials)
            self.topic_path = self.publisher.topic_path(project_id, topic_id)
            
            self.logger.info("Successfully initialized Pub/Sub publisher")
        except Exception as e:
            self.logger.error(f"Failed to initialize Pub/Sub publisher: {e}")
            raise
        
        # Configure retry settings for Pub/Sub
        self.retry_settings = retry.Retry(
            initial=1.0,
            maximum=5.0,
            multiplier=2.0,
            deadline=10.0,
        )

    async def process_market_data(self, data):
        """Process and publish market data to Pub/Sub"""
        if 'returnData' in data and isinstance(data['returnData'], dict):
            return_data = data['returnData']
            pubsub_data = {
                'symbol': return_data.get('symbol', 'N/A'),
                'timestamp': int(return_data.get('time', 0)),  # Must be numeric for int64
                'source': 'XTB_FEED',  # Enum must be valid value
                'data_type': 'MARKET_DATA',  # Enum must be valid value
                'bid': float(return_data.get('bid', -1.0)),  # Using -1.0 to indicate N/A
                'ask': float(return_data.get('ask', -1.0)),
                'price': -1.0,  # Using -1.0 to indicate N/A
                'volume': -1.0,
                'spread_raw': float(return_data.get('spreadRaw', -1.0)),
                'spread_table': float(return_data.get('spreadTable', -1.0)),
                'volatility': -1.0,
                'market_sentiment': -1.0,
                'trading_activity': -1.0
    }
            
            # Publish to Pub/Sub
            await self.publish_to_pubsub(pubsub_data)

    async def publish_to_pubsub(self, data):
        """Publish message to Pub/Sub"""
        try:
            # Convert data to JSON string
            json_string = json.dumps(data)
            
            # Encode as bytes
            message_bytes = json_string.encode('utf-8')
            
            # Publish message
            future = self.publisher.publish(
                self.topic_path,
                message_bytes,
                retry=self.retry_settings
            )
            
            # Handle the future in an async way
            message_id = asyncio.create_task(self.handle_publish_future(future, data))
            
        except Exception as e:
            self.logger.error(f"Error publishing to Pub/Sub: {e}")

    async def handle_publish_future(self, future, data):
        """Handle the future returned by publish()"""
        try:
            # Wait for the publish to complete
            message_id = await asyncio.get_event_loop().run_in_executor(
                None, future.result
            )
            self.logger.debug(f"Published message {message_id} for symbol {data.get('symbol')}")
        except Exception as e:
            self.logger.error(f"Publish failed: {e}")

    async def connect(self):
        try:
            self.websocket = await websockets.connect(
                self.ws_url,
                ssl=self.ssl_context
            )
            await self.login()
            # Start message router
            asyncio.create_task(self.message_router())
            return True
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False

    async def login(self):
        login_cmd = {
            "command": "login",
            "arguments": {
                "userId": self.userId,
                "password": self.password,
                "appName": self.app_name,
                "appId": "12345",
                "timeZone": "GMT+" + str(int((datetime.now() - datetime.utcnow()).total_seconds() / 3600))
            }
        }
        
        self.logger.info(f"Attempting login with credentials: {self.userId}")
        await self.send_command(login_cmd)
        response = await self.websocket.recv()
        response_data = json.loads(response)
        
        if "status" in response_data and response_data["status"]:
            self.session_id = response_data.get("streamSessionId")
            self.logger.info("Successfully logged in")
            return True
        else:
            self.logger.error(f"Login failed: {response_data}")
            raise Exception(f"Login failed: {response_data.get('errorDescr', 'Unknown error')}")

    async def message_router(self):
        """Routes incoming messages to appropriate queues based on message type"""
        while True:
            try:
                if self.websocket:
                    message = await self.websocket.recv()
                    data = json.loads(message)
                    
                    # Route messages based on their content
                    if "returnData" in data and "symbol" in data["returnData"]:
                        await self.symbol_queue.put(data)
                        await self.process_market_data(data)
                    elif "data" in data:
                        await self.price_queue.put(data)
                    else:
                        self.logger.debug(f"Unhandled message type: {data}")
            except websockets.exceptions.ConnectionClosed as e:
                self.logger.warning(f"Connection closed: {e}. Attempting to reconnect...")
                await self.reconnect()
                if not self.websocket:
                    break
            except Exception as e:
                self.logger.error(f"Error in message router: {e}")
                await self.reconnect()
                if not self.websocket:
                    break

    async def reconnect(self, max_attempts=3, delay=5):
        """Attempt to reconnect to the websocket"""
        attempts = 0
        while attempts < max_attempts and not self.websocket:
            try:
                self.logger.info(f"Reconnection attempt {attempts + 1}/{max_attempts}")
                success = await self.connect()
                if success:
                    self.logger.info("Successfully reconnected")
                    # Resubscribe to any active symbols
                    if self.running:
                        symbol_cmd = {
                            "command": "getSymbol",
                            "arguments": {
                                "symbol": "ETHEREUM"
                            }
                        }
                        await self.send_command(symbol_cmd)
                    return True
            except Exception as e:
                self.logger.error(f"Reconnection failed: {e}")
                attempts += 1
                if attempts < max_attempts:
                    self.logger.info(f"Waiting {delay} seconds before next attempt...")
                    await asyncio.sleep(delay)
        
        if attempts == max_attempts:
            self.logger.error("Max reconnection attempts reached")
            return False

    async def get_symbol_periodically(self, symbol):
        self.running = True
        while self.running:
            try:
                # Random delay between 3 and 6 seconds
                delay = random.uniform(3.0, 6.0)
                symbol_cmd = {
                    "command": "getSymbol",
                    "arguments": {
                        "symbol": symbol
                    }
                }
                
                await self.send_command(symbol_cmd)
                # Get response from the symbol queue
                symbol_response = await self.symbol_queue.get()
                
                # Wait for the random delay before next request
                await asyncio.sleep(delay)
                
            except Exception as e:
                self.logger.error(f"Error in periodic symbol request: {e}")
                break

    async def subscribe_to_prices(self, symbol):
        # Start periodic symbol requests in a separate task
        asyncio.create_task(self.get_symbol_periodically(symbol))
    
        # Subscribe to price updates
        cmd = {
            "command": "getTickPrices",
            "arguments": {
                "symbols": [symbol],
                "timestamp": 0,
                "level": -1
            }
        }
        await self.send_command(cmd)

    async def send_command(self, command):
        if self.websocket:
            await self.websocket.send(json.dumps(command))
        else:
            raise Exception("WebSocket not connected")

    async def listen_to_stream(self):
        while True:
            try:
                # Get price updates from the price queue
                data = await self.price_queue.get()
                if "data" in data:
                    self.logger.info(f"Received price data: {data['data']}")
                yield data
            except Exception as e:
                self.logger.error(f"Error in stream: {e}")
                break

    async def close(self):
        self.running = False  # Stop the periodic symbol requests
        if self.websocket:
            await self.websocket.close()

async def main():
    # Replace with your XTB credentials and Google Cloud settings
    client = XTBWebSocket(
        userId="17051761",
        password="PotezneBigData997",
        project_id="ccproject-419413",        # Replace with your Google Cloud project ID
        topic_id="topic-1",          # Replace with your Pub/Sub topic ID
        credentials_path="config.json"  # Replace with path to your JSON key file
    )
    
    try:
        if await client.connect():
            # Subscribe to ETHEREUM prices
            await client.subscribe_to_prices("ETHEREUM")
            
            # Keep the connection alive indefinitely
            while True:
                await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())