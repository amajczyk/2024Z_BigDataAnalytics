import websockets
import json
import asyncio
import ssl
import logging
from asyncio import Queue
from confluent_kafka import Producer
from datetime import datetime
import random
import backoff

class XTBWebSocket:
    def __init__(
        self,
        userId,
        password,
        kafka_bootstrap_servers,
        kafka_topic,
        app_name="Python_XTB_API",
    ):
        # Basic configuration
        self.userId = userId
        self.password = password
        self.app_name = app_name
        self.websocket = None
        self.session_id = None

        # SSL configuration
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

        # Server URL
        self.ws_url = "wss://ws.xtb.com/demo"

        # Logging setup
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

        # State flags
        self.running = False
        self.is_connected = False
        self._closing = False  # Flag for graceful shutdown
        self.reconnection_task = None

        # Message queues
        self.symbol_queue = Queue()
        self.price_queue = Queue()

        # Kafka configuration
        self.kafka_topic = kafka_topic
        try:
            self.kafka_producer = Producer(
                {"bootstrap.servers": kafka_bootstrap_servers}
            )
            self.logger.info("Successfully initialized Kafka producer")
        except Exception as e:
            self.logger.error(f"Failed to initialize Kafka producer: {e}")
            raise

    def is_websocket_open(self):
        """Check if websocket connection is open and valid"""
        return (
            self.websocket is not None and 
            hasattr(self.websocket, 'transport') and
            self.websocket.transport is not None and
            not self._closing
        )

    async def process_market_data(self, data):
        """Process and send market data to Kafka"""
        if "returnData" in data and isinstance(data["returnData"], dict):
            return_data = data["returnData"]
            kafka_data = {
                "symbol": return_data.get("symbol", "N/A"),
                "timestamp": int(return_data.get("time", 0)),
                "source": "XTB_FEED",
                "data_type": "MARKET_DATA",
                "bid": float(return_data.get("bid", -1.0)),
                "ask": float(return_data.get("ask", -1.0)),
                "price": -1.0,
                "volume": -1.0,
                "spread_raw": float(return_data.get("spreadRaw", -1.0)),
                "spread_table": float(return_data.get("spreadTable", -1.0)),
                "volatility": -1.0,
                "market_sentiment": -1.0,
                "trading_activity": -1.0,
            }
            await self.send_to_kafka(kafka_data)

    async def send_to_kafka(self, data):
        """Send message to Kafka"""
        try:
            json_string = json.dumps(data)
            topics = [self.kafka_topic, "model-topic", "model-topic-eth"]
            for topic in topics:
                self.kafka_producer.produce(
                    topic,
                    key=data["symbol"],
                    value=json_string,
                    callback=self.delivery_report
                )
            self.logger.debug(f"Sent to Kafka topics: {', '.join(topics)}: {json_string}")
        except Exception as e:
            self.logger.error(f"Error sending to Kafka: {e}")

    def delivery_report(self, err, msg):
        """Called once for each message produced to indicate delivery result."""
        if err is not None:
            self.logger.error(f"Message delivery failed: {err}")
        else:
            self.logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

    async def connect(self):
        """Connect to the WebSocket"""
        try:
            self.websocket = await websockets.connect(
                self.ws_url,
                ssl=self.ssl_context,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10,
                max_size=10_000_000
            )
            await self.login()
            self.is_connected = True
            asyncio.create_task(self.message_router())
            asyncio.create_task(self.connection_healthcheck())
            return True
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            self.is_connected = False
            return False

    async def login(self):
        """Log in to the XTB WebSocket server."""
        try:
            login_request = {
                "command": "login",
                "arguments": {
                    "userId": self.userId,
                    "password": self.password,
                    "appName": self.app_name,
                    "appId": str(random.randint(100000, 999999)),
                    "timeZone": "GMT+0"
                },
            }
            await self.websocket.send(json.dumps(login_request))
            response = await self.websocket.recv()
            response_data = json.loads(response)

            if response_data.get("status"):
                self.session_id = response_data.get("streamSessionId")
                self.logger.info(f"Login successful. Session ID: {self.session_id}")
            else:
                self.logger.error(f"Login failed: {response_data}")
                raise Exception("Login failed")
        except Exception as e:
            self.logger.error(f"Error during login: {e}")
            raise

    async def connection_healthcheck(self):
        """Monitor connection health"""
        while not self._closing:
            try:
                if self.is_websocket_open():
                    try:
                        pong_waiter = await self.websocket.ping()
                        await asyncio.wait_for(pong_waiter, timeout=10)
                    except Exception as e:
                        self.logger.warning(f"Ping failed: {e}")
                        await self.reconnect()
                else:
                    self.logger.warning("Connection appears to be closed, initiating reconnect...")
                    await self.reconnect()
            except Exception as e:
                self.logger.error(f"Healthcheck failed: {e}")
                if not self._closing:
                    await self.reconnect()
            await asyncio.sleep(30)

    @backoff.on_exception(
        backoff.expo,
        (websockets.ConnectionClosed, ConnectionError),
        max_tries=5,
        max_time=300
    )
    async def reconnect(self):
        """Attempt to reconnect with proper cleanup"""
        if self.reconnection_task and not self.reconnection_task.done():
            self.logger.debug("Reconnection already in progress")
            return

        try:
            self.logger.info("Starting reconnection process")
            if self.is_websocket_open():
                try:
                    await self.websocket.close()
                except Exception as e:
                    self.logger.warning(f"Error closing existing connection: {e}")

            self.is_connected = False
            self.websocket = None
            
            self.websocket = await websockets.connect(
                self.ws_url,
                ssl=self.ssl_context,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            await self.login()
            
            if self.running:
                await self.restore_subscriptions()
            
            self.is_connected = True
            asyncio.create_task(self.message_router())
            self.logger.info("Successfully reconnected")
            return True
            
        except Exception as e:
            self.logger.error(f"Reconnection attempt failed: {e}")
            raise

    async def restore_subscriptions(self):
        """Restore subscriptions after reconnection"""
        try:
            symbol_cmd = {
                "command": "getSymbol",
                "arguments": {
                    "symbol": "ETHEREUM"
                }
            }
            await self.send_command(symbol_cmd)
            self.logger.info("Successfully restored subscriptions")
        except Exception as e:
            self.logger.error(f"Failed to restore subscriptions: {e}")
            raise

    async def subscribe_to_prices(self, symbol):
        """Subscribe to price updates"""
        try:
            asyncio.create_task(self.get_symbol_periodically(symbol))
            subscribe_request = {
                "command": "getTickPrices",
                "arguments": {
                    "symbols": [symbol],
                    "timestamp": 0,
                    "level": -1
                }
            }
            await self.send_command(subscribe_request)
            self.logger.info(f"Subscribed to tick prices for symbol {symbol}")
        except Exception as e:
            self.logger.error(f"Error subscribing to prices for {symbol}: {e}")
            raise

    async def get_symbol_periodically(self, symbol):
        """Periodically request symbol data"""
        self.running = True
        while self.running and not self._closing:
            try:
                delay = random.uniform(3.0, 6.0)
                symbol_cmd = {
                    "command": "getSymbol",
                    "arguments": {
                        "symbol": symbol
                    }
                }

                await self.send_command(symbol_cmd)
                symbol_response = await self.symbol_queue.get()
                await self.process_market_data(symbol_response)
                await asyncio.sleep(delay)

            except Exception as e:
                self.logger.error(f"Error in periodic symbol request: {e}")
                if not self._closing:
                    await asyncio.sleep(1)

    async def message_router(self):
        """Route incoming WebSocket messages"""
        while not self._closing:
            try:
                if not self.is_websocket_open():
                    await asyncio.sleep(1)
                    continue

                message = await self.websocket.recv()
                data = json.loads(message)
                
                if "returnData" in data and "symbol" in data["returnData"]:
                    await self.symbol_queue.put(data)
                elif "data" in data:
                    await self.price_queue.put(data)
                else:
                    self.logger.debug(f"Unhandled message type: {data}")

                while not self.price_queue.empty():
                    price_data = await self.price_queue.get()
                    await self.process_market_data(price_data)
                    
            except websockets.ConnectionClosed as e:
                if not self._closing:
                    self.logger.warning(f"Connection closed: {e}")
                    await asyncio.sleep(1)
                    await self.reconnect()
            except Exception as e:
                if not self._closing:
                    self.logger.error(f"Error in message router: {e}")
                    await asyncio.sleep(1)

    async def send_command(self, command):
        """Send command with retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                if not self.is_websocket_open():
                    await self.reconnect()
                    
                await self.websocket.send(json.dumps(command))
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"Command send failed, attempt {attempt + 1}/{max_retries}: {e}")
                await asyncio.sleep(retry_delay * (attempt + 1))

    async def close(self):
        """Clean up resources"""
        self._closing = True
        self.running = False
        
        if self.websocket and hasattr(self.websocket, 'transport') and self.websocket.transport:
            try:
                await self.websocket.close()
            except Exception as e:
                self.logger.warning(f"Error during websocket closure: {e}")
        
        self.kafka_producer.flush()

async def main():
    client = XTBWebSocket(
        userId="17051761",
        password="PotezneBigData997",
        kafka_bootstrap_servers="localhost:9092",
        kafka_topic="test-topic",
    )

    try:
        if await client.connect():
            await client.subscribe_to_prices("ETHEREUM")
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
