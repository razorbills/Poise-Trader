# core/brain/feeds/market_data.py
import asyncio
import websockets
import json
from utils.logger import get_logger

logger = get_logger("MarketFeed")

async def listen_markets(queue: asyncio.Queue):
    uri = 'wss://stream.binance.com:9443/ws/btcusdt@trade'
    while True:
        try:
            async with websockets.connect(uri) as ws:
                logger.info("[Market] Connected to Binance BTC/USDT feed")
                while True:
                    msg = await ws.recv()
                    trade = json.loads(msg)
                    await queue.put(('market', trade))
        except Exception as e:
            logger.error(f"[Market] Connection error: {e}, reconnecting in 5s...")
            await asyncio.sleep(5)
