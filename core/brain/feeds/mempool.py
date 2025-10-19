# core/brain/feeds/mempool.py
import asyncio
import websockets
import json
from utils.logger import get_logger

logger = get_logger("MempoolFeed")

async def listen_mempool(queue: asyncio.Queue):
    uri = 'wss://ws.blockchain.info/inv'
    while True:
        try:
            async with websockets.connect(uri) as ws:
                await ws.send(json.dumps({"op": "unconfirmed_sub"}))
                logger.info("[Mempool] Connected to BTC mempool feed")
                while True:
                    msg = await ws.recv()
                    tx = json.loads(msg)
                    await queue.put(('mempool', tx))
        except Exception as e:
            logger.error(f"[Mempool] Connection error: {e}, reconnecting in 5s...")
            await asyncio.sleep(5)
