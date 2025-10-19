#!/usr/bin/env python3
import os
import asyncio
import logging
from modules.strategy import TonMintSniperStrategy
from modules.filter import TokenFilter
from modules.executor import TonExecutor
from modules.tracker import ProfitTracker

# ─── CONFIG ───────────────────────────────────────────────────────────────────

RPC_ENDPOINT   = os.getenv("TON_RPC", "https://mainnet.ton.dev")
PRIVATE_KEY    = os.getenv("TON_KEY_PATH", "./keys/ton_key.json")
MIN_PROFIT_PCT = float(os.getenv("MIN_PROFIT_PCT", 5.0))   # sell when +5%
TRADE_SIZE     = float(os.getenv("TRADE_SIZE_TON", 0.0001))# e.g., 0.0001 TON

# ─── SETUP LOGGER ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt  = "%H:%M:%S"
)
logger = logging.getLogger("PoiseTrader")

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────

async def main():
    strategy = TonMintSniperStrategy(rpc=RPC_ENDPOINT)
    token_filter = TokenFilter(whitelist_file="config/whitelist.json")
    executor = TonExecutor(rpc=RPC_ENDPOINT, key_path=PRIVATE_KEY)
    tracker = ProfitTracker(log_path="logs/trades.log")

    logger.info("🚀 PoiseTrader started: waiting for TON mints...")

    # stream mint events indefinitely
    async for mint_event in strategy.watch_mint_events():
        token = mint_event.token_address
        logger.info(f"🆕 Mint detected: {token}")

        # 1) Filter out spam tokens
        if not token_filter.is_legit(token, mint_event):
            logger.debug(f"↩️  Skipping {token} (filtered out)")
            continue

        # 2) Build a trade signal
        trade_signal = strategy.build_trade_signal(
            token_address=token,
            ton_amount=TRADE_SIZE,
            min_profit_pct=MIN_PROFIT_PCT
        )
        logger.info(f"⚡ Signal: BUY {TRADE_SIZE} TON → {token}")

        # 3) Execute swap buy ▶︎ sell cycle
        result = await executor.sniper_trade(trade_signal)
        if not result.success:
            logger.warning(f"❌ Trade failed: {result.error}")
            continue

        logger.info(f"✅ Trade succeeded: profit {result.profit_ton:.8f} TON")
        
        # 4) Record trade in log
        tracker.record(
            token=token,
            buy_tx=result.buy_tx,
            sell_tx=result.sell_tx,
            profit_ton=result.profit_ton
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 PoiseTrader stopped by user.")
