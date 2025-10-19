# core/brain/modules/strategy.py

import os
import json
import asyncio
import aiohttp
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, Any, List

# 4-byte selector for TIP-3 “mint” – replace if it changes
MINT_SELECTOR = "0x01234567"

# Path to JSON config
CONFIG_PATH = os.getenv(
    "STRATEGY_CONFIG",
    os.path.join(os.path.dirname(__file__), "../../config/strategy.json")
)

@dataclass
class MintEvent:
    token_address: str
    owner_address: str
    initial_supply: int
    block_time: int
    block_seqno: int

class TonMintSniperStrategy:
    def __init__(self, rpc: str):
        self.rpc = rpc
        self._last_seqno = -1

        # load thresholds & whitelist
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)

        self.whitelist        = set(w.lower() for w in cfg["whitelist"])
        self.min_liquidity    = cfg["min_liquidity_ton"]
        self.min_volume24h    = cfg["min_volume24h_ton"]
        self.max_price_impact = cfg["max_price_impact_pct"]
        self.test_ton_amount  = cfg["test_ton_amount"]
        self.trade_size_ton   = cfg["trade_size_ton"]
        self.min_profit_pct   = cfg["min_profit_pct"]

        self.logger = logging.getLogger("TonMintSniperStrategy")
        self.session = aiohttp.ClientSession()

    async def watch_mint_events(self) -> AsyncGenerator[MintEvent, None]:
        """
        Poll the masterchain for new blocks, extract TIP-3 mint events.
        """
        while True:
            info = await self._rpc("getMasterchainInfo", {})
            seqno = info["last"]["seqno"]
            if seqno != self._last_seqno:
                block = await self._rpc("getBlock", {
                    "chain":   "masterchain",
                    "workchain": info["last"]["workchain"],
                    "shard":   "-9223372036854775808:+9223372036854775807",
                    "seqno":   seqno
                })
                for ev in self._extract_mint_events(block, seqno):
                    yield ev
                self._last_seqno = seqno

            await asyncio.sleep(1.0)

    async def _rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        JSON-RPC helper for TON.
        """
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        async with self.session.post(self.rpc, json=payload) as resp:
            data = await resp.json()
            return data["result"]

    def _extract_mint_events(self, block: Dict[str, Any], seqno: int) -> List[MintEvent]:
        """
        Scan transactions for out_msgs whose body starts with MINT_SELECTOR.
        """
        events: List[MintEvent] = []
        for tx in block.get("transactions", []):
            for msg in tx.get("out_msgs", []):
                body = msg.get("body", "")
                if body.startswith(MINT_SELECTOR):
                    events.append(MintEvent(
                        token_address=msg["dst"],
                        owner_address=msg["src"],
                        initial_supply=int(msg.get("value", 0)),
                        block_time=block.get("gen_utime", 0),
                        block_seqno=seqno
                    ))
        return events

    async def is_legit(self, ev: MintEvent) -> bool:
        """
        Filter by whitelist, liquidity, 24h volume, and price-impact.
        """
        addr = ev.token_address.lower()
        if addr in self.whitelist:
            self.logger.debug(f"{addr} in whitelist → pass")
            return True

        pool = await self._fetch_pool_info(addr)
        if pool["liquidity_ton"] < self.min_liquidity:
            self.logger.debug(f"{addr}: liquidity {pool['liquidity_ton']} < {self.min_liquidity}")
            return False

        if pool["volume24h_ton"] < self.min_volume24h:
            self.logger.debug(f"{addr}: vol24h {pool['volume24h_ton']} < {self.min_volume24h}")
            return False

        impact = await self._price_impact_pct(addr)
        if impact > self.max_price_impact:
            self.logger.debug(f"{addr}: impact {impact:.2f}% > {self.max_price_impact}%")
            return False

        self.logger.info(f"{addr} passed all filters")
        return True

    async def _fetch_pool_info(self, token: str) -> Dict[str, float]:
        """
        Query STON.fi, fallback to DeDust for:
          - token‘s TON liquidity
          - 24h volume in TON
        """
        # STON.fi endpoint
        ston_url = f"https://api.ston.fi/v1/pools/{token}"
        async with self.session.get(ston_url) as resp:
            if resp.status == 200:
                d = await resp.json()
                return {
                    "liquidity_ton": float(d["tokenReserves"]["ton"]),
                    "volume24h_ton": float(d["volume24h"]["ton"])
                }

        # DeDust fallback
        dedust_url = f"https://api.dedust.org/v1/pools/{token}"
        async with self.session.get(dedust_url) as resp:
            d = await resp.json()
            return {
                "liquidity_ton": float(d["liquidity0"]),
                "volume24h_ton": float(d["volume24h"])
            }

    async def _fetch_quote(self, token: str, ton_amount: float) -> float:
        """
        Get a simulated swap quote from STON.fi or DeDust.
        """
        # STON.fi quote
        params = {"from": "TON", "to": token, "amount": ton_amount}
        async with self.session.get("https://api.ston.fi/v1/quote", params=params) as resp:
            if resp.status == 200:
                q = await resp.json()
                return float(q["outputAmount"])

        # DeDust fallback
        params = {"inputToken": "TON", "outputToken": token, "amount": ton_amount}
        async with self.session.get("https://api.dedust.org/v1/quote", params=params) as resp:
            q = await resp.json()
            return float(q["outputAmount"])

    async def _price_impact_pct(self, token: str) -> float:
        """
        Compute % price-impact for swapping test_ton_amount.
        """
        quote_out = await self._fetch_quote(token, self.test_ton_amount)
        pool = await self._fetch_pool_info(token)
        ideal_out = self.test_ton_amount  # ideal: 1 TON → 1 TON
        if pool["liquidity_ton"] > 0:
            ideal_out = (pool["liquidity_ton"] / pool["liquidity_ton"]) * self.test_ton_amount

        if ideal_out <= 0:
            return 100.0

        return max(0.0, (ideal_out - quote_out) / ideal_out * 100)

    def build_trade_signal(self, ev: MintEvent) -> Dict[str, Any]:
        """
        Assemble final trade parameters for executor.
        """
        return {
            "token_address": ev.token_address,
            "buy_ton":      self.trade_size_ton,
            "min_profit_pct": self.min_profit_pct,
            "block_seqno":  ev.block_seqno
        }

    async def close(self):
        """
        Cleanup HTTP session.
        """
        await self.session.close()


# -------------------------
# Demo harness – run a cycle
# -------------------------
if __name__ == "__main__":
    import asyncio
    rpc_url = os.getenv("TON_RPC_URL", "https://mainnet.ton.dev")
    strat = TonMintSniperStrategy(rpc=rpc_url)

    async def demo():
        async for ev in strat.watch_mint_events():
            print("🆕 MINT EVENT:", ev)
            if await strat.is_legit(ev):
                sig = strat.build_trade_signal(ev)
                print("✅ SIGNAL:", sig)
            else:
                print("⛔️ FILTERED:", ev.token_address)
            break
        await strat.close()

    asyncio.run(demo())
