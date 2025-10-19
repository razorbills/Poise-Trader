# core/brain/modules/executor.py

import os
import logging
from tonclient.client import TonClient
from tonclient.types import (
    ClientConfig,
    Network,
    Abi,
    CallSet,
    Signer,
    ParamsOfProcessMessage
)

class DexExecutor:
    """
    Executor for swapping TON → tokens via a DEX router (STON.fi / DeDust).
    """

    def __init__(self):
        # Load configuration from environment
        self.rpc_url           = os.getenv("TON_RPC_URL")
        self.router_address    = os.getenv("DEX_ROUTER_ADDRESS")
        self.wton_address      = os.getenv("WTON_TOKEN_ADDRESS")
        pub_key                = os.getenv("WALLET_PUBLIC_KEY")
        priv_key               = os.getenv("WALLET_PRIVATE_KEY")
        abi_path               = os.getenv(
            "DEX_ROUTER_ABI_PATH",
            os.path.join(os.path.dirname(__file__), "../../../ton_abi/stonfi_router.abi.json")
        )

        # Initialize TON SDK client
        self.client = TonClient(
            config=ClientConfig(
                network=Network(server_address=self.rpc_url)
            )
        )

        # Load the router ABI
        self.dex_abi = Abi.from_path(abi_path)

        # Prepare signer
        self.signer = Signer.Keys(
            keys={
                "public": pub_key,
                "secret": priv_key
            }
        )

        # Logger
        self.logger = logging.getLogger("DexExecutor")
        self.logger.setLevel(logging.INFO)

    def _ton_to_nano(self, ton_amount: float) -> int:
        """
        Convert TON to nanotons (1 TON = 10^9 nanotons).
        """
        return int(ton_amount * 1e9)

    def _ton_wrapped(self) -> str:
        """
        Wrapped TON token (WTON) address in TIP-3 format.
        """
        return self.wton_address

    async def swap_exact_ton_for_tokens(
        self,
        token_address: str,
        ton_amount: float,
        min_amount_out: int
    ) -> dict:
        """
        Execute swapExactTONForTokens on the router.

        :param token_address: target TIP-3 token address
        :param ton_amount: amount of TON to swap (float TON)
        :param min_amount_out: minimum acceptable output amount (in token's smallest units)
        :returns: transaction details dict
        """
        amount_in = self._ton_to_nano(ton_amount)
        self.logger.info(
            f"Swapping {ton_amount} TON ({amount_in} nanotons) → {token_address}, "
            f"min_out={min_amount_out}"
        )

        call_set = CallSet(
            function_name="swapExactTONForTokens",
            input={
                "amountIn":      amount_in,
                "minAmountOut":  min_amount_out,
                "path":          [self._ton_wrapped(), token_address]
            }
        )

        params = ParamsOfProcessMessage(
            message_encode_params={
                "abi":      self.dex_abi,
                "address":  self.router_address,
                "call_set": call_set,
                "signer":   self.signer
            },
            send_events=False
        )

        # Send transaction
        result = await self.client.processing.process_message(params=params)
        tx_hash = result.transaction

        self.logger.info(f"Transaction sent: {tx_hash}")
        return {"tx_hash": tx_hash, "raw": result}

    async def close(self):
        """
        Cleanly shut down the TON client.
        """
        await self.client.close()


# -------------------------
# Example standalone usage
# -------------------------
if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    exec_ = DexExecutor()

    async def demo():
        # Example parameters (replace with real signal values)
        token_addr   = "EQC..."    # TIP-3 token address
        ton_amount   = 0.01        # 0.01 TON
        min_amount   = 9500000     # e.g. require ≥ 0.0095 tokens (in minimal units)

        res = await exec_.swap_exact_ton_for_tokens(
            token_address=token_addr,
            ton_amount=ton_amount,
            min_amount_out=min_amount
        )
        print("Swap result:", res)
        await exec_.close()

    asyncio.run(demo())
