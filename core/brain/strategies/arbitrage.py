# core/brain/strategies/arbitrage.py
async def check_arbitrage(mempool_tx, market_trade):
    # Extract price
    price = float(market_trade.get('p', 0))
    outs = mempool_tx.get('x', {}).get('out', [])
    amount_btc = sum(o.get('value', 0) for o in outs) / 1e8

    # Very simple: look for >=1 BTC transfers, log it if price > threshold
    if amount_btc >= 1:
        return {
            'strategy': 'mempool_watch',
            'pair': 'BTC/USDT',
            'side': 'observe',
            'amount': amount_btc,
            'price': price
        }
    return None
