#!/usr/bin/env python3
"""
🔥 LIVE PAPER TRADING TEST
Test paper trading with REAL live MEXC prices instead of demo data
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import random
import sys
from pathlib import Path

# Add core modules to path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
import os
load_dotenv()

# MEXC API Configuration
MEXC_API_KEY = "mx0vglVSHm8sh7Nnvd"
MEXC_SECRET_KEY = "cb416a71d0ba45298eb1383dc7896a18"
PAPER_TRADING_MODE = True  # Set to False for real trading

class LiveMexcDataFeed:
    """Live MEXC data feed for paper trading"""
    
    def __init__(self):
        self.base_url = "https://api.mexc.com"
        self.prices_cache = {}
        self.last_update = None
        
    async def get_live_price(self, symbol: str) -> float:
        """Get current live price from MEXC"""
        
        # Convert symbol format (BTC/USDT -> BTCUSDT)
        mexc_symbol = symbol.replace('/', '')
        
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.base_url}/api/v3/ticker/24hr?symbol={mexc_symbol}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        price = float(data['lastPrice'])
                        
                        # Cache the price
                        self.prices_cache[symbol] = {
                            'price': price,
                            'timestamp': datetime.now(),
                            'change_24h': float(data['priceChangePercent'])
                        }
                        
                        return price
                    else:
                        print(f"⚠️ Failed to get {symbol} price: {response.status}")
                        return None
                        
            except Exception as e:
                print(f"❌ Error getting {symbol} price: {e}")
                return None
    
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get multiple live prices at once"""
        prices = {}
        
        for symbol in symbols:
            price = await self.get_live_price(symbol)
            if price:
                prices[symbol] = price
                
        return prices

class LivePaperTradingManager:
    """Paper trading manager using LIVE market prices"""
    
    def __init__(self, initial_capital: float = 5000.0):
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.positions = {}
        self.trade_history = []
        self.data_feed = LiveMexcDataFeed()
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        
        print(f"🔥 LIVE Paper Trading Manager initialized with ${initial_capital:,.2f}")
        print("📡 Using REAL-TIME MEXC market prices!")
    
    async def execute_live_trade(self, symbol: str, action: str, amount_usd: float, strategy: str = "test", stop_loss: float = None, take_profit: float = None, *args, **kwargs):
        """Execute trade using live market prices.
        Accepts optional stop_loss/take_profit and extra args for compatibility.
        """
        
        print(f"\n🎯 EXECUTING LIVE TRADE: {action} ${amount_usd} of {symbol}")
        
        # Get current live price
        current_price = await self.data_feed.get_live_price(symbol)
        
        if not current_price:
            return {"success": False, "error": "Could not get live price"}
        
        print(f"📈 LIVE {symbol} Price: ${current_price:,.2f}")
        
        # Add realistic slippage (0.1-0.5%)
        slippage_pct = random.uniform(0.001, 0.005)
        if action.upper() == "BUY":
            execution_price = current_price * (1 + slippage_pct)
        else:
            execution_price = current_price * (1 - slippage_pct)
        
        # Calculate quantity
        if action.upper() == "BUY":
            if amount_usd > self.cash_balance:
                return {"success": False, "error": f"Insufficient funds: ${self.cash_balance:.2f}"}
            
            commission = amount_usd * 0.001  # 0.1% commission
            net_amount = amount_usd - commission
            quantity = net_amount / execution_price
            
            # Update positions
            if symbol not in self.positions:
                self.positions[symbol] = {"quantity": 0, "avg_price": 0, "total_cost": 0}
            
            old_quantity = self.positions[symbol]["quantity"]
            old_cost = self.positions[symbol]["total_cost"]
            
            new_quantity = old_quantity + quantity
            new_cost = old_cost + net_amount
            
            self.positions[symbol] = {
                "quantity": new_quantity,
                "avg_price": new_cost / new_quantity if new_quantity > 0 else 0,
                "total_cost": new_cost
            }
            
            self.cash_balance -= amount_usd
            
        else:  # SELL
            if symbol not in self.positions or self.positions[symbol]["quantity"] <= 0:
                return {"success": False, "error": f"No {symbol} position to sell"}
            
            # Calculate quantity to sell
            max_sell_value = self.positions[symbol]["quantity"] * execution_price
            if amount_usd > max_sell_value:
                amount_usd = max_sell_value
            
            quantity = amount_usd / execution_price
            commission = amount_usd * 0.001
            net_proceeds = amount_usd - commission
            
            # Update position
            self.positions[symbol]["quantity"] -= quantity
            cost_basis = (amount_usd / execution_price) * self.positions[symbol]["avg_price"]
            self.positions[symbol]["total_cost"] -= cost_basis
            
            if self.positions[symbol]["quantity"] <= 0.0001:  # Close position
                self.positions[symbol] = {"quantity": 0, "avg_price": 0, "total_cost": 0}
            
            self.cash_balance += net_proceeds
        
        # Record trade
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action.upper(),
            "amount_usd": amount_usd,
            "quantity": quantity,
            "live_price": current_price,
            "execution_price": execution_price,
            "slippage_pct": slippage_pct * 100,
            "commission": commission,
            "strategy": strategy,
            "success": True
        }
        
        self.trade_history.append(trade_record)
        self.total_trades += 1
        
        print(f"✅ TRADE EXECUTED:")
        print(f"   💰 Quantity: {quantity:.6f} {symbol.split('/')[0]}")
        print(f"   💵 Price: ${execution_price:,.2f} (slippage: {slippage_pct*100:.2f}%)")
        print(f"   💸 Commission: ${commission:.2f}")
        print(f"   🏦 Cash Balance: ${self.cash_balance:,.2f}")
        
        return {"success": True, "trade": trade_record}
    
    async def get_portfolio_value(self):
        """Calculate total portfolio value using live prices"""
        
        portfolio_value = self.cash_balance
        position_values = {}
        
        # Get live prices for all positions
        symbols_with_positions = [symbol for symbol, pos in self.positions.items() if pos["quantity"] > 0]
        
        if symbols_with_positions:
            live_prices = await self.data_feed.get_multiple_prices(symbols_with_positions)
            
            for symbol, position in self.positions.items():
                if position["quantity"] > 0 and symbol in live_prices:
                    current_value = position["quantity"] * live_prices[symbol]
                    portfolio_value += current_value
                    position_values[symbol] = {
                        "quantity": position["quantity"],
                        "current_price": live_prices[symbol],
                        "current_value": current_value,
                        "cost_basis": position["total_cost"],
                        "unrealized_pnl": current_value - position["total_cost"]
                    }
        
        return {
            "total_value": portfolio_value,
            "cash": self.cash_balance,
            "positions": position_values,
            "total_return": (portfolio_value - self.initial_capital) / self.initial_capital,
            "total_pnl": portfolio_value - self.initial_capital
        }
    
    def get_portfolio_value_sync(self):
        """SYNC version for dashboard - uses last known/avg prices instead of live lookup"""
        portfolio_value = self.cash_balance
        position_values = {}
        
        for symbol, position in self.positions.items():
            if position["quantity"] > 0:
                # Use average price (cost basis) for sync calculation
                current_price = position["avg_price"]
                current_value = position["quantity"] * current_price
                portfolio_value += current_value
                position_values[symbol] = {
                    "quantity": position["quantity"],
                    "current_price": current_price,
                    "current_value": current_value,
                    "cost_basis": position["total_cost"],
                    "unrealized_pnl": current_value - position["total_cost"]
                }
        
        return {
            "total_value": portfolio_value,
            "cash": self.cash_balance,
            "positions": position_values,
            "total_return": (portfolio_value - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0,
            "total_pnl": portfolio_value - self.initial_capital
        }

async def run_live_paper_trading_test():
    """Test paper trading with live MEXC prices"""
    
    print("🚀 LIVE PAPER TRADING TEST")
    print("🔥 Using REAL-TIME market prices from MEXC!")
    print("=" * 60)
    
    # Initialize live paper trading manager
    trader = LivePaperTradingManager(5000.0)
    
    # Test symbols with live prices
    test_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]
    
    print(f"\n📊 CURRENT LIVE PRICES:")
    live_prices = await trader.data_feed.get_multiple_prices(test_symbols)
    for symbol, price in live_prices.items():
        print(f"   {symbol:10} ${price:>10,.2f}")
    
    print(f"\n🎯 EXECUTING TEST TRADES WITH LIVE PRICES:")
    
    # Test 1: Buy some BTC with live price
    await trader.execute_live_trade("BTC/USDT", "BUY", 1000, "live_test_btc")
    
    # Test 2: Buy some ETH with live price
    await trader.execute_live_trade("ETH/USDT", "BUY", 1500, "live_test_eth")
    
    # Test 3: Buy some SOL with live price
    await trader.execute_live_trade("SOL/USDT", "BUY", 800, "live_test_sol")
    
    # Test 4: Sell some ETH
    await trader.execute_live_trade("ETH/USDT", "SELL", 500, "live_test_eth_sell")
    
    print(f"\n📈 FINAL PORTFOLIO (with LIVE prices):")
    portfolio = await trader.get_portfolio_value()
    
    print(f"💰 Total Portfolio Value: ${portfolio['total_value']:,.2f}")
    print(f"💵 Cash Balance: ${portfolio['cash']:,.2f}")
    print(f"📈 Total Return: {portfolio['total_return']*100:+.2f}%")
    print(f"💎 Total P&L: ${portfolio['total_pnl']:+,.2f}")
    
    print(f"\n🎯 ACTIVE POSITIONS (with LIVE market values):")
    for symbol, pos in portfolio['positions'].items():
        pnl_pct = (pos['unrealized_pnl'] / pos['cost_basis']) * 100 if pos['cost_basis'] > 0 else 0
        print(f"   {symbol:10} {pos['quantity']:>8.4f} @ ${pos['current_price']:>8,.2f} = ${pos['current_value']:>8,.2f} (PnL: ${pos['unrealized_pnl']:+7,.2f} / {pnl_pct:+5.1f}%)")
    
    print(f"\n🔥 COMPARISON:")
    print(f"   Old Paper Trading: Used 2022 prices (BTC ~$42k)")
    print(f"   Live Paper Trading: Uses 2025 prices (BTC ~$111k)")
    print(f"   This is the REAL market data your bot should use!")
    
    return portfolio

if __name__ == "__main__":
    asyncio.run(run_live_paper_trading_test())
