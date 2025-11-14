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
import requests
import urllib3
import os

# Disable SSL warnings for requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
    """Live MEXC data feed with comprehensive market data for AI learning"""
    
    def __init__(self):
        self.base_url = "https://api.mexc.com"
        self.prices_cache = {}
        self.orderbook_cache = {}  # Order book depth data
        self.trades_cache = {}  # Recent trades
        self.klines_cache = {}  # OHLCV candles
        self.ticker_24h_cache = {}  # 24h statistics
        self.last_update = None
        # Try alternative endpoints if main one fails
        self.alternative_urls = [
            "https://api.mexc.com",
            "https://www.mexc.com",
            "https://contract.mexc.com"
        ]
        print("📊 Enhanced MEXC Data Feed initialized - collecting comprehensive market data")
        
    async def get_live_price(self, symbol: str) -> float:
        """Get current live price from MEXC using requests (works reliably)"""
        
        # Convert symbol format (BTC/USDT -> BTCUSDT)
        mexc_symbol = symbol.replace('/', '')
        
        # Use requests library which works reliably
        try:
            url = f"{self.base_url}/api/v3/ticker/price?symbol={mexc_symbol}"
            response = requests.get(url, timeout=5, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                price = float(data['price'])
                
                # Cache the price
                self.prices_cache[symbol] = {
                    'price': price,
                    'timestamp': datetime.now()
                }
                
                return price
            else:
                print(f"⚠️ Failed to get {symbol} price: Status {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error getting {symbol} price: {str(e)[:50]}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error getting {symbol} price: {str(e)[:50]}")
            return None
    
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get multiple live prices at once"""
        prices = {}
        
        for symbol in symbols:
            price = await self.get_live_price(symbol)
            if price:
                prices[symbol] = price
                
        return prices
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Get order book depth data - critical for AI learning!"""
        mexc_symbol = symbol.replace('/', '')
        
        try:
            url = f"{self.base_url}/api/v3/depth?symbol={mexc_symbol}&limit={limit}"
            response = requests.get(url, timeout=5, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                
                # Process orderbook data
                orderbook = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'bids': [[float(p), float(q)] for p, q in data.get('bids', [])[:limit]],
                    'asks': [[float(p), float(q)] for p, q in data.get('asks', [])[:limit]],
                }
                
                # Calculate orderbook metrics for AI
                if orderbook['bids'] and orderbook['asks']:
                    orderbook['spread'] = orderbook['asks'][0][0] - orderbook['bids'][0][0]
                    orderbook['spread_pct'] = (orderbook['spread'] / orderbook['bids'][0][0]) * 100
                    orderbook['bid_volume'] = sum(q for p, q in orderbook['bids'])
                    orderbook['ask_volume'] = sum(q for p, q in orderbook['asks'])
                    orderbook['volume_imbalance'] = (orderbook['bid_volume'] - orderbook['ask_volume']) / (orderbook['bid_volume'] + orderbook['ask_volume'])
                    orderbook['mid_price'] = (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2
                
                self.orderbook_cache[symbol] = orderbook
                return orderbook
            
            return None
        except Exception as e:
            print(f"⚠️ Error getting orderbook for {symbol}: {str(e)[:50]}")
            return None
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades - shows market momentum and pressure"""
        mexc_symbol = symbol.replace('/', '')
        
        try:
            url = f"{self.base_url}/api/v3/trades?symbol={mexc_symbol}&limit={limit}"
            response = requests.get(url, timeout=5, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                
                trades = []
                for trade in data:
                    trades.append({
                        'price': float(trade['price']),
                        'qty': float(trade['qty']),
                        'time': trade['time'],
                        'is_buyer_maker': trade.get('isBuyerMaker', True)
                    })
                
                # Calculate trade metrics for AI
                if trades:
                    buy_volume = sum(t['qty'] for t in trades if not t['is_buyer_maker'])
                    sell_volume = sum(t['qty'] for t in trades if t['is_buyer_maker'])
                    total_volume = buy_volume + sell_volume
                    
                    trade_data = {
                        'symbol': symbol,
                        'trades': trades,
                        'buy_volume': buy_volume,
                        'sell_volume': sell_volume,
                        'buy_sell_ratio': buy_volume / sell_volume if sell_volume > 0 else 999,
                        'volume_pressure': (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0,
                        'avg_trade_size': total_volume / len(trades),
                        'timestamp': datetime.now()
                    }
                    
                    self.trades_cache[symbol] = trade_data
                    return trade_data
            
            return None
        except Exception as e:
            print(f"⚠️ Error getting trades for {symbol}: {str(e)[:50]}")
            return None
    
    async def get_24h_ticker(self, symbol: str) -> Dict:
        """Get 24h ticker statistics - volume, price change, high/low"""
        mexc_symbol = symbol.replace('/', '')
        
        try:
            url = f"{self.base_url}/api/v3/ticker/24hr?symbol={mexc_symbol}"
            response = requests.get(url, timeout=5, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                
                # Safely parse values with None handling
                count_value = data.get('count', 0)
                if count_value is None:
                    count_value = 0
                
                ticker = {
                    'symbol': symbol,
                    'price_change': float(data.get('priceChange') or 0),
                    'price_change_pct': float(data.get('priceChangePercent') or 0),
                    'high_24h': float(data.get('highPrice') or 0),
                    'low_24h': float(data.get('lowPrice') or 0),
                    'volume_24h': float(data.get('volume') or 0),
                    'quote_volume_24h': float(data.get('quoteVolume') or 0),
                    'trades_count': int(float(count_value)) if count_value else 0,
                    'open_price': float(data.get('openPrice') or 0),
                    'close_price': float(data.get('lastPrice') or 0),
                    'timestamp': datetime.now()
                }
                
                # Calculate additional metrics
                if ticker['high_24h'] > 0 and ticker['low_24h'] > 0:
                    ticker['volatility_24h'] = ((ticker['high_24h'] - ticker['low_24h']) / ticker['low_24h']) * 100
                    ticker['price_position'] = ((ticker['close_price'] - ticker['low_24h']) / (ticker['high_24h'] - ticker['low_24h'])) if ticker['high_24h'] != ticker['low_24h'] else 0.5
                
                self.ticker_24h_cache[symbol] = ticker
                return ticker
            
            return None
        except Exception as e:
            print(f"⚠️ Error getting 24h ticker for {symbol}: {str(e)[:50]}")
            return None
    
    async def get_klines(self, symbol: str, interval: str = '1m', limit: int = 100) -> List[Dict]:
        """Get candlestick/kline data - OHLCV for pattern recognition"""
        mexc_symbol = symbol.replace('/', '')
        
        try:
            url = f"{self.base_url}/api/v3/klines?symbol={mexc_symbol}&interval={interval}&limit={limit}"
            response = requests.get(url, timeout=5, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate data is a list
                if not isinstance(data, list) or len(data) == 0:
                    return []
                
                klines = []
                for k in data:
                    # Validate kline has enough elements
                    if not isinstance(k, list) or len(k) < 9:
                        continue
                    
                    try:
                        klines.append({
                            'open_time': k[0],
                            'open': float(k[1]),
                            'high': float(k[2]),
                            'low': float(k[3]),
                            'close': float(k[4]),
                            'volume': float(k[5]),
                            'close_time': k[6],
                            'quote_volume': float(k[7]),
                            'trades_count': int(float(k[8])) if k[8] else 0
                        })
                    except (ValueError, TypeError, IndexError) as e:
                        # Skip invalid klines
                        continue
                
                self.klines_cache[f"{symbol}_{interval}"] = klines
                return klines if klines else []
            
            return None
        except Exception as e:
            print(f"⚠️ Error getting klines for {symbol}: {str(e)[:50]}")
            return None
    
    async def get_comprehensive_market_data(self, symbol: str) -> Dict:
        """🔥 Get ALL market data at once - feed AI everything!"""
        print(f"📊 Fetching comprehensive data for {symbol}...")
        
        # Fetch all data types in parallel would be ideal, but sequentially works
        price = await self.get_live_price(symbol)
        orderbook = await self.get_orderbook(symbol)
        trades = await self.get_recent_trades(symbol)
        ticker_24h = await self.get_24h_ticker(symbol)
        klines_1m = await self.get_klines(symbol, '1m', 100)
        klines_5m = await self.get_klines(symbol, '5m', 50)
        klines_15m = await self.get_klines(symbol, '15m', 30)
        
        comprehensive_data = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'price': price,
            'orderbook': orderbook,
            'recent_trades': trades,
            'ticker_24h': ticker_24h,
            'klines': {
                '1m': klines_1m,
                '5m': klines_5m,
                '15m': klines_15m
            },
            # Derived metrics for AI
            'market_sentiment': self._calculate_market_sentiment(orderbook, trades, ticker_24h),
            'liquidity_score': self._calculate_liquidity(orderbook, trades),
            'momentum_score': self._calculate_momentum(klines_1m, ticker_24h)
        }
        
        return comprehensive_data
    
    def _calculate_market_sentiment(self, orderbook: Dict, trades: Dict, ticker: Dict) -> float:
        """Calculate overall market sentiment from multiple indicators"""
        sentiment_score = 0.0
        factors = 0
        
        # Factor 1: Volume imbalance from orderbook
        if orderbook and 'volume_imbalance' in orderbook:
            sentiment_score += orderbook['volume_imbalance']
            factors += 1
        
        # Factor 2: Buy/sell pressure from trades
        if trades and 'volume_pressure' in trades:
            sentiment_score += trades['volume_pressure']
            factors += 1
        
        # Factor 3: Price change trend
        if ticker and 'price_change_pct' in ticker:
            sentiment_score += (ticker['price_change_pct'] / 10)  # Normalize
            factors += 1
        
        return sentiment_score / factors if factors > 0 else 0.0
    
    def _calculate_liquidity(self, orderbook: Dict, trades: Dict) -> float:
        """Calculate market liquidity score"""
        if not orderbook:
            return 0.0
        
        # Tight spread + high volume = high liquidity
        spread_score = 1 / (1 + orderbook.get('spread_pct', 1.0))  # Lower spread = higher score
        volume_score = min(orderbook.get('bid_volume', 0) + orderbook.get('ask_volume', 0), 1000) / 1000
        
        return (spread_score + volume_score) / 2
    
    def _calculate_momentum(self, klines: List[Dict], ticker: Dict) -> float:
        """Calculate price momentum score"""
        if not klines or len(klines) < 5:
            return 0.0
        
        # Recent price movement
        recent_closes = [k['close'] for k in klines[-5:]]
        momentum = (recent_closes[-1] - recent_closes[0]) / recent_closes[0] * 100
        
        return max(min(momentum, 5), -5) / 5  # Normalize to [-1, 1]

class LivePaperTradingManager:
    """Paper trading manager using LIVE market prices"""
    
    def __init__(self, initial_capital: float = 5.0):
        self.state_file = "trading_state.json"
        self.initial_capital = initial_capital
        
        # Try to load existing state
        if self._load_state():
            print("💾 Loaded existing trading state")
        else:
            # Initialize fresh state
            self.cash_balance = initial_capital
            self.positions = {}
            self.trade_history = []
            self.total_trades = 0
            self.winning_trades = 0
            print(f"🆕 NEW trading session with ${initial_capital:,.2f}")
        
        self.data_feed = LiveMexcDataFeed()
        
        print(f"🔥 LIVE Paper Trading Manager active")
        print(f"   💰 Current Balance: ${self.cash_balance:.2f}")
        print(f"   📊 Active Positions: {len(self.positions)}")
        print("📡 Using REAL-TIME MEXC market prices!")
        
        # Save state immediately
        self._save_state()
    
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
                self.positions[symbol] = {
                    "quantity": 0, 
                    "avg_price": 0, 
                    "total_cost": 0,
                    "take_profit": None,
                    "stop_loss": None
                }
            
            old_quantity = self.positions[symbol]["quantity"]
            old_cost = self.positions[symbol]["total_cost"]
            
            new_quantity = old_quantity + quantity
            new_cost = old_cost + net_amount
            
            self.positions[symbol] = {
                "quantity": new_quantity,
                "avg_price": new_cost / new_quantity if new_quantity > 0 else 0,
                "total_cost": new_cost,
                "take_profit": take_profit,  # Store TP/SL for position management
                "stop_loss": stop_loss
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
        
        # Save state after each trade
        self._save_state()
        
        return {"success": True, "trade": trade_record}
    
    async def get_portfolio_value(self):
        """Calculate total portfolio value using live prices"""
        
        portfolio_value = self.cash_balance
        position_values = {}
        
        # Get live prices for all positions
        symbols_with_positions = [symbol for symbol, pos in self.positions.items() if pos["quantity"] > 0]
        
        live_prices = {}
        if symbols_with_positions:
            fetched = await self.data_feed.get_multiple_prices(symbols_with_positions)
            live_prices = fetched or {}
        
        for symbol, position in self.positions.items():
            if position["quantity"] > 0:
                if symbol in live_prices:
                    current_price = live_prices[symbol]
                elif position.get("current_price") and position.get("current_price") > 0:
                    current_price = position["current_price"]
                else:
                    current_price = position.get("avg_price", 0)
                
                current_value = position["quantity"] * current_price
                portfolio_value += current_value
                
                position["current_price"] = current_price
                position["current_value"] = current_value
                position["cost_basis"] = position.get("total_cost", 0)
                position["unrealized_pnl"] = current_value - position.get("total_cost", 0)
                position["avg_price"] = position.get("avg_price", current_price)
                if "take_profit" not in position:
                    position["take_profit"] = None
                if "stop_loss" not in position:
                    position["stop_loss"] = None
                
                position_values[symbol] = {
                    "quantity": position["quantity"],
                    "current_price": current_price,
                    "current_value": current_value,
                    "cost_basis": position.get("total_cost", 0),
                    "unrealized_pnl": current_value - position.get("total_cost", 0),
                    "avg_price": position.get("avg_price", current_price),
                    "take_profit": position.get("take_profit"),
                    "stop_loss": position.get("stop_loss")
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
                    "unrealized_pnl": current_value - position["total_cost"],
                    "pnl_percentage": ((current_value - position["total_cost"]) / position["total_cost"]) * 100 if position["total_cost"] > 0 else 0
                }
        
        return {
            "total_value": portfolio_value,
            "cash": self.cash_balance,
            "positions": position_values,
            "total_return": (portfolio_value - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0,
            "total_pnl": portfolio_value - self.initial_capital
        }

    def _save_state(self):
        """Save trading state to file"""
        try:
            state = {
                "cash_balance": self.cash_balance,
                "positions": self.positions,
                "trade_history": self.trade_history[-100:],  # Keep last 100 trades
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "initial_capital": self.initial_capital,
                "last_save_time": datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            print(f"⚠️ Failed to save state: {e}")
            return False
    
    def _load_state(self):
        """Load trading state from file - with automatic stale data detection"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                # Check if state is stale (older than 30 minutes = probably from old deployment)
                last_save_time = state.get('last_save_time')
                is_stale = False
                
                if last_save_time:
                    try:
                        last_save = datetime.fromisoformat(last_save_time)
                        age_minutes = (datetime.now() - last_save).total_seconds() / 60
                        
                        if age_minutes > 30:
                            is_stale = True
                            print(f"⚠️ State is {age_minutes:.1f} minutes old - STALE DATA DETECTED")
                    except:
                        pass
                
                # Check for unrealistic position prices (indicates old/corrupted data)
                positions = state.get("positions", {})
                has_unrealistic_prices = False
                
                for symbol, pos in positions.items():
                    if pos.get('quantity', 0) > 0:
                        avg_price = pos.get('avg_price', 0)
                        # Check if prices are wildly unrealistic
                        if symbol == "BTC/USDT" and (avg_price > 120000 or avg_price < 20000):
                            has_unrealistic_prices = True
                            print(f"⚠️ Unrealistic BTC price detected: ${avg_price:,.2f}")
                        elif symbol == "ETH/USDT" and (avg_price > 10000 or avg_price < 1000):
                            has_unrealistic_prices = True
                            print(f"⚠️ Unrealistic ETH price detected: ${avg_price:,.2f}")
                
                # If data is stale or has unrealistic prices, start fresh
                if is_stale or has_unrealistic_prices:
                    print("🔄 STARTING FRESH - Stale/corrupted data detected")
                    print(f"   Reason: {'Stale timestamp' if is_stale else 'Unrealistic prices'}")
                    
                    # Delete the stale state file
                    try:
                        os.remove(self.state_file)
                        print(f"   🗑️ Deleted stale state file")
                    except:
                        pass
                    
                    return False  # Start fresh
                
                # Data looks good - load it
                self.cash_balance = state.get("cash_balance", self.initial_capital)
                self.positions = state.get("positions", {})
                self.trade_history = state.get("trade_history", [])
                self.total_trades = state.get("total_trades", 0)
                self.winning_trades = state.get("winning_trades", 0)
                
                # Keep initial capital from constructor if not in state
                if "initial_capital" in state:
                    self.initial_capital = state["initial_capital"]
                
                print(f"✅ Loaded valid state from {state.get('last_save_time', 'unknown time')}")
                return True
            return False
        except Exception as e:
            print(f"⚠️ Failed to load state: {e}")
            return False

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
