#!/usr/bin/env python3
"""
🏆 ENHANCED LIVE TRADING CHART WITH ADVANCED TP/SL VISUALIZATION 🏆

Features:
• Real-time candlestick charts with TP/SL levels
• Interactive trading dashboard with multiple timeframes  
• Advanced P&L visualization with drawdown analysis
• Trade history and performance analytics
• Risk management visualization
• Live order book depth (simulated)
• Volume profile analysis
• Support/resistance level detection
• Strategy performance comparison
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from matplotlib.widgets import Button, Slider
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Tuple
import threading
import time

try:
    import mplfinance as mpf
    CANDLESTICK_AVAILABLE = True
    print("📊 Advanced candlestick charts enabled!")
except ImportError:
    CANDLESTICK_AVAILABLE = False
    print("⚠️ mplfinance not available - using line charts")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
    print("🚀 Interactive Plotly charts enabled!")
except ImportError:
    PLOTLY_AVAILABLE = False
    print("📈 Using matplotlib for visualization")

class EnhancedLiveTradingChart:
    """
    🚀 ENHANCED LIVE TRADING CHART SYSTEM 🚀
    
    Advanced real-time visualization with:
    • Professional candlestick charts
    • Dynamic TP/SL level tracking
    • Real-time P&L and risk metrics
    • Interactive controls and analysis
    """
    
    def __init__(self, max_points=500, update_interval=5000):
        self.max_points = max_points
        self.update_interval = update_interval  # milliseconds
        self.is_running = False
        
        # Data storage
        self.ohlc_data = {}  # OHLC data for candlestick charts
        self.price_data = {}
        self.trade_data = {}
        self.performance_data = {
            'timestamps': deque(maxlen=max_points),
            'portfolio_values': deque(maxlen=max_points),
            'drawdown_values': deque(maxlen=max_points),
            'trade_outcomes': deque(maxlen=max_points),
            'win_count': 0,
            'loss_count': 0,
            'total_pnl': 0.0
        }
        
        # Trading levels tracking
        self.active_levels = {}  # Symbol -> {tp, sl, entry, side}
        self.support_resistance = {}  # Symbol -> List of levels
        
        # Chart components
        self.fig = None
        self.axes = {}
        self.animation = None
        
        # Interactive dashboard
        self.dashboard_thread = None
        self.use_plotly = PLOTLY_AVAILABLE
        
        # Initialize charts
        self.initialize_charts()
        
    def initialize_charts(self):
        """Initialize the trading dashboard"""
        if not plt or not plt.get_backend():
            print("⚠️ No display backend available")
            return
            
        try:
            # Create main dashboard figure
            plt.style.use('dark_background')  # Professional dark theme
            
            self.fig = plt.figure(figsize=(20, 14))
            self.fig.suptitle('🏆 ENHANCED POISE TRADER - LIVE DASHBOARD 🏆', 
                            fontsize=18, fontweight='bold', color='gold')
            
            # Create grid layout: 3 rows, 3 columns
            gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Main price chart with TP/SL (spans 2 columns)
            self.axes['price'] = self.fig.add_subplot(gs[0, :2])
            self.axes['price'].set_title('📈 Live Price Action with TP/SL Levels', fontweight='bold')
            self.axes['price'].grid(True, alpha=0.3)
            
            # 2. Portfolio performance (top right)
            self.axes['portfolio'] = self.fig.add_subplot(gs[0, 2])
            self.axes['portfolio'].set_title('💰 Portfolio Growth', fontweight='bold')
            self.axes['portfolio'].grid(True, alpha=0.3)
            
            # 3. Order book depth simulation (middle left)
            self.axes['orderbook'] = self.fig.add_subplot(gs[1, 0])
            self.axes['orderbook'].set_title('📊 Order Book Depth', fontweight='bold')
            
            # 4. Volume profile (middle center)
            self.axes['volume'] = self.fig.add_subplot(gs[1, 1])
            self.axes['volume'].set_title('📶 Volume Profile', fontweight='bold')
            
            # 5. Performance metrics (middle right)
            self.axes['metrics'] = self.fig.add_subplot(gs[1, 2])
            self.axes['metrics'].set_title('🎯 Performance Metrics', fontweight='bold')
            
            # 6. Trade history (bottom spanning all columns)
            self.axes['history'] = self.fig.add_subplot(gs[2, :])
            self.axes['history'].set_title('📋 Recent Trade History & P&L', fontweight='bold')
            self.axes['history'].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.ion()  # Interactive mode
            
            print("🚀 Enhanced live trading dashboard initialized!")
            print("📊 6-panel professional layout created")
            
        except Exception as e:
            print(f"❌ Error initializing enhanced charts: {e}")
    
    def add_ohlc_data(self, symbol: str, open_price: float, high: float, 
                     low: float, close: float, volume: float = 1000, timestamp=None):
        """Add OHLC data for candlestick charts"""
        if timestamp is None:
            timestamp = datetime.now()
            
        if symbol not in self.ohlc_data:
            self.ohlc_data[symbol] = {
                'timestamps': deque(maxlen=self.max_points),
                'open': deque(maxlen=self.max_points),
                'high': deque(maxlen=self.max_points),
                'low': deque(maxlen=self.max_points),
                'close': deque(maxlen=self.max_points),
                'volume': deque(maxlen=self.max_points)
            }
        
        data = self.ohlc_data[symbol]
        data['timestamps'].append(timestamp)
        data['open'].append(open_price)
        data['high'].append(high)
        data['low'].append(low)
        data['close'].append(close)
        data['volume'].append(volume)
    
    def add_price_point(self, symbol: str, price: float, timestamp=None):
        """Add price point and simulate OHLC data"""
        if timestamp is None:
            timestamp = datetime.now()
            
        if symbol not in self.price_data:
            self.price_data[symbol] = {
                'timestamps': deque(maxlen=self.max_points),
                'prices': deque(maxlen=self.max_points)
            }
        
        self.price_data[symbol]['timestamps'].append(timestamp)
        self.price_data[symbol]['prices'].append(price)
        
        # Generate simulated OHLC from price (for demonstration)
        if len(self.price_data[symbol]['prices']) >= 2:
            prev_price = list(self.price_data[symbol]['prices'])[-2]
            volatility = abs(price - prev_price) / prev_price * 0.1
            
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = prev_price
            volume = np.random.uniform(800, 1200)  # Simulated volume
            
            self.add_ohlc_data(symbol, open_price, high, low, price, volume, timestamp)
    
    def set_trade_levels(self, symbol: str, entry_price: float, take_profit: float, 
                        stop_loss: float, side: str = 'BUY'):
        """Set TP/SL levels for visualization"""
        self.active_levels[symbol] = {
            'entry_price': entry_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'side': side,
            'entry_time': datetime.now(),
            'active': True
        }
        
        print(f"📊 {symbol} Levels Set: Entry:{entry_price:.6f} TP:{take_profit:.6f} SL:{stop_loss:.6f}")
    
    def close_trade_on_chart(self, symbol: str, exit_price: float, reason: str, pnl: float):
        """Mark trade completion and update performance"""
        if symbol in self.active_levels:
            self.active_levels[symbol]['active'] = False
            self.active_levels[symbol]['exit_price'] = exit_price
            self.active_levels[symbol]['exit_time'] = datetime.now()
            self.active_levels[symbol]['pnl'] = pnl
            self.active_levels[symbol]['reason'] = reason
        
        # Update performance tracking
        self.performance_data['timestamps'].append(datetime.now())
        current_portfolio = 5000 + self.performance_data['total_pnl'] + pnl
        self.performance_data['portfolio_values'].append(current_portfolio)
        self.performance_data['trade_outcomes'].append(pnl)
        self.performance_data['total_pnl'] += pnl
        
        if pnl > 0:
            self.performance_data['win_count'] += 1
        else:
            self.performance_data['loss_count'] += 1
        
        print(f"📈 {symbol} Trade Completed: Exit:{exit_price:.6f} P&L:{pnl:+.2f} ({reason})")
    
    def update_support_resistance(self, symbol: str, levels: List[float]):
        """Update support and resistance levels"""
        self.support_resistance[symbol] = levels
    
    def update_live_chart(self, active_symbol: str = None, bot_stats: Dict = None):
        """Update all dashboard panels"""
        if not self.fig:
            return
            
        try:
            # Check if we're in the main thread - if not, skip update to avoid errors
            import threading
            if threading.current_thread() != threading.main_thread():
                return
            
            # Update main price chart
            if active_symbol:
                self._update_enhanced_price_chart(active_symbol)
            
            # Update portfolio chart
            self._update_enhanced_portfolio_chart()
            
            # Update order book simulation
            if active_symbol:
                self._update_orderbook_chart(active_symbol)
            
            # Update volume profile
            if active_symbol:
                self._update_volume_chart(active_symbol)
            
            # Update performance metrics
            self._update_metrics_chart(bot_stats)
            
            # Update trade history
            self._update_history_chart()
            
            plt.pause(0.05)  # Smooth updates
            
        except Exception as e:
            print(f"📊 Chart update error: {e}")
    
    def _update_enhanced_price_chart(self, symbol: str):
        """Enhanced price chart with professional styling"""
        self.axes['price'].clear()
        self.axes['price'].set_title(f'📈 {symbol} - Enhanced Live Analysis', fontweight='bold', color='gold')
        self.axes['price'].grid(True, alpha=0.3)
        
        if symbol not in self.price_data or len(self.price_data[symbol]['prices']) < 2:
            return
        
        times = list(self.price_data[symbol]['timestamps'])
        prices = list(self.price_data[symbol]['prices'])
        
        # Plot price with gradient fill
        self.axes['price'].plot(times, prices, color='cyan', linewidth=3, label='💎 Live Price', alpha=0.9)
        self.axes['price'].fill_between(times, prices, alpha=0.2, color='cyan')
        
        # Add moving averages
        if len(prices) >= 20:
            sma_20 = pd.Series(prices).rolling(20).mean()
            self.axes['price'].plot(times[-len(sma_20):], sma_20, 'orange', 
                                  linewidth=2, label='📊 SMA 20', alpha=0.8)
        
        if len(prices) >= 50:
            sma_50 = pd.Series(prices).rolling(50).mean()
            self.axes['price'].plot(times[-len(sma_50):], sma_50, 'purple', 
                                  linewidth=2, label='📈 SMA 50', alpha=0.8)
        
        # Add TP/SL levels for active position
        if symbol in self.active_levels and self.active_levels[symbol]['active']:
            levels = self.active_levels[symbol]
            
            # Take Profit line (green)
            self.axes['price'].axhline(y=levels['take_profit'], color='lime', linestyle='--', 
                                     linewidth=4, label=f'🎯 TP: ${levels["take_profit"]:.6f}', alpha=0.9)
            
            # Stop Loss line (red)  
            self.axes['price'].axhline(y=levels['stop_loss'], color='red', linestyle='--', 
                                     linewidth=4, label=f'🛡️ SL: ${levels["stop_loss"]:.6f}', alpha=0.9)
            
            # Entry price line (yellow)
            self.axes['price'].axhline(y=levels['entry_price'], color='gold', linestyle='-', 
                                     linewidth=3, label=f'📍 Entry: ${levels["entry_price"]:.6f}', alpha=0.9)
            
            # Calculate live P&L
            current_price = prices[-1]
            if levels['side'] == 'BUY':
                pnl_pct = (current_price - levels['entry_price']) / levels['entry_price'] * 100
            else:
                pnl_pct = (levels['entry_price'] - current_price) / levels['entry_price'] * 100
            
            # P&L zone visualization
            if levels['side'] == 'BUY':
                profit_zone_y = [levels['entry_price'], max(prices)]
                loss_zone_y = [min(prices), levels['entry_price']]
            else:
                profit_zone_y = [min(prices), levels['entry_price']]
                loss_zone_y = [levels['entry_price'], max(prices)]
            
            # Fill profit/loss zones
            self.axes['price'].fill_between(times, profit_zone_y[0], profit_zone_y[1], 
                                          color='green', alpha=0.1, label='💚 Profit Zone')
            self.axes['price'].fill_between(times, loss_zone_y[0], loss_zone_y[1], 
                                          color='red', alpha=0.1, label='❤️ Risk Zone')
            
            # Live P&L annotation
            pnl_color = 'lime' if pnl_pct >= 0 else 'red'
            self.axes['price'].text(0.02, 0.98, f'LIVE P&L: {pnl_pct:+.3f}%', 
                                  transform=self.axes['price'].transAxes,
                                  fontsize=14, fontweight='bold', color=pnl_color,
                                  bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
        
        # Add support/resistance levels
        if symbol in self.support_resistance:
            for i, level in enumerate(self.support_resistance[symbol][-5:]):  # Show last 5 levels
                self.axes['price'].axhline(y=level, color='white', linestyle=':', 
                                         alpha=0.6, linewidth=1)
                self.axes['price'].text(times[-1], level, f'S/R: {level:.4f}', 
                                      fontsize=8, color='white', alpha=0.8)
        
        self.axes['price'].legend(loc='upper left', fontsize=10)
        self.axes['price'].tick_params(axis='x', rotation=45)
        
        # Format x-axis for time
        self.axes['price'].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    def _update_enhanced_portfolio_chart(self):
        """Enhanced portfolio performance chart"""
        self.axes['portfolio'].clear()
        self.axes['portfolio'].set_title('💰 Portfolio Performance', fontweight='bold', color='gold')
        self.axes['portfolio'].grid(True, alpha=0.3)
        
        if not self.performance_data['timestamps']:
            self.axes['portfolio'].text(0.5, 0.5, 'No trading data yet', 
                                      ha='center', va='center', 
                                      transform=self.axes['portfolio'].transAxes, 
                                      fontsize=12, color='gray')
            return
        
        times = list(self.performance_data['timestamps'])
        values = list(self.performance_data['portfolio_values'])
        
        # Portfolio value line
        self.axes['portfolio'].plot(times, values, color='lime', linewidth=3, 
                                  label='💎 Portfolio Value', alpha=0.9)
        
        # Break-even line
        initial_value = 5000
        self.axes['portfolio'].axhline(y=initial_value, color='gray', linestyle='--', 
                                     alpha=0.7, label='⚖️ Break-Even')
        
        # Fill profit/loss areas
        self.axes['portfolio'].fill_between(times, values, initial_value, 
                                          where=np.array(values) >= initial_value,
                                          color='green', alpha=0.3, interpolate=True)
        self.axes['portfolio'].fill_between(times, values, initial_value, 
                                          where=np.array(values) < initial_value,
                                          color='red', alpha=0.3, interpolate=True)
        
        # Current performance metrics
        if values:
            current_value = values[-1]
            total_pnl = current_value - initial_value
            pnl_pct = (total_pnl / initial_value) * 100
            
            color = 'lime' if total_pnl >= 0 else 'red'
            self.axes['portfolio'].text(0.02, 0.95, 
                                      f'Value: ${current_value:.2f}\nP&L: ${total_pnl:+.2f}\nReturn: {pnl_pct:+.2f}%', 
                                      transform=self.axes['portfolio'].transAxes,
                                      fontweight='bold', color=color, fontsize=10,
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
        
        self.axes['portfolio'].legend(fontsize=9)
        self.axes['portfolio'].tick_params(axis='x', rotation=45)
        self.axes['portfolio'].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    def _update_orderbook_chart(self, symbol: str):
        """Simulate order book depth visualization"""
        self.axes['orderbook'].clear()
        self.axes['orderbook'].set_title(f'📊 {symbol} Order Book', fontweight='bold')
        
        if symbol not in self.price_data or not self.price_data[symbol]['prices']:
            return
        
        current_price = list(self.price_data[symbol]['prices'])[-1]
        
        # Simulate order book data
        spread = current_price * 0.001  # 0.1% spread
        levels = 10
        
        # Generate bid levels (below current price)
        bid_prices = [current_price - spread - (i * spread * 0.1) for i in range(levels)]
        bid_volumes = [np.random.uniform(100, 1000) for _ in range(levels)]
        
        # Generate ask levels (above current price)
        ask_prices = [current_price + spread + (i * spread * 0.1) for i in range(levels)]
        ask_volumes = [np.random.uniform(100, 1000) for _ in range(levels)]
        
        # Plot order book
        self.axes['orderbook'].barh(bid_prices, bid_volumes, color='green', alpha=0.7, label='Bids')
        self.axes['orderbook'].barh(ask_prices, ask_volumes, color='red', alpha=0.7, label='Asks')
        
        # Current price line
        self.axes['orderbook'].axhline(y=current_price, color='yellow', linewidth=3, 
                                     label=f'Price: ${current_price:.6f}')
        
        self.axes['orderbook'].set_xlabel('Volume')
        self.axes['orderbook'].legend(fontsize=9)
    
    def _update_volume_chart(self, symbol: str):
        """Volume profile analysis"""
        self.axes['volume'].clear()
        self.axes['volume'].set_title(f'📶 {symbol} Volume Profile', fontweight='bold')
        
        if symbol not in self.ohlc_data or not self.ohlc_data[symbol]['volume']:
            return
        
        volumes = list(self.ohlc_data[symbol]['volume'])
        prices = list(self.ohlc_data[symbol]['close'])
        
        if len(volumes) < 10:
            return
        
        # Create volume profile (price levels vs volume)
        price_bins = np.linspace(min(prices), max(prices), 20)
        volume_profile = np.zeros(len(price_bins)-1)
        
        for price, volume in zip(prices, volumes):
            for i in range(len(price_bins)-1):
                if price_bins[i] <= price <= price_bins[i+1]:
                    volume_profile[i] += volume
                    break
        
        # Plot volume profile horizontally
        bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
        self.axes['volume'].barh(bin_centers, volume_profile, 
                               color='skyblue', alpha=0.7, edgecolor='white')
        
        # Highlight high volume nodes (important levels)
        max_volume_idx = np.argmax(volume_profile)
        poc_price = bin_centers[max_volume_idx]  # Point of Control
        self.axes['volume'].axhline(y=poc_price, color='orange', linewidth=2, 
                                  linestyle='--', label=f'POC: ${poc_price:.4f}')
        
        self.axes['volume'].set_xlabel('Volume')
        self.axes['volume'].set_ylabel('Price Level')
        self.axes['volume'].legend(fontsize=9)
    
    def _update_metrics_chart(self, bot_stats: Dict = None):
        """Performance metrics visualization"""
        self.axes['metrics'].clear()
        self.axes['metrics'].set_title('🎯 Live Performance Metrics', fontweight='bold', color='gold')
        
        wins = self.performance_data['win_count']
        losses = self.performance_data['loss_count']
        total_trades = wins + losses
        
        if total_trades == 0:
            self.axes['metrics'].text(0.5, 0.5, 'No trades completed\nyet', 
                                    ha='center', va='center', 
                                    transform=self.axes['metrics'].transAxes, 
                                    fontsize=12, color='gray')
            return
        
        win_rate = wins / total_trades
        
        # Metrics to display
        metrics = {
            'Win Rate': win_rate,
            'Total Trades': min(1.0, total_trades / 50),  # Normalize for display
            'Profit Factor': min(1.0, (self.performance_data['total_pnl'] + 5000) / 5000),
            'AI Confidence': bot_stats.get('confidence_threshold', 0.5) if bot_stats else 0.5
        }
        
        # Create circular progress indicators
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        colors = ['lime', 'cyan', 'gold', 'orange']
        
        for i, (metric_name, value) in enumerate(metrics.items()):
            # Create circular progress
            theta = np.linspace(0, 2 * np.pi * value, 100)
            r_outer = 0.8
            r_inner = 0.6
            
            x_outer = r_outer * np.cos(theta)
            y_outer = r_outer * np.sin(theta)
            x_inner = r_inner * np.cos(theta)
            y_inner = r_inner * np.sin(theta)
            
            # Position circles
            center_x = np.cos(angles[i]) * 1.5
            center_y = np.sin(angles[i]) * 1.5
            
            self.axes['metrics'].fill_between(x_outer + center_x, y_outer + center_y, 
                                            x_inner + center_x, y_inner + center_y,
                                            color=colors[i], alpha=0.7)
            
            # Add metric labels
            self.axes['metrics'].text(center_x, center_y, f'{metric_name}\n{value:.1%}' if metric_name != 'Total Trades' else f'{metric_name}\n{wins + losses}', 
                                    ha='center', va='center', fontweight='bold', 
                                    fontsize=9, color='white')
        
        self.axes['metrics'].set_xlim(-3, 3)
        self.axes['metrics'].set_ylim(-3, 3)
        self.axes['metrics'].set_aspect('equal')
        self.axes['metrics'].axis('off')
    
    def _update_history_chart(self):
        """Trade history and P&L timeline"""
        self.axes['history'].clear()
        self.axes['history'].set_title('📋 Trade History & P&L Timeline', fontweight='bold', color='gold')
        self.axes['history'].grid(True, alpha=0.3)
        
        if not self.performance_data['trade_outcomes']:
            self.axes['history'].text(0.5, 0.5, 'No completed trades yet', 
                                    ha='center', va='center', 
                                    transform=self.axes['history'].transAxes, 
                                    fontsize=12, color='gray')
            return
        
        times = list(self.performance_data['timestamps'])
        outcomes = list(self.performance_data['trade_outcomes'])
        
        # Cumulative P&L line
        cumulative_pnl = np.cumsum(outcomes)
        self.axes['history'].plot(times, cumulative_pnl, color='gold', linewidth=3, 
                                label='💰 Cumulative P&L', alpha=0.9)
        
        # Individual trade markers
        colors = ['lime' if pnl > 0 else 'red' for pnl in outcomes]
        sizes = [abs(pnl) * 10 + 20 for pnl in outcomes]  # Size based on P&L magnitude
        
        self.axes['history'].scatter(times, outcomes, c=colors, s=sizes, alpha=0.8, 
                                   edgecolors='white', linewidths=1, label='Individual Trades')
        
        # Zero line
        self.axes['history'].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Performance statistics
        total_pnl = sum(outcomes)
        avg_trade = np.mean(outcomes) if outcomes else 0
        best_trade = max(outcomes) if outcomes else 0
        worst_trade = min(outcomes) if outcomes else 0
        
        stats_text = f'Total P&L: ${total_pnl:+.2f}\\nAvg Trade: ${avg_trade:+.2f}\\nBest: ${best_trade:+.2f}\\nWorst: ${worst_trade:+.2f}'
        self.axes['history'].text(0.02, 0.02, stats_text, 
                                transform=self.axes['history'].transAxes,
                                fontsize=10, fontweight='bold', color='white',
                                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
        
        self.axes['history'].legend(fontsize=10)
        self.axes['history'].tick_params(axis='x', rotation=45)
        self.axes['history'].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.axes['history'].set_ylabel('P&L ($)')
    
    def start_live_updates(self):
        """Start live chart updates"""
        if not self.fig:
            return
            
        self.is_running = True
        print("🚀 Enhanced live chart updates started!")
        
        # Start animation for live updates
        self.animation = animation.FuncAnimation(
            self.fig, self._animation_update, interval=self.update_interval,
            blit=False, cache_frame_data=False
        )
        
    def _animation_update(self, frame):
        """Animation update function"""
        if not self.is_running:
            return
        
        # Update charts if we have active data
        active_symbols = list(self.price_data.keys())
        if active_symbols:
            self.update_live_chart(active_symbols[0])
    
    def stop_live_updates(self):
        """Stop live updates"""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
        print("⏹️ Enhanced live chart updates stopped")
    
    def show_chart(self):
        """Display the enhanced dashboard"""
        if self.fig:
            plt.show()
    
    def save_chart_snapshot(self, filename: str = None):
        """Save current chart as image"""
        if not self.fig:
            return
            
        if filename is None:
            filename = f"trading_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        self.fig.savefig(filename, dpi=300, bbox_inches='tight', 
                        facecolor='black', edgecolor='none')
        print(f"📸 Chart snapshot saved: {filename}")


class PlotlyInteractiveDashboard:
    """🚀 Advanced Interactive Dashboard using Plotly"""
    
    def __init__(self):
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available - using matplotlib fallback")
            return
        
        self.app = None
        self.data_store = {
            'prices': {},
            'trades': [],
            'levels': {},
            'performance': {'timestamps': [], 'values': []}
        }
        
        print("🚀 Interactive Plotly dashboard initialized!")
    
    def create_interactive_dashboard(self):
        """Create interactive web-based dashboard"""
        if not PLOTLY_AVAILABLE:
            return None
        
        try:
            import dash
            from dash import dcc, html, Input, Output
            
            # Create Dash app
            self.app = dash.Dash(__name__)
            
            # Layout
            self.app.layout = html.Div([
                html.H1("🏆 POISE TRADER - LIVE DASHBOARD", 
                       style={'textAlign': 'center', 'color': 'gold'}),
                
                # Main chart row
                html.Div([
                    dcc.Graph(id='main-price-chart', style={'width': '70%', 'display': 'inline-block'}),
                    dcc.Graph(id='portfolio-chart', style={'width': '30%', 'display': 'inline-block'})
                ]),
                
                # Secondary charts row
                html.Div([
                    dcc.Graph(id='volume-chart', style={'width': '33%', 'display': 'inline-block'}),
                    dcc.Graph(id='metrics-chart', style={'width': '33%', 'display': 'inline-block'}),
                    dcc.Graph(id='trades-chart', style={'width': '34%', 'display': 'inline-block'})
                ]),
                
                # Auto-refresh interval
                dcc.Interval(
                    id='interval-component',
                    interval=5000,  # 5 seconds
                    n_intervals=0
                )
            ])
            
            # Callbacks for live updates
            @self.app.callback(
                [Output('main-price-chart', 'figure'),
                 Output('portfolio-chart', 'figure'),
                 Output('volume-chart', 'figure'),
                 Output('metrics-chart', 'figure'),
                 Output('trades-chart', 'figure')],
                [Input('interval-component', 'n_intervals')]
            )
            def update_charts(n):
                return (
                    self._create_main_chart(),
                    self._create_portfolio_chart(),
                    self._create_volume_chart(),
                    self._create_metrics_chart(),
                    self._create_trades_chart()
                )
            
            print("📱 Interactive dashboard created!")
            return self.app
            
        except Exception as e:
            print(f"❌ Error creating interactive dashboard: {e}")
            return None
    
    def _create_main_chart(self):
        """Create main price chart with TP/SL"""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           subplot_titles=['Price Action with TP/SL', 'Volume'],
                           row_heights=[0.7, 0.3])
        
        # Add price data if available
        for symbol, data in self.data_store['prices'].items():
            if data['timestamps'] and data['prices']:
                fig.add_trace(
                    go.Scatter(x=data['timestamps'], y=data['prices'],
                             mode='lines', name=f'{symbol} Price',
                             line=dict(color='cyan', width=3)),
                    row=1, col=1
                )
                
                # Add TP/SL levels
                if symbol in self.data_store['levels'] and self.data_store['levels'][symbol].get('active'):
                    levels = self.data_store['levels'][symbol]
                    
                    # Take Profit
                    fig.add_hline(y=levels['take_profit'], line_color='lime', 
                                line_dash='dash', line_width=3, 
                                annotation_text=f"TP: {levels['take_profit']:.6f}",
                                row=1, col=1)
                    
                    # Stop Loss
                    fig.add_hline(y=levels['stop_loss'], line_color='red', 
                                line_dash='dash', line_width=3,
                                annotation_text=f"SL: {levels['stop_loss']:.6f}",
                                row=1, col=1)
                    
                    # Entry Price
                    fig.add_hline(y=levels['entry_price'], line_color='gold', 
                                line_width=2,
                                annotation_text=f"Entry: {levels['entry_price']:.6f}",
                                row=1, col=1)
        
        fig.update_layout(
            title="🚀 ENHANCED LIVE PRICE CHART",
            template='plotly_dark',
            height=600
        )
        
        return fig
    
    def _create_portfolio_chart(self):
        """Create portfolio performance chart"""
        fig = go.Figure()
        
        if self.data_store['performance']['timestamps']:
            fig.add_trace(
                go.Scatter(x=self.data_store['performance']['timestamps'],
                         y=self.data_store['performance']['values'],
                         mode='lines+markers',
                         name='Portfolio Value',
                         line=dict(color='lime', width=3),
                         marker=dict(size=8))
            )
            
            # Break-even line
            fig.add_hline(y=5000, line_color='gray', line_dash='dash', 
                         annotation_text="Break-Even")
        
        fig.update_layout(
            title="💰 Portfolio Performance",
            template='plotly_dark',
            height=300
        )
        
        return fig
    
    def _create_volume_chart(self):
        """Create volume analysis chart"""
        fig = go.Figure()
        
        # Add volume bars if available
        for symbol, data in self.data_store['prices'].items():
            if symbol in self.app_data.get('volume', {}):
                volume_data = self.app_data['volume'][symbol]
                fig.add_trace(
                    go.Bar(x=volume_data['timestamps'], y=volume_data['volumes'],
                          name=f'{symbol} Volume', opacity=0.7)
                )
        
        fig.update_layout(
            title="📶 Volume Analysis",
            template='plotly_dark',
            height=300
        )
        
        return fig
    
    def _create_metrics_chart(self):
        """Create metrics gauge chart"""
        wins = self.performance_data.get('win_count', 0)
        total = wins + self.performance_data.get('loss_count', 0)
        win_rate = wins / total if total > 0 else 0
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=win_rate * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Win Rate %"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 85], 'color': "yellow"},
                    {'range': [85, 100], 'color': "lime"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            title="🎯 Performance Gauge",
            template='plotly_dark',
            height=300
        )
        
        return fig
    
    def _create_trades_chart(self):
        """Create recent trades chart"""
        fig = go.Figure()
        
        if self.data_store['trades']:
            recent_trades = self.data_store['trades'][-20:]  # Last 20 trades
            
            colors = ['lime' if trade['pnl'] > 0 else 'red' for trade in recent_trades]
            
            fig.add_trace(
                go.Bar(x=[f"Trade {i+1}" for i in range(len(recent_trades))],
                      y=[trade['pnl'] for trade in recent_trades],
                      marker_color=colors,
                      name='Trade P&L')
            )
        
        fig.update_layout(
            title="📊 Recent Trade Results",
            template='plotly_dark',
            height=300,
            yaxis_title="P&L ($)"
        )
        
        return fig
    
    def run_dashboard(self, debug=False, port=8050):
        """Run the interactive dashboard"""
        if self.app:
            print(f"🌐 Starting interactive dashboard on http://localhost:{port}")
            self.app.run_server(debug=debug, port=port, host='0.0.0.0')


# Integration helper functions
def create_enhanced_chart_system(bot_instance):
    """Create and integrate enhanced chart system with bot"""
    enhanced_chart = EnhancedLiveTradingChart(max_points=300, update_interval=3000)
    
    # Replace bot's live_chart with enhanced version
    bot_instance.live_chart = enhanced_chart
    
    # Start live updates
    enhanced_chart.start_live_updates()
    
    print("✅ Enhanced chart system integrated with bot!")
    return enhanced_chart

def create_interactive_dashboard(bot_instance):
    """Create interactive web dashboard"""
    dashboard = PlotlyInteractiveDashboard()
    dashboard_app = dashboard.create_interactive_dashboard()
    
    if dashboard_app:
        print("🚀 Interactive dashboard ready!")
        print("   📱 Access at: http://localhost:8050")
        return dashboard, dashboard_app
    
    return None, None
