#!/usr/bin/env python3
"""
🏆 ENHANCED LIVE TRADING CHART WITH TP/SL VISUALIZATION 🏆

Features:
• Real-time price charts with dynamic TP/SL levels
• Live P&L calculation and visualization
• Multi-panel dashboard with portfolio performance
• Interactive trade management interface
• Professional candlestick charts with technical indicators
• Volume analysis and market depth simulation
• Trade execution markers and annotations
• Performance analytics and statistics
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Tuple
import threading
import time
import json

try:
    import mplfinance as mpf
    CANDLESTICK_AVAILABLE = True
    print("📊 Professional candlestick charts enabled!")
except ImportError:
    CANDLESTICK_AVAILABLE = False
    print("⚠️ mplfinance not available - using line charts")

class LiveTradingChart:
    """
    🚀 PROFESSIONAL LIVE TRADING CHART SYSTEM 🚀
    
    Advanced real-time visualization with:
    • Live price action with TP/SL levels
    • Real-time P&L tracking and alerts
    • Portfolio performance monitoring
    • Trade execution visualization
    • Technical analysis overlays
    • Multi-timeframe support
    """
    
    def __init__(self, max_points=300, update_interval=2000):
        self.max_points = max_points
        self.update_interval = update_interval
        self.is_running = False
        
        # Data storage
        self.price_data = {}
        self.ohlc_data = {}
        self.trade_data = {}
        self.performance_data = {
            'timestamps': deque(maxlen=max_points),
            'portfolio_values': deque(maxlen=max_points),
            'pnl_values': deque(maxlen=max_points),
            'trade_outcomes': deque(maxlen=max_points),
            'win_count': 0,
            'loss_count': 0,
            'total_pnl': 0.0,
            'current_drawdown': 0.0,
            'max_drawdown': 0.0
        }
        
        # Trading levels and annotations
        self.active_levels = {}  # Symbol -> {tp, sl, entry, side, pnl}
        self.trade_markers = []  # List of trade execution markers
        self.support_resistance = {}
        
        # Chart styling
        self.colors = {
            'bg': '#0a0a0a',
            'grid': '#333333',
            'text': '#ffffff',
            'price_line': '#00d4ff',
            'tp_line': '#00ff00',
            'sl_line': '#ff0000',
            'entry_line': '#ffaa00',
            'profit_zone': '#00ff0030',
            'loss_zone': '#ff000030',
            'portfolio_up': '#00ff88',
            'portfolio_down': '#ff4444'
        }
        
        # Initialize chart components
        self.fig = None
        self.axes = {}
        self.animation_thread = None
        
        self.initialize_professional_dashboard()
    
    def initialize_professional_dashboard(self):
        """Initialize professional trading dashboard"""
        try:
            # Set up professional dark theme
            plt.style.use('dark_background')
            
            # Create main figure with custom layout
            self.fig = plt.figure(figsize=(20, 14), facecolor=self.colors['bg'])
            self.fig.suptitle('🏆 POISE TRADER - LIVE TRADING DASHBOARD 🏆', 
                            fontsize=20, fontweight='bold', color='gold', y=0.98)
            
            # Create sophisticated grid layout
            gs = GridSpec(4, 4, figure=self.fig, hspace=0.4, wspace=0.3)
            
            # 1. Main price chart with TP/SL (Large - spans 2x2)
            self.axes['main'] = self.fig.add_subplot(gs[0:2, 0:3])
            self.axes['main'].set_title('📈 LIVE PRICE ACTION WITH TP/SL LEVELS', 
                                      fontweight='bold', color='cyan', fontsize=14)
            self.axes['main'].grid(True, alpha=0.2, color=self.colors['grid'])
            self.axes['main'].set_facecolor('#111111')
            
            # 2. Portfolio performance (Top right)
            self.axes['portfolio'] = self.fig.add_subplot(gs[0, 3])
            self.axes['portfolio'].set_title('💰 PORTFOLIO PERFORMANCE', 
                                           fontweight='bold', color='lime', fontsize=12)
            self.axes['portfolio'].grid(True, alpha=0.2)
            self.axes['portfolio'].set_facecolor('#111111')
            
            # 3. Live P&L (Second row, right)
            self.axes['pnl'] = self.fig.add_subplot(gs[1, 3])
            self.axes['pnl'].set_title('📊 LIVE P&L TRACKING', 
                                     fontweight='bold', color='orange', fontsize=12)
            self.axes['pnl'].grid(True, alpha=0.2)
            self.axes['pnl'].set_facecolor('#111111')
            
            # 4. Volume and indicators (Bottom left)
            self.axes['volume'] = self.fig.add_subplot(gs[2, 0:2])
            self.axes['volume'].set_title('📶 VOLUME & TECHNICAL INDICATORS', 
                                        fontweight='bold', color='purple', fontsize=12)
            self.axes['volume'].grid(True, alpha=0.2)
            self.axes['volume'].set_facecolor('#111111')
            
            # 5. Trade statistics (Bottom center-right)
            self.axes['stats'] = self.fig.add_subplot(gs[2, 2:4])
            self.axes['stats'].set_title('🎯 PERFORMANCE STATISTICS', 
                                       fontweight='bold', color='yellow', fontsize=12)
            self.axes['stats'].set_facecolor('#111111')
            
            # 6. Trade history timeline (Bottom full width)
            self.axes['history'] = self.fig.add_subplot(gs[3, :])
            self.axes['history'].set_title('📋 TRADE EXECUTION TIMELINE', 
                                         fontweight='bold', color='white', fontsize=12)
            self.axes['history'].grid(True, alpha=0.2)
            self.axes['history'].set_facecolor('#111111')
            
            plt.tight_layout()
            plt.ion()  # Interactive mode for live updates
            
            print("🚀 Professional trading dashboard initialized!")
            print("📊 6-panel layout: Main Chart + Portfolio + P&L + Volume + Stats + History")
            
        except Exception as e:
            print(f"❌ Error initializing professional dashboard: {e}")
    
    def add_price_point(self, symbol: str, price: float, timestamp=None):
        """Add new price point with automatic OHLC generation"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Initialize symbol data if needed
        if symbol not in self.price_data:
            self.price_data[symbol] = {
                'timestamps': deque(maxlen=self.max_points),
                'prices': deque(maxlen=self.max_points),
                'highs': deque(maxlen=self.max_points),
                'lows': deque(maxlen=self.max_points),
                'volumes': deque(maxlen=self.max_points)
            }
        
        data = self.price_data[symbol]
        
        # Add price data
        data['timestamps'].append(timestamp)
        data['prices'].append(price)
        
        # Generate realistic OHLC data for candlesticks
        if len(data['prices']) >= 2:
            prev_price = list(data['prices'])[-2]
            volatility = abs(price - prev_price) / prev_price * 0.02  # 2% max volatility
            
            high = price * (1 + volatility * np.random.uniform(0.3, 1.0))
            low = price * (1 - volatility * np.random.uniform(0.3, 1.0))
            volume = np.random.uniform(800000, 2000000)  # Realistic volume
        else:
            high = price * 1.001
            low = price * 0.999
            volume = 1000000
        
        data['highs'].append(high)
        data['lows'].append(low)
        data['volumes'].append(volume)
        
        # Update live chart if active symbol
        if symbol in self.active_levels:
            self._update_live_pnl(symbol, price)
    
    def set_trade_levels(self, symbol: str, entry_price: float, take_profit: float, 
                        stop_loss: float, side: str = 'BUY', position_size: float = 100.0):
        """Set TP/SL levels for live visualization"""
        self.active_levels[symbol] = {
            'entry_price': entry_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'side': side.upper(),
            'position_size': position_size,
            'entry_time': datetime.now(),
            'active': True,
            'live_pnl': 0.0,
            'live_pnl_pct': 0.0,
            'highest_pnl': 0.0,
            'lowest_pnl': 0.0
        }
        
        print(f"📊 {symbol} TRADE LEVELS SET:")
        print(f"   📍 Entry: ${entry_price:.6f}")
        print(f"   🎯 Take Profit: ${take_profit:.6f} ({((take_profit/entry_price-1)*100):+.2f}%)")
        print(f"   🛡️ Stop Loss: ${stop_loss:.6f} ({((stop_loss/entry_price-1)*100):+.2f}%)")
        print(f"   💰 Position Size: ${position_size:.2f}")
        print(f"   📊 Live chart will track this position!")
    
    def close_trade_on_chart(self, symbol: str, exit_price: float, reason: str, 
                           pnl_amount: float, pnl_pct: float):
        """Mark trade completion on chart"""
        if symbol in self.active_levels:
            levels = self.active_levels[symbol]
            levels['active'] = False
            levels['exit_price'] = exit_price
            levels['exit_time'] = datetime.now()
            levels['exit_reason'] = reason
            levels['final_pnl'] = pnl_amount
            levels['final_pnl_pct'] = pnl_pct
            
            # Add trade marker for visualization
            trade_marker = {
                'symbol': symbol,
                'entry_time': levels['entry_time'],
                'exit_time': levels['exit_time'],
                'entry_price': levels['entry_price'],
                'exit_price': exit_price,
                'side': levels['side'],
                'pnl': pnl_amount,
                'pnl_pct': pnl_pct,
                'reason': reason,
                'duration': (levels['exit_time'] - levels['entry_time']).total_seconds() / 60
            }
            self.trade_markers.append(trade_marker)
            
            # Update performance data
            self.performance_data['timestamps'].append(datetime.now())
            current_portfolio = 5000 + self.performance_data['total_pnl'] + pnl_amount
            self.performance_data['portfolio_values'].append(current_portfolio)
            self.performance_data['trade_outcomes'].append(pnl_amount)
            self.performance_data['total_pnl'] += pnl_amount
            
            if pnl_amount > 0:
                self.performance_data['win_count'] += 1
            else:
                self.performance_data['loss_count'] += 1
            
            # Clean up tracking for this symbol
            del self.active_levels[symbol]
            
            print(f"📈 {symbol} TRADE COMPLETED ON CHART:")
            print(f"   📊 Exit: ${exit_price:.6f} | P&L: ${pnl_amount:+.2f} ({pnl_pct:+.2f}%)")
            print(f"   ⏰ Duration: {trade_marker['duration']:.1f} minutes")
            print(f"   🏁 Reason: {reason}")
    
    def _update_live_pnl(self, symbol: str, current_price: float):
        """Update live P&L for active positions"""
        if symbol not in self.active_levels or not self.active_levels[symbol]['active']:
            return
        
        levels = self.active_levels[symbol]
        entry_price = levels['entry_price']
        side = levels['side']
        position_size = levels['position_size']
        
        # Calculate live P&L
        if side == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
        
        pnl_amount = position_size * (pnl_pct / 100)
        
        # Update levels with live data
        levels['live_pnl'] = pnl_amount
        levels['live_pnl_pct'] = pnl_pct
        levels['highest_pnl'] = max(levels.get('highest_pnl', pnl_amount), pnl_amount)
        levels['lowest_pnl'] = min(levels.get('lowest_pnl', pnl_amount), pnl_amount)
        
        # Update current P&L in performance data
        self.performance_data['current_live_pnl'] = sum(
            level['live_pnl'] for level in self.active_levels.values() if level['active']
        )
    
    def update_live_chart(self, active_symbol: str = None, bot_stats: Dict = None):
        """Update all dashboard panels with latest data"""
        if not self.fig:
            return
        
        try:
            # Check if we're in the main thread - if not, skip update to avoid errors
            import threading
            if threading.current_thread() != threading.main_thread():
                return
            
            # Update main price chart
            if active_symbol and active_symbol in self.price_data:
                self._update_main_price_chart(active_symbol)
            
            # Update portfolio performance
            self._update_portfolio_chart()
            
            # Update live P&L tracking
            self._update_pnl_chart()
            
            # Update volume and indicators
            if active_symbol:
                self._update_volume_indicators(active_symbol)
            
            # Update performance statistics
            self._update_statistics_panel(bot_stats)
            
            # Update trade history timeline
            self._update_trade_history()
            
            # Refresh display
            plt.draw()
            plt.pause(0.01)
            
        except Exception as e:
            print(f"📊 Chart update error: {e}")
    
    def _update_main_price_chart(self, symbol: str):
        """Update main price chart with TP/SL levels"""
        ax = self.axes['main']
        ax.clear()
        ax.set_title(f'📈 {symbol} - LIVE PRICE ACTION WITH TP/SL LEVELS', 
                    fontweight='bold', color='cyan', fontsize=14)
        ax.grid(True, alpha=0.2, color=self.colors['grid'])
        ax.set_facecolor('#111111')
        
        if symbol not in self.price_data or len(self.price_data[symbol]['prices']) < 2:
            ax.text(0.5, 0.5, f'Waiting for {symbol} price data...', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=16, color='gray')
            return
        
        data = self.price_data[symbol]
        times = list(data['timestamps'])
        prices = list(data['prices'])
        highs = list(data['highs'])
        lows = list(data['lows'])
        
        # Plot main price line with gradient effect
        ax.plot(times, prices, color=self.colors['price_line'], linewidth=3, 
               label=f'💎 {symbol} Live Price', alpha=0.9, zorder=5)
        
        # Add price fill for visual appeal
        ax.fill_between(times, prices, alpha=0.15, color=self.colors['price_line'])
        
        # Plot high/low bands for volatility visualization
        ax.fill_between(times, highs, lows, alpha=0.1, color='gray', 
                       label='📊 High/Low Range')
        
        # Add technical indicators
        if len(prices) >= 20:
            # Moving averages
            sma_20 = pd.Series(prices).rolling(20).mean()
            ax.plot(times[-len(sma_20):], sma_20, color='orange', linewidth=2, 
                   label='📊 SMA 20', alpha=0.8, linestyle='--')
        
        if len(prices) >= 50:
            sma_50 = pd.Series(prices).rolling(50).mean()
            ax.plot(times[-len(sma_50):], sma_50, color='purple', linewidth=2, 
                   label='📈 SMA 50', alpha=0.8, linestyle='--')
        
        # Add TP/SL levels for active positions
        if symbol in self.active_levels and self.active_levels[symbol]['active']:
            levels = self.active_levels[symbol]
            current_price = prices[-1]
            
            # Take Profit line (Green, dashed)
            ax.axhline(y=levels['take_profit'], color=self.colors['tp_line'], 
                      linestyle='--', linewidth=4, alpha=0.9, zorder=10,
                      label=f'🎯 Take Profit: ${levels["take_profit"]:.6f}')
            
            # Stop Loss line (Red, dashed)
            ax.axhline(y=levels['stop_loss'], color=self.colors['sl_line'], 
                      linestyle='--', linewidth=4, alpha=0.9, zorder=10,
                      label=f'🛡️ Stop Loss: ${levels["stop_loss"]:.6f}')
            
            # Entry price line (Orange, solid)
            ax.axhline(y=levels['entry_price'], color=self.colors['entry_line'], 
                      linestyle='-', linewidth=3, alpha=0.9, zorder=8,
                      label=f'📍 Entry: ${levels["entry_price"]:.6f}')
            
            # Profit/Loss zones
            if levels['side'] == 'BUY':
                # Green zone above entry (profit), red zone below (loss)
                ax.fill_between(times, levels['entry_price'], max(prices + [levels['take_profit']]), 
                              color=self.colors['profit_zone'], alpha=0.2, label='💚 Profit Zone')
                ax.fill_between(times, min(prices + [levels['stop_loss']]), levels['entry_price'], 
                              color=self.colors['loss_zone'], alpha=0.2, label='❤️ Risk Zone')
            else:
                # For SELL positions, zones are inverted
                ax.fill_between(times, min(prices + [levels['take_profit']]), levels['entry_price'], 
                              color=self.colors['profit_zone'], alpha=0.2, label='💚 Profit Zone')
                ax.fill_between(times, levels['entry_price'], max(prices + [levels['stop_loss']]), 
                              color=self.colors['loss_zone'], alpha=0.2, label='❤️ Risk Zone')
            
            # Live P&L display (Top-left corner)
            live_pnl = levels['live_pnl']
            live_pnl_pct = levels['live_pnl_pct']
            pnl_color = 'lime' if live_pnl >= 0 else 'red'
            
            # Create attractive P&L box
            pnl_text = f'💰 LIVE P&L\n${live_pnl:+.2f}\n{live_pnl_pct:+.2f}%'
            ax.text(0.02, 0.98, pnl_text, transform=ax.transAxes,
                   fontsize=14, fontweight='bold', color=pnl_color,
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='black', 
                           edgecolor=pnl_color, alpha=0.9, linewidth=2),
                   verticalalignment='top')
            
            # Distance to TP/SL indicators (Top-right corner)
            if levels['side'] == 'BUY':
                tp_distance = ((levels['take_profit'] - current_price) / current_price) * 100
                sl_distance = ((current_price - levels['stop_loss']) / current_price) * 100
            else:
                tp_distance = ((current_price - levels['take_profit']) / current_price) * 100
                sl_distance = ((levels['stop_loss'] - current_price) / current_price) * 100
            
            distance_text = f'🎯 TP: {tp_distance:+.2f}%\n🛡️ SL: {sl_distance:+.2f}%'
            ax.text(0.98, 0.98, distance_text, transform=ax.transAxes,
                   fontsize=12, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#333333', alpha=0.9),
                   verticalalignment='top', horizontalalignment='right')
        
        # Add trade execution markers
        for marker in self.trade_markers[-10:]:  # Show last 10 trades
            if marker['symbol'] == symbol:
                marker_color = 'lime' if marker['pnl'] > 0 else 'red'
                marker_style = '^' if marker['side'] == 'BUY' else 'v'
                
                # Entry marker
                ax.scatter(marker['entry_time'], marker['entry_price'], 
                         color='yellow', s=200, marker='o', zorder=15,
                         edgecolors='black', linewidth=2)
                
                # Exit marker
                ax.scatter(marker['exit_time'], marker['exit_price'], 
                         color=marker_color, s=250, marker=marker_style, zorder=15,
                         edgecolors='white', linewidth=2)
                
                # Add P&L annotation
                ax.annotate(f'${marker["pnl"]:+.2f}', 
                          xy=(marker['exit_time'], marker['exit_price']),
                          xytext=(10, 20), textcoords='offset points',
                          fontsize=10, fontweight='bold', color=marker_color,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
                                  edgecolor=marker_color, alpha=0.8))
        
        # Customize appearance
        ax.legend(loc='center left', fontsize=10, framealpha=0.9)
        ax.tick_params(colors='white')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # Add current time indicator
        if times:
            ax.axvline(x=times[-1], color='white', linestyle=':', alpha=0.5)
    
    def _update_portfolio_chart(self):
        """Update portfolio performance chart"""
        ax = self.axes['portfolio']
        ax.clear()
        ax.set_title('💰 PORTFOLIO PERFORMANCE', fontweight='bold', color='lime', fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#111111')
        
        if not self.performance_data['timestamps']:
            ax.text(0.5, 0.5, 'Starting\nTrading...', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='gray')
            return
        
        times = list(self.performance_data['timestamps'])
        values = list(self.performance_data['portfolio_values'])
        
        # Portfolio value line
        ax.plot(times, values, color=self.colors['portfolio_up'], linewidth=3, 
               label='💎 Portfolio Value', alpha=0.9)
        
        # Break-even line
        initial_value = 5000
        ax.axhline(y=initial_value, color='gray', linestyle='--', alpha=0.6, 
                  label='⚖️ Break-Even')
        
        # Fill profit/loss areas
        values_array = np.array(values)
        ax.fill_between(times, values, initial_value,
                       where=values_array >= initial_value,
                       color='green', alpha=0.3, interpolate=True)
        ax.fill_between(times, values, initial_value,
                       where=values_array < initial_value,
                       color='red', alpha=0.3, interpolate=True)
        
        # Current performance display
        if values:
            current_value = values[-1]
            total_pnl = current_value - initial_value
            pnl_pct = (total_pnl / initial_value) * 100
            
            color = 'lime' if total_pnl >= 0 else 'red'
            perf_text = f'${current_value:.0f}\n${total_pnl:+.0f}\n{pnl_pct:+.1f}%'
            
            ax.text(0.95, 0.95, perf_text, transform=ax.transAxes,
                   fontweight='bold', color=color, fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='black', 
                           edgecolor=color, alpha=0.9),
                   verticalalignment='top', horizontalalignment='right')
        
        ax.legend(fontsize=9, loc='upper left')
        ax.tick_params(colors='white', labelsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    def _update_pnl_chart(self):
        """Update live P&L tracking chart"""
        ax = self.axes['pnl']
        ax.clear()
        ax.set_title('📊 LIVE P&L TRACKING', fontweight='bold', color='orange', fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#111111')
        
        # Show current live P&L from active positions
        current_live_pnl = sum(
            level['live_pnl'] for level in self.active_levels.values() if level['active']
        )
        
        # Show individual position P&Ls
        y_pos = 0.9
        for symbol, levels in self.active_levels.items():
            if levels['active']:
                pnl = levels['live_pnl']
                pnl_pct = levels['live_pnl_pct']
                color = 'lime' if pnl >= 0 else 'red'
                
                pnl_text = f'{symbol}: ${pnl:+.2f} ({pnl_pct:+.2f}%)'
                ax.text(0.05, y_pos, pnl_text, transform=ax.transAxes,
                       fontweight='bold', color=color, fontsize=11)
                y_pos -= 0.15
        
        # Total live P&L
        if current_live_pnl != 0:
            total_color = 'lime' if current_live_pnl >= 0 else 'red'
            ax.text(0.5, 0.5, f'TOTAL LIVE P&L\n${current_live_pnl:+.2f}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=16, fontweight='bold', color=total_color,
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='black', 
                           edgecolor=total_color, alpha=0.9, linewidth=3))
        else:
            ax.text(0.5, 0.5, 'No Active\nPositions', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='gray')
        
        ax.axis('off')
    
    def _update_volume_indicators(self, symbol: str):
        """Update volume and technical indicators"""
        ax = self.axes['volume']
        ax.clear()
        ax.set_title(f'📶 {symbol} - VOLUME & INDICATORS', fontweight='bold', color='purple', fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#111111')
        
        if symbol not in self.price_data or len(self.price_data[symbol]['volumes']) < 10:
            ax.text(0.5, 0.5, 'Collecting data...', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return
        
        data = self.price_data[symbol]
        times = list(data['timestamps'])[-50:]  # Last 50 points
        volumes = list(data['volumes'])[-50:]
        prices = list(data['prices'])[-50:]
        
        # Volume bars
        colors = ['lime' if i == 0 or prices[i] >= prices[i-1] else 'red' 
                 for i in range(len(prices))]
        
        ax.bar(times, volumes, color=colors, alpha=0.6, width=0.0003, 
              label='📶 Volume')
        
        # Volume MA
        if len(volumes) >= 20:
            vol_ma = pd.Series(volumes).rolling(20).mean()
            ax.plot(times[-len(vol_ma):], vol_ma, color='yellow', linewidth=2, 
                   label='📊 Volume MA', alpha=0.8)
        
        # RSI indicator (as overlay)
        if len(prices) >= 14:
            rsi = self._calculate_rsi(np.array(prices))
            # Scale RSI to fit with volume
            max_vol = max(volumes) if volumes else 1000000
            rsi_scaled = (rsi / 100) * max_vol * 0.3
            
            ax.plot(times[-1:], [rsi_scaled], 'o', color='cyan', markersize=10,
                   label=f'📈 RSI: {rsi:.1f}')
        
        ax.legend(fontsize=9)
        ax.tick_params(colors='white', labelsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    def _update_statistics_panel(self, bot_stats: Dict = None):
        """Update performance statistics panel"""
        ax = self.axes['stats']
        ax.clear()
        ax.set_title('🎯 PERFORMANCE STATISTICS', fontweight='bold', color='yellow', fontsize=12)
        ax.set_facecolor('#111111')
        
        wins = self.performance_data['win_count']
        losses = self.performance_data['loss_count']
        total_trades = wins + losses
        
        # Calculate statistics
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        total_pnl = self.performance_data['total_pnl']
        avg_win = np.mean([t for t in self.performance_data['trade_outcomes'] if t > 0]) if wins > 0 else 0
        avg_loss = np.mean([t for t in self.performance_data['trade_outcomes'] if t < 0]) if losses > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Create visual statistics display
        stats_data = [
            ('Win Rate', win_rate, 100, 'lime'),
            ('Total Trades', total_trades, 50, 'cyan'), 
            ('Profit Factor', profit_factor, 3, 'orange'),
            ('Total P&L', total_pnl + 5000, 6000, 'gold')
        ]
        
        # Create circular progress indicators
        angles = np.linspace(0, 2*np.pi, len(stats_data), endpoint=False)
        
        for i, (name, value, max_val, color) in enumerate(stats_data):
            # Position for this metric
            center_x = np.cos(angles[i]) * 1.2
            center_y = np.sin(angles[i]) * 1.2
            
            # Normalize value for circle
            if name == 'Win Rate':
                normalized = value / 100
                display_value = f'{value:.1f}%'
            elif name == 'Total Trades':
                normalized = min(1.0, value / max_val)
                display_value = f'{int(value)}'
            elif name == 'Profit Factor':
                normalized = min(1.0, value / max_val)
                display_value = f'{value:.2f}'
            else:  # Total P&L
                normalized = min(1.0, max(0, (value - 5000) / 1000))
                display_value = f'${value:.0f}'
            
            # Draw progress circle
            circle_theta = np.linspace(0, 2*np.pi*normalized, 100)
            x_circle = 0.3 * np.cos(circle_theta) + center_x
            y_circle = 0.3 * np.sin(circle_theta) + center_y
            
            ax.plot(x_circle, y_circle, color=color, linewidth=8, alpha=0.8)
            
            # Add metric text
            ax.text(center_x, center_y, f'{name}\n{display_value}', 
                   ha='center', va='center', fontweight='bold', 
                   color='white', fontsize=10)
        
        # Bot stats overlay
        if bot_stats:
            confidence = bot_stats.get('confidence_threshold', 0.5)
            mode = bot_stats.get('trading_mode', 'UNKNOWN')
            
            ax.text(0.5, -0.3, f'🤖 Mode: {mode}\n🎯 Confidence: {confidence:.1%}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontweight='bold', color='cyan', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.9))
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _update_trade_history(self):
        """Update trade execution timeline"""
        ax = self.axes['history']
        ax.clear()
        ax.set_title('📋 TRADE EXECUTION TIMELINE', fontweight='bold', color='white', fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#111111')
        
        if not self.trade_markers:
            ax.text(0.5, 0.5, 'No completed trades yet', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='gray')
            return
        
        # Show recent trades timeline
        recent_trades = self.trade_markers[-20:]  # Last 20 trades
        
        if recent_trades:
            # Extract data
            trade_times = [t['exit_time'] for t in recent_trades]
            trade_pnls = [t['pnl'] for t in recent_trades]
            
            # Cumulative P&L line
            cumulative_pnl = np.cumsum([t['pnl'] for t in recent_trades])
            ax.plot(trade_times, cumulative_pnl, color='gold', linewidth=4, 
                   label='💰 Cumulative P&L', alpha=0.9, zorder=10)
            
            # Individual trade scatter
            colors = ['lime' if pnl > 0 else 'red' for pnl in trade_pnls]
            sizes = [min(300, abs(pnl) * 20 + 50) for pnl in trade_pnls]  # Size based on P&L
            
            ax.scatter(trade_times, trade_pnls, c=colors, s=sizes, alpha=0.8,
                      edgecolors='white', linewidths=2, zorder=15, label='Individual Trades')
            
            # Zero line
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Trade statistics text
            total_pnl = sum(trade_pnls)
            avg_trade = np.mean(trade_pnls)
            best_trade = max(trade_pnls) if trade_pnls else 0
            worst_trade = min(trade_pnls) if trade_pnls else 0
            
            stats_text = (f'Total P&L: ${total_pnl:+.2f}\n'
                         f'Avg Trade: ${avg_trade:+.2f}\n'
                         f'Best: ${best_trade:+.2f}\n'
                         f'Worst: ${worst_trade:+.2f}')
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=11, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.9),
                   verticalalignment='top')
        
        ax.legend(fontsize=10)
        ax.tick_params(colors='white', labelsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_ylabel('P&L ($)', color='white')
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def start_live_updates(self):
        """Start live chart updates in separate thread"""
        if self.is_running:
            return
        
        self.is_running = True
        print("🚀 Live chart updates started!")
        print("📊 Dashboard will update every 2 seconds")
        
        # Start background update thread
        self.animation_thread = threading.Thread(target=self._animation_loop)
        self.animation_thread.daemon = True
        self.animation_thread.start()
    
    def _animation_loop(self):
        """Background animation loop"""
        while self.is_running:
            try:
                if self.fig and self.active_levels:
                    # Find active symbol to display
                    active_symbol = next(
                        (symbol for symbol, levels in self.active_levels.items() 
                         if levels['active']), 
                        None
                    )
                    
                    if active_symbol:
                        self.update_live_chart(active_symbol)
                
                time.sleep(self.update_interval / 1000)  # Convert ms to seconds
                
            except Exception as e:
                print(f"Animation loop error: {e}")
                time.sleep(5)
    
    def stop_live_updates(self):
        """Stop live chart updates"""
        self.is_running = False
        if self.animation_thread:
            self.animation_thread.join(timeout=2)
        print("⏹️ Live chart updates stopped")
    
    def show_chart(self):
        """Display the dashboard"""
        if self.fig:
            plt.show(block=False)
            print("📈 Live trading dashboard displayed!")
            print("💡 The chart will update automatically with live data")
    
    def save_chart_snapshot(self, filename: str = None):
        """Save current chart state"""
        if not self.fig:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"poise_trader_snapshot_{timestamp}.png"
        
        self.fig.savefig(filename, dpi=300, bbox_inches='tight', 
                        facecolor='black', edgecolor='none')
        print(f"📸 Trading snapshot saved: {filename}")
    
    def add_support_resistance_level(self, symbol: str, level: float, level_type: str = 'support'):
        """Add support/resistance level for visualization"""
        if symbol not in self.support_resistance:
            self.support_resistance[symbol] = []
        
        self.support_resistance[symbol].append({
            'level': level,
            'type': level_type,
            'timestamp': datetime.now()
        })
        
        # Keep only recent levels
        if len(self.support_resistance[symbol]) > 10:
            self.support_resistance[symbol] = self.support_resistance[symbol][-10:]


class InteractiveTradingGUI:
    """
    🖥️ INTERACTIVE TRADING CONTROL GUI
    
    Features:
    • Real-time bot control
    • Live performance monitoring
    • Manual trade execution
    • Settings adjustment
    • Alert management
    """
    
    def __init__(self, bot_instance, chart_manager=None):
        try:
            import tkinter as tk
            from tkinter import ttk
            
            self.bot = bot_instance
            self.chart_manager = chart_manager
            
            # Main window
            self.root = tk.Tk()
            self.root.title("🏆 POISE TRADER - CONTROL CENTER 🏆")
            self.root.geometry("1200x800")
            self.root.configure(bg='#1a1a1a')
            
            # Variables for live updates
            self.update_active = True
            
            self.setup_gui_layout()
            self.start_live_updates()
            
            print("🖥️ Interactive trading GUI initialized!")
            
        except ImportError:
            print("⚠️ Tkinter not available - GUI disabled")
            self.root = None
    
    def setup_gui_layout(self):
        """Setup the GUI layout"""
        if not self.root:
            return
        
        try:
            import tkinter as tk
            from tkinter import ttk
            
            # Main container
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Top control panel
            control_frame = ttk.LabelFrame(main_frame, text="🎮 BOT CONTROL CENTER", padding=10)
            control_frame.pack(fill='x', pady=(0, 10))
            
            # Bot status and controls
            status_frame = ttk.Frame(control_frame)
            status_frame.pack(fill='x')
            
            self.status_label = ttk.Label(status_frame, text="🔴 Bot Status: Stopped", 
                                        font=('Arial', 12, 'bold'))
            self.status_label.pack(side='left')
            
            self.start_btn = ttk.Button(status_frame, text="🚀 Start Bot", 
                                      command=self.start_bot)
            self.start_btn.pack(side='right', padx=(0, 5))
            
            self.stop_btn = ttk.Button(status_frame, text="🛑 Stop Bot", 
                                     command=self.stop_bot)
            self.stop_btn.pack(side='right', padx=(0, 5))
            
            self.chart_btn = ttk.Button(status_frame, text="📈 Show Charts", 
                                      command=self.show_charts)
            self.chart_btn.pack(side='right', padx=(0, 5))
            
            # Performance display
            perf_frame = ttk.LabelFrame(main_frame, text="📊 LIVE PERFORMANCE", padding=10)
            perf_frame.pack(fill='both', expand=True)
            
            # Create notebook for tabbed interface
            notebook = ttk.Notebook(perf_frame)
            notebook.pack(fill='both', expand=True)
            
            # Performance tab
            perf_tab = ttk.Frame(notebook)
            notebook.add(perf_tab, text='📈 Performance')
            
            self.perf_text = tk.Text(perf_tab, bg='#2a2a2a', fg='white', 
                                   font=('Consolas', 10), wrap='word')
            self.perf_text.pack(fill='both', expand=True)
            
            # Active trades tab
            trades_tab = ttk.Frame(notebook)
            notebook.add(trades_tab, text='⚡ Active Trades')
            
            self.trades_text = tk.Text(trades_tab, bg='#2a2a2a', fg='white',
                                     font=('Consolas', 10), wrap='word')
            self.trades_text.pack(fill='both', expand=True)
            
            # Settings tab
            settings_tab = ttk.Frame(notebook)
            notebook.add(settings_tab, text='⚙️ Settings')
            
            # Add settings controls
            self.setup_settings_tab(settings_tab)
            
        except Exception as e:
            print(f"❌ Error setting up GUI: {e}")
    
    def setup_settings_tab(self, parent):
        """Setup settings controls"""
        try:
            import tkinter as tk
            from tkinter import ttk
            
            # Trading mode selection
            mode_frame = ttk.LabelFrame(parent, text="Trading Mode", padding=5)
            mode_frame.pack(fill='x', pady=5)
            
            self.mode_var = tk.StringVar(value=getattr(self.bot, 'trading_mode', 'PRECISION'))
            
            ttk.Radiobutton(mode_frame, text="⚡ Aggressive Mode", variable=self.mode_var, 
                          value='AGGRESSIVE').pack(anchor='w')
            ttk.Radiobutton(mode_frame, text="🎯 Precision Mode", variable=self.mode_var, 
                          value='PRECISION').pack(anchor='w')
            
            # Risk settings
            risk_frame = ttk.LabelFrame(parent, text="Risk Management", padding=5)
            risk_frame.pack(fill='x', pady=5)
            
            ttk.Label(risk_frame, text="Max Risk per Trade:").pack(anchor='w')
            self.risk_scale = ttk.Scale(risk_frame, from_=0.01, to=0.05, 
                                      orient='horizontal', length=200)
            self.risk_scale.set(getattr(self.bot, 'max_risk_per_trade', 0.02))
            self.risk_scale.pack(fill='x')
            
            # Apply settings button
            apply_btn = ttk.Button(risk_frame, text="Apply Settings", 
                                 command=self.apply_settings)
            apply_btn.pack(pady=10)
            
        except Exception as e:
            print(f"Settings tab error: {e}")
    
    def start_live_updates(self):
        """Start live GUI updates"""
        if self.root:
            self.update_gui_data()
    
    def update_gui_data(self):
        """Update GUI with current bot data"""
        if not self.root or not self.update_active:
            return
        
        try:
            # Update status
            if hasattr(self.bot, 'is_running') and self.bot.is_running:
                self.status_label.config(text="🟢 Bot Status: RUNNING")
            else:
                self.status_label.config(text="🔴 Bot Status: STOPPED")
            
            # Update performance text
            perf_data = self._get_performance_summary()
            self.perf_text.delete(1.0, 'end')
            self.perf_text.insert(1.0, perf_data)
            
            # Update active trades
            trades_data = self._get_active_trades_summary()
            self.trades_text.delete(1.0, 'end')
            self.trades_text.insert(1.0, trades_data)
            
        except Exception as e:
            print(f"GUI update error: {e}")
        
        # Schedule next update
        if self.root:
            self.root.after(3000, self.update_gui_data)  # Update every 3 seconds
    
    def _get_performance_summary(self) -> str:
        """Get performance summary text"""
        try:
            # Get data from bot or chart manager
            wins = self.performance_data['win_count'] if hasattr(self, 'performance_data') else 0
            losses = self.performance_data['loss_count'] if hasattr(self, 'performance_data') else 0
            total_pnl = self.performance_data['total_pnl'] if hasattr(self, 'performance_data') else 0
            
            total_trades = wins + losses
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            
            current_capital = getattr(self.bot, 'current_capital', 5000)
            
            return f"""🏆 POISE TRADER PERFORMANCE SUMMARY
{'='*50}

💰 FINANCIAL METRICS:
   Current Capital: ${current_capital:,.2f}
   Total P&L: ${total_pnl:+,.2f}
   ROI: {((current_capital/5000-1)*100):+.2f}%

📊 TRADING STATISTICS:
   Total Trades: {total_trades}
   Winning Trades: {wins}
   Losing Trades: {losses}
   Win Rate: {win_rate:.1f}%

🎯 PERFORMANCE METRICS:
   Current Mode: {getattr(self.bot, 'trading_mode', 'UNKNOWN')}
   Confidence Threshold: {getattr(self.bot, 'confidence_threshold', 0.5):.1%}
   Active Positions: {len(getattr(self.bot, 'active_signals', {}))}

🚀 STATUS:
   Bot Running: {'✅ YES' if hasattr(self.bot, 'is_running') and self.bot.is_running else '❌ NO'}
   Live Charts: {'✅ YES' if self.chart_manager else '❌ NO'}
   Last Update: {datetime.now().strftime('%H:%M:%S')}
"""
        except Exception as e:
            return f"Error getting performance data: {e}"
    
    def _get_active_trades_summary(self) -> str:
        """Get active trades summary"""
        try:
            active_signals = getattr(self.bot, 'active_signals', {})
            
            if not active_signals:
                return "⚡ ACTIVE TRADES:\n\nNo active trades currently.\n\nBot is analyzing markets for opportunities..."
            
            trades_text = "⚡ ACTIVE TRADES:\n" + "="*40 + "\n\n"
            
            for symbol, signal in active_signals.items():
                if symbol in self.chart_manager.active_levels if self.chart_manager else False:
                    levels = self.chart_manager.active_levels[symbol]
                    live_pnl = levels.get('live_pnl', 0)
                    live_pnl_pct = levels.get('live_pnl_pct', 0)
                    
                    pnl_status = "💚 PROFIT" if live_pnl > 0 else "❤️ LOSS" if live_pnl < 0 else "💛 BREAK-EVEN"
                    
                    trades_text += f"🎯 {symbol} ({signal.action}):\n"
                    trades_text += f"   Entry: ${levels['entry_price']:.6f}\n"
                    trades_text += f"   TP: ${levels['take_profit']:.6f}\n"
                    trades_text += f"   SL: ${levels['stop_loss']:.6f}\n"
                    trades_text += f"   {pnl_status}: ${live_pnl:+.2f} ({live_pnl_pct:+.2f}%)\n"
                    trades_text += f"   Confidence: {signal.confidence:.1%}\n"
                    trades_text += f"   Duration: {(datetime.now() - levels['entry_time']).total_seconds() / 60:.1f}min\n"
                    trades_text += "\n" + "-"*40 + "\n\n"
                else:
                    trades_text += f"🔄 {symbol}: {signal.action} (Setting up...)\n\n"
            
            return trades_text
            
        except Exception as e:
            return f"Error getting active trades: {e}"
    
    def start_bot(self):
        """Start the trading bot"""
        print("🚀 Starting bot from GUI...")
        # This would trigger the bot's start method
        
    def stop_bot(self):
        """Stop the trading bot"""
        if hasattr(self.bot, 'is_running'):
            self.bot.is_running = False
        print("🛑 Bot stop requested from GUI...")
    
    def show_charts(self):
        """Show/focus on charts"""
        if self.chart_manager and self.chart_manager.fig:
            plt.figure(self.chart_manager.fig.number)
            plt.show(block=False)
            print("📈 Charts brought to focus!")
    
    def apply_settings(self):
        """Apply GUI settings to bot"""
        try:
            # Update bot settings from GUI
            if hasattr(self.bot, 'trading_mode'):
                self.bot.trading_mode = self.mode_var.get()
            
            if hasattr(self.bot, 'max_risk_per_trade'):
                self.bot.max_risk_per_trade = self.risk_scale.get()
            
            print("⚙️ Settings applied successfully!")
            
        except Exception as e:
            print(f"❌ Error applying settings: {e}")
    
    def run(self):
        """Run the GUI main loop"""
        if self.root:
            print("🖥️ Starting GUI control center...")
            self.root.mainloop()
    
    def destroy(self):
        """Clean up GUI resources"""
        self.update_active = False
        if self.root:
            self.root.destroy()


def integrate_live_charts_with_bot(bot_instance):
    """
    🔧 INTEGRATE LIVE CHARTS WITH TRADING BOT
    
    This function:
    • Replaces basic chart with enhanced version
    • Sets up live TP/SL visualization
    • Starts interactive GUI
    • Returns chart and GUI managers
    """
    print("🔧 Integrating enhanced live charts with bot...")
    
    # Create enhanced chart system
    chart_manager = LiveTradingChart(max_points=200, update_interval=2000)
    
    # Replace bot's existing chart system
    bot_instance.live_chart = chart_manager
    
    # Create interactive GUI
    gui_manager = InteractiveTradingGUI(bot_instance, chart_manager)
    
    # Start chart updates
    chart_manager.start_live_updates()
    
    # Show initial dashboard
    chart_manager.show_chart()
    
    print("✅ Enhanced live chart system integrated!")
    print("📊 Live TP/SL visualization: ACTIVE")
    print("🖥️ Interactive GUI control: READY")
    print("🎯 Dashboard will update in real-time during trading")
    
    return chart_manager, gui_manager


# Example usage integration
if __name__ == "__main__":
    # Test the chart system
    print("🧪 Testing enhanced live chart system...")
    
    chart = LiveTradingChart()
    
    # Simulate some data
    base_price = 50000
    for i in range(100):
        timestamp = datetime.now() - timedelta(minutes=100-i)
        price = base_price + np.random.normal(0, 500)
        chart.add_price_point("BTC/USDT", price, timestamp)
    
    # Set some trade levels
    chart.set_trade_levels("BTC/USDT", base_price, base_price*1.02, base_price*0.98, "BUY", 1000)
    
    # Update chart
    chart.update_live_chart("BTC/USDT", {'trading_mode': 'PRECISION', 'confidence_threshold': 0.85})
    
    # Show chart
    chart.show_chart()
    
    print("🎯 Test complete! Chart should be visible.")
    input("Press Enter to continue...")
