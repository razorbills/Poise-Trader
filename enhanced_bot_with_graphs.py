#!/usr/bin/env python3
"""
🤖 ENHANCED AUTONOMOUS AI TRADING BOT WITH LIVE GRAPHS

NEW FEATURES:
📈 Live price charts with TP/SL levels
📊 Real-time performance graphs
🎯 Visual trade execution
📉 P&L visualization
🔥 Interactive dashboard

IMPROVEMENTS:
✅ Live graph showing current price vs TP/SL
✅ Real-time trade visualization
✅ Performance charts
✅ Interactive display
✅ Visual trade management
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from decimal import Decimal
import random
import math
import threading
from collections import deque

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️ Matplotlib not available. Install with: pip install matplotlib")

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("⚠️ Tkinter not available for GUI")


class LivePriceChart:
    """
    📈 LIVE PRICE CHART WITH TP/SL VISUALIZATION
    
    Features:
    • Real-time price plotting
    • TP/SL level lines
    • Trade entry/exit markers
    • P&L visualization
    • Interactive controls
    """
    
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.price_data = {}
        self.trade_markers = {}
        
        if PLOTTING_AVAILABLE:
            self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
            self.fig.suptitle('🤖 AUTONOMOUS TRADING BOT - LIVE DASHBOARD', fontsize=16, fontweight='bold')
            
            # Chart 1: Price with TP/SL
            self.price_ax = self.axes[0, 0]
            self.price_ax.set_title('📈 Live Price with TP/SL Levels')
            self.price_ax.grid(True, alpha=0.3)
            
            # Chart 2: P&L Over Time  
            self.pnl_ax = self.axes[0, 1]
            self.pnl_ax.set_title('💰 Portfolio P&L Over Time')
            self.pnl_ax.grid(True, alpha=0.3)
            
            # Chart 3: Win/Loss Distribution
            self.stats_ax = self.axes[1, 0]
            self.stats_ax.set_title('📊 Win/Loss Statistics')
            
            # Chart 4: Trade Duration Analysis
            self.duration_ax = self.axes[1, 1]
            self.duration_ax.set_title('⏰ Trade Duration Analysis')
            
            plt.tight_layout()
            plt.ion()  # Interactive mode
        
        # Data storage
        self.timestamps = deque(maxlen=max_points)
        self.portfolio_values = deque(maxlen=max_points)
        self.trade_durations = []
        self.trade_pnls = []
    
    def add_price_point(self, symbol, price, timestamp=None):
        """Add new price point to the chart"""
        if not PLOTTING_AVAILABLE:
            return
        
        if timestamp is None:
            timestamp = datetime.now()
        
        if symbol not in self.price_data:
            self.price_data[symbol] = {
                'timestamps': deque(maxlen=self.max_points),
                'prices': deque(maxlen=self.max_points),
                'tp_level': None,
                'sl_level': None,
                'entry_price': None,
                'line': None
            }
        
        data = self.price_data[symbol]
        data['timestamps'].append(timestamp)
        data['prices'].append(price)
    
    def set_trade_levels(self, symbol, entry_price, take_profit, stop_loss):
        """Set TP/SL levels for visualization"""
        if not PLOTTING_AVAILABLE or symbol not in self.price_data:
            return
        
        data = self.price_data[symbol]
        data['entry_price'] = entry_price
        data['tp_level'] = take_profit
        data['sl_level'] = stop_loss
        
        print(f"📊 Chart updated: {symbol} Entry:{entry_price:.6f} TP:{take_profit:.6f} SL:{stop_loss:.6f}")
    
    def update_chart(self, symbol, performance_stats):
        """Update the live chart"""
        if not PLOTTING_AVAILABLE or symbol not in self.price_data:
            return
        
        try:
            data = self.price_data[symbol]
            
            if not data['prices']:
                return
            
            # Clear and update price chart
            self.price_ax.clear()
            self.price_ax.set_title(f'📈 {symbol} - Live Price with TP/SL')
            self.price_ax.grid(True, alpha=0.3)
            
            # Plot price line
            times = list(data['timestamps'])
            prices = list(data['prices'])
            
            if len(times) > 1:
                self.price_ax.plot(times, prices, 'b-', linewidth=2, label='Price')
                
                # Add TP/SL lines if available
                if data['tp_level'] and data['sl_level']:
                    self.price_ax.axhline(y=data['tp_level'], color='green', linestyle='--', 
                                        linewidth=2, label=f'Take Profit: {data["tp_level"]:.6f}')
                    self.price_ax.axhline(y=data['sl_level'], color='red', linestyle='--', 
                                        linewidth=2, label=f'Stop Loss: {data["sl_level"]:.6f}')
                
                # Add entry price line
                if data['entry_price']:
                    self.price_ax.axhline(y=data['entry_price'], color='orange', linestyle='-', 
                                        linewidth=2, label=f'Entry: {data["entry_price"]:.6f}')
                
                # Current price annotation
                current_price = prices[-1]
                self.price_ax.annotate(f'Current: {current_price:.6f}', 
                                     xy=(times[-1], current_price),
                                     xytext=(10, 10), textcoords='offset points',
                                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                     fontweight='bold')
                
                self.price_ax.legend()
                self.price_ax.tick_params(axis='x', rotation=45)
            
            # Update portfolio P&L chart
            self.pnl_ax.clear()
            self.pnl_ax.set_title('💰 Portfolio P&L Over Time')
            self.pnl_ax.grid(True, alpha=0.3)
            
            if len(self.timestamps) > 1:
                self.pnl_ax.plot(list(self.timestamps), list(self.portfolio_values), 'g-', linewidth=2)
                self.pnl_ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                
                # Highlight current P&L
                if self.portfolio_values:
                    current_pnl = self.portfolio_values[-1]
                    color = 'green' if current_pnl >= 0 else 'red'
                    self.pnl_ax.text(0.02, 0.95, f'Current P&L: ${current_pnl:+.2f}', 
                                   transform=self.pnl_ax.transAxes, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
            
            # Update win/loss pie chart
            self.stats_ax.clear()
            self.stats_ax.set_title('📊 Win/Loss Statistics')
            
            if performance_stats['total_trades'] > 0:
                wins = performance_stats['winning_trades']
                losses = performance_stats['losing_trades']
                
                if wins > 0 or losses > 0:
                    labels = ['Wins', 'Losses']
                    sizes = [wins, losses]
                    colors = ['#90EE90', '#FF6B6B']
                    
                    wedges, texts, autotexts = self.stats_ax.pie(sizes, labels=labels, colors=colors, 
                                                               autopct='%1.1f%%', startangle=90)
                    
                    # Add win rate in center
                    win_rate = performance_stats['win_rate']
                    self.stats_ax.text(0, 0, f'{win_rate:.1f}%\nWin Rate', 
                                     ha='center', va='center', fontweight='bold', fontsize=12)
            
            # Update trade duration histogram
            self.duration_ax.clear()
            self.duration_ax.set_title('⏰ Trade Duration Distribution')
            
            if self.trade_durations:
                self.duration_ax.hist(self.trade_durations, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
                self.duration_ax.set_xlabel('Duration (minutes)')
                self.duration_ax.set_ylabel('Number of Trades')
                
                avg_duration = sum(self.trade_durations) / len(self.trade_durations)
                self.duration_ax.axvline(x=avg_duration, color='red', linestyle='--', 
                                       label=f'Avg: {avg_duration:.1f}min')
                self.duration_ax.legend()
            
            plt.tight_layout()
            plt.pause(0.1)
            
        except Exception as e:
            print(f"❌ Chart update error: {e}")
    
    def add_trade_completion(self, duration_minutes, pnl_amount):
        """Add completed trade data for analysis"""
        self.trade_durations.append(duration_minutes)
        self.trade_pnls.append(pnl_amount)
    
    def update_portfolio_value(self, timestamp, portfolio_value):
        """Update portfolio value for P&L chart"""
        self.timestamps.append(timestamp)
        self.portfolio_values.append(portfolio_value - 5000)  # Show P&L from initial


class TradingGUI:
    """
    🖥️ INTERACTIVE TRADING GUI
    
    Features:
    • Live trade monitoring
    • Manual controls
    • Performance metrics
    • Trade history
    • Real-time updates
    """
    
    def __init__(self, bot_instance):
        if not GUI_AVAILABLE:
            return
        
        self.bot = bot_instance
        self.root = tk.Tk()
        self.root.title("🤖 Autonomous Trading Bot - Live Dashboard")
        self.root.geometry("800x600")
        self.root.configure(bg='#1e1e1e')
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI interface"""
        if not GUI_AVAILABLE:
            return
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bot status section
        status_frame = ttk.LabelFrame(main_frame, text="🤖 Bot Status", padding="5")
        status_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="🔴 Stopped")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.capital_label = ttk.Label(status_frame, text="💰 Capital: $5,000")
        self.capital_label.grid(row=0, column=1, sticky=tk.E)
        
        # Performance section
        perf_frame = ttk.LabelFrame(main_frame, text="📊 Performance", padding="5")
        perf_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        self.perf_text = tk.Text(perf_frame, height=15, width=40)
        self.perf_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Active trades section
        trades_frame = ttk.LabelFrame(main_frame, text="⚡ Active Trades", padding="5")
        trades_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        self.trades_text = tk.Text(trades_frame, height=15, width=40)
        self.trades_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        self.start_button = ttk.Button(control_frame, text="🚀 Start Bot", command=self.start_bot)
        self.start_button.grid(row=0, column=0, padx=(0, 5))
        
        self.stop_button = ttk.Button(control_frame, text="🛑 Stop Bot", command=self.stop_bot)
        self.stop_button.grid(row=0, column=1, padx=(5, 5))
        
        self.chart_button = ttk.Button(control_frame, text="📈 Show Chart", command=self.show_chart)
        self.chart_button.grid(row=0, column=2, padx=(5, 0))
        
        # Start update loop
        self.update_gui()
    
    def update_gui(self):
        """Update GUI with current data"""
        if not GUI_AVAILABLE:
            return
        
        try:
            # Update bot status
            if self.bot.is_running:
                self.status_label.config(text="🟢 Running")
            else:
                self.status_label.config(text="🔴 Stopped")
            
            self.capital_label.config(text=f"💰 Capital: ${self.bot.current_capital:,.2f}")
            
            # Update performance
            stats = self.bot.performance_tracker.get_live_stats()
            perf_text = f"""📊 PERFORMANCE SUMMARY
Total Trades: {stats['total_trades']}
Wins: {stats['winning_trades']} | Losses: {stats['losing_trades']}
Win Rate: {stats['win_rate']:.1f}%
Net P&L: ${stats['net_pnl']:+.2f}
Best Trade: ${stats['best_trade']:+.2f}
Worst Trade: ${stats['worst_trade']:+.2f}
Current Streak: {abs(stats['current_streak'])} {'wins' if stats['current_streak'] > 0 else 'losses'}

💰 TRADE QUALITY:
Average Win: ${stats['average_win']:.2f}
Average Loss: ${stats['average_loss']:.2f}
Profit Factor: {stats['profit_factor']:.2f}
"""
            
            self.perf_text.delete(1.0, tk.END)
            self.perf_text.insert(1.0, perf_text)
            
            # Update active trades
            active_trades_text = "⚡ ACTIVE TRADES:\n\n"
            for trade_id, trade in self.bot.performance_tracker.active_trades.items():
                duration = (datetime.now() - trade['entry_time']).total_seconds() / 60
                active_trades_text += f"{trade['symbol']} {trade['side']}\n"
                active_trades_text += f"Entry: ${trade['entry_price']:.6f}\n"
                active_trades_text += f"TP: ${trade['take_profit']:.6f}\n"
                active_trades_text += f"SL: ${trade['stop_loss']:.6f}\n"
                active_trades_text += f"Duration: {duration:.1f}min\n"
                active_trades_text += f"Confidence: {trade['confidence']*100:.1f}%\n"
                active_trades_text += "-" * 25 + "\n"
            
            if not self.bot.performance_tracker.active_trades:
                active_trades_text += "No active trades"
            
            self.trades_text.delete(1.0, tk.END)
            self.trades_text.insert(1.0, active_trades_text)
            
        except Exception as e:
            print(f"GUI update error: {e}")
        
        # Schedule next update
        self.root.after(5000, self.update_gui)  # Update every 5 seconds
    
    def start_bot(self):
        """Start the trading bot"""
        if not self.bot.is_running:
            asyncio.create_task(self.bot.start_improved_trading())
    
    def stop_bot(self):
        """Stop the trading bot"""
        self.bot.is_running = False
    
    def show_chart(self):
        """Show/update the live chart"""
        if PLOTTING_AVAILABLE:
            plt.show()
    
    def run(self):
        """Run the GUI"""
        if GUI_AVAILABLE:
            self.root.mainloop()


class EnhancedTradePerformanceTracker:
    """
    📊 ENHANCED WIN/LOSS TRACKING WITH VISUALIZATION
    
    Features:
    • Comprehensive statistics
    • Visual trade tracking
    • Real-time updates
    • Performance analytics
    • Trade outcome visualization
    """
    
    def __init__(self, chart_manager=None):
        self.trades = []
        self.active_trades = {}
        self.chart_manager = chart_manager
        
        # Performance Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'net_pnl': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'profit_factor': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'current_streak': 0,
            'longest_win_streak': 0,
            'longest_loss_streak': 0,
        }
        
        # Enhanced tracking
        self.hourly_performance = {}
        self.strategy_performance = {}
        self.symbol_performance = {}
    
    def start_trade(self, trade_info):
        """Start tracking a new trade with visualization"""
        trade_id = trade_info['trade_id']
        
        trade_record = {
            'trade_id': trade_id,
            'symbol': trade_info['symbol'],
            'side': trade_info['side'],
            'entry_price': trade_info['entry_price'],
            'position_size': trade_info['position_size'],
            'stop_loss': trade_info.get('stop_loss'),
            'take_profit': trade_info.get('take_profit'),
            'strategy': trade_info.get('strategy', 'Unknown'),
            'entry_time': datetime.now(),
            'status': 'ACTIVE',
            'confidence': trade_info.get('confidence', 0.5)
        }
        
        self.active_trades[trade_id] = trade_record
        
        # Update chart with TP/SL levels
        if self.chart_manager:
            self.chart_manager.set_trade_levels(
                trade_info['symbol'],
                trade_info['entry_price'],
                trade_info.get('take_profit'),
                trade_info.get('stop_loss')
            )
        
        print(f"📈 TRADE STARTED: {trade_info['symbol']} {trade_info['side']} @ {trade_info['entry_price']:.6f}")
        print(f"🎯 TP: {trade_info.get('take_profit', 0):.6f} | SL: {trade_info.get('stop_loss', 0):.6f}")
        
        return trade_id
    
    def close_trade(self, trade_id, exit_price, exit_reason='MANUAL'):
        """Close a trade and record the result with visualization"""
        if trade_id not in self.active_trades:
            print(f"❌ Trade {trade_id} not found")
            return None
        
        trade = self.active_trades[trade_id]
        trade['exit_price'] = exit_price
        trade['exit_time'] = datetime.now()
        trade['exit_reason'] = exit_reason
        trade['duration'] = (trade['exit_time'] - trade['entry_time']).total_seconds()
        
        # Calculate PnL
        if trade['side'].upper() == 'BUY':
            pnl_pct = (exit_price - trade['entry_price']) / trade['entry_price']
        else:  # SELL
            pnl_pct = (trade['entry_price'] - exit_price) / trade['entry_price']
        
        pnl_amount = trade['position_size'] * pnl_pct
        trade['pnl_pct'] = pnl_pct
        trade['pnl_amount'] = pnl_amount
        trade['status'] = 'CLOSED'
        
        # Determine win/loss
        is_winner = pnl_amount > 0
        trade['result'] = 'WIN' if is_winner else 'LOSS'
        
        # Update statistics
        self._update_statistics(trade, is_winner)
        
        # Update chart with trade completion
        if self.chart_manager:
            duration_minutes = trade['duration'] / 60
            self.chart_manager.add_trade_completion(duration_minutes, pnl_amount)
        
        # Move to completed trades
        self.trades.append(trade)
        del self.active_trades[trade_id]
        
        # Enhanced result display
        result_symbol = "✅ WIN" if is_winner else "❌ LOSS"
        reason_emoji = {
            'TAKE_PROFIT': '🎯',
            'STOP_LOSS': '🛡️',
            'MAX_DURATION': '⏰',
            'EARLY_EXIT_BIG_MOVE': '🚀',
            'MARKET_EXIT': '📊'
        }.get(exit_reason, '📝')
        
        print(f"\n{result_symbol} TRADE COMPLETED:")
        print(f"   Symbol: {trade['symbol']}")
        print(f"   P&L: {pnl_pct*100:+.2f}% (${pnl_amount:+.2f})")
        print(f"   Duration: {trade['duration']/60:.1f} minutes")
        print(f"   Exit Reason: {reason_emoji} {exit_reason}")
        print(f"   Entry: {trade['entry_price']:.6f} → Exit: {exit_price:.6f}")
        
        # Update running statistics display
        self._display_quick_stats()
        
        return trade
    
    def _update_statistics(self, trade, is_winner):
        """Update performance statistics"""
        self.stats['total_trades'] += 1
        
        if is_winner:
            self.stats['winning_trades'] += 1
            self.stats['total_profit'] += trade['pnl_amount']
            self.stats['current_streak'] = max(0, self.stats['current_streak']) + 1
            self.stats['longest_win_streak'] = max(self.stats['longest_win_streak'], self.stats['current_streak'])
            
            if trade['pnl_amount'] > self.stats['best_trade']:
                self.stats['best_trade'] = trade['pnl_amount']
        else:
            self.stats['losing_trades'] += 1
            self.stats['total_loss'] += abs(trade['pnl_amount'])
            self.stats['current_streak'] = min(0, self.stats['current_streak']) - 1
            self.stats['longest_loss_streak'] = max(self.stats['longest_loss_streak'], abs(self.stats['current_streak']))
            
            if trade['pnl_amount'] < self.stats['worst_trade']:
                self.stats['worst_trade'] = trade['pnl_amount']
        
        # Recalculate derived stats
        if self.stats['total_trades'] > 0:
            self.stats['win_rate'] = (self.stats['winning_trades'] / self.stats['total_trades']) * 100
        
        self.stats['net_pnl'] = self.stats['total_profit'] - self.stats['total_loss']
        
        if self.stats['winning_trades'] > 0:
            self.stats['average_win'] = self.stats['total_profit'] / self.stats['winning_trades']
        
        if self.stats['losing_trades'] > 0:
            self.stats['average_loss'] = self.stats['total_loss'] / self.stats['losing_trades']
            if self.stats['total_loss'] > 0:
                self.stats['profit_factor'] = self.stats['total_profit'] / self.stats['total_loss']
        
        # Update symbol-specific performance
        symbol = trade['symbol']
        if symbol not in self.symbol_performance:
            self.symbol_performance[symbol] = {'wins': 0, 'losses': 0, 'total_pnl': 0}
        
        if is_winner:
            self.symbol_performance[symbol]['wins'] += 1
        else:
            self.symbol_performance[symbol]['losses'] += 1
        
        self.symbol_performance[symbol]['total_pnl'] += trade['pnl_amount']
    
    def _display_quick_stats(self):
        """Display quick statistics after each trade"""
        print(f"📊 QUICK STATS: {self.stats['winning_trades']}W/{self.stats['losing_trades']}L "
              f"({self.stats['win_rate']:.1f}%) | Net: ${self.stats['net_pnl']:+.2f}")
    
    def get_live_stats(self):
        """Get current performance statistics"""
        return self.stats.copy()
    
    def display_enhanced_dashboard(self):
        """Display enhanced performance dashboard with symbol breakdown"""
        print("\n" + "="*80)
        print("📊 ENHANCED TRADING PERFORMANCE DASHBOARD")
        print("="*80)
        
        # Overall Performance
        print(f"📈 OVERALL PERFORMANCE:")
        print(f"   Total Trades: {self.stats['total_trades']}")
        print(f"   Wins: {self.stats['winning_trades']} | Losses: {self.stats['losing_trades']}")
        print(f"   Win Rate: {self.stats['win_rate']:.1f}%")
        print(f"   Net P&L: ${self.stats['net_pnl']:+.2f}")
        
        # Trade Quality
        print(f"\n💰 TRADE QUALITY:")
        print(f"   Average Win: ${self.stats['average_win']:.2f}")
        print(f"   Average Loss: ${self.stats['average_loss']:.2f}")
        print(f"   Profit Factor: {self.stats['profit_factor']:.2f}")
        print(f"   Best Trade: ${self.stats['best_trade']:+.2f}")
        print(f"   Worst Trade: ${self.stats['worst_trade']:+.2f}")
        
        # Streak Information
        print(f"\n🔥 STREAK ANALYSIS:")
        streak_text = f"{abs(self.stats['current_streak'])} {'wins' if self.stats['current_streak'] > 0 else 'losses'}"
        print(f"   Current Streak: {streak_text}")
        print(f"   Longest Win Streak: {self.stats['longest_win_streak']}")
        print(f"   Longest Loss Streak: {self.stats['longest_loss_streak']}")
        
        # Symbol Performance
        if self.symbol_performance:
            print(f"\n📈 PERFORMANCE BY SYMBOL:")
            for symbol, perf in self.symbol_performance.items():
                total = perf['wins'] + perf['losses']
                win_rate = (perf['wins'] / total * 100) if total > 0 else 0
                print(f"   {symbol}: {perf['wins']}W/{perf['losses']}L "
                      f"({win_rate:.1f}%) | P&L: ${perf['total_pnl']:+.2f}")
        
        # Active Trades
        if self.active_trades:
            print(f"\n⚡ ACTIVE TRADES:")
            for trade_id, trade in self.active_trades.items():
                duration = (datetime.now() - trade['entry_time']).total_seconds() / 60
                print(f"   {trade['symbol']} {trade['side']} @ {trade['entry_price']:.6f}")
                print(f"      TP: {trade['take_profit']:.6f} | SL: {trade['stop_loss']:.6f}")
                print(f"      Duration: {duration:.1f}min | Confidence: {trade['confidence']*100:.1f}%")
        
        print("="*80 + "\n")


class EnhancedRealisticTradeManager:
    """
    ⏰ ENHANCED TRADE MANAGEMENT WITH VISUALIZATION
    
    Features:
    • Live price tracking with TP/SL visualization
    • Real-time trade monitoring
    • Enhanced exit logic
    • Visual feedback
    • Better trade management
    """
    
    def __init__(self, performance_tracker, chart_manager=None):
        self.performance_tracker = performance_tracker
        self.chart_manager = chart_manager
        
        self.trade_settings = {
            'min_trade_duration_minutes': 15,   # Minimum 15 minutes
            'max_trade_duration_minutes': 120,  # Maximum 2 hours
            'tp_sl_check_interval': 30,         # Check TP/SL every 30 seconds
            'chart_update_interval': 10,        # Update chart every 10 seconds
            'early_exit_threshold': 0.15,       # 15% threshold for early exit
        }
        
        self.market_simulator = EnhancedMarketSimulator()
    
    async def execute_trade_with_visualization(self, trade_signal):
        """Execute trade with live visualization"""
        
        # Start the trade
        trade_id = self.performance_tracker.start_trade({
            'trade_id': f"TRADE_{int(time.time())}",
            'symbol': trade_signal['symbol'],
            'side': trade_signal['type'],
            'entry_price': trade_signal['entry_price'],
            'position_size': trade_signal.get('position_size', 100),
            'stop_loss': trade_signal.get('stop_loss'),
            'take_profit': trade_signal.get('profit_target'),
            'strategy': trade_signal.get('strategy', 'AI'),
            'confidence': trade_signal.get('confidence', 0.7)
        })
        
        # Start live price tracking for chart
        if self.chart_manager:
            asyncio.create_task(self._live_price_tracking(trade_signal['symbol'], trade_id))
        
        # Manage the trade until exit
        exit_result = await self._manage_trade_with_visualization(trade_id, trade_signal)
        
        return exit_result
    
    async def _live_price_tracking(self, symbol, trade_id):
        """Track live prices and update chart"""
        entry_price = self.performance_tracker.active_trades[trade_id]['entry_price']
        
        while trade_id in self.performance_tracker.active_trades:
            try:
                current_price = await self.market_simulator.get_current_price(symbol, entry_price)
                
                if self.chart_manager:
                    self.chart_manager.add_price_point(symbol, current_price)
                    
                    # Update chart every 10 seconds
                    if int(time.time()) % 10 == 0:
                        stats = self.performance_tracker.get_live_stats()
                        self.chart_manager.update_chart(symbol, stats)
                
                await asyncio.sleep(self.trade_settings['chart_update_interval'])
                
            except Exception as e:
                print(f"Price tracking error: {e}")
                await asyncio.sleep(30)
    
    async def _manage_trade_with_visualization(self, trade_id, trade_signal):
        """Manage trade with enhanced visualization"""
        
        if trade_id not in self.performance_tracker.active_trades:
            return {'success': False, 'error': 'Trade not found'}
        
        trade = self.performance_tracker.active_trades[trade_id]
        entry_time = trade['entry_time']
        entry_price = trade['entry_price']
        stop_loss = trade['stop_loss']
        take_profit = trade['take_profit']
        symbol = trade['symbol']
        side = trade['side']
        
        print(f"\n🎯 MANAGING {symbol} TRADE WITH LIVE VISUALIZATION:")
        print(f"   Entry Price: {entry_price:.6f}")
        print(f"   Take Profit: {take_profit:.6f} ({((take_profit/entry_price-1)*100):+.1f}%)")
        print(f"   Stop Loss: {stop_loss:.6f} ({((stop_loss/entry_price-1)*100):+.1f}%)")
        print(f"   Position Size: ${trade['position_size']}")
        print(f"   Strategy: {trade['strategy']}")
        print(f"   Confidence: {trade['confidence']*100:.1f}%")
        
        last_chart_update = datetime.now()
        
        while trade_id in self.performance_tracker.active_trades:
            current_time = datetime.now()
            trade_duration = (current_time - entry_time).total_seconds() / 60  # in minutes
            
            # Get current market price
            current_price = await self.market_simulator.get_current_price(symbol, entry_price)
            
            # Calculate current P&L
            if side.upper() == 'BUY':
                current_pnl_pct = (current_price - entry_price) / entry_price
                distance_to_tp = ((take_profit - current_price) / current_price) * 100
                distance_to_sl = ((current_price - stop_loss) / current_price) * 100
            else:
                current_pnl_pct = (entry_price - current_price) / entry_price
                distance_to_tp = ((current_price - take_profit) / current_price) * 100
                distance_to_sl = ((stop_loss - current_price) / current_price) * 100
            
            # Enhanced status display
            status_line = (f"⏰ {symbol} | {trade_duration:.1f}min | "
                         f"P&L: {current_pnl_pct*100:+.2f}% | "
                         f"Price: {current_price:.6f} | "
                         f"TP: {distance_to_tp:+.1f}% | SL: {distance_to_sl:+.1f}%")
            
            if int(trade_duration) % 2 == 0:  # Every 2 minutes
                print(status_line)
            
            # Check exit conditions
            exit_reason = None
            
            # 1. Take Profit Hit
            if take_profit:
                if (side.upper() == 'BUY' and current_price >= take_profit) or \
                   (side.upper() == 'SELL' and current_price <= take_profit):
                    exit_reason = 'TAKE_PROFIT'
                    print(f"🎯 TAKE PROFIT HIT! {current_price:.6f} >= {take_profit:.6f}")
            
            # 2. Stop Loss Hit
            if stop_loss and not exit_reason:
                if (side.upper() == 'BUY' and current_price <= stop_loss) or \
                   (side.upper() == 'SELL' and current_price >= stop_loss):
                    exit_reason = 'STOP_LOSS'
                    print(f"🛡️ STOP LOSS HIT! {current_price:.6f} <= {stop_loss:.6f}")
            
            # 3. Maximum duration reached
            if trade_duration >= self.trade_settings['max_trade_duration_minutes'] and not exit_reason:
                exit_reason = 'MAX_DURATION'
                print(f"⏰ MAX DURATION REACHED: {trade_duration:.1f} minutes")
            
            # 4. Early exit for big moves (after minimum duration)
            if trade_duration >= self.trade_settings['min_trade_duration_minutes'] and not exit_reason:
                if abs(current_pnl_pct) >= self.trade_settings['early_exit_threshold']:
                    exit_reason = 'EARLY_EXIT_BIG_MOVE'
                    print(f"🚀 BIG MOVE DETECTED: {current_pnl_pct*100:+.1f}% - Early exit!")
            
            # 5. Random market-based exit (simulate real market behavior)
            if trade_duration >= self.trade_settings['min_trade_duration_minutes'] and not exit_reason:
                if random.random() < 0.05:  # 5% chance per check after min duration
                    exit_reason = 'MARKET_EXIT'
                    print(f"📊 MARKET EXIT SIGNAL")
            
            # Exit if condition met
            if exit_reason:
                closed_trade = self.performance_tracker.close_trade(trade_id, current_price, exit_reason)
                return {
                    'success': True,
                    'trade_result': closed_trade,
                    'duration_minutes': trade_duration,
                    'exit_reason': exit_reason
                }
            
            # Wait before next check
            await asyncio.sleep(self.trade_settings['tp_sl_check_interval'])
        
        return {'success': False, 'error': 'Trade management loop ended unexpectedly'}


class EnhancedMarketSimulator:
    """
    📈 ENHANCED MARKET SIMULATOR WITH REALISTIC BEHAVIOR
    
    Features:
    • More realistic price movements
    • Support/resistance levels
    • Trend persistence
    • Volatility clustering
    • Market microstructure
    """
    
    def __init__(self):
        self.price_cache = {}
        self.support_resistance_levels = {}
        
    async def get_current_price(self, symbol, base_price):
        """Get enhanced realistic current market price"""
        
        if symbol not in self.price_cache:
            self.price_cache[symbol] = {
                'price': base_price,
                'last_update': datetime.now(),
                'trend': random.choice([1, -1]),
                'trend_strength': random.uniform(0.3, 0.8),
                'volatility': random.uniform(0.002, 0.015),
                'support_level': base_price * 0.97,
                'resistance_level': base_price * 1.03,
            }
        
        cache = self.price_cache[symbol]
        current_time = datetime.now()
        time_diff = (current_time - cache['last_update']).total_seconds()
        
        # Update price based on time passed
        if time_diff > 5:  # Update every 5 seconds
            
            # Trend persistence (trends tend to continue)
            if random.random() < 0.85:  # 85% chance to continue trend
                trend = cache['trend']
            else:
                trend = cache['trend'] * -1  # Reverse trend
                cache['trend'] = trend
            
            # Enhanced price movement with support/resistance
            base_move = trend * cache['trend_strength'] * cache['volatility'] * random.uniform(0.5, 2.0)
            
            # Check for support/resistance bounce
            current_price = cache['price']
            if current_price <= cache['support_level'] and trend < 0:
                base_move = abs(base_move)  # Bounce up from support
                print(f"🔄 {symbol} bounced off support at {cache['support_level']:.6f}")
            elif current_price >= cache['resistance_level'] and trend > 0:
                base_move = -abs(base_move)  # Reject at resistance
                print(f"🔄 {symbol} rejected at resistance at {cache['resistance_level']:.6f}")
            
            # Apply movement
            new_price = current_price * (1 + base_move)
            
            # Add realistic noise
            noise = random.uniform(-0.001, 0.001)  # ±0.1% noise
            new_price *= (1 + noise)
            
            # Update cache
            cache['price'] = new_price
            cache['last_update'] = current_time
            
            # Occasionally update support/resistance levels
            if random.random() < 0.1:  # 10% chance
                cache['support_level'] = new_price * random.uniform(0.95, 0.99)
                cache['resistance_level'] = new_price * random.uniform(1.01, 1.05)
        
        return cache['price']


class ImprovedAIBrain:
    """
    🧠 ENHANCED AI TRADING BRAIN
    
    Improvements:
    • Better signal quality
    • Market condition analysis
    • Multi-timeframe analysis
    • Confidence scoring
    • Quality filtering
    """
    
    def __init__(self):
        self.last_signal_time = {}
        self.signal_cooldown_minutes = 25  # Minimum 25 minutes between signals
        self.min_confidence_threshold = 0.75  # Minimum 75% confidence
        
        self.market_analysis_history = {}
    
    def should_generate_signal(self, symbol):
        """Enhanced signal timing logic"""
        if symbol not in self.last_signal_time:
            return True
        
        last_time = self.last_signal_time[symbol]
        time_diff = (datetime.now() - last_time).total_seconds() / 60
        
        return time_diff >= self.signal_cooldown_minutes
    
    async def analyze_and_generate_signal(self, market_data):
        """Generate high-quality trading signal with enhanced analysis"""
        
        # Enhanced market scan
        best_opportunity = None
        best_confidence = 0
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'PEPE/USDT', 'SHIB/USDT', 'DOGE/USDT', 'SOL/USDT']
        
        print(f"🧠 AI ANALYZING {len(symbols)} SYMBOLS...")
        
        for symbol in symbols:
            
            # Check cooldown
            if not self.should_generate_signal(symbol):
                print(f"   ⏰ {symbol}: Cooldown active ({self.signal_cooldown_minutes}min)")
                continue
            
            # Enhanced market analysis
            confidence, analysis = self._enhanced_market_analysis(symbol)
            print(f"   📊 {symbol}: Confidence {confidence*100:.1f}% - {analysis['market_condition']}")
            
            if confidence > best_confidence and confidence >= self.min_confidence_threshold:
                best_confidence = confidence
                
                # Generate enhanced signal
                base_price = random.uniform(0.0001, 50000)
                trend_direction = analysis['recommended_direction']
                
                # Dynamic TP/SL based on market conditions
                if analysis['volatility_score'] > 0.7:
                    profit_target_pct = random.uniform(0.05, 0.12)  # Higher targets for volatile markets
                else:
                    profit_target_pct = random.uniform(0.03, 0.08)  # Conservative targets for calm markets
                
                stop_loss_pct = profit_target_pct * 0.4  # Risk:Reward 1:2.5
                
                if trend_direction == 'BUY':
                    profit_target = base_price * (1 + profit_target_pct)
                    stop_loss = base_price * (1 - stop_loss_pct)
                else:
                    profit_target = base_price * (1 - profit_target_pct)
                    stop_loss = base_price * (1 + stop_loss_pct)
                
                best_opportunity = {
                    'symbol': symbol,
                    'type': trend_direction,
                    'entry_price': base_price,
                    'profit_target': profit_target,
                    'stop_loss': stop_loss,
                    'confidence': confidence,
                    'expected_profit_pct': profit_target_pct,
                    'strategy': 'ENHANCED_AI',
                    'position_size': random.randint(75, 250),
                    'analysis': analysis,
                }
                
                # Update last signal time
                self.last_signal_time[symbol] = datetime.now()
        
        if best_opportunity:
            print(f"🎯 BEST OPPORTUNITY SELECTED: {best_opportunity['symbol']} "
                  f"({best_opportunity['confidence']*100:.1f}% confidence)")
        else:
            print("🔍 No opportunities meet quality threshold - waiting...")
        
        return best_opportunity
    
    def _enhanced_market_analysis(self, symbol):
        """Enhanced market analysis with multiple factors"""
        
        # Simulate comprehensive analysis
        factors = {
            'trend_strength': random.uniform(0.3, 1.0),
            'volume_profile': random.uniform(0.2, 1.0),
            'volatility_score': random.uniform(0.1, 1.0),
            'momentum_score': random.uniform(0.2, 1.0),
            'support_resistance': random.uniform(0.3, 1.0),
            'market_sentiment': random.uniform(0.2, 1.0),
            'technical_indicators': random.uniform(0.4, 1.0),
        }
        
        # Weighted confidence calculation
        weights = {
            'trend_strength': 0.25,
            'volume_profile': 0.15,
            'volatility_score': 0.10,
            'momentum_score': 0.20,
            'support_resistance': 0.15,
            'market_sentiment': 0.10,
            'technical_indicators': 0.05,
        }
        
        confidence = sum(factors[key] * weights[key] for key in factors.keys())
        
        # Add quality filter
        if factors['volume_profile'] < 0.3 or factors['trend_strength'] < 0.4:
            confidence *= 0.7  # Reduce confidence for poor conditions
        
        # Determine market condition and direction
        if factors['momentum_score'] > 0.6 and factors['trend_strength'] > 0.6:
            market_condition = 'STRONG_TREND'
            recommended_direction = 'BUY' if random.random() > 0.5 else 'SELL'
        elif factors['volatility_score'] > 0.8:
            market_condition = 'HIGH_VOLATILITY'
            recommended_direction = random.choice(['BUY', 'SELL'])
        elif factors['support_resistance'] > 0.7:
            market_condition = 'RANGE_BOUND'
            recommended_direction = random.choice(['BUY', 'SELL'])
        else:
            market_condition = 'UNCLEAR'
            recommended_direction = 'BUY'
            confidence *= 0.6  # Reduce confidence for unclear conditions
        
        analysis = {
            'market_condition': market_condition,
            'recommended_direction': recommended_direction,
            'volatility_score': factors['volatility_score'],
            'trend_score': factors['trend_strength'],
            'volume_score': factors['volume_profile'],
            'factors': factors
        }
        
        return min(confidence, 1.0), analysis


class EnhancedAutonomousBot:
    """
    🤖 ENHANCED AUTONOMOUS TRADING BOT WITH LIVE VISUALIZATION
    
    New Features:
    📈 Live price charts with TP/SL levels
    📊 Real-time performance visualization
    🎯 Interactive trading dashboard
    📉 Visual trade management
    🔥 Enhanced AI analysis
    """
    
    def __init__(self, initial_capital=5000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Initialize enhanced components
        self.chart_manager = LivePriceChart() if PLOTTING_AVAILABLE else None
        self.performance_tracker = EnhancedTradePerformanceTracker(self.chart_manager)
        self.trade_manager = EnhancedRealisticTradeManager(self.performance_tracker, self.chart_manager)
        self.ai_brain = ImprovedAIBrain()
        
        # Enhanced settings
        self.settings = {
            'max_concurrent_trades': 2,          # Reduced for better focus
            'market_check_interval': 300,       # 5 minutes
            'dashboard_update_interval': 300,    # 5 minutes
            'chart_update_interval': 10,        # 10 seconds for charts
            'daily_target_pct': 0.05,           # 5% daily target
        }
        
        # Bot state
        self.is_running = False
        self.start_time = None
        
        # Initialize GUI if available
        self.gui = TradingGUI(self) if GUI_AVAILABLE else None
        
    async def start_enhanced_trading(self):
        """Start the enhanced autonomous trading bot"""
        print("🤖 STARTING ENHANCED AUTONOMOUS TRADING BOT")
        print("=" * 70)
        print("🆕 NEW FEATURES:")
        print("   📈 Live price charts with TP/SL visualization")
        print("   📊 Real-time performance graphs")
        print("   🎯 Visual trade execution monitoring")
        print("   📉 Interactive P&L tracking")
        print("   🔥 Enhanced AI market analysis")
        print("")
        
        if not PLOTTING_AVAILABLE:
            print("⚠️ Charts disabled - install matplotlib: pip install matplotlib")
        
        if not GUI_AVAILABLE:
            print("⚠️ GUI disabled - tkinter not available")
        
        self.is_running = True
        self.start_time = datetime.now()
        
        print(f"💰 Starting Capital: ${self.current_capital:,}")
        print(f"🎯 Daily Target: {self.settings['daily_target_pct']*100}%")
        print(f"📊 Max Concurrent Trades: {self.settings['max_concurrent_trades']}")
        print(f"⏰ Market Analysis: Every {self.settings['market_check_interval']/60} minutes")
        print(f"📈 Chart Updates: Every {self.settings['chart_update_interval']} seconds")
        
        # Start enhanced trading loop
        await self._run_enhanced_trading_loop()
    
    async def _run_enhanced_trading_loop(self):
        """Enhanced trading loop with visualization"""
        print("\n🚀 ENHANCED BOT WITH LIVE VISUALIZATION IS RUNNING!")
        print("📈 Live charts will show price action with TP/SL levels")
        print("📊 Performance dashboard updates in real-time")
        print("🎯 Watch your trades come to life!\n")
        
        last_dashboard_update = datetime.now()
        last_chart_update = datetime.now()
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # 1. Check if we can take new trades
                active_trades = len(self.performance_tracker.active_trades)
                
                if active_trades < self.settings['max_concurrent_trades']:
                    
                    # 2. Enhanced AI market analysis
                    print("🧠 ENHANCED AI MARKET ANALYSIS...")
                    signal = await self.ai_brain.analyze_and_generate_signal({})
                    
                    if signal:
                        print(f"\n⚡ HIGH-QUALITY SIGNAL DETECTED:")
                        print(f"   📊 Symbol: {signal['symbol']}")
                        print(f"   📈 Direction: {signal['type']}")
                        print(f"   🎯 Confidence: {signal['confidence']*100:.1f}%")
                        print(f"   💰 Expected Profit: {signal['expected_profit_pct']*100:.1f}%")
                        print(f"   📉 Market Condition: {signal['analysis']['market_condition']}")
                        print(f"   🔍 Entry: {signal['entry_price']:.6f}")
                        print(f"   🎯 TP: {signal['profit_target']:.6f}")
                        print(f"   🛡️ SL: {signal['stop_loss']:.6f}")
                        
                        # 3. Execute with live visualization
                        asyncio.create_task(self.trade_manager.execute_trade_with_visualization(signal))
                    else:
                        print("🔍 No high-confidence setups found - maintaining quality standards")
                else:
                    print(f"⏳ Max trades active ({active_trades}/{self.settings['max_concurrent_trades']}) - focusing on current positions")
                
                # 4. Update dashboard periodically
                if (current_time - last_dashboard_update).total_seconds() >= self.settings['dashboard_update_interval']:
                    self.performance_tracker.display_enhanced_dashboard()
                    self._display_enhanced_status()
                    last_dashboard_update = current_time
                
                # 5. Update portfolio value for chart
                if self.chart_manager:
                    net_pnl = self.performance_tracker.stats['net_pnl']
                    self.chart_manager.update_portfolio_value(current_time, self.initial_capital + net_pnl)
                
                # 6. Wait before next analysis
                print(f"💤 Next analysis in {self.settings['market_check_interval']/60} minutes...")
                await asyncio.sleep(self.settings['market_check_interval'])
                
            except Exception as e:
                print(f"⚠️ Enhanced loop error: {e}")
                await asyncio.sleep(60)
    
    def _display_enhanced_status(self):
        """Display enhanced capital and growth status"""
        runtime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        net_pnl = self.performance_tracker.stats['net_pnl']
        self.current_capital = self.initial_capital + net_pnl
        total_growth = (net_pnl / self.initial_capital) * 100
        
        print(f"\n💰 ENHANCED CAPITAL STATUS:")
        print(f"   💵 Initial Capital: ${self.initial_capital:,}")
        print(f"   💰 Current Capital: ${self.current_capital:,}")
        print(f"   📈 Net P&L: ${net_pnl:+.2f}")
        print(f"   📊 Total Growth: {total_growth:+.2f}%")
        print(f"   ⏰ Runtime: {runtime_hours:.1f} hours")
        
        # Performance rate calculations
        if runtime_hours > 0:
            hourly_return = total_growth / runtime_hours
            daily_projection = hourly_return * 24
            print(f"   📈 Hourly Rate: {hourly_return:+.2f}%/hr")
            print(f"   🎯 Daily Projection: {daily_projection:+.1f}%")


async def main():
    """Launch the enhanced autonomous bot with live visualization"""
    print("🤖 ENHANCED AUTONOMOUS TRADING BOT")
    print("📈 NEW: Live graphs with TP/SL visualization!")
    print("🎯 Fixed: Rapid trading, Missing stats, No visual feedback")
    print("✅ Features: Live charts, Performance graphs, Visual trade management")
    print("")
    
    if not PLOTTING_AVAILABLE:
        print("📦 To enable live charts, install matplotlib:")
        print("   pip install matplotlib")
        print("")
    
    # Create enhanced bot
    bot = EnhancedAutonomousBot(initial_capital=5000)
    
    try:
        if GUI_AVAILABLE and bot.gui:
            # Run bot in background, GUI in foreground
            asyncio.create_task(bot.start_enhanced_trading())
            bot.gui.run()
        else:
            # Run bot directly
            await bot.start_enhanced_trading()
            
    except KeyboardInterrupt:
        print("\n🛑 ENHANCED BOT STOPPED BY USER")
        
        # Final comprehensive report
        bot.performance_tracker.display_enhanced_dashboard()
        
        print(f"\n🏁 FINAL ENHANCED RESULTS:")
        print(f"   💰 Final Capital: ${bot.current_capital:,.2f}")
        print(f"   📈 Total Growth: {((bot.current_capital - bot.initial_capital) / bot.initial_capital * 100):+.2f}%")
        print(f"   ⏰ Runtime: {((datetime.now() - bot.start_time).total_seconds() / 3600):.1f} hours")
        print(f"   📊 Total Trades: {bot.performance_tracker.stats['total_trades']}")
        print(f"   🏆 Win Rate: {bot.performance_tracker.stats['win_rate']:.1f}%")
        
        if PLOTTING_AVAILABLE:
            input("📈 Press Enter to close charts...")
            plt.close('all')


if __name__ == "__main__":
    print("🤖 ENHANCED AUTONOMOUS TRADING BOT")
    print("📈 NEW: Live price graphs with TP/SL levels!")
    print("✅ FIXED: All previous issues + Visual feedback")
    print("🎯 FEATURES: Live charts, Real-time stats, Visual trade management")
    print("")
    
    asyncio.run(main())
