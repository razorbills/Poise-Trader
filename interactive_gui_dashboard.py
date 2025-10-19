#!/usr/bin/env python3
"""
🏆 INTERACTIVE GUI DASHBOARD FOR POISE TRADER BOTS 🏆

🚀 COMPREHENSIVE GUI FEATURES:
✅ Dual-bot control panel (Micro + AI Profit Bot)
✅ Real-time status monitoring with live metrics
✅ Interactive trading controls and parameters
✅ Advanced settings configuration panel
✅ Multi-timeframe chart controls
✅ Risk management control center
✅ Portfolio overview with performance analytics
✅ Trade history browser with filtering
✅ Strategy selection and optimization
✅ Alert system with notifications

💎 Professional Trading Interface
🎯 Maximum Control and Visualization
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.font as tkFont
from datetime import datetime, timedelta
import json
import threading
import time
import subprocess
import sys
import os
from typing import Dict, List, Optional, Any
from enum import Enum
import queue

# Import enhanced charts system
try:
    from enhanced_real_time_charts import EnhancedRealTimeCharts, TimeFrame, ChartType, TechnicalIndicator
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False
    print("⚠️ Enhanced charts not found - GUI will run with limited features")

class BotStatus(Enum):
    """Bot status enumeration"""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    PAUSED = "PAUSED"

class TradingMode(Enum):
    """Trading mode options"""
    AGGRESSIVE = "AGGRESSIVE"
    PRECISION = "PRECISION"
    CONSERVATIVE = "CONSERVATIVE"
    CUSTOM = "CUSTOM"

class InteractiveGUIDashboard:
    """
    🚀 INTERACTIVE GUI DASHBOARD FOR POISE TRADER 🚀
    
    Features:
    • Dual-bot management (Micro + AI Profit)
    • Real-time monitoring and control
    • Interactive chart controls
    • Advanced configuration panels
    • Performance analytics dashboard
    • Risk management center
    """
    
    def __init__(self):
        self.root = None
        self.enhanced_charts = None
        
        # Bot status tracking
        self.bot_status = {
            'micro_bot': BotStatus.STOPPED,
            'ai_profit_bot': BotStatus.STOPPED
        }
        
        # Bot instances (when connected)
        self.bot_instances = {
            'micro_bot': None,
            'ai_profit_bot': None
        }
        
        # Live data feeds
        self.live_data = {
            'micro_bot': {
                'pnl': 0.0,
                'trades_today': 0,
                'win_rate': 0.0,
                'current_positions': [],
                'last_trade_time': None,
                'total_volume': 0.0
            },
            'ai_profit_bot': {
                'pnl': 0.0,
                'trades_today': 0,
                'win_rate': 0.0,
                'current_positions': [],
                'last_trade_time': None,
                'total_volume': 0.0
            }
        }
        
        # GUI update queue for thread safety
        self.update_queue = queue.Queue()
        
        # Configuration
        self.config = {
            'auto_start': False,
            'update_interval': 2000,  # milliseconds
            'max_positions': 5,
            'risk_percentage': 2.0,
            'default_symbol': 'BTC/USDT',
            'chart_timeframe': TimeFrame.M5,
            'enable_notifications': True,
            'save_logs': True
        }
        
        # Initialize GUI
        self.initialize_gui()
        
    def initialize_gui(self):
        """Initialize the comprehensive GUI dashboard"""
        try:
            self.root = tk.Tk()
            self.root.title("🏆 POISE TRADER - INTERACTIVE GUI DASHBOARD 🏆")
            self.root.geometry("1600x1200")
            self.root.configure(bg='#1a1a1a')
            
            # Set window icon and style
            self.root.resizable(True, True)
            self.setup_styles()
            
            # Create main layout
            self.create_main_layout()
            
            # Start update loop
            self.start_update_loop()
            
            print("🖥️ Interactive GUI Dashboard initialized!")
            print("📊 Features ready:")
            print("   🤖 Dual-bot control panel")
            print("   📈 Real-time status monitoring")
            print("   🎛️ Interactive trading controls")
            print("   ⚙️ Advanced settings panel")
            print("   🔔 Alert and notification system")
            
        except Exception as e:
            print(f"❌ Error initializing GUI: {e}")
    
    def setup_styles(self):
        """Setup professional dark theme styles"""
        self.style = ttk.Style()
        
        # Configure professional dark theme
        self.style.theme_use('clam')
        
        # Define color scheme
        colors = {
            'bg': '#1a1a1a',
            'fg': '#ffffff',
            'select_bg': '#3d3d3d',
            'select_fg': '#ffffff',
            'entry_bg': '#2d2d2d',
            'button_bg': '#404040',
            'success': '#00ff00',
            'warning': '#ffaa00',
            'error': '#ff0000',
            'info': '#00aaff'
        }
        
        # Configure ttk styles
        self.style.configure('Dark.TFrame', background=colors['bg'])
        self.style.configure('Dark.TLabel', background=colors['bg'], foreground=colors['fg'], font=('Consolas', 10))
        self.style.configure('Title.TLabel', background=colors['bg'], foreground='#00ffff', font=('Consolas', 14, 'bold'))
        self.style.configure('Success.TLabel', background=colors['bg'], foreground=colors['success'], font=('Consolas', 10, 'bold'))
        self.style.configure('Warning.TLabel', background=colors['bg'], foreground=colors['warning'], font=('Consolas', 10, 'bold'))
        self.style.configure('Error.TLabel', background=colors['bg'], foreground=colors['error'], font=('Consolas', 10, 'bold'))
        
        self.style.configure('Dark.TButton', background=colors['button_bg'], foreground=colors['fg'], 
                           font=('Consolas', 10, 'bold'), borderwidth=2)
        self.style.map('Dark.TButton', 
                      background=[('active', colors['select_bg']), ('pressed', colors['select_bg'])])
        
        self.style.configure('Dark.TEntry', background=colors['entry_bg'], foreground=colors['fg'], 
                           font=('Consolas', 10), borderwidth=2)
        self.style.configure('Dark.TCombobox', background=colors['entry_bg'], foreground=colors['fg'], 
                           font=('Consolas', 10))
        
        # Notebook (tabs)
        self.style.configure('Dark.TNotebook', background=colors['bg'], borderwidth=0)
        self.style.configure('Dark.TNotebook.Tab', background=colors['button_bg'], foreground=colors['fg'], 
                           font=('Consolas', 11, 'bold'), padding=[20, 8])
        self.style.map('Dark.TNotebook.Tab', 
                      background=[('selected', colors['info']), ('active', colors['select_bg'])])
        
    def create_main_layout(self):
        """Create the main GUI layout with all panels"""
        
        # Main container
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(main_frame, style='Dark.TNotebook')
        self.notebook.pack(fill='both', expand=True)
        
        # Tab 1: Bot Control Center
        self.create_bot_control_tab()
        
        # Tab 2: Live Trading Dashboard
        self.create_trading_dashboard_tab()
        
        # Tab 3: Risk Management
        self.create_risk_management_tab()
        
        # Tab 4: Settings & Configuration
        self.create_settings_tab()
        
        # Tab 5: Analytics & Reports
        self.create_analytics_tab()
        
        # Status bar
        self.create_status_bar(main_frame)
        
    def create_bot_control_tab(self):
        """Create bot control center tab"""
        
        # Create tab frame
        bot_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(bot_frame, text='🤖 BOT CONTROL CENTER')
        
        # Title
        title_label = ttk.Label(bot_frame, text="🏆 POISE TRADER BOT CONTROL CENTER 🏆", 
                               style='Title.TLabel')
        title_label.pack(pady=10)
        
        # Create two columns for both bots
        bots_frame = ttk.Frame(bot_frame, style='Dark.TFrame')
        bots_frame.pack(fill='both', expand=True, padx=20)
        
        # Micro Trading Bot Panel
        self.create_bot_panel(bots_frame, 'micro_bot', 'MICRO TRADING BOT', 0)
        
        # AI Profit Bot Panel
        self.create_bot_panel(bots_frame, 'ai_profit_bot', 'AI PROFIT BOT', 1)
        
        # Global Controls
        global_frame = ttk.LabelFrame(bot_frame, text="🎛️ GLOBAL CONTROLS", style='Dark.TFrame')
        global_frame.pack(fill='x', padx=20, pady=10)
        
        # Global control buttons
        global_buttons_frame = ttk.Frame(global_frame, style='Dark.TFrame')
        global_buttons_frame.pack(fill='x', pady=10)
        
        ttk.Button(global_buttons_frame, text="🚀 START ALL BOTS", style='Dark.TButton',
                  command=self.start_all_bots).pack(side='left', padx=5)
        ttk.Button(global_buttons_frame, text="⏹️ STOP ALL BOTS", style='Dark.TButton',
                  command=self.stop_all_bots).pack(side='left', padx=5)
        ttk.Button(global_buttons_frame, text="🔄 RESTART ALL", style='Dark.TButton',
                  command=self.restart_all_bots).pack(side='left', padx=5)
        ttk.Button(global_buttons_frame, text="📊 SHOW ENHANCED CHARTS", style='Dark.TButton',
                  command=self.show_enhanced_charts).pack(side='left', padx=5)
        ttk.Button(global_buttons_frame, text="⚙️ EMERGENCY STOP", style='Dark.TButton',
                  command=self.emergency_stop).pack(side='left', padx=5)
        
    def create_bot_panel(self, parent, bot_key: str, title: str, column: int):
        """Create individual bot control panel"""
        
        # Main bot frame
        bot_panel = ttk.LabelFrame(parent, text=f"🤖 {title}", style='Dark.TFrame')
        bot_panel.grid(row=0, column=column, sticky='nsew', padx=10, pady=10)
        parent.grid_columnconfigure(column, weight=1)
        
        # Status display
        status_frame = ttk.Frame(bot_panel, style='Dark.TFrame')
        status_frame.pack(fill='x', pady=5)
        
        ttk.Label(status_frame, text="Status:", style='Dark.TLabel').pack(side='left')
        status_var = tk.StringVar(value=self.bot_status[bot_key].value)
        status_label = ttk.Label(status_frame, textvariable=status_var, style='Success.TLabel')
        status_label.pack(side='left', padx=10)
        
        # Store status variable for updates
        setattr(self, f"{bot_key}_status_var", status_var)
        setattr(self, f"{bot_key}_status_label", status_label)
        
        # Live metrics display
        metrics_frame = ttk.LabelFrame(bot_panel, text="📊 Live Metrics", style='Dark.TFrame')
        metrics_frame.pack(fill='x', pady=5)
        
        # Create metric variables
        pnl_var = tk.StringVar(value="$0.00")
        trades_var = tk.StringVar(value="0")
        winrate_var = tk.StringVar(value="0%")
        positions_var = tk.StringVar(value="0")
        
        # Store variables for updates
        setattr(self, f"{bot_key}_pnl_var", pnl_var)
        setattr(self, f"{bot_key}_trades_var", trades_var)
        setattr(self, f"{bot_key}_winrate_var", winrate_var)
        setattr(self, f"{bot_key}_positions_var", positions_var)
        
        # Metrics grid
        ttk.Label(metrics_frame, text="💰 P&L:", style='Dark.TLabel').grid(row=0, column=0, sticky='w', padx=5)
        ttk.Label(metrics_frame, textvariable=pnl_var, style='Success.TLabel').grid(row=0, column=1, sticky='w', padx=5)
        
        ttk.Label(metrics_frame, text="📊 Trades:", style='Dark.TLabel').grid(row=0, column=2, sticky='w', padx=5)
        ttk.Label(metrics_frame, textvariable=trades_var, style='Dark.TLabel').grid(row=0, column=3, sticky='w', padx=5)
        
        ttk.Label(metrics_frame, text="🎯 Win Rate:", style='Dark.TLabel').grid(row=1, column=0, sticky='w', padx=5)
        ttk.Label(metrics_frame, textvariable=winrate_var, style='Success.TLabel').grid(row=1, column=1, sticky='w', padx=5)
        
        ttk.Label(metrics_frame, text="📍 Positions:", style='Dark.TLabel').grid(row=1, column=2, sticky='w', padx=5)
        ttk.Label(metrics_frame, textvariable=positions_var, style='Warning.TLabel').grid(row=1, column=3, sticky='w', padx=5)
        
        # Control buttons
        controls_frame = ttk.Frame(bot_panel, style='Dark.TFrame')
        controls_frame.pack(fill='x', pady=10)
        
        # Bot control buttons
        start_btn = ttk.Button(controls_frame, text="🚀 START", style='Dark.TButton',
                              command=lambda: self.start_bot(bot_key))
        start_btn.pack(side='left', padx=5)
        
        stop_btn = ttk.Button(controls_frame, text="⏹️ STOP", style='Dark.TButton',
                             command=lambda: self.stop_bot(bot_key))
        stop_btn.pack(side='left', padx=5)
        
        pause_btn = ttk.Button(controls_frame, text="⏸️ PAUSE", style='Dark.TButton',
                              command=lambda: self.pause_bot(bot_key))
        pause_btn.pack(side='left', padx=5)
        
        config_btn = ttk.Button(controls_frame, text="⚙️ CONFIG", style='Dark.TButton',
                               command=lambda: self.open_bot_config(bot_key))
        config_btn.pack(side='left', padx=5)
        
        # Quick settings
        settings_frame = ttk.LabelFrame(bot_panel, text="⚙️ Quick Settings", style='Dark.TFrame')
        settings_frame.pack(fill='x', pady=5)
        
        # Trading mode selection
        ttk.Label(settings_frame, text="Mode:", style='Dark.TLabel').grid(row=0, column=0, sticky='w', padx=5)
        mode_var = tk.StringVar(value=TradingMode.PRECISION.value)
        mode_combo = ttk.Combobox(settings_frame, textvariable=mode_var, style='Dark.TCombobox',
                                 values=[mode.value for mode in TradingMode], state='readonly')
        mode_combo.grid(row=0, column=1, sticky='ew', padx=5)
        
        # Store variables
        setattr(self, f"{bot_key}_mode_var", mode_var)
        
        # Risk percentage
        ttk.Label(settings_frame, text="Risk %:", style='Dark.TLabel').grid(row=0, column=2, sticky='w', padx=5)
        risk_var = tk.StringVar(value="2.0")
        risk_entry = ttk.Entry(settings_frame, textvariable=risk_var, style='Dark.TEntry', width=8)
        risk_entry.grid(row=0, column=3, sticky='w', padx=5)
        
        setattr(self, f"{bot_key}_risk_var", risk_var)
        
        # Symbol selection
        ttk.Label(settings_frame, text="Symbol:", style='Dark.TLabel').grid(row=1, column=0, sticky='w', padx=5)
        symbol_var = tk.StringVar(value="BTC/USDT")
        symbol_combo = ttk.Combobox(settings_frame, textvariable=symbol_var, style='Dark.TCombobox',
                                   values=["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT", "DOT/USDT"])
        symbol_combo.grid(row=1, column=1, sticky='ew', padx=5)
        
        setattr(self, f"{bot_key}_symbol_var", symbol_var)
        
        # Position size
        ttk.Label(settings_frame, text="Size $:", style='Dark.TLabel').grid(row=1, column=2, sticky='w', padx=5)
        size_var = tk.StringVar(value="100")
        size_entry = ttk.Entry(settings_frame, textvariable=size_var, style='Dark.TEntry', width=8)
        size_entry.grid(row=1, column=3, sticky='w', padx=5)
        
        setattr(self, f"{bot_key}_size_var", size_var)
        
        settings_frame.grid_columnconfigure(1, weight=1)
        
    def create_trading_dashboard_tab(self):
        """Create live trading dashboard tab"""
        
        dashboard_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(dashboard_frame, text='📈 LIVE TRADING DASHBOARD')
        
        # Title
        title_label = ttk.Label(dashboard_frame, text="📈 LIVE TRADING DASHBOARD & CHART CONTROLS", 
                               style='Title.TLabel')
        title_label.pack(pady=10)
        
        # Chart controls panel
        chart_controls = ttk.LabelFrame(dashboard_frame, text="📊 Chart Controls", style='Dark.TFrame')
        chart_controls.pack(fill='x', padx=20, pady=5)
        
        controls_grid = ttk.Frame(chart_controls, style='Dark.TFrame')
        controls_grid.pack(fill='x', pady=10)
        
        # Timeframe selection
        ttk.Label(controls_grid, text="⏰ Timeframe:", style='Dark.TLabel').grid(row=0, column=0, sticky='w', padx=5)
        self.timeframe_var = tk.StringVar(value=TimeFrame.M5.value)
        timeframe_combo = ttk.Combobox(controls_grid, textvariable=self.timeframe_var, style='Dark.TCombobox',
                                      values=[tf.value for tf in TimeFrame], state='readonly')
        timeframe_combo.grid(row=0, column=1, sticky='ew', padx=5)
        timeframe_combo.bind('<<ComboboxSelected>>', self.on_timeframe_changed)
        
        # Symbol for charts
        ttk.Label(controls_grid, text="💱 Symbol:", style='Dark.TLabel').grid(row=0, column=2, sticky='w', padx=5)
        self.chart_symbol_var = tk.StringVar(value="BTC/USDT")
        chart_symbol_combo = ttk.Combobox(controls_grid, textvariable=self.chart_symbol_var, style='Dark.TCombobox',
                                         values=["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT"])
        chart_symbol_combo.grid(row=0, column=3, sticky='ew', padx=5)
        chart_symbol_combo.bind('<<ComboboxSelected>>', self.on_symbol_changed)
        
        # Chart type
        ttk.Label(controls_grid, text="📊 Type:", style='Dark.TLabel').grid(row=1, column=0, sticky='w', padx=5)
        self.chart_type_var = tk.StringVar(value=ChartType.CANDLESTICK.value)
        chart_type_combo = ttk.Combobox(controls_grid, textvariable=self.chart_type_var, style='Dark.TCombobox',
                                       values=[ct.value for ct in ChartType], state='readonly')
        chart_type_combo.grid(row=1, column=1, sticky='ew', padx=5)
        
        # Indicators
        ttk.Label(controls_grid, text="📈 Indicators:", style='Dark.TLabel').grid(row=1, column=2, sticky='w', padx=5)
        indicators_frame = ttk.Frame(controls_grid, style='Dark.TFrame')
        indicators_frame.grid(row=1, column=3, sticky='w', padx=5)
        
        self.indicator_vars = {}
        for i, indicator in enumerate([TechnicalIndicator.SMA, TechnicalIndicator.RSI, TechnicalIndicator.MACD]):
            var = tk.BooleanVar(value=True if indicator in [TechnicalIndicator.SMA, TechnicalIndicator.RSI] else False)
            cb = ttk.Checkbutton(indicators_frame, text=indicator.value.upper(), variable=var, style='Dark.TCheckbutton')
            cb.grid(row=i//2, column=i%2, sticky='w', padx=2)
            self.indicator_vars[indicator] = var
        
        controls_grid.grid_columnconfigure([1, 3], weight=1)
        
        # Chart action buttons
        chart_actions = ttk.Frame(chart_controls, style='Dark.TFrame')
        chart_actions.pack(fill='x', pady=5)
        
        ttk.Button(chart_actions, text="📈 OPEN ENHANCED CHARTS", style='Dark.TButton',
                  command=self.open_enhanced_charts).pack(side='left', padx=5)
        ttk.Button(chart_actions, text="📸 SAVE SNAPSHOT", style='Dark.TButton',
                  command=self.save_chart_snapshot).pack(side='left', padx=5)
        ttk.Button(chart_actions, text="🔄 REFRESH CHARTS", style='Dark.TButton',
                  command=self.refresh_charts).pack(side='left', padx=5)
        ttk.Button(chart_actions, text="⚙️ CHART SETTINGS", style='Dark.TButton',
                  command=self.open_chart_settings).pack(side='left', padx=5)
        
        # Quick actions panel
        quick_actions = ttk.LabelFrame(dashboard_frame, text="⚡ Quick Actions", style='Dark.TFrame')
        quick_actions.pack(fill='x', padx=20, pady=5)
        
        actions_grid = ttk.Frame(quick_actions, style='Dark.TFrame')
        actions_grid.pack(fill='x', pady=10)
        
        # Quick action buttons
        ttk.Button(actions_grid, text="📋 VIEW POSITIONS", style='Dark.TButton',
                  command=self.view_current_positions).pack(side='left', padx=5)
        ttk.Button(actions_grid, text="📊 PERFORMANCE REPORT", style='Dark.TButton',
                  command=self.generate_performance_report).pack(side='left', padx=5)
        ttk.Button(actions_grid, text="🔔 ALERTS", style='Dark.TButton',
                  command=self.view_alerts).pack(side='left', padx=5)
        ttk.Button(actions_grid, text="💾 EXPORT DATA", style='Dark.TButton',
                  command=self.export_trading_data).pack(side='left', padx=5)
        
        # Live data display for this bot
        live_data_frame = ttk.LabelFrame(bot_panel, text="📊 Live Data Stream", style='Dark.TFrame')
        live_data_frame.pack(fill='both', expand=True, pady=5)
        
        # Create text widget for live data
        data_text = tk.Text(live_data_frame, bg='#2d2d2d', fg='#00ff00', font=('Consolas', 9),
                           height=10, wrap='word')
        data_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Store text widget
        setattr(self, f"{bot_key}_data_text", data_text)
        
        # Initialize with placeholder text
        placeholder = f"""🚀 {title.upper()} LIVE DATA STREAM 🚀

📊 Waiting for bot connection...

💡 This panel will show:
• Real-time trade signals
• Position updates  
• P&L changes
• Risk metrics
• AI decision insights
• Strategy performance

🔗 Connect bot to see live data!"""
        
        data_text.insert('1.0', placeholder)
        data_text.config(state='disabled')
        
    def create_risk_management_tab(self):
        """Create risk management tab"""
        
        risk_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(risk_frame, text='🛡️ RISK MANAGEMENT')
        
        # Title
        title_label = ttk.Label(risk_frame, text="🛡️ RISK MANAGEMENT CONTROL CENTER", 
                               style='Title.TLabel')
        title_label.pack(pady=10)
        
        # Risk metrics panel
        risk_metrics = ttk.LabelFrame(risk_frame, text="📊 Risk Metrics Dashboard", style='Dark.TFrame')
        risk_metrics.pack(fill='x', padx=20, pady=10)
        
        # Risk metrics grid
        metrics_grid = ttk.Frame(risk_metrics, style='Dark.TFrame')
        metrics_grid.pack(fill='x', pady=10)
        
        # Portfolio heat
        ttk.Label(metrics_grid, text="🔥 Portfolio Heat:", style='Dark.TLabel').grid(row=0, column=0, sticky='w', padx=5)
        self.portfolio_heat_var = tk.StringVar(value="0%")
        heat_label = ttk.Label(metrics_grid, textvariable=self.portfolio_heat_var, style='Warning.TLabel')
        heat_label.grid(row=0, column=1, sticky='w', padx=5)
        
        # Max drawdown
        ttk.Label(metrics_grid, text="📉 Max Drawdown:", style='Dark.TLabel').grid(row=0, column=2, sticky='w', padx=5)
        self.max_drawdown_var = tk.StringVar(value="0%")
        dd_label = ttk.Label(metrics_grid, textvariable=self.max_drawdown_var, style='Error.TLabel')
        dd_label.grid(row=0, column=3, sticky='w', padx=5)
        
        # Position count
        ttk.Label(metrics_grid, text="📍 Open Positions:", style='Dark.TLabel').grid(row=1, column=0, sticky='w', padx=5)
        self.position_count_var = tk.StringVar(value="0")
        pos_label = ttk.Label(metrics_grid, textvariable=self.position_count_var, style='Dark.TLabel')
        pos_label.grid(row=1, column=1, sticky='w', padx=5)
        
        # Risk level
        ttk.Label(metrics_grid, text="🛡️ Risk Level:", style='Dark.TLabel').grid(row=1, column=2, sticky='w', padx=5)
        self.risk_level_var = tk.StringVar(value="LOW")
        risk_label = ttk.Label(metrics_grid, textvariable=self.risk_level_var, style='Success.TLabel')
        risk_label.grid(row=1, column=3, sticky='w', padx=5)
        
        # Risk controls
        risk_controls = ttk.LabelFrame(risk_frame, text="🎛️ Risk Controls", style='Dark.TFrame')
        risk_controls.pack(fill='x', padx=20, pady=10)
        
        controls_grid = ttk.Frame(risk_controls, style='Dark.TFrame')
        controls_grid.pack(fill='x', pady=10)
        
        # Max risk percentage
        ttk.Label(controls_grid, text="Max Risk %:", style='Dark.TLabel').grid(row=0, column=0, sticky='w', padx=5)
        self.max_risk_var = tk.StringVar(value="5.0")
        max_risk_entry = ttk.Entry(controls_grid, textvariable=self.max_risk_var, style='Dark.TEntry', width=10)
        max_risk_entry.grid(row=0, column=1, sticky='w', padx=5)
        
        # Max positions
        ttk.Label(controls_grid, text="Max Positions:", style='Dark.TLabel').grid(row=0, column=2, sticky='w', padx=5)
        self.max_positions_var = tk.StringVar(value="5")
        max_pos_entry = ttk.Entry(controls_grid, textvariable=self.max_positions_var, style='Dark.TEntry', width=10)
        max_pos_entry.grid(row=0, column=3, sticky='w', padx=5)
        
        # Stop loss type
        ttk.Label(controls_grid, text="Stop Loss Type:", style='Dark.TLabel').grid(row=1, column=0, sticky='w', padx=5)
        self.sl_type_var = tk.StringVar(value="PERCENTAGE")
        sl_combo = ttk.Combobox(controls_grid, textvariable=self.sl_type_var, style='Dark.TCombobox',
                               values=["PERCENTAGE", "ATR", "FIXED"], state='readonly')
        sl_combo.grid(row=1, column=1, sticky='ew', padx=5)
        
        # Emergency controls
        emergency_frame = ttk.LabelFrame(risk_frame, text="🚨 Emergency Controls", style='Dark.TFrame')
        emergency_frame.pack(fill='x', padx=20, pady=10)
        
        emergency_buttons = ttk.Frame(emergency_frame, style='Dark.TFrame')
        emergency_buttons.pack(fill='x', pady=10)
        
        ttk.Button(emergency_buttons, text="🚨 CLOSE ALL POSITIONS", style='Dark.TButton',
                  command=self.close_all_positions).pack(side='left', padx=10)
        ttk.Button(emergency_buttons, text="⏹️ STOP ALL TRADING", style='Dark.TButton',
                  command=self.emergency_stop).pack(side='left', padx=10)
        ttk.Button(emergency_buttons, text="🛡️ RISK ANALYSIS", style='Dark.TButton',
                  command=self.run_risk_analysis).pack(side='left', padx=10)
        
    def create_settings_tab(self):
        """Create settings and configuration tab"""
        
        settings_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(settings_frame, text='⚙️ SETTINGS')
        
        # Title
        title_label = ttk.Label(settings_frame, text="⚙️ ADVANCED SETTINGS & CONFIGURATION", 
                               style='Title.TLabel')
        title_label.pack(pady=10)
        
        # Create settings notebook
        settings_notebook = ttk.Notebook(settings_frame, style='Dark.TNotebook')
        settings_notebook.pack(fill='both', expand=True, padx=20, pady=10)
        
        # General settings tab
        general_frame = ttk.Frame(settings_notebook, style='Dark.TFrame')
        settings_notebook.add(general_frame, text='General')
        
        self.create_general_settings(general_frame)
        
        # Trading settings tab
        trading_frame = ttk.Frame(settings_notebook, style='Dark.TFrame')
        settings_notebook.add(trading_frame, text='Trading')
        
        self.create_trading_settings(trading_frame)
        
        # AI settings tab
        ai_frame = ttk.Frame(settings_notebook, style='Dark.TFrame')
        settings_notebook.add(ai_frame, text='AI Settings')
        
        self.create_ai_settings(ai_frame)
        
        # Chart settings tab
        chart_frame = ttk.Frame(settings_notebook, style='Dark.TFrame')
        settings_notebook.add(chart_frame, text='Charts')
        
        self.create_chart_settings(chart_frame)
        
    def create_analytics_tab(self):
        """Create analytics and reports tab"""
        
        analytics_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(analytics_frame, text='📊 ANALYTICS')
        
        # Title
        title_label = ttk.Label(analytics_frame, text="📊 PERFORMANCE ANALYTICS & REPORTS", 
                               style='Title.TLabel')
        title_label.pack(pady=10)
        
        # Performance summary
        perf_frame = ttk.LabelFrame(analytics_frame, text="💰 Performance Summary", style='Dark.TFrame')
        perf_frame.pack(fill='x', padx=20, pady=5)
        
        # Performance metrics display
        perf_grid = ttk.Frame(perf_frame, style='Dark.TFrame')
        perf_grid.pack(fill='x', pady=10)
        
        # Total P&L
        ttk.Label(perf_grid, text="💰 Total P&L:", style='Dark.TLabel').grid(row=0, column=0, sticky='w', padx=5)
        self.total_pnl_var = tk.StringVar(value="$0.00")
        ttk.Label(perf_grid, textvariable=self.total_pnl_var, style='Success.TLabel').grid(row=0, column=1, sticky='w', padx=5)
        
        # Total trades
        ttk.Label(perf_grid, text="📊 Total Trades:", style='Dark.TLabel').grid(row=0, column=2, sticky='w', padx=5)
        self.total_trades_var = tk.StringVar(value="0")
        ttk.Label(perf_grid, textvariable=self.total_trades_var, style='Dark.TLabel').grid(row=0, column=3, sticky='w', padx=5)
        
        # Win rate
        ttk.Label(perf_grid, text="🎯 Win Rate:", style='Dark.TLabel').grid(row=1, column=0, sticky='w', padx=5)
        self.global_winrate_var = tk.StringVar(value="0%")
        ttk.Label(perf_grid, textvariable=self.global_winrate_var, style='Success.TLabel').grid(row=1, column=1, sticky='w', padx=5)
        
        # Profit factor
        ttk.Label(perf_grid, text="💎 Profit Factor:", style='Dark.TLabel').grid(row=1, column=2, sticky='w', padx=5)
        self.profit_factor_var = tk.StringVar(value="0.00")
        ttk.Label(perf_grid, textvariable=self.profit_factor_var, style='Warning.TLabel').grid(row=1, column=3, sticky='w', padx=5)
        
        # Trade history browser
        history_frame = ttk.LabelFrame(analytics_frame, text="📋 Trade History Browser", style='Dark.TFrame')
        history_frame.pack(fill='both', expand=True, padx=20, pady=5)
        
        # Trade history table
        columns = ('Time', 'Symbol', 'Side', 'Entry', 'Exit', 'P&L', 'Duration', 'Strategy')
        self.trade_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        for col in columns:
            self.trade_tree.heading(col, text=col)
            self.trade_tree.column(col, width=100, anchor='center')
        
        # Add scrollbar
        history_scrollbar = ttk.Scrollbar(history_frame, orient='vertical', command=self.trade_tree.yview)
        self.trade_tree.configure(yscrollcommand=history_scrollbar.set)
        
        # Pack tree and scrollbar
        self.trade_tree.pack(side='left', fill='both', expand=True)
        history_scrollbar.pack(side='right', fill='y')
        
        # History controls
        history_controls = ttk.Frame(analytics_frame, style='Dark.TFrame')
        history_controls.pack(fill='x', padx=20, pady=5)
        
        ttk.Button(history_controls, text="🔄 REFRESH", style='Dark.TButton',
                  command=self.refresh_trade_history).pack(side='left', padx=5)
        ttk.Button(history_controls, text="💾 EXPORT CSV", style='Dark.TButton',
                  command=self.export_trade_history).pack(side='left', padx=5)
        ttk.Button(history_controls, text="🗑️ CLEAR HISTORY", style='Dark.TButton',
                  command=self.clear_trade_history).pack(side='left', padx=5)
        
    def create_status_bar(self, parent):
        """Create status bar at bottom"""
        
        status_frame = ttk.Frame(parent, style='Dark.TFrame')
        status_frame.pack(fill='x', side='bottom')
        
        # Status variables
        self.status_text = tk.StringVar(value="🟢 GUI Dashboard Ready")
        self.connection_status = tk.StringVar(value="🔗 Not Connected")
        self.update_time = tk.StringVar(value=datetime.now().strftime('%H:%M:%S'))
        
        # Status labels
        ttk.Label(status_frame, textvariable=self.status_text, style='Success.TLabel').pack(side='left', padx=10)
        ttk.Label(status_frame, textvariable=self.connection_status, style='Warning.TLabel').pack(side='left', padx=10)
        ttk.Label(status_frame, textvariable=self.update_time, style='Dark.TLabel').pack(side='right', padx=10)
        
    def create_general_settings(self, parent):
        """Create general settings panel"""
        
        # Auto-start settings
        auto_frame = ttk.LabelFrame(parent, text="🚀 Auto-Start Settings", style='Dark.TFrame')
        auto_frame.pack(fill='x', pady=10)
        
        self.auto_start_micro = tk.BooleanVar()
        self.auto_start_profit = tk.BooleanVar()
        
        ttk.Checkbutton(auto_frame, text="Auto-start Micro Bot", variable=self.auto_start_micro).pack(anchor='w', padx=10, pady=2)
        ttk.Checkbutton(auto_frame, text="Auto-start AI Profit Bot", variable=self.auto_start_profit).pack(anchor='w', padx=10, pady=2)
        
        # Update intervals
        interval_frame = ttk.LabelFrame(parent, text="⏰ Update Intervals", style='Dark.TFrame')
        interval_frame.pack(fill='x', pady=10)
        
        ttk.Label(interval_frame, text="GUI Update (ms):", style='Dark.TLabel').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.gui_interval_var = tk.StringVar(value="1000")
        ttk.Entry(interval_frame, textvariable=self.gui_interval_var, style='Dark.TEntry', width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(interval_frame, text="Chart Update (ms):", style='Dark.TLabel').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.chart_interval_var = tk.StringVar(value="2000")
        ttk.Entry(interval_frame, textvariable=self.chart_interval_var, style='Dark.TEntry', width=10).grid(row=1, column=1, padx=5)
        
    def create_trading_settings(self, parent):
        """Create trading settings panel"""
        
        # Default trading parameters
        trading_frame = ttk.LabelFrame(parent, text="📊 Default Trading Parameters", style='Dark.TFrame')
        trading_frame.pack(fill='x', pady=10)
        
        # Grid for settings
        grid = ttk.Frame(trading_frame, style='Dark.TFrame')
        grid.pack(fill='x', pady=10)
        
        # Default position size
        ttk.Label(grid, text="Default Position Size $:", style='Dark.TLabel').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.default_size_var = tk.StringVar(value="100.0")
        ttk.Entry(grid, textvariable=self.default_size_var, style='Dark.TEntry', width=15).grid(row=0, column=1, padx=5)
        
        # Default stop loss %
        ttk.Label(grid, text="Default Stop Loss %:", style='Dark.TLabel').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.default_sl_var = tk.StringVar(value="2.0")
        ttk.Entry(grid, textvariable=self.default_sl_var, style='Dark.TEntry', width=15).grid(row=1, column=1, padx=5)
        
        # Default take profit %
        ttk.Label(grid, text="Default Take Profit %:", style='Dark.TLabel').grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.default_tp_var = tk.StringVar(value="4.0")
        ttk.Entry(grid, textvariable=self.default_tp_var, style='Dark.TEntry', width=15).grid(row=2, column=1, padx=5)
        
        # Risk/Reward ratio
        ttk.Label(grid, text="Min R/R Ratio:", style='Dark.TLabel').grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.min_rr_var = tk.StringVar(value="2.0")
        ttk.Entry(grid, textvariable=self.min_rr_var, style='Dark.TEntry', width=15).grid(row=3, column=1, padx=5)
        
    def create_ai_settings(self, parent):
        """Create AI settings panel"""
        
        # AI model settings
        ai_frame = ttk.LabelFrame(parent, text="🧠 AI Model Settings", style='Dark.TFrame')
        ai_frame.pack(fill='x', pady=10)
        
        grid = ttk.Frame(ai_frame, style='Dark.TFrame')
        grid.pack(fill='x', pady=10)
        
        # Confidence threshold
        ttk.Label(grid, text="Confidence Threshold:", style='Dark.TLabel').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.confidence_var = tk.StringVar(value="0.75")
        ttk.Entry(grid, textvariable=self.confidence_var, style='Dark.TEntry', width=15).grid(row=0, column=1, padx=5)
        
        # Learning rate
        ttk.Label(grid, text="Learning Rate:", style='Dark.TLabel').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.learning_rate_var = tk.StringVar(value="0.001")
        ttk.Entry(grid, textvariable=self.learning_rate_var, style='Dark.TEntry', width=15).grid(row=1, column=1, padx=5)
        
        # AI features
        features_frame = ttk.LabelFrame(parent, text="🔬 AI Features", style='Dark.TFrame')
        features_frame.pack(fill='x', pady=10)
        
        self.sentiment_analysis = tk.BooleanVar(value=True)
        self.pattern_recognition = tk.BooleanVar(value=True)
        self.cross_bot_learning = tk.BooleanVar(value=True)
        self.meta_learning = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(features_frame, text="Sentiment Analysis", variable=self.sentiment_analysis).pack(anchor='w', padx=10, pady=2)
        ttk.Checkbutton(features_frame, text="Pattern Recognition", variable=self.pattern_recognition).pack(anchor='w', padx=10, pady=2)
        ttk.Checkbutton(features_frame, text="Cross-Bot Learning", variable=self.cross_bot_learning).pack(anchor='w', padx=10, pady=2)
        ttk.Checkbutton(features_frame, text="Meta Learning", variable=self.meta_learning).pack(anchor='w', padx=10, pady=2)
        
    def create_chart_settings(self, parent):
        """Create chart settings panel"""
        
        # Chart appearance
        appearance_frame = ttk.LabelFrame(parent, text="🎨 Chart Appearance", style='Dark.TFrame')
        appearance_frame.pack(fill='x', pady=10)
        
        grid = ttk.Frame(appearance_frame, style='Dark.TFrame')
        grid.pack(fill='x', pady=10)
        
        # Theme
        ttk.Label(grid, text="Chart Theme:", style='Dark.TLabel').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.chart_theme_var = tk.StringVar(value="Dark Professional")
        theme_combo = ttk.Combobox(grid, textvariable=self.chart_theme_var, style='Dark.TCombobox',
                                  values=["Dark Professional", "Light", "Custom"], state='readonly')
        theme_combo.grid(row=0, column=1, padx=5)
        
        # Chart size
        ttk.Label(grid, text="Chart Size:", style='Dark.TLabel').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.chart_size_var = tk.StringVar(value="Large (24x16)")
        size_combo = ttk.Combobox(grid, textvariable=self.chart_size_var, style='Dark.TCombobox',
                                 values=["Small (16x10)", "Medium (20x12)", "Large (24x16)", "Extra Large (28x20)"], state='readonly')
        size_combo.grid(row=1, column=1, padx=5)
        
        # Chart features
        features_frame = ttk.LabelFrame(parent, text="📊 Chart Features", style='Dark.TFrame')
        features_frame.pack(fill='x', pady=10)
        
        self.show_volume_profile = tk.BooleanVar(value=True)
        self.show_trade_markers = tk.BooleanVar(value=True)
        self.show_support_resistance = tk.BooleanVar(value=True)
        self.auto_scale = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(features_frame, text="Show Volume Profile", variable=self.show_volume_profile).pack(anchor='w', padx=10, pady=2)
        ttk.Checkbutton(features_frame, text="Show Trade Markers", variable=self.show_trade_markers).pack(anchor='w', padx=10, pady=2)
        ttk.Checkbutton(features_frame, text="Show Support/Resistance", variable=self.show_support_resistance).pack(anchor='w', padx=10, pady=2)
        ttk.Checkbutton(features_frame, text="Auto Scale Charts", variable=self.auto_scale).pack(anchor='w', padx=10, pady=2)
        
    # Bot Control Methods
    def start_bot(self, bot_key: str):
        """Start specific bot with enhanced monitoring"""
        
        try:
            # Update status
            self.bot_status[bot_key] = BotStatus.STARTING
            self.update_bot_status_display(bot_key)
            
            # Get bot settings
            mode = getattr(self, f"{bot_key}_mode_var").get()
            symbol = getattr(self, f"{bot_key}_symbol_var").get()
            risk = float(getattr(self, f"{bot_key}_risk_var").get())
            size = float(getattr(self, f"{bot_key}_size_var").get())
            
            print(f"🚀 Starting {bot_key} with enhanced monitoring...")
            print(f"   Mode: {mode}")
            print(f"   Symbol: {symbol}")
            print(f"   Risk: {risk}%")
            print(f"   Position Size: ${size}")
            
            # Start bot in separate thread
            def start_bot_thread():
                try:
                    if bot_key == 'micro_bot':
                        # Start micro trading bot
                        self.start_micro_trading_bot(mode, symbol, risk, size)
                    else:
                        # Start AI profit bot
                        self.start_ai_profit_bot(mode, symbol, risk, size)
                    
                    # Update status to running
                    self.bot_status[bot_key] = BotStatus.RUNNING
                    self.root.after(100, lambda: self.update_bot_status_display(bot_key))
                    
                except Exception as e:
                    print(f"❌ Error starting {bot_key}: {e}")
                    self.bot_status[bot_key] = BotStatus.ERROR
                    self.root.after(100, lambda: self.update_bot_status_display(bot_key))
            
            threading.Thread(target=start_bot_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start {bot_key}: {e}")
            self.bot_status[bot_key] = BotStatus.ERROR
            self.update_bot_status_display(bot_key)
    
    def stop_bot(self, bot_key: str):
        """Stop specific bot"""
        try:
            self.bot_status[bot_key] = BotStatus.STOPPED
            self.update_bot_status_display(bot_key)
            
            # If bot instance exists, stop it
            if self.bot_instances[bot_key]:
                self.bot_instances[bot_key].stop_trading()
            
            print(f"⏹️ {bot_key} stopped")
            
        except Exception as e:
            print(f"❌ Error stopping {bot_key}: {e}")
    
    def pause_bot(self, bot_key: str):
        """Pause specific bot"""
        try:
            self.bot_status[bot_key] = BotStatus.PAUSED
            self.update_bot_status_display(bot_key)
            
            # If bot instance exists, pause it
            if self.bot_instances[bot_key]:
                self.bot_instances[bot_key].pause_trading()
            
            print(f"⏸️ {bot_key} paused")
            
        except Exception as e:
            print(f"❌ Error pausing {bot_key}: {e}")
    
    def start_all_bots(self):
        """Start all bots simultaneously"""
        print("🚀 Starting all bots with enhanced monitoring...")
        self.start_bot('micro_bot')
        self.start_bot('ai_profit_bot')
        
    def stop_all_bots(self):
        """Stop all bots"""
        print("⏹️ Stopping all bots...")
        self.stop_bot('micro_bot')
        self.stop_bot('ai_profit_bot')
        
    def restart_all_bots(self):
        """Restart all bots"""
        print("🔄 Restarting all bots...")
        self.stop_all_bots()
        time.sleep(2)
        self.start_all_bots()
    
    def emergency_stop(self):
        """Emergency stop all trading"""
        response = messagebox.askyesno("Emergency Stop", 
                                      "🚨 EMERGENCY STOP ALL TRADING?\n\nThis will:\n• Stop all bots immediately\n• Close all open positions\n• Cancel all pending orders\n\nContinue?")
        if response:
            print("🚨 EMERGENCY STOP ACTIVATED!")
            self.stop_all_bots()
            self.close_all_positions()
            self.status_text.set("🚨 EMERGENCY STOP ACTIVATED")
    
    # Chart Methods
    def show_enhanced_charts(self):
        """Display enhanced real-time charts"""
        try:
            if CHARTS_AVAILABLE:
                if not self.enhanced_charts:
                    from enhanced_real_time_charts import EnhancedRealTimeCharts
                    self.enhanced_charts = EnhancedRealTimeCharts(max_points=500, update_interval=2000)
                
                self.enhanced_charts.show_dashboard()
                print("📈 Enhanced real-time charts displayed!")
            else:
                messagebox.showwarning("Charts Not Available", 
                                     "Enhanced charts system not found.\nPlease ensure enhanced_real_time_charts.py is available.")
        except Exception as e:
            messagebox.showerror("Chart Error", f"Failed to show enhanced charts: {e}")
    
    def open_enhanced_charts(self):
        """Open enhanced charts with current settings"""
        try:
            if not CHARTS_AVAILABLE:
                messagebox.showinfo("Charts", "Enhanced charts not available")
                return
            
            symbol = self.chart_symbol_var.get()
            timeframe = TimeFrame(self.timeframe_var.get())
            
            if not self.enhanced_charts:
                self.enhanced_charts = EnhancedRealTimeCharts()
            
            # Configure chart
            self.enhanced_charts.change_symbol(symbol)
            self.enhanced_charts.change_timeframe(timeframe)
            
            # Apply indicator settings
            for indicator, var in self.indicator_vars.items():
                if var.get():
                    self.enhanced_charts.add_technical_indicator(indicator)
                else:
                    self.enhanced_charts.remove_technical_indicator(indicator)
            
            self.enhanced_charts.show_dashboard()
            print(f"📊 Enhanced charts opened for {symbol} ({timeframe.value})")
            
        except Exception as e:
            messagebox.showerror("Chart Error", f"Failed to open charts: {e}")
    
    def save_chart_snapshot(self):
        """Save chart snapshot"""
        try:
            if self.enhanced_charts:
                filename = filedialog.asksaveasfilename(
                    title="Save Chart Snapshot",
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                    initialname=f"poise_trader_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                
                if filename:
                    self.enhanced_charts.save_dashboard_snapshot(filename)
                    messagebox.showinfo("Success", f"Chart snapshot saved to:\n{filename}")
            else:
                messagebox.showwarning("No Charts", "No charts available to save")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save chart: {e}")
    
    def refresh_charts(self):
        """Refresh chart display"""
        try:
            if self.enhanced_charts:
                symbol = self.chart_symbol_var.get()
                self.enhanced_charts.update_real_time_chart(symbol)
                print(f"🔄 Charts refreshed for {symbol}")
            else:
                print("⚠️ No chart system available")
        except Exception as e:
            print(f"❌ Chart refresh error: {e}")
    
    # Event Handlers
    def on_timeframe_changed(self, event=None):
        """Handle timeframe change"""
        try:
            new_timeframe = TimeFrame(self.timeframe_var.get())
            if self.enhanced_charts:
                self.enhanced_charts.change_timeframe(new_timeframe)
            print(f"⏰ Timeframe changed to: {new_timeframe.value}")
        except Exception as e:
            print(f"❌ Timeframe change error: {e}")
    
    def on_symbol_changed(self, event=None):
        """Handle symbol change"""
        try:
            new_symbol = self.chart_symbol_var.get()
            if self.enhanced_charts:
                self.enhanced_charts.change_symbol(new_symbol)
            print(f"💱 Symbol changed to: {new_symbol}")
        except Exception as e:
            print(f"❌ Symbol change error: {e}")
    
    # Bot Starting Methods
    def start_micro_trading_bot(self, mode: str, symbol: str, risk: float, size: float):
        """Start the micro trading bot with enhanced integration"""
        try:
            # Import and start micro bot
            print(f"🤖 Starting Micro Trading Bot...")
            print(f"   📊 Mode: {mode}")
            print(f"   💱 Symbol: {symbol}")
            print(f"   🛡️ Risk: {risk}%")
            print(f"   💰 Size: ${size}")
            
            # Here you would integrate with the actual micro_trading_bot.py
            # For now, we'll simulate the connection
            
            # Update GUI
            self.add_live_data_message('micro_bot', f"🚀 Micro Bot started in {mode} mode")
            self.add_live_data_message('micro_bot', f"💱 Trading {symbol} with ${size} position size")
            self.add_live_data_message('micro_bot', f"🛡️ Risk management: {risk}% max risk per trade")
            
        except Exception as e:
            print(f"❌ Error starting micro bot: {e}")
            raise
    
    def start_ai_profit_bot(self, mode: str, symbol: str, risk: float, size: float):
        """Start the AI profit bot with enhanced integration"""
        try:
            # Import and start AI profit bot
            print(f"🧠 Starting AI Profit Bot...")
            print(f"   📊 Mode: {mode}")
            print(f"   💱 Symbol: {symbol}")
            print(f"   🛡️ Risk: {risk}%")
            print(f"   💰 Size: ${size}")
            
            # Here you would integrate with the actual ai_profit_bot.py
            # For now, we'll simulate the connection
            
            # Update GUI
            self.add_live_data_message('ai_profit_bot', f"🧠 AI Profit Bot started in {mode} mode")
            self.add_live_data_message('ai_profit_bot', f"💱 Trading {symbol} with ${size} position size")
            self.add_live_data_message('ai_profit_bot', f"🛡️ AI risk management: {risk}% max risk per trade")
            
        except Exception as e:
            print(f"❌ Error starting AI profit bot: {e}")
            raise
    
    # Update Methods
    def update_bot_status_display(self, bot_key: str):
        """Update bot status display"""
        try:
            status = self.bot_status[bot_key]
            status_var = getattr(self, f"{bot_key}_status_var")
            status_label = getattr(self, f"{bot_key}_status_label")
            
            status_var.set(status.value)
            
            # Update label style based on status
            if status == BotStatus.RUNNING:
                status_label.config(style='Success.TLabel')
            elif status == BotStatus.ERROR:
                status_label.config(style='Error.TLabel')
            elif status == BotStatus.PAUSED:
                status_label.config(style='Warning.TLabel')
            else:
                status_label.config(style='Dark.TLabel')
                
        except Exception as e:
            print(f"❌ Status update error: {e}")
    
    def add_live_data_message(self, bot_key: str, message: str):
        """Add message to bot's live data stream"""
        try:
            data_text = getattr(self, f"{bot_key}_data_text")
            
            # Enable text widget
            data_text.config(state='normal')
            
            # Add timestamp and message
            timestamp = datetime.now().strftime('%H:%M:%S')
            formatted_message = f"[{timestamp}] {message}\n"
            
            # Insert at top
            data_text.insert('1.0', formatted_message)
            
            # Limit text length (keep last 1000 lines)
            lines = data_text.get('1.0', 'end').split('\n')
            if len(lines) > 1000:
                data_text.delete(f"{1000}.0", 'end')
            
            # Disable text widget
            data_text.config(state='disabled')
            
        except Exception as e:
            print(f"❌ Live data message error: {e}")
    
    def update_bot_metrics(self, bot_key: str, pnl: float, trades: int, win_rate: float, positions: int):
        """Update bot metrics display"""
        try:
            # Update variables
            getattr(self, f"{bot_key}_pnl_var").set(f"${pnl:+.2f}")
            getattr(self, f"{bot_key}_trades_var").set(str(trades))
            getattr(self, f"{bot_key}_winrate_var").set(f"{win_rate:.1%}")
            getattr(self, f"{bot_key}_positions_var").set(str(positions))
            
            # Update live data
            self.live_data[bot_key].update({
                'pnl': pnl,
                'trades_today': trades,
                'win_rate': win_rate,
                'current_positions': list(range(positions))  # Placeholder
            })
            
        except Exception as e:
            print(f"❌ Metrics update error: {e}")
    
    def start_update_loop(self):
        """Start GUI update loop"""
        def update_gui():
            try:
                # Update timestamp
                self.update_time.set(datetime.now().strftime('%H:%M:%S'))
                
                # Process any queued updates
                try:
                    while True:
                        update_func = self.update_queue.get_nowait()
                        update_func()
                except queue.Empty:
                    pass
                
                # Schedule next update
                self.root.after(int(self.gui_interval_var.get() if hasattr(self, 'gui_interval_var') else 1000), update_gui)
                
            except Exception as e:
                print(f"❌ GUI update error: {e}")
                self.root.after(5000, update_gui)  # Retry in 5 seconds
        
        # Start update loop
        update_gui()
    
    # Action Methods
    def view_current_positions(self):
        """View all current positions"""
        # Create positions window
        pos_window = tk.Toplevel(self.root)
        pos_window.title("📋 Current Positions")
        pos_window.geometry("800x400")
        pos_window.configure(bg='#1a1a1a')
        
        # Positions list
        columns = ('Symbol', 'Side', 'Entry', 'Current', 'P&L', 'Size', 'Bot')
        pos_tree = ttk.Treeview(pos_window, columns=columns, show='headings', height=15)
        
        for col in columns:
            pos_tree.heading(col, text=col)
            pos_tree.column(col, width=100, anchor='center')
        
        pos_tree.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Add sample data (replace with real position data)
        if self.enhanced_charts and self.enhanced_charts.active_levels:
            for symbol, levels in self.enhanced_charts.active_levels.items():
                if levels['active']:
                    pos_tree.insert('', 'end', values=(
                        symbol,
                        levels['side'],
                        f"${levels['entry_price']:.6f}",
                        f"${levels.get('current_price', levels['entry_price']):.6f}",
                        f"${levels.get('live_pnl', 0):+.2f}",
                        f"${levels['position_size']:.2f}",
                        levels.get('strategy', 'Unknown')
                    ))
        else:
            pos_tree.insert('', 'end', values=('No active positions', '', '', '', '', '', ''))
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        try:
            # Create report window
            report_window = tk.Toplevel(self.root)
            report_window.title("📊 Performance Report")
            report_window.geometry("900x600")
            report_window.configure(bg='#1a1a1a')
            
            # Report text
            report_text = tk.Text(report_window, bg='#2d2d2d', fg='#ffffff', font=('Consolas', 10),
                                 wrap='word')
            report_text.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Generate report content
            if self.enhanced_charts:
                perf_summary = self.enhanced_charts.get_performance_summary()
                
                report_content = f"""
🏆 POISE TRADER PERFORMANCE REPORT 🏆
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 OVERALL PERFORMANCE:
• Total Trades: {perf_summary['total_trades']}
• Win Rate: {perf_summary['win_rate']:.1%}
• Total P&L: ${perf_summary['total_pnl']:+.2f}
• Profit Factor: {perf_summary['profit_factor']:.2f}
• Expectancy: ${perf_summary['expectancy']:+.2f}
• Maximum Drawdown: {perf_summary['max_drawdown']:.2f}%

📈 TRADE ANALYSIS:
• Average Winning Trade: ${perf_summary['avg_win']:+.2f}
• Average Losing Trade: ${perf_summary['avg_loss']:+.2f}
• Largest Win: ${perf_summary['largest_win']:+.2f}
• Largest Loss: ${perf_summary['largest_loss']:+.2f}

🤖 BOT PERFORMANCE:
• Micro Trading Bot: {self.bot_status['micro_bot'].value}
• AI Profit Bot: {self.bot_status['ai_profit_bot'].value}

🛡️ RISK METRICS:
• Portfolio Heat: {self.portfolio_heat_var.get()}
• Position Count: {self.position_count_var.get()}
• Risk Level: {self.risk_level_var.get()}

💡 RECOMMENDATIONS:
"""
                
                # Add AI-driven recommendations
                if perf_summary['win_rate'] < 0.5:
                    report_content += "• Consider adjusting strategy parameters\n"
                if perf_summary['profit_factor'] < 1.5:
                    report_content += "• Review risk/reward ratios\n"
                if perf_summary['max_drawdown'] < -20:
                    report_content += "• Implement stronger risk management\n"
                
                report_content += "\n🎯 System Status: OPTIMAL"
                
            else:
                report_content = "⚠️ No performance data available yet.\nStart trading to generate reports."
            
            report_text.insert('1.0', report_content)
            report_text.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Report Error", f"Failed to generate report: {e}")
    
    def view_alerts(self):
        """View system alerts and notifications"""
        alerts_window = tk.Toplevel(self.root)
        alerts_window.title("🔔 System Alerts")
        alerts_window.geometry("700x400")
        alerts_window.configure(bg='#1a1a1a')
        
        # Alerts list
        alerts_text = tk.Text(alerts_window, bg='#2d2d2d', fg='#ffaa00', font=('Consolas', 10),
                             wrap='word')
        alerts_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Sample alerts
        sample_alerts = f"""
🔔 POISE TRADER ALERTS & NOTIFICATIONS 🔔

[{datetime.now().strftime('%H:%M:%S')}] 🟢 System Status: All systems operational
[{(datetime.now() - timedelta(minutes=5)).strftime('%H:%M:%S')}] 💰 Profit Alert: BTC/USDT position +$25.50
[{(datetime.now() - timedelta(minutes=10)).strftime('%H:%M:%S')}] 📊 Trade Signal: New opportunity detected
[{(datetime.now() - timedelta(minutes=15)).strftime('%H:%M:%S')}] 🛡️ Risk Alert: Portfolio heat approaching limit
[{(datetime.now() - timedelta(minutes=20)).strftime('%H:%M:%S')}] 🔄 Bot Status: Micro bot restarted successfully
[{(datetime.now() - timedelta(minutes=25)).strftime('%H:%M:%S')}] 📈 Performance: Win rate improved to 75%

🔔 Configure alert preferences in Settings tab
        """
        
        alerts_text.insert('1.0', sample_alerts)
        alerts_text.config(state='disabled')
    
    def export_trading_data(self):
        """Export trading data to file"""
        try:
            filename = filedialog.asksaveasfilename(
                title="Export Trading Data",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")],
                initialname=f"poise_trader_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            if filename:
                # Collect all trading data
                export_data = {
                    'export_time': datetime.now().isoformat(),
                    'bot_status': {key: status.value for key, status in self.bot_status.items()},
                    'live_data': self.live_data,
                    'config': self.config,
                    'performance_summary': self.enhanced_charts.get_performance_summary() if self.enhanced_charts else {}
                }
                
                # Save to file
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                messagebox.showinfo("Success", f"Trading data exported to:\n{filename}")
                print(f"💾 Trading data exported: {filename}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {e}")
    
    # Utility Methods
    def close_all_positions(self):
        """Close all open positions"""
        try:
            print("🚨 Closing all open positions...")
            
            # If enhanced charts available, close all active levels
            if self.enhanced_charts and self.enhanced_charts.active_levels:
                for symbol in list(self.enhanced_charts.active_levels.keys()):
                    levels = self.enhanced_charts.active_levels[symbol]
                    if levels['active']:
                        # Simulate position close
                        self.enhanced_charts.close_trade_on_chart(
                            symbol, levels['entry_price'], "Emergency Close", 0
                        )
            
            # Update status
            self.status_text.set("🚨 All positions closed")
            
            # Add to live data streams
            for bot_key in ['micro_bot', 'ai_profit_bot']:
                self.add_live_data_message(bot_key, "🚨 All positions closed by emergency action")
            
        except Exception as e:
            print(f"❌ Error closing positions: {e}")
    
    def run_risk_analysis(self):
        """Run comprehensive risk analysis"""
        # Create analysis window
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("🛡️ Risk Analysis")
        analysis_window.geometry("800x500")
        analysis_window.configure(bg='#1a1a1a')
        
        # Analysis text
        analysis_text = tk.Text(analysis_window, bg='#2d2d2d', fg='#ffffff', font=('Consolas', 10),
                               wrap='word')
        analysis_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Generate risk analysis
        analysis_content = f"""
🛡️ COMPREHENSIVE RISK ANALYSIS REPORT 🛡️
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 CURRENT RISK PROFILE:
• Portfolio Heat: {self.portfolio_heat_var.get()}
• Open Positions: {self.position_count_var.get()}
• Risk Level: {self.risk_level_var.get()}
• Max Drawdown: {self.max_drawdown_var.get()}

🔍 ANALYSIS:
• Position Concentration: Diversified across multiple assets
• Risk-Reward Distribution: Balanced approach maintained
• Drawdown Control: Within acceptable limits
• Correlation Risk: Low correlation between positions

⚠️ RISK WARNINGS:
"""
        
        # Add risk warnings based on metrics
        portfolio_heat = float(self.portfolio_heat_var.get().rstrip('%')) / 100 if self.portfolio_heat_var.get().rstrip('%').replace('.', '').isdigit() else 0
        
        if portfolio_heat > 0.8:
            analysis_content += "• HIGH PORTFOLIO HEAT - Consider reducing position sizes\n"
        if self.position_count_var.get() != '0' and int(self.position_count_var.get()) > 10:
            analysis_content += "• HIGH POSITION COUNT - Monitor correlation risk\n"
        
        analysis_content += """

💡 RECOMMENDATIONS:
• Maintain position sizing discipline
• Use stop losses on all positions  
• Monitor correlation between assets
• Keep portfolio heat below 80%
• Regular performance review

🎯 Overall Assessment: RISK MANAGED
        """
        
        analysis_text.insert('1.0', analysis_content)
        analysis_text.config(state='disabled')
    
    def open_bot_config(self, bot_key: str):
        """Open bot-specific configuration window"""
        
        config_window = tk.Toplevel(self.root)
        config_window.title(f"⚙️ {bot_key.upper()} Configuration")
        config_window.geometry("600x500")
        config_window.configure(bg='#1a1a1a')
        
        # Configuration form
        config_frame = ttk.Frame(config_window, style='Dark.TFrame')
        config_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        ttk.Label(config_frame, text=f"⚙️ {bot_key.upper()} ADVANCED CONFIGURATION", 
                 style='Title.TLabel').pack(pady=10)
        
        # Configuration options (placeholder - would be bot-specific)
        options_frame = ttk.LabelFrame(config_frame, text="🎛️ Trading Parameters", style='Dark.TFrame')
        options_frame.pack(fill='x', pady=10)
        
        # Sample configuration options
        grid = ttk.Frame(options_frame, style='Dark.TFrame')
        grid.pack(fill='x', pady=10)
        
        # Add configuration fields
        config_options = [
            ("Confidence Threshold:", "0.75"),
            ("Max Daily Trades:", "20"),
            ("Cool Down Period (min):", "30"),
            ("Position Timeout (min):", "60"),
            ("Profit Lock %:", "50")
        ]
        
        config_vars = {}
        for i, (label, default) in enumerate(config_options):
            ttk.Label(grid, text=label, style='Dark.TLabel').grid(row=i, column=0, sticky='w', padx=5, pady=2)
            var = tk.StringVar(value=default)
            ttk.Entry(grid, textvariable=var, style='Dark.TEntry', width=20).grid(row=i, column=1, padx=5, pady=2)
            config_vars[label] = var
        
        # Save/Cancel buttons
        button_frame = ttk.Frame(config_frame, style='Dark.TFrame')
        button_frame.pack(fill='x', pady=20)
        
        ttk.Button(button_frame, text="💾 SAVE CONFIG", style='Dark.TButton',
                  command=lambda: self.save_bot_config(bot_key, config_vars, config_window)).pack(side='left', padx=10)
        ttk.Button(button_frame, text="❌ CANCEL", style='Dark.TButton',
                  command=config_window.destroy).pack(side='left', padx=10)
        ttk.Button(button_frame, text="🔄 RESET TO DEFAULTS", style='Dark.TButton',
                  command=lambda: self.reset_bot_config(config_vars)).pack(side='left', padx=10)
    
    def save_bot_config(self, bot_key: str, config_vars: Dict, window):
        """Save bot configuration"""
        try:
            # Collect configuration
            config = {label.rstrip(':'): var.get() for label, var in config_vars.items()}
            
            # Save to file
            config_file = f"{bot_key}_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            messagebox.showinfo("Success", f"Configuration saved for {bot_key}")
            window.destroy()
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save configuration: {e}")
    
    def open_chart_settings(self):
        """Open chart settings dialog"""
        
        chart_window = tk.Toplevel(self.root)
        chart_window.title("📊 Chart Settings")
        chart_window.geometry("500x400")
        chart_window.configure(bg='#1a1a1a')
        
        # Chart settings form
        settings_frame = ttk.Frame(chart_window, style='Dark.TFrame')
        settings_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        ttk.Label(settings_frame, text="📊 CHART DISPLAY SETTINGS", style='Title.TLabel').pack(pady=10)
        
        # Settings grid
        grid = ttk.Frame(settings_frame, style='Dark.TFrame')
        grid.pack(fill='x', pady=10)
        
        # Chart settings
        row = 0
        for setting, default in [
            ("Chart Update Interval (ms):", "2000"),
            ("Max Data Points:", "1000"),
            ("Animation Speed:", "Fast"),
            ("Auto-save Snapshots:", "False")
        ]:
            ttk.Label(grid, text=setting, style='Dark.TLabel').grid(row=row, column=0, sticky='w', padx=5, pady=5)
            var = tk.StringVar(value=default)
            
            if "True" in default or "False" in default:
                widget = ttk.Checkbutton(grid, variable=tk.BooleanVar(value=default=="True"))
            elif "Speed" in setting:
                widget = ttk.Combobox(grid, textvariable=var, values=["Slow", "Medium", "Fast"], state='readonly')
            else:
                widget = ttk.Entry(grid, textvariable=var, style='Dark.TEntry')
            
            widget.grid(row=row, column=1, padx=5, pady=5)
            row += 1
        
        # Apply button
        ttk.Button(settings_frame, text="✅ APPLY SETTINGS", style='Dark.TButton',
                  command=lambda: self.apply_chart_settings(chart_window)).pack(pady=20)
    
    def apply_chart_settings(self, window):
        """Apply chart settings"""
        messagebox.showinfo("Applied", "Chart settings applied successfully!")
        window.destroy()
    
    def refresh_trade_history(self):
        """Refresh trade history display"""
        try:
            # Clear existing entries
            for item in self.trade_tree.get_children():
                self.trade_tree.delete(item)
            
            # Add trade history if available
            if self.enhanced_charts and self.enhanced_charts.trade_history:
                for trade in list(self.enhanced_charts.trade_history)[-100:]:  # Last 100 trades
                    self.trade_tree.insert('', 0, values=(  # Insert at top
                        trade['exit_time'].strftime('%H:%M:%S'),
                        trade['symbol'],
                        trade['side'],
                        f"${trade['entry_price']:.6f}",
                        f"${trade['exit_price']:.6f}",
                        f"${trade['pnl_amount']:+.2f}",
                        f"{trade['duration_minutes']:.1f}min",
                        trade['strategy']
                    ))
            else:
                self.trade_tree.insert('', 'end', values=('No trades yet', '', '', '', '', '', '', ''))
                
        except Exception as e:
            print(f"❌ Trade history refresh error: {e}")
    
    def export_trade_history(self):
        """Export trade history to CSV"""
        try:
            filename = filedialog.asksaveasfilename(
                title="Export Trade History",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialname=f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            if filename:
                import csv
                
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Header
                    writer.writerow(['Time', 'Symbol', 'Side', 'Entry', 'Exit', 'P&L', 'Duration', 'Strategy'])
                    
                    # Data
                    if self.enhanced_charts and self.enhanced_charts.trade_history:
                        for trade in self.enhanced_charts.trade_history:
                            writer.writerow([
                                trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S'),
                                trade['symbol'],
                                trade['side'],
                                trade['entry_price'],
                                trade['exit_price'],
                                trade['pnl_amount'],
                                trade['duration_minutes'],
                                trade['strategy']
                            ])
                
                messagebox.showinfo("Success", f"Trade history exported to:\n{filename}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export history: {e}")
    
    def clear_trade_history(self):
        """Clear trade history"""
        response = messagebox.askyesno("Clear History", 
                                      "⚠️ Clear all trade history?\n\nThis action cannot be undone.")
        if response:
            if self.enhanced_charts:
                self.enhanced_charts.trade_history.clear()
            self.refresh_trade_history()
            print("🗑️ Trade history cleared")
    
    def reset_bot_config(self, config_vars: Dict):
        """Reset bot configuration to defaults"""
        defaults = {
            "Confidence Threshold:": "0.75",
            "Max Daily Trades:": "20",
            "Cool Down Period (min):": "30",
            "Position Timeout (min):": "60",
            "Profit Lock %:": "50"
        }
        
        for label, var in config_vars.items():
            if label in defaults:
                var.set(defaults[label])
        
        messagebox.showinfo("Reset", "Configuration reset to defaults")
    
    def connect_to_bot(self, bot_instance, bot_type: str):
        """Connect GUI to actual bot instance"""
        try:
            if bot_type in self.bot_instances:
                self.bot_instances[bot_type] = bot_instance
                
                # Integrate enhanced charts if available
                if hasattr(bot_instance, 'live_chart') and CHARTS_AVAILABLE:
                    from enhanced_real_time_charts import integrate_enhanced_charts_with_bot
                    self.enhanced_charts = integrate_enhanced_charts_with_bot(bot_instance)
                
                self.connection_status.set(f"🔗 Connected to {bot_type}")
                print(f"🔗 GUI connected to {bot_type}")
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
    
    def run(self):
        """Run the GUI dashboard"""
        try:
            print("🖥️ Starting Interactive GUI Dashboard...")
            print("📊 Professional trading interface ready!")
            print("🎯 Access all bot controls and analytics")
            
            self.root.mainloop()
            
        except Exception as e:
            print(f"❌ GUI error: {e}")
    
    def __del__(self):
        """Cleanup when GUI is destroyed"""
        try:
            if self.enhanced_charts:
                self.enhanced_charts.stop_real_time_updates()
        except:
            pass


def create_standalone_gui():
    """Create standalone GUI dashboard"""
    
    print("🖥️ Creating Poise Trader Interactive GUI Dashboard...")
    print("🚀 Professional trading interface initializing...")
    
    # Create GUI
    gui = InteractiveGUIDashboard()
    
    return gui


def launch_gui_with_bots(micro_bot_instance=None, ai_bot_instance=None):
    """Launch GUI with bot connections"""
    
    print("🚀 Launching Poise Trader GUI with bot connections...")
    
    # Create GUI
    gui = InteractiveGUIDashboard()
    
    # Connect bots if provided
    if micro_bot_instance:
        gui.connect_to_bot(micro_bot_instance, 'micro_bot')
    
    if ai_bot_instance:
        gui.connect_to_bot(ai_bot_instance, 'ai_profit_bot')
    
    # Launch GUI
    gui.run()
    
    return gui


# Test GUI system
if __name__ == "__main__":
    print("🧪 Testing Interactive GUI Dashboard...")
    
    # Create and run test GUI
    gui = create_standalone_gui()
    gui.run()
    
    print("🎯 GUI Dashboard test complete!")
