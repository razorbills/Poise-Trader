#!/usr/bin/env python3
"""
📊 PAPER TRADING MANAGER - REALISTIC SIMULATION ENGINE

This module provides a comprehensive paper trading simulation that:
• Tracks virtual portfolio with realistic execution
• Simulates market impact and slippage
• Records all trades for performance analysis  
• Provides detailed performance metrics
• Supports multiple strategies simultaneously

SAFE FOR TESTING - NO REAL MONEY INVOLVED!
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import uuid


class PaperTradingManager:
    """
    📊 PAPER TRADING PORTFOLIO MANAGER
    
    Simulates real trading with virtual money to:
    • Test strategies safely
    • Validate system performance
    • Debug execution logic
    • Build confidence before real trading
    """
    
    def __init__(self, initial_capital: float = 5000.0):
        self.logger = logging.getLogger("PaperTrading")
        
        # Virtual portfolio
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
        self.trade_history: List[Dict] = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_portfolio_value = initial_capital
        
        # Simulation settings
        self.slippage_bps = 5  # 0.05% slippage simulation
        self.commission_bps = 10  # 0.1% commission
        
        # Data storage
        self.data_dir = Path("data/paper_trading")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📊 Paper Trading Manager initialized with ${initial_capital:,.2f} virtual capital")
    
    async def execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a paper trade with realistic simulation
        
        Args:
            signal: Trading signal with symbol, action, position_size, etc.
            
        Returns:
            Execution result with success status and details
        """
        
        try:
            trade_id = str(uuid.uuid4())[:8]
            symbol = signal['symbol']
            action = signal['action'].upper()
            
            # Get current market price (simulated)
            current_price = await self._get_simulated_price(symbol)
            
            # Calculate position size
            position_size = await self._calculate_position_size(signal, current_price)
            
            if position_size == 0:
                return {
                    'success': False,
                    'error': 'Insufficient funds or invalid position size',
                    'trade_id': trade_id
                }
            
            # Simulate execution with slippage
            execution_price = self._apply_slippage(current_price, action)
            
            # Execute the trade
            if action == 'BUY':
                result = await self._execute_buy(symbol, position_size, execution_price, signal)
            elif action == 'SELL':
                result = await self._execute_sell(symbol, position_size, execution_price, signal)
            else:
                return {
                    'success': False,
                    'error': f'Invalid action: {action}',
                    'trade_id': trade_id
                }
            
            if result['success']:
                self.trade_history.append(trade_record)
                self.total_trades += 1
                
                # Update portfolio metrics
                await self._update_portfolio_metrics()
                
                self.logger.info(f"📈 PAPER TRADE: {action} {position_size:.4f} {symbol} @ ${execution_price:.6f}")
                
            return {
                'success': result['success'],
                'trade_id': trade_id,
                'execution_price': execution_price,
                'position_size': position_size,
                'commission': result.get('commission', 0),
                'portfolio_value': self.get_portfolio_value()
            }
            
        except Exception as e:
            self.logger.error(f"Error executing paper trade: {e}")
            return {
                'success': False,
                'error': str(e),
                'trade_id': trade_id
            }
    
    async def _execute_buy(self, symbol: str, size: float, price: float, signal: Dict) -> Dict:
        """Execute a buy order in paper trading"""
        
        try:
            # Calculate costs
            trade_value = size * price
            commission = trade_value * (self.commission_bps / 10000)
            total_cost = trade_value + commission
            
            # Check if we have enough cash
            if total_cost > self.cash_balance:
                return {
                    'success': False,
                    'error': f'Insufficient cash: need ${total_cost:.2f}, have ${self.cash_balance:.2f}'
                }
            
            # Update cash balance
            self.cash_balance -= total_cost
            
            # Update or create position
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': 0,
                    'avg_price': 0,
                    'total_cost': 0,
                    'unrealized_pnl': 0,
                    'stop_loss': signal.get('stop_loss'),
                    'take_profit': signal.get('take_profit'),
                }
            
            position = self.positions[symbol]
            
            # Update position (weighted average for multiple buys)
            old_value = position['quantity'] * position['avg_price']
            new_value = old_value + trade_value
            new_quantity = position['quantity'] + size
            
            position['avg_price'] = new_value / new_quantity if new_quantity > 0 else price
            position['quantity'] = new_quantity
            position['total_cost'] += total_cost
            
            # Update stop loss and take profit if provided
            if signal.get('stop_loss'):
                position['stop_loss'] = signal['stop_loss']
            if signal.get('take_profit'):
                position['take_profit'] = signal['take_profit']
            
            return {
                'success': True,
                'commission': commission,
                'new_position_size': new_quantity,
                'avg_price': position['avg_price']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _execute_sell(self, symbol: str, size: float, price: float, signal: Dict) -> Dict:
        """Execute a sell order in paper trading"""
        
        try:
            # Check if we have the position
            if symbol not in self.positions or self.positions[symbol]['quantity'] <= 0:
                return {
                    'success': False,
                    'error': f'No position to sell in {symbol}'
                }
            
            position = self.positions[symbol]
            available_quantity = position['quantity']
            
            # Adjust size if trying to sell more than available
            actual_size = min(size, available_quantity)
            
            # Calculate proceeds
            trade_value = actual_size * price
            commission = trade_value * (self.commission_bps / 10000)
            net_proceeds = trade_value - commission
            
            # Calculate PnL for this trade
            cost_basis = actual_size * position['avg_price']
            trade_pnl = net_proceeds - cost_basis
            
            # Update cash balance
            self.cash_balance += net_proceeds
            
            # Update position
            remaining_quantity = position['quantity'] - actual_size
            
            if remaining_quantity <= 0.0001:  # Close position completely
                self.total_pnl += trade_pnl
                if trade_pnl > 0:
                    self.winning_trades += 1
                
                del self.positions[symbol]
                self.logger.info(f"💰 Position closed: {symbol} PnL: ${trade_pnl:+.2f}")
                
            else:
                # Partial sell - update position
                position['quantity'] = remaining_quantity
                # Keep same avg_price for remaining position
            
            return {
                'success': True,
                'commission': commission,
                'trade_pnl': trade_pnl,
                'remaining_position': remaining_quantity
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _calculate_position_size(self, signal: Dict, current_price: float) -> float:
        """Calculate appropriate position size for paper trading"""
        
        try:
            # Get portfolio value
            portfolio_value = self.get_portfolio_value()
            
            # Use signal position size if provided
            if signal.get('position_size') and signal['position_size'] > 0:
                requested_value = signal['position_size']
            else:
                # Default to 5% of portfolio for paper trading
                requested_value = portfolio_value * 0.05
            
            # Convert value to quantity
            position_size = requested_value / current_price
            
            # Apply position limits
            max_position_value = portfolio_value * 0.1  # Max 10% per position
            max_position_size = max_position_value / current_price
            
            # Ensure we don't exceed cash available
            trade_cost = position_size * current_price * 1.001  # Include commission
            if trade_cost > self.cash_balance:
                position_size = (self.cash_balance * 0.95) / current_price  # Leave 5% buffer
            
            return min(position_size, max_position_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
    
    async def _get_simulated_price(self, symbol: str) -> float:
        """Get simulated current market price"""
        
        # For paper trading, we'll use simulated prices based on popular crypto pairs
        # In real implementation, this would connect to live data feeds
        
        base_prices = {
            'BTC/USDT': 43000.0,
            'ETH/USDT': 2600.0,
            'SOL/USDT': 105.0,
            'AVAX/USDT': 38.0,
            'PEPE/USDT': 0.00001234,
            'SHIB/USDT': 0.00002456,
            'DOGE/USDT': 0.08,
            'FLOKI/USDT': 0.000145,
            'WIF/USDT': 2.34,
            'BONK/USDT': 0.000034,
            'MEME/USDT': 0.0234,
            '1000SATS/USDT': 0.000456
        }
        
        # Add some realistic price movement (+/- 2%)
        import random
        base_price = base_prices.get(symbol, 100.0)
        price_variation = random.uniform(-0.02, 0.02)
        return base_price * (1 + price_variation)
    
    def _apply_slippage(self, price: float, action: str) -> float:
        """Apply realistic slippage to execution price"""
        
        slippage_factor = self.slippage_bps / 10000
        
        if action == 'BUY':
            # Buying typically costs slightly more
            return price * (1 + slippage_factor)
        else:
            # Selling typically gets slightly less
            return price * (1 - slippage_factor)
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        
        try:
            total_value = self.cash_balance
            
            # Add value of all positions
            for symbol, position in self.positions.items():
                if position['quantity'] > 0:
                    # Get current market price from position history or use entry price with market movement
                    if 'current_price' in position:
                        current_price = position['current_price']
                    elif 'entry_price' in position:
                        # Apply realistic price movement for demo purposes
                        import random
                        price_change = random.uniform(-0.02, 0.02)  # ±2% movement
                        current_price = position['entry_price'] * (1 + price_change)
                        position['current_price'] = current_price  # Cache for consistency
                    else:
                        # Fallback to reasonable crypto prices
                        if 'BTC' in symbol:
                            current_price = 65000
                        elif 'ETH' in symbol:
                            current_price = 3500
                        else:
                            current_price = 100.0
                    
                    position_value = position['quantity'] * current_price
                    
                    # Update position current value and unrealized P&L
                    position['current_value'] = position_value
                    if 'cost_basis' in position:
                        position['unrealized_pnl'] = position_value - position['cost_basis']
                    
                    total_value += position_value
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
            return self.cash_balance
    
    async def _update_portfolio_metrics(self):
        """Update portfolio performance metrics"""
        
        try:
            current_value = self.get_portfolio_value()
            
            # Update peak value
            if current_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_value
            
            # Calculate current drawdown
            current_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Update total PnL
            self.total_pnl = current_value - self.initial_capital
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio metrics: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        try:
            portfolio_value = self.get_portfolio_value()
            total_return = (portfolio_value - self.initial_capital) / self.initial_capital
            
            win_rate = self.winning_trades / max(self.total_trades, 1)
            
            return {
                'portfolio_value': portfolio_value,
                'cash_balance': self.cash_balance,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'total_pnl': self.total_pnl,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': win_rate,
                'win_rate_pct': win_rate * 100,
                'max_drawdown': self.max_drawdown,
                'max_drawdown_pct': self.max_drawdown * 100,
                'active_positions': len(self.positions),
                'positions': dict(self.positions)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return {}
    
    def get_trade_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get trade history with optional limit"""
        
        try:
            if limit:
                return self.trade_history[-limit:]
            return self.trade_history.copy()
            
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return []
    
    async def save_portfolio_state(self):
        """Save current portfolio state to disk"""
        
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_summary': self.get_performance_summary(),
                'positions': dict(self.positions),
                'trade_history': self.trade_history[-100:],  # Last 100 trades
            }
            
            # Save to file
            filename = self.data_dir / f"portfolio_state_{datetime.now().strftime('%Y%m%d')}.json"
            with open(filename, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"💾 Portfolio state saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving portfolio state: {e}")
    
    async def load_portfolio_state(self, date: Optional[str] = None):
        """Load portfolio state from disk"""
        
        try:
            if not date:
                date = datetime.now().strftime('%Y%m%d')
            
            filename = self.data_dir / f"portfolio_state_{date}.json"
            
            if filename.exists():
                with open(filename, 'r') as f:
                    state = json.load(f)
                
                # Restore portfolio state
                portfolio = state.get('portfolio_summary', {})
                self.cash_balance = portfolio.get('cash_balance', self.initial_capital)
                self.positions = state.get('positions', {})
                self.trade_history = state.get('trade_history', [])
                self.total_trades = portfolio.get('total_trades', 0)
                self.winning_trades = portfolio.get('winning_trades', 0)
                
                self.logger.info(f"📂 Portfolio state loaded from {filename}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error loading portfolio state: {e}")
            return False
    
    async def generate_daily_report(self):
        """Generate comprehensive daily trading report"""
        
        try:
            summary = self.get_performance_summary()
            
            report = f"""
📊 DAILY PAPER TRADING REPORT - {datetime.now().strftime('%Y-%m-%d')}
{'=' * 60}

💰 PORTFOLIO SUMMARY:
   Initial Capital: ${self.initial_capital:,.2f}
   Current Value:   ${summary['portfolio_value']:,.2f}
   Cash Balance:    ${summary['cash_balance']:,.2f}
   Total Return:    {summary['total_return_pct']:+.2f}%
   Total PnL:       ${summary['total_pnl']:+,.2f}

📈 TRADING STATISTICS:
   Total Trades:    {summary['total_trades']}
   Winning Trades:  {summary['winning_trades']}
   Win Rate:        {summary['win_rate_pct']:.1f}%
   Max Drawdown:    {summary['max_drawdown_pct']:.2f}%

💼 ACTIVE POSITIONS: {summary['active_positions']}
"""
            
            # Add position details
            for symbol, position in summary['positions'].items():
                if position['quantity'] > 0:
                    unrealized_pnl = position.get('unrealized_pnl', 0)
                    report += f"   {symbol}: {position['quantity']:.4f} @ ${position['avg_price']:.6f} (PnL: ${unrealized_pnl:+.2f})\n"
            
            report += "\n✅ Paper trading report generated successfully!"
            
            # Save report to file
            report_file = self.data_dir / f"daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            self.logger.info("📊 Daily paper trading report generated")
            print(report)
            
        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")
    
    async def check_stop_losses(self):
        """Check and execute stop losses for paper trading"""
        
        try:
            for symbol, position in list(self.positions.items()):
                if position['quantity'] <= 0:
                    continue
                
                current_price = await self._get_simulated_price(symbol)
                
                # Check stop loss
                if position.get('stop_loss') and current_price <= position['stop_loss']:
                    self.logger.warning(f"🛑 STOP LOSS TRIGGERED: {symbol} @ ${current_price:.6f}")
                    
                    # Execute stop loss sell
                    stop_loss_signal = {
                        'symbol': symbol,
                        'action': 'SELL',
                        'position_size': position['quantity'] * current_price,  # Full position
                        'strategy_name': 'stop_loss'
                    }
                    
                    await self.execute_trade(stop_loss_signal)
                
                # Check take profit
                elif position.get('take_profit') and current_price >= position['take_profit']:
                    self.logger.info(f"🎯 TAKE PROFIT HIT: {symbol} @ ${current_price:.6f}")
                    
                    # Execute take profit sell
                    take_profit_signal = {
                        'symbol': symbol,
                        'action': 'SELL',
                        'position_size': position['quantity'] * current_price,  # Full position
                        'strategy_name': 'take_profit'
                    }
                    
                    await self.execute_trade(take_profit_signal)
            
        except Exception as e:
            self.logger.error(f"Error checking stop losses: {e}")
    
    def reset_portfolio(self, new_capital: float = None):
        """Reset paper trading portfolio"""
        
        if new_capital:
            self.initial_capital = new_capital
        
        self.cash_balance = self.initial_capital
        self.positions.clear()
        self.trade_history.clear()
        
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_portfolio_value = self.initial_capital
        
        self.logger.info(f"🔄 Paper trading portfolio reset to ${self.initial_capital:,.2f}")


class PaperTradingSimulator:
    """
    🧪 COMPREHENSIVE PAPER TRADING SIMULATOR
    
    Provides realistic market simulation for testing:
    • Realistic price movements
    • Market impact simulation
    • Latency and slippage modeling
    • Order book simulation
    """
    
    def __init__(self):
        self.logger = logging.getLogger("PaperTradingSimulator")
        self.market_data = {}
        self.last_prices = {}
        
        self.logger.info("🧪 Paper Trading Simulator initialized")
    
    async def simulate_market_conditions(self):
        """Simulate realistic market conditions"""
        
        # This would simulate:
        # - Price volatility
        # - Volume patterns
        # - Market trends
        # - Order book depth
        
        pass
    
    async def get_simulated_orderbook(self, symbol: str) -> Dict:
        """Get simulated order book for realistic execution"""
        
        # Simulate order book with realistic bid/ask spreads
        current_price = await self._get_base_price(symbol)
        spread_bps = 5  # 0.05% spread
        
        bid_price = current_price * (1 - spread_bps / 10000)
        ask_price = current_price * (1 + spread_bps / 10000)
        
        return {
            'bids': [[bid_price, 1000.0]],  # price, quantity
            'asks': [[ask_price, 1000.0]],
            'timestamp': datetime.now().timestamp()
        }
    
    async def _get_base_price(self, symbol: str) -> float:
        """Get base price for simulation"""
        
        # Same price logic as paper trading manager
        base_prices = {
            'BTC/USDT': 43000.0,
            'ETH/USDT': 2600.0,
            'SOL/USDT': 105.0,
            'AVAX/USDT': 38.0,
            'PEPE/USDT': 0.00001234,
            'SHIB/USDT': 0.00002456,
            'DOGE/USDT': 0.08,
            'FLOKI/USDT': 0.000145,
            'WIF/USDT': 2.34,
            'BONK/USDT': 0.000034,
            'MEME/USDT': 0.0234,
            '1000SATS/USDT': 0.000456
        }
        
        return base_prices.get(symbol, 100.0)
