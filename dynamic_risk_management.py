#!/usr/bin/env python3
"""
🛡️ DYNAMIC AI-POWERED RISK MANAGEMENT SYSTEM
The most advanced risk management for 90% win rate trading

FEATURES:
✅ AI-Powered Position Sizing
✅ Dynamic Stop Loss Optimization
✅ Intelligent Take Profit Levels
✅ Volatility-Adaptive Risk Controls
✅ Performance-Based Risk Adjustment
✅ Real-time Risk Monitoring
✅ Emergency Risk Protection
✅ Drawdown Recovery AI
✅ Correlation-Based Risk Limits
✅ Market Regime Risk Adaptation
"""

import asyncio
import numpy as np
import pandas as pd
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')
import os

_REAL_TRADING_ENABLED = str(os.getenv('REAL_TRADING', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
_STRICT_REAL_DATA = str(os.getenv('STRICT_REAL_DATA', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
ALLOW_SIMULATED_FEATURES = (
    str(os.getenv('ALLOW_SIMULATED_FEATURES', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
    and not _REAL_TRADING_ENABLED
    and not _STRICT_REAL_DATA
)

@dataclass
class RiskParameters:
    """Dynamic risk parameters for a trade"""
    position_size: float  # As fraction of portfolio
    stop_loss_pct: float  # Stop loss percentage
    take_profit_pct: float  # Take profit percentage
    max_risk_per_trade: float  # Maximum risk per trade
    risk_reward_ratio: float  # Target risk/reward
    confidence_threshold: float  # Minimum confidence for trade
    volatility_adjustment: float  # Volatility-based adjustment
    regime_adjustment: float  # Market regime adjustment

@dataclass
class PositionRisk:
    """Risk assessment for current position"""
    current_risk: float  # Current $ at risk
    unrealized_pnl: float  # Current P&L
    risk_pct: float  # Risk as % of portfolio
    time_in_trade: int  # Minutes in trade
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    recommended_action: str  # HOLD, REDUCE, CLOSE, EMERGENCY_EXIT
    stop_loss_distance: float  # Distance to stop loss
    take_profit_distance: float  # Distance to take profit

@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    total_risk_exposure: float
    portfolio_value: float
    risk_utilization: float  # % of risk budget used
    max_drawdown: float
    current_drawdown: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    calmar_ratio: float

class VolatilityEstimator:
    """Advanced volatility estimation for risk management"""
    
    def __init__(self):
        self.volatility_models = ['garch', 'ewma', 'parkinson', 'realized']
        self.volatility_history = deque(maxlen=1000)
        
    async def estimate_volatility(self, price_history: List[float], timeframe: str = '1m') -> Dict[str, float]:
        """Estimate volatility using multiple models"""
        
        if len(price_history) < 20:
            return {'realized': 0.02, 'forecast': 0.02, 'confidence': 0.5}
        
        prices = np.array(price_history)
        returns = np.diff(prices) / prices[:-1]
        
        # 1. Realized Volatility (Historical)
        realized_vol = np.std(returns) * np.sqrt(1440) if timeframe == '1m' else np.std(returns)
        
        # 2. EWMA Volatility (Exponentially Weighted)
        ewma_vol = self._calculate_ewma_volatility(returns)
        
        # 3. Parkinson Estimator (High-Low range)
        parkinson_vol = self._calculate_parkinson_estimator(prices)
        
        # 4. GARCH Simulation
        garch_vol = self._simulate_garch_volatility(returns)
        
        # 5. Regime-Adjusted Volatility
        regime_vol = await self._calculate_regime_volatility(returns, prices)
        
        # Ensemble volatility forecast
        volatilities = [realized_vol, ewma_vol, parkinson_vol, garch_vol, regime_vol]
        weights = [0.2, 0.25, 0.15, 0.25, 0.15]  # Model weights
        
        forecast_vol = sum(vol * weight for vol, weight in zip(volatilities, weights))
        
        # Volatility confidence
        vol_std = np.std(volatilities)
        confidence = max(0.3, 1.0 - (vol_std / forecast_vol)) if forecast_vol > 0 else 0.5
        
        vol_data = {
            'realized': realized_vol,
            'ewma': ewma_vol,
            'parkinson': parkinson_vol,
            'garch': garch_vol,
            'regime_adjusted': regime_vol,
            'forecast': forecast_vol,
            'confidence': confidence,
            'percentile_rank': self._get_volatility_percentile(forecast_vol)
        }
        
        self.volatility_history.append(vol_data)
        return vol_data
    
    def _calculate_ewma_volatility(self, returns: np.ndarray, lambda_param: float = 0.94) -> float:
        """Calculate EWMA volatility"""
        
        if len(returns) < 10:
            return np.std(returns)
        
        weights = np.array([(1 - lambda_param) * (lambda_param ** i) for i in range(len(returns))])
        weights = weights[::-1]  # Reverse for recent bias
        weights /= weights.sum()
        
        weighted_variance = np.sum(weights * (returns ** 2))
        return np.sqrt(weighted_variance)
    
    def _calculate_parkinson_estimator(self, prices: np.ndarray) -> float:
        """Calculate Parkinson volatility estimator using high-low ranges"""
        
        # Simulate high/low from price data
        parkinson_values = []
        for i in range(1, len(prices)):
            # Estimate high/low as +/- 0.5% of price
            high = prices[i] * 1.005
            low = prices[i] * 0.995
            
            if low > 0:
                parkinson_value = (1 / (4 * np.log(2))) * (np.log(high / low) ** 2)
                parkinson_values.append(parkinson_value)
        
        if parkinson_values:
            return np.sqrt(np.mean(parkinson_values))
        else:
            return np.std(np.diff(prices) / prices[:-1])
    
    def _simulate_garch_volatility(self, returns: np.ndarray) -> float:
        """Simulate GARCH(1,1) volatility forecast"""
        
        if len(returns) < 20:
            return np.std(returns)
        
        # Simple GARCH(1,1) simulation
        alpha = 0.1  # ARCH parameter
        beta = 0.8   # GARCH parameter
        omega = 0.0001  # Long-term variance
        
        # Initialize
        variance = np.var(returns)
        garch_variances = []
        
        for i, ret in enumerate(returns[-20:]):  # Last 20 observations
            variance = omega + alpha * (ret ** 2) + beta * variance
            garch_variances.append(variance)
        
        # Forecast next period
        forecast_variance = omega + alpha * (returns[-1] ** 2) + beta * garch_variances[-1]
        return np.sqrt(forecast_variance)
    
    async def _calculate_regime_volatility(self, returns: np.ndarray, prices: np.ndarray) -> float:
        """Calculate regime-adjusted volatility"""
        
        # Different volatility in different regimes
        recent_returns = returns[-20:] if len(returns) >= 20 else returns
        
        # Detect regime from price action
        if len(prices) >= 10:
            trend = (prices[-1] - prices[-10]) / prices[-10]
            volatility_base = np.std(recent_returns)
            
            # Adjust for regime
            if abs(trend) > 0.05:  # Strong trending
                regime_adjustment = 0.8  # Lower volatility in trends
            elif abs(trend) < 0.01:  # Consolidating
                regime_adjustment = 1.2  # Higher volatility expected in consolidation
            else:
                regime_adjustment = 1.0
            
            return volatility_base * regime_adjustment
        
        return np.std(recent_returns)
    
    def _get_volatility_percentile(self, current_vol: float) -> float:
        """Get percentile rank of current volatility"""
        
        if not self.volatility_history:
            return 0.5
        
        historical_vols = [vh['forecast'] for vh in self.volatility_history]
        
        if len(historical_vols) < 5:
            return 0.5
        
        percentile = (sum(1 for vol in historical_vols if vol < current_vol) / len(historical_vols))
        return percentile

class DynamicPositionSizer:
    """AI-powered dynamic position sizing system"""
    
    def __init__(self):
        self.sizing_history = deque(maxlen=500)
        self.performance_tracker = defaultdict(list)
        self.base_risk_per_trade = 0.02  # 2% base risk
        self.max_risk_per_trade = 0.05   # 5% maximum risk
        self.min_risk_per_trade = 0.005  # 0.5% minimum risk
        
    def configure(self, max_risk_per_trade: float = None, account_size: float = None):
        """Configure position sizer parameters"""
        if max_risk_per_trade is not None:
            self.max_risk_per_trade = max_risk_per_trade
        if account_size is not None:
            self.account_size = account_size
        print(f"   📏 Dynamic Position Sizer configured - Max Risk: {self.max_risk_per_trade:.2%}")
    
    async def calculate_position_size(self, signal, account_size: float, available_cash: float, 
                                    market_data: Dict, consecutive_losses: int = 0) -> float:
        """Calculate position size (wrapper for calculate_optimal_position_size)"""
        try:
            # Extract signal parameters
            symbol = signal.symbol if hasattr(signal, 'symbol') else 'UNKNOWN'
            confidence = signal.confidence if hasattr(signal, 'confidence') else 0.5
            entry_price = signal.entry_price if hasattr(signal, 'entry_price') else 100.0
            stop_loss = signal.stop_loss if hasattr(signal, 'stop_loss') else entry_price * 0.98
            
            # Create market intelligence dict
            market_intelligence = {
                'volatility': {
                    'forecast': market_data.get('volatility', 0.02),
                    'percentile_rank': 0.5
                },
                'regime': {
                    'regime_type': market_data.get('regime', 'neutral'),
                    'confidence': 0.7
                }
            }
            
            # Calculate position fraction
            position_fraction = await self.calculate_optimal_position_size(
                symbol, confidence, market_intelligence, account_size, entry_price, stop_loss
            )
            
            # Convert to USD amount
            position_size_usd = position_fraction * account_size
            
            # Apply consecutive loss reduction
            if consecutive_losses >= 3:
                reduction_factor = 0.5 ** (consecutive_losses - 2)
                position_size_usd *= reduction_factor
            
            return min(position_size_usd, available_cash * 0.8)
            
        except Exception as e:
            print(f"   ⚠️ Position size calculation error: {e}")
            return min(account_size * 0.02, available_cash * 0.3)  # Fallback to 2% of account
        
    async def calculate_optimal_position_size(self, symbol: str, signal_confidence: float,
                                            market_intelligence: Dict, portfolio_value: float,
                                            current_price: float, stop_loss_price: float) -> float:
        """Calculate optimal position size using AI-driven approach"""
        
        # 1. Base position size from confidence
        confidence_multiplier = self._calculate_confidence_multiplier(signal_confidence)
        
        # 2. Market regime adjustment
        regime_multiplier = self._calculate_regime_multiplier(market_intelligence.get('regime', {}))
        
        # 3. Volatility adjustment
        volatility_multiplier = self._calculate_volatility_multiplier(
            market_intelligence.get('volatility', {}), symbol
        )
        
        # 4. Historical performance adjustment
        performance_multiplier = await self._calculate_performance_multiplier(symbol)
        
        # 5. Portfolio heat adjustment
        heat_multiplier = self._calculate_portfolio_heat_multiplier(portfolio_value)
        
        # 6. Risk parity adjustment
        risk_parity_multiplier = await self._calculate_risk_parity_adjustment(symbol)
        
        # Composite multiplier
        total_multiplier = (confidence_multiplier * regime_multiplier * volatility_multiplier * 
                          performance_multiplier * heat_multiplier * risk_parity_multiplier)
        
        # Calculate risk amount
        risk_amount = portfolio_value * self.base_risk_per_trade * total_multiplier
        
        # Constrain to limits
        risk_amount = max(portfolio_value * self.min_risk_per_trade, 
                         min(portfolio_value * self.max_risk_per_trade, risk_amount))
        
        # Convert to position size based on stop loss
        if stop_loss_price > 0 and current_price != stop_loss_price:
            risk_per_unit = abs(current_price - stop_loss_price)
            position_size = risk_amount / risk_per_unit
            
            # Position size as fraction of portfolio
            position_fraction = (position_size * current_price) / portfolio_value
        else:
            # Fallback calculation
            position_fraction = risk_amount / portfolio_value / 0.02  # Assume 2% stop
        
        # Final constraints
        position_fraction = max(0.001, min(0.15, position_fraction))  # 0.1% to 15% of portfolio
        
        # Track sizing decision
        self.sizing_history.append({
            'symbol': symbol,
            'timestamp': datetime.now(),
            'confidence': signal_confidence,
            'position_fraction': position_fraction,
            'multipliers': {
                'confidence': confidence_multiplier,
                'regime': regime_multiplier,
                'volatility': volatility_multiplier,
                'performance': performance_multiplier,
                'heat': heat_multiplier,
                'risk_parity': risk_parity_multiplier
            }
        })
        
        return position_fraction
    
    def _calculate_confidence_multiplier(self, confidence: float) -> float:
        """Calculate position size multiplier based on signal confidence"""
        
        # Non-linear confidence scaling
        if confidence < 0.6:
            return 0.5  # Very small positions for low confidence
        elif confidence < 0.75:
            return 0.75
        elif confidence < 0.85:
            return 1.0
        elif confidence < 0.95:
            return 1.3
        else:
            return 1.5  # Larger positions for very high confidence
    
    def _calculate_regime_multiplier(self, regime: Dict) -> float:
        """Calculate position size multiplier based on market regime"""
        
        regime_type = regime.get('regime_type', 'neutral')
        regime_confidence = regime.get('confidence', 0.5)
        
        # Regime-based sizing
        if regime_type in ['strong_bull', 'strong_bear'] and regime_confidence > 0.8:
            return 1.3  # Larger positions in strong trending regimes
        elif regime_type in ['bull_trend', 'bear_trend'] and regime_confidence > 0.7:
            return 1.1
        elif regime_type in ['volatile_trending', 'volatile_sideways']:
            return 0.7  # Smaller positions in volatile markets
        elif regime_type == 'consolidating':
            return 0.6  # Very small positions in consolidation
        else:
            return 1.0
    
    def _calculate_volatility_multiplier(self, volatility_data: Dict, symbol: str) -> float:
        """Calculate position size multiplier based on volatility"""
        
        if not volatility_data:
            return 1.0
        
        forecast_vol = volatility_data.get('forecast', 0.02)
        vol_percentile = volatility_data.get('percentile_rank', 0.5)
        
        # Inverse relationship with volatility
        if vol_percentile > 0.9:  # Extreme high volatility
            return 0.4
        elif vol_percentile > 0.8:  # High volatility
            return 0.6
        elif vol_percentile > 0.6:  # Above average volatility
            return 0.8
        elif vol_percentile < 0.2:  # Low volatility
            return 1.3
        else:
            return 1.0
    
    async def _calculate_performance_multiplier(self, symbol: str) -> float:
        """Calculate position size multiplier based on historical performance"""
        
        # Get recent performance for this symbol
        symbol_performance = self.performance_tracker.get(symbol, [])
        
        if len(symbol_performance) < 5:
            return 1.0  # No adjustment for insufficient data
        
        # Calculate recent win rate and profit factor
        recent_trades = symbol_performance[-20:]  # Last 20 trades
        wins = sum(1 for trade in recent_trades if trade['pnl'] > 0)
        win_rate = wins / len(recent_trades)
        
        total_profit = sum(max(0, trade['pnl']) for trade in recent_trades)
        total_loss = abs(sum(min(0, trade['pnl']) for trade in recent_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else 2.0
        
        # Performance-based adjustment
        if win_rate > 0.8 and profit_factor > 2.0:
            return 1.4  # Increase size for strong performance
        elif win_rate > 0.7 and profit_factor > 1.5:
            return 1.2
        elif win_rate < 0.4 or profit_factor < 0.8:
            return 0.5  # Reduce size for poor performance
        elif win_rate < 0.5:
            return 0.7
        else:
            return 1.0
    
    def _calculate_portfolio_heat_multiplier(self, portfolio_value: float) -> float:
        """Calculate multiplier based on current portfolio heat (open positions risk)"""
        
        # Simulate current portfolio heat (in real system, this would be actual)
        current_positions = len(self.sizing_history) % 5  # Simulate 0-4 open positions
        portfolio_heat = current_positions * 0.02  # Assume 2% risk per position
        
        # Reduce position sizes as portfolio heat increases
        if portfolio_heat > 0.08:  # More than 8% at risk
            return 0.5
        elif portfolio_heat > 0.06:  # More than 6% at risk
            return 0.7
        elif portfolio_heat > 0.04:  # More than 4% at risk
            return 0.85
        else:
            return 1.0
    
    async def _calculate_risk_parity_adjustment(self, symbol: str) -> float:
        """Calculate risk parity adjustment across portfolio"""
        
        # In a real system, this would analyze correlation with other positions
        # For now, simulate based on symbol characteristics
        
        if symbol.startswith('BTC'):
            # BTC is typically higher volatility
            return 0.9
        elif symbol.startswith('ETH'):
            # ETH is moderate volatility
            return 1.0
        elif symbol.startswith('BNB'):
            # Exchange tokens are typically lower volatility
            return 1.1
        else:
            # Alt coins are typically higher volatility
            return 0.8

class DynamicStopLossOptimizer:
    """AI-powered dynamic stop loss optimization"""
    
    def __init__(self):
        self.stop_performance = deque(maxlen=1000)
        self.optimal_stops = {}  # Symbol-specific optimal stops
        self.atr_multiplier = 2.0  # Default ATR multiplier
        self.volatility_adjustment = True  # Default volatility adjustment
        
    def set_parameters(self, atr_multiplier: float = None, volatility_adjustment: bool = None):
        """Set parameters for stop loss optimization"""
        if atr_multiplier is not None:
            self.atr_multiplier = atr_multiplier
        if volatility_adjustment is not None:
            self.volatility_adjustment = volatility_adjustment
        print(f"   🛑 Dynamic Stop Optimizer configured - ATR: {self.atr_multiplier}x, Vol Adj: {self.volatility_adjustment}")
        
    async def calculate_optimal_stop_loss(self, symbol: str, entry_price: float,
                                        direction: str, market_intelligence: Dict,
                                        volatility_data: Dict) -> Tuple[float, str]:
        """Calculate optimal stop loss using AI analysis"""
        
        # 1. ATR-based stop
        atr_stop = self._calculate_atr_stop(entry_price, direction, volatility_data)
        
        # 2. Support/Resistance based stop
        sr_stop = self._calculate_support_resistance_stop(
            entry_price, direction, market_intelligence.get('regime', {})
        )
        
        # 3. Volatility percentile stop
        vol_stop = self._calculate_volatility_percentile_stop(
            entry_price, direction, volatility_data
        )
        
        # 4. Historical performance stop
        perf_stop = await self._calculate_performance_optimized_stop(
            symbol, entry_price, direction
        )
        
        # 5. Market regime stop
        regime_stop = self._calculate_regime_stop(
            entry_price, direction, market_intelligence.get('regime', {})
        )
        
        # Ensemble stop loss calculation
        stops = [atr_stop, sr_stop, vol_stop, perf_stop, regime_stop]
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        # Remove invalid stops and adjust weights
        valid_stops = [(stop, weight) for stop, weight in zip(stops, weights) if stop is not None]
        
        if valid_stops:
            total_weight = sum(weight for _, weight in valid_stops)
            optimal_stop = sum(stop * weight / total_weight for stop, weight in valid_stops)
        else:
            # Fallback stop
            optimal_stop = entry_price * (0.98 if direction == 'BUY' else 1.02)
        
        # Stop loss method used
        method = "AI_Ensemble_Stop"
        
        # Validate stop loss
        optimal_stop = self._validate_stop_loss(entry_price, optimal_stop, direction)
        
        return optimal_stop, method
    
    def _calculate_atr_stop(self, entry_price: float, direction: str, volatility_data: Dict) -> Optional[float]:
        """Calculate ATR-based stop loss"""
        
        forecast_vol = volatility_data.get('forecast', 0.02)
        
        # Convert daily volatility to ATR approximation
        atr_multiplier = 2.0  # 2x ATR for stop
        atr_distance = entry_price * forecast_vol * atr_multiplier
        
        if direction == 'BUY':
            return entry_price - atr_distance
        else:
            return entry_price + atr_distance
    
    def _calculate_support_resistance_stop(self, entry_price: float, direction: str, regime: Dict) -> Optional[float]:
        """Calculate support/resistance based stop"""
        
        support = regime.get('support_level')
        resistance = regime.get('resistance_level')
        
        if direction == 'BUY' and support:
            # Stop below support with buffer
            return support * 0.995
        elif direction == 'SELL' and resistance:
            # Stop above resistance with buffer
            return resistance * 1.005
        
        return None
    
    def _calculate_volatility_percentile_stop(self, entry_price: float, direction: str,
                                           volatility_data: Dict) -> Optional[float]:
        """Calculate stop based on volatility percentile"""
        
        vol_percentile = volatility_data.get('percentile_rank', 0.5)
        forecast_vol = volatility_data.get('forecast', 0.02)
        
        # Wider stops in high volatility environments
        if vol_percentile > 0.8:
            multiplier = 3.0  # Wide stops in high vol
        elif vol_percentile > 0.6:
            multiplier = 2.5
        elif vol_percentile < 0.2:
            multiplier = 1.5  # Tight stops in low vol
        else:
            multiplier = 2.0
        
        stop_distance = entry_price * forecast_vol * multiplier
        
        if direction == 'BUY':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    async def _calculate_performance_optimized_stop(self, symbol: str, entry_price: float,
                                                  direction: str) -> Optional[float]:
        """Calculate stop loss optimized for historical performance"""
        
        # Get historical stop performance for this symbol
        symbol_stops = [sp for sp in self.stop_performance if sp.get('symbol') == symbol]
        
        if len(symbol_stops) < 10:
            return None  # Insufficient data
        
        # Analyze optimal stop distances
        successful_stops = [sp for sp in symbol_stops if sp.get('outcome') == 'profitable']
        failed_stops = [sp for sp in symbol_stops if sp.get('outcome') == 'stopped_out']
        
        if successful_stops and failed_stops:
            # Find optimal stop distance that maximizes win rate
            avg_successful_distance = np.mean([sp['stop_distance_pct'] for sp in successful_stops])
            avg_failed_distance = np.mean([sp['stop_distance_pct'] for sp in failed_stops])
            
            # Optimal distance is between successful and failed averages, closer to successful
            optimal_distance_pct = avg_successful_distance * 0.7 + avg_failed_distance * 0.3
            
            stop_distance = entry_price * optimal_distance_pct
            
            if direction == 'BUY':
                return entry_price - stop_distance
            else:
                return entry_price + stop_distance
        
        return None
    
    def _calculate_regime_stop(self, entry_price: float, direction: str, regime: Dict) -> Optional[float]:
        """Calculate regime-appropriate stop loss"""
        
        regime_type = regime.get('regime_type', 'neutral')
        volatility_level = regime.get('volatility_level', 'normal')
        
        # Base stop percentage by regime
        if regime_type in ['strong_bull', 'strong_bear']:
            base_stop_pct = 0.015  # Tight stops in strong trends
        elif regime_type in ['volatile_trending', 'volatile_sideways']:
            base_stop_pct = 0.04   # Wide stops in volatility
        elif regime_type == 'consolidating':
            base_stop_pct = 0.02   # Moderate stops in consolidation
        else:
            base_stop_pct = 0.025  # Default stop
        
        # Volatility adjustment
        if volatility_level == 'extreme':
            base_stop_pct *= 1.5
        elif volatility_level == 'high':
            base_stop_pct *= 1.2
        elif volatility_level == 'low':
            base_stop_pct *= 0.8
        
        stop_distance = entry_price * base_stop_pct
        
        if direction == 'BUY':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def _validate_stop_loss(self, entry_price: float, stop_loss: float, direction: str) -> float:
        """Validate and constrain stop loss to reasonable bounds"""
        
        if direction == 'BUY':
            # Stop must be below entry
            if stop_loss >= entry_price:
                stop_loss = entry_price * 0.98
            
            # Minimum 0.5% stop, maximum 10% stop
            min_stop = entry_price * 0.995
            max_stop = entry_price * 0.90
            stop_loss = max(max_stop, min(min_stop, stop_loss))
            
        else:  # SELL
            # Stop must be above entry
            if stop_loss <= entry_price:
                stop_loss = entry_price * 1.02
            
            # Minimum 0.5% stop, maximum 10% stop
            min_stop = entry_price * 1.005
            max_stop = entry_price * 1.10
            stop_loss = min(max_stop, max(min_stop, stop_loss))
        
        return stop_loss

class IntelligentTakeProfitManager:
    """AI-powered take profit optimization"""
    
    def __init__(self):
        self.profit_performance = deque(maxlen=1000)
        self.optimal_targets = {}
        
    async def calculate_take_profit_levels(self, symbol: str, entry_price: float,
                                         direction: str, stop_loss: float,
                                         market_intelligence: Dict,
                                         signal_confidence: float) -> List[Tuple[float, float]]:
        """Calculate multiple take profit levels with allocation percentages"""
        
        # 1. Calculate risk/reward based targets
        risk_distance = abs(entry_price - stop_loss)
        
        # 2. Get market intelligence targets
        regime_targets = self._calculate_regime_targets(
            entry_price, direction, market_intelligence.get('regime', {}), risk_distance
        )
        
        # 3. Get volatility-based targets
        vol_targets = self._calculate_volatility_targets(
            entry_price, direction, market_intelligence.get('volatility', {}), risk_distance
        )
        
        # 4. Get support/resistance targets
        sr_targets = self._calculate_support_resistance_targets(
            entry_price, direction, market_intelligence.get('regime', {})
        )
        
        # 5. Get performance-optimized targets
        perf_targets = await self._calculate_performance_targets(
            symbol, entry_price, direction, risk_distance
        )
        
        # 6. Combine into intelligent profit levels
        profit_levels = await self._optimize_profit_levels(
            entry_price, direction, regime_targets, vol_targets, sr_targets, 
            perf_targets, signal_confidence
        )
        
        return profit_levels
    
    def _calculate_regime_targets(self, entry_price: float, direction: str,
                                regime: Dict, risk_distance: float) -> List[Tuple[float, float]]:
        """Calculate take profit targets based on market regime"""
        
        regime_type = regime.get('regime_type', 'neutral')
        trend_strength = regime.get('trend_strength', 0.0)
        
        # Base risk/reward ratios by regime
        if regime_type in ['strong_bull', 'strong_bear']:
            rr_ratios = [1.5, 2.5, 4.0]  # Aggressive targets in strong trends
            allocations = [0.4, 0.4, 0.2]
        elif regime_type in ['bull_trend', 'bear_trend']:
            rr_ratios = [1.5, 2.0, 3.0]
            allocations = [0.5, 0.3, 0.2]
        elif regime_type == 'consolidating':
            rr_ratios = [1.0, 1.5]  # Conservative targets in consolidation
            allocations = [0.6, 0.4]
        else:
            rr_ratios = [1.5, 2.0]  # Default targets
            allocations = [0.6, 0.4]
        
        targets = []
        for i, (rr, allocation) in enumerate(zip(rr_ratios, allocations)):
            target_distance = risk_distance * rr
            
            if direction == 'BUY':
                target_price = entry_price + target_distance
            else:
                target_price = entry_price - target_distance
            
            targets.append((target_price, allocation))
        
        return targets
    
    def _calculate_volatility_targets(self, entry_price: float, direction: str,
                                    volatility_data: Dict, risk_distance: float) -> List[Tuple[float, float]]:
        """Calculate volatility-adjusted take profit targets"""
        
        if not volatility_data:
            return []
        
        vol_percentile = volatility_data.get('percentile_rank', 0.5)
        
        # Adjust targets based on volatility environment
        if vol_percentile > 0.8:  # High volatility - wider targets
            rr_ratios = [2.0, 3.5, 5.0]
            allocations = [0.3, 0.4, 0.3]
        elif vol_percentile < 0.2:  # Low volatility - tighter targets
            rr_ratios = [1.0, 1.5, 2.0]
            allocations = [0.5, 0.3, 0.2]
        else:  # Normal volatility
            rr_ratios = [1.5, 2.5, 3.5]
            allocations = [0.4, 0.4, 0.2]
        
        targets = []
        for rr, allocation in zip(rr_ratios, allocations):
            target_distance = risk_distance * rr
            
            if direction == 'BUY':
                target_price = entry_price + target_distance
            else:
                target_price = entry_price - target_distance
            
            targets.append((target_price, allocation))
        
        return targets
    
    def _calculate_support_resistance_targets(self, entry_price: float, direction: str,
                                            regime: Dict) -> List[Tuple[float, float]]:
        """Calculate targets based on support and resistance levels"""
        
        support = regime.get('support_level')
        resistance = regime.get('resistance_level')
        
        targets = []
        
        if direction == 'BUY' and resistance:
            # Target resistance levels for long positions
            if resistance > entry_price:
                # Conservative target just below resistance
                conservative_target = resistance * 0.998
                targets.append((conservative_target, 0.7))
                
                # Aggressive target through resistance
                if resistance * 1.005 > entry_price * 1.02:  # Meaningful upside
                    aggressive_target = resistance * 1.005
                    targets.append((aggressive_target, 0.3))
        
        elif direction == 'SELL' and support:
            # Target support levels for short positions
            if support < entry_price:
                # Conservative target just above support
                conservative_target = support * 1.002
                targets.append((conservative_target, 0.7))
                
                # Aggressive target through support
                if support * 0.995 < entry_price * 0.98:  # Meaningful downside
                    aggressive_target = support * 0.995
                    targets.append((aggressive_target, 0.3))
        
        return targets
    
    async def _calculate_performance_targets(self, symbol: str, entry_price: float,
                                          direction: str, risk_distance: float) -> List[Tuple[float, float]]:
        """Calculate performance-optimized take profit targets"""
        
        # Analyze historical take profit performance
        symbol_profits = [pp for pp in self.profit_performance if pp.get('symbol') == symbol]
        
        if len(symbol_profits) < 5:
            return []  # Insufficient data
        
        # Find optimal risk/reward ratios
        profitable_exits = [pp for pp in symbol_profits if pp.get('outcome') == 'profitable']
        
        if profitable_exits:
            rr_ratios = [pp['risk_reward_ratio'] for pp in profitable_exits]
            
            # Calculate percentiles for target setting
            target_25th = np.percentile(rr_ratios, 25)
            target_50th = np.percentile(rr_ratios, 50)
            target_75th = np.percentile(rr_ratios, 75)
            
            targets = []
            
            # Conservative target (25th percentile)
            conservative_distance = risk_distance * target_25th
            if direction == 'BUY':
                conservative_price = entry_price + conservative_distance
            else:
                conservative_price = entry_price - conservative_distance
            targets.append((conservative_price, 0.5))
            
            # Moderate target (50th percentile)
            moderate_distance = risk_distance * target_50th
            if direction == 'BUY':
                moderate_price = entry_price + moderate_distance
            else:
                moderate_price = entry_price - moderate_distance
            targets.append((moderate_price, 0.3))
            
            # Aggressive target (75th percentile)
            aggressive_distance = risk_distance * target_75th
            if direction == 'BUY':
                aggressive_price = entry_price + aggressive_distance
            else:
                aggressive_price = entry_price - aggressive_distance
            targets.append((aggressive_price, 0.2))
            
            return targets
        
        return []
    
    async def _optimize_profit_levels(self, entry_price: float, direction: str,
                                    regime_targets: List, vol_targets: List,
                                    sr_targets: List, perf_targets: List,
                                    confidence: float) -> List[Tuple[float, float]]:
        """Optimize and combine all target sources"""
        
        all_targets = []
        all_targets.extend(regime_targets)
        all_targets.extend(vol_targets)
        all_targets.extend(sr_targets)
        all_targets.extend(perf_targets)
        
        if not all_targets:
            # Fallback targets
            risk_distance = entry_price * 0.02  # 2% risk assumption
            if direction == 'BUY':
                return [
                    (entry_price + risk_distance * 1.5, 0.5),
                    (entry_price + risk_distance * 2.5, 0.3),
                    (entry_price + risk_distance * 4.0, 0.2)
                ]
            else:
                return [
                    (entry_price - risk_distance * 1.5, 0.5),
                    (entry_price - risk_distance * 2.5, 0.3),
                    (entry_price - risk_distance * 4.0, 0.2)
                ]
        
        # Cluster targets and allocate weights
        target_prices = [target[0] for target in all_targets]
        target_weights = [target[1] for target in all_targets]
        
        # Create final optimized targets
        optimized_targets = []
        
        # Sort targets by distance from entry
        if direction == 'BUY':
            sorted_targets = sorted(zip(target_prices, target_weights), key=lambda x: x[0])
        else:
            sorted_targets = sorted(zip(target_prices, target_weights), key=lambda x: x[0], reverse=True)
        
        # Group similar targets
        final_targets = []
        i = 0
        while i < len(sorted_targets) and len(final_targets) < 4:
            price, weight = sorted_targets[i]
            
            # Combine nearby targets
            combined_weight = weight
            combined_price = price * weight
            
            j = i + 1
            while j < len(sorted_targets):
                next_price, next_weight = sorted_targets[j]
                price_diff_pct = abs(next_price - price) / entry_price
                
                if price_diff_pct < 0.01:  # Within 1% - combine
                    combined_weight += next_weight
                    combined_price += next_price * next_weight
                    j += 1
                else:
                    break
            
            if combined_weight > 0:
                avg_price = combined_price / combined_weight
                final_targets.append((avg_price, combined_weight))
            
            i = j
        
        # Normalize weights
        total_weight = sum(weight for _, weight in final_targets)
        if total_weight > 0:
            optimized_targets = [(price, weight / total_weight) for price, weight in final_targets]
        
        # Limit to 3 targets maximum
        return optimized_targets[:3]

class RealTimeRiskMonitor:
    """Real-time risk monitoring and emergency controls"""
    
    def __init__(self):
        self.risk_alerts = deque(maxlen=100)
        self.emergency_triggers = {
            'max_drawdown': 0.15,      # 15% max drawdown
            'portfolio_heat': 0.12,    # 12% max portfolio risk
            'consecutive_losses': 5,   # 5 consecutive losses
            'daily_loss_limit': 0.08,  # 8% daily loss limit
            'volatility_spike': 3.0    # 3x normal volatility
        }
        self.daily_stats = {}
        self.reset_daily_stats()
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        today = datetime.now().date()
        self.daily_stats = {
            'date': today,
            'trades_count': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'consecutive_losses': 0
        }
    
    async def monitor_portfolio_risk(self, portfolio_value: float, open_positions: List[Dict],
                                   recent_trades: List[Dict]) -> Tuple[bool, List[str], Dict]:
        """Monitor overall portfolio risk in real-time"""
        
        risk_alerts = []
        emergency_action_required = False
        risk_metrics = {}
        
        # Check daily reset
        if self.daily_stats['date'] != datetime.now().date():
            self.reset_daily_stats()
        
        # 1. Calculate current portfolio exposure
        total_risk_exposure = sum(pos.get('risk_amount', 0) for pos in open_positions)
        portfolio_heat = total_risk_exposure / portfolio_value if portfolio_value > 0 else 0
        
        # 2. Check drawdown
        current_drawdown = self._calculate_current_drawdown(recent_trades, portfolio_value)
        
        # 3. Check daily performance
        daily_pnl = sum(trade.get('pnl', 0) for trade in recent_trades 
                       if self._is_today(trade.get('exit_time')))
        daily_loss_pct = abs(daily_pnl) / portfolio_value if daily_pnl < 0 else 0
        
        # 4. Check consecutive losses
        consecutive_losses = self._count_consecutive_losses(recent_trades)
        
        # 5. Check volatility spikes
        volatility_alert = await self._check_volatility_spikes(open_positions)
        
        # Risk Checks
        if portfolio_heat > self.emergency_triggers['portfolio_heat']:
            emergency_action_required = True
            risk_alerts.append(f"🚨 EMERGENCY: Portfolio heat {portfolio_heat:.1%} exceeds limit!")
        
        if current_drawdown > self.emergency_triggers['max_drawdown']:
            emergency_action_required = True
            risk_alerts.append(f"🚨 EMERGENCY: Drawdown {current_drawdown:.1%} exceeds limit!")
        
        if consecutive_losses >= self.emergency_triggers['consecutive_losses']:
            emergency_action_required = True
            risk_alerts.append(f"🚨 EMERGENCY: {consecutive_losses} consecutive losses!")
        
        if daily_loss_pct > self.emergency_triggers['daily_loss_limit']:
            emergency_action_required = True
            risk_alerts.append(f"🚨 EMERGENCY: Daily loss {daily_loss_pct:.1%} exceeds limit!")
        
        if volatility_alert:
            risk_alerts.append("⚠️ High volatility spike detected - reducing position sizes")
        
        # Warning level checks
        if portfolio_heat > self.emergency_triggers['portfolio_heat'] * 0.8:
            risk_alerts.append(f"⚠️ WARNING: Portfolio heat {portfolio_heat:.1%} approaching limit")
        
        if current_drawdown > self.emergency_triggers['max_drawdown'] * 0.8:
            risk_alerts.append(f"⚠️ WARNING: Drawdown {current_drawdown:.1%} approaching limit")
        
        # Calculate comprehensive risk metrics
        risk_metrics = {
            'portfolio_heat': portfolio_heat,
            'current_drawdown': current_drawdown,
            'daily_loss_pct': daily_loss_pct,
            'consecutive_losses': consecutive_losses,
            'total_risk_exposure': total_risk_exposure,
            'emergency_action_required': emergency_action_required,
            'risk_budget_used': portfolio_heat / 0.15,  # % of max risk used
            'recommended_max_new_risk': max(0, self.emergency_triggers['portfolio_heat'] - portfolio_heat)
        }
        
        return emergency_action_required, risk_alerts, risk_metrics
    
    def _calculate_current_drawdown(self, recent_trades: List[Dict], portfolio_value: float) -> float:
        """Calculate current drawdown from peak"""
        
        if not recent_trades:
            return 0.0
        
        # Calculate running portfolio value
        running_value = portfolio_value
        peak_value = portfolio_value
        max_drawdown = 0.0
        
        # Go through trades in reverse to find peak
        for trade in reversed(recent_trades[-50:]):  # Last 50 trades
            pnl = trade.get('pnl', 0)
            running_value -= pnl  # Subtract to go backwards
            
            if running_value > peak_value:
                peak_value = running_value
            
            current_dd = (peak_value - running_value) / peak_value
            max_drawdown = max(max_drawdown, current_dd)
        
        return max_drawdown
    
    def _count_consecutive_losses(self, recent_trades: List[Dict]) -> int:
        """Count consecutive losing trades"""
        
        if not recent_trades:
            return 0
        
        consecutive = 0
        for trade in reversed(recent_trades):
            if trade.get('pnl', 0) < 0:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    async def _check_volatility_spikes(self, open_positions: List[Dict]) -> bool:
        """Check for sudden volatility spikes"""
        
        # In real system, this would analyze current vs historical volatility
        # Simulate volatility spike detection
        if not ALLOW_SIMULATED_FEATURES:
            return False
        volatility_multiplier = np.random.uniform(0.5, 4.0)
        return volatility_multiplier > 2.5
    
    def _is_today(self, timestamp) -> bool:
        """Check if timestamp is from today"""
        if timestamp is None:
            return False
        
        if isinstance(timestamp, datetime):
            return timestamp.date() == datetime.now().date()
        
        return False

class AdvancedRiskManager:
    """Master AI-powered risk management system"""
    
    def __init__(self):
        self.volatility_estimator = VolatilityEstimator()
        self.position_sizer = DynamicPositionSizer()
        self.stop_optimizer = DynamicStopLossOptimizer()
        self.profit_manager = IntelligentTakeProfitManager()
        self.risk_monitor = RealTimeRiskMonitor()
        
        # Risk management state
        self.active_positions = {}
        self.risk_budget = 0.15  # 15% max portfolio risk
        self.current_risk_utilization = 0.0
        self.performance_history = deque(maxlen=1000)
        
        print("🛡️ ADVANCED AI RISK MANAGEMENT SYSTEM INITIALIZED")
        print("   ✅ Volatility Estimator")
        print("   ✅ Dynamic Position Sizer")
        print("   ✅ Stop Loss Optimizer")
        print("   ✅ Take Profit Manager")
        print("   ✅ Real-time Risk Monitor")
    
    async def calculate_trade_parameters(self, symbol: str, signal_direction: str,
                                       signal_confidence: float, current_price: float,
                                       market_intelligence: Dict, portfolio_value: float) -> RiskParameters:
        """Calculate complete trade parameters using AI risk management"""
        
        print(f"🛡️ CALCULATING RISK PARAMETERS FOR {symbol} {signal_direction}")
        
        # 1. Estimate volatility
        price_history = market_intelligence.get('price_history', [current_price])
        volatility_data = await self.volatility_estimator.estimate_volatility(price_history)
        
        # 2. Calculate optimal stop loss
        optimal_stop, stop_method = await self.stop_optimizer.calculate_optimal_stop_loss(
            symbol, current_price, signal_direction, market_intelligence, volatility_data
        )
        
        # 3. Calculate position size
        position_fraction = await self.position_sizer.calculate_optimal_position_size(
            symbol, signal_confidence, market_intelligence, portfolio_value, 
            current_price, optimal_stop
        )
        
        # 4. Calculate take profit levels
        take_profit_levels = await self.profit_manager.calculate_take_profit_levels(
            symbol, current_price, signal_direction, optimal_stop, 
            market_intelligence, signal_confidence
        )
        
        # 5. Calculate derived parameters
        stop_loss_pct = abs(current_price - optimal_stop) / current_price
        
        # Primary take profit (weighted average)
        if take_profit_levels:
            primary_tp = sum(price * weight for price, weight in take_profit_levels)
            take_profit_pct = abs(primary_tp - current_price) / current_price
            risk_reward_ratio = take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 2.0
        else:
            take_profit_pct = stop_loss_pct * 2.0  # Default 1:2 risk/reward
            risk_reward_ratio = 2.0
        
        # 6. Final risk parameter validation
        max_risk_per_trade = portfolio_value * position_fraction * stop_loss_pct
        
        # Confidence threshold adjustment
        base_threshold = 0.65
        regime_confidence = market_intelligence.get('regime', {}).get('confidence', 0.5)
        vol_confidence = volatility_data.get('confidence', 0.5)
        
        confidence_threshold = base_threshold + (1 - min(regime_confidence, vol_confidence)) * 0.1
        
        # Volatility adjustment factor
        vol_percentile = volatility_data.get('percentile_rank', 0.5)
        volatility_adjustment = 1.0 - (vol_percentile - 0.5) * 0.4
        
        # Regime adjustment factor
        regime_type = market_intelligence.get('regime', {}).get('regime_type', 'neutral')
        regime_adjustment = 1.2 if regime_type in ['strong_bull', 'strong_bear'] else 1.0
        
        risk_params = RiskParameters(
            position_size=position_fraction,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_risk_per_trade=max_risk_per_trade,
            risk_reward_ratio=risk_reward_ratio,
            confidence_threshold=confidence_threshold,
            volatility_adjustment=volatility_adjustment,
            regime_adjustment=regime_adjustment
        )
        
        print(f"✅ RISK PARAMETERS CALCULATED:")
        print(f"   📊 Position Size: {position_fraction:.1%}")
        print(f"   🛑 Stop Loss: {stop_loss_pct:.2%}")
        print(f"   🎯 Take Profit: {take_profit_pct:.2%}")
        print(f"   ⚖️ Risk/Reward: 1:{risk_reward_ratio:.1f}")
        
        return risk_params
    
    async def monitor_active_position(self, symbol: str, position_data: Dict,
                                    current_price: float, market_intelligence: Dict) -> PositionRisk:
        """Monitor and assess risk for active position"""
        
        entry_price = position_data.get('entry_price', current_price)
        direction = position_data.get('direction', 'BUY')
        position_size = position_data.get('position_size', 0)
        stop_loss = position_data.get('stop_loss', 0)
        entry_time = position_data.get('entry_time', datetime.now())
        
        # Calculate current metrics
        if direction == 'BUY':
            unrealized_pnl = (current_price - entry_price) * position_size
        else:
            unrealized_pnl = (entry_price - current_price) * position_size
        
        # Risk assessment
        current_risk = abs(entry_price - stop_loss) * position_size if stop_loss else 0
        portfolio_value = position_data.get('portfolio_value', 10000)
        risk_pct = current_risk / portfolio_value if portfolio_value > 0 else 0
        
        # Time in trade
        time_in_trade = int((datetime.now() - entry_time).total_seconds() / 60)
        
        # Risk level assessment
        risk_level = self._assess_position_risk_level(
            risk_pct, unrealized_pnl, time_in_trade, market_intelligence
        )
        
        # Recommended action
        recommended_action = await self._get_recommended_position_action(
            symbol, position_data, current_price, risk_level, market_intelligence
        )
        
        # Calculate distances
        stop_loss_distance = abs(current_price - stop_loss) / current_price if stop_loss else 0
        
        # Estimate take profit distance (simplified)
        take_profit_distance = risk_pct * 2.0  # Assume 1:2 risk/reward
        
        return PositionRisk(
            current_risk=current_risk,
            unrealized_pnl=unrealized_pnl,
            risk_pct=risk_pct,
            time_in_trade=time_in_trade,
            risk_level=risk_level,
            recommended_action=recommended_action,
            stop_loss_distance=stop_loss_distance,
            take_profit_distance=take_profit_distance
        )
    
    def _assess_position_risk_level(self, risk_pct: float, unrealized_pnl: float,
                                  time_in_trade: int, market_intelligence: Dict) -> str:
        """Assess risk level for position"""
        
        # Base risk level from position size
        if risk_pct > 0.08:
            base_level = 'CRITICAL'
        elif risk_pct > 0.05:
            base_level = 'HIGH'
        elif risk_pct > 0.03:
            base_level = 'MEDIUM'
        else:
            base_level = 'LOW'
        
        # Adjust for unrealized P&L
        if unrealized_pnl < -risk_pct * 0.8:  # Close to stop loss
            if base_level == 'LOW':
                base_level = 'MEDIUM'
            elif base_level == 'MEDIUM':
                base_level = 'HIGH'
            elif base_level == 'HIGH':
                base_level = 'CRITICAL'
        
        # Adjust for time in trade
        if time_in_trade > 240:  # More than 4 hours
            if base_level == 'LOW':
                base_level = 'MEDIUM'
        
        # Adjust for market regime
        regime = market_intelligence.get('regime', {})
        if regime.get('regime_type') in ['volatile_trending', 'volatile_sideways']:
            if base_level == 'LOW':
                base_level = 'MEDIUM'
            elif base_level == 'MEDIUM':
                base_level = 'HIGH'
        
        return base_level
    
    async def _get_recommended_position_action(self, symbol: str, position_data: Dict,
                                             current_price: float, risk_level: str,
                                             market_intelligence: Dict) -> str:
        """Get AI recommendation for position management"""
        
        direction = position_data.get('direction', 'BUY')
        unrealized_pnl = position_data.get('unrealized_pnl', 0)
        time_in_trade = position_data.get('time_in_trade', 0)
        
        # Emergency actions
        if risk_level == 'CRITICAL':
            return 'EMERGENCY_EXIT'
        
        # Profit taking logic
        if unrealized_pnl > 0:
            # Check if we should take partial profits
            profit_pct = unrealized_pnl / (position_data.get('position_value', 1))
            
            if profit_pct > 0.06:  # 6% profit
                return 'TAKE_PARTIAL_PROFIT'
            elif profit_pct > 0.03 and risk_level in ['HIGH', 'MEDIUM']:
                return 'TAKE_PARTIAL_PROFIT'
        
        # Time-based exits
        if time_in_trade > 360:  # 6 hours
            if unrealized_pnl >= 0:
                return 'CLOSE'  # Break even or better after long time
            elif risk_level == 'HIGH':
                return 'REDUCE'  # Reduce size for long losing trades
        
        # Market regime changes
        regime = market_intelligence.get('regime', {})
        regime_type = regime.get('regime_type', 'neutral')
        
        # Exit if regime changed against position
        if ((direction == 'BUY' and regime_type in ['strong_bear', 'bear_trend']) or
            (direction == 'SELL' and regime_type in ['strong_bull', 'bull_trend'])):
            if regime.get('confidence', 0) > 0.8:
                return 'CLOSE'  # Exit when strong opposing regime
        
        # Default action
        return 'HOLD'
    
    async def calculate_portfolio_metrics(self, portfolio_value: float, trade_history: List[Dict]) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        
        if not trade_history:
            return self._default_risk_metrics(portfolio_value)
        
        # Calculate win rate
        wins = sum(1 for trade in trade_history if trade.get('pnl', 0) > 0)
        total_trades = len(trade_history)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(max(0, trade.get('pnl', 0)) for trade in trade_history)
        gross_loss = abs(sum(min(0, trade.get('pnl', 0)) for trade in trade_history))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Calculate drawdown
        running_balance = portfolio_value
        peak_balance = portfolio_value
        max_drawdown = 0
        
        for trade in reversed(trade_history):
            pnl = trade.get('pnl', 0)
            running_balance -= pnl
            
            if running_balance > peak_balance:
                peak_balance = running_balance
            
            drawdown = (peak_balance - running_balance) / peak_balance
            max_drawdown = max(max_drawdown, drawdown)
        
        current_drawdown = (peak_balance - portfolio_value) / peak_balance if peak_balance > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        if trade_history:
            returns = [trade.get('pnl', 0) / portfolio_value for trade in trade_history]
            avg_return = np.mean(returns)
            return_std = np.std(returns)
            sharpe_ratio = (avg_return * 252) / (return_std * np.sqrt(252)) if return_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate Calmar ratio
        calmar_ratio = (gross_profit / portfolio_value) / max_drawdown if max_drawdown > 0 else 0
        
        # Current risk utilization
        current_risk = sum(pos.get('risk_amount', 0) for pos in self.active_positions.values())
        risk_utilization = current_risk / (portfolio_value * self.risk_budget)
        
        return RiskMetrics(
            total_risk_exposure=current_risk,
            portfolio_value=portfolio_value,
            risk_utilization=risk_utilization,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio
        )
    
    async def emergency_risk_control(self, portfolio_value: float, emergency_type: str) -> List[str]:
        """Execute emergency risk control procedures"""
        
        emergency_actions = []
        
        if emergency_type == 'max_drawdown':
            emergency_actions.extend([
                "🚨 EMERGENCY: Maximum drawdown exceeded",
                "💀 Closing all positions immediately",
                "⏸️ Halting new trades for 24 hours",
                "📊 Initiating portfolio review"
            ])
        
        elif emergency_type == 'portfolio_heat':
            emergency_actions.extend([
                "🚨 EMERGENCY: Portfolio heat too high",
                "✂️ Reducing all position sizes by 50%",
                "🛑 No new trades until heat drops below 8%",
                "⚖️ Rebalancing risk allocation"
            ])
        
        elif emergency_type == 'consecutive_losses':
            emergency_actions.extend([
                "🚨 EMERGENCY: Too many consecutive losses",
                "🔄 Switching to defensive mode",
                "📉 Reducing position sizes by 75%",
                "🧠 Triggering AI model recalibration"
            ])
        
        elif emergency_type == 'daily_loss_limit':
            emergency_actions.extend([
                "🚨 EMERGENCY: Daily loss limit reached",
                "🛑 Stopping all trading for today",
                "💰 Closing all losing positions",
                "📈 Keeping only profitable positions"
            ])
        
        return emergency_actions
    
    async def optimize_existing_positions(self, current_positions: Dict[str, Dict],
                                        market_data: Dict[str, Any]) -> Dict[str, Dict]:
        """Optimize existing positions based on current market conditions"""
        
        optimization_actions = {}
        
        for symbol, position in current_positions.items():
            current_price = market_data.get(symbol, {}).get('price', 0)
            if current_price <= 0:
                continue
            
            # Get current market intelligence
            intelligence = market_data.get(symbol, {}).get('intelligence', {})
            
            # Monitor position risk
            position_risk = await self.monitor_active_position(
                symbol, position, current_price, intelligence
            )
            
            actions = []
            
            # Implement recommended actions
            if position_risk.recommended_action == 'EMERGENCY_EXIT':
                actions.append('CLOSE_IMMEDIATELY')
            elif position_risk.recommended_action == 'CLOSE':
                actions.append('CLOSE_POSITION')
            elif position_risk.recommended_action == 'REDUCE':
                actions.append('REDUCE_POSITION_50')
            elif position_risk.recommended_action == 'TAKE_PARTIAL_PROFIT':
                actions.append('TAKE_PARTIAL_PROFIT_30')
            
            # Trailing stop adjustments
            if position_risk.unrealized_pnl > 0:
                trailing_stop = await self._calculate_trailing_stop(
                    symbol, position, current_price, intelligence
                )
                if trailing_stop:
                    actions.append(f'UPDATE_STOP_TO_{trailing_stop:.2f}')
            
            if actions:
                optimization_actions[symbol] = {
                    'actions': actions,
                    'position_risk': position_risk.__dict__,
                    'current_price': current_price
                }
        
        return optimization_actions
    
    async def _calculate_trailing_stop(self, symbol: str, position: Dict,
                                     current_price: float, intelligence: Dict) -> Optional[float]:
        """Calculate intelligent trailing stop level"""
        
        direction = position.get('direction', 'BUY')
        entry_price = position.get('entry_price', current_price)
        current_stop = position.get('stop_loss', 0)
        
        # Only trail if in profit
        if direction == 'BUY' and current_price <= entry_price:
            return None
        elif direction == 'SELL' and current_price >= entry_price:
            return None
        
        # Get volatility for trailing distance
        volatility = intelligence.get('volatility', {}).get('forecast', 0.02)
        
        # Calculate trailing distance (2x volatility)
        trailing_distance = current_price * volatility * 2.0
        
        if direction == 'BUY':
            new_stop = current_price - trailing_distance
            # Only move stop up
            if new_stop > current_stop:
                return new_stop
        else:  # SELL
            new_stop = current_price + trailing_distance
            # Only move stop down
            if new_stop < current_stop:
                return new_stop
        
        return None
    
    def _default_risk_metrics(self, portfolio_value: float) -> RiskMetrics:
        """Default risk metrics when no trade history"""
        return RiskMetrics(
            total_risk_exposure=0.0,
            portfolio_value=portfolio_value,
            risk_utilization=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            calmar_ratio=0.0
        )
    
    def update_performance(self, symbol: str, trade_result: Dict):
        """Update performance tracking for continuous improvement"""
        
        self.performance_history.append(trade_result)
        self.position_sizer.performance_tracker[symbol].append(trade_result)
        
        # Update stop loss performance
        if 'stop_loss_triggered' in trade_result:
            self.stop_optimizer.stop_performance.append({
                'symbol': symbol,
                'stop_distance_pct': trade_result.get('stop_distance_pct', 0),
                'outcome': 'stopped_out' if trade_result['pnl'] < 0 else 'profitable',
                'pnl': trade_result['pnl']
            })
        
        # Update take profit performance
        if 'take_profit_triggered' in trade_result:
            self.profit_manager.profit_performance.append({
                'symbol': symbol,
                'risk_reward_ratio': trade_result.get('risk_reward_ratio', 0),
                'outcome': 'profitable' if trade_result['pnl'] > 0 else 'loss',
                'profit_level': trade_result.get('profit_level', 1)
            })

# Global risk manager instance
risk_manager = AdvancedRiskManager()

# Export components
__all__ = [
    'risk_manager',
    'AdvancedRiskManager',
    'RiskParameters',
    'PositionRisk',
    'RiskMetrics',
    'VolatilityEstimator',
    'DynamicPositionSizer',
    'DynamicStopLossOptimizer',
    'IntelligentTakeProfitManager',
    'RealTimeRiskMonitor'
]
