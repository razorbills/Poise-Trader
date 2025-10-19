"""
🕵️ MARKET MANIPULATION DETECTOR 🕵️
Detects pump groups, coordinated dumps, psychological opponent modeling
Portfolio diversification AI for 10+ assets
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

class ManipulationType(Enum):
    PUMP_AND_DUMP = "pump_and_dump"
    COORDINATED_SELL = "coordinated_sell"
    WASH_TRADING = "wash_trading"
    SPOOFING = "spoofing"
    BEAR_RAID = "bear_raid"
    BULL_TRAP = "bull_trap"

class HerdBehavior(Enum):
    EXTREME_GREED = "extreme_greed"
    EXTREME_FEAR = "extreme_fear"
    FOMO = "fomo"
    PANIC_SELLING = "panic_selling"
    EUPHORIA = "euphoria"
    DESPAIR = "despair"

@dataclass
class ManipulationAlert:
    """Market manipulation detection alert"""
    asset: str
    manipulation_type: ManipulationType
    confidence: float
    severity: str  # low/medium/high/critical
    evidence: List[str]
    estimated_impact: float
    time_detected: datetime
    recommended_action: str

@dataclass
class PsychologicalSignal:
    """Psychological opponent modeling signal"""
    behavior_type: HerdBehavior
    intensity: float
    affected_assets: List[str]
    exploitation_opportunity: str
    confidence: float
    duration_estimate: int  # minutes

@dataclass
class PortfolioAllocation:
    """AI-driven portfolio allocation"""
    asset: str
    current_weight: float
    target_weight: float
    rebalance_amount: float
    reasoning: str
    risk_contribution: float

class MarketManipulationDetector:
    """Detects various forms of market manipulation"""
    
    def __init__(self):
        self.manipulation_patterns = {}
        self.volume_anomalies = deque(maxlen=100)
        self.price_anomalies = deque(maxlen=100)
        self.social_sentiment_spikes = deque(maxlen=50)
        self.alerts = []
        
    async def detect_pump_and_dump(self, market_data: Dict[str, Any]) -> List[ManipulationAlert]:
        """Detect pump and dump schemes"""
        alerts = []
        
        for asset, data in market_data.items():
            price_change = data.get('price_change_1h', 0)
            volume_change = data.get('volume_change_1h', 0)
            social_mentions = data.get('social_mentions', 0)
            
            # Classic pump and dump pattern
            if (price_change > 0.15 and  # 15%+ price spike
                volume_change > 3.0 and  # 300%+ volume spike
                social_mentions > 100):  # High social activity
                
                evidence = [
                    f"Price spike: +{price_change:.1%}",
                    f"Volume spike: +{volume_change:.1%}",
                    f"Social mentions: {social_mentions}"
                ]
                
                alert = ManipulationAlert(
                    asset=asset,
                    manipulation_type=ManipulationType.PUMP_AND_DUMP,
                    confidence=0.8,
                    severity="high",
                    evidence=evidence,
                    estimated_impact=-0.25,  # Expect 25% drop
                    time_detected=datetime.now(),
                    recommended_action="AVOID or SHORT"
                )
                alerts.append(alert)
        
        return alerts
    
    async def detect_coordinated_selling(self, whale_activities: List[Any]) -> List[ManipulationAlert]:
        """Detect coordinated selling by multiple whales"""
        alerts = []
        
        # Group activities by asset and time window
        asset_activities = defaultdict(list)
        time_window = timedelta(hours=2)
        
        for activity in whale_activities:
            if activity.transaction_type == 'sell':
                asset_activities[activity.asset].append(activity)
        
        for asset, activities in asset_activities.items():
            # Check for multiple large sells in short timeframe
            recent_sells = [a for a in activities 
                          if datetime.now() - a.timestamp < time_window]
            
            if len(recent_sells) >= 3:  # 3+ whale sells
                total_volume = sum(a.usd_value for a in recent_sells)
                
                if total_volume > 50000000:  # $50M+ coordinated selling
                    evidence = [
                        f"{len(recent_sells)} whale sells detected",
                        f"Total volume: ${total_volume:,.0f}",
                        f"Time window: {time_window.total_seconds()/3600:.1f} hours"
                    ]
                    
                    alert = ManipulationAlert(
                        asset=asset,
                        manipulation_type=ManipulationType.COORDINATED_SELL,
                        confidence=0.75,
                        severity="high",
                        evidence=evidence,
                        estimated_impact=-0.15,
                        time_detected=datetime.now(),
                        recommended_action="DEFENSIVE positioning"
                    )
                    alerts.append(alert)
        
        return alerts
    
    async def detect_wash_trading(self, order_book_data: Dict[str, Any]) -> List[ManipulationAlert]:
        """Detect wash trading patterns"""
        alerts = []
        
        for asset, book_data in order_book_data.items():
            bid_ask_ratio = book_data.get('bid_ask_ratio', 1.0)
            trade_size_variance = book_data.get('trade_size_variance', 0.5)
            round_number_trades = book_data.get('round_number_percentage', 0.1)
            
            # Wash trading indicators
            if (abs(bid_ask_ratio - 1.0) < 0.05 and  # Balanced book (suspicious)
                trade_size_variance < 0.2 and        # Low variance in trade sizes
                round_number_trades > 0.3):          # Many round number trades
                
                evidence = [
                    f"Suspicious bid/ask balance: {bid_ask_ratio:.3f}",
                    f"Low trade size variance: {trade_size_variance:.3f}",
                    f"Round number trades: {round_number_trades:.1%}"
                ]
                
                alert = ManipulationAlert(
                    asset=asset,
                    manipulation_type=ManipulationType.WASH_TRADING,
                    confidence=0.65,
                    severity="medium",
                    evidence=evidence,
                    estimated_impact=0.0,  # Neutral price impact
                    time_detected=datetime.now(),
                    recommended_action="MONITOR closely"
                )
                alerts.append(alert)
        
        return alerts
    
    async def detect_spoofing(self, order_book_data: Dict[str, Any]) -> List[ManipulationAlert]:
        """Detect order spoofing (fake walls)"""
        alerts = []
        
        for asset, book_data in order_book_data.items():
            large_orders = book_data.get('large_orders', [])
            order_cancellation_rate = book_data.get('cancellation_rate', 0.1)
            
            # Look for large orders that get cancelled frequently
            suspicious_orders = [order for order in large_orders 
                               if order.get('size') > 1000000 and  # $1M+ orders
                               order.get('time_active') < 300]     # Active < 5 minutes
            
            if len(suspicious_orders) >= 2 and order_cancellation_rate > 0.4:
                evidence = [
                    f"{len(suspicious_orders)} large orders cancelled quickly",
                    f"Cancellation rate: {order_cancellation_rate:.1%}",
                    "Potential spoofing detected"
                ]
                
                alert = ManipulationAlert(
                    asset=asset,
                    manipulation_type=ManipulationType.SPOOFING,
                    confidence=0.7,
                    severity="medium",
                    evidence=evidence,
                    estimated_impact=0.05,  # Small price impact
                    time_detected=datetime.now(),
                    recommended_action="WAIT for real liquidity"
                )
                alerts.append(alert)
        
        return alerts
    
    async def detect_pump_dump(self, symbols: List[str] = None, price_data: Dict[str, List[float]] = None, **kwargs) -> List[ManipulationAlert]:
        """Detect pump and dump schemes with flexible parameters"""
        # Convert parameters to expected format
        market_data = {}
        
        if symbols and price_data:
            for symbol in symbols:
                if symbol in price_data and len(price_data[symbol]) > 0:
                    prices = price_data[symbol]
                    if len(prices) >= 2:
                        price_change = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0
                        volume_change = kwargs.get('volume_change', 2.0)  # Default volume spike
                        social_mentions = kwargs.get('social_mentions', 50)  # Default social activity
                        
                        market_data[symbol] = {
                            'price_change_1h': price_change,
                            'volume_change_1h': volume_change,
                            'social_mentions': social_mentions
                        }
        
        # If no specific data provided, use kwargs directly
        if not market_data and kwargs:
            market_data = kwargs
        
        # Call the main pump and dump detection method
        return await self.detect_pump_and_dump(market_data)

class PsychologicalOpponentModeler:
    """Models herd behavior and psychological market patterns"""
    
    def __init__(self):
        self.fear_greed_index = 50  # 0-100 scale
        self.social_sentiment_history = deque(maxlen=100)
        self.volume_emotion_correlation = {}
        self.psychological_signals = []
        
    async def analyze_herd_behavior(self, market_data: Dict[str, Any], 
                                  social_data: Dict[str, Any]) -> List[PsychologicalSignal]:
        """Analyze and exploit herd behavior patterns"""
        signals = []
        
        # Update fear & greed index
        self._update_fear_greed_index(market_data, social_data)
        
        # Detect extreme emotions
        if self.fear_greed_index >= 80:  # Extreme greed
            signal = PsychologicalSignal(
                behavior_type=HerdBehavior.EXTREME_GREED,
                intensity=self.fear_greed_index / 100,
                affected_assets=list(market_data.keys()),
                exploitation_opportunity="CONTRARIAN selling opportunity",
                confidence=0.8,
                duration_estimate=240  # 4 hours
            )
            signals.append(signal)
            
        elif self.fear_greed_index <= 20:  # Extreme fear
            signal = PsychologicalSignal(
                behavior_type=HerdBehavior.EXTREME_FEAR,
                intensity=(100 - self.fear_greed_index) / 100,
                affected_assets=list(market_data.keys()),
                exploitation_opportunity="CONTRARIAN buying opportunity",
                confidence=0.85,
                duration_estimate=180  # 3 hours
            )
            signals.append(signal)
        
        # Detect FOMO patterns
        fomo_assets = self._detect_fomo_patterns(market_data, social_data)
        for asset in fomo_assets:
            signal = PsychologicalSignal(
                behavior_type=HerdBehavior.FOMO,
                intensity=0.7,
                affected_assets=[asset],
                exploitation_opportunity="FADE the FOMO - short opportunity",
                confidence=0.75,
                duration_estimate=120  # 2 hours
            )
            signals.append(signal)
        
        # Detect panic selling
        panic_assets = self._detect_panic_selling(market_data)
        for asset in panic_assets:
            signal = PsychologicalSignal(
                behavior_type=HerdBehavior.PANIC_SELLING,
                intensity=0.8,
                affected_assets=[asset],
                exploitation_opportunity="BUY the panic - contrarian opportunity",
                confidence=0.8,
                duration_estimate=90  # 1.5 hours
            )
            signals.append(signal)
        
        self.psychological_signals = signals
        return signals
    
    def _update_fear_greed_index(self, market_data: Dict[str, Any], 
                                social_data: Dict[str, Any]):
        """Update fear & greed index based on multiple factors"""
        factors = []
        
        # Price momentum factor
        avg_price_change = np.mean([data.get('price_change_24h', 0) 
                                   for data in market_data.values()])
        momentum_score = min(100, max(0, 50 + avg_price_change * 200))
        factors.append(momentum_score)
        
        # Volume factor
        avg_volume_change = np.mean([data.get('volume_change_24h', 0) 
                                    for data in market_data.values()])
        volume_score = min(100, max(0, 50 + avg_volume_change * 25))
        factors.append(volume_score)
        
        # Social sentiment factor
        social_sentiment = social_data.get('composite_sentiment', 0.5)
        sentiment_score = social_sentiment * 100
        factors.append(sentiment_score)
        
        # Volatility factor (inverse - high vol = fear)
        avg_volatility = np.mean([data.get('volatility', 0.02) 
                                 for data in market_data.values()])
        volatility_score = max(0, 100 - avg_volatility * 2000)
        factors.append(volatility_score)
        
        # Calculate weighted average
        self.fear_greed_index = np.mean(factors)
    
    def _detect_fomo_patterns(self, market_data: Dict[str, Any], 
                             social_data: Dict[str, Any]) -> List[str]:
        """Detect FOMO patterns in specific assets"""
        fomo_assets = []
        
        for asset, data in market_data.items():
            price_change = data.get('price_change_1h', 0)
            volume_change = data.get('volume_change_1h', 0)
            social_mentions = social_data.get(f'{asset}_mentions', 0)
            
            # FOMO indicators
            if (price_change > 0.05 and      # 5%+ price rise
                volume_change > 1.5 and     # 150%+ volume increase
                social_mentions > 50):      # High social activity
                fomo_assets.append(asset)
        
        return fomo_assets
    
    def _detect_panic_selling(self, market_data: Dict[str, Any]) -> List[str]:
        """Detect panic selling patterns"""
        panic_assets = []
        
        for asset, data in market_data.items():
            price_change = data.get('price_change_1h', 0)
            volume_change = data.get('volume_change_1h', 0)
            rsi = data.get('rsi', 50)
            
            # Panic selling indicators
            if (price_change < -0.08 and     # 8%+ price drop
                volume_change > 2.0 and     # 200%+ volume spike
                rsi < 25):                  # Oversold RSI
                panic_assets.append(asset)
        
        return panic_assets

class PortfolioDiversificationAI:
    """AI-driven portfolio diversification for 10+ assets"""
    
    def __init__(self):
        self.target_assets = [
            'BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE', 'MATIC',
            'GOLD', 'SILVER', 'OIL', 'SPY', 'QQQ'
        ]
        self.correlation_matrix = {}
        self.risk_budgets = {}
        self.current_allocations = {}
        self.rebalance_threshold = 0.05  # 5% deviation triggers rebalance
        
    async def calculate_optimal_allocation(self, market_data: Dict[str, Any], 
                                         risk_tolerance: float = 0.5) -> List[PortfolioAllocation]:
        """Calculate optimal portfolio allocation using modern portfolio theory"""
        allocations = []
        
        # Update correlation matrix
        await self._update_correlations(market_data)
        
        # Calculate expected returns and volatilities
        expected_returns = self._calculate_expected_returns(market_data)
        volatilities = self._calculate_volatilities(market_data)
        
        # Optimize portfolio using simplified Markowitz approach
        optimal_weights = self._optimize_portfolio(expected_returns, volatilities, risk_tolerance)
        
        for asset in self.target_assets:
            if asset in optimal_weights:
                current_weight = self.current_allocations.get(asset, 0.0)
                target_weight = optimal_weights[asset]
                rebalance_amount = target_weight - current_weight
                
                # Only suggest rebalance if significant deviation
                if abs(rebalance_amount) > self.rebalance_threshold:
                    reasoning = self._generate_allocation_reasoning(asset, market_data.get(asset, {}), 
                                                                 current_weight, target_weight)
                    
                    allocation = PortfolioAllocation(
                        asset=asset,
                        current_weight=current_weight,
                        target_weight=target_weight,
                        rebalance_amount=rebalance_amount,
                        reasoning=reasoning,
                        risk_contribution=self._calculate_risk_contribution(asset, target_weight)
                    )
                    allocations.append(allocation)
        
        # Sort by rebalance amount (largest changes first)
        allocations.sort(key=lambda x: abs(x.rebalance_amount), reverse=True)
        
        return allocations
    
    async def _update_correlations(self, market_data: Dict[str, Any]):
        """Update correlation matrix between assets"""
        # Simplified correlation calculation (replace with historical data)
        correlations = {
            ('BTC', 'ETH'): 0.8,
            ('BTC', 'GOLD'): -0.2,
            ('BTC', 'SPY'): 0.4,
            ('ETH', 'SOL'): 0.7,
            ('GOLD', 'SILVER'): 0.9,
            ('SPY', 'QQQ'): 0.95,
            ('OIL', 'GOLD'): 0.3
        }
        
        self.correlation_matrix = correlations
    
    def _calculate_expected_returns(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate expected returns for each asset"""
        expected_returns = {}
        
        for asset in self.target_assets:
            if asset in market_data:
                # Use recent performance as proxy for expected return
                recent_return = market_data[asset].get('price_change_7d', 0.0)
                momentum = market_data[asset].get('momentum', 0.0)
                
                # Adjust for momentum and mean reversion
                expected_return = recent_return * 0.3 + momentum * 0.7
                expected_returns[asset] = expected_return
            else:
                # Default expected return
                expected_returns[asset] = 0.05  # 5% annual
        
        return expected_returns
    
    def _calculate_volatilities(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate volatility for each asset"""
        volatilities = {}
        
        # Default volatilities by asset class
        default_vols = {
            'BTC': 0.8, 'ETH': 0.9, 'BNB': 0.7, 'SOL': 1.2, 'ADA': 1.0,
            'DOT': 1.1, 'LINK': 1.0, 'UNI': 1.3, 'AAVE': 1.2, 'MATIC': 1.1,
            'GOLD': 0.2, 'SILVER': 0.3, 'OIL': 0.4, 'SPY': 0.15, 'QQQ': 0.2
        }
        
        for asset in self.target_assets:
            if asset in market_data:
                volatility = market_data[asset].get('volatility', default_vols.get(asset, 0.5))
            else:
                volatility = default_vols.get(asset, 0.5)
            
            volatilities[asset] = volatility
        
        return volatilities
    
    def _optimize_portfolio(self, expected_returns: Dict[str, float], 
                           volatilities: Dict[str, float], 
                           risk_tolerance: float) -> Dict[str, float]:
        """Optimize portfolio weights using simplified approach"""
        
        # Risk parity base allocation
        risk_contributions = {}
        total_inv_vol = 0
        
        for asset in self.target_assets:
            inv_vol = 1 / volatilities.get(asset, 0.5)
            risk_contributions[asset] = inv_vol
            total_inv_vol += inv_vol
        
        # Normalize to get base weights
        base_weights = {asset: contrib / total_inv_vol 
                       for asset, contrib in risk_contributions.items()}
        
        # Adjust for expected returns and risk tolerance
        adjusted_weights = {}
        total_adjustment = 0
        
        for asset in self.target_assets:
            expected_return = expected_returns.get(asset, 0.05)
            base_weight = base_weights[asset]
            
            # Tilt towards higher expected returns based on risk tolerance
            return_adjustment = expected_return * risk_tolerance * 2
            adjusted_weight = base_weight * (1 + return_adjustment)
            
            adjusted_weights[asset] = max(0.01, adjusted_weight)  # Minimum 1%
            total_adjustment += adjusted_weights[asset]
        
        # Normalize to sum to 1
        final_weights = {asset: weight / total_adjustment 
                        for asset, weight in adjusted_weights.items()}
        
        return final_weights
    
    def _generate_allocation_reasoning(self, asset: str, asset_data: Dict[str, Any], 
                                     current_weight: float, target_weight: float) -> str:
        """Generate reasoning for allocation change"""
        change = target_weight - current_weight
        
        if change > 0:
            action = "INCREASE"
            reasons = []
            
            if asset_data.get('momentum', 0) > 0.02:
                reasons.append("strong momentum")
            if asset_data.get('volatility', 0.5) < 0.3:
                reasons.append("low volatility")
            if asset in ['GOLD', 'SILVER'] and asset_data.get('price_change_24h', 0) < -0.02:
                reasons.append("safe haven demand")
            
            reason_text = ", ".join(reasons) if reasons else "portfolio rebalancing"
            return f"{action} allocation due to {reason_text}"
        
        else:
            action = "DECREASE"
            reasons = []
            
            if asset_data.get('volatility', 0.5) > 0.8:
                reasons.append("high volatility")
            if asset_data.get('rsi', 50) > 75:
                reasons.append("overbought conditions")
            
            reason_text = ", ".join(reasons) if reasons else "risk management"
            return f"{action} allocation for {reason_text}"
    
    def _calculate_risk_contribution(self, asset: str, weight: float) -> float:
        """Calculate risk contribution of asset to portfolio"""
        # Simplified risk contribution calculation
        asset_vol = {
            'BTC': 0.8, 'ETH': 0.9, 'GOLD': 0.2, 'SPY': 0.15
        }.get(asset, 0.5)
        
        return weight * asset_vol
    
    async def optimize_allocation(self, market_data: Dict[str, Any] = None, risk_tolerance: float = 0.5, **kwargs) -> List[PortfolioAllocation]:
        """Optimize portfolio allocation (alias for calculate_optimal_allocation)"""
        if market_data is None:
            # Create default market data if none provided
            market_data = {
                'BTC': {'price_change_7d': 0.05, 'momentum': 0.02, 'volatility': 0.8, 'rsi': 60},
                'ETH': {'price_change_7d': 0.03, 'momentum': 0.01, 'volatility': 0.9, 'rsi': 55},
                'GOLD': {'price_change_7d': -0.01, 'momentum': 0.0, 'volatility': 0.2, 'rsi': 45},
                'SPY': {'price_change_7d': 0.02, 'momentum': 0.005, 'volatility': 0.15, 'rsi': 50}
            }
        
        return await self.calculate_optimal_allocation(market_data, risk_tolerance)
