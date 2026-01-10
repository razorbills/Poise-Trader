"""
🐋 WHALE INTELLIGENCE SYSTEM 🐋
Dark pool tracking, whale wallet monitoring, front-running detection
Copycat mode to mirror best-performing wallets
"""

import asyncio
import hashlib
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

_REAL_TRADING_ENABLED = str(os.getenv('REAL_TRADING', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
_STRICT_REAL_DATA = str(os.getenv('STRICT_REAL_DATA', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
ALLOW_SIMULATED_FEATURES = (
    str(os.getenv('ALLOW_SIMULATED_FEATURES', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
    and not _REAL_TRADING_ENABLED
    and not _STRICT_REAL_DATA
)

@dataclass
class WhaleActivity:
    """Dark pool/whale tracking data"""
    wallet_address: str
    transaction_hash: str
    asset: str
    amount: float
    usd_value: float
    transaction_type: str  # buy/sell/transfer
    timestamp: datetime
    exchange: Optional[str]
    is_otc: bool
    confidence: float
    wallet_label: Optional[str] = None

@dataclass
class FrontRunSignal:
    """Front-running detection signal"""
    asset: str
    predicted_direction: str  # buy/sell
    whale_wallet: str
    estimated_size: float
    confidence: float
    time_window: int  # seconds until execution
    risk_reward_ratio: float
    whale_success_rate: float

@dataclass
class CopycatSignal:
    """Signal to copy successful wallet"""
    target_wallet: str
    wallet_performance: float
    action: str  # buy/sell
    asset: str
    position_size: float
    confidence: float
    wallet_win_rate: float
    avg_hold_time: int

class DarkPoolTracker:
    """Tracks dark pools and whale wallet activities"""
    
    def __init__(self):
        self.whale_wallets = {}
        self.otc_trades = []
        self.whale_threshold = 1000000  # $1M+ transactions
        self.tracked_addresses = set()
        self.wallet_performance = defaultdict(list)
        
        # Known whale wallets with labels
        self.known_whales = {
            '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa': 'Satoshi Genesis',
            '3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy': 'Binance Cold Storage',
            '1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ': 'Bitfinex Cold Storage',
            '37XuVSEpWW4trkfmvWzegTHQt7BdktSKUs': 'Coinbase Custody',
            '3FupZp8gDaHmBBWCqRXsWjTLCv4R4GhQjY': 'Grayscale Bitcoin Trust'
        }
        
    async def track_whale_activities(self) -> List[WhaleActivity]:
        """Track whale wallet activities across multiple chains"""
        if not ALLOW_SIMULATED_FEATURES:
            return []
        activities = []
        
        # Simulated whale activities (replace with real blockchain monitoring)
        simulated_activities = [
            {
                'wallet': '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
                'asset': 'BTC',
                'amount': 500.0,
                'usd_value': 32500000,
                'type': 'transfer',
                'exchange': None,
                'is_otc': True,
                'pattern': 'accumulation'
            },
            {
                'wallet': '0x742d35Cc6634C0532925a3b8D0C9e7C5e7B8a9eD',
                'asset': 'ETH',
                'amount': 10000.0,
                'usd_value': 35000000,
                'type': 'sell',
                'exchange': 'Binance',
                'is_otc': False,
                'pattern': 'distribution'
            },
            {
                'wallet': '3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy',
                'asset': 'BTC',
                'amount': 1000.0,
                'usd_value': 65000000,
                'type': 'buy',
                'exchange': 'Coinbase',
                'is_otc': True,
                'pattern': 'strategic_accumulation'
            },
            {
                'wallet': 'bc1qa5wkgaew2dkv56kfvj49j0av5nml45x9ek9hz6',
                'asset': 'BTC',
                'amount': 2500.0,
                'usd_value': 162500000,
                'type': 'transfer',
                'exchange': 'Kraken',
                'is_otc': True,
                'pattern': 'institutional_flow'
            }
        ]
        
        for activity_data in simulated_activities:
            if activity_data['usd_value'] >= self.whale_threshold:
                confidence = self._calculate_activity_confidence(activity_data)
                
                activity = WhaleActivity(
                    wallet_address=activity_data['wallet'],
                    transaction_hash=self._generate_tx_hash(activity_data['wallet']),
                    asset=activity_data['asset'],
                    amount=activity_data['amount'],
                    usd_value=activity_data['usd_value'],
                    transaction_type=activity_data['type'],
                    timestamp=datetime.now(),
                    exchange=activity_data['exchange'],
                    is_otc=activity_data['is_otc'],
                    confidence=confidence,
                    wallet_label=self.known_whales.get(activity_data['wallet'], 'Unknown Whale')
                )
                activities.append(activity)
                
                # Track this wallet for future monitoring
                self.tracked_addresses.add(activity_data['wallet'])
                
                # Update wallet performance tracking
                self._update_wallet_performance(activity_data['wallet'], activity_data)
        
        return activities
    
    def _calculate_activity_confidence(self, activity_data: Dict[str, Any]) -> float:
        """Calculate confidence in whale activity detection"""
        base_confidence = 0.6
        
        # Size factor
        if activity_data['usd_value'] > 100000000:  # $100M+
            base_confidence += 0.3
        elif activity_data['usd_value'] > 50000000:  # $50M+
            base_confidence += 0.2
        elif activity_data['usd_value'] > 10000000:  # $10M+
            base_confidence += 0.1
        
        # Known wallet factor
        if activity_data['wallet'] in self.known_whales:
            base_confidence += 0.1
        
        # OTC factor
        if activity_data['is_otc']:
            base_confidence += 0.05
        
        # Exchange factor
        if activity_data['exchange'] in ['Binance', 'Coinbase', 'Kraken']:
            base_confidence += 0.05
        
        return min(base_confidence, 0.95)
    
    def _generate_tx_hash(self, wallet: str) -> str:
        """Generate simulated transaction hash"""
        return hashlib.sha256(f"{wallet}{datetime.now()}".encode()).hexdigest()[:16]
    
    def _update_wallet_performance(self, wallet: str, activity_data: Dict[str, Any]):
        """Update performance tracking for whale wallet"""
        if wallet not in self.wallet_performance:
            self.wallet_performance[wallet] = deque(maxlen=50)  # Keep last 50 activities
        
        performance_record = {
            'timestamp': datetime.now(),
            'type': activity_data['type'],
            'asset': activity_data['asset'],
            'usd_value': activity_data['usd_value'],
            'pattern': activity_data.get('pattern', 'unknown')
        }
        
        self.wallet_performance[wallet].append(performance_record)
    
    async def analyze_dark_pools(self, symbols: List[str], price_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze dark pool activities for given symbols"""
        try:
            if not ALLOW_SIMULATED_FEATURES:
                return {
                    'large_block_trades': [],
                    'hidden_volume': {},
                    'institutional_flow': {},
                    'dark_pool_sentiment': 'neutral'
                }
            dark_pool_analysis = {
                'large_block_trades': [],
                'hidden_volume': {},
                'institutional_flow': {},
                'dark_pool_sentiment': 'neutral'
            }
            
            # Simulate dark pool analysis for each symbol
            for symbol in symbols:
                if symbol.replace('/USDT', '') in ['BTC', 'ETH', 'BNB']:
                    # Simulate large block detection
                    dark_pool_analysis['large_block_trades'].append({
                        'symbol': symbol,
                        'estimated_size': np.random.uniform(1000000, 50000000),
                        'direction': np.random.choice(['buy', 'sell']),
                        'confidence': np.random.uniform(0.6, 0.9)
                    })
                    
                    # Hidden volume estimation
                    if symbol in price_data and len(price_data[symbol]) > 10:
                        prices = price_data[symbol]
                        volatility = np.std(prices[-10:]) / np.mean(prices[-10:])
                        dark_pool_analysis['hidden_volume'][symbol] = {
                            'estimated_ratio': min(0.3, volatility * 10),
                            'confidence': 0.7
                        }
            
            return dark_pool_analysis
        except Exception as e:
            logger.error(f"Dark pool analysis error: {e}")
            return {'large_block_trades': [], 'hidden_volume': {}}
    
    async def detect_whale_accumulation(self) -> List[Dict[str, Any]]:
        """Detect whale accumulation patterns"""
        try:
            if not ALLOW_SIMULATED_FEATURES:
                return []
            accumulation_signals = []
            
            # Simulate whale accumulation detection
            whale_patterns = [
                {
                    'symbol': 'BTC/USDT',
                    'wallet': '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
                    'accumulation_score': 0.85,
                    'time_frame': '7d',
                    'estimated_size': 25000000,
                    'pattern': 'gradual_accumulation'
                },
                {
                    'symbol': 'ETH/USDT',
                    'wallet': '0x742d35Cc6634C0532925a3b8D0C9e7C5e7B8a9eD',
                    'accumulation_score': 0.72,
                    'time_frame': '3d',
                    'estimated_size': 15000000,
                    'pattern': 'aggressive_accumulation'
                }
            ]
            
            # Filter high-confidence patterns
            for pattern in whale_patterns:
                if pattern['accumulation_score'] > 0.7:
                    accumulation_signals.append(pattern)
            
            return accumulation_signals
        except Exception as e:
            logger.error(f"Whale accumulation detection error: {e}")
            return []
    
    async def detect_hidden_liquidity(self) -> Dict[str, Any]:
        """Detect hidden liquidity movements and iceberg orders"""
        if not ALLOW_SIMULATED_FEATURES:
            return {
                'large_orders': [],
                'iceberg_orders': [],
                'dark_pool_volume': {},
                'institutional_flow': {},
                'order_book_anomalies': []
            }
        hidden_liquidity = {
            'large_orders': [],
            'iceberg_orders': [],
            'dark_pool_volume': {},
            'institutional_flow': {},
            'order_book_anomalies': []
        }
        
        # Simulate large order detection
        hidden_liquidity['large_orders'] = [
            {
                'asset': 'BTC',
                'size': 500,
                'price': 64800,
                'side': 'buy',
                'confidence': 0.85,
                'estimated_impact': 0.002,  # 0.2% price impact
                'time_to_execution': 1800  # 30 minutes
            },
            {
                'asset': 'ETH',
                'size': 2000,
                'price': 3480,
                'side': 'sell',
                'confidence': 0.78,
                'estimated_impact': -0.0015,
                'time_to_execution': 900  # 15 minutes
            }
        ]
        
        # Simulate iceberg order detection
        hidden_liquidity['iceberg_orders'] = [
            {
                'asset': 'BTC',
                'total_size': 1500,
                'visible_size': 50,
                'side': 'buy',
                'price_levels': [64750, 64700, 64650],
                'confidence': 0.72
            }
        ]
        
        # Dark pool volume estimates
        hidden_liquidity['dark_pool_volume'] = {
            'BTC': {
                'volume_24h': 25000000,  # $25M
                'percentage_of_total': 0.15,  # 15% of total volume
                'avg_trade_size': 500000
            },
            'ETH': {
                'volume_24h': 18000000,  # $18M
                'percentage_of_total': 0.12,
                'avg_trade_size': 350000
            }
        }
        
        # Institutional flow detection
        hidden_liquidity['institutional_flow'] = {
            'net_flow_btc': 15000000,  # $15M net inflow
            'net_flow_eth': -8000000,  # $8M net outflow
            'flow_confidence': 0.8,
            'major_participants': ['Grayscale', 'MicroStrategy', 'Tesla']
        }
        
        return hidden_liquidity

class FrontRunningDetector:
    """Detects front-running opportunities by monitoring whale patterns"""
    
    def __init__(self):
        self.whale_patterns = defaultdict(list)
        self.front_run_signals = []
        self.pattern_confidence_threshold = 0.7
        self.whale_success_rates = {}
        
    async def analyze_whale_patterns(self, whale_activities: List[WhaleActivity]) -> List[FrontRunSignal]:
        """Analyze whale patterns for front-running opportunities"""
        signals = []
        
        for activity in whale_activities:
            # Analyze historical patterns for this whale
            pattern_confidence = self._calculate_pattern_confidence(activity)
            
            if pattern_confidence >= self.pattern_confidence_threshold:
                # Predict whale's next move
                predicted_direction = self._predict_whale_direction(activity)
                time_window = self._estimate_execution_window(activity)
                risk_reward = self._calculate_risk_reward(activity)
                
                # Get whale's historical success rate
                success_rate = self._get_whale_success_rate(activity.wallet_address)
                
                signal = FrontRunSignal(
                    asset=activity.asset,
                    predicted_direction=predicted_direction,
                    whale_wallet=activity.wallet_address,
                    estimated_size=activity.usd_value,
                    confidence=pattern_confidence,
                    time_window=time_window,
                    risk_reward_ratio=risk_reward,
                    whale_success_rate=success_rate
                )
                signals.append(signal)
                
                # Update whale patterns
                self._update_whale_patterns(activity)
        
        # Sort by confidence and potential profit
        signals.sort(key=lambda s: s.confidence * s.risk_reward_ratio, reverse=True)
        
        self.front_run_signals = signals
        return signals
    
    def _calculate_pattern_confidence(self, activity: WhaleActivity) -> float:
        """Calculate confidence in whale pattern recognition"""
        base_confidence = 0.5
        
        # Historical pattern factor
        if activity.wallet_address in self.whale_patterns:
            pattern_consistency = self._calculate_pattern_consistency(activity.wallet_address)
            base_confidence += pattern_consistency * 0.3
        
        # Size factor
        if activity.usd_value > 50000000:  # $50M+
            base_confidence += 0.2
        elif activity.usd_value > 10000000:  # $10M+
            base_confidence += 0.1
        
        # OTC factor (higher predictability)
        if activity.is_otc:
            base_confidence += 0.1
        
        # Known whale factor
        if activity.wallet_label and activity.wallet_label != 'Unknown Whale':
            base_confidence += 0.1
        
        # Exchange factor
        if activity.exchange in ['Binance', 'Coinbase', 'Kraken']:
            base_confidence += 0.05
        
        return min(base_confidence, 0.95)
    
    def _calculate_pattern_consistency(self, wallet_address: str) -> float:
        """Calculate how consistent whale's patterns are"""
        if wallet_address not in self.whale_patterns:
            return 0.0
        
        patterns = self.whale_patterns[wallet_address]
        if len(patterns) < 3:
            return 0.0
        
        # Analyze consistency in timing, size, and direction
        recent_patterns = patterns[-10:]  # Last 10 activities
        
        # Time consistency (similar times of day)
        times = [p['timestamp'].hour for p in recent_patterns]
        time_std = np.std(times)
        time_consistency = max(0, 1 - time_std / 12)  # Normalize by 12 hours
        
        # Size consistency
        sizes = [p['usd_value'] for p in recent_patterns]
        size_cv = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 1
        size_consistency = max(0, 1 - size_cv)
        
        # Direction consistency
        directions = [p['type'] for p in recent_patterns]
        direction_consistency = max(directions.count('buy'), directions.count('sell')) / len(directions)
        
        return (time_consistency + size_consistency + direction_consistency) / 3
    
    def _predict_whale_direction(self, activity: WhaleActivity) -> str:
        """Predict whale's next trading direction"""
        if activity.wallet_address in self.whale_patterns:
            recent_activities = self.whale_patterns[activity.wallet_address][-5:]
            
            # Look for patterns
            buy_count = sum(1 for a in recent_activities if a['type'] == 'buy')
            sell_count = sum(1 for a in recent_activities if a['type'] == 'sell')
            
            # If whale has been accumulating, likely to continue
            if buy_count > sell_count * 2:
                return 'buy'
            elif sell_count > buy_count * 2:
                return 'sell'
        
        # Default prediction based on current activity
        if activity.transaction_type == 'buy':
            return 'buy'  # Whale accumulating
        elif activity.transaction_type == 'sell':
            return 'sell'  # Whale distributing
        else:  # transfer
            # Transfers to exchanges often precede selling
            if activity.exchange:
                return 'sell'
            else:
                return 'buy'  # Cold storage = accumulation
    
    def _estimate_execution_window(self, activity: WhaleActivity) -> int:
        """Estimate time window for whale's next move"""
        base_window = 3600  # 1 hour default
        
        if activity.is_otc:
            return base_window * 2  # OTC trades take longer
        elif activity.usd_value > 50000000:
            return base_window // 2  # Large trades execute faster
        elif activity.exchange:
            return base_window // 3  # Exchange trades are faster
        
        return base_window
    
    def _calculate_risk_reward(self, activity: WhaleActivity) -> float:
        """Calculate risk/reward ratio for front-running"""
        base_ratio = 2.0
        
        # Higher volume = better risk/reward
        if activity.usd_value > 100000000:  # $100M+
            return 4.0
        elif activity.usd_value > 50000000:  # $50M+
            return 3.5
        elif activity.usd_value > 20000000:  # $20M+
            return 3.0
        elif activity.usd_value > 10000000:  # $10M+
            return 2.5
        
        return base_ratio
    
    def _get_whale_success_rate(self, wallet_address: str) -> float:
        """Get historical success rate for whale"""
        if wallet_address not in self.whale_success_rates:
            # Default success rate for unknown whales
            return 0.65
        
        return self.whale_success_rates[wallet_address]
    
    def _update_whale_patterns(self, activity: WhaleActivity):
        """Update whale pattern tracking"""
        pattern_record = {
            'timestamp': activity.timestamp,
            'type': activity.transaction_type,
            'asset': activity.asset,
            'usd_value': activity.usd_value,
            'exchange': activity.exchange,
            'is_otc': activity.is_otc
        }
        
        self.whale_patterns[activity.wallet_address].append(pattern_record)
        
        # Keep only recent patterns (last 50)
        if len(self.whale_patterns[activity.wallet_address]) > 50:
            self.whale_patterns[activity.wallet_address] = self.whale_patterns[activity.wallet_address][-50:]
    
    async def detect_front_running_opportunities(self, market_data: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Detect front-running opportunities from market data"""
        try:
            opportunities = []
            
            for symbol, data in market_data.items():
                prices = data.get('prices', [])
                volumes = data.get('volumes', [])
                volatility = data.get('volatility', 0.02)
                
                # Analyze for front-running patterns
                if len(prices) >= 10:
                    # Look for unusual price/volume patterns that suggest whale activity
                    recent_moves = []
                    for i in range(max(0, len(prices)-5), len(prices)):
                        if i > 0:
                            move = (prices[i] - prices[i-1]) / prices[i-1]
                            recent_moves.append(move)
                    
                    if len(recent_moves) >= 3:
                        avg_move = np.mean(recent_moves)
                        move_consistency = 1 - np.std(recent_moves) / (abs(avg_move) + 0.001)
                        
                        # Check for unusual volume if available
                        volume_spike = 1.0
                        if volumes and len(volumes) >= 5:
                            recent_vol = np.mean(volumes[-3:])
                            historical_vol = np.mean(volumes[-10:-3])
                            if historical_vol > 0:
                                volume_spike = recent_vol / historical_vol
                        
                        # Detect front-running opportunity
                        if move_consistency > 0.7 and abs(avg_move) > 0.01 and volume_spike > 1.5:
                            opportunities.append({
                                'symbol': symbol,
                                'direction': 'buy' if avg_move > 0 else 'sell',
                                'confidence': min(0.9, move_consistency * 0.7 + (volume_spike - 1) * 0.1),
                                'estimated_impact': abs(avg_move) * 2,
                                'time_window': 300,  # 5 minutes
                                'whale_activity_detected': True,
                                'volume_spike': volume_spike,
                                'price_momentum': avg_move
                            })
            
            # Sort by confidence
            opportunities.sort(key=lambda x: x['confidence'], reverse=True)
            return opportunities
            
        except Exception as e:
            logger.error(f"Front-running detection error: {e}")
            return []

class CopycatTrader:
    """Mirror best-performing wallets in real-time"""
    
    def __init__(self):
        self.tracked_wallets = {}
        self.wallet_rankings = {}
        self.copycat_signals = []
        self.performance_threshold = 0.15  # 15% minimum performance
        
        # Elite wallets to track (known successful traders)
        self.elite_wallets = {
            '1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2': {'label': 'Legendary Trader A', 'track_since': '2023-01-01'},
            '3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy': {'label': 'Institutional Whale', 'track_since': '2022-06-01'},
            'bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh': {'label': 'DeFi Master', 'track_since': '2023-03-01'}
        }
        
    async def analyze_wallet_performance(self, whale_activities: List[WhaleActivity]) -> Dict[str, Any]:
        """Analyze performance of tracked wallets"""
        if not ALLOW_SIMULATED_FEATURES:
            return {}
        performance_data = {}
        
        for activity in whale_activities:
            wallet = activity.wallet_address
            
            if wallet not in self.tracked_wallets:
                self.tracked_wallets[wallet] = {
                    'trades': [],
                    'total_pnl': 0.0,
                    'win_rate': 0.0,
                    'avg_hold_time': 0,
                    'risk_score': 0.5,
                    'last_activity': activity.timestamp
                }
            
            # Simulate performance calculation
            simulated_pnl = self._simulate_trade_pnl(activity)
            
            self.tracked_wallets[wallet]['trades'].append({
                'timestamp': activity.timestamp,
                'asset': activity.asset,
                'type': activity.transaction_type,
                'size': activity.usd_value,
                'pnl': simulated_pnl
            })
            
            # Update performance metrics
            self._update_wallet_metrics(wallet)
            
            performance_data[wallet] = {
                'total_pnl_pct': self.tracked_wallets[wallet]['total_pnl'],
                'win_rate': self.tracked_wallets[wallet]['win_rate'],
                'trade_count': len(self.tracked_wallets[wallet]['trades']),
                'avg_hold_time': self.tracked_wallets[wallet]['avg_hold_time'],
                'risk_score': self.tracked_wallets[wallet]['risk_score'],
                'label': activity.wallet_label or 'Unknown'
            }
        
        return performance_data
    
    def _simulate_trade_pnl(self, activity: WhaleActivity) -> float:
        """Simulate P&L for whale trade (replace with real tracking)"""
        if not ALLOW_SIMULATED_FEATURES:
            return 0.0
        # Simulate based on market conditions and whale success patterns
        base_return = np.random.normal(0.05, 0.15)  # 5% average with 15% volatility
        
        # Adjust based on wallet reputation
        if activity.wallet_address in self.elite_wallets:
            base_return += 0.03  # Elite wallets perform better
        
        # Adjust based on trade size (larger trades often more informed)
        if activity.usd_value > 50000000:
            base_return += 0.02
        elif activity.usd_value > 10000000:
            base_return += 0.01
        
        return base_return
    
    def _update_wallet_metrics(self, wallet: str):
        """Update performance metrics for wallet"""
        trades = self.tracked_wallets[wallet]['trades']
        
        if len(trades) < 2:
            return
        
        # Calculate total P&L
        total_pnl = sum(trade['pnl'] for trade in trades)
        self.tracked_wallets[wallet]['total_pnl'] = total_pnl
        
        # Calculate win rate
        winning_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades)
        self.tracked_wallets[wallet]['win_rate'] = win_rate
        
        # Calculate average hold time (simulated)
        avg_hold_time = 0
        if ALLOW_SIMULATED_FEATURES:
            avg_hold_time = np.random.randint(3600, 86400)  # 1-24 hours
        self.tracked_wallets[wallet]['avg_hold_time'] = avg_hold_time
        
        # Calculate risk score
        pnl_values = [t['pnl'] for t in trades]
        volatility = np.std(pnl_values)
        risk_score = min(1.0, volatility / 0.2)  # Normalize volatility
        self.tracked_wallets[wallet]['risk_score'] = risk_score
    
    async def generate_copycat_signals(self) -> List[CopycatSignal]:
        """Generate signals to copy successful wallets"""
        signals = []
        
        # Rank wallets by performance
        ranked_wallets = self._rank_wallets_by_performance()
        
        # Generate copycat signals for top performers
        for wallet_addr, metrics in ranked_wallets[:5]:  # Top 5 wallets
            if metrics['total_pnl'] > self.performance_threshold and metrics['win_rate'] > 0.6:
                
                # Simulate recent activity to copy
                recent_trades = self.tracked_wallets[wallet_addr]['trades'][-3:]
                
                for trade in recent_trades:
                    if trade['timestamp'] > datetime.now() - timedelta(hours=6):  # Recent activity
                        
                        # Determine position size based on wallet performance
                        position_size = self._calculate_copycat_position_size(metrics)
                        
                        signal = CopycatSignal(
                            target_wallet=wallet_addr,
                            wallet_performance=metrics['total_pnl'],
                            action=trade['type'],
                            asset=trade['asset'],
                            position_size=position_size,
                            confidence=min(0.9, metrics['win_rate'] + 0.1),
                            wallet_win_rate=metrics['win_rate'],
                            avg_hold_time=metrics['avg_hold_time']
                        )
                        signals.append(signal)
        
        # Sort by confidence and performance
        signals.sort(key=lambda s: s.confidence * s.wallet_performance, reverse=True)
        
        self.copycat_signals = signals[:10]  # Keep top 10 signals
        return self.copycat_signals
    
    def _rank_wallets_by_performance(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Rank wallets by performance metrics"""
        wallet_scores = []
        
        for wallet_addr, data in self.tracked_wallets.items():
            if len(data['trades']) < 5:  # Need minimum trade history
                continue
            
            # Calculate composite score
            pnl_score = data['total_pnl'] * 2  # Weight P&L heavily
            win_rate_score = data['win_rate'] * 1.5
            risk_adjusted_score = pnl_score / (1 + data['risk_score'])
            
            composite_score = pnl_score + win_rate_score + risk_adjusted_score
            
            wallet_scores.append((wallet_addr, {
                'total_pnl': data['total_pnl'],
                'win_rate': data['win_rate'],
                'trade_count': len(data['trades']),
                'risk_score': data['risk_score'],
                'composite_score': composite_score,
                'avg_hold_time': data['avg_hold_time']
            }))
        
        # Sort by composite score
        wallet_scores.sort(key=lambda x: x[1]['composite_score'], reverse=True)
        return wallet_scores
    
    def _calculate_copycat_position_size(self, wallet_metrics: Dict[str, Any]) -> float:
        """Calculate position size for copying wallet"""
        base_size = 0.05  # 5% of portfolio
        
        # Adjust based on wallet performance
        performance_multiplier = min(2.0, 1 + wallet_metrics['total_pnl'])
        
        # Adjust based on win rate
        win_rate_multiplier = wallet_metrics['win_rate'] * 1.5
        
        # Adjust based on risk
        risk_adjustment = 1 / (1 + wallet_metrics['risk_score'])
        
        final_size = base_size * performance_multiplier * win_rate_multiplier * risk_adjustment
        
        return min(0.15, final_size)  # Cap at 15% of portfolio
    
    def get_top_wallets_to_copy(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get top wallets recommended for copying"""
        ranked_wallets = self._rank_wallets_by_performance()
        
        top_wallets = []
        for wallet_addr, metrics in ranked_wallets[:top_n]:
            wallet_info = {
                'address': wallet_addr,
                'label': self.elite_wallets.get(wallet_addr, {}).get('label', 'Unknown Trader'),
                'performance': metrics['total_pnl'],
                'win_rate': metrics['win_rate'],
                'trade_count': metrics['trade_count'],
                'risk_score': metrics['risk_score'],
                'recommended_allocation': self._calculate_copycat_position_size(metrics)
            }
            top_wallets.append(wallet_info)
        
        return top_wallets
    
    async def analyze_successful_wallets(self, timeframe: str = '1h', min_success_rate: float = 0.75) -> List[Dict[str, Any]]:
        """Analyze successful wallets for copying"""
        try:
            successful_signals = []
            
            # Analyze tracked wallets
            for wallet_addr, data in self.tracked_wallets.items():
                if data['win_rate'] >= min_success_rate and len(data['trades']) >= 5:
                    # Recent successful pattern
                    recent_trades = [t for t in data['trades'] if t['timestamp'] > datetime.now() - timedelta(hours=24)]
                    
                    if recent_trades:
                        latest_trade = recent_trades[-1]
                        
                        successful_signals.append({
                            'symbol': latest_trade['asset'] + '/USDT',
                            'wallet': wallet_addr,
                            'action': latest_trade['type'],
                            'confidence': data['win_rate'],
                            'wallet_performance': data['total_pnl'],
                            'success_pattern': 'recent_winner',
                            'estimated_hold_time': data['avg_hold_time']
                        })
            
            # Sort by confidence and performance
            successful_signals.sort(key=lambda x: x['confidence'] * x['wallet_performance'], reverse=True)
            return successful_signals[:10]  # Top 10 signals
            
        except Exception as e:
            logger.error(f"Successful wallet analysis error: {e}")
            return []

    async def predict_whale_movements(self) -> List[Dict[str, Any]]:
        """Predict whale movement patterns"""
        try:
            predictions = []
            
            # Analyze each tracked whale
            for wallet_addr, data in self.tracked_wallets.items():
                if len(data['trades']) >= 3:
                    recent_trades = data['trades'][-5:]  # Last 5 trades
                    
                    # Look for patterns
                    buy_pattern = sum(1 for t in recent_trades if t['type'] == 'buy')
                    sell_pattern = sum(1 for t in recent_trades if t['type'] == 'sell')
                    
                    if buy_pattern > sell_pattern * 2:  # Accumulation pattern
                        predictions.append({
                            'wallet': wallet_addr,
                            'predicted_action': 'continue_accumulation',
                            'confidence': min(0.85, buy_pattern / len(recent_trades)),
                            'estimated_timeframe': '6-12h',
                            'pattern_type': 'accumulation_phase',
                            'assets': list(set(t['asset'] for t in recent_trades))
                        })
                    elif sell_pattern > buy_pattern * 2:  # Distribution pattern
                        predictions.append({
                            'wallet': wallet_addr,
                            'predicted_action': 'continue_distribution',
                            'confidence': min(0.85, sell_pattern / len(recent_trades)),
                            'estimated_timeframe': '3-6h',
                            'pattern_type': 'distribution_phase',
                            'assets': list(set(t['asset'] for t in recent_trades))
                        })
            
            return predictions
        except Exception as e:
            logger.error(f"Whale movement prediction error: {e}")
            return []
    
    def _predict_whale_direction(self, activity: WhaleActivity) -> str:
        """Predict whale's next trading direction"""
        if activity.wallet_address in self.whale_patterns:
            recent_activities = self.whale_patterns[activity.wallet_address][-5:]
            
            # Look for patterns
            buy_count = sum(1 for a in recent_activities if a['type'] == 'buy')
            sell_count = sum(1 for a in recent_activities if a['type'] == 'sell')
            
            # If whale has been accumulating, likely to continue
            if buy_count > sell_count * 2:
                return 'buy'
            elif sell_count > buy_count * 2:
                return 'sell'
        
        # Default prediction based on current activity
        if activity.transaction_type == 'buy':
            return 'buy'  # Whale accumulating
        elif activity.transaction_type == 'sell':
            return 'sell'  # Whale distributing
        else:  # transfer
            # Transfers to exchanges often precede selling
            if activity.exchange:
                return 'sell'
            else:
                return 'buy'  # Cold storage = accumulation
    
    def _estimate_execution_window(self, activity: WhaleActivity) -> int:
        """Estimate time window for whale's next move"""
        base_window = 3600  # 1 hour default
        
        if activity.is_otc:
            return base_window * 2  # OTC trades take longer
        elif activity.usd_value > 50000000:
            return base_window // 2  # Large trades execute faster
        elif activity.exchange:
            return base_window // 3  # Exchange trades are faster
        
        return base_window
    
    def _calculate_risk_reward(self, activity: WhaleActivity) -> float:
        """Calculate risk/reward ratio for front-running"""
        base_ratio = 2.0
        
        # Higher volume = better risk/reward
        if activity.usd_value > 100000000:  # $100M+
            return 4.0
        elif activity.usd_value > 50000000:  # $50M+
            return 3.5
        elif activity.usd_value > 20000000:  # $20M+
            return 3.0
        elif activity.usd_value > 10000000:  # $10M+
            return 2.5
        
        return base_ratio
    
    def _get_whale_success_rate(self, wallet_address: str) -> float:
        """Get historical success rate for whale"""
        if wallet_address not in self.whale_success_rates:
            # Default success rate for unknown whales
            return 0.65
        
        return self.whale_success_rates[wallet_address]
    
    def _update_whale_patterns(self, activity: WhaleActivity):
        """Update whale pattern tracking"""
        pattern_record = {
            'timestamp': activity.timestamp,
            'type': activity.transaction_type,
            'asset': activity.asset,
            'usd_value': activity.usd_value,
            'exchange': activity.exchange,
            'is_otc': activity.is_otc
        }
        
        self.whale_patterns[activity.wallet_address].append(pattern_record)
        
        # Keep only recent patterns (last 50)
        if len(self.whale_patterns[activity.wallet_address]) > 50:
            self.whale_patterns[activity.wallet_address] = self.whale_patterns[activity.wallet_address][-50:]
