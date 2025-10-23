#!/usr/bin/env python3
"""
💧 PROFESSIONAL LIQUIDITY & ORDER FLOW ANALYSIS
Advanced market microstructure and liquidity analysis

FEATURES:
✅ Order Book Analysis & Imbalance Detection
✅ Liquidity Heat Maps
✅ Volume Profile Analysis
✅ Market Depth Analysis
✅ Bid/Ask Spread Tracking
✅ Large Order Detection
✅ Absorption & Exhaustion Patterns
✅ Delta Analysis (Buy vs Sell Volume)
✅ Footprint Charts
✅ Market Maker Detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum

class LiquidityCondition(Enum):
    """Market liquidity conditions"""
    DEEP = "deep"
    NORMAL = "normal"
    THIN = "thin"
    ILLIQUID = "illiquid"

class OrderFlowPattern(Enum):
    """Order flow patterns"""
    ABSORPTION = "absorption"          # Large orders being absorbed
    EXHAUSTION = "exhaustion"         # Buying/selling exhaustion
    ACCUMULATION = "accumulation"     # Smart money accumulating
    DISTRIBUTION = "distribution"     # Smart money distributing
    STOP_HUNT = "stop_hunt"          # Stop loss hunting
    BREAKOUT = "breakout"            # Real breakout with volume

@dataclass
class OrderBookSnapshot:
    """Order book state at a moment"""
    timestamp: datetime
    best_bid: float
    best_ask: float
    bid_volume: float
    ask_volume: float
    spread: float
    spread_pct: float
    depth_10_bid: float  # Volume within 10 levels
    depth_10_ask: float
    imbalance: float     # Bid/Ask imbalance
    large_orders_bid: List[Tuple[float, float]]  # (price, size)
    large_orders_ask: List[Tuple[float, float]]

@dataclass
class VolumeProfile:
    """Volume profile analysis"""
    poc: float           # Point of Control (highest volume price)
    value_area_high: float
    value_area_low: float
    volume_nodes: List[Tuple[float, float]]  # (price, volume)
    total_volume: float
    buy_volume: float
    sell_volume: float
    delta: float         # Buy - Sell volume
    cumulative_delta: float

@dataclass
class LiquidityZone:
    """Liquidity concentration zone"""
    price_level: float
    liquidity_score: float
    zone_type: str      # support, resistance, magnet
    strength: float     # 0-100
    touches: int        # How many times tested

class OrderFlowAnalyzer:
    """Professional order flow analysis"""
    
    def __init__(self):
        self.order_book_history = deque(maxlen=1000)
        self.volume_profiles = {}
        self.liquidity_zones = {}
        self.footprint_data = defaultdict(list)
        self.delta_history = deque(maxlen=500)
        
    async def analyze_order_flow(self, symbol: str, order_book: Dict, 
                                trades: List[Dict]) -> Dict:
        """Complete order flow analysis"""
        
        # Take order book snapshot
        snapshot = await self._create_order_book_snapshot(order_book)
        self.order_book_history.append(snapshot)
        
        # Analyze volume profile
        volume_profile = await self._analyze_volume_profile(trades)
        self.volume_profiles[symbol] = volume_profile
        
        # Detect order flow patterns
        patterns = await self._detect_order_flow_patterns(snapshot, trades)
        
        # Identify liquidity zones
        zones = await self._identify_liquidity_zones(symbol, volume_profile)
        self.liquidity_zones[symbol] = zones
        
        # Calculate market delta
        delta_analysis = await self._analyze_delta(trades)
        
        # Detect large orders and icebergs
        large_orders = await self._detect_large_orders(order_book, trades)
        
        # Market maker analysis
        mm_analysis = await self._analyze_market_maker_activity(order_book, trades)
        
        # Liquidity condition assessment
        liquidity_condition = self._assess_liquidity_condition(snapshot)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'order_book': {
                'spread': snapshot.spread,
                'spread_pct': snapshot.spread_pct,
                'imbalance': snapshot.imbalance,
                'bid_depth': snapshot.depth_10_bid,
                'ask_depth': snapshot.depth_10_ask
            },
            'volume_profile': {
                'poc': volume_profile.poc,
                'value_area': (volume_profile.value_area_low, volume_profile.value_area_high),
                'delta': volume_profile.delta,
                'cumulative_delta': volume_profile.cumulative_delta
            },
            'patterns': patterns,
            'liquidity_zones': zones,
            'delta_analysis': delta_analysis,
            'large_orders': large_orders,
            'market_maker': mm_analysis,
            'liquidity_condition': liquidity_condition.value,
            'tradeable': self._is_tradeable(liquidity_condition, snapshot)
        }
    
    async def _create_order_book_snapshot(self, order_book: Dict) -> OrderBookSnapshot:
        """Create snapshot of current order book"""
        
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        if not bids or not asks:
            # Return empty snapshot
            return OrderBookSnapshot(
                timestamp=datetime.now(),
                best_bid=0, best_ask=0,
                bid_volume=0, ask_volume=0,
                spread=0, spread_pct=0,
                depth_10_bid=0, depth_10_ask=0,
                imbalance=0,
                large_orders_bid=[], large_orders_ask=[]
            )
        
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid) * 100 if best_bid > 0 else 0
        
        # Calculate depth (first 10 levels)
        bid_depth = sum(float(b[1]) for b in bids[:10])
        ask_depth = sum(float(a[1]) for a in asks[:10])
        
        # Calculate imbalance
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
        
        # Find large orders (top 5% by size)
        all_bid_sizes = [float(b[1]) for b in bids]
        all_ask_sizes = [float(a[1]) for a in asks]
        
        bid_threshold = np.percentile(all_bid_sizes, 95) if all_bid_sizes else 0
        ask_threshold = np.percentile(all_ask_sizes, 95) if all_ask_sizes else 0
        
        large_bids = [(float(b[0]), float(b[1])) for b in bids if float(b[1]) > bid_threshold]
        large_asks = [(float(a[0]), float(a[1])) for a in asks if float(a[1]) > ask_threshold]
        
        return OrderBookSnapshot(
            timestamp=datetime.now(),
            best_bid=best_bid,
            best_ask=best_ask,
            bid_volume=sum(float(b[1]) for b in bids[:1]),
            ask_volume=sum(float(a[1]) for a in asks[:1]),
            spread=spread,
            spread_pct=spread_pct,
            depth_10_bid=bid_depth,
            depth_10_ask=ask_depth,
            imbalance=imbalance,
            large_orders_bid=large_bids[:5],
            large_orders_ask=large_asks[:5]
        )
    
    async def _analyze_volume_profile(self, trades: List[Dict]) -> VolumeProfile:
        """Analyze volume profile from trades"""
        
        if not trades:
            return VolumeProfile(
                poc=0, value_area_high=0, value_area_low=0,
                volume_nodes=[], total_volume=0,
                buy_volume=0, sell_volume=0, delta=0, cumulative_delta=0
            )
        
        # Group trades by price level
        price_volumes = defaultdict(float)
        buy_volumes = defaultdict(float)
        sell_volumes = defaultdict(float)
        
        for trade in trades:
            price = float(trade['price'])
            volume = float(trade['amount'])
            side = trade.get('side', 'buy')
            
            # Round price to nearest tick
            price = round(price, 2)
            
            price_volumes[price] += volume
            if side == 'buy':
                buy_volumes[price] += volume
            else:
                sell_volumes[price] += volume
        
        # Find Point of Control (highest volume price)
        if price_volumes:
            poc = max(price_volumes.keys(), key=lambda x: price_volumes[x])
        else:
            poc = 0
        
        # Calculate value area (70% of volume)
        sorted_prices = sorted(price_volumes.items(), key=lambda x: x[1], reverse=True)
        total_volume = sum(v for _, v in sorted_prices)
        
        value_area_volume = 0
        value_area_prices = []
        target_volume = total_volume * 0.7
        
        for price, volume in sorted_prices:
            value_area_volume += volume
            value_area_prices.append(price)
            if value_area_volume >= target_volume:
                break
        
        value_area_high = max(value_area_prices) if value_area_prices else 0
        value_area_low = min(value_area_prices) if value_area_prices else 0
        
        # Calculate delta
        total_buy = sum(buy_volumes.values())
        total_sell = sum(sell_volumes.values())
        delta = total_buy - total_sell
        
        # Update cumulative delta
        self.delta_history.append(delta)
        cumulative_delta = sum(self.delta_history)
        
        return VolumeProfile(
            poc=poc,
            value_area_high=value_area_high,
            value_area_low=value_area_low,
            volume_nodes=sorted_prices[:10],
            total_volume=total_volume,
            buy_volume=total_buy,
            sell_volume=total_sell,
            delta=delta,
            cumulative_delta=cumulative_delta
        )
    
    async def _detect_order_flow_patterns(self, snapshot: OrderBookSnapshot, 
                                        trades: List[Dict]) -> List[str]:
        """Detect order flow patterns"""
        
        patterns = []
        
        # Check for absorption (large orders being absorbed without price movement)
        if len(self.order_book_history) > 10:
            recent_spreads = [s.spread_pct for s in list(self.order_book_history)[-10:]]
            spread_stability = np.std(recent_spreads)
            
            large_volume = sum(float(t['amount']) for t in trades[-100:]) if len(trades) > 100 else 0
            
            if spread_stability < 0.1 and large_volume > 10000:
                patterns.append(OrderFlowPattern.ABSORPTION.value)
        
        # Check for exhaustion (decreasing volume at extremes)
        if len(trades) > 50:
            recent_volumes = [float(t['amount']) for t in trades[-50:]]
            volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
            
            if volume_trend < -100:  # Declining volume
                patterns.append(OrderFlowPattern.EXHAUSTION.value)
        
        # Check for stop hunts (quick spike with reversal)
        if len(self.order_book_history) > 5:
            recent_imbalances = [s.imbalance for s in list(self.order_book_history)[-5:]]
            imbalance_reversal = (max(recent_imbalances) - min(recent_imbalances)) > 0.5
            
            if imbalance_reversal:
                patterns.append(OrderFlowPattern.STOP_HUNT.value)
        
        # Check for accumulation/distribution
        if snapshot.imbalance > 0.3 and snapshot.depth_10_bid > snapshot.depth_10_ask * 2:
            patterns.append(OrderFlowPattern.ACCUMULATION.value)
        elif snapshot.imbalance < -0.3 and snapshot.depth_10_ask > snapshot.depth_10_bid * 2:
            patterns.append(OrderFlowPattern.DISTRIBUTION.value)
        
        return patterns
    
    async def _identify_liquidity_zones(self, symbol: str, 
                                      volume_profile: VolumeProfile) -> List[LiquidityZone]:
        """Identify key liquidity zones"""
        
        zones = []
        
        # POC is always a liquidity magnet
        if volume_profile.poc > 0:
            zones.append(LiquidityZone(
                price_level=volume_profile.poc,
                liquidity_score=100,
                zone_type='magnet',
                strength=100,
                touches=0
            ))
        
        # Value area boundaries are support/resistance
        if volume_profile.value_area_high > 0:
            zones.append(LiquidityZone(
                price_level=volume_profile.value_area_high,
                liquidity_score=80,
                zone_type='resistance',
                strength=80,
                touches=0
            ))
        
        if volume_profile.value_area_low > 0:
            zones.append(LiquidityZone(
                price_level=volume_profile.value_area_low,
                liquidity_score=80,
                zone_type='support',
                strength=80,
                touches=0
            ))
        
        # High volume nodes
        for price, volume in volume_profile.volume_nodes[:5]:
            if volume > volume_profile.total_volume * 0.1:  # >10% of total volume
                zones.append(LiquidityZone(
                    price_level=price,
                    liquidity_score=60,
                    zone_type='magnet',
                    strength=60,
                    touches=0
                ))
        
        return zones
    
    async def _analyze_delta(self, trades: List[Dict]) -> Dict:
        """Analyze buy/sell volume delta"""
        
        if not trades:
            return {'delta': 0, 'cumulative_delta': 0, 'delta_trend': 'neutral'}
        
        # Calculate delta for recent trades
        buy_volume = sum(float(t['amount']) for t in trades if t.get('side') == 'buy')
        sell_volume = sum(float(t['amount']) for t in trades if t.get('side') == 'sell')
        delta = buy_volume - sell_volume
        
        # Determine trend
        if len(self.delta_history) > 10:
            recent_deltas = list(self.delta_history)[-10:]
            delta_trend = np.polyfit(range(len(recent_deltas)), recent_deltas, 1)[0]
            
            if delta_trend > 100:
                trend = 'bullish'
            elif delta_trend < -100:
                trend = 'bearish'
            else:
                trend = 'neutral'
        else:
            trend = 'neutral'
        
        return {
            'delta': delta,
            'cumulative_delta': sum(self.delta_history) if self.delta_history else 0,
            'delta_trend': trend,
            'buy_pressure': buy_volume / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0.5
        }
    
    async def _detect_large_orders(self, order_book: Dict, trades: List[Dict]) -> Dict:
        """Detect large and iceberg orders"""
        
        large_orders = {
            'detected': [],
            'potential_icebergs': [],
            'whale_activity': False
        }
        
        if not trades:
            return large_orders
        
        # Calculate average trade size
        trade_sizes = [float(t['amount']) for t in trades]
        if trade_sizes:
            avg_size = np.mean(trade_sizes)
            large_threshold = avg_size * 10  # 10x average
            
            # Find large trades
            for trade in trades[-50:]:  # Last 50 trades
                if float(trade['amount']) > large_threshold:
                    large_orders['detected'].append({
                        'price': trade['price'],
                        'size': trade['amount'],
                        'side': trade.get('side', 'unknown'),
                        'timestamp': trade.get('timestamp')
                    })
            
            # Detect iceberg orders (repeated trades at same price)
            price_counts = defaultdict(list)
            for trade in trades[-100:]:
                price_counts[trade['price']].append(float(trade['amount']))
            
            for price, sizes in price_counts.items():
                if len(sizes) > 5 and np.std(sizes) < avg_size * 0.2:
                    large_orders['potential_icebergs'].append({
                        'price': price,
                        'estimated_size': sum(sizes),
                        'chunks': len(sizes)
                    })
            
            # Check for whale activity
            if len(large_orders['detected']) > 3:
                large_orders['whale_activity'] = True
        
        return large_orders
    
    async def _analyze_market_maker_activity(self, order_book: Dict, 
                                           trades: List[Dict]) -> Dict:
        """Analyze market maker presence and behavior"""
        
        mm_analysis = {
            'presence_detected': False,
            'mm_spread': 0,
            'liquidity_provision': 'normal',
            'quote_stuffing': False
        }
        
        if not order_book.get('bids') or not order_book.get('asks'):
            return mm_analysis
        
        # Check for tight, symmetric quotes (market maker signature)
        bids = order_book['bids'][:5]
        asks = order_book['asks'][:5]
        
        bid_sizes = [float(b[1]) for b in bids]
        ask_sizes = [float(a[1]) for a in asks]
        
        # Market makers often place similar sizes on both sides
        if bid_sizes and ask_sizes:
            size_symmetry = 1 - abs(np.mean(bid_sizes) - np.mean(ask_sizes)) / max(np.mean(bid_sizes), np.mean(ask_sizes))
            
            if size_symmetry > 0.8:
                mm_analysis['presence_detected'] = True
                
                # Calculate MM spread
                mm_analysis['mm_spread'] = float(asks[0][0]) - float(bids[0][0])
                
                # Assess liquidity provision
                total_liquidity = sum(bid_sizes) + sum(ask_sizes)
                if total_liquidity > 100000:
                    mm_analysis['liquidity_provision'] = 'high'
                elif total_liquidity < 10000:
                    mm_analysis['liquidity_provision'] = 'low'
        
        # Check for quote stuffing (rapid order placement/cancellation)
        if len(self.order_book_history) > 10:
            recent_spreads = [s.spread for s in list(self.order_book_history)[-10:]]
            spread_changes = sum(1 for i in range(1, len(recent_spreads)) 
                               if abs(recent_spreads[i] - recent_spreads[i-1]) > 0)
            
            if spread_changes > 7:  # High frequency of changes
                mm_analysis['quote_stuffing'] = True
        
        return mm_analysis
    
    def _assess_liquidity_condition(self, snapshot: OrderBookSnapshot) -> LiquidityCondition:
        """Assess overall liquidity condition"""
        
        # Based on spread and depth
        if snapshot.spread_pct < 0.1 and snapshot.depth_10_bid + snapshot.depth_10_ask > 100000:
            return LiquidityCondition.DEEP
        elif snapshot.spread_pct < 0.3 and snapshot.depth_10_bid + snapshot.depth_10_ask > 10000:
            return LiquidityCondition.NORMAL
        elif snapshot.spread_pct < 0.5:
            return LiquidityCondition.THIN
        else:
            return LiquidityCondition.ILLIQUID
    
    def _is_tradeable(self, condition: LiquidityCondition, 
                     snapshot: OrderBookSnapshot) -> bool:
        """Determine if market is tradeable"""
        
        # Don't trade illiquid markets
        if condition == LiquidityCondition.ILLIQUID:
            return False
        
        # Don't trade if spread is too wide
        if snapshot.spread_pct > 0.5:
            return False
        
        # Don't trade if order book is too imbalanced
        if abs(snapshot.imbalance) > 0.8:
            return False
        
        return True


class FootprintChart:
    """Professional footprint chart analysis"""
    
    def __init__(self):
        self.footprints = defaultdict(lambda: defaultdict(dict))
        
    async def create_footprint(self, symbol: str, timeframe: str, 
                              trades: List[Dict]) -> Dict:
        """Create footprint chart data"""
        
        footprint = {
            'symbol': symbol,
            'timeframe': timeframe,
            'bars': []
        }
        
        # Group trades by time period
        period_trades = defaultdict(list)
        
        for trade in trades:
            # Round timestamp to timeframe
            timestamp = trade.get('timestamp', datetime.now())
            period_key = self._round_to_timeframe(timestamp, timeframe)
            period_trades[period_key].append(trade)
        
        # Create footprint for each period
        for period, trades_in_period in sorted(period_trades.items()):
            bar = await self._create_footprint_bar(trades_in_period)
            footprint['bars'].append({
                'time': period,
                'data': bar
            })
        
        return footprint
    
    async def _create_footprint_bar(self, trades: List[Dict]) -> Dict:
        """Create single footprint bar"""
        
        # Group by price level
        price_levels = defaultdict(lambda: {'bid_volume': 0, 'ask_volume': 0})
        
        for trade in trades:
            price = round(float(trade['price']), 2)
            volume = float(trade['amount'])
            side = trade.get('side', 'buy')
            
            if side == 'buy':
                price_levels[price]['ask_volume'] += volume
            else:
                price_levels[price]['bid_volume'] += volume
        
        # Calculate imbalances
        footprint_bar = {}
        for price, volumes in sorted(price_levels.items()):
            total = volumes['bid_volume'] + volumes['ask_volume']
            imbalance = (volumes['ask_volume'] - volumes['bid_volume']) / total if total > 0 else 0
            
            footprint_bar[price] = {
                'bid': volumes['bid_volume'],
                'ask': volumes['ask_volume'],
                'total': total,
                'imbalance': imbalance,
                'dominant': 'ask' if imbalance > 0 else 'bid'
            }
        
        return footprint_bar
    
    def _round_to_timeframe(self, timestamp: datetime, timeframe: str) -> datetime:
        """Round timestamp to timeframe"""
        
        if timeframe == '1m':
            return timestamp.replace(second=0, microsecond=0)
        elif timeframe == '5m':
            minute = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == '15m':
            minute = (timestamp.minute // 15) * 15
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == '1h':
            return timestamp.replace(minute=0, second=0, microsecond=0)
        else:
            return timestamp
