#!/usr/bin/env python3
"""
🔍 ADVANCED MARKET SCANNER
Elite market opportunity scanner with AI-powered analysis
"""

import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from collections import deque, defaultdict
from dataclasses import dataclass, field
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class MarketOpportunity:
    """Market trading opportunity"""
    symbol: str
    opportunity_type: str  # 'breakout', 'reversal', 'momentum', 'arbitrage'
    direction: str  # 'BUY', 'SELL'
    confidence: float
    urgency: float  # 0-1, how quickly to act
    expected_return: float
    risk_score: float
    time_horizon: str  # 'immediate', 'short', 'medium'
    reasoning: str
    technical_indicators: Dict
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ScanResults:
    """Market scan results"""
    opportunities: List[MarketOpportunity]
    market_regime: str
    volatility_index: float
    fear_greed_index: float
    total_opportunities: int
    high_confidence_count: int
    scan_timestamp: datetime = field(default_factory=datetime.now)

class AdvancedMarketScanner:
    """🔍 Elite market opportunity detection system"""
    
    def __init__(self):
        self.scan_history = deque(maxlen=500)
        self.opportunity_tracking = defaultdict(list)
        
        # Scanner components
        self.technical_scanner = TechnicalPatternScanner()
        self.momentum_scanner = MomentumBreakoutScanner()
        self.reversal_scanner = ReversalOpportunityScanner()
        self.arbitrage_scanner = ArbitrageOpportunityScanner()
        self.volume_scanner = VolumeAnomalyScanner()
        
        # Market analysis
        self.market_regime_detector = MarketRegimeDetector()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        
        # Opportunity filters
        self.quality_filters = OpportunityQualityFilters()
        self.risk_filters = RiskBasedFilters()
        
        # Tracking
        self.scan_count = 0
        self.opportunities_found = 0
        self.success_rate = 0.0
        
        print("🔍 Advanced Market Scanner initialized!")
        print("   📊 Features: Pattern detection, momentum analysis, arbitrage scanning")
        print("   🎯 Target: High-probability opportunities with 90%+ win rate")
    
    async def scan_market(self, symbols: List[str], market_data: Dict) -> ScanResults:
        """Comprehensive market scan for opportunities"""
        try:
            self.scan_count += 1
            all_opportunities = []
            
            print(f"   🔍 Scanning {len(symbols)} assets for opportunities...")
            
            # Parallel scanning across different opportunity types
            scan_tasks = []
            
            for symbol in symbols:
                symbol_data = market_data.get(symbol, {})
                if not symbol_data:
                    continue
                
                # Technical pattern opportunities
                scan_tasks.append(self._scan_technical_patterns(symbol, symbol_data))
                
                # Momentum opportunities
                scan_tasks.append(self._scan_momentum_breakouts(symbol, symbol_data))
                
                # Reversal opportunities
                scan_tasks.append(self._scan_reversal_patterns(symbol, symbol_data))
                
                # Volume anomalies
                scan_tasks.append(self._scan_volume_anomalies(symbol, symbol_data))
            
            # Execute all scans concurrently
            scan_results = await asyncio.gather(*scan_tasks, return_exceptions=True)
            
            # Collect valid opportunities
            for result in scan_results:
                if isinstance(result, list):
                    all_opportunities.extend(result)
                elif isinstance(result, MarketOpportunity):
                    all_opportunities.append(result)
            
            # Filter and rank opportunities
            filtered_opportunities = await self._filter_and_rank_opportunities(all_opportunities, market_data)
            
            # Analyze market conditions
            market_regime = await self.market_regime_detector.detect_regime(market_data)
            volatility_index = await self.volatility_analyzer.calculate_volatility_index(market_data)
            sentiment_score = await self.sentiment_analyzer.analyze_market_sentiment(market_data)
            
            # Create scan results
            results = ScanResults(
                opportunities=filtered_opportunities,
                market_regime=market_regime,
                volatility_index=volatility_index,
                fear_greed_index=sentiment_score,
                total_opportunities=len(filtered_opportunities),
                high_confidence_count=len([o for o in filtered_opportunities if o.confidence >= 0.8])
            )
            
            # Record scan
            self.scan_history.append(results)
            self.opportunities_found += len(filtered_opportunities)
            
            print(f"   ✅ Scan complete: Found {len(filtered_opportunities)} opportunities "
                  f"({results.high_confidence_count} high-confidence)")
            
            return results
            
        except Exception as e:
            logger.error(f"Market scan error: {e}")
            return ScanResults([], "unknown", 0.0, 0.5, 0, 0)
    
    async def _scan_technical_patterns(self, symbol: str, data: Dict) -> List[MarketOpportunity]:
        """Scan for technical pattern opportunities"""
        opportunities = []
        
        try:
            prices = data.get('prices', [])
            if len(prices) < 50:
                return opportunities
            
            # Scan for various patterns
            patterns = await self.technical_scanner.detect_patterns(symbol, prices)
            
            for pattern in patterns:
                if pattern['confidence'] >= 0.6:
                    opportunity = MarketOpportunity(
                        symbol=symbol,
                        opportunity_type='pattern',
                        direction=pattern['direction'],
                        confidence=pattern['confidence'],
                        urgency=pattern.get('urgency', 0.5),
                        expected_return=pattern.get('expected_return', 0.02),
                        risk_score=pattern.get('risk', 0.3),
                        time_horizon=pattern.get('timeframe', 'short'),
                        reasoning=f"Technical pattern: {pattern['pattern_name']}",
                        technical_indicators=pattern.get('indicators', {})
                    )
                    opportunities.append(opportunity)
            
        except Exception as e:
            logger.error(f"Technical pattern scan error for {symbol}: {e}")
        
        return opportunities
    
    async def _scan_momentum_breakouts(self, symbol: str, data: Dict) -> List[MarketOpportunity]:
        """Scan for momentum breakout opportunities"""
        opportunities = []
        
        try:
            breakouts = await self.momentum_scanner.detect_breakouts(symbol, data)
            
            for breakout in breakouts:
                if breakout['strength'] >= 0.7:
                    opportunity = MarketOpportunity(
                        symbol=symbol,
                        opportunity_type='breakout',
                        direction=breakout['direction'],
                        confidence=breakout['confidence'],
                        urgency=breakout.get('urgency', 0.8),  # Breakouts are urgent
                        expected_return=breakout.get('target_return', 0.03),
                        risk_score=breakout.get('risk', 0.4),
                        time_horizon='immediate',
                        reasoning=f"Momentum breakout: {breakout['breakout_type']}",
                        technical_indicators=breakout.get('indicators', {})
                    )
                    opportunities.append(opportunity)
            
        except Exception as e:
            logger.error(f"Momentum breakout scan error for {symbol}: {e}")
        
        return opportunities
    
    async def _scan_reversal_patterns(self, symbol: str, data: Dict) -> List[MarketOpportunity]:
        """Scan for reversal opportunities"""
        opportunities = []
        
        try:
            reversals = await self.reversal_scanner.detect_reversals(symbol, data)
            
            for reversal in reversals:
                if reversal['probability'] >= 0.6:
                    opportunity = MarketOpportunity(
                        symbol=symbol,
                        opportunity_type='reversal',
                        direction=reversal['direction'],
                        confidence=reversal['probability'],
                        urgency=reversal.get('urgency', 0.6),
                        expected_return=reversal.get('expected_return', 0.025),
                        risk_score=reversal.get('risk', 0.5),
                        time_horizon='medium',
                        reasoning=f"Reversal pattern: {reversal['pattern_type']}",
                        technical_indicators=reversal.get('indicators', {})
                    )
                    opportunities.append(opportunity)
            
        except Exception as e:
            logger.error(f"Reversal pattern scan error for {symbol}: {e}")
        
        return opportunities
    
    async def _scan_volume_anomalies(self, symbol: str, data: Dict) -> List[MarketOpportunity]:
        """Scan for volume-based opportunities"""
        opportunities = []
        
        try:
            anomalies = await self.volume_scanner.detect_anomalies(symbol, data)
            
            for anomaly in anomalies:
                if anomaly['significance'] >= 0.7:
                    opportunity = MarketOpportunity(
                        symbol=symbol,
                        opportunity_type='volume_anomaly',
                        direction=anomaly['direction'],
                        confidence=anomaly['confidence'],
                        urgency=anomaly.get('urgency', 0.7),
                        expected_return=anomaly.get('expected_return', 0.02),
                        risk_score=anomaly.get('risk', 0.3),
                        time_horizon='short',
                        reasoning=f"Volume anomaly: {anomaly['anomaly_type']}",
                        technical_indicators=anomaly.get('indicators', {})
                    )
                    opportunities.append(opportunity)
            
        except Exception as e:
            logger.error(f"Volume anomaly scan error for {symbol}: {e}")
        
        return opportunities
    
    async def _filter_and_rank_opportunities(self, opportunities: List[MarketOpportunity], market_data: Dict) -> List[MarketOpportunity]:
        """Filter and rank opportunities by quality"""
        try:
            if not opportunities:
                return []
            
            # Apply quality filters
            quality_filtered = await self.quality_filters.filter_opportunities(opportunities, market_data)
            
            # Apply risk filters
            risk_filtered = await self.risk_filters.filter_by_risk(quality_filtered, market_data)
            
            # Rank by combined score
            ranked_opportunities = self._rank_opportunities(risk_filtered)
            
            # Return top opportunities
            return ranked_opportunities[:20]  # Top 20 opportunities
            
        except Exception as e:
            logger.error(f"Opportunity filtering error: {e}")
            return opportunities[:10]  # Fallback to first 10
    
    def _rank_opportunities(self, opportunities: List[MarketOpportunity]) -> List[MarketOpportunity]:
        """Rank opportunities by combined score"""
        try:
            def calculate_score(opp: MarketOpportunity) -> float:
                # Multi-factor scoring
                confidence_score = opp.confidence * 0.3
                return_score = min(1.0, opp.expected_return * 20) * 0.25  # Cap at 5% return
                urgency_score = opp.urgency * 0.2
                risk_score = (1 - opp.risk_score) * 0.25  # Lower risk = higher score
                
                return confidence_score + return_score + urgency_score + risk_score
            
            # Sort by score descending
            return sorted(opportunities, key=calculate_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Opportunity ranking error: {e}")
            return opportunities
    
    def get_scan_statistics(self) -> Dict:
        """Get market scanner statistics"""
        try:
            recent_scans = list(self.scan_history)[-10:]  # Last 10 scans
            
            if not recent_scans:
                return {'scans': 0, 'avg_opportunities': 0, 'success_rate': 0.0}
            
            avg_opportunities = np.mean([scan.total_opportunities for scan in recent_scans])
            avg_high_confidence = np.mean([scan.high_confidence_count for scan in recent_scans])
            
            return {
                'total_scans': self.scan_count,
                'opportunities_found': self.opportunities_found,
                'avg_opportunities_per_scan': avg_opportunities,
                'avg_high_confidence_per_scan': avg_high_confidence,
                'success_rate': self.success_rate,
                'last_scan': recent_scans[-1].scan_timestamp if recent_scans else None
            }
            
        except Exception as e:
            logger.error(f"Scan statistics error: {e}")
            return {}

class TechnicalPatternScanner:
    """📈 Technical pattern detection scanner"""
    
    def __init__(self):
        self.patterns = {
            'double_bottom': self._detect_double_bottom,
            'double_top': self._detect_double_top,
            'head_shoulders': self._detect_head_shoulders,
            'triangle': self._detect_triangle,
            'flag': self._detect_flag_pattern,
            'cup_handle': self._detect_cup_handle,
            'ascending_triangle': self._detect_ascending_triangle,
            'descending_triangle': self._detect_descending_triangle
        }
    
    async def detect_patterns(self, symbol: str, prices: List[float]) -> List[Dict]:
        """Detect technical patterns in price data"""
        detected_patterns = []
        
        try:
            if len(prices) < 30:
                return detected_patterns
            
            # Run pattern detection
            for pattern_name, detector in self.patterns.items():
                try:
                    pattern_result = detector(prices)
                    if pattern_result and pattern_result['confidence'] >= 0.5:
                        pattern_result['pattern_name'] = pattern_name
                        pattern_result['symbol'] = symbol
                        detected_patterns.append(pattern_result)
                except Exception as e:
                    logger.error(f"Pattern {pattern_name} detection error: {e}")
            
            return detected_patterns
            
        except Exception as e:
            logger.error(f"Pattern detection error for {symbol}: {e}")
            return []
    
    def _detect_double_bottom(self, prices: List[float]) -> Optional[Dict]:
        """Detect double bottom pattern"""
        try:
            if len(prices) < 40:
                return None
            
            # Find local minima
            lows = []
            for i in range(5, len(prices) - 5):
                if all(prices[i] <= prices[j] for j in range(i-3, i+4)):
                    lows.append((i, prices[i]))
            
            if len(lows) < 2:
                return None
            
            # Check for double bottom pattern
            for i in range(len(lows) - 1):
                low1_idx, low1_price = lows[i]
                low2_idx, low2_price = lows[i + 1]
                
                # Price similarity (within 2%)
                price_diff = abs(low1_price - low2_price) / low1_price
                if price_diff < 0.02 and (low2_idx - low1_idx) > 15:
                    # Check for higher low trend
                    if low2_price > low1_price * 0.99:
                        return {
                            'confidence': 0.8 - price_diff * 10,
                            'direction': 'BUY',
                            'expected_return': 0.04,
                            'risk': 0.25,
                            'urgency': 0.7,
                            'timeframe': 'short',
                            'indicators': {
                                'low1_price': low1_price,
                                'low2_price': low2_price,
                                'price_similarity': 1 - price_diff
                            }
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Double bottom detection error: {e}")
            return None
    
    def _detect_double_top(self, prices: List[float]) -> Optional[Dict]:
        """Detect double top pattern"""
        try:
            if len(prices) < 40:
                return None
            
            # Find local maxima
            highs = []
            for i in range(5, len(prices) - 5):
                if all(prices[i] >= prices[j] for j in range(i-3, i+4)):
                    highs.append((i, prices[i]))
            
            if len(highs) < 2:
                return None
            
            # Check for double top pattern
            for i in range(len(highs) - 1):
                high1_idx, high1_price = highs[i]
                high2_idx, high2_price = highs[i + 1]
                
                # Price similarity (within 2%)
                price_diff = abs(high1_price - high2_price) / high1_price
                if price_diff < 0.02 and (high2_idx - high1_idx) > 15:
                    # Check for lower high trend
                    if high2_price < high1_price * 1.01:
                        return {
                            'confidence': 0.8 - price_diff * 10,
                            'direction': 'SELL',
                            'expected_return': 0.04,
                            'risk': 0.25,
                            'urgency': 0.7,
                            'timeframe': 'short',
                            'indicators': {
                                'high1_price': high1_price,
                                'high2_price': high2_price,
                                'price_similarity': 1 - price_diff
                            }
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Double top detection error: {e}")
            return None
    
    def _detect_head_shoulders(self, prices: List[float]) -> Optional[Dict]:
        """Detect head and shoulders pattern"""
        try:
            if len(prices) < 50:
                return None
            
            # Find three consecutive peaks
            peaks = []
            for i in range(10, len(prices) - 10):
                if all(prices[i] >= prices[j] for j in range(i-5, i+6)):
                    peaks.append((i, prices[i]))
            
            if len(peaks) < 3:
                return None
            
            # Check for head and shoulders formation
            for i in range(len(peaks) - 2):
                left_shoulder = peaks[i]
                head = peaks[i + 1]
                right_shoulder = peaks[i + 2]
                
                # Head should be higher than both shoulders
                if (head[1] > left_shoulder[1] * 1.05 and 
                    head[1] > right_shoulder[1] * 1.05 and
                    abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] < 0.03):
                    
                    confidence = 0.75 + min(0.2, (head[1] - max(left_shoulder[1], right_shoulder[1])) / head[1] * 5)
                    
                    return {
                        'confidence': confidence,
                        'direction': 'SELL',
                        'expected_return': 0.05,
                        'risk': 0.3,
                        'urgency': 0.8,
                        'timeframe': 'medium',
                        'indicators': {
                            'left_shoulder': left_shoulder[1],
                            'head': head[1],
                            'right_shoulder': right_shoulder[1],
                            'neckline': (left_shoulder[1] + right_shoulder[1]) / 2
                        }
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Head and shoulders detection error: {e}")
            return None
    
    def _detect_triangle(self, prices: List[float]) -> Optional[Dict]:
        """Detect triangle patterns"""
        try:
            if len(prices) < 30:
                return None
            
            # Calculate trend lines for highs and lows
            recent_prices = prices[-30:]
            highs = [max(recent_prices[i:i+3]) for i in range(0, len(recent_prices)-2, 3)]
            lows = [min(recent_prices[i:i+3]) for i in range(0, len(recent_prices)-2, 3)]
            
            if len(highs) < 4 or len(lows) < 4:
                return None
            
            # Calculate slopes
            high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
            low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
            
            # Detect triangle type
            if abs(high_slope) < 0.01 and low_slope > 0.01:
                # Ascending triangle
                direction = 'BUY'
                confidence = 0.75
                pattern_type = 'ascending'
            elif high_slope < -0.01 and abs(low_slope) < 0.01:
                # Descending triangle
                direction = 'SELL'
                confidence = 0.75
                pattern_type = 'descending'
            elif high_slope < -0.01 and low_slope > 0.01:
                # Symmetrical triangle
                direction = 'BUY' if recent_prices[-1] > np.mean(recent_prices) else 'SELL'
                confidence = 0.65
                pattern_type = 'symmetrical'
            else:
                return None
            
            return {
                'confidence': confidence,
                'direction': direction,
                'expected_return': 0.03,
                'risk': 0.35,
                'urgency': 0.6,
                'timeframe': 'short',
                'indicators': {
                    'pattern_type': pattern_type,
                    'high_slope': high_slope,
                    'low_slope': low_slope,
                    'convergence_point': len(recent_prices)
                }
            }
            
        except Exception as e:
            logger.error(f"Triangle pattern detection error: {e}")
            return None
    
    def _detect_flag_pattern(self, prices: List[float]) -> Optional[Dict]:
        """Detect flag/pennant patterns"""
        try:
            if len(prices) < 25:
                return None
            
            # Look for sharp move followed by consolidation
            recent_prices = prices[-25:]
            
            # Find sharp move (first 10 candles)
            initial_move = (recent_prices[9] - recent_prices[0]) / recent_prices[0]
            
            if abs(initial_move) < 0.03:  # Need at least 3% move
                return None
            
            # Check consolidation (next 15 candles)
            consolidation_range = max(recent_prices[10:]) - min(recent_prices[10:])
            consolidation_pct = consolidation_range / np.mean(recent_prices[10:])
            
            if consolidation_pct < 0.02:  # Tight consolidation
                direction = 'BUY' if initial_move > 0 else 'SELL'
                confidence = 0.7 + min(0.2, abs(initial_move) * 5)
                
                return {
                    'confidence': confidence,
                    'direction': direction,
                    'expected_return': abs(initial_move) * 0.8,  # Expect similar move
                    'risk': 0.3,
                    'urgency': 0.8,
                    'timeframe': 'immediate',
                    'indicators': {
                        'initial_move': initial_move,
                        'consolidation_range': consolidation_pct,
                        'flag_type': 'bull_flag' if initial_move > 0 else 'bear_flag'
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Flag pattern detection error: {e}")
            return None
    
    def _detect_cup_handle(self, prices: List[float]) -> Optional[Dict]:
        """Detect cup and handle pattern"""
        try:
            if len(prices) < 60:
                return None
            
            # Look for cup formation (U-shape)
            cup_prices = prices[-60:-10]  # Exclude handle
            cup_start = cup_prices[0]
            cup_end = cup_prices[-1]
            cup_bottom = min(cup_prices)
            
            # Cup criteria
            cup_depth = (cup_start - cup_bottom) / cup_start
            if cup_depth < 0.15 or cup_depth > 0.50:  # 15-50% correction
                return None
            
            # Handle formation (smaller correction)
            handle_prices = prices[-15:]
            handle_start = handle_prices[0]
            handle_low = min(handle_prices)
            handle_correction = (handle_start - handle_low) / handle_start
            
            if handle_correction > 0.15:  # Handle too deep
                return None
            
            # Breakout check
            current_price = prices[-1]
            breakout_level = max(cup_start, handle_start)
            
            if current_price > breakout_level * 1.01:  # 1% breakout
                return {
                    'confidence': 0.8,
                    'direction': 'BUY',
                    'expected_return': cup_depth * 0.6,  # Target based on cup depth
                    'risk': 0.25,
                    'urgency': 0.9,
                    'timeframe': 'medium',
                    'indicators': {
                        'cup_depth': cup_depth,
                        'handle_correction': handle_correction,
                        'breakout_level': breakout_level,
                        'current_price': current_price
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Cup and handle detection error: {e}")
            return None
    
    def _detect_ascending_triangle(self, prices: List[float]) -> Optional[Dict]:
        """Detect ascending triangle pattern"""
        try:
            if len(prices) < 30:
                return None
            
            recent_prices = prices[-30:]
            
            # Find resistance level (horizontal)
            highs = []
            for i in range(5, len(recent_prices) - 5):
                if all(recent_prices[i] >= recent_prices[j] for j in range(i-2, i+3)):
                    highs.append(recent_prices[i])
            
            if len(highs) < 3:
                return None
            
            # Check if highs are at similar level (resistance)
            resistance_level = np.mean(highs[-3:])
            resistance_variance = np.std(highs[-3:]) / resistance_level
            
            if resistance_variance > 0.02:  # Too much variance
                return None
            
            # Check for ascending lows
            lows = []
            for i in range(5, len(recent_prices) - 5):
                if all(recent_prices[i] <= recent_prices[j] for j in range(i-2, i+3)):
                    lows.append(recent_prices[i])
            
            if len(lows) < 2:
                return None
            
            # Calculate slope of lows
            low_slope = (lows[-1] - lows[0]) / len(lows)
            
            if low_slope > 0.001:  # Ascending lows
                return {
                    'confidence': 0.75,
                    'direction': 'BUY',
                    'expected_return': 0.04,
                    'risk': 0.3,
                    'urgency': 0.7,
                    'timeframe': 'short',
                    'indicators': {
                        'resistance_level': resistance_level,
                        'low_slope': low_slope,
                        'breakout_target': resistance_level * 1.02
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Ascending triangle detection error: {e}")
            return None
    
    def _detect_descending_triangle(self, prices: List[float]) -> Optional[Dict]:
        """Detect descending triangle pattern"""
        try:
            if len(prices) < 30:
                return None
            
            recent_prices = prices[-30:]
            
            # Find support level (horizontal)
            lows = []
            for i in range(5, len(recent_prices) - 5):
                if all(recent_prices[i] <= recent_prices[j] for j in range(i-2, i+3)):
                    lows.append(recent_prices[i])
            
            if len(lows) < 3:
                return None
            
            # Check if lows are at similar level (support)
            support_level = np.mean(lows[-3:])
            support_variance = np.std(lows[-3:]) / support_level
            
            if support_variance > 0.02:  # Too much variance
                return None
            
            # Check for descending highs
            highs = []
            for i in range(5, len(recent_prices) - 5):
                if all(recent_prices[i] >= recent_prices[j] for j in range(i-2, i+3)):
                    highs.append(recent_prices[i])
            
            if len(highs) < 2:
                return None
            
            # Calculate slope of highs
            high_slope = (highs[-1] - highs[0]) / len(highs)
            
            if high_slope < -0.001:  # Descending highs
                return {
                    'confidence': 0.75,
                    'direction': 'SELL',
                    'expected_return': 0.04,
                    'risk': 0.3,
                    'urgency': 0.7,
                    'timeframe': 'short',
                    'indicators': {
                        'support_level': support_level,
                        'high_slope': high_slope,
                        'breakdown_target': support_level * 0.98
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Descending triangle detection error: {e}")
            return None

class MomentumBreakoutScanner:
    """🚀 Momentum breakout detection scanner"""
    
    def __init__(self):
        self.breakout_thresholds = {
            'volume_multiplier': 2.0,
            'price_movement': 0.02,
            'momentum_strength': 0.6
        }
    
    async def detect_breakouts(self, symbol: str, data: Dict) -> List[Dict]:
        """Detect momentum breakouts"""
        breakouts = []
        
        try:
            prices = data.get('prices', [])
            volumes = data.get('volumes', [])
            
            if len(prices) < 20:
                return breakouts
            
            # Recent price action
            recent_prices = prices[-20:]
            recent_volumes = volumes[-20:] if volumes else [1] * 20
            
            # Calculate momentum indicators
            momentum_signals = self._calculate_momentum_indicators(recent_prices, recent_volumes)
            
            # Detect breakout conditions
            for signal in momentum_signals:
                if signal['strength'] >= self.breakout_thresholds['momentum_strength']:
                    breakout = {
                        'breakout_type': signal['type'],
                        'direction': signal['direction'],
                        'strength': signal['strength'],
                        'confidence': signal['confidence'],
                        'target_return': signal.get('target_return', 0.03),
                        'risk': 0.35,
                        'urgency': 0.9,
                        'indicators': signal['indicators']
                    }
                    breakouts.append(breakout)
            
            return breakouts
            
        except Exception as e:
            logger.error(f"Breakout detection error for {symbol}: {e}")
            return []
    
    def _calculate_momentum_indicators(self, prices: List[float], volumes: List[float]) -> List[Dict]:
        """Calculate various momentum indicators"""
        indicators = []
        
        try:
            if len(prices) < 10:
                return indicators
            
            # 1. Price momentum breakout
            sma_10 = np.mean(prices[-10:])
            sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else sma_10
            current_price = prices[-1]
            
            if current_price > sma_10 * 1.02 and sma_10 > sma_20:
                momentum_strength = min(1.0, (current_price - sma_10) / sma_10 * 20)
                indicators.append({
                    'type': 'price_momentum',
                    'direction': 'BUY',
                    'strength': momentum_strength,
                    'confidence': 0.7 + momentum_strength * 0.2,
                    'target_return': momentum_strength * 0.1,
                    'indicators': {
                        'sma_10': sma_10,
                        'sma_20': sma_20,
                        'momentum_strength': momentum_strength
                    }
                })\n            \n            # 2. Volume breakout\n            avg_volume = np.mean(volumes[-10:])\n            current_volume = volumes[-1]\n            \n            if current_volume > avg_volume * self.breakout_thresholds['volume_multiplier']:\n                volume_strength = min(1.0, current_volume / avg_volume / 5)\n                price_change = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0\n                \n                if abs(price_change) > 0.01:  # Significant price move with volume\n                    indicators.append({\n                        'type': 'volume_breakout',\n                        'direction': 'BUY' if price_change > 0 else 'SELL',\n                        'strength': volume_strength,\n                        'confidence': 0.6 + volume_strength * 0.3,\n                        'target_return': abs(price_change) * 2,\n                        'indicators': {\n                            'volume_ratio': current_volume / avg_volume,\n                            'price_change': price_change,\n                            'volume_strength': volume_strength\n                        }\n                    })\n            \n            # 3. RSI divergence breakout\n            rsi_values = self._calculate_rsi(prices)\n            if len(rsi_values) >= 5:\n                current_rsi = rsi_values[-1]\n                \n                if current_rsi < 30:  # Oversold breakout\n                    indicators.append({\n                        'type': 'rsi_oversold',\n                        'direction': 'BUY',\n                        'strength': (30 - current_rsi) / 30,\n                        'confidence': 0.65,\n                        'target_return': 0.025,\n                        'indicators': {\n                            'rsi': current_rsi,\n                            'oversold_strength': (30 - current_rsi) / 30\n                        }\n                    })\n                elif current_rsi > 70:  # Overbought breakdown\n                    indicators.append({\n                        'type': 'rsi_overbought',\n                        'direction': 'SELL',\n                        'strength': (current_rsi - 70) / 30,\n                        'confidence': 0.65,\n                        'target_return': 0.025,\n                        'indicators': {\n                            'rsi': current_rsi,\n                            'overbought_strength': (current_rsi - 70) / 30\n                        }\n                    })\n            \n            return indicators\n            \n        except Exception as e:\n            logger.error(f\"Momentum indicators calculation error: {e}\")\n            return []\n    \n    def _calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:\n        \"\"\"Calculate RSI values\"\"\"\n        try:\n            if len(prices) < period + 1:\n                return []\n            \n            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]\n            gains = [max(0, delta) for delta in deltas]\n            losses = [max(0, -delta) for delta in deltas]\n            \n            rsi_values = []\n            \n            for i in range(period - 1, len(gains)):\n                avg_gain = np.mean(gains[i-period+1:i+1])\n                avg_loss = np.mean(losses[i-period+1:i+1])\n                \n                if avg_loss == 0:\n                    rsi = 100\n                else:\n                    rs = avg_gain / avg_loss\n                    rsi = 100 - (100 / (1 + rs))\n                \n                rsi_values.append(rsi)\n            \n            return rsi_values\n            \n        except Exception as e:\n            logger.error(f\"RSI calculation error: {e}\")\n            return []

class ReversalOpportunityScanner:
    \"\"\"🔄 Market reversal opportunity scanner\"\"\"\n    \n    def __init__(self):\n        self.reversal_indicators = {\n            'divergence_threshold': 0.7,\n            'oversold_level': 25,\n            'overbought_level': 75,\n            'volume_confirmation': 1.5\n        }\n    \n    async def detect_reversals(self, symbol: str, data: Dict) -> List[Dict]:\n        \"\"\"Detect reversal opportunities\"\"\"\n        reversals = []\n        \n        try:\n            prices = data.get('prices', [])\n            volumes = data.get('volumes', [])\n            \n            if len(prices) < 30:\n                return reversals\n            \n            # Detect different types of reversals\n            reversal_signals = [\n                self._detect_divergence_reversal(prices, volumes),\n                self._detect_support_resistance_reversal(prices),\n                self._detect_momentum_exhaustion(prices),\n                self._detect_volume_climax_reversal(prices, volumes)\n            ]\n            \n            # Filter valid reversals\n            for signal in reversal_signals:\n                if signal and signal['probability'] >= 0.6:\n                    reversals.append(signal)\n            \n            return reversals\n            \n        except Exception as e:\n            logger.error(f\"Reversal detection error for {symbol}: {e}\")\n            return []\n    \n    def _detect_divergence_reversal(self, prices: List[float], volumes: List[float]) -> Optional[Dict]:\n        \"\"\"Detect price-momentum divergence reversals\"\"\"\n        try:\n            if len(prices) < 20:\n                return None\n            \n            # Calculate momentum\n            momentum = [(prices[i] - prices[i-5]) / prices[i-5] for i in range(5, len(prices))]\n            \n            if len(momentum) < 10:\n                return None\n            \n            # Check for momentum divergence\n            recent_prices = prices[-10:]\n            recent_momentum = momentum[-10:]\n            \n            # Price trend vs momentum trend\n            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]\n            momentum_trend = recent_momentum[-1] - recent_momentum[0]\n            \n            # Bearish divergence: price up, momentum down\n            if price_trend > 0.02 and momentum_trend < -0.01:\n                return {\n                    'pattern_type': 'bearish_divergence',\n                    'direction': 'SELL',\n                    'probability': 0.7,\n                    'expected_return': 0.03,\n                    'risk': 0.4,\n                    'urgency': 0.6,\n                    'indicators': {\n                        'price_trend': price_trend,\n                        'momentum_trend': momentum_trend,\n                        'divergence_strength': abs(price_trend + momentum_trend)\n                    }\n                }\n            \n            # Bullish divergence: price down, momentum up\n            elif price_trend < -0.02 and momentum_trend > 0.01:\n                return {\n                    'pattern_type': 'bullish_divergence',\n                    'direction': 'BUY',\n                    'probability': 0.7,\n                    'expected_return': 0.03,\n                    'risk': 0.4,\n                    'urgency': 0.6,\n                    'indicators': {\n                        'price_trend': price_trend,\n                        'momentum_trend': momentum_trend,\n                        'divergence_strength': abs(price_trend + momentum_trend)\n                    }\n                }\n            \n            return None\n            \n        except Exception as e:\n            logger.error(f\"Divergence reversal detection error: {e}\")\n            return None
    
    def _detect_support_resistance_reversal(self, prices: List[float]) -> Optional[Dict]:
        \"\"\"Detect support/resistance level reversals\"\"\"\n        try:\n            if len(prices) < 30:\n                return None\n            \n            current_price = prices[-1]\n            recent_prices = prices[-30:]\n            \n            # Find support and resistance levels\n            support_levels = self._find_support_levels(recent_prices)\n            resistance_levels = self._find_resistance_levels(recent_prices)\n            \n            # Check for bounces off support/resistance\n            for support in support_levels:\n                distance_to_support = abs(current_price - support) / support\n                if distance_to_support < 0.005:  # Within 0.5% of support\n                    return {\n                        'pattern_type': 'support_bounce',\n                        'direction': 'BUY',\n                        'probability': 0.65,\n                        'expected_return': 0.025,\n                        'risk': 0.3,\n                        'urgency': 0.8,\n                        'indicators': {\n                            'support_level': support,\n                            'distance_to_support': distance_to_support,\n                            'bounce_strength': 1 - distance_to_support * 100\n                        }\n                    }\n            \n            for resistance in resistance_levels:\n                distance_to_resistance = abs(current_price - resistance) / resistance\n                if distance_to_resistance < 0.005:  # Within 0.5% of resistance\n                    return {\n                        'pattern_type': 'resistance_rejection',\n                        'direction': 'SELL',\n                        'probability': 0.65,\n                        'expected_return': 0.025,\n                        'risk': 0.3,\n                        'urgency': 0.8,\n                        'indicators': {\n                            'resistance_level': resistance,\n                            'distance_to_resistance': distance_to_resistance,\n                            'rejection_strength': 1 - distance_to_resistance * 100\n                        }\n                    }\n            \n            return None\n            \n        except Exception as e:\n            logger.error(f\"Support/resistance reversal detection error: {e}\")\n            return None
    
    def _find_support_levels(self, prices: List[float]) -> List[float]:\n        \"\"\"Find key support levels\"\"\"\n        try:\n            # Find local minima\n            support_levels = []\n            \n            for i in range(3, len(prices) - 3):\n                if all(prices[i] <= prices[j] for j in range(i-2, i+3)):\n                    support_levels.append(prices[i])\n            \n            # Cluster similar levels\n            clustered_supports = []\n            for support in support_levels:\n                # Check if similar level already exists\n                similar_exists = any(abs(support - existing) / existing < 0.01 \n                                   for existing in clustered_supports)\n                if not similar_exists:\n                    clustered_supports.append(support)\n            \n            return sorted(clustered_supports)[-3:]  # Return top 3 supports\n            \n        except Exception as e:\n            logger.error(f\"Support level detection error: {e}\")\n            return []\n    \n    def _find_resistance_levels(self, prices: List[float]) -> List[float]:\n        \"\"\"Find key resistance levels\"\"\"\n        try:\n            # Find local maxima\n            resistance_levels = []\n            \n            for i in range(3, len(prices) - 3):\n                if all(prices[i] >= prices[j] for j in range(i-2, i+3)):\n                    resistance_levels.append(prices[i])\n            \n            # Cluster similar levels\n            clustered_resistances = []\n            for resistance in resistance_levels:\n                # Check if similar level already exists\n                similar_exists = any(abs(resistance - existing) / existing < 0.01 \n                                   for existing in clustered_resistances)\n                if not similar_exists:\n                    clustered_resistances.append(resistance)\n            \n            return sorted(clustered_resistances, reverse=True)[:3]  # Return top 3 resistances\n            \n        except Exception as e:\n            logger.error(f\"Resistance level detection error: {e}\")\n            return []\n    \n    def _detect_momentum_exhaustion(self, prices: List[float]) -> Optional[Dict]:\n        \"\"\"Detect momentum exhaustion reversals\"\"\"\n        try:\n            if len(prices) < 15:\n                return None\n            \n            # Calculate momentum over different periods\n            momentum_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0\n            momentum_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 else 0\n            momentum_15 = (prices[-1] - prices[-16]) / prices[-16] if len(prices) >= 16 else 0\n            \n            # Check for momentum exhaustion\n            if momentum_5 > 0.05 and momentum_10 > 0.08 and momentum_15 > 0.10:\n                # Strong uptrend, potential exhaustion\n                exhaustion_score = min(1.0, (momentum_5 + momentum_10 + momentum_15) / 0.3)\n                return {\n                    'pattern_type': 'bullish_exhaustion',\n                    'direction': 'SELL',\n                    'probability': 0.6 + exhaustion_score * 0.2,\n                    'expected_return': 0.04,\n                    'risk': 0.5,\n                    'urgency': 0.5,\n                    'indicators': {\n                        'momentum_5': momentum_5,\n                        'momentum_10': momentum_10,\n                        'momentum_15': momentum_15,\n                        'exhaustion_score': exhaustion_score\n                    }\n                }\n            \n            elif momentum_5 < -0.05 and momentum_10 < -0.08 and momentum_15 < -0.10:\n                # Strong downtrend, potential exhaustion\n                exhaustion_score = min(1.0, abs(momentum_5 + momentum_10 + momentum_15) / 0.3)\n                return {\n                    'pattern_type': 'bearish_exhaustion',\n                    'direction': 'BUY',\n                    'probability': 0.6 + exhaustion_score * 0.2,\n                    'expected_return': 0.04,\n                    'risk': 0.5,\n                    'urgency': 0.5,\n                    'indicators': {\n                        'momentum_5': momentum_5,\n                        'momentum_10': momentum_10,\n                        'momentum_15': momentum_15,\n                        'exhaustion_score': exhaustion_score\n                    }\n                }\n            \n            return None\n            \n        except Exception as e:\n            logger.error(f\"Momentum exhaustion detection error: {e}\")\n            return None
    
    def _detect_volume_climax_reversal(self, prices: List[float], volumes: List[float]) -> Optional[Dict]:\n        \"\"\"Detect volume climax reversals\"\"\"\n        try:\n            if len(prices) < 10 or len(volumes) < 10:\n                return None\n            \n            current_volume = volumes[-1]\n            avg_volume = np.mean(volumes[-10:])\n            current_price = prices[-1]\n            prev_price = prices[-2]\n            \n            # Volume spike with price rejection\n            if current_volume > avg_volume * 3:  # 3x volume spike\n                price_change = (current_price - prev_price) / prev_price\n                \n                # High volume but small price move = exhaustion\n                if abs(price_change) < 0.005:  # Less than 0.5% move despite volume\n                    return {\n                        'pattern_type': 'volume_climax',\n                        'direction': 'SELL' if price_change >= 0 else 'BUY',\n                        'probability': 0.7,\n                        'expected_return': 0.02,\n                        'risk': 0.4,\n                        'urgency': 0.9,\n                        'indicators': {\n                            'volume_ratio': current_volume / avg_volume,\n                            'price_change': price_change,\n                            'climax_strength': current_volume / avg_volume / 5\n                        }\n                    }\n            \n            return None\n            \n        except Exception as e:\n            logger.error(f\"Volume climax detection error: {e}\")\n            return None

class ArbitrageOpportunityScanner:
    \"\"\"💰 Arbitrage opportunity scanner\"\"\"\n    \n    def __init__(self):\n        self.arbitrage_threshold = 0.005  # 0.5% minimum spread\n        self.opportunity_history = deque(maxlen=100)\n    \n    async def scan_arbitrage(self, symbols: List[str], exchange_data: Dict) -> List[Dict]:\n        \"\"\"Scan for arbitrage opportunities\"\"\"\n        opportunities = []\n        \n        try:\n            # Cross-exchange arbitrage (placeholder)\n            for symbol in symbols:\n                # This would compare prices across exchanges\n                # For now, simulate opportunities\n                if np.random.random() < 0.1:  # 10% chance of opportunity\n                    spread = np.random.uniform(0.006, 0.015)  # 0.6-1.5% spread\n                    \n                    opportunities.append({\n                        'type': 'cross_exchange',\n                        'symbol': symbol,\n                        'spread': spread,\n                        'direction': 'BUY' if np.random.random() < 0.5 else 'SELL',\n                        'confidence': 0.9,\n                        'expected_profit': spread * 0.8,  # Account for fees\n                        'urgency': 0.95,  # Arbitrage opportunities are time-sensitive\n                        'risk': 0.1\n                    })\n            \n            return opportunities\n            \n        except Exception as e:\n            logger.error(f\"Arbitrage scan error: {e}\")\n            return []\n\nclass VolumeAnomalyScanner:\n    \"\"\"📊 Volume anomaly detection scanner\"\"\"\n    \n    def __init__(self):\n        self.volume_thresholds = {\n            'spike_multiplier': 3.0,\n            'dry_up_threshold': 0.3,\n            'accumulation_threshold': 1.5\n        }\n    \n    async def detect_anomalies(self, symbol: str, data: Dict) -> List[Dict]:\n        \"\"\"Detect volume anomalies\"\"\"\n        anomalies = []\n        \n        try:\n            prices = data.get('prices', [])\n            volumes = data.get('volumes', [])\n            \n            if len(volumes) < 20:\n                return anomalies\n            \n            # Detect volume spikes\n            spike_anomaly = self._detect_volume_spike(prices, volumes)\n            if spike_anomaly:\n                anomalies.append(spike_anomaly)\n            \n            # Detect volume dry-up\n            dryup_anomaly = self._detect_volume_dryup(prices, volumes)\n            if dryup_anomaly:\n                anomalies.append(dryup_anomaly)\n            \n            # Detect accumulation/distribution\n            accumulation_anomaly = self._detect_accumulation_distribution(prices, volumes)\n            if accumulation_anomaly:\n                anomalies.append(accumulation_anomaly)\n            \n            return anomalies\n            \n        except Exception as e:\n            logger.error(f\"Volume anomaly detection error for {symbol}: {e}\")\n            return []\n    \n    def _detect_volume_spike(self, prices: List[float], volumes: List[float]) -> Optional[Dict]:\n        \"\"\"Detect volume spikes\"\"\"\n        try:\n            current_volume = volumes[-1]\n            avg_volume = np.mean(volumes[-20:])\n            \n            if current_volume > avg_volume * self.volume_thresholds['spike_multiplier']:\n                price_change = (prices[-1] - prices[-2]) / prices[-2]\n                \n                return {\n                    'anomaly_type': 'volume_spike',\n                    'direction': 'BUY' if price_change > 0 else 'SELL',\n                    'significance': min(1.0, current_volume / avg_volume / 5),\n                    'confidence': 0.7,\n                    'expected_return': abs(price_change) * 1.5,\n                    'risk': 0.35,\n                    'urgency': 0.9,\n                    'indicators': {\n                        'volume_ratio': current_volume / avg_volume,\n                        'price_change': price_change\n                    }\n                }\n            \n            return None\n            \n        except Exception as e:\n            logger.error(f\"Volume spike detection error: {e}\")\n            return None
    
    def _detect_volume_dryup(self, prices: List[float], volumes: List[float]) -> Optional[Dict]:\n        \"\"\"Detect volume dry-up before moves\"\"\"\n        try:\n            recent_volumes = volumes[-10:]\n            avg_volume = np.mean(volumes[-20:-10])\n            current_avg_volume = np.mean(recent_volumes)\n            \n            if current_avg_volume < avg_volume * self.volume_thresholds['dry_up_threshold']:\n                # Volume dried up, potential breakout coming\n                price_compression = np.std(prices[-10:]) / np.mean(prices[-10:])\n                \n                if price_compression < 0.02:  # Price also compressed\n                    return {\n                        'anomaly_type': 'volume_dryup',\n                        'direction': 'BUY',  # Assume upward breakout\n                        'significance': 1 - (current_avg_volume / avg_volume),\n                        'confidence': 0.6,\n                        'expected_return': 0.03,\n                        'risk': 0.4,\n                        'urgency': 0.6,\n                        'indicators': {\n                            'volume_dryup_ratio': current_avg_volume / avg_volume,\n                            'price_compression': price_compression\n                        }\n                    }\n            \n            return None\n            \n        except Exception as e:\n            logger.error(f\"Volume dry-up detection error: {e}\")\n            return None
    
    def _detect_accumulation_distribution(self, prices: List[float], volumes: List[float]) -> Optional[Dict]:\n        \"\"\"Detect accumulation/distribution patterns\"\"\"\n        try:\n            if len(prices) < 15 or len(volumes) < 15:\n                return None\n            \n            # Calculate price-volume relationship\n            recent_prices = prices[-15:]\n            recent_volumes = volumes[-15:]\n            \n            # On-Balance Volume (OBV) approximation\n            obv = []\n            obv_value = 0\n            \n            for i in range(1, len(recent_prices)):\n                if recent_prices[i] > recent_prices[i-1]:\n                    obv_value += recent_volumes[i]\n                elif recent_prices[i] < recent_prices[i-1]:\n                    obv_value -= recent_volumes[i]\n                obv.append(obv_value)\n            \n            if len(obv) < 10:\n                return None\n            \n            # OBV trend\n            obv_trend = (obv[-1] - obv[0]) / abs(obv[0] + 1)\n            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]\n            \n            # Accumulation: OBV up, price stable/up\n            if obv_trend > 0.1 and price_trend > -0.02:\n                return {\n                    'anomaly_type': 'accumulation',\n                    'direction': 'BUY',\n                    'significance': min(1.0, obv_trend * 2),\n                    'confidence': 0.65,\n                    'expected_return': 0.035,\n                    'risk': 0.3,\n                    'urgency': 0.5,\n                    'indicators': {\n                        'obv_trend': obv_trend,\n                        'price_trend': price_trend,\n                        'accumulation_strength': obv_trend\n                    }\n                }\n            \n            # Distribution: OBV down, price stable/down\n            elif obv_trend < -0.1 and price_trend < 0.02:\n                return {\n                    'anomaly_type': 'distribution',\n                    'direction': 'SELL',\n                    'significance': min(1.0, abs(obv_trend) * 2),\n                    'confidence': 0.65,\n                    'expected_return': 0.035,\n                    'risk': 0.3,\n                    'urgency': 0.5,\n                    'indicators': {\n                        'obv_trend': obv_trend,\n                        'price_trend': price_trend,\n                        'distribution_strength': abs(obv_trend)\n                    }\n                }\n            \n            return None\n            \n        except Exception as e:\n            logger.error(f\"Accumulation/distribution detection error: {e}\")\n            return None

class MarketRegimeDetector:\n    \"\"\"🌊 Market regime detection system\"\"\"\n    \n    def __init__(self):\n        self.regime_history = deque(maxlen=50)\n        self.regime_indicators = {\n            'trend_strength_threshold': 0.6,\n            'volatility_threshold': 0.3,\n            'volume_threshold': 1.5\n        }\n    \n    async def detect_regime(self, market_data: Dict) -> str:\n        \"\"\"Detect current market regime\"\"\"\n        try:\n            # Aggregate data across all symbols\n            all_prices = []\n            all_volumes = []\n            \n            for symbol_data in market_data.values():\n                if 'prices' in symbol_data:\n                    all_prices.extend(symbol_data['prices'][-20:])\n                if 'volumes' in symbol_data:\n                    all_volumes.extend(symbol_data['volumes'][-20:])\n            \n            if len(all_prices) < 50:\n                return 'insufficient_data'\n            \n            # Calculate regime indicators\n            trend_strength = self._calculate_trend_strength(all_prices)\n            volatility = self._calculate_market_volatility(all_prices)\n            volume_activity = self._calculate_volume_activity(all_volumes)\n            \n            # Classify regime\n            if trend_strength > 0.6 and volatility < 0.3:\n                regime = 'trending_low_vol'\n            elif trend_strength > 0.6 and volatility > 0.3:\n                regime = 'trending_high_vol'\n            elif trend_strength < 0.3 and volatility < 0.2:\n                regime = 'sideways_low_vol'\n            elif trend_strength < 0.3 and volatility > 0.4:\n                regime = 'sideways_high_vol'\n            elif volatility > 0.5:\n                regime = 'high_volatility'\n            else:\n                regime = 'mixed'\n            \n            # Record regime\n            self.regime_history.append({\n                'regime': regime,\n                'timestamp': datetime.now(),\n                'trend_strength': trend_strength,\n                'volatility': volatility,\n                'volume_activity': volume_activity\n            })\n            \n            return regime\n            \n        except Exception as e:\n            logger.error(f\"Regime detection error: {e}\")\n            return 'unknown'\n    \n    def _calculate_trend_strength(self, prices: List[float]) -> float:\n        \"\"\"Calculate overall trend strength\"\"\"\n        try:\n            if len(prices) < 20:\n                return 0.0\n            \n            # Linear regression slope\n            x = np.arange(len(prices))\n            slope, _ = np.polyfit(x, prices, 1)\n            \n            # Normalize slope\n            avg_price = np.mean(prices)\n            normalized_slope = abs(slope) / avg_price * len(prices)\n            \n            return min(1.0, normalized_slope * 10)\n            \n        except Exception as e:\n            logger.error(f\"Trend strength calculation error: {e}\")\n            return 0.0\n    \n    def _calculate_market_volatility(self, prices: List[float]) -> float:\n        \"\"\"Calculate market volatility\"\"\"\n        try:\n            if len(prices) < 10:\n                return 0.0\n            \n            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]\n            volatility = np.std(returns)\n            \n            return min(1.0, volatility * 50)  # Normalize\n            \n        except Exception as e:\n            logger.error(f\"Market volatility calculation error: {e}\")\n            return 0.0\n    \n    def _calculate_volume_activity(self, volumes: List[float]) -> float:\n        \"\"\"Calculate volume activity level\"\"\"\n        try:\n            if len(volumes) < 10:\n                return 0.0\n            \n            recent_avg = np.mean(volumes[-10:])\n            historical_avg = np.mean(volumes)\n            \n            activity = recent_avg / historical_avg if historical_avg > 0 else 1.0\n            \n            return min(2.0, activity)\n            \n        except Exception as e:\n            logger.error(f\"Volume activity calculation error: {e}\")\n            return 1.0

class VolatilityAnalyzer:
    \"\"\"📈 Market volatility analysis\"\"\"\n    \n    async def calculate_volatility_index(self, market_data: Dict) -> float:\n        \"\"\"Calculate market-wide volatility index\"\"\"\n        try:\n            all_volatilities = []\n            \n            for symbol_data in market_data.values():\n                prices = symbol_data.get('prices', [])\n                if len(prices) >= 20:\n                    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]\n                    volatility = np.std(returns)\n                    all_volatilities.append(volatility)\n            \n            if not all_volatilities:\n                return 0.5  # Default medium volatility\n            \n            avg_volatility = np.mean(all_volatilities)\n            \n            # Normalize to 0-1 scale\n            normalized_volatility = min(1.0, avg_volatility * 50)\n            \n            return normalized_volatility\n            \n        except Exception as e:\n            logger.error(f\"Volatility index calculation error: {e}\")\n            return 0.5

class MarketSentimentAnalyzer:
    \"\"\"😱 Market sentiment analysis\"\"\"\n    \n    async def analyze_market_sentiment(self, market_data: Dict) -> float:\n        \"\"\"Analyze overall market sentiment (Fear & Greed Index)\"\"\"\n        try:\n            # Aggregate sentiment indicators\n            price_momentum = self._calculate_price_momentum(market_data)\n            volume_sentiment = self._calculate_volume_sentiment(market_data)\n            volatility_sentiment = self._calculate_volatility_sentiment(market_data)\n            \n            # Combine sentiment indicators\n            sentiment_score = (\n                price_momentum * 0.4 +\n                volume_sentiment * 0.3 +\n                volatility_sentiment * 0.3\n            )\n            \n            # Normalize to 0-1 (0 = extreme fear, 1 = extreme greed)\n            return max(0.0, min(1.0, sentiment_score))\n            \n        except Exception as e:\n            logger.error(f\"Sentiment analysis error: {e}\")\n            return 0.5  # Neutral sentiment\n    \n    def _calculate_price_momentum(self, market_data: Dict) -> float:\n        \"\"\"Calculate price momentum sentiment\"\"\"\n        try:\n            momentum_scores = []\n            \n            for symbol_data in market_data.values():\n                prices = symbol_data.get('prices', [])\n                if len(prices) >= 10:\n                    momentum = (prices[-1] - prices[-10]) / prices[-10]\n                    momentum_scores.append(momentum)\n            \n            if not momentum_scores:\n                return 0.5\n            \n            avg_momentum = np.mean(momentum_scores)\n            \n            # Convert to 0-1 scale (centered at 0.5)\n            return 0.5 + np.tanh(avg_momentum * 10) * 0.5\n            \n        except Exception as e:\n            logger.error(f\"Price momentum sentiment error: {e}\")\n            return 0.5\n    \n    def _calculate_volume_sentiment(self, market_data: Dict) -> float:\n        \"\"\"Calculate volume-based sentiment\"\"\"\n        try:\n            volume_ratios = []\n            \n            for symbol_data in market_data.values():\n                volumes = symbol_data.get('volumes', [])\n                if len(volumes) >= 10:\n                    recent_volume = np.mean(volumes[-5:])\n                    historical_volume = np.mean(volumes[-20:-5])\n                    ratio = recent_volume / historical_volume if historical_volume > 0 else 1.0\n                    volume_ratios.append(ratio)\n            \n            if not volume_ratios:\n                return 0.5\n            \n            avg_ratio = np.mean(volume_ratios)\n            \n            # Higher volume = more greed/activity\n            return min(1.0, avg_ratio / 3)  # Cap at 3x volume\n            \n        except Exception as e:\n            logger.error(f\"Volume sentiment error: {e}\")\n            return 0.5\n    \n    def _calculate_volatility_sentiment(self, market_data: Dict) -> float:\n        \"\"\"Calculate volatility-based sentiment\"\"\"\n        try:\n            volatilities = []\n            \n            for symbol_data in market_data.values():\n                prices = symbol_data.get('prices', [])\n                if len(prices) >= 10:\n                    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]\n                    volatility = np.std(returns)\n                    volatilities.append(volatility)\n            \n            if not volatilities:\n                return 0.5\n            \n            avg_volatility = np.mean(volatilities)\n            \n            # Higher volatility = more fear (inverted)\n            return max(0.0, 1.0 - avg_volatility * 50)\n            \n        except Exception as e:\n            logger.error(f\"Volatility sentiment error: {e}\")\n            return 0.5

class OpportunityQualityFilters:
    \"\"\"🎯 Quality filters for trading opportunities\"\"\"\n    \n    def __init__(self):\n        self.quality_thresholds = {\n            'min_confidence': 0.6,\n            'min_expected_return': 0.015,\n            'max_risk_score': 0.8,\n            'min_urgency': 0.3\n        }\n    \n    async def filter_opportunities(self, opportunities: List[MarketOpportunity], market_data: Dict) -> List[MarketOpportunity]:\n        \"\"\"Filter opportunities by quality criteria\"\"\"\n        filtered = []\n        \n        try:\n            for opp in opportunities:\n                # Basic quality checks\n                if (opp.confidence >= self.quality_thresholds['min_confidence'] and\n                    opp.expected_return >= self.quality_thresholds['min_expected_return'] and\n                    opp.risk_score <= self.quality_thresholds['max_risk_score'] and\n                    opp.urgency >= self.quality_thresholds['min_urgency']):\n                    \n                    # Additional quality checks\n                    if self._passes_advanced_quality_checks(opp, market_data):\n                        filtered.append(opp)\n            \n            return filtered\n            \n        except Exception as e:\n            logger.error(f\"Opportunity filtering error: {e}\")\n            return opportunities\n    \n    def _passes_advanced_quality_checks(self, opportunity: MarketOpportunity, market_data: Dict) -> bool:\n        \"\"\"Advanced quality checks for opportunities\"\"\"\n        try:\n            # Check risk-reward ratio\n            risk_reward_ratio = opportunity.expected_return / (opportunity.risk_score + 0.01)\n            if risk_reward_ratio < 1.5:  # Need at least 1.5:1 ratio\n                return False\n            \n            # Check opportunity freshness\n            time_since_detection = datetime.now() - opportunity.timestamp\n            if time_since_detection > timedelta(minutes=30):  # Too old\n                return False\n            \n            # Check for conflicting signals\n            symbol_data = market_data.get(opportunity.symbol, {})\n            if self._has_conflicting_signals(opportunity, symbol_data):\n                return False\n            \n            return True\n            \n        except Exception as e:\n            logger.error(f\"Advanced quality check error: {e}\")\n            return True  # Default to passing\n    \n    def _has_conflicting_signals(self, opportunity: MarketOpportunity, symbol_data: Dict) -> bool:\n        \"\"\"Check for conflicting signals\"\"\"\n        try:\n            # This would check for contradictory indicators\n            # For now, return False (no conflicts)\n            return False\n            \n        except Exception as e:\n            logger.error(f\"Conflicting signals check error: {e}\")\n            return False

class RiskBasedFilters:\n    \"\"\"⚠️ Risk-based opportunity filters\"\"\"\n    \n    def __init__(self):\n        self.risk_limits = {\n            'max_portfolio_risk': 0.15,  # 15% max portfolio risk\n            'max_single_asset_risk': 0.05,  # 5% max single asset risk\n            'max_correlation_exposure': 0.3  # 30% max correlated exposure\n        }\n    \n    async def filter_by_risk(self, opportunities: List[MarketOpportunity], market_data: Dict) -> List[MarketOpportunity]:\n        \"\"\"Filter opportunities by risk criteria\"\"\"\n        filtered = []\n        \n        try:\n            for opp in opportunities:\n                # Risk score check\n                if opp.risk_score > 0.8:  # Too risky\n                    continue\n                \n                # Expected return vs risk check\n                if opp.expected_return / opp.risk_score < 2.0:  # Need 2:1 return:risk\n                    continue\n                \n                # Portfolio impact check\n                if self._calculate_portfolio_impact(opp) > self.risk_limits['max_single_asset_risk']:\n                    continue\n                \n                filtered.append(opp)\n            \n            return filtered\n            \n        except Exception as e:\n            logger.error(f\"Risk filtering error: {e}\")\n            return opportunities\n    \n    def _calculate_portfolio_impact(self, opportunity: MarketOpportunity) -> float:\n        \"\"\"Calculate potential portfolio impact of opportunity\"\"\"\n        try:\n            # Estimate position size based on risk\n            position_risk = opportunity.risk_score * 0.02  # 2% base position\n            return position_risk\n            \n        except Exception as e:\n            logger.error(f\"Portfolio impact calculation error: {e}\")\n            return 0.02  # Default 2%\n\n# Global market scanner\nmarket_scanner = AdvancedMarketScanner()\n\nprint(\"🔍 Advanced Market Scanner ready!\")\nprint(\"   📊 Features: Pattern detection, breakout scanning, reversal analysis\")\nprint(\"   🎯 Goal: Identify high-probability opportunities with 90%+ success rate\")
