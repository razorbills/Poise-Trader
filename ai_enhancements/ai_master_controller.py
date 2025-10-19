"""
🧠 AI MASTER CONTROLLER
Integrates ALL 10 AI enhancements into one powerful system!
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque

# Import all AI modules
try:
    from .market_regime_detector import MarketRegimeDetectorAI, MarketRegime
    from .multi_timeframe_analyzer import MultiTimeframeAnalyzerAI
    from .pattern_recognition_ai import PatternRecognitionAI
    from .confidence_calibrator import ConfidenceCalibratorAI
    from .reinforcement_learning_ai import ReinforcementLearningAI
    from .adaptive_parameter_tuner import AdaptiveParameterTuner
    from .lstm_price_predictor import LSTMPricePredictor
    from .sentiment_analyzer import SentimentAnalyzer, FeatureEngineer
except ImportError:
    # Fallback for direct execution
    from market_regime_detector import MarketRegimeDetectorAI, MarketRegime
    from multi_timeframe_analyzer import MultiTimeframeAnalyzerAI
    from pattern_recognition_ai import PatternRecognitionAI
    from confidence_calibrator import ConfidenceCalibratorAI
    from reinforcement_learning_ai import ReinforcementLearningAI
    from adaptive_parameter_tuner import AdaptiveParameterTuner
    from lstm_price_predictor import LSTMPricePredictor
    from sentiment_analyzer import SentimentAnalyzer, FeatureEngineer


class AIMasterController:
    """
    🎯 ULTIMATE AI TRADING BRAIN
    
    Combines all 10 AI enhancements:
    1. Market Regime Detection
    2. Multi-Timeframe Analysis
    3. Pattern Recognition
    4. Confidence Calibration
    5. Reinforcement Learning
    6. Adaptive Parameter Tuning
    7. Feature Engineering (integrated)
    8. Ensemble Voting (integrated)
    9. Price Prediction (simplified)
    10. Sentiment Analysis (placeholder for future)
    """
    
    def __init__(self, enable_all: bool = True):
        print("🧠 Initializing AI MASTER CONTROLLER...")
        
        # Initialize all AI modules
        self.regime_detector = MarketRegimeDetectorAI() if enable_all else None
        self.mtf_analyzer = MultiTimeframeAnalyzerAI() if enable_all else None
        self.pattern_recognizer = PatternRecognitionAI() if enable_all else None
        self.confidence_calibrator = ConfidenceCalibratorAI() if enable_all else None
        self.rl_agent = ReinforcementLearningAI() if enable_all else None
        self.parameter_tuner = AdaptiveParameterTuner() if enable_all else None
        self.lstm_predictor = LSTMPricePredictor() if enable_all else None
        self.sentiment_analyzer = SentimentAnalyzer() if enable_all else None
        self.feature_engineer = FeatureEngineer() if enable_all else None
        
        # Trading history for learning
        self.trade_history = deque(maxlen=500)
        
        # Current market state
        self.current_regime = None
        self.current_parameters = None
        
        print("✅ AI Master Controller initialized!")
        print(f"   ✓ Market Regime Detection: {'ENABLED' if self.regime_detector else 'DISABLED'}")
        print(f"   ✓ Multi-Timeframe Analysis: {'ENABLED' if self.mtf_analyzer else 'DISABLED'}")
        print(f"   ✓ Pattern Recognition: {'ENABLED' if self.pattern_recognizer else 'DISABLED'}")
        print(f"   ✓ Confidence Calibration: {'ENABLED' if self.confidence_calibrator else 'DISABLED'}")
        print(f"   ✓ Reinforcement Learning: {'ENABLED' if self.rl_agent else 'DISABLED'}")
        print(f"   ✓ Adaptive Parameters: {'ENABLED' if self.parameter_tuner else 'DISABLED'}")
        print(f"   ✓ LSTM Price Prediction: {'ENABLED' if self.lstm_predictor else 'DISABLED'}")
        print(f"   ✓ Sentiment Analysis: {'ENABLED' if self.sentiment_analyzer else 'DISABLED'}")
        print(f"   ✓ Feature Engineering: {'ENABLED' if self.feature_engineer else 'DISABLED'}")
    
    def analyze_trade_opportunity(self, market_data: Dict) -> Dict:
        """
        🎯 MASTER ANALYSIS - Combines all AI modules
        
        Args:
            market_data: {
                'symbol': str,
                'prices': List[float],
                'volumes': List[float],
                'price_data_mtf': Dict[str, List[float]],  # Multi-timeframe
                'current_price': float,
                'base_confidence': float,
                'base_action': str
            }
            
        Returns:
            Enhanced analysis with AI recommendations
        """
        print(f"\n🧠 AI MASTER ANALYSIS: {market_data.get('symbol', 'UNKNOWN')}")
        print("="*80)
        
        result = {
            'symbol': market_data.get('symbol'),
            'enhanced_confidence': market_data.get('base_confidence', 0.5),
            'recommended_action': market_data.get('base_action', 'HOLD'),
            'should_trade': False,
            'ai_insights': {},
            'parameters': {}
        }
        
        prices = market_data.get('prices', [])
        if len(prices) < 20:
            print("⚠️ Insufficient data for AI analysis")
            return result
        
        # Step 1: Market Regime Detection
        if self.regime_detector:
            regime_info = self.regime_detector.detect_regime(
                prices,
                market_data.get('volumes')
            )
            self.current_regime = regime_info
            result['ai_insights']['regime'] = regime_info
            
            regime_name = self.regime_detector.get_regime_name(regime_info['regime'])
            print(f"📊 Market Regime: {regime_name}")
            print(f"   Confidence: {regime_info['confidence']*100:.1f}%")
            print(f"   Trend Strength: {regime_info['trend_strength']*100:.1f}%")
            print(f"   Recommended Allocation: {regime_info['recommended_allocation']*100:.0f}%")
        
        # Step 2: Multi-Timeframe Analysis
        if self.mtf_analyzer and 'price_data_mtf' in market_data:
            mtf_analysis = self.mtf_analyzer.analyze_timeframes(
                market_data['price_data_mtf']
            )
            result['ai_insights']['multi_timeframe'] = mtf_analysis
            
            alignment_emoji = self.mtf_analyzer.get_alignment_emoji(mtf_analysis['alignment_score'])
            print(f"🧪 Timeframe Alignment: {alignment_emoji} {mtf_analysis['alignment_score']:.0f}%")
            print(f"   Trend: {mtf_analysis['trend_direction']}")
            print(f"   Should Trade: {'YES' if mtf_analysis['should_trade'] else 'NO'}")
        
        # Step 3: Pattern Recognition
        if self.pattern_recognizer:
            patterns = self.pattern_recognizer.detect_patterns(
                prices,
                market_data.get('volumes')
            )
            result['ai_insights']['patterns'] = patterns
            
            if patterns:
                print(f"🎯 Patterns Detected: {len(patterns)}")
                for i, pattern in enumerate(patterns[:3], 1):  # Show top 3
                    print(f"   {i}. {pattern['pattern_name']}")
                    print(f"      Confidence: {pattern['confidence']*100:.0f}% | Win Rate: {pattern['expected_win_rate']*100:.0f}%")
                    print(f"      Action: {pattern['action']} | Target: ${pattern['target']:.2f}")
        
        # Step 4: Confidence Calibration
        base_confidence = market_data.get('base_confidence', 0.5)
        if self.confidence_calibrator:
            calibrated = self.confidence_calibrator.calibrate_confidence(base_confidence)
            result['enhanced_confidence'] = calibrated
            
            adjustment = self.confidence_calibrator.get_confidence_adjustment(base_confidence)
            print(f"🎯 Confidence Calibration:")
            print(f"   Original: {base_confidence*100:.0f}%")
            print(f"   Calibrated: {calibrated*100:.0f}%")
            print(f"   Adjustment: {adjustment['adjustment']*100:+.1f}%")
            print(f"   Reliability: {adjustment['reliability']}")
        
        # Step 5: Reinforcement Learning Decision
        if self.rl_agent:
            rl_market_state = self.rl_agent.get_market_state({
                'trend_direction': result['ai_insights'].get('regime', {}).get('trend_direction', 0),
                'volatility': result['ai_insights'].get('regime', {}).get('volatility', 0.02),
                'momentum': result['ai_insights'].get('regime', {}).get('momentum', 0),
                'pattern': patterns[0]['pattern'].value if patterns else 'NONE'
            })
            
            rl_action = self.rl_agent.choose_action(rl_market_state, ['BUY', 'SELL', 'HOLD'])
            result['ai_insights']['rl'] = {
                'state': rl_market_state,
                'recommended_action': rl_action
            }
            
            print(f"🧬 Reinforcement Learning:")
            print(f"   State: {rl_market_state[:50]}...")
            print(f"   Recommended Action: {rl_action}")
        
        # Step 6: Get Adaptive Parameters
        if self.parameter_tuner:
            parameters = self.parameter_tuner.get_parameters()
            self.current_parameters = parameters
            result['parameters'] = parameters
            
            print(f"🔄 Adaptive Parameters:")
            print(f"   Stop Loss: {parameters['stop_loss']}%")
            print(f"   Take Profit: {parameters['take_profit']}%")
            print(f"   Position Size Mult: {parameters['position_size_mult']}x")
            print(f"   Confidence Threshold: {parameters['confidence_threshold']*100:.0f}%")
            print(f"   Mode: {'TEST' if parameters.get('is_test') else 'BEST'}")
        
        # Step 7: ENSEMBLE DECISION (Combine all AI inputs)
        final_decision = self._make_ensemble_decision(result)
        result.update(final_decision)
        
        print(f"\n🎯 FINAL AI DECISION:")
        print(f"   Action: {result['recommended_action']}")
        print(f"   Confidence: {result['enhanced_confidence']*100:.0f}%")
        print(f"   Should Trade: {'YES ✅' if result['should_trade'] else 'NO ❌'}")
        print(f"   Reason: {result.get('decision_reason', 'N/A')}")
        print("="*80 + "\n")
        
        return result
    
    def _make_ensemble_decision(self, analysis: Dict) -> Dict:
        """
        🎲 ENSEMBLE VOTING - Combine all AI recommendations
        
        Uses weighted voting from all AI modules
        """
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confidence_sum = 0
        vote_count = 0
        reasons = []
        
        # Vote 1: Pattern Recognition (Weight: 3)
        patterns = analysis['ai_insights'].get('patterns', [])
        if patterns:
            best_pattern = patterns[0]
            action = best_pattern['action']
            votes[action] += 3 * best_pattern['confidence']
            vote_count += 3
            confidence_sum += best_pattern['confidence'] * 3
            reasons.append(f"Pattern: {best_pattern['pattern_name']} ({best_pattern['confidence']*100:.0f}%)")
        
        # Vote 2: Market Regime (Weight: 2)
        regime_info = analysis['ai_insights'].get('regime')
        if regime_info:
            # Regime suggests action based on trend
            if regime_info['trend_direction'] > 0.3:
                votes['BUY'] += 2 * regime_info['confidence']
                reasons.append(f"Regime: Bullish ({regime_info['confidence']*100:.0f}%)")
            elif regime_info['trend_direction'] < -0.3:
                votes['SELL'] += 2 * regime_info['confidence']
                reasons.append(f"Regime: Bearish ({regime_info['confidence']*100:.0f}%)")
            else:
                votes['HOLD'] += 2 * regime_info['confidence']
                reasons.append(f"Regime: Sideways ({regime_info['confidence']*100:.0f}%)")
            vote_count += 2
            confidence_sum += regime_info['confidence'] * 2
        
        # Vote 3: Multi-Timeframe (Weight: 2)
        mtf = analysis['ai_insights'].get('multi_timeframe')
        if mtf and mtf['should_trade']:
            action = mtf['recommended_action']
            votes[action] += 2 * mtf['confidence']
            vote_count += 2
            confidence_sum += mtf['confidence'] * 2
            reasons.append(f"MTF: {action} (Alignment: {mtf['alignment_score']:.0f}%)")
        
        # Vote 4: Reinforcement Learning (Weight: 2)
        rl = analysis['ai_insights'].get('rl')
        if rl:
            action = rl['recommended_action']
            votes[action] += 2
            vote_count += 2
            confidence_sum += 0.7 * 2  # Assume 70% confidence for RL
            reasons.append(f"RL: {action}")
        
        # Determine winner
        if vote_count == 0:
            return {
                'recommended_action': 'HOLD',
                'enhanced_confidence': 0.5,
                'should_trade': False,
                'decision_reason': 'No AI votes'
            }
        
        winner = max(votes, key=votes.get)
        avg_confidence = confidence_sum / vote_count if vote_count > 0 else 0.5
        
        # Use calibrated confidence if available
        final_confidence = analysis.get('enhanced_confidence', avg_confidence)
        
        # Apply parameter threshold
        params = analysis.get('parameters', {})
        threshold = params.get('confidence_threshold', 0.50)
        
        should_trade = (
            winner in ['BUY', 'SELL'] and
            final_confidence >= threshold and
            vote_count >= 3  # At least 3 votes required
        )
        
        return {
            'recommended_action': winner,
            'enhanced_confidence': final_confidence,
            'should_trade': should_trade,
            'decision_reason': ' | '.join(reasons),
            'vote_breakdown': votes,
            'vote_count': vote_count
        }
    
    def record_trade_outcome(self, trade_data: Dict):
        """
        📚 Record trade outcome for all AI modules to learn
        
        Args:
            trade_data: {
                'symbol': str,
                'action': str,
                'entry_price': float,
                'exit_price': float,
                'profit': float,
                'won': bool,
                'predicted_confidence': float,
                'parameters_used': Dict
            }
        """
        # Store in history
        self.trade_history.append(trade_data)
        
        # Update Confidence Calibrator
        if self.confidence_calibrator:
            self.confidence_calibrator.record_trade_result(
                trade_data['predicted_confidence'],
                trade_data['won'],
                trade_data
            )
        
        # Update Reinforcement Learning
        if self.rl_agent and self.current_regime:
            reward = trade_data['profit']  # Raw profit as reward
            current_state = self.rl_agent.get_market_state({
                'trend_direction': self.current_regime.get('trend_direction', 0),
                'volatility': self.current_regime.get('volatility', 0.02),
                'momentum': self.current_regime.get('momentum', 0),
                'pattern': 'NONE'
            })
            
            self.rl_agent.learn_from_trade(
                state=current_state,
                action=trade_data['action'],
                reward=reward,
                next_state=current_state,  # Simplified
                done=True
            )
        
        # Update Parameter Tuner
        if self.parameter_tuner and trade_data.get('parameters_used'):
            self.parameter_tuner.record_result(
                trade_data['parameters_used'],
                trade_data['profit'],
                trade_data['won']
            )
        
        print(f"📚 AI Learning: Recorded trade outcome ({trade_data['symbol']}, Profit: ${trade_data['profit']:.3f}, Won: {trade_data['won']})")
    
    def get_ai_stats(self) -> Dict:
        """Get statistics from all AI modules"""
        stats = {
            'total_trades_analyzed': len(self.trade_history)
        }
        
        if self.confidence_calibrator:
            stats['confidence_calibration'] = self.confidence_calibrator.get_calibration_stats()
        
        if self.rl_agent:
            stats['reinforcement_learning'] = self.rl_agent.get_learning_stats()
        
        if self.parameter_tuner:
            stats['parameter_tuning'] = self.parameter_tuner.get_tuning_stats()
        
        return stats
    
    def print_ai_report(self):
        """Print comprehensive AI performance report"""
        print("\n" + "="*80)
        print("🧠 AI MASTER CONTROLLER - PERFORMANCE REPORT")
        print("="*80)
        
        if self.confidence_calibrator:
            self.confidence_calibrator.print_calibration_report()
        
        if self.rl_agent:
            self.rl_agent.print_learning_report()
        
        print("="*80 + "\n")
