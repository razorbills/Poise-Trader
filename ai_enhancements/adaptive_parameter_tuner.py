"""
🔄 ADAPTIVE PARAMETER TUNING AI
Auto-tunes stop loss, take profit, position sizes
Constantly tests and keeps what works!
"""

import numpy as np
from typing import Dict, List
from collections import deque, defaultdict
import json
import os


class AdaptiveParameterTuner:
    """
    🎯 Adaptive Parameter Optimization
    
    Dynamically adjusts trading parameters based on performance
    """
    
    def __init__(self, tuning_file: str = "parameter_tuning.json"):
        self.tuning_file = tuning_file
        
        # Current parameters
        self.parameters = {
            'stop_loss': 2.0,
            'take_profit': 3.5,
            'position_size_mult': 1.0,
            'confidence_threshold': 0.50
        }
        
        # Performance tracking per parameter set
        self.performance_history = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'total_pnl': 0.0
        })
        
        # Test variants
        self.test_variants = self._generate_test_variants()
        self.test_frequency = 0.10  # Test variants 10% of time
        
        # Load saved state
        self.load_state()
    
    def _generate_test_variants(self) -> List[Dict]:
        """Generate parameter variants to test"""
        variants = []
        
        # Stop loss variants
        for sl in [1.5, 2.0, 2.5, 3.0]:
            # Take profit variants (should be > stop loss)
            for tp in [2.5, 3.0, 3.5, 4.0, 5.0]:
                if tp > sl:
                    variants.append({
                        'stop_loss': sl,
                        'take_profit': tp,
                        'position_size_mult': 1.0,
                        'confidence_threshold': 0.50
                    })
        
        # Position size multipliers
        for mult in [0.8, 1.0, 1.2]:
            variants.append({
                'stop_loss': 2.0,
                'take_profit': 3.5,
                'position_size_mult': mult,
                'confidence_threshold': 0.50
            })
        
        # Confidence thresholds
        for threshold in [0.40, 0.50, 0.60, 0.70]:
            variants.append({
                'stop_loss': 2.0,
                'take_profit': 3.5,
                'position_size_mult': 1.0,
                'confidence_threshold': threshold
            })
        
        return variants
    
    def get_parameters(self, force_test: bool = False) -> Dict:
        """
        Get trading parameters (either best or test variant)
        
        Args:
            force_test: Force testing a variant
            
        Returns:
            Parameter dictionary
        """
        # Decide whether to test or use best
        if force_test or np.random.random() < self.test_frequency:
            # Test a random variant
            variant = np.random.choice(self.test_variants)
            variant['is_test'] = True
            return variant
        else:
            # Use best known parameters
            params = self.parameters.copy()
            params['is_test'] = False
            return params
    
    def record_result(self, parameters: Dict, profit: float, won: bool):
        """
        Record trade result for parameter set
        
        Args:
            parameters: Parameter dict used for trade
            profit: Profit/loss from trade
            won: Whether trade won
        """
        # Create parameter signature
        sig = self._param_signature(parameters)
        
        # Update performance
        perf = self.performance_history[sig]
        perf['trades'] += 1
        perf['total_pnl'] += profit
        if won:
            perf['wins'] += 1
        
        # Update best parameters every 20 trades
        if sum(p['trades'] for p in self.performance_history.values()) % 20 == 0:
            self._update_best_parameters()
            self.save_state()
    
    def _param_signature(self, params: Dict) -> str:
        """Create unique signature for parameter set"""
        return f"SL{params['stop_loss']}_TP{params['take_profit']}_PS{params['position_size_mult']}_CT{params['confidence_threshold']}"
    
    def _update_best_parameters(self):
        """Update best parameters based on performance"""
        best_sig = None
        best_score = -float('inf')
        
        for sig, perf in self.performance_history.items():
            if perf['trades'] < 5:  # Need at least 5 trades
                continue
            
            # Calculate score: win rate + avg profit
            win_rate = perf['wins'] / perf['trades']
            avg_profit = perf['total_pnl'] / perf['trades']
            
            # Combined score
            score = win_rate * 100 + avg_profit * 10
            
            if score > best_score:
                best_score = score
                best_sig = sig
        
        if best_sig:
            # Parse signature back to parameters
            self._parse_signature_to_params(best_sig)
            print(f"🔄 Updated best parameters: {best_sig} (Score: {best_score:.2f})")
    
    def _parse_signature_to_params(self, sig: str):
        """Parse signature string back to parameters"""
        parts = sig.split('_')
        self.parameters['stop_loss'] = float(parts[0].replace('SL', ''))
        self.parameters['take_profit'] = float(parts[1].replace('TP', ''))
        self.parameters['position_size_mult'] = float(parts[2].replace('PS', ''))
        self.parameters['confidence_threshold'] = float(parts[3].replace('CT', ''))
    
    def get_tuning_stats(self) -> Dict:
        """Get parameter tuning statistics"""
        stats = {
            'current_best': self.parameters,
            'variants_tested': len(self.performance_history),
            'total_trades': sum(p['trades'] for p in self.performance_history.values()),
            'top_performers': []
        }
        
        # Get top 5 performers
        performers = []
        for sig, perf in self.performance_history.items():
            if perf['trades'] >= 5:
                win_rate = perf['wins'] / perf['trades']
                avg_profit = perf['total_pnl'] / perf['trades']
                score = win_rate * 100 + avg_profit * 10
                
                performers.append({
                    'signature': sig,
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'trades': perf['trades'],
                    'score': score
                })
        
        performers.sort(key=lambda x: x['score'], reverse=True)
        stats['top_performers'] = performers[:5]
        
        return stats
    
    def save_state(self):
        """Save tuning state"""
        state = {
            'parameters': self.parameters,
            'performance_history': {
                k: v for k, v in self.performance_history.items()
            }
        }
        
        try:
            with open(self.tuning_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save tuning state: {e}")
    
    def load_state(self):
        """Load tuning state"""
        if not os.path.exists(self.tuning_file):
            return
        
        try:
            with open(self.tuning_file, 'r') as f:
                state = json.load(f)
            
            self.parameters = state.get('parameters', self.parameters)
            
            for sig, perf in state.get('performance_history', {}).items():
                self.performance_history[sig] = perf
            
            print(f"✅ Loaded parameter tuning: {len(self.performance_history)} variants tested")
        except Exception as e:
            print(f"⚠️ Failed to load tuning state: {e}")
