"""
🎯 CONFIDENCE CALIBRATION AI
Makes confidence scores more accurate
70% confidence → Actually wins 70% of the time!
"""

import numpy as np
from typing import Dict, List
from collections import defaultdict, deque
import json
import os


class ConfidenceCalibratorAI:
    """
    🧠 Confidence Score Calibration
    
    Learns the true win rate for each confidence level
    Adjusts scores to match reality
    """
    
    def __init__(self, calibration_file: str = "confidence_calibration.json"):
        self.calibration_file = calibration_file
        
        # Track actual win rates per confidence bucket
        self.confidence_buckets = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total': 0})
        
        # Historical trades for calibration
        self.trade_history = deque(maxlen=500)
        
        # Calibration curve (maps reported confidence → true win rate)
        self.calibration_curve = {}
        
        # Load existing calibration data
        self.load_calibration()
        
    def calibrate_confidence(self, reported_confidence: float) -> float:
        """
        🎯 Calibrate confidence score
        
        Args:
            reported_confidence: AI's reported confidence (0-1)
            
        Returns:
            Calibrated confidence based on historical accuracy
        """
        if not self.calibration_curve:
            return reported_confidence  # No calibration data yet
        
        # Find closest calibration bucket
        confidence_pct = int(reported_confidence * 100)
        
        # Get calibrated value
        if confidence_pct in self.calibration_curve:
            calibrated = self.calibration_curve[confidence_pct]
        else:
            # Interpolate between nearest buckets
            calibrated = self._interpolate_confidence(confidence_pct)
        
        return calibrated
    
    def record_trade_result(self, predicted_confidence: float, 
                           actual_result: bool, 
                           trade_info: Dict = None):
        """
        📊 Record a trade result for calibration learning
        
        Args:
            predicted_confidence: What AI predicted (0-1)
            actual_result: True if win, False if loss
            trade_info: Additional trade details
        """
        # Round to nearest 5% bucket
        bucket = self._get_confidence_bucket(predicted_confidence)
        
        # Update bucket stats
        self.confidence_buckets[bucket]['total'] += 1
        if actual_result:
            self.confidence_buckets[bucket]['wins'] += 1
        else:
            self.confidence_buckets[bucket]['losses'] += 1
        
        # Add to trade history
        self.trade_history.append({
            'predicted_confidence': predicted_confidence,
            'actual_result': actual_result,
            'timestamp': trade_info.get('timestamp') if trade_info else None
        })
        
        # Recalculate calibration curve every 10 trades
        if len(self.trade_history) % 10 == 0:
            self.update_calibration_curve()
            self.save_calibration()
    
    def _get_confidence_bucket(self, confidence: float) -> int:
        """Get 5% confidence bucket (0, 5, 10, ..., 95, 100)"""
        bucket = int(round(confidence * 100 / 5) * 5)
        return max(0, min(100, bucket))
    
    def update_calibration_curve(self):
        """Recalculate calibration curve from all trade data"""
        new_curve = {}
        
        for bucket, stats in self.confidence_buckets.items():
            if stats['total'] >= 5:  # At least 5 trades for reliability
                true_win_rate = stats['wins'] / stats['total']
                new_curve[bucket] = true_win_rate
        
        # Smooth the curve
        if len(new_curve) >= 3:
            new_curve = self._smooth_calibration_curve(new_curve)
        
        self.calibration_curve = new_curve
    
    def _smooth_calibration_curve(self, curve: Dict) -> Dict:
        """Apply smoothing to calibration curve"""
        buckets = sorted(curve.keys())
        win_rates = [curve[b] for b in buckets]
        
        # Simple moving average smoothing
        smoothed = {}
        for i, bucket in enumerate(buckets):
            if i == 0:
                smoothed[bucket] = win_rates[i]
            elif i == len(buckets) - 1:
                smoothed[bucket] = win_rates[i]
            else:
                # Average of current and neighbors
                avg = (win_rates[i-1] + win_rates[i] + win_rates[i+1]) / 3
                smoothed[bucket] = avg
        
        return smoothed
    
    def _interpolate_confidence(self, confidence_pct: int) -> float:
        """Interpolate confidence for buckets without data"""
        if not self.calibration_curve:
            return confidence_pct / 100.0
        
        buckets = sorted(self.calibration_curve.keys())
        
        # Find surrounding buckets
        lower = max([b for b in buckets if b <= confidence_pct], default=None)
        upper = min([b for b in buckets if b >= confidence_pct], default=None)
        
        if lower is None and upper is None:
            return confidence_pct / 100.0
        elif lower is None:
            return self.calibration_curve[upper]
        elif upper is None:
            return self.calibration_curve[lower]
        elif lower == upper:
            return self.calibration_curve[lower]
        else:
            # Linear interpolation
            lower_rate = self.calibration_curve[lower]
            upper_rate = self.calibration_curve[upper]
            weight = (confidence_pct - lower) / (upper - lower)
            return lower_rate + weight * (upper_rate - lower_rate)
    
    def get_calibration_stats(self) -> Dict:
        """Get calibration statistics"""
        stats = {
            'total_trades': len(self.trade_history),
            'calibration_buckets': len(self.calibration_curve),
            'bucket_details': {}
        }
        
        for bucket in sorted(self.confidence_buckets.keys()):
            data = self.confidence_buckets[bucket]
            if data['total'] > 0:
                stats['bucket_details'][bucket] = {
                    'predicted_confidence': f"{bucket}%",
                    'actual_win_rate': f"{data['wins']/data['total']*100:.1f}%",
                    'total_trades': data['total'],
                    'wins': data['wins'],
                    'losses': data['losses'],
                    'calibrated_confidence': f"{self.calibration_curve.get(bucket, bucket/100)*100:.1f}%"
                }
        
        return stats
    
    def save_calibration(self):
        """Save calibration data to file"""
        data = {
            'calibration_curve': self.calibration_curve,
            'buckets': {
                str(k): v for k, v in self.confidence_buckets.items()
            }
        }
        
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save calibration: {e}")
    
    def load_calibration(self):
        """Load calibration data from file"""
        if not os.path.exists(self.calibration_file):
            return
        
        try:
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)
            
            # Convert keys back to integers
            self.calibration_curve = {
                int(k): v for k, v in data.get('calibration_curve', {}).items()
            }
            
            for k, v in data.get('buckets', {}).items():
                self.confidence_buckets[int(k)] = v
            
            print(f"✅ Loaded calibration data: {len(self.calibration_curve)} buckets")
        except Exception as e:
            print(f"⚠️ Failed to load calibration: {e}")
    
    def get_confidence_adjustment(self, confidence: float) -> Dict:
        """
        Get confidence adjustment info
        
        Returns:
            {
                'original': float,
                'calibrated': float,
                'adjustment': float,
                'reliability': str
            }
        """
        calibrated = self.calibrate_confidence(confidence)
        bucket = self._get_confidence_bucket(confidence)
        
        # Get reliability
        trades_in_bucket = self.confidence_buckets[bucket]['total']
        if trades_in_bucket >= 20:
            reliability = "HIGH"
        elif trades_in_bucket >= 10:
            reliability = "MEDIUM"
        elif trades_in_bucket >= 5:
            reliability = "LOW"
        else:
            reliability = "INSUFFICIENT_DATA"
        
        return {
            'original': confidence,
            'calibrated': calibrated,
            'adjustment': calibrated - confidence,
            'reliability': reliability,
            'trades_in_bucket': trades_in_bucket
        }
    
    def print_calibration_report(self):
        """Print detailed calibration report"""
        print("\n" + "="*80)
        print("🎯 CONFIDENCE CALIBRATION REPORT")
        print("="*80)
        
        stats = self.get_calibration_stats()
        
        print(f"\n📊 Total Trades Analyzed: {stats['total_trades']}")
        print(f"📈 Calibration Buckets: {stats['calibration_buckets']}")
        
        if stats['bucket_details']:
            print("\n" + "-"*80)
            print(f"{'Predicted':<15} {'Actual':<15} {'Calibrated':<15} {'Trades':<10} {'Status'}")
            print("-"*80)
            
            for bucket, details in sorted(stats['bucket_details'].items()):
                predicted = details['predicted_confidence']
                actual = details['actual_win_rate']
                calibrated = details['calibrated_confidence']
                trades = details['total_trades']
                
                # Color code status
                pred_val = int(predicted.rstrip('%'))
                act_val = float(actual.rstrip('%'))
                diff = abs(pred_val - act_val)
                
                if diff < 5:
                    status = "✅ Accurate"
                elif diff < 10:
                    status = "🟡 Slight Off"
                else:
                    status = "🔴 Needs Calibration"
                
                print(f"{predicted:<15} {actual:<15} {calibrated:<15} {trades:<10} {status}")
        
        print("="*80 + "\n")
