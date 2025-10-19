"""
🔮 LSTM PRICE PREDICTION AI
Predicts price movement in next 5-30 minutes
Neural network-based forecasting
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import deque


class LSTMPricePredictor:
    """
    🧠 Simplified LSTM-Style Price Prediction
    
    Uses weighted moving averages and momentum to predict future prices
    (Simplified version - full LSTM requires TensorFlow/PyTorch)
    """
    
    def __init__(self):
        self.prediction_history = deque(maxlen=100)
        self.accuracy_tracker = deque(maxlen=50)
        
    def predict_price(self, prices: List[float], time_horizon: int = 5) -> Dict:
        """
        🔮 Predict future price
        
        Args:
            prices: Recent price history (50-100 bars)
            time_horizon: Minutes ahead to predict (5, 10, 30)
            
        Returns:
            {
                'predicted_price': float,
                'predicted_change': float (%),
                'confidence': float (0-1),
                'direction': 'UP', 'DOWN', or 'NEUTRAL',
                'prediction_range': (low, high)
            }
        """
        if len(prices) < 20:
            return self._default_prediction(prices[-1] if prices else 0)
        
        prices = np.array(prices)
        current_price = prices[-1]
        
        # Calculate multiple trend indicators
        short_term_trend = self._calculate_trend(prices[-10:])
        medium_term_trend = self._calculate_trend(prices[-20:])
        long_term_trend = self._calculate_trend(prices[-50:]) if len(prices) >= 50 else medium_term_trend
        
        # Calculate momentum
        momentum = self._calculate_momentum(prices)
        
        # Calculate volatility
        volatility = self._calculate_volatility(prices)
        
        # Weighted prediction
        trend_weight = 0.5
        momentum_weight = 0.3
        volatility_weight = 0.2
        
        # Combine trends with exponential weighting (recent = more important)
        combined_trend = (
            short_term_trend * 0.5 +
            medium_term_trend * 0.3 +
            long_term_trend * 0.2
        )
        
        # Predict price change
        predicted_change_pct = (
            combined_trend * trend_weight +
            momentum * momentum_weight
        ) * (time_horizon / 5)  # Scale by time horizon
        
        # Add volatility buffer
        volatility_buffer = volatility * volatility_weight * (time_horizon / 5)
        
        # Calculate predicted price
        predicted_price = current_price * (1 + predicted_change_pct)
        
        # Prediction range (confidence interval)
        prediction_low = current_price * (1 + predicted_change_pct - volatility_buffer)
        prediction_high = current_price * (1 + predicted_change_pct + volatility_buffer)
        
        # Determine direction
        if predicted_change_pct > 0.005:  # > 0.5%
            direction = 'UP'
        elif predicted_change_pct < -0.005:  # < -0.5%
            direction = 'DOWN'
        else:
            direction = 'NEUTRAL'
        
        # Calculate confidence based on trend consistency
        confidence = self._calculate_prediction_confidence(
            short_term_trend,
            medium_term_trend,
            long_term_trend,
            volatility
        )
        
        result = {
            'predicted_price': predicted_price,
            'predicted_change': predicted_change_pct * 100,  # As percentage
            'confidence': confidence,
            'direction': direction,
            'prediction_range': (prediction_low, prediction_high),
            'time_horizon_minutes': time_horizon,
            'current_price': current_price,
            'volatility': volatility
        }
        
        # Store prediction for accuracy tracking
        self.prediction_history.append({
            'timestamp': None,
            'predicted_price': predicted_price,
            'predicted_change': predicted_change_pct,
            'current_price': current_price,
            'time_horizon': time_horizon
        })
        
        return result
    
    def _calculate_trend(self, prices: np.ndarray) -> float:
        """Calculate linear trend slope"""
        if len(prices) < 2:
            return 0
        
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        # Normalize by average price
        trend = slope / np.mean(prices)
        
        return trend
    
    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate price momentum"""
        if len(prices) < 10:
            return 0
        
        recent_avg = np.mean(prices[-5:])
        older_avg = np.mean(prices[-10:-5])
        
        momentum = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        
        return momentum
    
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate price volatility"""
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        return volatility
    
    def _calculate_prediction_confidence(self, short_trend: float, 
                                         medium_trend: float, 
                                         long_trend: float,
                                         volatility: float) -> float:
        """
        Calculate confidence in prediction
        
        High confidence when:
        - All trends agree in direction
        - Low volatility
        """
        # Trend alignment
        trends = [short_trend, medium_trend, long_trend]
        
        # Check if all same sign
        if all(t > 0 for t in trends):
            alignment = 1.0  # Perfect bullish alignment
        elif all(t < 0 for t in trends):
            alignment = 1.0  # Perfect bearish alignment
        else:
            # Calculate alignment score
            avg_trend = np.mean(trends)
            std_trend = np.std(trends)
            alignment = max(0, 1 - std_trend * 10)
        
        # Volatility penalty (higher volatility = lower confidence)
        volatility_factor = max(0, 1 - volatility * 20)
        
        # Combined confidence
        confidence = (alignment * 0.7 + volatility_factor * 0.3)
        
        return max(0, min(1, confidence))
    
    def _default_prediction(self, current_price: float) -> Dict:
        """Default prediction when insufficient data"""
        return {
            'predicted_price': current_price,
            'predicted_change': 0.0,
            'confidence': 0.50,
            'direction': 'NEUTRAL',
            'prediction_range': (current_price * 0.99, current_price * 1.01),
            'time_horizon_minutes': 5,
            'current_price': current_price,
            'volatility': 0.02
        }
    
    def verify_prediction(self, predicted_price: float, actual_price: float):
        """Verify prediction accuracy"""
        error_pct = abs(predicted_price - actual_price) / actual_price if actual_price > 0 else 0
        
        # Consider accurate if within 1%
        accurate = error_pct < 0.01
        
        self.accuracy_tracker.append({
            'error_pct': error_pct,
            'accurate': accurate
        })
        
        return accurate
    
    def get_prediction_accuracy(self) -> Dict:
        """Get prediction accuracy statistics"""
        if not self.accuracy_tracker:
            return {'accuracy': 0.0, 'avg_error': 0.0, 'sample_size': 0}
        
        accurate_count = sum(1 for p in self.accuracy_tracker if p['accurate'])
        accuracy = accurate_count / len(self.accuracy_tracker)
        avg_error = np.mean([p['error_pct'] for p in self.accuracy_tracker])
        
        return {
            'accuracy': accuracy,
            'avg_error_pct': avg_error * 100,
            'sample_size': len(self.accuracy_tracker)
        }
