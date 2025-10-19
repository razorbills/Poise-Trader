#!/usr/bin/env python3
"""
🤖 MACHINE LEARNING COMPONENTS FOR ADVANCED AI TRADING
Neural Networks, Reinforcement Learning, and Pattern Recognition
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Advanced ML imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ Install tensorflow and scikit-learn for full ML features")

# Optional PyTorch detection for future GPU extensions
try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
    GPU_TORCH_AVAILABLE = torch.cuda.is_available()
except Exception:
    TORCH_AVAILABLE = False
    GPU_TORCH_AVAILABLE = False

# TensorFlow GPU detection and configuration
GPU_AVAILABLE = False
try:
    if 'tf' in globals():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

@dataclass
class MarketPattern:
    """Advanced market pattern recognition"""
    pattern_type: str
    confidence: float
    expected_move: float
    timeframe: int
    indicators: Dict[str, float]

@dataclass
class TradingSignalML:
    """ML-enhanced trading signal"""
    symbol: str
    action: str
    confidence: float
    ml_score: float
    nn_prediction: float
    pattern_score: float
    risk_reward: float
    optimal_size: float
    expected_return: float
    stop_loss: float
    take_profit: float

class NeuralPricePredictor:
    """Deep Learning model for price prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.sequence_length = 20
        self.is_trained = False
        self.use_gpu = GPU_AVAILABLE
        
        if ML_AVAILABLE:
            self._build_model()
    
    def _build_model(self):
        """Build LSTM neural network for price prediction with GPU support"""
        if not ML_AVAILABLE:
            return
        try:
            # Use all available GPUs if present, otherwise default strategy
            try:
                gpus = tf.config.list_physical_devices('GPU')
                strategy = tf.distribute.MirroredStrategy() if gpus else tf.distribute.get_strategy()
            except Exception:
                strategy = tf.distribute.get_strategy()
            
            with strategy.scope():
                self.model = keras.Sequential([
                    layers.LSTM(128, return_sequences=True, input_shape=(self.sequence_length, 7)),
                    layers.Dropout(0.2),
                    layers.LSTM(64, return_sequences=True),
                    layers.Dropout(0.2),
                    layers.LSTM(32),
                    layers.Dropout(0.2),
                    layers.Dense(16, activation='relu'),
                    layers.Dense(1)
                ])
                
                self.model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['mae']
                )
        except Exception:
            # Fallback to CPU-only build if strategy configuration fails
            self.model = keras.Sequential([
                layers.LSTM(64, return_sequences=True, input_shape=(self.sequence_length, 7)),
                layers.Dropout(0.2),
                layers.LSTM(32),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(1)
            ])
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
    
    def prepare_features(self, price_history: List[float], volume_history: List[float] = None) -> np.ndarray:
        """Prepare features for neural network"""
        if len(price_history) < self.sequence_length + 1:
            return None
        
        features = []
        for i in range(len(price_history) - self.sequence_length):
            sequence = price_history[i:i + self.sequence_length]
            
            # Calculate technical features
            returns = [(sequence[j] - sequence[j-1]) / sequence[j-1] if j > 0 else 0 
                      for j in range(len(sequence))]
            
            sma_5 = np.mean(sequence[-5:]) if len(sequence) >= 5 else sequence[-1]
            sma_10 = np.mean(sequence[-10:]) if len(sequence) >= 10 else sequence[-1]
            
            # RSI calculation
            gains = [r if r > 0 else 0 for r in returns]
            losses = [-r if r < 0 else 0 for r in returns]
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0
            rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))
            
            # Volatility
            volatility = np.std(returns) if len(returns) > 1 else 0
            
            # Create feature vector for each timestep
            seq_features = []
            for j in range(self.sequence_length):
                if j < len(sequence):
                    seq_features.append([
                        sequence[j],
                        returns[j] if j < len(returns) else 0,
                        sma_5,
                        sma_10,
                        rsi,
                        volatility,
                        volume_history[i+j] if volume_history and i+j < len(volume_history) else 1.0
                    ])
            
            features.append(seq_features)
        
        return np.array(features) if features else None
    
    def train(self, price_data: Dict[str, List[float]], epochs: int = 50):
        """Train the neural network on historical data"""
        if not ML_AVAILABLE or not price_data:
            return
        
        all_features = []
        all_targets = []
        
        for symbol, prices in price_data.items():
            features = self.prepare_features(prices)
            if features is not None and len(features) > 0:
                # Target is next price movement
                targets = [prices[i + self.sequence_length] for i in range(len(prices) - self.sequence_length)]
                all_features.extend(features)
                all_targets.extend(targets)
        
        if all_features:
            X = np.array(all_features)
            y = np.array(all_targets)
            
            # Normalize features
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.fit_transform(X_reshaped)
            X = X_scaled.reshape(X.shape)
            
            # Train model
            batch_size = 128 if GPU_AVAILABLE else 32
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
            self.is_trained = True
    
    def predict(self, price_history: List[float]) -> Tuple[float, float]:
        """Predict next price and confidence"""
        if not ML_AVAILABLE or not self.is_trained or len(price_history) < self.sequence_length:
            return price_history[-1] if price_history else 0, 0.5
        
        features = self.prepare_features(price_history)
        if features is None or len(features) == 0:
            return price_history[-1], 0.5
        
        # Use last sequence for prediction
        X = features[-1:] 
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped)
        X = X_scaled.reshape(X.shape)
        
        prediction = self.model.predict(X, verbose=0)[0][0]
        
        # Calculate confidence based on recent accuracy
        confidence = 0.6
        
        return float(prediction), float(confidence)

class ReinforcementLearningOptimizer:
    """Q-Learning based strategy optimizer"""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = 0.1  # Exploration rate
        self.action_space = ['BUY', 'SELL', 'HOLD']
        self.state_history = deque(maxlen=1000)
        
    def get_state(self, market_data: Dict) -> str:
        """Convert market conditions to state"""
        rsi = market_data.get('rsi', 50)
        trend = market_data.get('trend', 'neutral')
        volatility = market_data.get('volatility', 'medium')
        
        # Discretize RSI
        if rsi < 30:
            rsi_state = 'oversold'
        elif rsi > 70:
            rsi_state = 'overbought'
        else:
            rsi_state = 'neutral'
        
        return f"{rsi_state}_{trend}_{volatility}"
    
    def choose_action(self, state: str, training: bool = False) -> str:
        """Choose action using epsilon-greedy strategy"""
        if training and np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        
        # Get best action from Q-table
        state_actions = self.q_table[state]
        if not state_actions:
            return 'HOLD'
        
        return max(state_actions.items(), key=lambda x: x[1])[0]
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value using Q-learning formula"""
        current_q = self.q_table[state][action]
        
        # Get max Q-value for next state
        next_state_actions = self.q_table[next_state]
        max_next_q = max(next_state_actions.values()) if next_state_actions else 0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        # Store in history
        self.state_history.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'q_value': new_q
        })
    
    def calculate_reward(self, trade_result: Dict) -> float:
        """Calculate reward from trade result"""
        pnl = trade_result.get('pnl', 0)
        risk = trade_result.get('risk', 1)
        
        # Risk-adjusted reward
        reward = pnl / risk if risk > 0 else pnl
        
        # Add bonus for profitable trades
        if pnl > 0:
            reward *= 1.2
        
        return reward

class PatternRecognitionEngine:
    """Advanced pattern recognition for technical analysis"""
    
    def __init__(self):
        self.patterns = {
            'head_and_shoulders': self._detect_head_shoulders,
            'double_top': self._detect_double_top,
            'double_bottom': self._detect_double_bottom,
            'triangle': self._detect_triangle,
            'flag': self._detect_flag,
            'wedge': self._detect_wedge
        }
        self.ml_model = None
        if ML_AVAILABLE:
            self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def detect_patterns(self, price_history: List[float]) -> List[MarketPattern]:
        """Detect all patterns in price history"""
        if len(price_history) < 20:
            return []
        
        detected_patterns = []
        for pattern_name, detector in self.patterns.items():
            pattern = detector(price_history)
            if pattern:
                detected_patterns.append(pattern)
        
        return detected_patterns
    
    def _detect_head_shoulders(self, prices: List[float]) -> MarketPattern:
        """Detect head and shoulders pattern"""
        if len(prices) < 15:
            return None
        
        # Find peaks and troughs
        peaks = []
        troughs = []
        
        for i in range(2, len(prices) - 2):
            if prices[i] > prices[i-1] and prices[i] > prices[i-2] and prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                peaks.append((i, prices[i]))
            elif prices[i] < prices[i-1] and prices[i] < prices[i-2] and prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                troughs.append((i, prices[i]))
        
        # Check for head and shoulders formation
        if len(peaks) >= 3 and len(troughs) >= 2:
            # Check if middle peak is highest (head)
            if peaks[1][1] > peaks[0][1] and peaks[1][1] > peaks[2][1]:
                # Check if shoulders are roughly equal
                shoulder_diff = abs(peaks[0][1] - peaks[2][1]) / peaks[0][1]
                if shoulder_diff < 0.05:  # Within 5%
                    confidence = 0.8 - shoulder_diff * 2
                    expected_move = -(peaks[1][1] - np.mean([t[1] for t in troughs])) / peaks[1][1]
                    
                    return MarketPattern(
                        pattern_type='head_and_shoulders',
                        confidence=confidence,
                        expected_move=expected_move,
                        timeframe=peaks[2][0] - peaks[0][0],
                        indicators={'neckline': np.mean([t[1] for t in troughs])}
                    )
        
        return None
    
    def _detect_double_top(self, prices: List[float]) -> MarketPattern:
        """Detect double top pattern"""
        if len(prices) < 10:
            return None
        
        # Find two peaks of similar height
        peaks = []
        for i in range(2, len(prices) - 2):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append((i, prices[i]))
        
        if len(peaks) >= 2:
            # Check if peaks are similar height
            peak_diff = abs(peaks[-1][1] - peaks[-2][1]) / peaks[-2][1]
            if peak_diff < 0.03:  # Within 3%
                confidence = 0.75 - peak_diff * 5
                expected_move = -0.05  # Expect 5% drop
                
                return MarketPattern(
                    pattern_type='double_top',
                    confidence=confidence,
                    expected_move=expected_move,
                    timeframe=peaks[-1][0] - peaks[-2][0],
                    indicators={'resistance': np.mean([peaks[-1][1], peaks[-2][1]])}
                )
        
        return None
    
    def _detect_double_bottom(self, prices: List[float]) -> MarketPattern:
        """Detect double bottom pattern"""
        if len(prices) < 10:
            return None
        
        # Find two troughs of similar depth
        troughs = []
        for i in range(2, len(prices) - 2):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                troughs.append((i, prices[i]))
        
        if len(troughs) >= 2:
            # Check if troughs are similar depth
            trough_diff = abs(troughs[-1][1] - troughs[-2][1]) / troughs[-2][1]
            if trough_diff < 0.03:  # Within 3%
                confidence = 0.75 - trough_diff * 5
                expected_move = 0.05  # Expect 5% rise
                
                return MarketPattern(
                    pattern_type='double_bottom',
                    confidence=confidence,
                    expected_move=expected_move,
                    timeframe=troughs[-1][0] - troughs[-2][0],
                    indicators={'support': np.mean([troughs[-1][1], troughs[-2][1]])}
                )
        
        return None
    
    def _detect_triangle(self, prices: List[float]) -> MarketPattern:
        """Detect triangle pattern"""
        if len(prices) < 15:
            return None
        
        # Calculate trend lines
        highs = []
        lows = []
        
        for i in range(len(prices)):
            if i == 0 or prices[i] > prices[i-1]:
                highs.append(prices[i])
            if i == 0 or prices[i] < prices[i-1]:
                lows.append(prices[i])
        
        if len(highs) >= 3 and len(lows) >= 3:
            # Check for converging trend lines
            high_slope = (highs[-1] - highs[0]) / len(highs)
            low_slope = (lows[-1] - lows[0]) / len(lows)
            
            if abs(high_slope) < abs(low_slope):  # Converging
                confidence = 0.7
                expected_move = 0.03 if low_slope > 0 else -0.03
                
                return MarketPattern(
                    pattern_type='triangle',
                    confidence=confidence,
                    expected_move=expected_move,
                    timeframe=len(prices),
                    indicators={'apex': (highs[-1] + lows[-1]) / 2}
                )
        
        return None
    
    def _detect_flag(self, prices: List[float]) -> MarketPattern:
        """Detect flag pattern"""
        if len(prices) < 10:
            return None
        
        # Look for strong move followed by consolidation
        initial_move = (prices[5] - prices[0]) / prices[0]
        consolidation = np.std(prices[5:]) / np.mean(prices[5:])
        
        if abs(initial_move) > 0.05 and consolidation < 0.02:
            confidence = 0.65
            expected_move = initial_move * 0.5  # Expect continuation
            
            return MarketPattern(
                pattern_type='flag',
                confidence=confidence,
                expected_move=expected_move,
                timeframe=len(prices),
                indicators={'breakout_level': prices[-1] * (1 + initial_move * 0.1)}
            )
        
        return None
    
    def _detect_wedge(self, prices: List[float]) -> MarketPattern:
        """Detect wedge pattern"""
        if len(prices) < 12:
            return None
        
        # Calculate converging trend lines with same direction
        highs = [prices[i] for i in range(len(prices)) if i == 0 or prices[i] > prices[i-1]]
        lows = [prices[i] for i in range(len(prices)) if i == 0 or prices[i] < prices[i-1]]
        
        if len(highs) >= 3 and len(lows) >= 3:
            high_slope = (highs[-1] - highs[0]) / len(highs)
            low_slope = (lows[-1] - lows[0]) / len(lows)
            
            # Both slopes same direction and converging
            if high_slope * low_slope > 0 and abs(high_slope - low_slope) < abs(high_slope) * 0.5:
                confidence = 0.6
                expected_move = -high_slope * 2  # Expect reversal
                
                return MarketPattern(
                    pattern_type='wedge',
                    confidence=confidence,
                    expected_move=expected_move,
                    timeframe=len(prices),
                    indicators={'breakout_direction': 'down' if high_slope > 0 else 'up'}
                )
        
        return None

# Global ML components
neural_predictor = NeuralPricePredictor()
rl_optimizer = ReinforcementLearningOptimizer()
pattern_engine = PatternRecognitionEngine()
