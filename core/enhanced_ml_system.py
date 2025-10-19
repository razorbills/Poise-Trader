#!/usr/bin/env python3
"""
🤖 ENHANCED ML SYSTEM WITH MODEL PERSISTENCE & ONLINE LEARNING
Advanced machine learning with persistent models and continuous adaptation
"""

import os
import pickle
import json
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
from pathlib import Path

# ML imports with fallbacks
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ml_components import TradingSignalML, MarketPattern

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Metadata for saved models"""
    model_name: str
    model_type: str
    created_at: datetime
    last_trained: datetime
    training_samples: int
    performance_metrics: Dict[str, float]
    feature_columns: List[str]
    target_column: str
    version: str = "1.0"

@dataclass
class OnlineLearningConfig:
    """Configuration for online learning"""
    update_frequency_minutes: int = 30
    min_samples_for_update: int = 50
    performance_threshold: float = 0.05
    max_model_age_hours: int = 24
    enable_drift_detection: bool = True
    drift_threshold: float = 0.1

class ModelRegistry:
    """Registry for managing ML models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self._models: Dict[str, Any] = {}
        self._metadata: Dict[str, ModelMetadata] = {}
        self._scalers: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        self.load_all_models()
    
    def save_model(self, name: str, model: Any, metadata: ModelMetadata, 
                   scaler: Any = None) -> bool:
        """Save model with metadata"""
        try:
            with self._lock:
                model_path = self.models_dir / f"{name}.pkl"
                metadata_path = self.models_dir / f"{name}_metadata.json"
                scaler_path = self.models_dir / f"{name}_scaler.pkl"
                
                # Save model
                if TF_AVAILABLE and isinstance(model, keras.Model):
                    model_dir = self.models_dir / f"{name}_tf"
                    model.save(str(model_dir))
                else:
                    joblib.dump(model, model_path)
                
                # Save metadata
                metadata_dict = asdict(metadata)
                metadata_dict['created_at'] = metadata.created_at.isoformat()
                metadata_dict['last_trained'] = metadata.last_trained.isoformat()
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata_dict, f, indent=2)
                
                # Save scaler if provided
                if scaler is not None:
                    joblib.dump(scaler, scaler_path)
                    self._scalers[name] = scaler
                
                # Update registry
                self._models[name] = model
                self._metadata[name] = metadata
                
                logger.info(f"💾 Saved model: {name}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to save model {name}: {e}")
            return False
    
    def load_model(self, name: str) -> Tuple[Any, ModelMetadata, Any]:
        """Load model with metadata and scaler"""
        try:
            with self._lock:
                if name in self._models:
                    return self._models[name], self._metadata[name], self._scalers.get(name)
                
                model_path = self.models_dir / f"{name}.pkl"
                metadata_path = self.models_dir / f"{name}_metadata.json"
                scaler_path = self.models_dir / f"{name}_scaler.pkl"
                tf_model_dir = self.models_dir / f"{name}_tf"
                
                # Load metadata
                if not metadata_path.exists():
                    raise FileNotFoundError(f"Metadata not found for model {name}")
                
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                
                metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
                metadata_dict['last_trained'] = datetime.fromisoformat(metadata_dict['last_trained'])
                metadata = ModelMetadata(**metadata_dict)
                
                # Load model
                model = None
                if tf_model_dir.exists() and TF_AVAILABLE:
                    model = keras.models.load_model(str(tf_model_dir))
                elif model_path.exists():
                    model = joblib.load(model_path)
                else:
                    raise FileNotFoundError(f"Model file not found for {name}")
                
                # Load scaler
                scaler = None
                if scaler_path.exists():
                    scaler = joblib.load(scaler_path)
                
                # Cache in registry
                self._models[name] = model
                self._metadata[name] = metadata
                if scaler:
                    self._scalers[name] = scaler
                
                logger.info(f"📥 Loaded model: {name}")
                return model, metadata, scaler
                
        except Exception as e:
            logger.error(f"❌ Failed to load model {name}: {e}")
            return None, None, None
    
    def load_all_models(self):
        """Load all available models"""
        if not self.models_dir.exists():
            return
        
        metadata_files = list(self.models_dir.glob("*_metadata.json"))
        
        for metadata_file in metadata_files:
            model_name = metadata_file.stem.replace("_metadata", "")
            try:
                self.load_model(model_name)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load model {model_name}: {e}")
        
        logger.info(f"📚 Loaded {len(self._models)} models from registry")
    
    def list_models(self) -> List[str]:
        """List available models"""
        return list(self._models.keys())
    
    def get_model_info(self, name: str) -> Optional[ModelMetadata]:
        """Get model metadata"""
        return self._metadata.get(name)
    
    def delete_model(self, name: str) -> bool:
        """Delete model and associated files"""
        try:
            with self._lock:
                # Remove files
                files_to_remove = [
                    self.models_dir / f"{name}.pkl",
                    self.models_dir / f"{name}_metadata.json",
                    self.models_dir / f"{name}_scaler.pkl"
                ]
                
                tf_model_dir = self.models_dir / f"{name}_tf"
                if tf_model_dir.exists():
                    import shutil
                    shutil.rmtree(tf_model_dir)
                
                for file_path in files_to_remove:
                    if file_path.exists():
                        file_path.unlink()
                
                # Remove from registry
                self._models.pop(name, None)
                self._metadata.pop(name, None)
                self._scalers.pop(name, None)
                
                logger.info(f"🗑️ Deleted model: {name}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to delete model {name}: {e}")
            return False

class OnlineLearningEngine:
    """Online learning engine for continuous model improvement"""
    
    def __init__(self, config: OnlineLearningConfig = None):
        self.config = config or OnlineLearningConfig()
        self.model_registry = ModelRegistry()
        
        # Learning data buffers
        self._training_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Drift detection
        self._baseline_distributions: Dict[str, Dict] = {}
        self._drift_detected: Dict[str, bool] = defaultdict(bool)
        
        # Learning state
        self._last_update: Dict[str, datetime] = {}
        self._update_lock = threading.Lock()
        self._learning_active = False
        self._learning_thread = None
        
    def start_online_learning(self):
        """Start online learning background process"""
        if not self._learning_active:
            self._learning_active = True
            self._learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
            self._learning_thread.start()
            logger.info("🎯 Online learning engine started")
    
    def stop_online_learning(self):
        """Stop online learning"""
        self._learning_active = False
        if self._learning_thread:
            self._learning_thread.join(timeout=5)
        logger.info("🎯 Online learning engine stopped")
    
    def add_training_sample(self, model_name: str, features: List[float], 
                          target: float, metadata: Dict = None):
        """Add new training sample to buffer"""
        sample = {
            'features': features,
            'target': target,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        self._training_buffer[model_name].append(sample)
        
        # Check if we need immediate update
        if len(self._training_buffer[model_name]) >= self.config.min_samples_for_update:
            self._schedule_model_update(model_name)
    
    def add_prediction_result(self, model_name: str, predicted: float, 
                            actual: float, features: List[float]):
        """Add prediction result for performance tracking"""
        error = abs(predicted - actual)
        relative_error = error / max(abs(actual), 1e-8)
        
        result = {
            'predicted': predicted,
            'actual': actual,
            'error': error,
            'relative_error': relative_error,
            'timestamp': datetime.now(),
            'features': features
        }
        
        self._performance_history[model_name].append(result)
        
        # Check for concept drift
        if self.config.enable_drift_detection:
            self._check_concept_drift(model_name, features)
    
    def _learning_loop(self):
        """Main learning loop"""
        while self._learning_active:
            try:
                for model_name in list(self._training_buffer.keys()):
                    if self._should_update_model(model_name):
                        self._update_model(model_name)
                
                # Sleep until next check
                time.sleep(self.config.update_frequency_minutes * 60)
                
            except Exception as e:
                logger.error(f"Learning loop error: {e}")
                time.sleep(60)  # Wait a minute on error
    
    def _should_update_model(self, model_name: str) -> bool:
        """Check if model should be updated"""
        # Check if enough time has passed
        last_update = self._last_update.get(model_name)
        if last_update:
            time_since_update = datetime.now() - last_update
            if time_since_update.total_seconds() < self.config.update_frequency_minutes * 60:
                return False
        
        # Check if we have enough samples
        if len(self._training_buffer[model_name]) < self.config.min_samples_for_update:
            return False
        
        # Check if performance has degraded
        if self._has_performance_degraded(model_name):
            return True
        
        # Check for concept drift
        if self._drift_detected[model_name]:
            return True
        
        # Check model age
        metadata = self.model_registry.get_model_info(model_name)
        if metadata:
            model_age = datetime.now() - metadata.last_trained
            if model_age.total_seconds() > self.config.max_model_age_hours * 3600:
                return True
        
        return False
    
    def _update_model(self, model_name: str):
        """Update model with new training data"""
        try:
            with self._update_lock:
                logger.info(f"🔄 Updating model: {model_name}")
                
                # Get current model
                model, metadata, scaler = self.model_registry.load_model(model_name)
                if model is None:
                    logger.warning(f"Model {model_name} not found, creating new one")
                    model, scaler = self._create_new_model(model_name)
                    if model is None:
                        return
                
                # Prepare training data
                samples = list(self._training_buffer[model_name])
                X = np.array([sample['features'] for sample in samples])
                y = np.array([sample['target'] for sample in samples])
                
                if len(X) == 0:
                    return
                
                # Scale features if scaler exists
                if scaler is not None:
                    X = scaler.fit_transform(X)
                
                # Train model
                if hasattr(model, 'partial_fit'):
                    # Online learning capable model
                    model.partial_fit(X, y)
                else:
                    # Retrain from scratch with recent data
                    model.fit(X, y)
                
                # Evaluate performance
                performance_metrics = self._evaluate_model(model, X, y)
                
                # Update metadata
                new_metadata = ModelMetadata(
                    model_name=model_name,
                    model_type=type(model).__name__,
                    created_at=metadata.created_at if metadata else datetime.now(),
                    last_trained=datetime.now(),
                    training_samples=len(samples),
                    performance_metrics=performance_metrics,
                    feature_columns=metadata.feature_columns if metadata else [f"feature_{i}" for i in range(X.shape[1])],
                    target_column=metadata.target_column if metadata else "target"
                )
                
                # Save updated model
                self.model_registry.save_model(model_name, model, new_metadata, scaler)
                
                # Clear processed samples
                self._training_buffer[model_name].clear()
                self._last_update[model_name] = datetime.now()
                self._drift_detected[model_name] = False
                
                logger.info(f"✅ Model {model_name} updated. Performance: {performance_metrics}")
                
        except Exception as e:
            logger.error(f"❌ Failed to update model {model_name}: {e}")
    
    def _create_new_model(self, model_name: str) -> Tuple[Any, Any]:
        """Create new model for online learning"""
        try:
            if SKLEARN_AVAILABLE:
                # Use SGD-based models for online learning
                from sklearn.linear_model import SGDRegressor
                model = SGDRegressor(learning_rate='adaptive', eta0=0.01)
                scaler = StandardScaler()
                return model, scaler
            else:
                logger.error("Scikit-learn not available for model creation")
                return None, None
                
        except Exception as e:
            logger.error(f"Failed to create new model: {e}")
            return None, None
    
    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            predictions = model.predict(X)
            
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)
            mae = np.mean(np.abs(y - predictions))
            
            return {
                'mse': float(mse),
                'r2': float(r2),
                'mae': float(mae),
                'rmse': float(np.sqrt(mse))
            }
            
        except Exception as e:
            logger.error(f"Model evaluation error: {e}")
            return {'mse': float('inf'), 'r2': 0.0, 'mae': float('inf'), 'rmse': float('inf')}
    
    def _has_performance_degraded(self, model_name: str) -> bool:
        """Check if model performance has degraded"""
        if model_name not in self._performance_history:
            return False
        
        recent_results = list(self._performance_history[model_name])[-100:]  # Last 100 predictions
        if len(recent_results) < 20:
            return False
        
        recent_errors = [r['relative_error'] for r in recent_results[-20:]]
        older_errors = [r['relative_error'] for r in recent_results[-100:-20]]
        
        if len(older_errors) == 0:
            return False
        
        recent_avg_error = np.mean(recent_errors)
        older_avg_error = np.mean(older_errors)
        
        # Check if error increased significantly
        error_increase = (recent_avg_error - older_avg_error) / max(older_avg_error, 1e-8)
        
        return error_increase > self.config.performance_threshold
    
    def _check_concept_drift(self, model_name: str, features: List[float]):
        """Check for concept drift using feature distribution monitoring"""
        try:
            if model_name not in self._baseline_distributions:
                # Initialize baseline
                self._baseline_distributions[model_name] = {
                    'mean': np.array(features),
                    'std': np.ones_like(features),
                    'count': 1
                }
                return
            
            baseline = self._baseline_distributions[model_name]
            
            # Update baseline statistics incrementally
            count = baseline['count']
            old_mean = baseline['mean']
            
            # Incremental mean and std update
            new_count = count + 1
            new_mean = (old_mean * count + np.array(features)) / new_count
            
            # Simple drift detection using mean shift
            mean_shift = np.linalg.norm(new_mean - old_mean)
            
            if mean_shift > self.config.drift_threshold:
                self._drift_detected[model_name] = True
                logger.warning(f"⚠️ Concept drift detected for model {model_name}")
            
            # Update baseline
            baseline['mean'] = new_mean
            baseline['count'] = new_count
            
        except Exception as e:
            logger.error(f"Drift detection error: {e}")
    
    def _schedule_model_update(self, model_name: str):
        """Schedule immediate model update"""
        # This could trigger an immediate update in a separate thread
        # For now, we'll just mark it for the next update cycle
        pass
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        stats = {
            'models': {},
            'total_models': len(self._training_buffer),
            'learning_active': self._learning_active,
            'config': asdict(self.config)
        }
        
        for model_name in self._training_buffer.keys():
            buffer_size = len(self._training_buffer[model_name])
            perf_history_size = len(self._performance_history[model_name])
            drift_detected = self._drift_detected[model_name]
            last_update = self._last_update.get(model_name)
            
            stats['models'][model_name] = {
                'buffer_size': buffer_size,
                'performance_history_size': perf_history_size,
                'drift_detected': drift_detected,
                'last_update': last_update.isoformat() if last_update else None,
                'should_update': self._should_update_model(model_name)
            }
        
        return stats

class AdvancedFeatureEngineering:
    """Advanced feature engineering for trading signals"""
    
    def __init__(self):
        self.feature_generators = {
            'technical_indicators': self._generate_technical_features,
            'price_patterns': self._generate_pattern_features,
            'volume_features': self._generate_volume_features,
            'market_microstructure': self._generate_microstructure_features
        }
    
    def generate_features(self, price_data: List[float], volume_data: List[float] = None,
                         additional_data: Dict[str, List[float]] = None) -> Dict[str, float]:
        """Generate comprehensive feature set"""
        features = {}
        
        for feature_type, generator in self.feature_generators.items():
            try:
                if feature_type == 'volume_features' and volume_data is None:
                    continue
                
                type_features = generator(price_data, volume_data, additional_data)
                features.update(type_features)
                
            except Exception as e:
                logger.warning(f"Feature generation error for {feature_type}: {e}")
        
        return features
    
    def _generate_technical_features(self, prices: List[float], volumes: List[float] = None,
                                   additional_data: Dict = None) -> Dict[str, float]:
        """Generate technical indicator features"""
        if len(prices) < 20:
            return {}
        
        prices_array = np.array(prices)
        features = {}
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            if len(prices) >= period:
                ma = np.mean(prices_array[-period:])
                features[f'sma_{period}'] = ma
                features[f'price_to_sma_{period}'] = prices_array[-1] / ma
        
        # RSI
        if len(prices) >= 14:
            deltas = np.diff(prices_array)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                features['rsi'] = rsi
        
        # Bollinger Bands
        if len(prices) >= 20:
            sma_20 = np.mean(prices_array[-20:])
            std_20 = np.std(prices_array[-20:])
            
            features['bollinger_upper'] = sma_20 + (2 * std_20)
            features['bollinger_lower'] = sma_20 - (2 * std_20)
            features['bollinger_position'] = (prices_array[-1] - features['bollinger_lower']) / (features['bollinger_upper'] - features['bollinger_lower'])
        
        # Volatility
        if len(prices) >= 10:
            returns = np.diff(prices_array) / prices_array[:-1]
            features['volatility_10'] = np.std(returns[-10:])
            features['volatility_20'] = np.std(returns[-20:]) if len(returns) >= 20 else features['volatility_10']
        
        return features
    
    def _generate_pattern_features(self, prices: List[float], volumes: List[float] = None,
                                 additional_data: Dict = None) -> Dict[str, float]:
        """Generate price pattern features"""
        if len(prices) < 10:
            return {}
        
        prices_array = np.array(prices)
        features = {}
        
        # Recent price changes
        features['price_change_1'] = (prices_array[-1] - prices_array[-2]) / prices_array[-2] if len(prices) >= 2 else 0
        features['price_change_5'] = (prices_array[-1] - prices_array[-6]) / prices_array[-6] if len(prices) >= 6 else 0
        features['price_change_10'] = (prices_array[-1] - prices_array[-11]) / prices_array[-11] if len(prices) >= 11 else 0
        
        # Trend strength
        if len(prices) >= 10:
            x = np.arange(10)
            y = prices_array[-10:]
            slope = np.polyfit(x, y, 1)[0]
            features['trend_strength'] = slope / np.mean(y)
        
        # Support/Resistance levels
        if len(prices) >= 20:
            recent_high = np.max(prices_array[-20:])
            recent_low = np.min(prices_array[-20:])
            
            features['distance_to_high'] = (recent_high - prices_array[-1]) / recent_high
            features['distance_to_low'] = (prices_array[-1] - recent_low) / recent_low
        
        return features
    
    def _generate_volume_features(self, prices: List[float], volumes: List[float] = None,
                                additional_data: Dict = None) -> Dict[str, float]:
        """Generate volume-based features"""
        if volumes is None or len(volumes) < 10:
            return {}
        
        volumes_array = np.array(volumes)
        features = {}
        
        # Volume moving averages
        features['volume_sma_10'] = np.mean(volumes_array[-10:])
        features['volume_ratio'] = volumes_array[-1] / features['volume_sma_10']
        
        # Price-volume relationship
        if len(prices) == len(volumes) and len(prices) >= 5:
            price_changes = np.diff(prices[-5:])
            volume_changes = np.diff(volumes_array[-5:])
            
            if len(price_changes) > 0 and len(volume_changes) > 0:
                correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
                features['price_volume_correlation'] = correlation if not np.isnan(correlation) else 0
        
        return features
    
    def _generate_microstructure_features(self, prices: List[float], volumes: List[float] = None,
                                        additional_data: Dict = None) -> Dict[str, float]:
        """Generate market microstructure features"""
        features = {}
        
        if len(prices) >= 5:
            # Price acceleration
            if len(prices) >= 3:
                price_changes = np.diff(prices[-3:])
                if len(price_changes) >= 2:
                    acceleration = np.diff(price_changes)[0]
                    features['price_acceleration'] = acceleration
        
        # Add more microstructure features as needed
        return features

# Global instances
model_registry = ModelRegistry()
online_learning_engine = OnlineLearningEngine()
feature_engineer = AdvancedFeatureEngineering()
