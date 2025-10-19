#!/usr/bin/env python3
"""
⚙️ ADVANCED CONFIGURATION MANAGEMENT SYSTEM
Dynamic configuration with hot-reload, validation, and environment-based settings
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
from cryptography.fernet import Fernet
import base64
from pydantic import BaseModel, validator, Field
from enum import Enum

logger = logging.getLogger(__name__)

class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    enabled: bool = True
    api_key: str = ""
    secret_key: str = ""
    passphrase: str = ""
    testnet: bool = False
    trading_pairs: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    default_pair: str = "BTCUSDT"
    rate_limit_per_minute: int = 1200
    timeout_seconds: int = 30

@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_daily_loss: float = 0.03
    max_position_size: float = 0.15
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    portfolio_heat: float = 0.6
    correlation_limit: float = 0.7
    drawdown_limit: float = 0.10
    risk_free_rate: float = 0.02
    volatility_adjustment: bool = True
    dynamic_position_sizing: bool = True

@dataclass
class StrategyConfig:
    """Trading strategy configuration"""
    enabled: bool = True
    capital_allocation: float = 1000.0
    confidence_threshold: float = 0.6
    max_positions: int = 5
    rebalance_frequency_hours: int = 6
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AIConfig:
    """AI system configuration"""
    enabled: bool = True
    model_update_frequency_hours: int = 6
    min_training_samples: int = 100
    confidence_threshold: float = 0.7
    enable_online_learning: bool = True
    enable_meta_learning: bool = True
    max_model_age_hours: int = 48
    drift_detection_threshold: float = 0.1

@dataclass
class SystemConfig:
    """System configuration"""
    log_level: str = "INFO"
    max_workers: int = 6
    stealth_mode: bool = True
    process_name: str = "Windows Audio Service"
    cpu_affinity: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    memory_limit_mb: int = 2048
    data_retention_days: int = 30
    backup_frequency_hours: int = 12

class ConfigValidator(BaseModel):
    """Pydantic model for configuration validation"""
    
    # Trading configuration
    trading_mode: TradingMode = TradingMode.PAPER
    initial_capital: float = Field(gt=0, description="Initial capital must be positive")
    
    # Exchange configurations
    exchanges: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Risk management
    risk: Dict[str, Any] = Field(default_factory=dict)
    
    # Strategies
    strategies: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # AI configuration
    ai: Dict[str, Any] = Field(default_factory=dict)
    
    # System configuration
    system: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('initial_capital')
    def validate_capital(cls, v):
        if v <= 0:
            raise ValueError('Initial capital must be positive')
        return v
    
    @validator('exchanges')
    def validate_exchanges(cls, v):
        for exchange_name, config in v.items():
            if not config.get('enabled', False):
                continue
            if not config.get('api_key') and not config.get('testnet', False):
                logger.warning(f"Exchange {exchange_name} enabled but no API key provided")
        return v

class ConfigurationManager:
    """Advanced configuration management with hot-reload and encryption"""
    
    def __init__(self, config_dir: str = "config", environment: str = "development"):
        self.config_dir = Path(config_dir)
        self.environment = environment
        self.config_file = self.config_dir / f"config_{environment}.yaml"
        self.default_config_file = self.config_dir / "config.yaml"
        
        # Encryption
        self.encryption_key_file = self.config_dir / ".encryption_key"
        self.salt_file = self.config_dir / ".salt"
        self._cipher = None
        
        # Configuration state
        self._config: Dict[str, Any] = {}
        self._config_lock = threading.RLock()
        self._observers: List[callable] = []
        
        # File watching
        self._file_observer = None
        self._watching = False
        
        # Load initial configuration
        self._ensure_config_dir()
        self._setup_encryption()
        self.load_config()
        
    def _ensure_config_dir(self):
        """Ensure configuration directory exists"""
        self.config_dir.mkdir(exist_ok=True, parents=True)
        
        # Create default config if it doesn't exist
        if not self.default_config_file.exists():
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration file"""
        default_config = {
            'trading_mode': 'paper',
            'initial_capital': 5000.0,
            
            'exchanges': {
                'mexc': {
                    'enabled': True,
                    'api_key': '${MEXC_API_KEY}',
                    'secret_key': '${MEXC_SECRET_KEY}',
                    'testnet': False,
                    'trading_pairs': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
                    'default_pair': 'BTCUSDT'
                }
            },
            
            'risk': {
                'max_daily_loss': 0.03,
                'max_position_size': 0.15,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'portfolio_heat': 0.6,
                'dynamic_position_sizing': True
            },
            
            'strategies': {
                'momentum': {
                    'enabled': True,
                    'capital_allocation': 2000.0,
                    'confidence_threshold': 0.7,
                    'parameters': {
                        'lookback_period': 20,
                        'momentum_threshold': 0.02
                    }
                },
                'mean_reversion': {
                    'enabled': True,
                    'capital_allocation': 1500.0,
                    'confidence_threshold': 0.6,
                    'parameters': {
                        'rsi_threshold': 30,
                        'bollinger_factor': 2.0
                    }
                }
            },
            
            'ai': {
                'enabled': True,
                'model_update_frequency_hours': 6,
                'confidence_threshold': 0.7,
                'enable_online_learning': True,
                'enable_meta_learning': True
            },
            
            'system': {
                'log_level': 'INFO',
                'max_workers': 6,
                'stealth_mode': True,
                'memory_limit_mb': 2048,
                'data_retention_days': 30
            }
        }
        
        with open(self.default_config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Created default configuration: {self.default_config_file}")
    
    def _setup_encryption(self):
        """Setup encryption for sensitive configuration data"""
        try:
            if self.encryption_key_file.exists():
                with open(self.encryption_key_file, 'rb') as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                with open(self.encryption_key_file, 'wb') as f:
                    f.write(key)
                os.chmod(self.encryption_key_file, 0o600)  # Read only for owner
            
            self._cipher = Fernet(key)
            
        except Exception as e:
            logger.warning(f"Encryption setup failed: {e}")
            self._cipher = None
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from files"""
        with self._config_lock:
            try:
                # Load default config first
                config = {}
                if self.default_config_file.exists():
                    with open(self.default_config_file, 'r') as f:
                        config = yaml.safe_load(f) or {}
                
                # Override with environment-specific config
                if self.config_file.exists() and self.config_file != self.default_config_file:
                    with open(self.config_file, 'r') as f:
                        env_config = yaml.safe_load(f) or {}
                    config.update(env_config)
                
                # Expand environment variables
                config = self._expand_environment_variables(config)
                
                # Validate configuration
                validated_config = self._validate_config(config)
                
                self._config = validated_config
                
                logger.info(f"Configuration loaded successfully from {self.config_file}")
                
                # Notify observers
                self._notify_observers()
                
                return self._config.copy()
                
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                if not self._config:
                    # Use minimal fallback config
                    self._config = self._get_fallback_config()
                return self._config.copy()
    
    def _expand_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Expand environment variables in configuration"""
        def expand_value(value):
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                return os.getenv(env_var, value)
            elif isinstance(value, dict):
                return {k: expand_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [expand_value(item) for item in value]
            else:
                return value
        
        return expand_value(config)
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration using Pydantic"""
        try:
            validator = ConfigValidator(**config)
            return validator.dict()
        except Exception as e:
            logger.warning(f"Configuration validation failed: {e}")
            return config  # Return unvalidated config with warning
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Get minimal fallback configuration"""
        return {
            'trading_mode': 'paper',
            'initial_capital': 1000.0,
            'exchanges': {},
            'risk': {'max_daily_loss': 0.05},
            'strategies': {},
            'ai': {'enabled': False},
            'system': {'log_level': 'INFO'}
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        with self._config_lock:
            keys = key.split('.')
            value = self._config
            
            try:
                for k in keys:
                    value = value[k]
                return value
            except (KeyError, TypeError):
                return default
    
    def set(self, key: str, value: Any, persist: bool = True) -> bool:
        """Set configuration value with dot notation support"""
        with self._config_lock:
            try:
                keys = key.split('.')
                config = self._config
                
                # Navigate to parent dictionary
                for k in keys[:-1]:
                    if k not in config:
                        config[k] = {}
                    config = config[k]
                
                # Set the value
                config[keys[-1]] = value
                
                if persist:
                    self.save_config()
                
                # Notify observers
                self._notify_observers()
                
                logger.debug(f"Configuration updated: {key} = {value}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to set configuration {key}: {e}")
                return False
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        with self._config_lock:
            try:
                # Create backup
                if self.config_file.exists():
                    backup_file = self.config_dir / f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
                    import shutil
                    shutil.copy2(self.config_file, backup_file)
                
                # Save current config
                with open(self.config_file, 'w') as f:
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
                
                logger.info(f"Configuration saved to {self.config_file}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save configuration: {e}")
                return False
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive configuration data"""
        if self._cipher is None:
            return data
        
        try:
            encrypted = self._cipher.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive configuration data"""
        if self._cipher is None:
            return encrypted_data
        
        try:
            decoded = base64.b64decode(encrypted_data.encode())
            decrypted = self._cipher.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_data
    
    def start_file_watching(self):
        """Start watching configuration files for changes"""
        if self._watching:
            return
        
        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, config_manager):
                self.config_manager = config_manager
            
            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith('.yaml'):
                    logger.info(f"Configuration file changed: {event.src_path}")
                    self.config_manager.load_config()
        
        try:
            self._file_observer = Observer()
            handler = ConfigFileHandler(self)
            self._file_observer.schedule(handler, str(self.config_dir), recursive=False)
            self._file_observer.start()
            self._watching = True
            
            logger.info("Configuration file watching started")
            
        except Exception as e:
            logger.error(f"Failed to start file watching: {e}")
    
    def stop_file_watching(self):
        """Stop watching configuration files"""
        if self._file_observer and self._watching:
            self._file_observer.stop()
            self._file_observer.join()
            self._watching = False
            logger.info("Configuration file watching stopped")
    
    def register_observer(self, callback: callable):
        """Register callback for configuration changes"""
        self._observers.append(callback)
    
    def unregister_observer(self, callback: callable):
        """Unregister configuration change callback"""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def _notify_observers(self):
        """Notify all observers of configuration changes"""
        for observer in self._observers:
            try:
                observer(self._config.copy())
            except Exception as e:
                logger.error(f"Observer notification failed: {e}")
    
    def get_exchange_config(self, exchange_name: str) -> Optional[ExchangeConfig]:
        """Get typed exchange configuration"""
        exchange_data = self.get(f'exchanges.{exchange_name}')
        if exchange_data:
            try:
                return ExchangeConfig(**exchange_data)
            except Exception as e:
                logger.error(f"Failed to parse exchange config for {exchange_name}: {e}")
        return None
    
    def get_risk_config(self) -> RiskConfig:
        """Get typed risk configuration"""
        risk_data = self.get('risk', {})
        try:
            return RiskConfig(**risk_data)
        except Exception as e:
            logger.error(f"Failed to parse risk config: {e}")
            return RiskConfig()
    
    def get_strategy_config(self, strategy_name: str) -> Optional[StrategyConfig]:
        """Get typed strategy configuration"""
        strategy_data = self.get(f'strategies.{strategy_name}')
        if strategy_data:
            try:
                return StrategyConfig(**strategy_data)
            except Exception as e:
                logger.error(f"Failed to parse strategy config for {strategy_name}: {e}")
        return None
    
    def get_ai_config(self) -> AIConfig:
        """Get typed AI configuration"""
        ai_data = self.get('ai', {})
        try:
            return AIConfig(**ai_data)
        except Exception as e:
            logger.error(f"Failed to parse AI config: {e}")
            return AIConfig()
    
    def get_system_config(self) -> SystemConfig:
        """Get typed system configuration"""
        system_data = self.get('system', {})
        try:
            return SystemConfig(**system_data)
        except Exception as e:
            logger.error(f"Failed to parse system config: {e}")
            return SystemConfig()
    
    def export_config(self, file_path: str, format: str = 'yaml') -> bool:
        """Export configuration to file"""
        try:
            with self._config_lock:
                if format.lower() == 'json':
                    with open(file_path, 'w') as f:
                        json.dump(self._config, f, indent=2, default=str)
                else:  # yaml
                    with open(file_path, 'w') as f:
                        yaml.dump(self._config, f, default_flow_style=False, indent=2)
                
                logger.info(f"Configuration exported to {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get configuration manager statistics"""
        with self._config_lock:
            return {
                'config_file': str(self.config_file),
                'environment': self.environment,
                'file_watching': self._watching,
                'observers_count': len(self._observers),
                'config_keys': list(self._config.keys()),
                'encryption_enabled': self._cipher is not None,
                'last_modified': datetime.fromtimestamp(
                    self.config_file.stat().st_mtime
                ).isoformat() if self.config_file.exists() else None
            }

# Global configuration manager instance
config_manager = ConfigurationManager(
    environment=os.getenv('TRADING_ENVIRONMENT', 'development')
)
