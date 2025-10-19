"""
Configuration Manager for Poise Trader

Handles secure configuration management with:
- Environment-specific configs
- Encrypted sensitive data
- Dynamic configuration updates
- Validation and schema support
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from enum import Enum
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import hashlib


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class ExchangeConfig:
    """Exchange-specific configuration"""
    name: str
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""
    sandbox: bool = True
    base_url: str = ""
    websocket_url: str = ""
    rate_limit: int = 100
    timeout: int = 30


@dataclass
class StrategyConfig:
    """Strategy configuration"""
    name: str
    enabled: bool = True
    symbols: List[str] = None
    parameters: Dict[str, Any] = None
    risk_limits: Dict[str, float] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = []
        if self.parameters is None:
            self.parameters = {}
        if self.risk_limits is None:
            self.risk_limits = {}


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size: float = 0.1
    max_daily_loss: float = 0.05
    max_drawdown: float = 0.15
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_open_positions: int = 5
    position_sizing: str = "fixed"  # fixed, percentage, kelly
    emergency_stop: bool = True


@dataclass
class SystemConfig:
    """System-wide configuration"""
    log_level: str = "INFO"
    log_file: str = "logs/poise_trader.log"
    data_directory: str = "data"
    backup_directory: str = "backups"
    max_workers: int = 4
    heartbeat_interval: int = 60
    performance_tracking: bool = True
    stealth_mode: bool = False


class ConfigManager:
    """
    Secure configuration manager with encryption and validation
    """
    
    def __init__(self, config_dir: str = "config", environment: Environment = None):
        self.config_dir = Path(config_dir)
        self.environment = environment or self._detect_environment()
        self.logger = logging.getLogger("ConfigManager")
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize encryption
        self.cipher_suite = None
        self._init_encryption()
        
        # Configuration cache
        self._config_cache: Dict[str, Any] = {}
        self._encrypted_keys: set = {
            'api_key', 'api_secret', 'passphrase', 'private_key',
            'webhook_secret', 'telegram_token', 'discord_token'
        }
        
        # Load configurations
        self._load_configurations()
    
    def _detect_environment(self) -> Environment:
        """Detect current environment from environment variables"""
        env = os.getenv('POISE_ENV', 'development').lower()
        try:
            return Environment(env)
        except ValueError:
            self.logger.warning("Unknown environment '%s', defaulting to development", env)
            return Environment.DEVELOPMENT
    
    def _init_encryption(self):
        """Initialize encryption for sensitive data"""
        try:
            # Try to load existing key
            key_file = self.config_dir / ".encryption_key"
            
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    key = f.read()
            else:
                # Generate new key
                password = os.getenv('POISE_ENCRYPTION_PASSWORD', 'default_password_change_me')
                salt = os.urandom(16)
                
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000
                )
                key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
                
                # Save key
                with open(key_file, 'wb') as f:
                    f.write(key)
                
                # Also save salt for future use
                with open(self.config_dir / ".salt", 'wb') as f:
                    f.write(salt)
            
            self.cipher_suite = Fernet(key)
            
        except Exception as e:
            self.logger.error("Failed to initialize encryption: %s", e)
            self.cipher_suite = None
    
    def _load_configurations(self):
        """Load all configuration files"""
        # Base config file
        base_config_file = self.config_dir / "config.yaml"
        if base_config_file.exists():
            self._load_config_file(base_config_file, "base")
        
        # Environment-specific config
        env_config_file = self.config_dir / f"config_{self.environment.value}.yaml"
        if env_config_file.exists():
            self._load_config_file(env_config_file, "environment")
        
        # Plugin configs
        plugins_dir = self.config_dir / "plugins"
        if plugins_dir.exists():
            for config_file in plugins_dir.glob("*.yaml"):
                plugin_name = config_file.stem
                self._load_config_file(config_file, f"plugin_{plugin_name}")
    
    def _load_config_file(self, file_path: Path, config_type: str):
        """Load a single configuration file"""
        try:
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if config_data:
                # Decrypt sensitive values
                self._decrypt_config(config_data)
                self._config_cache[config_type] = config_data
                
            self.logger.debug("Loaded config: %s", config_type)
            
        except Exception as e:
            self.logger.error("Failed to load config file %s: %s", file_path, e)
    
    def _decrypt_config(self, config: Dict[str, Any]):
        """Recursively decrypt sensitive values in config"""
        if not self.cipher_suite:
            return
        
        for key, value in config.items():
            if isinstance(value, dict):
                self._decrypt_config(value)
            elif key in self._encrypted_keys and isinstance(value, str):
                try:
                    # Check if value is encrypted (starts with our marker)
                    if value.startswith("ENC:"):
                        encrypted_value = value[4:]
                        decrypted = self.cipher_suite.decrypt(
                            base64.urlsafe_b64decode(encrypted_value)
                        )
                        config[key] = decrypted.decode()
                except Exception as e:
                    self.logger.error("Failed to decrypt %s: %s", key, e)
    
    def _encrypt_value(self, value: str) -> str:
        """Encrypt a sensitive value"""
        if not self.cipher_suite:
            return value
        
        try:
            encrypted = self.cipher_suite.encrypt(value.encode())
            return "ENC:" + base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            self.logger.error("Failed to encrypt value: %s", e)
            return value
    
    def get_config(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            path: Configuration path (e.g., 'exchanges.binance.api_key')
            default: Default value if path not found
            
        Returns:
            Configuration value
        """
        parts = path.split('.')
        
        # Try environment config first, then base config
        for config_type in ["environment", "base"]:
            if config_type in self._config_cache:
                value = self._config_cache[config_type]
                
                try:
                    for part in parts:
                        value = value[part]
                    return value
                except (KeyError, TypeError):
                    continue
        
        return default
    
    def set_config(self, path: str, value: Any, save: bool = True):
        """
        Set configuration value
        
        Args:
            path: Configuration path
            value: Value to set
            save: Whether to save to file immediately
        """
        parts = path.split('.')
        config_type = "environment"
        
        if config_type not in self._config_cache:
            self._config_cache[config_type] = {}
        
        config = self._config_cache[config_type]
        
        # Navigate to the parent dictionary
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        
        # Set the value
        final_key = parts[-1]
        
        # Encrypt if sensitive
        if final_key in self._encrypted_keys and isinstance(value, str):
            value = self._encrypt_value(value)
        
        config[final_key] = value
        
        if save:
            self._save_config(config_type)
    
    def _save_config(self, config_type: str):
        """Save configuration to file"""
        if config_type not in self._config_cache:
            return
        
        try:
            if config_type == "environment":
                file_path = self.config_dir / f"config_{self.environment.value}.yaml"
            elif config_type == "base":
                file_path = self.config_dir / "config.yaml"
            elif config_type.startswith("plugin_"):
                plugin_name = config_type[7:]  # Remove 'plugin_' prefix
                plugins_dir = self.config_dir / "plugins"
                plugins_dir.mkdir(exist_ok=True)
                file_path = plugins_dir / f"{plugin_name}.yaml"
            else:
                return
            
            with open(file_path, 'w') as f:
                yaml.dump(self._config_cache[config_type], f, default_flow_style=False)
            
            self.logger.debug("Saved config: %s", config_type)
            
        except Exception as e:
            self.logger.error("Failed to save config %s: %s", config_type, e)
    
    def get_exchange_config(self, exchange_name: str) -> Optional[ExchangeConfig]:
        """Get configuration for a specific exchange"""
        exchange_data = self.get_config(f"exchanges.{exchange_name}")
        
        if not exchange_data:
            return None
        
        return ExchangeConfig(
            name=exchange_name,
            **exchange_data
        )
    
    def get_strategy_config(self, strategy_name: str) -> Optional[StrategyConfig]:
        """Get configuration for a specific strategy"""
        strategy_data = self.get_config(f"strategies.{strategy_name}")
        
        if not strategy_data:
            return None
        
        return StrategyConfig(
            name=strategy_name,
            **strategy_data
        )
    
    def get_risk_config(self) -> RiskConfig:
        """Get risk management configuration"""
        risk_data = self.get_config("risk", {})
        return RiskConfig(**risk_data)
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration"""
        system_data = self.get_config("system", {})
        return SystemConfig(**system_data)
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get configuration for a specific plugin"""
        plugin_config_type = f"plugin_{plugin_name}"
        if plugin_config_type in self._config_cache:
            return self._config_cache[plugin_config_type]
        
        # Try to get from main config
        return self.get_config(f"plugins.{plugin_name}", {})
    
    def create_default_configs(self):
        """Create default configuration files"""
        # Main config
        main_config = {
            'system': {
                'log_level': 'INFO',
                'max_workers': 4,
                'stealth_mode': False
            },
            'risk': {
                'max_position_size': 0.1,
                'max_daily_loss': 0.05,
                'stop_loss_pct': 0.02
            },
            'exchanges': {},
            'strategies': {},
            'plugins': {}
        }
        
        config_file = self.config_dir / "config.yaml"
        if not config_file.exists():
            with open(config_file, 'w') as f:
                yaml.dump(main_config, f, default_flow_style=False)
        
        # Environment-specific config
        env_config_file = self.config_dir / f"config_{self.environment.value}.yaml"
        if not env_config_file.exists():
            env_config = {
                'system': {
                    'log_level': 'DEBUG' if self.environment == Environment.DEVELOPMENT else 'INFO'
                }
            }
            
            with open(env_config_file, 'w') as f:
                yaml.dump(env_config, f, default_flow_style=False)
    
    def validate_config(self) -> List[str]:
        """
        Validate current configuration
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check required system settings
        system_config = self.get_system_config()
        if not system_config.log_level:
            errors.append("Missing system.log_level")
        
        # Check exchange configurations
        exchanges = self.get_config("exchanges", {})
        for exchange_name, exchange_config in exchanges.items():
            if not exchange_config.get('api_key'):
                errors.append(f"Missing API key for exchange: {exchange_name}")
        
        return errors
    
    def reload(self):
        """Reload all configurations from disk"""
        self._config_cache.clear()
        self._load_configurations()
        self.logger.info("Configuration reloaded")
    
    def get_environment(self) -> Environment:
        """Get current environment"""
        return self.environment
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == Environment.PRODUCTION
    
    def backup_config(self) -> str:
        """
        Create a backup of current configuration
        
        Returns:
            Path to backup file
        """
        import datetime
        
        backup_dir = Path("backups")
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"config_backup_{timestamp}.yaml"
        
        # Combine all configs
        full_config = {}
        for config_type, config_data in self._config_cache.items():
            full_config[config_type] = config_data
        
        with open(backup_file, 'w') as f:
            yaml.dump(full_config, f, default_flow_style=False)
        
        self.logger.info("Configuration backed up to: %s", backup_file)
        return str(backup_file)
