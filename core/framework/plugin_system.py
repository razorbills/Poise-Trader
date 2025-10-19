"""
Plugin System for Poise Trader

Provides a flexible plugin architecture for loading and managing
trading system components dynamically.
"""

import os
import sys
import importlib
import inspect
import logging
from typing import Dict, Any, List, Type, Optional, Set
from pathlib import Path
import json
import yaml
from abc import ABC, abstractmethod


class PluginMetadata:
    """Metadata for a plugin"""
    
    def __init__(self, name: str, version: str, description: str = "",
                 dependencies: List[str] = None, category: str = "general"):
        self.name = name
        self.version = version
        self.description = description
        self.dependencies = dependencies or []
        self.category = category
        self.is_loaded = False
        self.instance = None


class BasePlugin(ABC):
    """Base class for all plugins"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"Plugin.{self.__class__.__name__}")
        self.is_initialized = False
        self.metadata = self._get_metadata()
    
    @abstractmethod
    def _get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the plugin cleanly"""
        pass
    
    def get_name(self) -> str:
        """Get plugin name"""
        return self.metadata.name
    
    def get_version(self) -> str:
        """Get plugin version"""
        return self.metadata.version


class PluginManager:
    """
    Manages loading, initialization, and lifecycle of plugins
    
    Features:
    - Dynamic plugin loading
    - Dependency resolution
    - Configuration management  
    - Plugin isolation
    - Hot-reloading capability
    """
    
    def __init__(self, plugin_dirs: List[str] = None, config_manager=None):
        self.logger = logging.getLogger("PluginManager")
        self.plugin_dirs = plugin_dirs or []
        self.config_manager = config_manager
        
        # Plugin registry
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self.plugin_classes: Dict[str, Type[BasePlugin]] = {}
        
        # Category mappings
        self.categories: Dict[str, Set[str]] = {
            'feeds': set(),
            'strategies': set(),
            'executors': set(),
            'risk_managers': set(),
            'monitors': set(),
            'backtesting': set()
        }
        
        # Dependency graph
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.loaded_plugins: Set[str] = set()
        
    def add_plugin_directory(self, directory: str) -> None:
        """Add a directory to scan for plugins"""
        if directory not in self.plugin_dirs:
            self.plugin_dirs.append(directory)
            self.logger.info("Added plugin directory: %s", directory)
    
    async def discover_plugins(self) -> Dict[str, PluginMetadata]:
        """
        Discover all plugins in registered directories
        
        Returns:
            Dictionary of discovered plugin metadata
        """
        discovered = {}
        
        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                self.logger.warning("Plugin directory not found: %s", plugin_dir)
                continue
            
            self.logger.info("Discovering plugins in: %s", plugin_dir)
            
            # Add directory to Python path
            if plugin_dir not in sys.path:
                sys.path.append(plugin_dir)
            
            # Scan for Python modules
            for item in os.listdir(plugin_dir):
                item_path = os.path.join(plugin_dir, item)
                
                if os.path.isfile(item_path) and item.endswith('.py'):
                    # Single module file
                    module_name = item[:-3]
                    await self._discover_module(plugin_dir, module_name, discovered)
                    
                elif os.path.isdir(item_path) and os.path.exists(
                    os.path.join(item_path, '__init__.py')
                ):
                    # Package directory
                    await self._discover_module(plugin_dir, item, discovered)
        
        self.plugin_metadata.update(discovered)
        self.logger.info("Discovered %d plugins", len(discovered))
        return discovered
    
    async def _discover_module(self, plugin_dir: str, module_name: str, 
                              discovered: Dict[str, PluginMetadata]) -> None:
        """Discover plugins in a specific module"""
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(
                module_name, 
                os.path.join(plugin_dir, f"{module_name}.py")
                if os.path.isfile(os.path.join(plugin_dir, f"{module_name}.py"))
                else os.path.join(plugin_dir, module_name, "__init__.py")
            )
            
            if spec is None or spec.loader is None:
                return
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BasePlugin) and 
                    obj is not BasePlugin and 
                    not inspect.isabstract(obj)):
                    
                    try:
                        # Create temporary instance to get metadata
                        temp_plugin = obj({})
                        metadata = temp_plugin.metadata
                        
                        self.plugin_classes[metadata.name] = obj
                        discovered[metadata.name] = metadata
                        
                        # Categorize plugin
                        self._categorize_plugin(metadata.name, metadata.category)
                        
                        self.logger.debug("Discovered plugin: %s v%s", 
                                        metadata.name, metadata.version)
                        
                    except Exception as e:
                        self.logger.error("Failed to load metadata for %s: %s", 
                                        name, e)
                        
        except Exception as e:
            self.logger.error("Failed to discover plugins in %s: %s", 
                            module_name, e)
    
    def _categorize_plugin(self, plugin_name: str, category: str) -> None:
        """Categorize a plugin"""
        if category in self.categories:
            self.categories[category].add(plugin_name)
        else:
            # Create new category
            self.categories[category] = {plugin_name}
    
    async def load_plugin(self, plugin_name: str, config: Dict[str, Any] = None) -> bool:
        """
        Load and initialize a specific plugin
        
        Args:
            plugin_name: Name of the plugin to load
            config: Configuration for the plugin
            
        Returns:
            True if plugin was loaded successfully
        """
        if plugin_name in self.loaded_plugins:
            self.logger.warning("Plugin %s is already loaded", plugin_name)
            return True
        
        if plugin_name not in self.plugin_classes:
            self.logger.error("Plugin %s not found", plugin_name)
            return False
        
        try:
            # Get plugin configuration
            plugin_config = config or {}
            if self.config_manager:
                plugin_config = self.config_manager.get_plugin_config(plugin_name)
            
            # Create plugin instance
            plugin_class = self.plugin_classes[plugin_name]
            plugin_instance = plugin_class(plugin_config)
            
            # Check dependencies
            metadata = self.plugin_metadata[plugin_name]
            for dep in metadata.dependencies:
                if dep not in self.loaded_plugins:
                    self.logger.info("Loading dependency: %s", dep)
                    if not await self.load_plugin(dep):
                        self.logger.error("Failed to load dependency %s for %s", 
                                        dep, plugin_name)
                        return False
            
            # Initialize plugin
            if await plugin_instance.initialize():
                self.plugins[plugin_name] = plugin_instance
                self.loaded_plugins.add(plugin_name)
                metadata.is_loaded = True
                metadata.instance = plugin_instance
                
                self.logger.info("Plugin loaded successfully: %s v%s", 
                               plugin_name, metadata.version)
                return True
            else:
                self.logger.error("Failed to initialize plugin: %s", plugin_name)
                return False
                
        except Exception as e:
            self.logger.error("Error loading plugin %s: %s", plugin_name, e, exc_info=True)
            return False
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a specific plugin
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if plugin was unloaded successfully
        """
        if plugin_name not in self.loaded_plugins:
            self.logger.warning("Plugin %s is not loaded", plugin_name)
            return True
        
        try:
            # Check if other plugins depend on this one
            dependents = self._get_dependents(plugin_name)
            if dependents:
                self.logger.error("Cannot unload %s: required by %s", 
                                plugin_name, ', '.join(dependents))
                return False
            
            # Shutdown plugin
            plugin = self.plugins[plugin_name]
            await plugin.shutdown()
            
            # Remove from registry
            del self.plugins[plugin_name]
            self.loaded_plugins.remove(plugin_name)
            
            metadata = self.plugin_metadata[plugin_name]
            metadata.is_loaded = False
            metadata.instance = None
            
            self.logger.info("Plugin unloaded: %s", plugin_name)
            return True
            
        except Exception as e:
            self.logger.error("Error unloading plugin %s: %s", plugin_name, e)
            return False
    
    def _get_dependents(self, plugin_name: str) -> List[str]:
        """Get list of plugins that depend on the given plugin"""
        dependents = []
        for name, metadata in self.plugin_metadata.items():
            if plugin_name in metadata.dependencies and name in self.loaded_plugins:
                dependents.append(name)
        return dependents
    
    async def load_plugins_by_category(self, category: str, 
                                     configs: Dict[str, Dict[str, Any]] = None) -> int:
        """
        Load all plugins in a specific category
        
        Args:
            category: Plugin category to load
            configs: Per-plugin configurations
            
        Returns:
            Number of plugins successfully loaded
        """
        if category not in self.categories:
            self.logger.warning("Unknown category: %s", category)
            return 0
        
        loaded_count = 0
        plugins = self.categories[category]
        configs = configs or {}
        
        for plugin_name in plugins:
            plugin_config = configs.get(plugin_name, {})
            if await self.load_plugin(plugin_name, plugin_config):
                loaded_count += 1
        
        self.logger.info("Loaded %d/%d plugins in category '%s'", 
                       loaded_count, len(plugins), category)
        return loaded_count
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a loaded plugin instance"""
        return self.plugins.get(plugin_name)
    
    def get_plugins_by_category(self, category: str) -> List[BasePlugin]:
        """Get all loaded plugins in a category"""
        if category not in self.categories:
            return []
        
        result = []
        for plugin_name in self.categories[category]:
            if plugin_name in self.plugins:
                result.append(self.plugins[plugin_name])
        
        return result
    
    def get_plugin_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all discovered plugins"""
        status = {}
        for name, metadata in self.plugin_metadata.items():
            status[name] = {
                'version': metadata.version,
                'description': metadata.description,
                'category': metadata.category,
                'loaded': metadata.is_loaded,
                'dependencies': metadata.dependencies
            }
        return status
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """
        Hot-reload a plugin (unload and load again)
        
        Args:
            plugin_name: Name of the plugin to reload
            
        Returns:
            True if plugin was reloaded successfully
        """
        if plugin_name not in self.plugins:
            self.logger.warning("Plugin %s is not loaded", plugin_name)
            return False
        
        # Get current config
        current_config = self.plugins[plugin_name].config
        
        # Unload plugin
        if not await self.unload_plugin(plugin_name):
            return False
        
        # Rediscover plugins to get updated code
        await self.discover_plugins()
        
        # Load plugin again
        return await self.load_plugin(plugin_name, current_config)
    
    async def shutdown_all(self) -> None:
        """Shutdown all loaded plugins"""
        self.logger.info("Shutting down all plugins")
        
        # Shutdown in reverse dependency order
        shutdown_order = self._get_shutdown_order()
        
        for plugin_name in shutdown_order:
            if plugin_name in self.plugins:
                try:
                    await self.plugins[plugin_name].shutdown()
                    self.logger.info("Shutdown plugin: %s", plugin_name)
                except Exception as e:
                    self.logger.error("Error shutting down %s: %s", plugin_name, e)
        
        self.plugins.clear()
        self.loaded_plugins.clear()
    
    def _get_shutdown_order(self) -> List[str]:
        """Get the order to shutdown plugins (reverse dependency order)"""
        # Simple approach: reverse of load order
        # TODO: Implement proper topological sort for complex dependencies
        return list(reversed(list(self.loaded_plugins)))
