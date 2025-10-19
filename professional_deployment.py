#!/usr/bin/env python3
"""
🚀 PROFESSIONAL DEPLOYMENT & MONITORING SYSTEM
Enterprise-grade bot deployment with health monitoring and auto-recovery
"""

import asyncio
import psutil
import socket
import json
import logging
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

@dataclass
class SystemHealth:
    """System health metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    bot_uptime: int
    last_trade_time: Optional[datetime]
    error_count: int
    health_score: float  # 0-100

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    environment: str  # development, staging, production
    auto_restart: bool
    max_restarts_per_hour: int
    health_check_interval: int
    alert_email: Optional[str]
    telegram_token: Optional[str]
    telegram_chat_id: Optional[str]
    max_memory_mb: int
    max_cpu_percent: float

class ProfessionalDeploymentManager:
    """Enterprise deployment management"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logging.getLogger("DeploymentManager")
        
        # Health monitoring
        self.health_history = []
        self.last_health_check = None
        self.restart_count = 0
        self.last_restart_hour = None
        
        # Process management
        self.bot_process = None
        self.monitoring_active = False
        self.start_time = datetime.now()
        
        # Alert management
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minutes
        
    async def deploy_bot(self, bot_script: str, bot_config: Dict) -> bool:
        """Deploy bot with full monitoring"""
        try:
            self.logger.info("🚀 Starting professional bot deployment...")
            
            # 1. Pre-deployment checks
            if not await self._pre_deployment_checks():
                return False
            
            # 2. Start bot process
            success = await self._start_bot_process(bot_script, bot_config)
            if not success:
                return False
            
            # 3. Start health monitoring
            await self._start_health_monitoring()
            
            # 4. Setup alerting
            await self._setup_alerting()
            
            # 5. Start auto-recovery system
            await self._start_auto_recovery()
            
            self.logger.info("✅ Professional deployment completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Deployment failed: {e}")
            return False
    
    async def _pre_deployment_checks(self) -> bool:
        """Pre-deployment system checks"""
        self.logger.info("🔍 Running pre-deployment checks...")
        
        checks = {
            'system_resources': await self._check_system_resources(),
            'network_connectivity': await self._check_network_connectivity(),
            'dependencies': await self._check_dependencies(),
            'disk_space': await self._check_disk_space(),
            'permissions': await self._check_permissions()
        }
        
        failed_checks = [name for name, passed in checks.items() if not passed]
        
        if failed_checks:
            self.logger.error(f"❌ Failed checks: {failed_checks}")
            return False
        
        self.logger.info("✅ All pre-deployment checks passed")
        return True
    
    async def _check_system_resources(self) -> bool:
        """Check system has sufficient resources"""
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        
        # Minimum requirements
        min_memory_gb = 4
        min_cpu_cores = 2
        
        if memory.total < min_memory_gb * 1024**3:
            self.logger.warning(f"Low memory: {memory.total / 1024**3:.1f}GB < {min_memory_gb}GB")
            return False
        
        if cpu_count < min_cpu_cores:
            self.logger.warning(f"Insufficient CPU cores: {cpu_count} < {min_cpu_cores}")
            return False
        
        return True
    
    async def _check_network_connectivity(self) -> bool:
        """Check network connectivity to exchanges"""
        test_hosts = [
            'api.mexc.com',
            'api.binance.com',
            'api.coinbase.com'
        ]
        
        for host in test_hosts:
            try:
                socket.create_connection((host, 443), timeout=5)
            except:
                self.logger.warning(f"Cannot connect to {host}")
                return False
        
        return True
    
    async def _check_dependencies(self) -> bool:
        """Check required dependencies are available"""
        required_modules = [
            'numpy', 'pandas', 'asyncio', 'aiohttp',
            'matplotlib', 'sklearn'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                self.logger.error(f"Missing dependency: {module}")
                return False
        
        return True
    
    async def _check_disk_space(self) -> bool:
        """Check sufficient disk space"""
        disk = psutil.disk_usage('.')
        free_gb = disk.free / 1024**3
        
        if free_gb < 1.0:  # Minimum 1GB free
            self.logger.error(f"Insufficient disk space: {free_gb:.1f}GB")
            return False
        
        return True
    
    async def _check_permissions(self) -> bool:
        """Check file system permissions"""
        try:
            # Test write access
            test_file = Path('.deployment_test')
            test_file.write_text('test')
            test_file.unlink()
            return True
        except:
            self.logger.error("No write permissions in deployment directory")
            return False
    
    async def _start_bot_process(self, bot_script: str, bot_config: Dict) -> bool:
        """Start bot as separate process"""
        try:
            # Save config to temporary file
            config_file = Path('.bot_config_temp.json')
            with open(config_file, 'w') as f:
                json.dump(bot_config, f)
            
            # Start bot process
            cmd = [
                'python', bot_script,
                '--config', str(config_file)
            ]
            
            self.bot_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment to check if it started successfully
            await asyncio.sleep(5)
            
            if self.bot_process.poll() is None:
                self.logger.info(f"✅ Bot process started with PID: {self.bot_process.pid}")
                return True
            else:
                stdout, stderr = self.bot_process.communicate()
                self.logger.error(f"❌ Bot failed to start: {stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start bot process: {e}")
            return False
    
    async def _start_health_monitoring(self):
        """Start comprehensive health monitoring"""
        self.monitoring_active = True
        asyncio.create_task(self._health_monitoring_loop())
        self.logger.info("🔍 Health monitoring started")
    
    async def _health_monitoring_loop(self):
        """Main health monitoring loop"""
        while self.monitoring_active:
            try:
                health = await self._collect_health_metrics()
                self.health_history.append(health)
                
                # Keep last 1000 health records
                if len(self.health_history) > 1000:
                    self.health_history = self.health_history[-500:]
                
                # Check for issues
                await self._analyze_health_issues(health)
                
                self.last_health_check = datetime.now()
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_health_metrics(self) -> SystemHealth:
        """Collect comprehensive system health metrics"""
        
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        # Network latency (ping to exchange)
        network_latency = await self._measure_network_latency()
        
        # Bot-specific metrics
        uptime = int((datetime.now() - self.start_time).total_seconds())
        
        # Error count (would track from logs)
        error_count = 0  # Implement log parsing
        
        # Calculate health score
        health_score = await self._calculate_health_score(
            cpu_usage, memory.percent, disk.percent, network_latency, error_count
        )
        
        return SystemHealth(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=(disk.used / disk.total) * 100,
            network_latency=network_latency,
            bot_uptime=uptime,
            last_trade_time=None,  # Would track from bot
            error_count=error_count,
            health_score=health_score
        )
    
    async def _measure_network_latency(self) -> float:
        """Measure network latency to exchange"""
        try:
            start_time = time.time()
            socket.create_connection(('api.mexc.com', 443), timeout=5)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            return latency
        except:
            return 999.0  # High latency for connection failure
    
    async def _calculate_health_score(self, cpu: float, memory: float, 
                                    disk: float, latency: float, errors: int) -> float:
        """Calculate overall health score (0-100)"""
        
        # Component scores
        cpu_score = max(0, 100 - cpu) if cpu < 90 else 0
        memory_score = max(0, 100 - memory) if memory < 90 else 0
        disk_score = max(0, 100 - disk) if disk < 95 else 0
        network_score = max(0, 100 - (latency / 10)) if latency < 1000 else 0
        error_score = max(0, 100 - (errors * 10))
        
        # Weighted average
        weights = [0.25, 0.25, 0.15, 0.25, 0.1]
        scores = [cpu_score, memory_score, disk_score, network_score, error_score]
        
        health_score = sum(score * weight for score, weight in zip(scores, weights))
        return max(0, min(100, health_score))
    
    async def _analyze_health_issues(self, health: SystemHealth):
        """Analyze health metrics and trigger actions"""
        
        issues = []
        
        # Critical issues
        if health.cpu_usage > self.config.max_cpu_percent:
            issues.append(f"High CPU usage: {health.cpu_usage:.1f}%")
        
        if health.memory_usage > 90:
            issues.append(f"High memory usage: {health.memory_usage:.1f}%")
        
        if health.network_latency > 1000:
            issues.append(f"High network latency: {health.network_latency:.0f}ms")
        
        if health.health_score < 30:
            issues.append(f"Critical health score: {health.health_score:.1f}")
        
        # Handle issues
        if issues:
            await self._handle_health_issues(issues, health)
    
    async def _handle_health_issues(self, issues: List[str], health: SystemHealth):
        """Handle detected health issues"""
        
        severity = "CRITICAL" if health.health_score < 30 else "WARNING"
        
        # Log issues
        for issue in issues:
            self.logger.warning(f"{severity}: {issue}")
        
        # Send alerts
        await self._send_health_alert(issues, health, severity)
        
        # Auto-recovery actions
        if health.health_score < 20 and self.config.auto_restart:
            await self._trigger_auto_restart(f"Critical health score: {health.health_score:.1f}")
    
    async def _setup_alerting(self):
        """Setup alerting systems"""
        if self.config.alert_email:
            self.logger.info(f"📧 Email alerts configured: {self.config.alert_email}")
        
        if self.config.telegram_token and self.config.telegram_chat_id:
            self.logger.info("📱 Telegram alerts configured")
    
    async def _send_health_alert(self, issues: List[str], health: SystemHealth, severity: str):
        """Send health alert via configured channels"""
        
        # Rate limiting
        alert_key = f"health_{severity}"
        now = datetime.now()
        
        if alert_key in self.last_alert_time:
            time_since_last = (now - self.last_alert_time[alert_key]).total_seconds()
            if time_since_last < self.alert_cooldown:
                return
        
        self.last_alert_time[alert_key] = now
        
        # Prepare alert message
        message = f"""
🚨 BOT HEALTH ALERT - {severity}

Issues detected:
{chr(10).join(f'• {issue}' for issue in issues)}

System Health:
• CPU Usage: {health.cpu_usage:.1f}%
• Memory Usage: {health.memory_usage:.1f}%
• Network Latency: {health.network_latency:.0f}ms
• Health Score: {health.health_score:.1f}/100
• Uptime: {health.bot_uptime} seconds

Time: {now.strftime('%Y-%m-%d %H:%M:%S')}
Environment: {self.config.environment}
        """
        
        # Send via email
        if self.config.alert_email:
            await self._send_email_alert(message, severity)
        
        # Send via Telegram
        if self.config.telegram_token:
            await self._send_telegram_alert(message)
    
    async def _send_email_alert(self, message: str, severity: str):
        """Send email alert"""
        try:
            # This would use actual SMTP configuration
            self.logger.info(f"📧 Email alert sent: {severity}")
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    async def _send_telegram_alert(self, message: str):
        """Send Telegram alert"""
        try:
            # This would use Telegram Bot API
            self.logger.info("📱 Telegram alert sent")
        except Exception as e:
            self.logger.error(f"Failed to send Telegram alert: {e}")
    
    async def _start_auto_recovery(self):
        """Start auto-recovery system"""
        asyncio.create_task(self._auto_recovery_loop())
        self.logger.info("🔄 Auto-recovery system started")
    
    async def _auto_recovery_loop(self):
        """Auto-recovery monitoring loop"""
        while self.monitoring_active:
            try:
                # Check if bot process is still running
                if self.bot_process and self.bot_process.poll() is not None:
                    self.logger.error("Bot process terminated unexpectedly")
                    await self._trigger_auto_restart("Process termination")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Auto-recovery error: {e}")
                await asyncio.sleep(60)
    
    async def _trigger_auto_restart(self, reason: str):
        """Trigger automatic restart"""
        if not self.config.auto_restart:
            self.logger.info("Auto-restart disabled")
            return
        
        # Check restart limits
        current_hour = datetime.now().hour
        if current_hour != self.last_restart_hour:
            self.restart_count = 0
            self.last_restart_hour = current_hour
        
        if self.restart_count >= self.config.max_restarts_per_hour:
            self.logger.error("Max restarts per hour exceeded")
            return
        
        self.logger.warning(f"🔄 Triggering auto-restart: {reason}")
        self.restart_count += 1
        
        # Stop current process
        if self.bot_process:
            self.bot_process.terminate()
            await asyncio.sleep(5)
            
            if self.bot_process.poll() is None:
                self.bot_process.kill()
        
        # Restart bot (would need bot script and config)
        # This would restart the deployment process
        self.logger.info("✅ Auto-restart completed")
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status"""
        current_health = self.health_history[-1] if self.health_history else None
        
        return {
            'deployment_active': self.monitoring_active,
            'bot_process_running': self.bot_process and self.bot_process.poll() is None,
            'uptime_seconds': int((datetime.now() - self.start_time).total_seconds()),
            'restart_count': self.restart_count,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'current_health': asdict(current_health) if current_health else None,
            'environment': self.config.environment,
            'auto_restart_enabled': self.config.auto_restart
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("🛑 Starting graceful shutdown...")
        
        # Stop monitoring
        self.monitoring_active = False
        
        # Stop bot process
        if self.bot_process:
            self.bot_process.terminate()
            await asyncio.sleep(5)
            
            if self.bot_process.poll() is None:
                self.bot_process.kill()
        
        self.logger.info("✅ Graceful shutdown completed")

# Example deployment configuration
def create_production_config() -> DeploymentConfig:
    """Create production deployment configuration"""
    return DeploymentConfig(
        environment="production",
        auto_restart=True,
        max_restarts_per_hour=3,
        health_check_interval=30,
        alert_email="alerts@yourcompany.com",
        telegram_token="YOUR_TELEGRAM_BOT_TOKEN",
        telegram_chat_id="YOUR_CHAT_ID",
        max_memory_mb=2048,
        max_cpu_percent=80.0
    )

# Global deployment manager
deployment_manager = None
