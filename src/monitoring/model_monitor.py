"""
Comprehensive monitoring and observability system for production ML models.
Includes performance monitoring, data drift detection, model quality tracking, and alerting.
"""

import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import numpy as np
from collections import deque, defaultdict
from contextlib import contextmanager

import torch
import psutil
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, push_to_gateway, generate_latest
)
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace, metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
import mlflow
import wandb
from loguru import logger

# Handle imports gracefully
try:
    from ..data.drift_detector import DataDriftDetector
    from ..optimization.performance_analyzer import PerformanceAnalyzer
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    try:
        from data.drift_detector import DataDriftDetector
        from optimization.performance_analyzer import PerformanceAnalyzer
    except ImportError:
        # Mock classes if imports fail
        class DataDriftDetector:
            def detect_drift(self, reference_data, current_data): return {"drift_detected": False}
            
        class PerformanceAnalyzer:
            def analyze_performance(self, metrics): return {"status": "ok"}


@dataclass
class MonitoringConfig:
    """Configuration for model monitoring."""
    
    # Basic settings
    model_name: str
    model_version: str
    environment: str = "production"
    
    # Monitoring intervals (seconds)
    performance_interval: float = 30.0
    health_check_interval: float = 60.0
    drift_check_interval: float = 300.0
    
    # Thresholds
    latency_threshold_p95: float = 1000.0  # ms
    error_rate_threshold: float = 0.05  # 5%
    memory_threshold_mb: float = 8192.0
    cpu_threshold_percent: float = 80.0
    drift_threshold: float = 0.1
    
    # Storage
    metrics_retention_days: int = 30
    enable_azure_monitor: bool = True
    enable_prometheus: bool = True
    enable_mlflow: bool = True
    enable_wandb: bool = False
    
    # Alerting
    alert_webhook_url: Optional[str] = None
    alert_email: Optional[str] = None
    
    # Features
    enable_gpu_monitoring: bool = True
    enable_drift_detection: bool = True
    enable_model_performance: bool = True
    enable_business_metrics: bool = True


class MetricsCollector:
    """Collects and manages various metrics."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.registry = CollectorRegistry()
        
        # Model performance metrics
        self.request_count = Counter(
            'model_requests_total',
            'Total number of model requests',
            ['model_name', 'model_version', 'status'],
            registry=self.registry
        )
        
        self.request_latency = Histogram(
            'model_request_latency_seconds',
            'Model request latency in seconds',
            ['model_name', 'model_version'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        self.model_errors = Counter(
            'model_errors_total',
            'Total number of model errors',
            ['model_name', 'model_version', 'error_type'],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_mb',
            'Memory usage in MB',
            registry=self.registry
        )
        
        self.gpu_memory_usage = Gauge(
            'gpu_memory_usage_mb',
            'GPU memory usage in MB',
            ['gpu_id'],
            registry=self.registry
        )
        
        self.gpu_utilization = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id'],
            registry=self.registry
        )
        
        # Model quality metrics
        self.prediction_drift = Gauge(
            'model_prediction_drift',
            'Prediction drift score',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        self.data_drift = Gauge(
            'model_data_drift',
            'Input data drift score',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        # Business metrics
        self.throughput = Gauge(
            'model_throughput_rps',
            'Model throughput in requests per second',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        self.queue_size = Gauge(
            'model_queue_size',
            'Current queue size',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        # Model info
        self.model_info = Info(
            'model_info',
            'Model information',
            registry=self.registry
        )
        
        # Set model info
        self.model_info.info({
            'name': config.model_name,
            'version': config.model_version,
            'environment': config.environment,
            'monitoring_start_time': datetime.now().isoformat(),
        })


class PerformanceMonitor:
    """Monitors model performance metrics."""
    
    def __init__(self, config: MonitoringConfig, metrics_collector: MetricsCollector):
        self.config = config
        self.metrics = metrics_collector
        self.latency_buffer = deque(maxlen=1000)
        self.error_buffer = deque(maxlen=1000)
        self.request_timestamps = deque(maxlen=1000)
        
    @contextmanager
    def measure_latency(self, operation: str = "inference"):
        """Context manager to measure operation latency."""
        start_time = time.time()
        try:
            yield
        finally:
            latency = time.time() - start_time
            self.latency_buffer.append(latency)
            
            # Update metrics
            self.metrics.request_latency.labels(
                model_name=self.config.model_name,
                model_version=self.config.model_version
            ).observe(latency)
            
            self.request_timestamps.append(time.time())
            
    def record_request(self, status: str = "success"):
        """Record a model request."""
        self.metrics.request_count.labels(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            status=status
        ).inc()
        
    def record_error(self, error_type: str):
        """Record a model error."""
        self.error_buffer.append(time.time())
        
        self.metrics.model_errors.labels(
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            error_type=error_type
        ).inc()
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        stats = {}
        
        if self.latency_buffer:
            latencies = list(self.latency_buffer)
            stats.update({
                'latency_mean': np.mean(latencies),
                'latency_p50': np.percentile(latencies, 50),
                'latency_p95': np.percentile(latencies, 95),
                'latency_p99': np.percentile(latencies, 99),
            })
            
        # Calculate error rate
        now = time.time()
        recent_errors = [t for t in self.error_buffer if now - t < 300]  # Last 5 minutes
        recent_requests = [t for t in self.request_timestamps if now - t < 300]
        
        if recent_requests:
            stats['error_rate'] = len(recent_errors) / len(recent_requests)
        else:
            stats['error_rate'] = 0.0
            
        # Calculate throughput (RPS)
        if len(self.request_timestamps) >= 2:
            time_window = self.request_timestamps[-1] - self.request_timestamps[0]
            if time_window > 0:
                stats['throughput'] = len(self.request_timestamps) / time_window
            else:
                stats['throughput'] = 0.0
        else:
            stats['throughput'] = 0.0
            
        return stats


class SystemMonitor:
    """Monitors system resources."""
    
    def __init__(self, config: MonitoringConfig, metrics_collector: MetricsCollector):
        self.config = config
        self.metrics = metrics_collector
        self.running = False
        self.monitor_thread = None
        
    def start(self):
        """Start system monitoring."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """Stop system monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics.cpu_usage.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                self.metrics.memory_usage.set(memory_mb)
                
                # GPU monitoring
                if self.config.enable_gpu_monitoring and torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        # GPU memory
                        gpu_memory = torch.cuda.memory_allocated(i) / (1024 * 1024)
                        self.metrics.gpu_memory_usage.labels(gpu_id=str(i)).set(gpu_memory)
                        
                        # GPU utilization (simplified)
                        # In production, use nvidia-ml-py for accurate metrics
                        gpu_util = min(100, gpu_memory / 1024)  # Placeholder
                        self.metrics.gpu_utilization.labels(gpu_id=str(i)).set(gpu_util)
                        
                time.sleep(self.config.performance_interval)
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(5)


class AlertManager:
    """Manages alerts based on metric thresholds."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_history = defaultdict(list)
        self.alert_cooldown = {}
        
    def check_thresholds(self, metrics: Dict[str, float]):
        """Check if any metrics exceed thresholds."""
        alerts = []
        
        # Latency threshold
        if 'latency_p95' in metrics and metrics['latency_p95'] > self.config.latency_threshold_p95 / 1000:
            alerts.append({
                'type': 'high_latency',
                'severity': 'warning',
                'message': f"P95 latency {metrics['latency_p95']*1000:.1f}ms exceeds threshold {self.config.latency_threshold_p95}ms",
                'value': metrics['latency_p95'] * 1000,
                'threshold': self.config.latency_threshold_p95,
            })
            
        # Error rate threshold
        if 'error_rate' in metrics and metrics['error_rate'] > self.config.error_rate_threshold:
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'critical',
                'message': f"Error rate {metrics['error_rate']*100:.2f}% exceeds threshold {self.config.error_rate_threshold*100:.2f}%",
                'value': metrics['error_rate'] * 100,
                'threshold': self.config.error_rate_threshold * 100,
            })
            
        # Send alerts
        for alert in alerts:
            self._send_alert(alert)
            
    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert notification."""
        alert_key = f"{alert['type']}_{alert['severity']}"
        now = time.time()
        
        # Check cooldown (don't spam alerts)
        if alert_key in self.alert_cooldown:
            if now - self.alert_cooldown[alert_key] < 300:  # 5 minutes cooldown
                return
                
        self.alert_cooldown[alert_key] = now
        
        # Log alert
        logger.warning(f"ALERT: {alert['message']}")
        
        # Store in history
        alert['timestamp'] = datetime.now().isoformat()
        self.alert_history[alert['type']].append(alert)
        
        # Send webhook if configured
        if self.config.alert_webhook_url:
            self._send_webhook(alert)
            
    def _send_webhook(self, alert: Dict[str, Any]):
        """Send webhook notification."""
        import httpx
        
        try:
            payload = {
                'model_name': self.config.model_name,
                'model_version': self.config.model_version,
                'environment': self.config.environment,
                'alert': alert,
            }
            
            # Send async webhook
            asyncio.create_task(self._async_webhook(payload))
            
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            
    async def _async_webhook(self, payload: Dict[str, Any]):
        """Send webhook asynchronously."""
        import httpx
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.config.alert_webhook_url,
                    json=payload,
                    timeout=10.0
                )
                response.raise_for_status()
            except Exception as e:
                logger.error(f"Webhook failed: {e}")


class ModelMonitor:
    """Main model monitoring orchestrator."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.performance_monitor = PerformanceMonitor(config, self.metrics_collector)
        self.system_monitor = SystemMonitor(config, self.metrics_collector)
        self.alert_manager = AlertManager(config)
        
        # Optional components
        self.drift_detector = None
        if config.enable_drift_detection:
            self.drift_detector = DataDriftDetector()
            
        self.performance_analyzer = None
        if config.enable_model_performance:
            self.performance_analyzer = PerformanceAnalyzer()
            
        # External integrations
        self._setup_external_monitoring()
        
        # Background tasks
        self.running = False
        self.monitor_tasks = []
        
    def _setup_external_monitoring(self):
        """Setup external monitoring services."""
        # Azure Monitor
        if self.config.enable_azure_monitor:
            try:
                configure_azure_monitor()
                logger.info("Azure Monitor configured")
            except Exception as e:
                logger.warning(f"Failed to configure Azure Monitor: {e}")
                
        # MLflow
        if self.config.enable_mlflow:
            try:
                mlflow.set_tracking_uri("http://localhost:5000")  # Configure as needed
                mlflow.set_experiment(f"{self.config.model_name}_{self.config.environment}")
                logger.info("MLflow tracking configured")
            except Exception as e:
                logger.warning(f"Failed to configure MLflow: {e}")
                
        # Weights & Biases
        if self.config.enable_wandb:
            try:
                wandb.init(
                    project=f"{self.config.model_name}-monitoring",
                    name=f"{self.config.model_version}-{self.config.environment}",
                    job_type="monitoring",
                )
                logger.info("Weights & Biases configured")
            except Exception as e:
                logger.warning(f"Failed to configure W&B: {e}")
                
    def start(self):
        """Start monitoring."""
        logger.info("Starting model monitoring")
        self.running = True
        
        # Start system monitoring
        self.system_monitor.start()
        
        # Start background monitoring tasks
        self.monitor_tasks.append(
            asyncio.create_task(self._monitoring_loop())
        )
        
    def stop(self):
        """Stop monitoring."""
        logger.info("Stopping model monitoring")
        self.running = False
        
        # Stop system monitoring
        self.system_monitor.stop()
        
        # Cancel background tasks
        for task in self.monitor_tasks:
            task.cancel()
            
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Get performance stats
                perf_stats = self.performance_monitor.get_performance_stats()
                
                # Update throughput metric
                if 'throughput' in perf_stats:
                    self.metrics_collector.throughput.labels(
                        model_name=self.config.model_name,
                        model_version=self.config.model_version
                    ).set(perf_stats['throughput'])
                    
                # Check alert thresholds
                self.alert_manager.check_thresholds(perf_stats)
                
                # Log to external services
                await self._log_to_external_services(perf_stats)
                
                await asyncio.sleep(self.config.performance_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
                
    async def _log_to_external_services(self, metrics: Dict[str, float]):
        """Log metrics to external monitoring services."""
        timestamp = time.time()
        
        # MLflow
        if self.config.enable_mlflow and mlflow.active_run():
            try:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=int(timestamp))
            except Exception as e:
                logger.debug(f"MLflow logging failed: {e}")
                
        # Weights & Biases
        if self.config.enable_wandb and wandb.run:
            try:
                wandb.log(metrics, step=int(timestamp))
            except Exception as e:
                logger.debug(f"W&B logging failed: {e}")
                
    def record_prediction(
        self,
        inputs: Any,
        outputs: Any,
        latency: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a model prediction for monitoring."""
        # Update metrics
        self.performance_monitor.record_request("success")
        
        # Check for drift if enabled
        if self.drift_detector:
            try:
                drift_score = self.drift_detector.detect_drift(inputs, outputs)
                if drift_score > self.config.drift_threshold:
                    self.alert_manager._send_alert({
                        'type': 'data_drift',
                        'severity': 'warning',
                        'message': f"Data drift detected with score {drift_score:.3f}",
                        'value': drift_score,
                        'threshold': self.config.drift_threshold,
                    })
                    
                # Update drift metric
                self.metrics_collector.data_drift.labels(
                    model_name=self.config.model_name,
                    model_version=self.config.model_version
                ).set(drift_score)
                
            except Exception as e:
                logger.debug(f"Drift detection failed: {e}")
                
    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        return {
            'performance': self.performance_monitor.get_performance_stats(),
            'alerts': dict(self.alert_manager.alert_history),
            'system': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'timestamp': datetime.now().isoformat(),
            }
        }
        
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(self.metrics_collector.registry)