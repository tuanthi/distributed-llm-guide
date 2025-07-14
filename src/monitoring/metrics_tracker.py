"""
Metrics tracking and aggregation utilities.
"""

import time
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
import json
from pathlib import Path

import numpy as np
from loguru import logger


class MetricsTracker:
    """Tracks and aggregates various metrics during training and inference."""
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.metrics = defaultdict(lambda: deque(maxlen=buffer_size))
        self.timestamps = defaultdict(lambda: deque(maxlen=buffer_size))
        
    def log_metric(self, name: str, value: float, timestamp: Optional[float] = None):
        """Log a single metric value."""
        if timestamp is None:
            timestamp = time.time()
            
        self.metrics[name].append(value)
        self.timestamps[name].append(timestamp)
        
    def log_metrics(self, metrics_dict: Dict[str, float], timestamp: Optional[float] = None):
        """Log multiple metrics at once."""
        if timestamp is None:
            timestamp = time.time()
            
        for name, value in metrics_dict.items():
            self.log_metric(name, value, timestamp)
            
    def get_metric_stats(self, name: str, window_size: Optional[int] = None) -> Dict[str, float]:
        """Get statistics for a metric."""
        if name not in self.metrics:
            return {}
            
        values = list(self.metrics[name])
        if window_size:
            values = values[-window_size:]
            
        if not values:
            return {}
            
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "p50": np.percentile(values, 50),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
            "count": len(values),
        }
        
    def get_all_stats(self, window_size: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """Get statistics for all tracked metrics."""
        return {
            name: self.get_metric_stats(name, window_size)
            for name in self.metrics.keys()
        }
        
    def export_metrics(self, filepath: str, format: str = "json"):
        """Export metrics to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert deques to lists for serialization
        export_data = {
            "metrics": {name: list(values) for name, values in self.metrics.items()},
            "timestamps": {name: list(timestamps) for name, timestamps in self.timestamps.items()},
            "export_time": time.time(),
        }
        
        if format == "json":
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        logger.info(f"Metrics exported to {filepath}")
        
    def clear_metrics(self, metric_name: Optional[str] = None):
        """Clear metrics data."""
        if metric_name:
            if metric_name in self.metrics:
                self.metrics[metric_name].clear()
                self.timestamps[metric_name].clear()
        else:
            self.metrics.clear()
            self.timestamps.clear()
            
    def get_metric_names(self) -> List[str]:
        """Get list of tracked metric names."""
        return list(self.metrics.keys())