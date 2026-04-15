"""Performance benchmarking for Diet Recommendation System"""
import time
from functools import wraps
from typing import Callable, Any
from logging_config import logger


def benchmark(func: Callable) -> Callable:
    """
    Decorator to benchmark function execution time
    
    Args:
        func: Function to benchmark
    
    Returns:
        Wrapped function with benchmarking
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        duration_ms = (end_time - start_time) * 1000
        logger.info(f"{func.__name__} executed in {duration_ms:.2f}ms")
        
        return result
    
    return wrapper


class PerformanceMetrics:
    """Track performance metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, metric_name: str, value: float) -> None:
        """
        Record a performance metric
        
        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append(value)
    
    def get_average(self, metric_name: str) -> float:
        """
        Get average value of a metric
        
        Args:
            metric_name: Name of the metric
        
        Returns:
            Average value
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return 0.0
        
        return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])
    
    def get_statistics(self, metric_name: str) -> dict:
        """
        Get statistics of a metric
        
        Args:
            metric_name: Name of the metric
        
        Returns:
            Dictionary of statistics
        """
        values = self.metrics.get(metric_name, [])
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "average": sum(values) / len(values),
            "latest": values[-1]
        }
    
    def get_all_statistics(self) -> dict:
        """
        Get all statistics
        
        Returns:
            Dictionary of all statistics
        """
        return {
            metric: self.get_statistics(metric)
            for metric in self.metrics.keys()
        }


# Global metrics instance
performance_metrics = PerformanceMetrics()


class InferenceTimer:
    """Context manager for timing inference"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            performance_metrics.record_metric(f"inference_time_{self.model_name}", duration_ms)
            logger.debug(f"Inference time for {self.model_name}: {duration_ms:.2f}ms")
