"""Performance benchmarking utilities for Diet Recommendation System"""
import time
import logging
from functools import wraps
from typing import Dict, Any, Callable, Optional
from contextlib import contextmanager
import psutil
import os

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor performance metrics for model operations"""

    def __init__(self):
        self.metrics = {}

    @contextmanager
    def measure(self, operation_name: str):
        """Context manager to measure operation performance"""
        start_time = time.perf_counter()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

            duration = (end_time - start_time) * 1000  # milliseconds
            memory_delta = end_memory - start_memory

            self.metrics[operation_name] = {
                'duration_ms': round(duration, 2),
                'memory_delta_mb': round(memory_delta, 2),
                'timestamp': time.time()
            }

            logger.info(f"{operation_name}: {duration:.2f}ms, Memory: {memory_delta:.2f}MB")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.copy()

    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics.clear()

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def benchmark_inference(func: Callable) -> Callable:
    """Decorator to benchmark inference operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        operation_name = f"{func.__name__}"
        with performance_monitor.measure(operation_name):
            result = func(*args, **kwargs)
        return result
    return wrapper

def benchmark_model_loading(func: Callable) -> Callable:
    """Decorator to benchmark model loading operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        operation_name = f"model_loading_{func.__name__}"
        with performance_monitor.measure(operation_name):
            result = func(*args, **kwargs)
        return result
    return wrapper

def benchmark_data_processing(func: Callable) -> Callable:
    """Decorator to benchmark data processing operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        operation_name = f"data_processing_{func.__name__}"
        with performance_monitor.measure(operation_name):
            result = func(*args, **kwargs)
        return result
    return wrapper

def get_system_info() -> Dict[str, Any]:
    """Get system information for benchmarking context"""
    return {
        'cpu_count': psutil.cpu_count(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_total': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
        'memory_available': psutil.virtual_memory().available / 1024 / 1024 / 1024,  # GB
        'memory_percent': psutil.virtual_memory().percent,
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
    }

def run_performance_test(test_func: Callable, iterations: int = 10, warmup: int = 2) -> Dict[str, Any]:
    """Run performance test with multiple iterations"""
    logger.info(f"Running performance test: {test_func.__name__}")

    # Warmup runs
    for i in range(warmup):
        test_func()
        logger.debug(f"Warmup iteration {i+1}/{warmup} completed")

    # Actual test runs
    durations = []
    for i in range(iterations):
        start_time = time.perf_counter()
        test_func()
        end_time = time.perf_counter()
        duration = (end_time - start_time) * 1000  # ms
        durations.append(duration)
        logger.debug(f"Test iteration {i+1}/{iterations}: {duration:.2f}ms")

    # Calculate statistics
    avg_duration = sum(durations) / len(durations)
    min_duration = min(durations)
    max_duration = max(durations)
    std_duration = (sum((x - avg_duration) ** 2 for x in durations) / len(durations)) ** 0.5

    results = {
        'test_name': test_func.__name__,
        'iterations': iterations,
        'avg_duration_ms': round(avg_duration, 2),
        'min_duration_ms': round(min_duration, 2),
        'max_duration_ms': round(max_duration, 2),
        'std_duration_ms': round(std_duration, 2),
        'system_info': get_system_info()
    }

    logger.info(f"Performance test completed: {avg_duration:.2f}ms avg ({std_duration:.2f}ms std)")
    return results

# Convenience functions for common benchmarks
def benchmark_recommendation(nutrition_input: list, metric: str = 'nutritional_mae',
                           bmi: float = 25.0, goal: str = 'maintenance',
                           iterations: int = 5) -> Dict[str, Any]:
    """Benchmark recommendation performance"""
    from model import recommend

    def test_recommendation():
        recommend(
            dataframe=None,
            _input=nutrition_input,
            ingredients=[],
            params={'n_neighbors': 5, 'return_distance': False},
            metric=metric,
            bmi=bmi,
            goal=goal
        )

    return run_performance_test(test_recommendation, iterations=iterations)

def benchmark_model_loading() -> Dict[str, Any]:
    """Benchmark model loading performance"""
    def test_model_loading():
        import importlib
        import model
        importlib.reload(model)

    return run_performance_test(test_model_loading, iterations=3)

if __name__ == "__main__":
    # Example usage
    sample_input = [500, 20, 5, 50, 400, 40, 10, 5, 35]

    print("Running performance benchmarks...")

    # Test recommendation performance
    rec_results = benchmark_recommendation(sample_input)
    print(f"Recommendation Performance: {rec_results['avg_duration_ms']:.2f}ms avg")

    # Test model loading performance
    load_results = benchmark_model_loading()
    print(f"Model Loading Performance: {load_results['avg_duration_ms']:.2f}ms avg")

    # Print system info
    system_info = get_system_info()
    print(f"System: {system_info['cpu_count']} CPUs, {system_info['memory_total']:.1f}GB RAM")