"""
FinVector - Performance Optimization Module
Includes caching, warmup, and benchmarking utilities
"""
import time
import hashlib
import json
from functools import lru_cache, wraps
from typing import Dict, Any, Optional, Callable
from collections import OrderedDict
import threading


class LRUCache:
    """Thread-safe LRU cache for search results"""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def _make_key(self, *args, **kwargs) -> str:
        """Create a cache key from arguments"""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            # Check TTL
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                del self.cache[key]
                del self.timestamps[key]
                self.misses += 1
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]

    def set(self, key: str, value: Any):
        """Set item in cache"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]

            self.cache[key] = value
            self.timestamps[key] = time.time()

    def clear(self):
        """Clear the cache"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> Dict:
        """Get cache statistics"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate_percent": round(hit_rate, 2)
            }


# Global search cache
search_cache = LRUCache(max_size=100, ttl_seconds=300)


def cached_search(func: Callable) -> Callable:
    """Decorator to cache search results"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key
        key = search_cache._make_key(*args, **kwargs)

        # Check cache
        cached = search_cache.get(key)
        if cached is not None:
            return cached

        # Execute and cache
        result = func(*args, **kwargs)
        search_cache.set(key, result)
        return result

    return wrapper


class PerformanceProfiler:
    """Profile and benchmark search operations"""

    def __init__(self):
        self.timings: Dict[str, list] = {}
        self.lock = threading.Lock()

    def record(self, operation: str, duration_ms: float):
        """Record a timing measurement"""
        with self.lock:
            if operation not in self.timings:
                self.timings[operation] = []
            self.timings[operation].append(duration_ms)

            # Keep only last 1000 measurements per operation
            if len(self.timings[operation]) > 1000:
                self.timings[operation] = self.timings[operation][-1000:]

    def get_stats(self, operation: str = None) -> Dict:
        """Get performance statistics"""
        with self.lock:
            if operation:
                times = self.timings.get(operation, [])
                if not times:
                    return {}
                return {
                    "operation": operation,
                    "count": len(times),
                    "avg_ms": round(sum(times) / len(times), 2),
                    "min_ms": round(min(times), 2),
                    "max_ms": round(max(times), 2),
                    "p95_ms": round(sorted(times)[int(len(times) * 0.95)] if times else 0, 2)
                }
            else:
                return {op: self.get_stats(op) for op in self.timings}


# Global profiler
profiler = PerformanceProfiler()


def timed(operation_name: str):
    """Decorator to time and profile operations"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start) * 1000
                profiler.record(operation_name, duration_ms)
        return wrapper
    return decorator


class ModelWarmup:
    """Warmup models to reduce cold start latency"""

    @staticmethod
    def warmup_text_model(search_engine):
        """Warmup the text embedding model"""
        print("Warming up text model...")
        start = time.time()

        # Run a few dummy encodings
        warmup_texts = [
            "wireless headphones",
            "gaming laptop",
            "mechanical keyboard"
        ]

        for text in warmup_texts:
            search_engine.text_model.encode(text)

        duration = (time.time() - start) * 1000
        print(f"Text model warmup complete: {duration:.0f}ms")

    @staticmethod
    def warmup_image_model(search_engine):
        """Warmup the CLIP image model"""
        print("Warming up CLIP model...")
        start = time.time()

        # Trigger lazy loading and warmup
        warmup_texts = [
            "a photo of headphones",
            "a gaming laptop",
            "a wireless mouse"
        ]

        for text in warmup_texts:
            search_engine.image_model.encode(text)

        duration = (time.time() - start) * 1000
        print(f"CLIP model warmup complete: {duration:.0f}ms")

    @staticmethod
    def warmup_all(search_engine):
        """Full warmup of all models"""
        print("\n=== Starting FinVector Warmup ===\n")
        total_start = time.time()

        ModelWarmup.warmup_text_model(search_engine)

        # Optionally warmup image model (can be slow)
        try:
            ModelWarmup.warmup_image_model(search_engine)
        except Exception as e:
            print(f"Image model warmup skipped: {e}")

        total_duration = (time.time() - total_start) * 1000
        print(f"\n=== Warmup complete: {total_duration:.0f}ms ===\n")


def run_benchmark(search_engine, iterations: int = 10):
    """Run a performance benchmark"""
    print("\n=== Running Performance Benchmark ===\n")

    queries = [
        ("laptop programming", 1000.0),
        ("wireless headphones", 200.0),
        ("gaming mouse", 100.0),
        ("mechanical keyboard", 150.0),
        ("4k monitor", 500.0),
    ]

    results = []

    for query, budget in queries:
        times = []
        for _ in range(iterations):
            start = time.time()
            search_engine.search_products(query=query, max_budget=budget, limit=10)
            duration = (time.time() - start) * 1000
            times.append(duration)

        avg = sum(times) / len(times)
        results.append({
            "query": query,
            "budget": budget,
            "iterations": iterations,
            "avg_ms": round(avg, 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2)
        })

        print(f"  '{query}' (${budget}): avg {avg:.0f}ms, min {min(times):.0f}ms, max {max(times):.0f}ms")

    # Overall stats
    all_avgs = [r["avg_ms"] for r in results]
    print(f"\n  Overall Average: {sum(all_avgs)/len(all_avgs):.0f}ms")

    # Check against target
    target_ms = 2000
    if max(all_avgs) < target_ms:
        print(f"  Target (<{target_ms}ms): PASS")
    else:
        print(f"  Target (<{target_ms}ms): FAIL")

    return results


if __name__ == "__main__":
    # Demo the performance utilities
    print("FinVector Performance Module")
    print("============================\n")

    # Cache demo
    cache = LRUCache(max_size=5, ttl_seconds=60)
    cache.set("key1", "value1")
    cache.set("key2", "value2")

    print("Cache Demo:")
    print(f"  Get 'key1': {cache.get('key1')}")
    print(f"  Get 'missing': {cache.get('missing')}")
    print(f"  Stats: {cache.stats()}")

    # Profiler demo
    print("\nProfiler Demo:")
    for i in range(10):
        profiler.record("test_operation", 100 + i * 10)

    print(f"  Stats: {profiler.get_stats('test_operation')}")
