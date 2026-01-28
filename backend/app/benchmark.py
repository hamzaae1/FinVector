#!/usr/bin/env python3
"""
FinVector - Performance Benchmark
Measures search performance and generates a report
"""
import time
import statistics
import json
from datetime import datetime
from .finvector_core import FinVectorSearch
from .performance import ModelWarmup, run_benchmark, profiler

def run_full_benchmark():
    """Run comprehensive performance benchmark"""
    print("="*60)
    print("  FINVECTOR PERFORMANCE BENCHMARK")
    print("="*60)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Initialize
    print("\n[1/4] Initializing search engine...")
    start = time.time()
    search = FinVectorSearch(load_image_model=False)  # Skip image model for speed
    init_time = (time.time() - start) * 1000
    print(f"       Initialization: {init_time:.0f}ms")

    # Warmup
    print("\n[2/4] Running model warmup...")
    ModelWarmup.warmup_text_model(search)

    # Benchmark searches
    print("\n[3/4] Running search benchmarks...")
    benchmark_results = run_benchmark(search, iterations=10)

    # Stress test
    print("\n[4/4] Running stress test (50 rapid searches)...")
    stress_times = []
    queries = ["laptop", "headphones", "mouse", "keyboard", "monitor"] * 10

    stress_start = time.time()
    for query in queries:
        start = time.time()
        search.search_products(query=query, max_budget=500.0, limit=5)
        stress_times.append((time.time() - start) * 1000)
    stress_total = (time.time() - stress_start) * 1000

    print(f"       Total time: {stress_total:.0f}ms for 50 searches")
    print(f"       Avg per search: {stress_total/50:.0f}ms")
    print(f"       Min: {min(stress_times):.0f}ms, Max: {max(stress_times):.0f}ms")
    print(f"       Std Dev: {statistics.stdev(stress_times):.0f}ms")

    # Generate report
    print("\n" + "="*60)
    print("  BENCHMARK REPORT")
    print("="*60)

    report = {
        "timestamp": datetime.now().isoformat(),
        "initialization_ms": round(init_time, 2),
        "search_benchmarks": benchmark_results,
        "stress_test": {
            "total_searches": 50,
            "total_time_ms": round(stress_total, 2),
            "avg_ms": round(stress_total/50, 2),
            "min_ms": round(min(stress_times), 2),
            "max_ms": round(max(stress_times), 2),
            "std_dev_ms": round(statistics.stdev(stress_times), 2)
        },
        "performance_target_ms": 2000,
        "target_met": max(stress_times) < 2000
    }

    # Print summary
    print(f"\n  Initialization Time: {report['initialization_ms']:.0f}ms")
    print(f"\n  Search Performance:")
    for bench in benchmark_results:
        print(f"    - {bench['query']}: {bench['avg_ms']:.0f}ms avg")

    print(f"\n  Stress Test (50 searches):")
    print(f"    - Total: {report['stress_test']['total_time_ms']:.0f}ms")
    print(f"    - Average: {report['stress_test']['avg_ms']:.0f}ms")
    print(f"    - Std Dev: {report['stress_test']['std_dev_ms']:.0f}ms")

    print(f"\n  Performance Target (<2000ms): ", end="")
    if report['target_met']:
        print("PASS")
    else:
        print("FAIL")

    # Save report
    report_file = "benchmark_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {report_file}")

    print("\n" + "="*60)
    return report


if __name__ == "__main__":
    run_full_benchmark()
