"""
FinVector - End-to-End Test Suite
Tests the complete system including API endpoints, data integrity, and edge cases
"""
import requests
import time
import json
from typing import Dict, List, Tuple

BASE_URL = "http://localhost:8000"


class TestResult:
    def __init__(self, name: str, passed: bool, message: str, duration_ms: float):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration_ms = duration_ms


def api_call(method: str, endpoint: str, data: Dict = None, timeout: int = 30) -> Tuple[int, Dict, float]:
    """Make API call and return status, response, and duration"""
    start = time.time()
    url = f"{BASE_URL}{endpoint}"

    try:
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=timeout)
        else:
            raise ValueError(f"Unknown method: {method}")

        duration_ms = (time.time() - start) * 1000
        return response.status_code, response.json(), duration_ms
    except requests.exceptions.ConnectionError:
        return 0, {"error": "Connection refused - is the server running?"}, 0
    except Exception as e:
        return 0, {"error": str(e)}, 0


def test_health_endpoint() -> TestResult:
    """Test 1: Health endpoint returns healthy status"""
    status, data, duration = api_call("GET", "/health")

    if status == 200 and data.get("status") == "healthy":
        return TestResult("Health Check", True, "API is healthy", duration)
    else:
        return TestResult("Health Check", False, f"Unhealthy: {data}", duration)


def test_root_endpoint() -> TestResult:
    """Test 2: Root endpoint returns API info"""
    status, data, duration = api_call("GET", "/")

    if status == 200 and "FinVector" in data.get("message", ""):
        return TestResult("Root Endpoint", True, "API info returned correctly", duration)
    else:
        return TestResult("Root Endpoint", False, f"Unexpected response: {data}", duration)


def test_search_basic() -> TestResult:
    """Test 3: Basic search returns results"""
    status, data, duration = api_call("POST", "/search", {
        "query": "laptop",
        "max_budget": 1000.0,
        "limit": 5
    })

    if status == 200 and data.get("success") and data.get("count", 0) > 0:
        return TestResult("Basic Search", True, f"Found {data['count']} products", duration)
    else:
        return TestResult("Basic Search", False, f"Search failed: {data}", duration)


def test_search_budget_compliance() -> TestResult:
    """Test 4: All search results respect budget constraint"""
    budget = 100.0
    status, data, duration = api_call("POST", "/search", {
        "query": "electronics",
        "max_budget": budget,
        "limit": 20
    })

    if status != 200:
        return TestResult("Budget Compliance", False, f"API error: {data}", duration)

    results = data.get("results", [])
    over_budget = [r for r in results if r["price"] > budget]

    if over_budget:
        return TestResult("Budget Compliance", False,
                         f"{len(over_budget)} products over ${budget} budget", duration)
    else:
        return TestResult("Budget Compliance", True,
                         f"All {len(results)} products within ${budget} budget", duration)


def test_search_rating_filter() -> TestResult:
    """Test 5: Rating filter works correctly"""
    min_rating = 4.5
    status, data, duration = api_call("POST", "/search", {
        "query": "headphones",
        "max_budget": 500.0,
        "min_rating": min_rating,
        "limit": 10
    })

    if status != 200:
        return TestResult("Rating Filter", False, f"API error: {data}", duration)

    results = data.get("results", [])
    low_rated = [r for r in results if r["rating"] < min_rating]

    if low_rated:
        return TestResult("Rating Filter", False,
                         f"{len(low_rated)} products below {min_rating} rating", duration)
    else:
        return TestResult("Rating Filter", True,
                         f"All {len(results)} products have rating >= {min_rating}", duration)


def test_search_empty_results() -> TestResult:
    """Test 6: Search with impossible constraints returns empty gracefully"""
    status, data, duration = api_call("POST", "/search", {
        "query": "laptop",
        "max_budget": 1.0,  # No laptop costs $1
        "limit": 10
    })

    if status == 200 and data.get("success"):
        count = data.get("count", 0)
        if count == 0:
            return TestResult("Empty Results", True, "Gracefully returned 0 results", duration)
        else:
            return TestResult("Empty Results", False,
                             f"Found {count} laptops for $1?", duration)
    else:
        return TestResult("Empty Results", False, f"API error: {data}", duration)


def test_alternatives_endpoint() -> TestResult:
    """Test 7: Alternatives endpoint works"""
    # First get a product
    status, data, _ = api_call("POST", "/search", {
        "query": "mouse",
        "max_budget": 200.0,
        "limit": 1
    })

    if status != 200 or not data.get("results"):
        return TestResult("Alternatives", False, "Could not find product for test", 0)

    product_id = data["results"][0]["product_id"]
    original_price = data["results"][0]["price"]

    # Now find alternatives
    status, data, duration = api_call("POST", "/alternatives", {
        "product_id": str(product_id),
        "max_budget": original_price - 5,
        "min_rating": 4.0,
        "limit": 3
    })

    if status == 200 and data.get("success"):
        alts = data.get("alternatives", [])
        expensive = [a for a in alts if a["price"] >= original_price]
        if expensive:
            return TestResult("Alternatives", False,
                             "Some alternatives are not cheaper", duration)
        return TestResult("Alternatives", True,
                         f"Found {len(alts)} cheaper alternatives", duration)
    else:
        return TestResult("Alternatives", False, f"API error: {data}", duration)


def test_budget_stretcher_endpoint() -> TestResult:
    """Test 8: Budget stretcher returns bundle suggestions"""
    # First get a product
    status, data, _ = api_call("POST", "/search", {
        "query": "keyboard",
        "max_budget": 200.0,
        "limit": 1
    })

    if status != 200 or not data.get("results"):
        return TestResult("Budget Stretcher", False, "Could not find product for test", 0)

    product_id = data["results"][0]["product_id"]

    status, data, duration = api_call("POST", "/budget-stretcher", {
        "product_id": str(product_id),
        "total_budget": 200.0
    })

    if status == 200:
        if data.get("success"):
            return TestResult("Budget Stretcher", True,
                             f"Bundle total: ${data.get('bundle_total', 0):.2f}", duration)
        else:
            return TestResult("Budget Stretcher", True,
                             f"Graceful handling: {data.get('message', 'no alternatives')}", duration)
    else:
        return TestResult("Budget Stretcher", False, f"API error: {data}", duration)


def test_analytics_endpoint() -> TestResult:
    """Test 9: Analytics endpoint returns data"""
    status, data, duration = api_call("GET", "/analytics/summary")

    if status == 200 and data.get("success"):
        return TestResult("Analytics", True,
                         f"Total searches tracked: {data.get('total_searches', 0)}", duration)
    elif "error" in data and "not enabled" in str(data.get("error", "")):
        return TestResult("Analytics", True, "Analytics disabled (acceptable)", duration)
    else:
        return TestResult("Analytics", False, f"Analytics error: {data}", duration)


def test_response_time_performance() -> TestResult:
    """Test 10: Response time is under 2 seconds (as per README)"""
    times = []

    for query in ["laptop", "headphones", "mouse", "keyboard", "monitor"]:
        status, data, duration = api_call("POST", "/search", {
            "query": query,
            "max_budget": 500.0,
            "limit": 10
        })
        if status == 200:
            times.append(duration)

    if not times:
        return TestResult("Response Time", False, "No successful searches", 0)

    avg_time = sum(times) / len(times)
    max_time = max(times)

    if max_time > 2000:
        return TestResult("Response Time", False,
                         f"Max response time {max_time:.0f}ms exceeds 2s limit", avg_time)
    else:
        return TestResult("Response Time", True,
                         f"Avg: {avg_time:.0f}ms, Max: {max_time:.0f}ms (under 2s)", avg_time)


def test_invalid_input_handling() -> TestResult:
    """Test 11: API handles invalid input gracefully"""
    # Test with negative budget
    status, data, duration = api_call("POST", "/search", {
        "query": "laptop",
        "max_budget": -100.0
    })

    # Should either return empty results or handle gracefully
    if status in [200, 400, 422]:
        return TestResult("Invalid Input", True,
                         f"Handled negative budget gracefully (status {status})", duration)
    else:
        return TestResult("Invalid Input", False,
                         f"Unexpected status {status}: {data}", duration)


def test_concurrent_requests() -> TestResult:
    """Test 12: API handles concurrent requests"""
    import concurrent.futures

    def make_search():
        return api_call("POST", "/search", {
            "query": "laptop",
            "max_budget": 1000.0,
            "limit": 5
        })

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_search) for _ in range(5)]
        results = [f.result() for f in futures]

    total_duration = (time.time() - start) * 1000
    successes = sum(1 for status, _, _ in results if status == 200)

    if successes == 5:
        return TestResult("Concurrent Requests", True,
                         f"5/5 concurrent requests succeeded in {total_duration:.0f}ms", total_duration)
    else:
        return TestResult("Concurrent Requests", False,
                         f"Only {successes}/5 requests succeeded", total_duration)


def run_all_tests():
    """Run all end-to-end tests"""
    print("\n" + "="*70)
    print("   FINVECTOR - END-TO-END TEST SUITE")
    print("="*70)

    tests = [
        test_health_endpoint,
        test_root_endpoint,
        test_search_basic,
        test_search_budget_compliance,
        test_search_rating_filter,
        test_search_empty_results,
        test_alternatives_endpoint,
        test_budget_stretcher_endpoint,
        test_analytics_endpoint,
        test_response_time_performance,
        test_invalid_input_handling,
        test_concurrent_requests,
    ]

    results = []
    for test_func in tests:
        print(f"\nRunning: {test_func.__doc__}")
        result = test_func()
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.name}: {result.message}")
        if result.duration_ms > 0:
            print(f"        Duration: {result.duration_ms:.0f}ms")

    # Summary
    print("\n" + "="*70)
    print("   TEST SUMMARY")
    print("="*70)

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}")

    print(f"\n  Total: {passed} passed, {failed} failed out of {len(results)} tests")

    if failed == 0:
        print("\n  ALL TESTS PASSED!")
        return True
    else:
        print(f"\n  {failed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
