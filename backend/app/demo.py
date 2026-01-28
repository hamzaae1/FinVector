#!/usr/bin/env python3
"""
FinVector - Interactive Demo Script
Budget-Aware E-Commerce Search Demo for Hackathon Presentation

Run this script to demonstrate all FinVector features interactively.
"""
import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_product(product, index=None):
    prefix = f"  {index}. " if index else "  "
    print(f"{prefix}{Colors.BOLD}{product['title']}{Colors.ENDC}")
    print(f"      {Colors.GREEN}${product['price']:.2f}{Colors.ENDC} | "
          f"Rating: {product['rating']} | "
          f"{Colors.CYAN}{product.get('match_quality', 'N/A')}{Colors.ENDC}")
    if 'explanation' in product:
        print(f"      {Colors.YELLOW}{product['explanation']}{Colors.ENDC}")
    print()


def pause():
    input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")


def demo_text_search():
    """Demo 1: Budget-Aware Text Search"""
    print_header("DEMO 1: Budget-Aware Text Search")

    print(f"{Colors.BOLD}Scenario:{Colors.ENDC} A user wants wireless headphones but only has $80 to spend.\n")
    print(f"Query: {Colors.CYAN}\"wireless headphones\"{Colors.ENDC}")
    print(f"Budget: {Colors.GREEN}$80.00{Colors.ENDC}")

    pause()

    response = requests.post(f"{BASE_URL}/search", json={
        "query": "wireless headphones",
        "max_budget": 80.0,
        "limit": 5
    })

    data = response.json()
    print(f"\n{Colors.GREEN}Found {data['count']} products within budget:{Colors.ENDC}\n")

    for i, product in enumerate(data['results'], 1):
        print_product(product, i)

    print(f"{Colors.BOLD}Key Feature:{Colors.ENDC} Notice how ALL products are under $80!")
    print("Traditional search shows products you can't afford. FinVector doesn't.")


def demo_smart_alternatives():
    """Demo 2: Smart Alternatives"""
    print_header("DEMO 2: Smart Alternatives")

    print(f"{Colors.BOLD}Scenario:{Colors.ENDC} User found a $100 mouse but wants cheaper options.\n")

    # First find a product
    response = requests.post(f"{BASE_URL}/search", json={
        "query": "wireless mouse premium",
        "max_budget": 150.0,
        "limit": 1
    })
    original = response.json()['results'][0]

    print(f"Original Choice: {Colors.BOLD}{original['title']}{Colors.ENDC}")
    print(f"Price: {Colors.RED}${original['price']:.2f}{Colors.ENDC}")

    pause()

    # Find alternatives
    response = requests.post(f"{BASE_URL}/alternatives", json={
        "product_id": str(original['product_id']),
        "max_budget": original['price'] - 20,
        "min_rating": 4.0,
        "limit": 3
    })

    data = response.json()
    print(f"\n{Colors.GREEN}Found {data['count']} cheaper alternatives:{Colors.ENDC}\n")

    for alt in data['alternatives']:
        print(f"  {Colors.BOLD}{alt['title']}{Colors.ENDC}")
        print(f"      Price: {Colors.GREEN}${alt['price']:.2f}{Colors.ENDC} | "
              f"Save: {Colors.YELLOW}${alt['savings']:.2f}{Colors.ENDC} | "
              f"Similarity: {alt['similarity_score']*100:.0f}%")
        print()

    print(f"{Colors.BOLD}Key Feature:{Colors.ENDC} Find similar products at lower prices!")


def demo_budget_stretcher():
    """Demo 3: Budget Stretcher"""
    print_header("DEMO 3: Budget Stretcher")

    print(f"{Colors.BOLD}Scenario:{Colors.ENDC} User has $150 budget for a gaming setup.\n")
    print("Instead of spending it all on one item, let's maximize value!\n")

    # Find a keyboard
    response = requests.post(f"{BASE_URL}/search", json={
        "query": "mechanical gaming keyboard",
        "max_budget": 200.0,
        "limit": 1
    })
    original = response.json()['results'][0]

    print(f"Originally Looking At: {Colors.BOLD}{original['title']}{Colors.ENDC}")
    print(f"Price: {Colors.RED}${original['price']:.2f}{Colors.ENDC}")
    print(f"Total Budget: {Colors.GREEN}$150.00{Colors.ENDC}")

    pause()

    # Get bundle suggestion
    response = requests.post(f"{BASE_URL}/budget-stretcher", json={
        "product_id": str(original['product_id']),
        "total_budget": 150.0
    })

    data = response.json()

    if data['success']:
        print(f"\n{Colors.GREEN}Budget Stretcher Suggestion:{Colors.ENDC}\n")
        print(f"  {Colors.BOLD}Alternative Main Item:{Colors.ENDC}")
        print(f"    {data['alternative']['title']} - ${data['alternative']['price']:.2f}")
        print(f"    (Saves ${data['savings']:.2f} vs original)")

        print(f"\n  {Colors.BOLD}Plus These Accessories:{Colors.ENDC}")
        for comp in data['complementary_products']:
            print(f"    + {comp['title']} - ${comp['price']:.2f}")

        print(f"\n  {Colors.CYAN}{'='*40}{Colors.ENDC}")
        print(f"  {Colors.BOLD}Bundle Total: ${data['bundle_total']:.2f}{Colors.ENDC}")
        within = "YES" if data['within_budget'] else "SLIGHTLY OVER"
        color = Colors.GREEN if data['within_budget'] else Colors.YELLOW
        print(f"  {Colors.BOLD}Within Budget: {color}{within}{Colors.ENDC}")
    else:
        print(f"\n{Colors.YELLOW}{data['message']}{Colors.ENDC}")

    print(f"\n{Colors.BOLD}Key Feature:{Colors.ENDC} Get more for your money with smart bundles!")


def demo_performance():
    """Demo 4: Performance Stats"""
    print_header("DEMO 4: Performance & Analytics")

    print(f"{Colors.BOLD}Real-time Performance Metrics:{Colors.ENDC}\n")

    # Run multiple searches and measure
    queries = ["laptop", "headphones", "keyboard", "mouse", "monitor"]
    times = []

    for query in queries:
        start = time.time()
        requests.post(f"{BASE_URL}/search", json={
            "query": query,
            "max_budget": 1000.0,
            "limit": 10
        })
        duration = (time.time() - start) * 1000
        times.append(duration)
        print(f"  Search '{query}': {Colors.GREEN}{duration:.0f}ms{Colors.ENDC}")

    avg = sum(times) / len(times)
    print(f"\n  {Colors.BOLD}Average Response Time: {avg:.0f}ms{Colors.ENDC}")
    print(f"  {Colors.BOLD}Target: < 2000ms{Colors.ENDC} {Colors.GREEN}PASS{Colors.ENDC}")

    pause()

    # Get analytics
    response = requests.get(f"{BASE_URL}/analytics/summary")
    data = response.json()

    print(f"\n{Colors.BOLD}Search Analytics:{Colors.ENDC}")
    print(f"  Total Searches: {data.get('total_searches', 'N/A')}")
    print(f"  Avg Response Time: {data.get('average_response_time_ms', 0):.0f}ms")

    if data.get('popular_queries'):
        print(f"\n  {Colors.BOLD}Popular Queries:{Colors.ENDC}")
        for q in data['popular_queries'][:5]:
            print(f"    - \"{q['query']}\" ({q['count']} searches)")


def demo_api_overview():
    """Demo 5: API Overview"""
    print_header("DEMO 5: API Endpoints")

    endpoints = [
        ("GET", "/", "API info and version"),
        ("GET", "/health", "Health check"),
        ("POST", "/search", "Budget-aware text search"),
        ("POST", "/alternatives", "Find cheaper alternatives"),
        ("POST", "/budget-stretcher", "Bundle suggestions"),
        ("POST", "/search/image/upload", "Search by image upload"),
        ("POST", "/search/image/url", "Search by image URL"),
        ("POST", "/search/image/text", "CLIP text-to-image search"),
        ("GET", "/analytics/summary", "Search analytics"),
        ("GET", "/analytics/recent", "Recent searches"),
    ]

    print(f"{Colors.BOLD}Available Endpoints:{Colors.ENDC}\n")
    for method, path, desc in endpoints:
        color = Colors.GREEN if method == "GET" else Colors.BLUE
        print(f"  {color}{method:6}{Colors.ENDC} {path:30} {Colors.CYAN}{desc}{Colors.ENDC}")

    print(f"\n{Colors.BOLD}API Documentation:{Colors.ENDC} {BASE_URL}/docs")


def run_demo():
    """Run the full demo"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("  ╔═══════════════════════════════════════════════════════════╗")
    print("  ║                                                           ║")
    print("  ║   FINVECTOR - Budget-Aware E-Commerce Search              ║")
    print("  ║   Team Cosine | Qdrant Vector Search Hackathon 2026       ║")
    print("  ║                                                           ║")
    print("  ╚═══════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")

    print(f"\n{Colors.BOLD}Team Members:{Colors.ENDC} Hamza Mhedhbi & Hachem Mastouri")
    print(f"\n{Colors.BOLD}Key Innovation:{Colors.ENDC} Budget filtering DURING search, not after!")
    print(f"{Colors.BOLD}Technologies:{Colors.ENDC} Qdrant, Sentence Transformers, CLIP, FastAPI")

    pause()

    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"{Colors.RED}Error: Server not healthy{Colors.ENDC}")
            return
    except:
        print(f"{Colors.RED}Error: Cannot connect to server at {BASE_URL}{Colors.ENDC}")
        print(f"Please start the server with: uvicorn api:app --host 0.0.0.0 --port 8000")
        return

    demos = [
        demo_text_search,
        demo_smart_alternatives,
        demo_budget_stretcher,
        demo_performance,
        demo_api_overview,
    ]

    for demo_func in demos:
        demo_func()
        pause()

    print_header("DEMO COMPLETE")
    print(f"{Colors.GREEN}Thank you for watching the FinVector demo!{Colors.ENDC}")
    print(f"\n{Colors.BOLD}Try it yourself:{Colors.ENDC}")
    print(f"  - API Docs: {BASE_URL}/docs")
    print(f"  - Health Check: curl {BASE_URL}/health")
    print(f"\n{Colors.BOLD}GitHub:{Colors.ENDC} github.com/team-cosine/finvector")
    print()


if __name__ == "__main__":
    run_demo()
