from app.finvector_core import FinVectorSearch

# Initialize
search = FinVectorSearch()

# Test 1: Basic search
print("🔍 Test 1: Budget-aware search")
results = search.search_products(
    query="wireless gaming headphones",
    max_budget=100.0,
    limit=5
)

for r in results:
    print(f"  • {r['title']} - ${r['price']} ({r['match_quality']})")
    print(f"    {r['explanation']}\n")

# Test 2: Find alternatives
print("\n💡 Test 2: Smart alternatives")
if results:
    alternatives = search.find_alternatives(
        product_id=results[0]['product_id'],
        max_budget=50.0
    )
    for a in alternatives:
        print(f"  • {a['title']} - ${a['price']} (Save ${a['savings']})")

# Test 3: Budget stretcher
print("\n🎁 Test 3: Budget Stretcher")
if results:
    bundle = search.budget_stretcher(
        product_id=results[0]['product_id'],
        total_budget=100.0
    )
    if bundle['success']:
        print(f"  Alternative: {bundle['alternative']['title']} - ${bundle['alternative']['price']}")
        print(f"  + Accessories:")
        for comp in bundle['complementary_products']:
            print(f"    • {comp['title']} - ${comp['price']}")
        print(f"  Total: ${bundle['bundle_total']} (Savings: ${bundle['savings']})")
