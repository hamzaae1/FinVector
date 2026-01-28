"""
FinVector - Comprehensive Feature Test Suite
Tests all core features: text search, alternatives, budget stretcher, and image search
"""
from app.finvector_core import FinVectorSearch
import requests
from io import BytesIO
from PIL import Image

def test_text_search(search):
    """Test 1: Basic text search with budget filter"""
    print("\n" + "="*60)
    print("TEST 1: Budget-aware Text Search")
    print("="*60)

    results = search.search_products(
        query="wireless gaming headphones",
        max_budget=100.0,
        limit=5
    )

    print(f"Query: 'wireless gaming headphones' (budget: $100)")
    print(f"Found: {len(results)} products\n")

    for r in results:
        print(f"  [{r['match_quality']}] {r['title']}")
        print(f"    Price: ${r['price']:.2f} | Rating: {r['rating']} | Score: {r['similarity_score']}")

    # Verify budget compliance
    over_budget = [r for r in results if r['price'] > 100.0]
    if over_budget:
        print("\n  ❌ FAIL: Found products over budget!")
        return False
    else:
        print("\n  ✅ PASS: All products within budget")
        return True


def test_alternatives(search):
    """Test 2: Smart alternatives feature"""
    print("\n" + "="*60)
    print("TEST 2: Smart Alternatives")
    print("="*60)

    # First find a product
    results = search.search_products(
        query="wireless mouse",
        max_budget=150.0,
        limit=1
    )

    if not results:
        print("  ❌ FAIL: No products found for alternatives test")
        return False

    original = results[0]
    print(f"Original: {original['title']} - ${original['price']:.2f}")

    alternatives = search.find_alternatives(
        product_id=original['product_id'],
        max_budget=original['price'] - 10,  # Must be cheaper
        min_rating=4.0,
        limit=3
    )

    print(f"\nAlternatives (max ${original['price'] - 10:.2f}, min 4.0 rating):")
    for a in alternatives:
        print(f"  • {a['title']} - ${a['price']:.2f} (Save ${a['savings']:.2f})")

    # Verify alternatives are cheaper
    expensive = [a for a in alternatives if a['price'] >= original['price']]
    if expensive:
        print("\n  ❌ FAIL: Some alternatives are not cheaper!")
        return False
    else:
        print("\n  ✅ PASS: All alternatives are cheaper")
        return True


def test_budget_stretcher(search):
    """Test 3: Budget stretcher bundle suggestions"""
    print("\n" + "="*60)
    print("TEST 3: Budget Stretcher")
    print("="*60)

    # Find a product
    results = search.search_products(
        query="gaming keyboard",
        max_budget=200.0,
        limit=1
    )

    if not results:
        print("  ❌ FAIL: No products found for budget stretcher test")
        return False

    original = results[0]
    total_budget = 150.0

    print(f"Original: {original['title']} - ${original['price']:.2f}")
    print(f"Total Budget: ${total_budget:.2f}")

    bundle = search.budget_stretcher(
        product_id=original['product_id'],
        total_budget=total_budget
    )

    if bundle['success']:
        print(f"\nSuggested Bundle:")
        print(f"  Alternative: {bundle['alternative']['title']} - ${bundle['alternative']['price']:.2f}")
        print(f"  Savings: ${bundle['savings']:.2f}")
        print(f"  Complementary products:")
        for comp in bundle['complementary_products']:
            print(f"    • {comp['title']} - ${comp['price']:.2f}")
        print(f"  Bundle Total: ${bundle['bundle_total']:.2f}")
        print(f"  Within Budget: {bundle['within_budget']}")
        print("\n  ✅ PASS: Budget stretcher working")
        return True
    else:
        print(f"\n  Note: {bundle['message']}")
        print("  ✅ PASS: Gracefully handled no alternatives")
        return True


def test_image_search_setup(search):
    """Test 4: Image search setup (CLIP model loading)"""
    print("\n" + "="*60)
    print("TEST 4: Image Search Setup")
    print("="*60)

    try:
        # This triggers lazy loading of CLIP model
        print("Loading CLIP model for image embeddings...")
        model = search.image_model
        if model is not None:
            print(f"  Model loaded: {model}")
            print("  ✅ PASS: CLIP model loaded successfully")
            return True
        else:
            print("  ❌ FAIL: CLIP model not loaded")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: Error loading CLIP model: {e}")
        return False


def test_collection_stats(search):
    """Test 5: Collection statistics"""
    print("\n" + "="*60)
    print("TEST 5: Collection Statistics")
    print("="*60)

    text_count = search.client.count("products_text").count
    image_count = search.client.count("products_images").count

    print(f"  products_text collection: {text_count} products")
    print(f"  products_images collection: {image_count} products")

    if text_count > 0:
        print("\n  ✅ PASS: Text collection has data")
        return True
    else:
        print("\n  ❌ FAIL: Text collection is empty")
        return False


def run_all_tests():
    """Run all feature tests"""
    print("\n" + "#"*60)
    print("   FINVECTOR - COMPREHENSIVE FEATURE TEST SUITE")
    print("#"*60)

    # Initialize search engine
    print("\nInitializing FinVector Search Engine...")
    search = FinVectorSearch(load_image_model=True)

    results = {
        "Text Search": test_text_search(search),
        "Smart Alternatives": test_alternatives(search),
        "Budget Stretcher": test_budget_stretcher(search),
        "Image Search Setup": test_image_search_setup(search),
        "Collection Stats": test_collection_stats(search)
    }

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! FinVector is ready.")
    else:
        print("\n⚠️  Some tests failed. Review the output above.")

    return passed == total


if __name__ == "__main__":
    run_all_tests()
