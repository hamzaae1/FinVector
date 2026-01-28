"""Test the alternatives feature"""
from app.finvector_core import FinVectorSearch

search = FinVectorSearch(load_image_model=False)

# Find an Xbox product first
print("Searching for Xbox products...")
results = search.search_products("xbox game", max_budget=500, limit=3)

if not results:
    print("No Xbox products found!")
else:
    print(f"\nFound {len(results)} Xbox products:\n")
    for r in results:
        print(f"ID: {r['product_id']}")
        print(f"  Title: {r['title'][:60]}...")
        print(f"  Category: {r['category']}")
        print(f"  Price: ${r['price']}")
        print(f"  Similarity: {r['similarity_score']}")
        print()

    # Now test alternatives for the first product
    first = results[0]
    print(f"\n{'='*60}")
    print(f"Testing alternatives for: {first['title'][:50]}...")
    print(f"Product ID: {first['product_id']}, Category: {first['category']}")
    print(f"{'='*60}\n")

    alternatives = search.find_alternatives(
        product_id=first['product_id'],
        max_budget=first['price'] * 0.9 if first['price'] else 50,
        limit=5
    )

    if not alternatives:
        print("No alternatives found!")
    else:
        print(f"Found {len(alternatives)} alternatives:\n")
        for a in alternatives:
            print(f"ID: {a['product_id']}")
            print(f"  Title: {a['title'][:60]}...")
            print(f"  Category: {a.get('category', 'N/A')}")
            print(f"  Price: ${a['price']}")
            print(f"  Similarity: {a['similarity_score']}")
            print()
