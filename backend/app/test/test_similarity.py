"""Test similarity thresholds"""
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Get Xbox product with its vector
print("Getting Xbox product...")
product = client.retrieve(
    collection_name="products_text",
    ids=[21368],
    with_vectors=True
)[0]

print(f"Product: {product.payload['title']}")
print(f"Category: {product.payload['category']}")
print(f"Price: ${product.payload['price']:.2f}")

# Test different similarity thresholds
for threshold in [0.9, 0.8, 0.7, 0.6, 0.5]:
    print(f"\n--- Threshold: {threshold} ---")

    # Search with same category filter
    response = client.query_points(
        collection_name="products_text",
        query=product.vector,
        query_filter=Filter(must=[
            FieldCondition(key="price", range=Range(lte=200)),
            FieldCondition(key="category", match=MatchValue(value="Xbox 360 Games, Consoles & Accessories"))
        ]),
        limit=5,
        with_payload=True,
        score_threshold=threshold
    )

    results = [r for r in response.points if r.id != 21368]
    print(f"Found {len(results)} alternatives (same category, price <= $200)")

    for r in results[:3]:
        print(f"  [{r.score:.3f}] ${r.payload['price']:.2f} - {r.payload['title'][:50]}...")

# Now test WITHOUT category filter
print(f"\n{'='*60}")
print("Testing WITHOUT category filter (threshold 0.7):")
print(f"{'='*60}")

response = client.query_points(
    collection_name="products_text",
    query=product.vector,
    query_filter=Filter(must=[
        FieldCondition(key="price", range=Range(lte=200))
    ]),
    limit=10,
    with_payload=True,
    score_threshold=0.7
)

results = [r for r in response.points if r.id != 21368]
print(f"Found {len(results)} products")
for r in results[:5]:
    print(f"  [{r.score:.3f}] {r.payload['category']} - ${r.payload['price']:.2f} - {r.payload['title'][:40]}...")
