"""Quick diagnostic to check Qdrant data quality"""
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Get sample products
print("Fetching sample products...")
response = client.scroll(
    collection_name="products_text",
    limit=20,
    with_payload=True,
    with_vectors=False
)

products = response[0]

print(f"\n{'='*60}")
print("Sample Products in Qdrant:")
print(f"{'='*60}\n")

for p in products[:10]:
    print(f"ID: {p.id}")
    print(f"  Title: {p.payload.get('title', 'N/A')[:60]}...")
    print(f"  Category: {p.payload.get('category', 'N/A')}")
    print(f"  Price: ${p.payload.get('price', 'N/A')}")
    print()

# Get unique categories
print(f"\n{'='*60}")
print("Checking categories...")
print(f"{'='*60}\n")

categories = {}
offset = None
while True:
    response = client.scroll(
        collection_name="products_text",
        limit=1000,
        offset=offset,
        with_payload=["category"]
    )
    points, offset = response

    if not points:
        break

    for p in points:
        cat = p.payload.get('category', 'MISSING')
        categories[cat] = categories.get(cat, 0) + 1

    if offset is None:
        break

print(f"Total products: {sum(categories.values())}")
print(f"Unique categories: {len(categories)}")
print(f"\nTop 15 categories:")
for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:15]:
    print(f"  {count:5d} - {cat}")
