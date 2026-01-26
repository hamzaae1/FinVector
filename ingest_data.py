from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

# Initialize
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
model = SentenceTransformer('all-mpnet-base-v2')

# Load products
df = pd.read_csv('products.csv')

# Process in batches
batch_size = 100
points = []

for idx, row in df.iterrows():
    # Create combined text for embedding
    text = f"{row['title']} {row['description']}"
    
    # Generate embedding
    embedding = model.encode(text).tolist()
    
    # Create point
    point = PointStruct(
        id=idx,
        vector=embedding,
        payload={
            "product_id": str(row['product_id']),
            "title": row['title'],
            "description": row['description'],
            "price": float(row['price']),
            "rating": float(row['rating']),
            "category": row['category'],
            "image_url": row.get('image_url', '')
        }
    )
    points.append(point)
    
    # Upload in batches
    if len(points) >= batch_size:
        client.upsert(
            collection_name="products_text",
            points=points
        )
        print(f"✅ Uploaded {len(points)} products")
        points = []

# Upload remaining
if points:
    client.upsert(collection_name="products_text", points=points)
    print(f"✅ Uploaded final {len(points)} products")

print(f"🎉 Total products in database: {client.count('products_text').count}")
