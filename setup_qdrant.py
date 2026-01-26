from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Create text embeddings collection
if client.collection_exists("products_text"):
    client.delete_collection("products_text")

client.create_collection(
    collection_name="products_text",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)  # all-mpnet-base-v2
)

# Create payload indexes for filtering
client.create_payload_index(
    collection_name="products_text",
    field_name="price",
    field_schema=PayloadSchemaType.FLOAT
)

client.create_payload_index(
    collection_name="products_text",
    field_name="rating",
    field_schema=PayloadSchemaType.FLOAT
)

client.create_payload_index(
    collection_name="products_text",
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)

# Create image embeddings collection
if client.collection_exists("products_images"):
    client.delete_collection("products_images")

client.create_collection(
    collection_name="products_images",
    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
)

# Create payload indexes for image collection filtering
client.create_payload_index(
    collection_name="products_images",
    field_name="price",
    field_schema=PayloadSchemaType.FLOAT
)

client.create_payload_index(
    collection_name="products_images",
    field_name="rating",
    field_schema=PayloadSchemaType.FLOAT
)

client.create_payload_index(
    collection_name="products_images",
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)

print("✅ Collections created successfully!")
print("✅ Payload indexes created for price, rating, and category!")
