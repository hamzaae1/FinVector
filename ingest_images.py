"""
FinVector - Image Indexing Script
Indexes product images into Qdrant using CLIP embeddings for visual search
Optimized with batch encoding and parallel downloads for GPU acceleration
"""
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, PayloadSchemaType
from sentence_transformers import SentenceTransformer
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
import os
import sys
import torch

load_dotenv()

# Check for GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")


def download_image(product):
    """Download a single image - used for parallel downloads"""
    image_url = product['payload']['image_url']
    try:
        response = requests.get(image_url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return {'product': product, 'image': img, 'success': True}
    except Exception as e:
        return {'product': product, 'image': None, 'success': False, 'error': str(e)}


def index_images(max_products=999999, batch_size=32, download_workers=8):
    """
    Index product images from products_text into products_images collection

    Args:
        max_products: Maximum number of products to index
        batch_size: Number of images to encode at once on GPU (higher = faster but more VRAM)
        download_workers: Number of parallel download threads
    """

    print(f"\n{'='*60}")
    print(f"  FINVECTOR - Image Indexing with CLIP (GPU Batch Mode)")
    print(f"{'='*60}")
    print(f"  Batch size: {batch_size} | Download workers: {download_workers}")

    # Initialize Qdrant
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=120
    )

    # Initialize CLIP model on GPU
    print(f"\nLoading CLIP model (clip-ViT-B-32) on {DEVICE}...")
    clip_model = SentenceTransformer('clip-ViT-B-32', device=DEVICE)
    print(f"CLIP model loaded on {DEVICE}!")

    # Recreate products_images collection
    print("\nRecreating products_images collection...")
    if client.collection_exists("products_images"):
        client.delete_collection("products_images")

    client.create_collection(
        collection_name="products_images",
        vectors_config=VectorParams(size=512, distance=Distance.COSINE)
    )

    # Create indexes
    client.create_payload_index("products_images", "price", PayloadSchemaType.FLOAT)
    client.create_payload_index("products_images", "rating", PayloadSchemaType.FLOAT)
    client.create_payload_index("products_images", "category", PayloadSchemaType.KEYWORD)

    # Fetch products from products_text
    print(f"\nFetching products with images from products_text...")

    # Get total count first
    total_in_collection = client.count("products_text").count
    print(f"Total products in collection: {total_in_collection}")

    products_to_index = []
    offset = None

    with tqdm(total=total_in_collection, desc="Fetching products", unit="products") as pbar:
        while True:
            result = client.scroll(
                collection_name="products_text",
                limit=100,
                offset=offset,
                with_payload=True
            )

            points, next_offset = result

            # Break if no more points
            if not points:
                break

            for point in points:
                image_url = point.payload.get('image_url', '')
                if image_url and image_url.startswith('http'):
                    products_to_index.append({
                        'id': point.id,
                        'payload': point.payload
                    })
                pbar.update(1)

            # Break if no next page
            if next_offset is None:
                break

            offset = next_offset

    print(f"Found {len(products_to_index)} products with image URLs")

    # Process in batches
    indexed = 0
    failed = 0
    all_points = []

    print(f"\nIndexing images with batch encoding...")

    # Main progress bar for overall progress
    with tqdm(total=len(products_to_index), desc="Indexing", unit="img") as pbar:

        # Process in batches
        for batch_start in range(0, len(products_to_index), batch_size):
            batch_products = products_to_index[batch_start:batch_start + batch_size]

            # Parallel download images for this batch
            batch_images = []
            batch_metadata = []

            with ThreadPoolExecutor(max_workers=download_workers) as executor:
                futures = {executor.submit(download_image, p): p for p in batch_products}

                for future in as_completed(futures):
                    result = future.result()
                    if result['success']:
                        batch_images.append(result['image'])
                        batch_metadata.append(result['product'])
                    else:
                        failed += 1
                    pbar.update(1)

            if not batch_images:
                continue

            # Batch encode all images at once on GPU
            embeddings = clip_model.encode(batch_images, batch_size=len(batch_images), show_progress_bar=False)

            # Create points
            for i, (embedding, product) in enumerate(zip(embeddings, batch_metadata)):
                point = PointStruct(
                    id=indexed,
                    vector=embedding.tolist(),
                    payload={
                        "product_id": str(product['id']),
                        "title": product['payload']['title'],
                        "description": product['payload'].get('description', ''),
                        "price": float(product['payload']['price']),
                        "rating": float(product['payload']['rating']),
                        "category": product['payload']['category'],
                        "image_url": product['payload']['image_url']
                    }
                )
                all_points.append(point)
                indexed += 1

            # Upload to Qdrant in batches of 100
            if len(all_points) >= 100:
                client.upsert(collection_name="products_images", points=all_points)
                all_points = []

    # Upload remaining points
    if all_points:
        client.upsert(collection_name="products_images", points=all_points)

    print(f"\n{'='*60}")
    print(f"  IMAGE INDEXING COMPLETE")
    print(f"{'='*60}")
    print(f"  Successfully indexed: {indexed}")
    print(f"  Failed: {failed}")
    print(f"  Products in collection: {client.count('products_images').count}")
    print(f"{'='*60}\n")

    return indexed


if __name__ == "__main__":
    max_products = int(sys.argv[1]) if len(sys.argv) > 1 else 999999  # Index all by default
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    index_images(max_products=max_products, batch_size=batch_size)
