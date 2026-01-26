"""
FinVector - Amazon 50K Dataset Ingestion Script
Ingests the Kaggle Amazon Products 50K dataset into Qdrant
"""
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, PayloadSchemaType
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
import glob
import sys
import torch

load_dotenv()

# Check for GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

def find_dataset():
    """Find the Amazon dataset CSV file"""
    # Look for common patterns
    patterns = [
        "amazon*.csv",
        "Amazon*.csv",
        "*50k*.csv",
        "*products*.csv"
    ]

    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            # Exclude our original small products.csv
            matches = [m for m in matches if m != "products.csv"]
            if matches:
                return matches[0]

    return None


def analyze_dataset(df):
    """Analyze dataset structure and map columns"""
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head(2).to_string())

    # Try to map columns to our schema
    column_mapping = {}

    columns_lower = {col.lower().replace(' ', '_'): col for col in df.columns}

    # Direct exact matches first (case insensitive)
    def find_exact_column(names):
        for name in names:
            if name.lower() in columns_lower:
                return columns_lower[name.lower()]
        return None

    def find_partial_column(patterns):
        for pattern in patterns:
            for col_lower, col_orig in columns_lower.items():
                if pattern in col_lower and 'unnamed' not in col_lower:
                    return col_orig
        return None

    # Title - exact match first
    column_mapping['title'] = find_exact_column(['title', 'name', 'product_name', 'productname'])
    if not column_mapping['title']:
        column_mapping['title'] = find_partial_column(['title', 'name', 'product'])

    # Description
    column_mapping['description'] = find_exact_column(['description', 'desc', 'about', 'details'])
    if not column_mapping['description']:
        column_mapping['description'] = find_partial_column(['description', 'desc', 'about'])

    # Price - exact match
    column_mapping['price'] = find_exact_column(['price', 'actual_price', 'selling_price', 'discounted_price'])
    if not column_mapping['price']:
        column_mapping['price'] = find_partial_column(['price', 'cost'])

    # Rating
    column_mapping['rating'] = find_exact_column(['stars', 'rating', 'ratings', 'avg_rating'])
    if not column_mapping['rating']:
        column_mapping['rating'] = find_partial_column(['star', 'rating'])

    # Category
    column_mapping['category'] = find_exact_column(['category_name', 'category', 'main_category'])
    if not column_mapping['category']:
        column_mapping['category'] = find_partial_column(['category', 'categories'])

    # Image URL
    column_mapping['image_url'] = find_exact_column(['imgUrl', 'image_url', 'img_link', 'image'])
    if not column_mapping['image_url']:
        column_mapping['image_url'] = find_partial_column(['img', 'image'])

    # Review count (for popularity signal)
    column_mapping['review_count'] = find_exact_column([
        'reviews', 'review_count', 'rating_count', 'no_of_ratings',
        'num_reviews', 'total_reviews', 'reviewcount', 'boughtInLastMonth'
    ])
    if not column_mapping['review_count']:
        column_mapping['review_count'] = find_partial_column(['review', 'rating_count', 'bought'])

    print(f"\nColumn mapping detected:")
    for key, value in column_mapping.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")

    return column_mapping


def clean_price(price_str):
    """Clean price string and convert to float"""
    if pd.isna(price_str):
        return None

    price_str = str(price_str)

    # Remove currency symbols and commas
    price_str = price_str.replace('$', '').replace('₹', '').replace(',', '').replace(' ', '')

    # Handle ranges like "100-200" by taking the first value
    if '-' in price_str:
        price_str = price_str.split('-')[0]

    try:
        price = float(price_str)
        # Convert INR to USD if price seems to be in INR (rough heuristic)
        if price > 10000:  # Likely INR
            price = price / 83  # Approximate INR to USD
        return round(price, 2)
    except:
        return None


def clean_rating(rating_str):
    """Clean rating and convert to float"""
    if pd.isna(rating_str):
        return 4.0  # Default rating

    rating_str = str(rating_str)

    # Extract numeric value
    import re
    match = re.search(r'(\d+\.?\d*)', rating_str)
    if match:
        rating = float(match.group(1))
        # Normalize to 5-star scale if needed
        if rating > 5:
            rating = rating / 2  # Assume 10-point scale
        return min(5.0, max(0.0, rating))

    return 4.0  # Default


def upsert_with_retry(client, collection_name, points, max_retries=3):
    """Upsert with retry logic"""
    import time
    for attempt in range(max_retries):
        try:
            client.upsert(collection_name=collection_name, points=points)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"    Retry {attempt + 1}/{max_retries} after {wait_time}s: {str(e)[:50]}...")
                time.sleep(wait_time)
            else:
                print(f"    Failed after {max_retries} retries: {str(e)[:100]}")
                return False
    return False


def ingest_data(csv_file, max_products=50000, batch_size=50):
    """Ingest Amazon dataset into Qdrant"""

    print(f"\n{'='*60}")
    print(f"  FINVECTOR - Amazon 50K Dataset Ingestion")
    print(f"{'='*60}")

    # Load dataset
    print(f"\nLoading dataset: {csv_file}")
    df = pd.read_csv(csv_file, nrows=max_products)

    # Analyze and map columns
    column_mapping = analyze_dataset(df)

    # Check required columns
    if not column_mapping['title']:
        print("\n❌ ERROR: Could not find product name/title column!")
        print("Please check the dataset structure.")
        return False

    # Initialize Qdrant with longer timeout
    print("\nConnecting to Qdrant...")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=120  # 2 minute timeout
    )

    # Recreate collection
    print("Recreating products_text collection...")
    if client.collection_exists("products_text"):
        client.delete_collection("products_text")

    client.create_collection(
        collection_name="products_text",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)  # all-mpnet-base-v2
    )

    # Create indexes
    client.create_payload_index("products_text", "price", PayloadSchemaType.FLOAT)
    client.create_payload_index("products_text", "rating", PayloadSchemaType.FLOAT)
    client.create_payload_index("products_text", "review_count", PayloadSchemaType.INTEGER)
    client.create_payload_index("products_text", "category", PayloadSchemaType.KEYWORD)

    # Initialize model on GPU
    print(f"Loading embedding model on {DEVICE}...")
    model = SentenceTransformer('all-mpnet-base-v2', device=DEVICE)

    # First pass: collect all data
    print(f"\nPreparing {len(df)} products...")
    products_data = []
    texts_to_encode = []

    from tqdm import tqdm

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing"):
        # Get title
        title = str(row[column_mapping['title']]) if column_mapping['title'] else ""
        if not title or title == 'nan':
            continue

        # Get description
        description = ""
        if column_mapping['description']:
            desc = row[column_mapping['description']]
            if pd.notna(desc):
                description = str(desc)

        # Get price - preserve None for unknown prices (don't fake it)
        price = None
        price_available = False
        if column_mapping['price']:
            price = clean_price(row[column_mapping['price']])
        if price is not None and price > 0:
            price_available = True
        else:
            price = None  # Keep as None, not random - prevents poisoned relevance

        # Get rating
        rating = 4.0
        if column_mapping['rating']:
            rating = clean_rating(row[column_mapping['rating']])

        # Get category
        category = "General"
        if column_mapping['category']:
            cat = row[column_mapping['category']]
            if pd.notna(cat):
                category = str(cat).split('|')[0].strip()

        # Get image URL
        image_url = ""
        if column_mapping['image_url']:
            img = row[column_mapping['image_url']]
            if pd.notna(img):
                image_url = str(img)

        # Get review count (for popularity signal)
        review_count = 0
        if column_mapping['review_count']:
            rc = row[column_mapping['review_count']]
            if pd.notna(rc):
                try:
                    # Handle formats like "1,234" or "1234" or "1.2K"
                    rc_str = str(rc).replace(',', '').strip()
                    if 'K' in rc_str.upper():
                        review_count = int(float(rc_str.upper().replace('K', '')) * 1000)
                    elif 'M' in rc_str.upper():
                        review_count = int(float(rc_str.upper().replace('M', '')) * 1000000)
                    else:
                        review_count = int(float(rc_str))
                except:
                    review_count = 0

        # Store data
        text = f"{title} {description}"[:500]
        texts_to_encode.append(text)
        products_data.append({
            "idx": idx,
            "title": title[:200],
            "description": description[:500],
            "price": float(price) if price is not None else None,
            "price_available": price_available,
            "rating": float(rating),
            "review_count": review_count,
            "category": category[:100],
            "image_url": image_url
        })

    print(f"\nEncoding {len(texts_to_encode)} products on {DEVICE} (batch size: 64)...")

    # Batch encode ALL texts at once on GPU - much faster!
    all_embeddings = model.encode(
        texts_to_encode,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"\nUploading to Qdrant...")
    points = []
    processed = 0

    for i, (embedding, data) in enumerate(tqdm(zip(all_embeddings, products_data), total=len(products_data), desc="Uploading")):
        point = PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                "product_id": str(data["idx"]),
                "title": data["title"],
                "description": data["description"],
                "price": data["price"],
                "rating": data["rating"],
                "review_count": data["review_count"],
                "category": data["category"],
                "image_url": data["image_url"]
            }
        )
        points.append(point)
        processed += 1

        # Upload in batches
        if len(points) >= batch_size:
            upsert_with_retry(client, "products_text", points)
            points = []

    # Upload remaining
    if points:
        upsert_with_retry(client, "products_text", points)

    skipped = len(df) - len(products_data)

    print(f"\n{'='*60}")
    print(f"  INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total processed: {processed}")
    print(f"  Skipped (invalid): {skipped}")
    print(f"  Products in Qdrant: {client.count('products_text').count}")
    print(f"{'='*60}\n")

    return True


if __name__ == "__main__":
    # Find or specify dataset
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = find_dataset()
        if not csv_file:
            print("❌ No Amazon dataset found!")
            print("\nPlease either:")
            print("  1. Place the CSV file in the current directory")
            print("  2. Run: python ingest_amazon.py <path_to_csv>")
            print("\nLooking for files matching: amazon*.csv, *50k*.csv, *products*.csv")
            sys.exit(1)

    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)

    success = ingest_data(csv_file)
    sys.exit(0 if success else 1)
