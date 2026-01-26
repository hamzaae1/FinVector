from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, PointStruct, MatchValue
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Union
from PIL import Image
from functools import lru_cache
from collections import OrderedDict
import requests
from io import BytesIO
import os
import time
import torch
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Check for GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Import logger (optional - won't fail if not available)
try:
    from logger import SearchLogger, timed_operation
    LOGGING_ENABLED = True
except ImportError:
    LOGGING_ENABLED = False
    def timed_operation(name):
        def decorator(func):
            return func
        return decorator

# Import reranker (optional - won't fail if not available)
try:
    from reranker import get_reranker, RerankConfig
    RERANKING_ENABLED = True
except ImportError:
    RERANKING_ENABLED = False
    get_reranker = None
    RerankConfig = None

class QueryEmbeddingCache:
    """
    LRU cache for query embeddings.

    Caches computed embeddings to avoid redundant encoding.
    Same query with different budget = same embedding (reusable).
    """

    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, query: str) -> Optional[np.ndarray]:
        """Get cached embedding or None"""
        key = query.lower().strip()
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, query: str, embedding: np.ndarray):
        """Cache an embedding"""
        key = query.lower().strip()

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = embedding

    def stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_rate": round(hit_rate, 4)
        }


class FinVectorSearch:
    def __init__(self, load_image_model: bool = True):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.device = DEVICE
        self.text_model = SentenceTransformer('all-mpnet-base-v2', device=DEVICE)
        self._image_model = None
        self._load_image_model = load_image_model

        # Query embedding cache
        self._embedding_cache = QueryEmbeddingCache(max_size=500)

    def _encode_query(self, query: str) -> List[float]:
        """
        Encode query with caching.

        Checks cache first, computes embedding only if not cached.
        Returns list for Qdrant compatibility.
        """
        # Check cache
        cached = self._embedding_cache.get(query)
        if cached is not None:
            return cached.tolist()

        # Compute embedding
        embedding = self.text_model.encode(query)

        # Cache it
        self._embedding_cache.put(query, embedding)

        return embedding.tolist()

    def get_embedding_cache_stats(self) -> Dict:
        """Get embedding cache statistics"""
        return self._embedding_cache.stats()

    @property
    def image_model(self):
        """Lazy load the CLIP model for image embeddings on GPU"""
        if self._image_model is None and self._load_image_model:
            self._image_model = SentenceTransformer('clip-ViT-B-32', device=DEVICE)
        return self._image_model
        
    def smart_search(
        self,
        query: str,
        max_budget: Optional[float] = None,
        min_rating: Optional[float] = None,
        limit: int = 10
    ) -> Dict:
        """
        Intelligent search that understands natural language queries.

        Features:
        - Extracts budget from text: "shoes under $50" → max_budget=50
        - Detects intent: "cheap but good" → adjusts rating filter
        - Expands queries: "wireless headphones" → adds "bluetooth earbuds"
        - Understands context: "laptop for gaming" → adds gaming terms

        Args:
            query: Natural language query (e.g., "I need running shoes under $80")
            max_budget: Override budget (uses extracted if None)
            min_rating: Override rating (uses intent-based if None)
            limit: Number of results

        Returns:
            Dict with results and query understanding metadata
        """
        from query_understanding import understand_query

        # Parse the query
        parsed = understand_query(query, default_budget=max_budget or 200.0)

        # Use extracted/detected values unless overridden
        effective_budget = max_budget if max_budget is not None else (parsed.extracted_budget or 200.0)
        effective_rating = min_rating if min_rating is not None else parsed.min_rating

        # Search with enhanced query
        results = self.search_products(
            query=parsed.enhanced_query,
            max_budget=effective_budget,
            min_rating=effective_rating,
            limit=limit
        )

        # Return results with understanding metadata
        return {
            "results": results,
            "understanding": {
                "original_query": parsed.original_query,
                "enhanced_query": parsed.enhanced_query,
                "extracted_budget": parsed.extracted_budget,
                "effective_budget": effective_budget,
                "detected_intent": parsed.intent,
                "min_rating_applied": effective_rating,
                "category_hints": parsed.categories_hint,
                "attributes": parsed.attributes,
                "confidence": parsed.confidence
            }
        }

    def search_products(
        self,
        query: str,
        max_budget: float,
        min_rating: float = 0.0,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search for products within budget

        Args:
            query: User search query
            max_budget: Maximum price constraint
            min_rating: Minimum product rating
            limit: Number of results to return
        """
        start_time = time.time()

        # Generate query embedding (with caching)
        query_vector = self._encode_query(query)

        # Build filter conditions
        filter_conditions = [
            FieldCondition(key="price", range=Range(lte=max_budget))
        ]

        if min_rating > 0:
            filter_conditions.append(
                FieldCondition(key="rating", range=Range(gte=min_rating))
            )

        # Search with budget filter
        response = self.client.query_points(
            collection_name="products_text",
            query=query_vector,
            query_filter=Filter(must=filter_conditions),
            limit=limit,
            with_payload=True
        )
        results = response.points
        formatted_results = self._format_results(results, query)

        # Log search
        if LOGGING_ENABLED:
            duration_ms = (time.time() - start_time) * 1000
            SearchLogger.log_search(
                query=query,
                max_budget=max_budget,
                results_count=len(formatted_results),
                response_time_ms=duration_ms,
                search_type="text",
                filters={"min_rating": min_rating, "limit": limit}
            )

        return formatted_results
    
    def find_alternatives(
        self,
        product_id: Union[str, int],
        max_budget: float,
        min_rating: float = 4.0,
        limit: int = 5,
        same_category: bool = True,
        min_similarity: float = 0.70,
        sort_by: str = "similarity"
    ) -> List[Dict]:
        """
        Find cheaper alternatives to a product.

        Args:
            product_id: ID of the original product
            max_budget: Maximum price for alternatives
            min_rating: Minimum rating filter
            limit: Number of alternatives to return
            same_category: Require alternatives to be in same category
            min_similarity: Minimum similarity score (0-1) to be considered an alternative
            sort_by: How to sort results - "similarity", "price_low", "price_high", "rating", "value"
        """

        # Convert product_id to int if it's a string
        point_id = int(product_id) if isinstance(product_id, str) else product_id

        # Get the original product
        original = self.client.retrieve(
            collection_name="products_text",
            ids=[point_id],
            with_vectors=True
        )[0]

        original_category = original.payload.get('category', '')
        original_price = original.payload.get('price') or 0

        # Build filter conditions
        filter_conditions = [
            FieldCondition(key="price", range=Range(lte=max_budget)),
            FieldCondition(key="rating", range=Range(gte=min_rating))
        ]

        # Add category filter if we have a valid category
        if same_category and original_category and original_category != 'General':
            filter_conditions.append(
                FieldCondition(key="category", match=MatchValue(value=original_category))
            )

        # Search for similar but cheaper products in same category
        # Fetch extra results for sorting flexibility
        response = self.client.query_points(
            collection_name="products_text",
            query=original.vector,
            query_filter=Filter(must=filter_conditions),
            limit=max(limit * 3, 20),  # Get more for sorting
            with_payload=True,
            score_threshold=min_similarity
        )
        results = response.points

        # Remove the original product from results
        alternatives = [r for r in results if r.id != point_id]

        # Sort based on user preference
        if sort_by == "price_low":
            alternatives.sort(key=lambda x: x.payload.get('price') or float('inf'))
        elif sort_by == "price_high":
            alternatives.sort(key=lambda x: x.payload.get('price') or 0, reverse=True)
        elif sort_by == "rating":
            alternatives.sort(key=lambda x: x.payload.get('rating', 0), reverse=True)
        elif sort_by == "value":
            # Value score: balance similarity, savings, and rating
            def value_score(item):
                similarity = item.score
                price = item.payload.get('price') or original_price
                rating = item.payload.get('rating', 0) / 5.0
                # Savings ratio (0 to 1, higher = more savings)
                savings_ratio = (original_price - price) / original_price if original_price > 0 else 0
                savings_ratio = max(0, min(1, savings_ratio))
                # Combined: 40% similarity + 30% savings + 30% rating
                return (0.4 * similarity) + (0.3 * savings_ratio) + (0.3 * rating)
            alternatives.sort(key=value_score, reverse=True)
        # else: "similarity" - already sorted by Qdrant

        return self._format_alternatives(alternatives[:limit], original.payload)

    def budget_stretcher(
        self,
        product_id: Union[str, int],
        total_budget: float,
        sort_by: str = "value"
    ) -> Dict:
        """
        Suggest cheaper alternative + complementary products.

        Args:
            product_id: ID of the product to find alternatives for
            total_budget: User's total budget
            sort_by: How to sort alternatives - "similarity", "price_low", "price_high", "rating", "value"
        """

        # Convert product_id to int if it's a string
        point_id = int(product_id) if isinstance(product_id, str) else product_id

        # Get original product
        original = self.client.retrieve(
            collection_name="products_text",
            ids=[point_id]
        )[0]

        original_price = original.payload.get('price')
        original_category = original.payload.get('category', '')

        # Can't do budget stretching without a known price
        if original_price is None:
            return {"success": False, "message": "Cannot stretch budget - original product price unknown"}

        # Find a cheaper alternative with same category
        has_category = original_category and original_category != 'General'
        alternatives = self.find_alternatives(
            product_id=product_id,
            max_budget=original_price * 0.85,  # 15% cheaper
            limit=1,
            same_category=has_category,  # Only filter by category if we have one
            min_similarity=0.70,
            sort_by=sort_by
        )

        if not alternatives:
            return {"success": False, "message": "No cheaper alternatives found"}

        alternative = alternatives[0]
        alt_price = alternative.get('price')

        # Alternative must have a known price
        if alt_price is None:
            return {"success": False, "message": "Alternative product price unknown"}

        savings = original_price - alt_price

        # Find complementary products with the savings
        category = original.payload['category']
        complementary_query = f"accessories for {category}"

        complementary = self.search_products(
            query=complementary_query,
            max_budget=savings,
            limit=3
        )

        # Filter complementary products with known prices for bundle calculation
        complementary_with_price = [c for c in complementary if c.get('price') is not None]
        bundle_total = alt_price + sum(c['price'] for c in complementary_with_price)

        return {
            "success": True,
            "original": {
                "product_id": point_id,
                "title": original.payload['title'],
                "price": original_price,
                "category": original_category
            },
            "alternative": alternative,
            "savings": round(savings, 2),
            "complementary_products": complementary,
            "bundle_total": round(bundle_total, 2),
            "within_budget": bundle_total <= total_budget
        }

    def search_by_image(
        self,
        image: Union[str, Image.Image],
        max_budget: float,
        min_rating: float = 0.0,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search for similar products using an image

        Args:
            image: Either a file path, URL, or PIL Image object
            max_budget: Maximum price constraint
            min_rating: Minimum product rating
            limit: Number of results to return
        """
        # Load and encode the image
        if isinstance(image, str):
            if image.startswith(('http://', 'https://')):
                response = requests.get(image)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image)
        else:
            img = image

        # Generate image embedding using CLIP
        image_vector = self.image_model.encode(img).tolist()

        # Build filter conditions
        filter_conditions = [
            FieldCondition(key="price", range=Range(lte=max_budget))
        ]

        if min_rating > 0:
            filter_conditions.append(
                FieldCondition(key="rating", range=Range(gte=min_rating))
            )

        # Search in the images collection
        response = self.client.query_points(
            collection_name="products_images",
            query=image_vector,
            query_filter=Filter(must=filter_conditions),
            limit=limit,
            with_payload=True
        )
        results = response.points

        return self._format_image_results(results)

    def index_product_images(self, products_df) -> int:
        """
        Index product images into the products_images collection

        Args:
            products_df: DataFrame with product_id, title, description, price, rating, category, image_url

        Returns:
            Number of products indexed
        """
        points = []
        indexed_count = 0

        for idx, row in products_df.iterrows():
            image_url = row.get('image_url', '')
            if not image_url or not image_url.startswith(('http://', 'https://')):
                continue

            try:
                # Download and encode image
                response = requests.get(image_url, timeout=10)
                img = Image.open(BytesIO(response.content))
                image_vector = self.image_model.encode(img).tolist()

                point = PointStruct(
                    id=idx,
                    vector=image_vector,
                    payload={
                        "product_id": str(row['product_id']),
                        "title": row['title'],
                        "description": row.get('description', ''),
                        "price": float(row['price']),
                        "rating": float(row['rating']),
                        "category": row['category'],
                        "image_url": image_url
                    }
                )
                points.append(point)
                indexed_count += 1

                # Upload in batches
                if len(points) >= 50:
                    self.client.upsert(
                        collection_name="products_images",
                        points=points
                    )
                    points = []

            except Exception as e:
                print(f"Failed to index image for product {row['product_id']}: {e}")
                continue

        # Upload remaining points
        if points:
            self.client.upsert(
                collection_name="products_images",
                points=points
            )

        return indexed_count

    def search_by_text_for_images(
        self,
        query: str,
        max_budget: float,
        min_rating: float = 0.0,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search for products using text query against image embeddings (CLIP text-to-image)

        Args:
            query: Text description of what you're looking for
            max_budget: Maximum price constraint
            min_rating: Minimum product rating
            limit: Number of results to return
        """
        # CLIP can encode both images and text into the same embedding space
        text_vector = self.image_model.encode(query).tolist()

        # Build filter conditions
        filter_conditions = [
            FieldCondition(key="price", range=Range(lte=max_budget))
        ]

        if min_rating > 0:
            filter_conditions.append(
                FieldCondition(key="rating", range=Range(gte=min_rating))
            )

        # Search in the images collection using text embedding
        response = self.client.query_points(
            collection_name="products_images",
            query=text_vector,
            query_filter=Filter(must=filter_conditions),
            limit=limit,
            with_payload=True
        )
        results = response.points

        return self._format_image_results(results)

    def _format_image_results(self, results) -> List[Dict]:
        """Format image search results"""
        formatted = []
        for hit in results:
            price = hit.payload.get('price')
            price_available = hit.payload.get('price_available', price is not None)
            formatted.append({
                "product_id": hit.id,
                "title": hit.payload["title"],
                "description": hit.payload.get("description", ""),
                "price": price,
                "price_available": price_available,
                "rating": hit.payload["rating"],
                "review_count": hit.payload.get("review_count", 0),
                "category": hit.payload["category"],
                "image_url": hit.payload.get("image_url", ""),
                "similarity_score": round(hit.score, 3),
                "match_quality": self._score_to_label(hit.score)
            })
        return formatted

    def _format_results(self, results, query: str) -> List[Dict]:
        """Format search results with explanations"""
        formatted = []
        for hit in results:
            price = hit.payload.get("price")
            price_available = hit.payload.get("price_available", price is not None)
            formatted.append({
                "product_id": hit.id,
                "title": hit.payload["title"],
                "description": hit.payload.get("description", ""),
                "price": price,
                "price_available": price_available,
                "rating": hit.payload["rating"],
                "review_count": hit.payload.get("review_count", 0),
                "category": hit.payload["category"],
                "image_url": hit.payload.get("image_url", ""),
                "similarity_score": round(hit.score, 3),
                "match_quality": self._score_to_label(hit.score),
                "explanation": self._generate_explanation(hit, query)
            })
        return formatted
    
    def _format_alternatives(self, results, original_payload) -> List[Dict]:
        """Format alternative products with comparisons"""
        formatted = []
        original_price = original_payload.get('price')

        for hit in results:
            alt_price = hit.payload.get('price')
            price_available = hit.payload.get('price_available', alt_price is not None)

            # Calculate savings only if both prices are available
            if original_price is not None and alt_price is not None:
                price_diff = original_price - alt_price
                savings = round(price_diff, 2)
                explanation = f"Save ${price_diff:.2f} - {int(hit.score * 100)}% similar"
            else:
                savings = None
                explanation = f"{int(hit.score * 100)}% similar"

            formatted.append({
                "product_id": hit.id,
                "title": hit.payload["title"],
                "price": alt_price,
                "price_available": price_available,
                "rating": hit.payload["rating"],
                "review_count": hit.payload.get("review_count", 0),
                "image_url": hit.payload.get("image_url", ""),
                "similarity_score": round(hit.score, 3),
                "savings": savings,
                "explanation": explanation
            })
        return formatted

    def _score_to_label(self, score: float) -> str:
        """Convert similarity score to quality label"""
        if score > 0.9:
            return "Excellent match"
        elif score > 0.75:
            return "Good match"
        elif score > 0.6:
            return "Fair match"
        else:
            return "Weak match"
    
    def _generate_explanation(self, hit, query: str) -> str:
        """Generate detailed human-readable explanation"""
        score_pct = int(hit.score * 100)
        price = hit.payload.get('price')
        price_available = hit.payload.get('price_available', price is not None)
        rating = hit.payload['rating']
        category = hit.payload.get('category', '')

        # Build detailed explanation
        reasons = []

        # Relevance
        if score_pct >= 90:
            reasons.append(f"Excellent {score_pct}% match to '{query}'")
        elif score_pct >= 75:
            reasons.append(f"Strong {score_pct}% match to '{query}'")
        elif score_pct >= 60:
            reasons.append(f"Good {score_pct}% match to '{query}'")
        else:
            reasons.append(f"{score_pct}% relevance to '{query}'")

        # Rating
        if rating >= 4.5:
            reasons.append(f"Highly rated: {rating}★ by customers")
        elif rating >= 4.0:
            reasons.append(f"Well rated: {rating}★")
        elif rating >= 3.5:
            reasons.append(f"Rated {rating}★")

        # Price positioning
        if price_available and price is not None:
            reasons.append(f"Priced at ${price:.2f}")
        else:
            reasons.append("Price unavailable")

        # Category context
        if category:
            reasons.append(f"Category: {category}")

        return " • ".join(reasons)

    def hybrid_search(
        self,
        image: Union[str, Image.Image],
        text_query: str,
        max_budget: float,
        min_rating: float = 0.0,
        limit: int = 10,
        image_weight: float = 0.7,
        min_image_similarity: float = 0.75
    ) -> List[Dict]:
        """
        Hybrid search: Find visually similar products, then refine by text.

        Two-stage approach:
        1. Find products visually similar to the image (must meet minimum threshold)
        2. Re-rank by text relevance (e.g., "black", "for running")

        Args:
            image: Reference image (file path, URL, or PIL Image)
            text_query: Text to refine/filter (e.g., "black", "leather", "for running")
            max_budget: Maximum price constraint
            min_rating: Minimum product rating
            limit: Number of results to return
            image_weight: Weight for image vs text (0.7 = 70% image, 30% text)
            min_image_similarity: Minimum visual similarity required (default 0.75)
        """
        # Load image
        if isinstance(image, str):
            if image.startswith(('http://', 'https://')):
                response = requests.get(image)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image)
        else:
            img = image

        # Get embeddings from CLIP
        image_vector = self.image_model.encode(img)
        text_vector = self.image_model.encode(text_query)

        # Normalize vectors for cosine similarity
        image_vector = image_vector / np.linalg.norm(image_vector)
        text_vector = text_vector / np.linalg.norm(text_vector)

        # Build filter conditions
        filter_conditions = [
            FieldCondition(key="price", range=Range(lte=max_budget))
        ]
        if min_rating > 0:
            filter_conditions.append(
                FieldCondition(key="rating", range=Range(gte=min_rating))
            )

        # Stage 1: Get candidates using IMAGE similarity only
        response = self.client.query_points(
            collection_name="products_images",
            query=image_vector.tolist(),
            query_filter=Filter(must=filter_conditions),
            limit=limit * 10,  # Get more candidates
            with_payload=True,
            with_vectors=True,
            score_threshold=min_image_similarity  # ONLY return visually similar items
        )

        candidates = response.points

        if not candidates:
            return []

        # Detect the dominant category from top image matches
        # This ensures we stay within the same product type
        top_category = candidates[0].payload.get('category', '')

        # Stage 2: Filter to same category, then score by text
        scored_results = []
        for hit in candidates:
            # Skip if different category (ensures shoes stay shoes)
            if hit.payload.get('category', '') != top_category:
                continue

            # Get product's image vector
            product_vector = np.array(hit.vector)
            product_vector = product_vector / np.linalg.norm(product_vector)

            # Calculate text similarity
            text_similarity = float(np.dot(product_vector, text_vector))

            # Image similarity from Qdrant
            image_similarity = hit.score

            # Combined score
            combined_score = (image_weight * image_similarity) + ((1 - image_weight) * text_similarity)

            scored_results.append({
                'hit': hit,
                'image_score': image_similarity,
                'text_score': text_similarity,
                'combined_score': combined_score
            })

        # Sort by combined score (text refinement)
        scored_results.sort(key=lambda x: x['combined_score'], reverse=True)

        # Take top results
        top_results = scored_results[:limit]

        return self._format_hybrid_results_v2(top_results, text_query, image_weight)

    def _format_hybrid_results(self, results, text_query: str, image_weight: float) -> List[Dict]:
        """Format hybrid search results with explanations (legacy)"""
        formatted = []
        text_weight = 1 - image_weight

        for hit in results:
            price = hit.payload.get('price')
            price_available = hit.payload.get('price_available', price is not None)
            price_str = f"${price:.2f}" if price is not None else "Price N/A"

            explanation = (
                f"{int(hit.score * 100)}% combined match • "
                f"Visual similarity: {int(image_weight * 100)}% weight • "
                f"Text refinement '{text_query}': {int(text_weight * 100)}% weight • "
                f"{price_str} • {hit.payload['rating']}★"
            )

            formatted.append({
                "product_id": hit.id,
                "title": hit.payload["title"],
                "description": hit.payload.get("description", ""),
                "price": price,
                "price_available": price_available,
                "rating": hit.payload["rating"],
                "review_count": hit.payload.get("review_count", 0),
                "category": hit.payload["category"],
                "image_url": hit.payload.get("image_url", ""),
                "similarity_score": round(hit.score, 3),
                "match_quality": self._score_to_label(hit.score),
                "explanation": explanation,
                "search_type": "hybrid"
            })
        return formatted

    def _format_hybrid_results_v2(self, scored_results: List[Dict], text_query: str, image_weight: float) -> List[Dict]:
        """Format hybrid search results with detailed scoring breakdown"""
        formatted = []
        text_weight = 1 - image_weight

        for item in scored_results:
            hit = item['hit']
            image_score = item['image_score']
            text_score = item['text_score']
            combined_score = item['combined_score']

            price = hit.payload.get('price')
            price_available = hit.payload.get('price_available', price is not None)
            price_str = f"${price:.2f}" if price is not None else "Price N/A"

            # Build detailed explanation
            explanation = (
                f"Combined: {int(combined_score * 100)}% • "
                f"Visual match: {int(image_score * 100)}% • "
                f"'{text_query}' match: {int(text_score * 100)}% • "
                f"{price_str} • {hit.payload['rating']}★"
            )

            formatted.append({
                "product_id": hit.id,
                "title": hit.payload["title"],
                "description": hit.payload.get("description", ""),
                "price": price,
                "price_available": price_available,
                "rating": hit.payload["rating"],
                "review_count": hit.payload.get("review_count", 0),
                "category": hit.payload["category"],
                "image_url": hit.payload.get("image_url", ""),
                "similarity_score": round(combined_score, 3),
                "image_similarity": round(image_score, 3),
                "text_similarity": round(text_score, 3),
                "match_quality": self._score_to_label(combined_score),
                "explanation": explanation,
                "search_type": "hybrid"
            })
        return formatted

    def search_by_category(
        self,
        category: str,
        max_budget: float,
        min_budget: float = 0.0,
        min_rating: float = 0.0,
        limit: int = 10,
        sort_by: str = "rating"  # rating, price_low, price_high
    ) -> List[Dict]:
        """
        Browse products by category with price range filtering

        Args:
            category: Category to filter by
            max_budget: Maximum price
            min_budget: Minimum price
            min_rating: Minimum rating
            limit: Number of results
            sort_by: Sort order - 'rating', 'price_low', 'price_high'
        """
        # Build filter
        filter_conditions = [
            FieldCondition(key="category", match=MatchValue(value=category)),
            FieldCondition(key="price", range=Range(gte=min_budget, lte=max_budget))
        ]

        if min_rating > 0:
            filter_conditions.append(
                FieldCondition(key="rating", range=Range(gte=min_rating))
            )

        # Scroll through filtered results
        response = self.client.scroll(
            collection_name="products_text",
            scroll_filter=Filter(must=filter_conditions),
            limit=limit * 3,  # Get more to sort
            with_payload=True
        )

        points = response[0]

        # Sort results - unknown prices sort to end for price sorts
        if sort_by == "price_low":
            points.sort(key=lambda x: (x.payload.get('price') is None, x.payload.get('price') or 0))
        elif sort_by == "price_high":
            points.sort(key=lambda x: (x.payload.get('price') is None, -(x.payload.get('price') or 0)))
        else:  # rating
            points.sort(key=lambda x: x.payload.get('rating', 0), reverse=True)

        # Format results
        formatted = []
        for hit in points[:limit]:
            price = hit.payload.get('price')
            price_available = hit.payload.get('price_available', price is not None)
            price_str = f"${price:.2f}" if price is not None else "Price N/A"

            formatted.append({
                "product_id": hit.id,
                "title": hit.payload["title"],
                "description": hit.payload.get("description", ""),
                "price": price,
                "price_available": price_available,
                "rating": hit.payload["rating"],
                "category": hit.payload["category"],
                "image_url": hit.payload.get("image_url", ""),
                "explanation": f"{price_str} • {hit.payload['rating']}★ • {category}"
            })
        return formatted

    def get_categories(self) -> List[str]:
        """Get all unique categories in the database"""
        categories = set()
        offset = None

        while True:
            response = self.client.scroll(
                collection_name="products_text",
                limit=1000,
                offset=offset,
                with_payload=["category"]
            )
            points, offset = response

            if not points:
                break

            for point in points:
                cat = point.payload.get('category', '')
                if cat:
                    categories.add(cat)

            if offset is None:
                break

        return sorted(list(categories))

    def search_with_price_range(
        self,
        query: str,
        min_budget: float,
        max_budget: float,
        min_rating: float = 0.0,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search with full price range support and optional category filter

        Args:
            query: Search query
            min_budget: Minimum price
            max_budget: Maximum price
            min_rating: Minimum rating
            category: Optional category filter
            limit: Number of results
        """
        # Use cached encoding
        query_vector = self._encode_query(query)

        # Build filter conditions
        filter_conditions = [
            FieldCondition(key="price", range=Range(gte=min_budget, lte=max_budget))
        ]

        if min_rating > 0:
            filter_conditions.append(
                FieldCondition(key="rating", range=Range(gte=min_rating))
            )

        if category:
            filter_conditions.append(
                FieldCondition(key="category", match=MatchValue(value=category))
            )

        response = self.client.query_points(
            collection_name="products_text",
            query=query_vector,
            query_filter=Filter(must=filter_conditions),
            limit=limit,
            with_payload=True
        )

        return self._format_results(response.points, query)

    def search_with_reranking(
        self,
        query: str,
        max_budget: float,
        min_rating: float = 0.0,
        limit: int = 10,
        enable_cross_encoder: bool = True,
        enable_mmr: bool = True,
        enable_boosting: bool = True,
        category_hints: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search with advanced reranking for better quality results.

        Pipeline:
        1. Initial semantic search (fetch more candidates)
        2. Cross-encoder reranking (more accurate relevance)
        3. Rating/popularity/category boosting (prefer matching categories)
        4. MMR diversity (avoid redundant results)

        Args:
            query: Search query
            max_budget: Maximum price constraint
            min_rating: Minimum product rating
            limit: Number of final results
            enable_cross_encoder: Use cross-encoder for accurate scoring
            enable_mmr: Use MMR for diverse results
            enable_boosting: Boost by rating, popularity, and value
            category_hints: Detected category hints for affinity boosting

        Returns:
            Reranked, diverse, high-quality results
        """
        if not RERANKING_ENABLED:
            # Fall back to regular search
            return self.search_products(query, max_budget, min_rating, limit)

        # Get more candidates than needed for reranking
        candidates = self.search_products(
            query=query,
            max_budget=max_budget,
            min_rating=min_rating,
            limit=limit * 5  # Get 5x more for reranking
        )

        if not candidates:
            return candidates

        # Configure reranker
        config = RerankConfig(
            enable_cross_encoder=enable_cross_encoder,
            enable_mmr=enable_mmr,
            enable_boosting=enable_boosting
        )
        reranker = get_reranker(config)

        # Rerank results
        reranked = reranker.rerank(
            query=query,
            results=candidates,
            max_budget=max_budget,
            top_k=limit,
            category_hints=category_hints
        )

        # Add reranking metadata to explanation
        for r in reranked:
            original_explanation = r.get('explanation', '')
            rerank_info = []

            if 'cross_encoder_score' in r:
                rerank_info.append(f"Cross-encoder: {r['cross_encoder_score']:.2f}")
            if 'boosts' in r:
                boosts = r['boosts']
                if boosts.get('rating_boost', 0) > 0:
                    rerank_info.append(f"Rating boost: +{boosts['rating_boost']:.3f}")
                if boosts.get('popularity_boost', 0) > 0:
                    rerank_info.append(f"Popularity: +{boosts['popularity_boost']:.3f}")
                if boosts.get('category_boost', 0) > 0:
                    rerank_info.append(f"Category: +{boosts['category_boost']:.3f}")
                if boosts.get('value_boost', 0) > 0:
                    rerank_info.append(f"Value boost: +{boosts['value_boost']:.3f}")
            if 'final_rank' in r:
                rerank_info.append(f"Rank: #{r['final_rank']}")

            if rerank_info:
                r['explanation'] = f"{original_explanation} | Reranking: {' • '.join(rerank_info)}"
            r['reranked'] = True

        return reranked

    def smart_search_reranked(
        self,
        query: str,
        max_budget: Optional[float] = None,
        min_rating: Optional[float] = None,
        limit: int = 10
    ) -> Dict:
        """
        Intelligent search with query understanding AND reranking.

        Combines:
        - Query understanding (budget extraction, intent detection, query expansion)
        - Cross-encoder reranking (more accurate relevance scoring)
        - MMR diversity (avoid showing duplicate-like results)
        - Rating/value boosting (surface best deals)

        Args:
            query: Natural language query (e.g., "I need running shoes under $80")
            max_budget: Override budget (uses extracted if None)
            min_rating: Override rating (uses intent-based if None)
            limit: Number of results

        Returns:
            Dict with reranked results and query understanding metadata
        """
        from query_understanding import understand_query

        # Parse the query
        parsed = understand_query(query, default_budget=max_budget or 200.0)

        # Use extracted/detected values unless overridden
        effective_budget = max_budget if max_budget is not None else (parsed.extracted_budget or 200.0)
        effective_rating = min_rating if min_rating is not None else parsed.min_rating

        # Search with reranking (pass category hints for affinity boosting)
        results = self.search_with_reranking(
            query=parsed.enhanced_query,
            max_budget=effective_budget,
            min_rating=effective_rating,
            limit=limit,
            category_hints=parsed.categories_hint
        )

        # Return results with understanding metadata
        return {
            "results": results,
            "understanding": {
                "original_query": parsed.original_query,
                "enhanced_query": parsed.enhanced_query,
                "extracted_budget": parsed.extracted_budget,
                "effective_budget": effective_budget,
                "detected_intent": parsed.intent,
                "min_rating_applied": effective_rating,
                "category_hints": parsed.categories_hint,
                "attributes": parsed.attributes,
                "confidence": parsed.confidence
            },
            "reranking_applied": RERANKING_ENABLED
        }
