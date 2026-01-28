"""
FinVector - Reranking & Quality Enhancement Module
Improves search result quality with cross-encoder reranking and diversity
"""
import torch
import numpy as np
import time
import hashlib
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple, Any
from sentence_transformers import CrossEncoder, SentenceTransformer
from dataclasses import dataclass, field

# GPU support
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RerankConfig:
    """Configuration for reranking"""
    enable_cross_encoder: bool = True
    enable_mmr: bool = True  # Maximal Marginal Relevance for diversity
    enable_boosting: bool = True  # Boost by rating/popularity
    cross_encoder_weight: float = 0.6  # Weight for cross-encoder score
    mmr_lambda: float = 0.7  # Balance between relevance and diversity
    rating_boost_weight: float = 0.1  # How much rating affects final score
    popularity_boost_weight: float = 0.08  # How much review count affects score
    category_boost_weight: float = 0.12  # How much category match affects score


@dataclass
class CacheConfig:
    """Configuration for semantic cache"""
    enabled: bool = True
    similarity_threshold: float = 0.85  # Min similarity for cache hit (0.85 = close synonyms)
    max_size: int = 1000  # Max cached entries
    ttl_seconds: int = 3600  # 1 hour default TTL
    budget_tolerance: float = 0.15  # 15% budget difference tolerance
    embedding_model: str = "all-MiniLM-L6-v2"  # Fast model for cache lookups


class SemanticCache:
    """
    Semantic caching for search results.

    Caches results by query meaning, not exact string match.
    "running shoes" and "jogging sneakers" can share cached results.

    Features:
    - Semantic similarity matching via sentence embeddings
    - Budget-aware caching (different budgets = different cache entries)
    - LRU eviction when cache is full
    - TTL-based expiration
    - Cache statistics for monitoring
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._encoder = None

        # Cache storage: OrderedDict for LRU ordering
        # Key: cache_key (hash)
        # Value: CacheEntry dict
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

    @property
    def encoder(self) -> SentenceTransformer:
        """Lazy load embedding model"""
        if self._encoder is None:
            print(f"Loading semantic cache encoder ({self.config.embedding_model})...")
            self._encoder = SentenceTransformer(self.config.embedding_model, device=DEVICE)
            print("Semantic cache encoder loaded!")
        return self._encoder

    def _compute_embedding(self, query: str) -> np.ndarray:
        """Compute normalized embedding for query"""
        embedding = self.encoder.encode(query, convert_to_numpy=True)
        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two normalized vectors"""
        return float(np.dot(a, b))

    def _is_budget_compatible(self, query_budget: Optional[float], cached_budget: Optional[float]) -> bool:
        """Check if budgets are close enough to share cache"""
        if query_budget is None or cached_budget is None:
            # If either has no budget, only match if both have no budget
            return query_budget is None and cached_budget is None

        if query_budget == 0 or cached_budget == 0:
            return query_budget == cached_budget

        # Check percentage difference
        diff_ratio = abs(query_budget - cached_budget) / max(query_budget, cached_budget)
        return diff_ratio <= self.config.budget_tolerance

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry has expired"""
        age = time.time() - entry["timestamp"]
        return age > self.config.ttl_seconds

    def _cleanup_expired(self):
        """Remove expired entries"""
        expired_keys = []
        for key, entry in self._cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]
            self.stats["expirations"] += 1

    def get(
        self,
        query: str,
        budget: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> Optional[List[Dict]]:
        """
        Look up query in semantic cache.

        Args:
            query: Search query
            budget: User's budget constraint
            top_k: Number of results needed

        Returns:
            Cached results if found, None otherwise
        """
        if not self.config.enabled:
            return None

        # Periodic cleanup
        if len(self._cache) > 0 and np.random.random() < 0.1:
            self._cleanup_expired()

        query_embedding = self._compute_embedding(query)

        best_match = None
        best_similarity = 0.0

        for cache_key, entry in self._cache.items():
            # Skip expired entries
            if self._is_expired(entry):
                continue

            # Check budget compatibility
            if not self._is_budget_compatible(budget, entry.get("budget")):
                continue

            # Check if cached results have enough items
            if top_k and len(entry["results"]) < top_k:
                continue

            # Compute semantic similarity
            similarity = self._cosine_similarity(query_embedding, entry["embedding"])

            if similarity >= self.config.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        if best_match:
            # Cache hit - move to end (most recently used)
            cache_key = best_match["cache_key"]
            self._cache.move_to_end(cache_key)
            self.stats["hits"] += 1

            # Return copy with cache metadata
            results = [r.copy() for r in best_match["results"]]
            for r in results:
                r["_cache_hit"] = True
                r["_cache_similarity"] = round(best_similarity, 4)

            return results[:top_k] if top_k else results

        self.stats["misses"] += 1
        return None

    def put(
        self,
        query: str,
        results: List[Dict],
        budget: Optional[float] = None
    ):
        """
        Store query results in cache.

        Args:
            query: Search query
            results: Reranked results to cache
            budget: Budget constraint used
        """
        if not self.config.enabled or not results:
            return

        # Evict oldest if at capacity
        while len(self._cache) >= self.config.max_size:
            self._cache.popitem(last=False)
            self.stats["evictions"] += 1

        # Create cache entry
        embedding = self._compute_embedding(query)
        cache_key = hashlib.md5(f"{query}:{budget}:{time.time()}".encode()).hexdigest()

        # Store clean copies (remove transient fields)
        clean_results = []
        for r in results:
            clean = r.copy()
            # Remove cache-specific fields from stored results
            clean.pop("_cache_hit", None)
            clean.pop("_cache_similarity", None)
            clean_results.append(clean)

        self._cache[cache_key] = {
            "cache_key": cache_key,
            "query": query,
            "embedding": embedding,
            "results": clean_results,
            "budget": budget,
            "timestamp": time.time(),
        }

    def invalidate(self, query: Optional[str] = None):
        """
        Invalidate cache entries.

        Args:
            query: Specific query to invalidate (None = clear all)
        """
        if query is None:
            self._cache.clear()
            return

        # Find and remove semantically similar entries
        query_embedding = self._compute_embedding(query)
        to_remove = []

        for cache_key, entry in self._cache.items():
            similarity = self._cosine_similarity(query_embedding, entry["embedding"])
            if similarity >= self.config.similarity_threshold:
                to_remove.append(cache_key)

        for key in to_remove:
            del self._cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0

        return {
            **self.stats,
            "size": len(self._cache),
            "max_size": self.config.max_size,
            "hit_rate": round(hit_rate, 4),
            "total_requests": total,
        }

    def __len__(self) -> int:
        return len(self._cache)


class CrossEncoderReranker:
    """
    Cross-encoder reranking for more accurate relevance scoring.

    Cross-encoders are more accurate than bi-encoders because they
    see query and document together, but they're slower.
    We use them to rerank top candidates from fast vector search.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading cross-encoder on {DEVICE}...")
        self.model = CrossEncoder(model_name, device=DEVICE)
        print("Cross-encoder loaded!")

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Rerank results using cross-encoder.

        Args:
            query: The search query
            results: List of result dicts with 'title' and 'description'
            top_k: Return top K results (None = return all)

        Returns:
            Reranked results with updated scores
        """
        if not results:
            return results

        # Create query-document pairs
        pairs = []
        for r in results:
            # Combine title and description for scoring
            title = r.get('title', '')
            description = r.get('description', '')
            text = f"{title}. {description}"[:512]
            pairs.append([query, text])

        # Score all pairs
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Update results with scores
        for i, r in enumerate(results):
            r['cross_encoder_score'] = float(scores[i])

        # Sort by cross-encoder score
        results.sort(key=lambda x: x['cross_encoder_score'], reverse=True)

        if top_k:
            results = results[:top_k]

        return results


class MMRReranker:
    """
    Maximal Marginal Relevance for diverse results.

    Balances relevance with diversity - avoids showing 10 very similar items.
    """

    def __init__(self, lambda_param: float = 0.7):
        """
        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
        """
        self.lambda_param = lambda_param

    def rerank(
        self,
        results: List[Dict],
        embeddings: Optional[np.ndarray] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Apply MMR to diversify results.

        Args:
            results: List of results with 'similarity_score'
            embeddings: Optional embeddings for each result (for similarity calc)
            top_k: Number of results to return

        Returns:
            Diversified results
        """
        if not results or len(results) <= 1:
            return results

        n = len(results)
        top_k = top_k or n

        # Use similarity scores if no embeddings provided
        if embeddings is None:
            # Simple diversity based on scores
            return self._simple_mmr(results, top_k)

        # Full MMR with embeddings
        return self._embedding_mmr(results, embeddings, top_k)

    def _simple_mmr(self, results: List[Dict], top_k: int) -> List[Dict]:
        """Simple MMR without embeddings - uses category diversity"""
        selected = []
        remaining = results.copy()
        seen_categories = set()

        while len(selected) < top_k and remaining:
            # Score each candidate
            best_idx = 0
            best_score = float('-inf')

            for i, r in enumerate(remaining):
                relevance = r.get('similarity_score', 0) or r.get('cross_encoder_score', 0)

                # Penalize if same category already selected
                category = r.get('category', '')
                diversity_penalty = 0.1 if category in seen_categories else 0

                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * diversity_penalty

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            # Select best
            selected_item = remaining.pop(best_idx)
            selected_item['mmr_score'] = best_score
            selected.append(selected_item)
            seen_categories.add(selected_item.get('category', ''))

        return selected

    def _embedding_mmr(
        self,
        results: List[Dict],
        embeddings: np.ndarray,
        top_k: int
    ) -> List[Dict]:
        """Full MMR with embedding-based similarity"""
        n = len(results)
        selected_indices = []
        remaining_indices = list(range(n))

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)

        while len(selected_indices) < top_k and remaining_indices:
            best_idx = None
            best_score = float('-inf')

            for i in remaining_indices:
                # Relevance score
                relevance = results[i].get('similarity_score', 0)

                # Max similarity to already selected items
                if selected_indices:
                    selected_embeddings = normalized[selected_indices]
                    similarities = np.dot(selected_embeddings, normalized[i])
                    max_sim = np.max(similarities)
                else:
                    max_sim = 0

                # MMR score
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            results[best_idx]['mmr_score'] = best_score

        return [results[i] for i in selected_indices]


class ResultBooster:
    """
    Boost results based on additional signals like rating, popularity, value, and category.

    Signals:
    - Rating: Higher rated products get a boost (0-5 stars normalized)
    - Popularity: Products with more reviews are more trusted (log-scaled)
    - Value: Products under budget get a value boost
    - Category: Products matching detected category hints get a boost
    """

    def __init__(
        self,
        rating_weight: float = 0.1,
        price_value_weight: float = 0.05,
        popularity_weight: float = 0.08,
        category_weight: float = 0.12
    ):
        self.rating_weight = rating_weight
        self.price_value_weight = price_value_weight
        self.popularity_weight = popularity_weight
        self.category_weight = category_weight

        # Log scale for review count normalization
        # log10(10000) ≈ 4, so products with 10k+ reviews get max boost
        self.max_log_reviews = 4.0

    def _normalize_review_count(self, review_count: int) -> float:
        """
        Normalize review count using log scale.

        Log scale prevents products with millions of reviews from
        completely dominating. A product with 1000 reviews vs 100
        reviews has a meaningful but not overwhelming difference.

        Examples:
            1 review     -> ~0.0
            10 reviews   -> ~0.25
            100 reviews  -> ~0.5
            1000 reviews -> ~0.75
            10000+ reviews -> ~1.0
        """
        import math
        if review_count <= 0:
            return 0.0
        log_count = math.log10(review_count + 1)
        return min(1.0, log_count / self.max_log_reviews)

    def _category_match_score(self, product_category: str, category_hints: List[str]) -> float:
        """
        Calculate category match score.

        Returns 1.0 for exact match, 0.5 for partial match, 0 otherwise.
        """
        if not category_hints or not product_category:
            return 0.0

        product_cat_lower = product_category.lower()

        for hint in category_hints:
            hint_lower = hint.lower()

            # Exact match
            if product_cat_lower == hint_lower:
                return 1.0

            # Partial match (category contains hint or vice versa)
            if hint_lower in product_cat_lower or product_cat_lower in hint_lower:
                return 0.7

            # Word overlap (e.g., "Men's Shoes" matches "Shoes")
            hint_words = set(hint_lower.split())
            cat_words = set(product_cat_lower.split())
            overlap = hint_words & cat_words
            if overlap:
                return 0.5

        return 0.0

    def boost(
        self,
        results: List[Dict],
        max_budget: float,
        category_hints: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Apply boosting based on rating, popularity, value, and category affinity.

        Args:
            results: Search results
            max_budget: User's max budget (for value calculation)
            category_hints: Detected category hints from query understanding

        Returns:
            Results with boosted scores
        """
        for r in results:
            base_score = r.get('cross_encoder_score') or r.get('similarity_score', 0)

            # Rating boost (normalized to 0-1)
            rating = r.get('rating', 0)
            rating_boost = (rating / 5.0) * self.rating_weight

            # Popularity boost (review count, log-scaled)
            # Look for review_count or rating_count field
            review_count = r.get('review_count') or r.get('rating_count') or r.get('reviews', 0)
            popularity_score = self._normalize_review_count(review_count)
            popularity_boost = popularity_score * self.popularity_weight

            # Category affinity boost
            product_category = r.get('category', '')
            category_match = self._category_match_score(product_category, category_hints or [])
            category_boost = category_match * self.category_weight

            # Value boost (how much under budget)
            # Skip value boost for items with unknown/None prices - don't poison rankings
            price = r.get('price')
            price_available = r.get('price_available', price is not None)

            if price_available and price is not None and max_budget > 0:
                value_ratio = 1 - (price / max_budget)  # Higher if cheaper
                value_boost = max(0, value_ratio) * self.price_value_weight
            else:
                value_boost = 0  # No value boost for unknown prices

            # Combined score
            r['boosted_score'] = base_score + rating_boost + popularity_boost + category_boost + value_boost
            r['boosts'] = {
                'rating_boost': round(rating_boost, 4),
                'popularity_boost': round(popularity_boost, 4),
                'category_boost': round(category_boost, 4),
                'value_boost': round(value_boost, 4),
                'price_available': price_available,
                'review_count': review_count,
                'category_match': round(category_match, 2)
            }

        # Sort by boosted score
        results.sort(key=lambda x: x.get('boosted_score', 0), reverse=True)

        return results


class SearchReranker:
    """
    Main reranking pipeline combining all techniques.

    Pipeline: Cache Check → Cross-Encoder → Boosting → MMR → Cache Store
    """

    def __init__(
        self,
        config: Optional[RerankConfig] = None,
        cache_config: Optional[CacheConfig] = None
    ):
        self.config = config or RerankConfig()
        self.cache_config = cache_config or CacheConfig()

        self._cross_encoder = None
        self._mmr = None
        self._booster = None
        self._cache = None

    @property
    def cache(self) -> SemanticCache:
        """Lazy load semantic cache"""
        if self._cache is None:
            self._cache = SemanticCache(self.cache_config)
        return self._cache

    @property
    def cross_encoder(self):
        """Lazy load cross-encoder"""
        if self._cross_encoder is None and self.config.enable_cross_encoder:
            self._cross_encoder = CrossEncoderReranker()
        return self._cross_encoder

    @property
    def mmr(self):
        """Lazy load MMR"""
        if self._mmr is None and self.config.enable_mmr:
            self._mmr = MMRReranker(lambda_param=self.config.mmr_lambda)
        return self._mmr

    @property
    def booster(self):
        """Lazy load booster"""
        if self._booster is None and self.config.enable_boosting:
            self._booster = ResultBooster(
                rating_weight=self.config.rating_boost_weight,
                popularity_weight=self.config.popularity_boost_weight,
                category_weight=self.config.category_boost_weight
            )
        return self._booster

    def rerank(
        self,
        query: str,
        results: List[Dict],
        max_budget: float = 0,
        top_k: Optional[int] = None,
        embeddings: Optional[np.ndarray] = None,
        use_cache: bool = True,
        category_hints: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Full reranking pipeline with semantic caching.

        Args:
            query: Search query
            results: Initial search results
            max_budget: User's budget (for value boosting)
            top_k: Number of final results
            embeddings: Optional embeddings for MMR
            use_cache: Whether to use semantic cache
            category_hints: Detected category hints for affinity boosting

        Returns:
            Reranked and diversified results
        """
        if not results:
            return results

        # Step 0: Check semantic cache
        if use_cache and self.cache_config.enabled:
            cached = self.cache.get(query, budget=max_budget, top_k=top_k)
            if cached:
                # Update ranks for cached results
                for i, r in enumerate(cached):
                    r['final_rank'] = i + 1
                return cached

        # Step 1: Cross-encoder reranking
        if self.config.enable_cross_encoder and self.cross_encoder:
            results = self.cross_encoder.rerank(query, results)

        # Step 2: Apply boosting (rating, popularity, category, value)
        if self.config.enable_boosting and self.booster:
            results = self.booster.boost(results, max_budget, category_hints)

        # Step 3: MMR for diversity
        if self.config.enable_mmr and self.mmr:
            results = self.mmr.rerank(results, embeddings, top_k)
        elif top_k:
            results = results[:top_k]

        # Update final scores
        for i, r in enumerate(results):
            r['final_rank'] = i + 1
            r['final_score'] = r.get('boosted_score') or r.get('cross_encoder_score') or r.get('similarity_score', 0)

        # Step 4: Store in cache
        if use_cache and self.cache_config.enabled:
            self.cache.put(query, results, budget=max_budget)

        return results

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get semantic cache statistics"""
        return self.cache.get_stats()

    def invalidate_cache(self, query: Optional[str] = None):
        """Invalidate cache entries (all or semantically similar to query)"""
        self.cache.invalidate(query)


# Global instance (lazy loaded)
_reranker = None
_semantic_cache = None


def get_reranker(
    config: Optional[RerankConfig] = None,
    cache_config: Optional[CacheConfig] = None
) -> SearchReranker:
    """Get or create global reranker instance"""
    global _reranker
    if _reranker is None:
        _reranker = SearchReranker(config, cache_config)
    return _reranker


def get_semantic_cache(config: Optional[CacheConfig] = None) -> SemanticCache:
    """Get or create standalone semantic cache instance"""
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticCache(config)
    return _semantic_cache


def rerank_results(
    query: str,
    results: List[Dict],
    max_budget: float = 0,
    top_k: Optional[int] = None,
    use_cache: bool = True,
    category_hints: Optional[List[str]] = None
) -> List[Dict]:
    """
    Convenience function to rerank results with semantic caching.

    Example:
        results = search_engine.search_products("running shoes", 100)
        reranked = rerank_results("running shoes", results, max_budget=100, top_k=10)

        # Similar query hits cache
        reranked2 = rerank_results("jogging sneakers", results, max_budget=100, top_k=10)

        # With category hints for affinity boosting
        reranked3 = rerank_results("running shoes", results, category_hints=["Shoes"])
    """
    reranker = get_reranker()
    return reranker.rerank(
        query, results, max_budget, top_k,
        use_cache=use_cache, category_hints=category_hints
    )


def get_cache_stats() -> Dict[str, Any]:
    """Get global reranker cache statistics"""
    return get_reranker().get_cache_stats()


def invalidate_cache(query: Optional[str] = None):
    """Invalidate cache entries (all or semantically similar to query)"""
    get_reranker().invalidate_cache(query)


# Test
if __name__ == "__main__":
    import copy

    # Create sample results with review counts for popularity testing
    sample_results = [
        {"title": "Running Shoes Nike", "description": "Great for running", "similarity_score": 0.85, "rating": 4.5, "price": 80, "category": "Shoes", "review_count": 5000},
        {"title": "Running Shoes Adidas", "description": "Comfortable running shoes", "similarity_score": 0.82, "rating": 4.2, "price": 90, "category": "Shoes", "review_count": 12000},
        {"title": "Walking Shoes", "description": "Good for walking", "similarity_score": 0.78, "rating": 4.8, "price": 60, "category": "Shoes", "review_count": 200},
        {"title": "Sports Watch", "description": "Track your runs", "similarity_score": 0.75, "rating": 4.0, "price": 50, "category": "Accessories", "review_count": 8500},
        {"title": "Running Socks", "description": "Moisture wicking socks", "similarity_score": 0.70, "rating": 4.3, "price": 15, "category": "Accessories", "review_count": 50},
    ]

    print("=" * 60)
    print("Reranking Test")
    print("=" * 60)

    print("\nOriginal order:")
    for i, r in enumerate(sample_results):
        print(f"  {i+1}. [{r['similarity_score']:.2f}] {r['title']}")

    reranked = rerank_results("running shoes", copy.deepcopy(sample_results), max_budget=100, top_k=5)

    print("\nAfter reranking:")
    for r in reranked:
        print(f"  {r['final_rank']}. [{r.get('final_score', 0):.2f}] {r['title']}")
        if 'boosts' in r:
            print(f"      Boosts: {r['boosts']}")

    # Test semantic caching
    print("\n" + "=" * 60)
    print("Semantic Caching Test")
    print("=" * 60)

    print("\nCache stats after first query:")
    stats = get_cache_stats()
    print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}, Size: {stats['size']}")

    # Similar query should hit cache
    print("\nTrying semantically similar query: 'nike running shoes'...")
    reranked2 = rerank_results("nike running shoes", copy.deepcopy(sample_results), max_budget=100, top_k=5)

    cache_hit = reranked2[0].get('_cache_hit', False)
    print(f"  Cache hit: {cache_hit}")
    if cache_hit:
        print(f"  Similarity to cached query: {reranked2[0].get('_cache_similarity', 0):.4f}")

    print("\nCache stats after similar query:")
    stats = get_cache_stats()
    print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}, Hit rate: {stats['hit_rate']:.2%}")

    # Different budget should miss cache
    print("\nTrying same query with different budget (200)...")
    reranked3 = rerank_results("running shoes", copy.deepcopy(sample_results), max_budget=200, top_k=5)
    cache_hit = reranked3[0].get('_cache_hit', False)
    print(f"  Cache hit: {cache_hit} (expected: False - different budget)")

    print("\nFinal cache stats:")
    stats = get_cache_stats()
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Cache size: {stats['size']}/{stats['max_size']}")
