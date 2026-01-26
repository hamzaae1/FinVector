from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Literal
from finvector_core import FinVectorSearch
from PIL import Image
from io import BytesIO
import base64
import os

# Import analytics (optional)
try:
    from logger import SearchAnalytics, APILogger
    import time
    ANALYTICS_ENABLED = True
except ImportError:
    ANALYTICS_ENABLED = False

# Import feedback system
try:
    from feedback import feedback_loop, FeedbackLoop
    FEEDBACK_ENABLED = True
except ImportError:
    FEEDBACK_ENABLED = False
    feedback_loop = None

app = FastAPI(
    title="FinVector API",
    version="1.0.0",
    description="Budget-Aware E-Commerce Search with Text and Image Support"
)

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search engine
search_engine = FinVectorSearch()

# Request models
class SearchRequest(BaseModel):
    query: str
    max_budget: float
    min_rating: Optional[float] = 0.0
    limit: Optional[int] = 10

class AlternativesRequest(BaseModel):
    product_id: str
    max_budget: float
    min_rating: Optional[float] = 4.0
    limit: Optional[int] = 5
    sort_by: Optional[Literal["similarity", "price_low", "price_high", "rating", "value"]] = "similarity"

class BudgetStretcherRequest(BaseModel):
    product_id: str
    total_budget: float
    sort_by: Optional[Literal["similarity", "price_low", "price_high", "rating", "value"]] = "value"

# Endpoints
@app.get("/")
async def root():
    return {
        "message": "FinVector API - Budget-Aware Product Search",
        "version": "1.0.0",
        "endpoints": ["/search", "/alternatives", "/budget-stretcher"]
    }

@app.post("/search")
async def search_products(request: SearchRequest):
    """Search for products within budget"""
    try:
        results = search_engine.search_products(
            query=request.query,
            max_budget=request.max_budget,
            min_rating=request.min_rating,
            limit=request.limit
        )
        return {
            "success": True,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SmartSearchRequest(BaseModel):
    query: str  # Natural language query like "I need running shoes under $80"
    max_budget: Optional[float] = None  # Optional - extracted from query if not provided
    min_rating: Optional[float] = None  # Optional - based on intent if not provided
    limit: Optional[int] = 10


@app.post("/search/smart")
async def smart_search(request: SmartSearchRequest):
    """
    Intelligent search that understands natural language.

    Examples:
    - "cheap running shoes under $80" → extracts budget=$80, intent=budget
    - "best laptop for programming" → expands query, intent=quality
    - "I need a gift under $50" → extracts budget=$50
    - "wireless headphones" → expands to include "bluetooth earbuds"
    """
    try:
        result = search_engine.smart_search(
            query=request.query,
            max_budget=request.max_budget,
            min_rating=request.min_rating,
            limit=request.limit
        )
        return {
            "success": True,
            "count": len(result["results"]),
            "understanding": result["understanding"],
            "results": result["results"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alternatives")
async def get_alternatives(request: AlternativesRequest):
    """
    Find cheaper alternatives to a product.

    Sort options:
    - similarity: Most similar first (default)
    - price_low: Cheapest first
    - price_high: Most expensive first (within budget)
    - rating: Highest rated first
    - value: Best value (balance of similarity, savings, rating)
    """
    try:
        alternatives = search_engine.find_alternatives(
            product_id=request.product_id,
            max_budget=request.max_budget,
            min_rating=request.min_rating,
            limit=request.limit,
            sort_by=request.sort_by
        )
        return {
            "success": True,
            "count": len(alternatives),
            "sort_by": request.sort_by,
            "alternatives": alternatives
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/budget-stretcher")
async def budget_stretcher(request: BudgetStretcherRequest):
    """
    Get bundle suggestions to maximize budget value.

    Sort options for alternatives:
    - similarity: Most similar first
    - price_low: Cheapest first
    - price_high: Most expensive first (within budget)
    - rating: Highest rated first
    - value: Best value (default - balance of similarity, savings, rating)
    """
    try:
        result = search_engine.budget_stretcher(
            product_id=request.product_id,
            total_budget=request.total_budget,
            sort_by=request.sort_by
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/app")
async def serve_app():
    """Serve the frontend application"""
    return FileResponse(os.path.join(static_dir, "index.html"))


# Image Search Endpoints

class ImageSearchByURLRequest(BaseModel):
    image_url: str
    max_budget: float
    min_rating: Optional[float] = 0.0
    limit: Optional[int] = 10


class TextToImageSearchRequest(BaseModel):
    query: str
    max_budget: float
    min_rating: Optional[float] = 0.0
    limit: Optional[int] = 10


@app.post("/search/image/upload")
async def search_by_image_upload(
    file: UploadFile = File(...),
    max_budget: float = Form(...),
    min_rating: float = Form(0.0),
    limit: int = Form(10)
):
    """Search for similar products by uploading an image"""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        results = search_engine.search_by_image(
            image=image,
            max_budget=max_budget,
            min_rating=min_rating,
            limit=limit
        )
        return {
            "success": True,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image/url")
async def search_by_image_url(request: ImageSearchByURLRequest):
    """Search for similar products using an image URL"""
    try:
        results = search_engine.search_by_image(
            image=request.image_url,
            max_budget=request.max_budget,
            min_rating=request.min_rating,
            limit=request.limit
        )
        return {
            "success": True,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image/text")
async def search_images_by_text(request: TextToImageSearchRequest):
    """Search for products using text query against image embeddings (CLIP text-to-image)"""
    try:
        results = search_engine.search_by_text_for_images(
            query=request.query,
            max_budget=request.max_budget,
            min_rating=request.min_rating,
            limit=request.limit
        )
        return {
            "success": True,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Analytics Endpoints

@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get search analytics summary"""
    if not ANALYTICS_ENABLED:
        return {"error": "Analytics not enabled"}

    try:
        recent_searches = SearchAnalytics.get_recent_searches(100)
        avg_response_time = SearchAnalytics.get_average_response_time()
        popular_queries = SearchAnalytics.get_popular_queries(10)

        return {
            "success": True,
            "total_searches": len(recent_searches),
            "average_response_time_ms": round(avg_response_time, 2),
            "popular_queries": [
                {"query": q, "count": c} for q, c in popular_queries
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/recent")
async def get_recent_searches(limit: int = 20):
    """Get recent search queries"""
    if not ANALYTICS_ENABLED:
        return {"error": "Analytics not enabled"}

    try:
        searches = SearchAnalytics.get_recent_searches(limit)
        return {
            "success": True,
            "count": len(searches),
            "searches": searches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Feedback & Personalization Endpoints

class InteractionRequest(BaseModel):
    user_id: str
    product_id: str
    action: str  # click, add_to_cart, purchase
    product_data: Optional[dict] = {}
    query: Optional[str] = ""


class PersonalizedSearchRequest(BaseModel):
    user_id: str
    query: str
    max_budget: float
    min_rating: Optional[float] = 0.0
    limit: Optional[int] = 10


@app.post("/feedback/interaction")
async def record_interaction(request: InteractionRequest):
    """Record a user interaction (click, add_to_cart, purchase)"""
    if not FEEDBACK_ENABLED:
        return {"success": False, "error": "Feedback system not enabled"}

    try:
        if request.action == "click":
            feedback_loop.record_click(
                user_id=request.user_id,
                product_id=request.product_id,
                product_data=request.product_data,
                query=request.query
            )
        elif request.action == "add_to_cart":
            feedback_loop.record_add_to_cart(
                user_id=request.user_id,
                product_id=request.product_id,
                product_data=request.product_data
            )
        elif request.action == "purchase":
            feedback_loop.record_purchase(
                user_id=request.user_id,
                product_id=request.product_id,
                product_data=request.product_data
            )

        return {"success": True, "message": f"Recorded {request.action} interaction"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/profile/{user_id}")
async def get_user_profile(user_id: str):
    """Get user preference profile"""
    if not FEEDBACK_ENABLED:
        return {"success": False, "error": "Feedback system not enabled"}

    try:
        profile = feedback_loop.get_user_profile(user_id)
        return {"success": True, "profile": profile}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/personalized")
async def personalized_search(request: PersonalizedSearchRequest):
    """Search with personalization based on user history"""
    try:
        # Get base search results
        results = search_engine.search_products(
            query=request.query,
            max_budget=request.max_budget,
            min_rating=request.min_rating,
            limit=request.limit * 2  # Get more results to re-rank
        )

        # Apply personalization if enabled
        if FEEDBACK_ENABLED and request.user_id:
            results = feedback_loop.apply_personalization(request.user_id, results)
            results = results[:request.limit]

        return {
            "success": True,
            "count": len(results),
            "personalized": FEEDBACK_ENABLED and bool(request.user_id),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== NEW FEATURES ==============

# Hybrid Search (Image + Text)
class HybridSearchRequest(BaseModel):
    image_url: str
    text_query: Optional[str] = ""  # Optional - empty means pure image search
    max_budget: float
    min_rating: Optional[float] = 0.0
    limit: Optional[int] = 10
    image_weight: Optional[float] = 0.7


@app.post("/search/hybrid")
async def hybrid_search(request: HybridSearchRequest):
    """
    Hybrid search: combine image similarity with text refinement.
    - If text_query is empty: pure image search
    - If text_query provided: image + text hybrid search
    """
    try:
        # If no text query, do pure image search
        if not request.text_query or request.text_query.strip() == "":
            results = search_engine.search_by_image(
                image=request.image_url,
                max_budget=request.max_budget,
                min_rating=request.min_rating,
                limit=request.limit
            )
            return {
                "success": True,
                "count": len(results),
                "search_type": "image_only",
                "results": results
            }

        # Hybrid search with text refinement
        results = search_engine.hybrid_search(
            image=request.image_url,
            text_query=request.text_query,
            max_budget=request.max_budget,
            min_rating=request.min_rating,
            limit=request.limit,
            image_weight=request.image_weight
        )
        return {
            "success": True,
            "count": len(results),
            "search_type": "hybrid",
            "image_weight": request.image_weight,
            "text_weight": 1 - request.image_weight,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/hybrid/upload")
async def hybrid_search_upload(
    file: UploadFile = File(...),
    text_query: str = Form(""),  # Optional - empty means pure image search
    max_budget: float = Form(...),
    min_rating: float = Form(0.0),
    limit: int = Form(10),
    image_weight: float = Form(0.7)
):
    """Hybrid search with image upload + optional text refinement"""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        # If no text query, do pure image search
        if not text_query or text_query.strip() == "":
            results = search_engine.search_by_image(
                image=image,
                max_budget=max_budget,
                min_rating=min_rating,
                limit=limit
            )
            return {
                "success": True,
                "count": len(results),
                "search_type": "image_only",
                "results": results
            }

        # Hybrid search
        results = search_engine.hybrid_search(
            image=image,
            text_query=text_query,
            max_budget=max_budget,
            min_rating=min_rating,
            limit=limit,
            image_weight=image_weight
        )
        return {
            "success": True,
            "count": len(results),
            "search_type": "hybrid",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Category Browsing
class CategorySearchRequest(BaseModel):
    category: str
    max_budget: float
    min_budget: Optional[float] = 0.0
    min_rating: Optional[float] = 0.0
    limit: Optional[int] = 10
    sort_by: Optional[str] = "rating"  # rating, price_low, price_high


@app.get("/categories")
async def get_categories():
    """Get all available product categories"""
    try:
        categories = search_engine.get_categories()
        return {
            "success": True,
            "count": len(categories),
            "categories": categories
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/category")
async def search_by_category(request: CategorySearchRequest):
    """Browse products by category with sorting options"""
    try:
        results = search_engine.search_by_category(
            category=request.category,
            max_budget=request.max_budget,
            min_budget=request.min_budget,
            min_rating=request.min_rating,
            limit=request.limit,
            sort_by=request.sort_by
        )
        return {
            "success": True,
            "count": len(results),
            "category": request.category,
            "sort_by": request.sort_by,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Advanced Search with Price Range
class AdvancedSearchRequest(BaseModel):
    query: str
    min_budget: Optional[float] = 0.0
    max_budget: float
    min_rating: Optional[float] = 0.0
    category: Optional[str] = None
    limit: Optional[int] = 10


@app.post("/search/advanced")
async def advanced_search(request: AdvancedSearchRequest):
    """Advanced search with price range and category filters"""
    try:
        results = search_engine.search_with_price_range(
            query=request.query,
            min_budget=request.min_budget,
            max_budget=request.max_budget,
            min_rating=request.min_rating,
            category=request.category,
            limit=request.limit
        )
        return {
            "success": True,
            "count": len(results),
            "filters": {
                "price_range": [request.min_budget, request.max_budget],
                "min_rating": request.min_rating,
                "category": request.category
            },
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== RERANKING ENDPOINTS ==============

class RerankedSearchRequest(BaseModel):
    query: str
    max_budget: float
    min_rating: Optional[float] = 0.0
    limit: Optional[int] = 10
    enable_cross_encoder: Optional[bool] = True
    enable_mmr: Optional[bool] = True
    enable_boosting: Optional[bool] = True


@app.post("/search/reranked")
async def reranked_search(request: RerankedSearchRequest):
    """
    Search with advanced reranking for higher quality results.

    Pipeline:
    1. Initial semantic search (fetches more candidates)
    2. Cross-encoder reranking (more accurate relevance scoring)
    3. Rating/value boosting (surfaces highly-rated, good value items)
    4. MMR diversity (avoids showing duplicate-like results)

    Options:
    - enable_cross_encoder: Use cross-encoder for accurate scoring (slower but better)
    - enable_mmr: Use MMR for diverse results (avoids showing 10 similar items)
    - enable_boosting: Boost by rating and value (surfaces best deals)
    """
    try:
        results = search_engine.search_with_reranking(
            query=request.query,
            max_budget=request.max_budget,
            min_rating=request.min_rating,
            limit=request.limit,
            enable_cross_encoder=request.enable_cross_encoder,
            enable_mmr=request.enable_mmr,
            enable_boosting=request.enable_boosting
        )
        return {
            "success": True,
            "count": len(results),
            "reranking": {
                "cross_encoder": request.enable_cross_encoder,
                "mmr_diversity": request.enable_mmr,
                "boosting": request.enable_boosting
            },
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SmartRerankedSearchRequest(BaseModel):
    query: str  # Natural language query
    max_budget: Optional[float] = None  # Extracted from query if not provided
    min_rating: Optional[float] = None  # Based on intent if not provided
    limit: Optional[int] = 10


@app.post("/search/smart/reranked")
async def smart_reranked_search(request: SmartRerankedSearchRequest):
    """
    The best of both worlds: Query understanding + Reranking.

    Combines:
    - Query understanding (budget extraction, intent detection, query expansion)
    - Cross-encoder reranking (more accurate relevance scoring)
    - MMR diversity (avoid redundant results)
    - Rating/value boosting (surface best deals)

    Examples:
    - "cheap running shoes under $80" → budget=$80, intent=budget, expands query, reranks
    - "best laptop for programming" → intent=quality, high min_rating, expands, reranks
    - "wireless headphones" → expands to "bluetooth earbuds", reranks for diversity
    """
    try:
        result = search_engine.smart_search_reranked(
            query=request.query,
            max_budget=request.max_budget,
            min_rating=request.min_rating,
            limit=request.limit
        )
        return {
            "success": True,
            "count": len(result["results"]),
            "understanding": result["understanding"],
            "reranking_applied": result.get("reranking_applied", False),
            "results": result["results"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
