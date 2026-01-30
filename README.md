<p align="center">
  <img src="https://img.shields.io/badge/Qdrant-Vector%20Search-red?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTV6Ii8+PC9zdmc+" alt="Qdrant"/>
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/PyTorch-GPU-orange?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/CLIP-Vision-purple?style=for-the-badge" alt="CLIP"/>
</p>

<h1 align="center">FinVector</h1>
<h3 align="center">Budget-First Smart Shopping with Semantic Search</h3>

<p align="center">
  <b>Team Cosine</b> | Qdrant Vector Search Hackathon 2026<br/>
  <i>Hamza Mhedhbi & Hachem Mastouri</i>
</p>

<p align="center">
  <a href="#-the-problem">Problem</a> •
  <a href="#-our-solution">Solution</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-getting-started">Setup</a> •
  <a href="#-demo">Demo</a>
</p>

---

##  The Problem

Traditional e-commerce search is fundamentally broken for budget-conscious shoppers:

1. **Dream Product Trap**: You search for "wireless headphones," fall in love with a $300 pair, then realize it's way over budget
2. **Endless Scrolling**: Products within your budget are buried on page 5
3. **No Budget Understanding**: Search engines don't understand "cheap but good" or "under $50"
4. **Text-Only Limitations**: You can't search by showing an image of what you want

**The result?** Frustration, wasted time, and impulse purchases beyond budget.

---

##  Our Solution

**FinVector** is a budget-aware semantic search engine that guarantees you'll **never see products you can't afford**. It combines:

- **Natural Language Understanding**: Extracts budget, intent, and preferences from plain English
- **Multi-Modal Search**: Find products by text, image, or both
- **Smart Recommendations**: Cheaper alternatives and bundle suggestions
- **Personalization**: Learns your preferences over time

### How It Works

```
You: "I need running shoes under $80"

FinVector understands:
  ✓ Budget: $80 (extracted automatically)
  ✓ Intent: Value-focused (wants good deal)
  ✓ Category: Running shoes, sneakers, athletic footwear
  ✓ Min Rating: 3.5★ (decent quality for budget intent)

Result: Only shows running shoes ≤$80, ranked by relevance + value
```

---

##  Features

### Core Search Capabilities

| Feature | Description |
|---------|-------------|
| **Smart Search** | Natural language queries with automatic budget/intent extraction |
| **Visual Search** | Upload an image to find visually similar products |
| **Hybrid Search** | Combine image + text ("like this but in blue") |
| **Category Browse** | Filter by category with price range and rating |

### Budget Optimization Tools

| Feature | Description |
|---------|-------------|
| **Find Alternatives** | Discover cheaper alternatives to any product with 5 sort modes |
| **Budget Stretcher** | Get a cheaper alternative + accessories for the same price |
| **Value Scoring** | Smart ranking balancing similarity, savings, and ratings |

### Intelligence & Quality

| Feature | Description |
|---------|-------------|
| **Query Expansion** | "wireless" automatically includes "bluetooth, cordless" |
| **Typo Correction** | Handles misspellings gracefully |
| **Cross-Encoder Reranking** | More accurate relevance scoring |
| **MMR Diversity** | Avoids showing 10 identical products |
| **Semantic Caching** | "running shoes" reuses cache for "jogging sneakers" |

### Personalization

| Feature | Description |
|---------|-------------|
| **Interaction Tracking** | Learns from clicks, cart adds, purchases |
| **Category Preferences** | Surfaces products from your favorite categories |
| **Price Range Learning** | Understands your typical budget |
| **Homepage Suggestions** | Personalized recommendations based on history |

---

##  Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              User Interface                              │
│                    (Tailwind CSS + Vanilla JavaScript)                   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            FastAPI Backend                               │
│                         18+ RESTful Endpoints                            │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
┌───────────────┐          ┌─────────────────┐          ┌─────────────────┐
│    Query      │          │    Embedding    │          │  Personalization│
│ Understanding │          │   Generation    │          │     Engine      │
│               │          │                 │          │                 │
│ • Budget      │          │ • Text: MPNet   │          │ • User profiles │
│ • Intent      │          │ • Image: CLIP   │          │ • Preferences   │
│ • Expansion   │          │ • Cross-encoder │          │ • Suggestions   │
└───────┬───────┘          └────────┬────────┘          └────────┬────────┘
        │                           │                            │
        └───────────────────────────┼────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Qdrant Cloud                                  │
│                                                                         │
│  ┌─────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │   products_text         │    │   products_images                   │ │
│  │   768-dim (MPNet)       │    │   512-dim (CLIP ViT-B-32)           │ │
│  │   50K+ products         │    │   Visual similarity                 │ │
│  └─────────────────────────┘    └─────────────────────────────────────┘ │
│                                                                         │
│  Features: Vector similarity • Filtered queries • Payload storage       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Search Pipeline

```
User Query: "cheap wireless headphones under $50 for gaming"
                              │
                              ▼
              ┌───────────────────────────────┐
              │      Query Understanding      │
              │                               │
              │  Budget: $50 (extracted)      │
              │  Intent: budget (cheap)       │
              │  Context: gaming              │
              │  Expand: +bluetooth +earbuds  │
              │  Min Rating: 3.0★             │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │     Embedding Generation      │
              │                               │
              │  Model: all-mpnet-base-v2     │
              │  Dimensions: 768              │
              │  Cache: LRU (500 queries)     │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │        Qdrant Search          │
              │                               │
              │  Collection: products_text    │
              │  Filter: price ≤ $50          │
              │  Filter: rating ≥ 3.0         │
              │  Limit: 50 candidates         │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │         Reranking             │
              │                               │
              │  1. Cross-encoder scoring     │
              │  2. Rating boost (0-30%)      │
              │  3. Value boost (0-20%)       │
              │  4. MMR diversity (λ=0.7)     │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │      Personalization          │
              │                               │
              │  Category boost (0.8-1.5x)    │
              │  Price preference (0.7-1.0x)  │
              └───────────────┬───────────────┘
                              │
                              ▼
                    Top K Results
            (Best matches within budget)
```

---

## 🔌 API Reference

### Search Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/search/smart` | Natural language search with query understanding |
| `POST` | `/search` | Basic semantic search with filters |
| `POST` | `/search/reranked` | Search with cross-encoder reranking |
| `POST` | `/search/smart/reranked` | Smart search + reranking combined |
| `POST` | `/search/advanced` | Price range + category + rating filters |
| `POST` | `/search/category` | Browse by category with sorting |

### Image Search Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/search/image/upload` | Search by uploading an image |
| `POST` | `/search/image/url` | Search by image URL |
| `POST` | `/search/image/text` | Text query against image embeddings |
| `POST` | `/search/hybrid/upload` | Image + text combined search |
| `POST` | `/search/hybrid` | Hybrid search with image URL |

### Product Discovery

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/alternatives` | Find cheaper alternatives (5 sort modes) |
| `POST` | `/budget-stretcher` | Alternative + complementary bundle |
| `GET` | `/categories` | List all available categories |
| `GET` | `/suggestions/{user_id}` | Personalized homepage suggestions |

### Personalization & Feedback

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/feedback/interaction` | Record click/cart/purchase |
| `GET` | `/feedback/profile/{user_id}` | Get user preference profile |
| `POST` | `/search/personalized` | Search with personalization boost |

### Utility

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/analytics/summary` | Search analytics summary |
| `GET` | `/app` | Serve web UI |

### Example Requests

**Smart Search:**
```bash
curl -X POST http://localhost:8000/search/smart \
  -H "Content-Type: application/json" \
  -d '{
    "query": "wireless headphones under $50 for gaming",
    "limit": 10
  }'
```

**Response:**
```json
{
  "success": true,
  "count": 10,
  "understanding": {
    "original_query": "wireless headphones under $50 for gaming",
    "enhanced_query": "wireless headphones gaming bluetooth earbuds cordless",
    "extracted_budget": 50.0,
    "effective_budget": 50.0,
    "detected_intent": "budget",
    "min_rating_applied": 3.0,
    "corrections": {},
    "excluded_terms": []
  },
  "results": [
    {
      "product_id": "12345",
      "title": "Gaming Wireless Headset with Microphone",
      "price": 39.99,
      "rating": 4.2,
      "category": "Electronics",
      "similarity_score": 0.87,
      "image_url": "https://..."
    }
  ]
}
```

**Find Alternatives:**
```bash
curl -X POST http://localhost:8000/alternatives \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "12345",
    "max_budget": 35.0,
    "sort_by": "value"
  }'
```

**Image Search:**
```bash
curl -X POST http://localhost:8000/search/image/upload \
  -F "file=@product_photo.jpg" \
  -F "max_budget=100" \
  -F "limit=10"
```

---

##  Getting Started

### Prerequisites

- Python 3.10+
- Qdrant Cloud account (free tier works)
- ~4GB RAM (for ML models)
- GPU optional (CUDA supported for faster inference)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/hamzaae1/FinVector.git
cd FinVector

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
QDRANT_URL=https://your-cluster-id.cloud.qdrant.io
QDRANT_API_KEY=your-api-key-here
```

### Data Ingestion

```bash
# Ingest the Amazon products dataset (50K+ products)
python ingest_amazon.py --amazon_products_path path/to/amazon_products.csv

# This will:
# - Create products_text collection (768-dim text embeddings)
# - Create products_images collection (512-dim CLIP embeddings)
# - Index all product metadata as payload
```

### Run the Server

```bash
# Development mode with auto-reload
uvicorn api:app --reload --port 8000

# Production mode
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Access the App

- **Web UI**: http://localhost:8000/app
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

##  Demo

### Text Search with Budget Extraction

![Text Search Demo](docs/text-search.gif)

*Search "laptop for programming under $800" - budget automatically extracted and enforced*

### Visual Search

![Visual Search Demo](docs/visual-search.gif)

*Upload a product image to find visually similar items within budget*

### Budget Stretcher

![Budget Stretcher Demo](docs/budget-stretcher.gif)

*See how switching to a cheaper alternative lets you buy accessories too*

### Find Alternatives

![Alternatives Demo](docs/alternatives.gif)

*Discover 5 cheaper alternatives with similarity scores and savings*

---

##  Technical Deep Dive

### Query Understanding Engine

The query understanding module (`query_understanding.py`) parses natural language:

```python
Input: "I need cheap black leather shoes for work under $120"

Output:
{
  "enhanced_query": "black leather shoes work office business professional",
  "extracted_budget": 120.0,
  "detected_intent": "professional",  # work context
  "min_rating": 4.0,  # professional = quality matters
  "attributes": {
    "color": "black",
    "material": "leather"
  },
  "categories_hint": ["Men's Shoes", "Women's Shoes"],
  "confidence": 0.85
}
```

**Budget Extraction Patterns:**
- `under $X`, `below $X`, `less than $X`
- `max $X`, `budget of $X`, `around $X`
- `up to $X`, `within $X`, `$X or less`

**Intent Detection:**
- `budget`: "cheap", "affordable", "budget" → min_rating: 3.0
- `quality`: "best", "top", "premium" → min_rating: 4.0
- `balanced`: "good value", "worth it" → min_rating: 3.5
- `professional`: "for work", "office" → min_rating: 4.0
- `gift`: "gift for", "present" → min_rating: 4.0

### Cross-Encoder Reranking

Traditional bi-encoder search encodes query and documents separately:

```
Query: "gaming mouse" → [0.2, 0.5, ...]
Doc: "Logitech G502" → [0.3, 0.4, ...]
Score: cosine_similarity(query_vec, doc_vec)
```

Our cross-encoder sees both together for more accurate scoring:

```
Input: "[CLS] gaming mouse [SEP] Logitech G502 Hero Gaming Mouse [SEP]"
Output: 0.92 (direct relevance score)
```

**Result:** 5-15% improvement in relevance accuracy.

### MMR Diversity Algorithm

Maximal Marginal Relevance prevents redundant results:

```
score(doc) = λ × relevance(doc) - (1-λ) × max_sim(doc, selected_docs)

With λ=0.7:
- 70% weight on relevance
- 30% penalty for similarity to already-selected items
```

**Result:** Instead of 10 nearly identical products, you get diverse options.

### Semantic Cache

Similar queries share cached results:

```python
Cache key: (query_embedding, budget_range)
Cache hit: cosine_similarity(new_query, cached_query) ≥ 0.85
           AND budget within 15% tolerance

"running shoes $80" → cache miss, compute results
"jogging sneakers $75" → cache HIT (0.89 similarity, budget in range)
```

**Result:** 10x faster response for semantically similar queries.

---

##  Performance

### Benchmark Results

| Metric | Value |
|--------|-------|
| Average Search Latency | 309ms |
| P95 Latency | 330ms |
| Model Initialization | 5.2s (one-time) |
| Concurrent Requests (50) | Stable ±14.6ms |
| Cache Hit Rate | ~35% |

### Scalability

- **50K+ products** indexed and searchable
- **HNSW index** for sub-linear search complexity
- **Query-time filtering** for instant budget enforcement
- **Semantic caching** reduces redundant computation
- **GPU acceleration** supported for inference

---

##  Project Structure

```
FinVector/
├── api.py                    # FastAPI application (18+ endpoints)
├── finvector_core.py         # Core search engine (1,261 lines)
├── query_understanding.py    # NLP query parsing (352 lines)
├── reranker.py               # Cross-encoder & MMR (400+ lines)
├── feedback.py               # Personalization system (350 lines)
├── logger.py                 # Analytics & logging (224 lines)
├── ingest_amazon.py          # Data ingestion pipeline
│
├── static/
│   └── index.html            # Web UI (800+ lines)
│
├── feedback_data/
│   ├── interactions.jsonl    # User interaction log
│   └── preferences.json      # Computed preferences
│
├── tests/
│   ├── test_search.py        # Search functionality tests
│   ├── test_alternatives.py  # Alternative finding tests
│   ├── test_e2e.py           # End-to-end tests
│   └── benchmark.py          # Performance benchmarks
│
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

---

##  Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Vector Database** | Qdrant Cloud | Similarity search + filtering |
| **Text Embeddings** | all-mpnet-base-v2 | 768-dim semantic vectors |
| **Image Embeddings** | CLIP ViT-B-32 | 512-dim visual vectors |
| **Reranking** | ms-marco-MiniLM-L6-v2 | Cross-encoder scoring |
| **Backend** | FastAPI + Uvicorn | Async REST API |
| **Frontend** | Tailwind CSS + JS | Responsive web UI |
| **ML Framework** | PyTorch | GPU-accelerated inference |
| **Data Processing** | Pandas + NumPy | Dataset handling |

---

##  Future Enhancements

- [ ] **Price History**: Track price changes and alert on drops
- [ ] **Multi-Language**: Support queries in French, Arabic, Spanish
- [ ] **Voice Search**: "Hey FinVector, find me running shoes under fifty dollars"
- [ ] **Mobile App**: React Native companion app
- [ ] **Browser Extension**: Auto-detect products and show cheaper alternatives
- [ ] **Collaborative Filtering**: "Users who bought X also liked Y"

---

##  Team

<table>
  <tr>
    <td align="center">
      <b>Hamza Mhedhbi</b><br/>
      <a href="https://github.com/hamzaae1">GitHub</a> •
      <a href="https://linkedin.com/in/hamza-mhedhbi">LinkedIn</a>
    </td>
    <td align="center">
      <b>Hachem Mastouri</b><br/>
      <a href="https://github.com/hachemmastouri">GitHub</a> •
      <a href="https://linkedin.com/in/hachem-mastouri">LinkedIn</a>
    </td>
  </tr>
</table>

---


---

<p align="center">
  <b>FinVector</b> — Budget-First Shopping. Smarter Search. Better Value.<br/>
  <i>Built with ❤️ for the Qdrant Vector Search Hackathon 2026</i>
</p>
