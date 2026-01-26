# FinVector - Budget-First Smart Shopping

<<<<<<< HEAD
**Team Cosine** | Qdrant Vector Search Hackathon 2026
=======
**Team Cosine** | Qdrant Vector Search Hackathon 2025
>>>>>>> e03f30945d5cb710e2e0f26cb61b32b5df2e70e9
**Team Members:** Hamza Mhedhbi & Hachem Mastouri

---

## The Problem

Traditional e-commerce search shows you the "best" products first, regardless of your budget. You search for "wireless headphones," fall in love with a $300 pair, then realize it's way over budget. Frustrating.

## Our Solution

**FinVector** is a budget-aware semantic search engine. It understands your budget constraint from the start and *never shows products you can't afford*. It also understands natural language:

- `"running shoes under $80"` → Extracts budget automatically
- `"cheap but good laptop"` → Understands you want value + quality
- `"wireless headphones"` → Expands to include "bluetooth earbuds"

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Smart Search** | Natural language queries with budget/intent extraction |
| **Visual Search** | Upload an image to find similar products |
| **Find Alternatives** | Discover cheaper alternatives to any product |
| **Budget Stretcher** | Get alternative + accessories for the same price |
| **Reranking** | Cross-encoder + MMR diversity for better results |
| **Semantic Cache** | Similar queries share cached results |

---

## Architecture

```
User Query: "wireless headphones under $50"
                    │
                    ▼
        ┌───────────────────────┐
        │  Query Understanding  │
        │  • Budget: $50        │
        │  • Intent: budget     │
        │  • Expand: +bluetooth │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Embedding Generation │
        │  all-mpnet-base-v2    │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │      Qdrant Cloud     │
        │  • Vector similarity  │
        │  • Filter: price ≤$50 │
        │  • Filter: rating ≥3★ │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │      Reranking        │
        │  • Cross-encoder      │
        │  • Rating boost       │
        │  • MMR diversity      │
        └───────────┬───────────┘
                    │
                    ▼
        Results: Best matches within budget
```

## Qdrant Integration

Two collections power the search:

| Collection | Model | Dimensions | Purpose |
|------------|-------|------------|---------|
| `products_text` | all-mpnet-base-v2 | 768 | Semantic text search |
| `products_images` | CLIP ViT-B-32 | 512 | Visual similarity search |

Qdrant handles:
- Vector similarity search with cosine distance
- **Filtered search** - budget and rating filters at query time
- Payload storage for product metadata

---

## Quick Start

```bash
# 1. Clone and setup
<<<<<<< HEAD
git clone https://github.com/hamzamhedhbi/FinVector.git
=======
git clone <repository>
>>>>>>> e03f30945d5cb710e2e0f26cb61b32b5df2e70e9
cd FinVector
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure .env
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-api-key

# 3. Ingest data
<<<<<<< HEAD
python ingest_amazon.py --amazon_products_path 
=======
python ingest_amazon.py
>>>>>>> e03f30945d5cb710e2e0f26cb61b32b5df2e70e9

# 4. Run
uvicorn api:app --reload --port 8000
```

**Open:** http://localhost:8000/static/index.html

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /search/smart` | Natural language search with query understanding |
| `POST /search` | Basic semantic search |
| `POST /search/image/upload` | Search by image |
| `POST /search/hybrid/upload` | Image + text combined search |
| `POST /alternatives` | Find cheaper alternatives |
| `POST /budget-stretcher` | Alternative + complementary products |
| `GET /categories` | List all categories |

### Example Request

```bash
curl -X POST http://localhost:8000/search/smart \
  -H "Content-Type: application/json" \
  -d '{"query": "wireless headphones under $50", "limit": 10}'
```

---

## Tech Stack

- **Vector DB:** Qdrant Cloud
- **Embeddings:** Sentence Transformers, CLIP
- **Reranking:** Cross-encoder (ms-marco-MiniLM)
- **Backend:** FastAPI
- **Frontend:** HTML/Tailwind CSS

---

## Project Structure

```
FinVector/
├── api.py                 # FastAPI endpoints
├── finvector_core.py      # Core search engine
├── query_understanding.py # Query parsing & expansion
├── reranker.py            # Cross-encoder & MMR
├── ingest_amazon.py       # Data pipeline
├── static/index.html      # Web UI
└── requirements.txt
```

---

## Team

<<<<<<< HEAD
- **Hamza Mhedhbi** 
- **Hachem Mastouri** 
=======
- **Hamza Mhedhbi** - 
- **Hachem Mastouri** -
>>>>>>> e03f30945d5cb710e2e0f26cb61b32b5df2e70e9

---

<p align="center"><b>Budget-First Shopping. Smarter Search. Better Value.</b></p>
