# Python Q&A Assistant

A production-oriented Python question answering assistant built with FastAPI and RAG. It grounds answers in Stack Overflow Python Q&A data and routes between Gemini 2.0 Flash and HuggingFace Mistral-7B for resilient generation.

**Live Demo:** https://ai-native-learning.onrender.com  
**GitHub:** https://github.com/Raman909/AI-native-learning

---

## Architecture

```mermaid
flowchart LR
    U[User] --> A[POST /ask]
    A --> R[FAISS Retriever]
    R --> C[Context Builder]
    C --> L[LLMRouter]
    L --> G[Gemini 2.0 Flash]
    L --> H[HuggingFace Mistral-7B]
```

LLMRouter prefers Gemini 2.0 Flash first. If Gemini hits its free-tier quota or returns a quota-related error, the router automatically switches to HuggingFace Mistral-7B, then retries Gemini after one hour. No manual intervention needed — check `GET /api/v1/llm-status` to see which provider is active.

---

## Setup

### Prerequisites

- Python 3.11+
- Stack Overflow Python Q&A dataset (Questions.csv + Answers.csv) from [Kaggle](https://www.kaggle.com/datasets/stackoverflow/pythonquestions)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Raman909/AI-native-learning.git
cd AI-native-learning
```

**2. Create a virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**
```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

**5. Add the dataset**

Download `Questions.csv` and `Answers.csv` from Kaggle and place them in the `data/` folder.

**6. Build the vector index**
```bash
python scripts/build_index.py --data-dir data --output-dir vector_store --limit 10000
```

**7. Start the server**
```bash
uvicorn app.main:app --reload --port 8000
```

API available at `http://localhost:8000`  
Interactive Swagger docs at `http://localhost:8000/docs`

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```env
GEMINI_API_KEY=your_gemini_api_key_here     # https://aistudio.google.com/app/apikey
GEMINI_MODEL=gemini-2.0-flash
HF_API_KEY=your_huggingface_token_here       # https://huggingface.co/settings/tokens
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3
VECTOR_STORE_PATH=vector_store
```

---

## API Reference

Base URL (production): `https://ai-native-learning.onrender.com`  
Base URL (local): `http://localhost:8000`

---

### `GET /`
Landing page — returns the React frontend UI.

---

### `GET /api/v1/health`
Health check. Returns current status and which LLM provider is active.

**Request**
```bash
curl https://ai-native-learning.onrender.com/api/v1/health
```

**Response**
```json
{
  "status": "healthy",
  "llm_status": {
    "current_provider": "gemini",
    "gemini_available": true,
    "fallback_available": true
  },
  "vector_store": "loaded"
}
```

---

### `GET /api/v1/llm-status`
Returns detailed LLM router state — which provider is active and whether fallback is available.

**Request**
```bash
curl https://ai-native-learning.onrender.com/api/v1/llm-status
```

**Response**
```json
{
  "current_provider": "gemini",
  "gemini_available": true,
  "fallback_available": true
}
```

---

### `GET /api/v1/examples`
Returns 5 sample Python questions you can use to test the `/ask` endpoint.

**Request**
```bash
curl https://ai-native-learning.onrender.com/api/v1/examples
```

**Response**
```json
[
  "How do I reverse a list in Python?",
  "How to handle missing values in a pandas DataFrame?",
  "What causes RecursionError in Python and how do I fix it?",
  "How do I implement multiple inheritance in Python?",
  "What is the fastest way to read a large CSV file in Python?"
]
```

---

### `POST /api/v1/ask`
Main endpoint. Accepts a Python question and returns a grounded answer retrieved from Stack Overflow data.

**Request**
```bash
curl -X POST https://ai-native-learning.onrender.com/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I reverse a list in Python?", "max_sources": 3}'
```

**Request Body**
```json
{
  "question": "How do I reverse a list in Python?",
  "max_sources": 3
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | string | Yes | A Python-related question |
| `max_sources` | integer | No | Number of source documents to retrieve (default: 3) |

**Response**
```json
{
  "question": "How do I reverse a list in Python?",
  "answer": "You can reverse a list using my_list.reverse() which modifies in-place, or my_list[::-1] which returns a new list...",
  "sources": [
    {
      "content": "def reversed_lines(file): ..."
    }
  ],
  "model": "gemini-2.0-flash",
  "provider_used": "gemini",
  "processing_time_ms": 3915.6
}
```

**Error Responses**

| Status | Description |
|---|---|
| `422` | Question too short, too long, or missing |
| `503` | Both Gemini and HuggingFace providers unavailable |

---

## Testing

```bash
pytest tests/ -v
```

To execute the evaluation notebook:
```bash
jupyter nbconvert --to notebook --execute notebooks/test_results.ipynb \
  --output test_results.ipynb --output-dir notebooks
```

---

## Deployment

Deployed on **Render** using the included `Dockerfile`.

**Live URL:** https://ai-native-learning.onrender.com

To deploy your own instance:
1. Fork this repo
2. Connect to [Render](https://render.com) → New Web Service → Select repo
3. Set environment variables in Render dashboard (same as `.env.example`)
4. Set Docker Command: `uvicorn app.main:app --host 0.0.0.0 --port 10000`
5. Deploy — Render auto-detects the Dockerfile

---

## Scaling to 100+ Concurrent Users

| Layer | Current | At Scale |
|---|---|---|
| Workers | 1 uvicorn process | `uvicorn --workers 4` + Gunicorn |
| Caching | None | Redis with 1h TTL (>40% cache hit rate) |
| Vector Store | FAISS (in-process) | Pinecone / Weaviate (distributed) |
| LLM | Single Gemini key | Pool of 3+ API keys (round-robin) |
| Fallback | HuggingFace Mistral | Groq API (10× faster free tier) |

**Estimated cost at 100k queries/month:** ~$7/mo (Render Starter) + $0 Gemini free tier.
