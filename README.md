# Upskill Advisor Backend

FastAPI backend that recommends upskilling courses and generates concise 3-step learning plans based on a user's skills and target role. Retrieval is hybrid (BM25 keyword + optional vector semantics) and plan generation can use an LLM via OpenRouter or fall back to heuristics. Optional PDF plan export is supported.

## 🚀 Features

- Personalized course recommendations and 3-step learning plans
- Hybrid retrieval: BM25 keyword search with optional semantic vectors (local or Pinecone)
- LLM plan generation via OpenRouter (LangChain ChatOpenAI) with graceful local fallback
- Skill gap analysis and per-run metrics (coverage/diversity)
- Versioned API mounted under a configurable prefix (default `/api/v1`)
- Structured JSON logging with request IDs
- Optional response caching (transparent no-op if dependency missing)

## 🏗️ Architecture & Key Files

```
UpskillAdvisorBackend/
├── main.py                      # FastAPI app, mounts v1 router, CORS, health, crawl, performance
├── api/
│   └── v1/
│       └── routes.py            # Versioned endpoints (advise, courses search, metadata, demo, metrics)
├── core/
│   ├── config.py                # Settings loader (.env), feature flags, API prefix
│   └── logging_config.py        # JSON logging + request_id helpers
├── schemas/
│   ├── api.py                   # AdviseRequest/Result, ApiResponse[T], Demo persona models
│   └── course.py                # Course model (provider/url optional; metadata dict)
├── services/
│   ├── advisor_service.py       # Advise flow, gap map, LLM plan (OpenRouter), PDF export hook
│   ├── retriever.py             # BM25 + optional vector search and hybrid fusion
│   ├── course_manager.py        # Manage courses.json, search/filter, stats, recommend
│   ├── pdf_service.py           # PDF generation (ReportLab) with .txt fallback
│   ├── crawler_service.py       # Crawl courses from the web
│   ├── web_scraper.py           # Scraping utilities used by crawler
│   └── metrics_service.py       # Simple performance/metrics helpers
├── ingest.py                    # Ingest local courses.json; optional Pinecone upsert
├── ingest_jd.py                 # Job description ingestion (optional, configurable)
├── generate_metrics_report.py   # Collate historical metrics for the UI
├── requirements.txt             # Dependencies
├── test_main.http               # Example HTTP calls
├── tests/ and test_*.py         # Test suite
└── reports/                     # Generated PDF/TXT plans
```

## 🔧 Core Components

- API layer (api/v1/routes.py)
  - POST /advise returns ApiResponse[AdviseResult]
  - Rich course search/filtering endpoints
  - Demo persona and historical metrics aggregation
- Advisor service (services/advisor_service.py)
  - Hybrid candidate retrieval via services/retriever.py
  - Gap map, plan generation (LLM via OpenRouter or heuristic), PDF export
- Retriever (services/retriever.py)
  - BM25 keyword search; optional local embeddings; optional Pinecone
  - Hybrid fusion with simple score normalization
- Configuration (core/config.py)
  - .env-driven flags: API prefix, model paths, Pinecone, OpenRouter, feature toggles

## 📦 Data Model (schemas)

- Course
  - course_id: str, title: str, skills: List[str], difficulty: str, duration_weeks: int
  - provider?: str, url?: str, metadata: Dict[str, Any]
- AdviseRequest
  - profile: UserProfile { current_skills: List[{name, expertise}], goal_role: str, years_experience?: int }
  - user_context?: Dict[str, Any], search_online?: bool, retrieval_mode?: "vector|keyword|hybrid|hybrid_rerank", target_skills?: List[str], generate_pdf?: bool
- AdviseResult
  - plan: List[step], gap_map: Dict[str, List[str]], recommended_courses: List[Course], notes?: str, metrics?: Dict

## 🛠️ Technology Stack

- FastAPI (Python 3.9+ recommended)
- Pydantic v2
- rank-bm25 for keyword search
- Optional embeddings via sentence-transformers (with hashing fallback)
- Optional Pinecone vector DB
- LangChain ChatOpenAI pointed to OpenRouter (optional)
- ReportLab for PDF (optional; auto-falls back to .txt)

## ⚙️ Configuration (.env)

See core/config.py for full list. Common keys:

- ENVIRONMENT=dev
- API_V1_PREFIX=/api/v1
- COURSES_JSON=courses.json
- USE_PINECONE=false
- PINECONE_API_KEY=...
- PINECONE_INDEX=...
- EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
- HF_HOME=./models_cache
- USE_CROSS_ENCODER=true
- CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
- OPENROUTER_API_KEY=...
- OPENROUTER_API_BASE=https://openrouter.ai/api/v1
- OPENROUTER_MODEL=mistralai/mistral-7b-instruct
- USE_LOCAL_LLM=true
- ENABLE_JD_INGESTION=true
- JD_SOURCE_PATH=job_descriptions

## ▶️ Run locally

- Create venv and install deps:
  - python -m venv venv
  - venv\Scripts\activate  (Windows) or source venv/bin/activate (macOS/Linux)
  - pip install -r requirements.txt
- Start API:
  - uvicorn main:app --reload --host 0.0.0.0 --port 8000
- Open docs at http://localhost:8000/docs

The app also exposes:
- GET / → basic health JSON
- GET /crawl → trigger a lightweight crawl via services.crawler_service
- GET /performance → simple performance metrics

## 🌐 Endpoints (mounted under API_V1_PREFIX, default /api/v1)

- POST /advise
- POST /advise/compare
- POST /demo/persona
- GET  /metrics/reports
- GET  /courses/search
- POST /courses/search
- GET  /courses/{course_id}
- GET  /courses/stats
- GET  /courses/providers
- GET  /courses/categories
- GET  /courses/skills
- GET  /courses/difficulties

## 📥 Ingest and data files

- Local courses live in courses.json; backups are written to backups/
- Run python ingest.py to embed and optionally upsert to Pinecone
- Optional JD ingestion via python ingest_jd.py (see file header for modes)
- Generated plans (PDF or .txt) are saved under reports/

## 📝 Example request

Payload matches schemas/api.py exactly:

```
POST /api/v1/advise
{
  "profile": {
    "current_skills": [
      {"name": "Python", "expertise": "Intermediate"},
      {"name": "SQL", "expertise": "Beginner"}
    ],
    "goal_role": "Machine Learning Engineer",
    "years_experience": 2
  },
  "user_context": {"time_per_week": 6},
  "search_online": true,
  "retrieval_mode": "hybrid",
  "generate_pdf": true
}
```

Response is wrapped as ApiResponse with request_id, status, and data fields.

## 🧪 Tests

- pytest
- Example requests in test_main.http
- Additional scenarios in test_*.py (crawler, performance, duplicates, online recommendations)

## 📈 Metrics & reports

- generate_metrics_report.py parses runs into reports/ for the UI
- api/v1/routes.py exposes GET /metrics/reports to fetch parsed history

## 📄 License and support

MIT License. Open issues for bugs/questions and see /docs for interactive API.