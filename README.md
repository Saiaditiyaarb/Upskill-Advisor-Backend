# Upskill Advisor Backend

A FastAPI-based recommendation system that provides personalized learning paths and course recommendations based on user skills and target goals. The system uses hybrid retrieval combining semantic search (Pinecone) and keyword search (BM25) with optional LLM-powered plan generation.

## 🚀 Features

- **Personalized Course Recommendations**: Get tailored course suggestions based on your current skills and learning goals
- **Hybrid Search**: Combines semantic vector search (Pinecone) with keyword-based search (BM25) for optimal results
- **Learning Path Generation**: Automatically generates step-by-step learning plans using heuristics or LLM (OpenAI GPT)
- **Skill Gap Analysis**: Identifies missing skills and maps them to recommended courses
- **Future-Proof Architecture**: Designed with extensibility and backward compatibility in mind
- **Structured Logging**: JSON-based logging with request ID tracking for observability
- **Caching Support**: Built-in response caching for improved performance

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  Advisor Service │    │   Retriever     │
│                 │    │                 │    │                 │
│ • API Routes    │───▶│ • Plan Generation│───▶│ • Hybrid Search │
│ • Middleware    │    │ • Skill Analysis │    │ • BM25 + Vector │
│ • Validation    │    │ • Course Ranking │    │ • Pinecone      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Schemas   │    │   Core Config   │    │  Data Ingestion │
│                 │    │                 │    │                 │
│ • Request/Response│    │ • Settings     │    │ • Course Loading│
│ • Generic Wrapper│    │ • Environment  │    │ • Embeddings    │
│ • Type Safety   │    │ • Logging      │    │ • Vector Store  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Directory Structure

```
UpskillAdvisorBackend/
├── main.py                 # FastAPI application entry point
├── ingest.py              # Data ingestion script for courses
├── requirements.txt       # Python dependencies
├── test_main.http        # HTTP test requests
├── api/                  # API layer
│   └── v1/
│       └── routes.py     # Versioned API endpoints
├── core/                 # Core configuration and utilities
│   ├── config.py         # Environment settings management
│   └── logging_config.py # Structured logging setup
├── schemas/              # Pydantic data models
│   ├── api.py           # API request/response schemas
│   └── course.py        # Course data model
├── services/            # Business logic layer
│   ├── advisor_service.py # Main recommendation logic
│   └── retriever.py     # Hybrid search implementation
└── tests/               # Test suite
    ├── test_advisor_service.py
    └── test_routes.py
```

## 🔧 Core Components

### 1. API Layer (`api/v1/routes.py`)
- **Versioned Endpoints**: Future-proof API versioning with `/api/v1` prefix
- **Generic Response Wrapper**: Consistent response format with `request_id`, `status`, and `data`
- **Request Validation**: Pydantic-based input validation and type safety
- **Caching**: Optional response caching with configurable TTL

### 2. Advisor Service (`services/advisor_service.py`)
- **Hybrid Recommendation**: Combines retrieval results with skill overlap scoring
- **Learning Path Generation**:
  - Primary: LLM-powered plans using LangChain + OpenAI GPT
  - Fallback: Heuristic-based planning when LLM unavailable
- **Skill Gap Analysis**: Maps target skills to missing prerequisites
- **Flexible Scoring**: Extensible scoring system for course ranking

### 3. Retriever (`services/retriever.py`)
- **Hybrid Search**: Combines semantic and keyword-based retrieval
- **Vector Search**: Pinecone integration for semantic similarity
- **Keyword Search**: BM25 algorithm for exact term matching
- **Graceful Degradation**: Falls back to local JSON when external services unavailable
- **Async Interface**: Non-blocking operations for FastAPI compatibility

### 4. Data Ingestion (`ingest.py`)
- **Course Validation**: Pydantic-based schema validation
- **Embedding Generation**:
  - Primary: SentenceTransformers for semantic embeddings
  - Fallback: Deterministic hashing when model unavailable
- **Vector Storage**: Batch upserts to Pinecone with deduplication
- **Re-runnable**: Idempotent operations using course IDs

### 5. Configuration (`core/config.py`)
- **Environment Management**: Centralized settings with `.env` support
- **Service Discovery**: Optional external service configuration
- **Future-Proof**: Easy addition of new configuration parameters
- **Caching**: LRU cached settings for performance

## 📊 Data Models

### Course Schema
```python
class Course(BaseModel):
    course_id: str
    title: str
    skills: List[str]
    difficulty: str  # beginner, intermediate, advanced
    duration_weeks: int
    provider: Optional[str]
    url: Optional[str]
    metadata: Dict[str, Any]  # Extensible for future fields
```

### API Request/Response
```python
class AdviseRequest(BaseModel):
    profile: UserProfile
    user_context: Optional[Dict[str, Any]]

class UserProfile(BaseModel):
    skills: List[str]
    target_skills: List[str]
    years_experience: Optional[int]

class AdviseResult(BaseModel):
    plan: List[Dict[str, Any]]
    gap_map: Dict[str, List[str]]
    recommended_courses: List[Course]
    notes: Optional[str]
```

## 🛠️ Technology Stack

- **Framework**: FastAPI (async web framework)
- **Language**: Python 3.8+
- **Data Validation**: Pydantic v2
- **Vector Database**: Pinecone (optional)
- **Search**: BM25 (rank-bm25)
- **Embeddings**: SentenceTransformers
- **LLM Integration**: LangChain + OpenAI GPT
- **Testing**: pytest + pytest-asyncio
- **Logging**: Structured JSON logging
- **Caching**: fastapi-cache2

## 🚦 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Optional: Pinecone account for vector search
- Optional: OpenAI API key for LLM-powered planning

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd UpskillAdvisorBackend
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment Configuration** (Optional)
Create a `.env` file in the project root:
```env
# Optional: Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=your_index_name

# Optional: OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Optional: Custom Settings
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
API_V1_PREFIX=/api/v1
ENVIRONMENT=dev
```

### Running the Application

1. **Start the FastAPI server**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. **Access the API**
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/
- API Endpoint: http://localhost:8000/api/v1/advise

### Data Ingestion (Optional)

If you have course data to ingest:

1. **Prepare course data** (`courses.json`)
```json
[
  {
    "course_id": "python-101",
    "title": "Python Programming Fundamentals",
    "skills": ["python", "programming", "basics"],
    "difficulty": "beginner",
    "duration_weeks": 6,
    "provider": "TechEdu",
    "url": "https://example.com/python-101"
  }
]
```

2. **Run ingestion**
```bash
python ingest.py
```

## 📝 API Usage

### Making a Recommendation Request

```bash
curl -X POST "http://localhost:8000/api/v1/advise" \
  -H "Content-Type: application/json" \
  -d '{
    "profile": {
      "skills": ["python", "sql"],
      "target_skills": ["machine learning", "data science"],
      "years_experience": 2
    },
    "user_context": {
      "time_per_week": 10,
      "preferred_difficulty": "intermediate"
    }
  }'
```

### Response Format

```json
{
  "request_id": "uuid-string",
  "status": "ok",
  "data": {
    "plan": [
      {
        "skill": "machine learning",
        "action": "learn",
        "resource": "ML Fundamentals Course",
        "course_id": "ml-101"
      }
    ],
    "gap_map": {
      "machine learning": [],
      "data science": []
    },
    "recommended_courses": [
      {
        "course_id": "ml-101",
        "title": "Machine Learning Fundamentals",
        "skills": ["machine learning", "python"],
        "difficulty": "intermediate",
        "duration_weeks": 8
      }
    ],
    "notes": "Plan generated via LLM based on retrieved courses and user profile."
  }
}
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_advisor_service.py -v
```

## 🔍 Monitoring & Observability

### Structured Logging
The application uses JSON-structured logging with request ID tracking:

```json
{
  "level": "INFO",
  "logger": "advisor",
  "message": "advisor_completed",
  "request_id": "uuid-string",
  "top_k": 5,
  "retrieved": 10,
  "selected": 3,
  "missing_skills": ["machine learning"]
}
```

### Health Checks
- Basic health: `GET /`
- API health: `GET /api/v1/advise` (with valid payload)

## 🔧 Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ENVIRONMENT` | `dev` | Application environment |
| `PINECONE_API_KEY` | `None` | Pinecone API key (optional) |
| `PINECONE_INDEX` | `None` | Pinecone index name (optional) |
| `OPENAI_API_KEY` | `None` | OpenAI API key (optional) |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `API_V1_PREFIX` | `/api/v1` | API version prefix |
| `COURSES_JSON` | `courses.json` | Path to course data file |

## 🚀 Deployment Considerations

### Production Checklist
- [ ] Set `ENVIRONMENT=production`
- [ ] Configure proper CORS origins
- [ ] Set up external vector database (Pinecone)
- [ ] Configure LLM API keys
- [ ] Set up monitoring and alerting
- [ ] Configure proper logging aggregation
- [ ] Set up load balancing
- [ ] Configure caching layer (Redis)

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔮 Future Enhancements

### Planned Features
- **User Authentication**: JWT-based user management
- **Course Ratings**: User feedback and rating system
- **Advanced Analytics**: Learning progress tracking
- **Multi-language Support**: Internationalization
- **Real-time Updates**: WebSocket-based notifications
- **A/B Testing**: Recommendation algorithm experimentation

### Architecture Improvements
- **Microservices**: Split into domain-specific services
- **Event Sourcing**: Track user interactions and learning progress
- **GraphQL**: More flexible API queries
- **Kubernetes**: Container orchestration for scalability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write tests for new features
- Update documentation for API changes
- Use structured logging for observability

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the test files for usage examples

---

**Built with ❤️ using FastAPI and modern Python practices**