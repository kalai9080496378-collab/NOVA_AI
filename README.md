# NOVA AI — AI Story-to-Video Platform

NOVA AI is a complete full-stack AI story-to-video generation and social platform.

## Architecture & Technology Stack
- **Backend API**: FastAPI + Uvicorn + Pydantic v2
- **ORM & Database**: SQLAlchemy 2.0 (Asyncio) + PostgreSQL
- **Migrations**: Alembic with full declarative schema versioning
- **Task Queue & Broker**: Celery + Redis
- **Security & Cryptography**: Bcrypt (passlib) password hashing, RFC 7519 JWT access tokens with revocation blocklist, sliding-window login rate limiting
- **Cost Protection**: Strict monthly user generation quota (default 3 videos/month) deducted atomically at job submission time
- **Video Generation**: RunAPI Veo 3.1 Fast integration (`runapi-veo-3-1`) with 2-stage retry resilience and immediate 7-day expiration asset ingestion, plus MockVideoProvider for local testing
- **Audio & Subtitles**: Modular TTS with phonetic Tamil script transliteration for Thanglish input, and SRT subtitle formatting with Romanized Thanglish caption preservation
- **Assembly Pipeline**: FFmpeg concat demuxer, audio ducking (background music ducked to 15% with `amix`), and subtitle burning
- **Social & Discovery**: Likes, threaded comments (max 500 chars), user follows, saved bookmarks, and cursor-based public feed pagination
- **Content Moderation & AI-Only Enforcement**: Pre-submission keyword scanning, post-generation moderation gate, and strict AI lineage verification for publishing

---

## Getting Started

### 1. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```
Key configuration settings:
- `DATABASE_URL`: `postgresql+asyncpg://user:pass@localhost:5432/nova_ai`
- `DATABASE_URL_SYNC`: `postgresql+psycopg://user:pass@localhost:5432/nova_ai`
- `REDIS_URL`: `redis://localhost:6379/0`
- `RUNAPI_API_KEY`: Your RunAPI API key
- `VIDEO_PROVIDER`: `runapi` (or `mock` for offline dev)

### 3. Database Migrations
Run the initial Alembic migration to create all tables and constraints:
```powershell
alembic upgrade head
```

### 4. Running the Application
```powershell
# Start FastAPI backend server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Start Celery worker in a separate terminal
celery -A src.workers.celery_app worker --loglevel=info
```

API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Liveness Probe: `http://localhost:8000/health`
- Readiness Probe: `http://localhost:8000/health/ready`

### 5. Running Automated Tests
```powershell
pytest -v
```
