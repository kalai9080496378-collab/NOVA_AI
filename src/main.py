import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from src.api.v1 import api_router
from src.config import settings
from src.database import engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nova_ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up NOVA AI backend services...")
    # Optional startup verification
    yield
    logger.info("Shutting down NOVA AI backend services...")
    await engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Backend foundation and API services for NOVA AI story-to-video studio.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    tags=["System"],
    summary="Liveness check probe",
)
async def health_check():
    """Liveness probe reporting service status."""
    return {
        "status": "healthy",
        "service": "nova-ai-backend",
        "version": "1.0.0",
        "environment": settings.APP_ENV,
    }


@app.get(
    "/health/ready",
    status_code=status.HTTP_200_OK,
    tags=["System"],
    summary="Readiness probe checking database connectivity",
)
async def readiness_check():
    """Readiness probe checking database connectivity via SELECT 1."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {
            "status": "ready",
            "database": "connected",
        }
    except Exception as e:
        return {
            "status": "degraded",
            "database": "unreachable",
            "error": str(e),
        }
