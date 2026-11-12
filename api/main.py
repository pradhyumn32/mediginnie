# Production FastAPI Application for Medijinn

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import settings
from db.db import engine, Base
from rag.embeddings import get_embeddings_model
from api.routes import health, chat, ingest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SimpleRateLimiterMiddleware(BaseHTTPMiddleware):
    """In-memory sliding window rate limiter per client IP."""
    def __init__(self, app, max_requests: int = 60, window_secs: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_secs = window_secs
        self.client_records: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/static") or path in ("/", "/api/health"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "anonymous"
        now = time.time()
        timestamps = self.client_records[client_ip]

        # Prune expired timestamps
        self.client_records[client_ip] = [t for t in timestamps if now - t < self.window_secs]

        if len(self.client_records[client_ip]) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down."},
                headers={"Retry-After": str(self.window_secs)},
            )

        self.client_records[client_ip].append(now)
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: runs startup verifications and clean teardown.
    """
    logger.info("Initializing Medijinn Clinical AI Service...")

    # 1. Ensure database schema exists
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema validated.")
    except Exception as exc:
        logger.error("Failed to connect or validate DB schema: %s", exc)

    # 2. Warm up embeddings model
    try:
        emb = get_embeddings_model()
        logger.info("Embedding provider preheated successfully.")
    except Exception as exc:
        logger.warning("Embedding model preheating encountered an issue: %s", exc)

    logger.info("Medijinn API is ready to serve requests.")
    yield
    logger.info("Shutting down Medijinn service.")


app = FastAPI(
    title="Medijinn Clinical RAG & Triage API",
    description=(
        "Production-grade, grounded pharmaceutical AI assistant for drug side effects, "
        "adverse reactions, drug-drug interactions, and clinical safety triage."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS) with explicit origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Apply in-memory sliding window rate limiting
app.add_middleware(
    SimpleRateLimiterMiddleware,
    max_requests=settings.RATE_LIMIT_PER_MINUTE,
    window_secs=60,
)

# Register route modules under /api
app.include_router(health.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(ingest.router, prefix="/api")

# Mount Frontend UI and Static Assets
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", tags=["Frontend"])
def serve_frontend():
    """Serves the primary clinical chatbot web interface."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        content={
            "app": "Medijinn",
            "version": "1.0.0",
            "status": "online",
            "documentation": "/docs",
        }
    )
