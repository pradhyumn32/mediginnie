# Production FastAPI Application for Medijinn

import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from db.db import engine, Base
from rag.embeddings import get_embeddings_model
from api.routes import health, chat, ingest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
