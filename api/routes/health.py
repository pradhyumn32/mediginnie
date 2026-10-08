# Health and Diagnostic Endpoints for Medijinn

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from db.db import get_db
from api.schemas import HealthResponse
from config.settings import settings

router = APIRouter(prefix="/health", tags=["Health & Diagnostics"])


@router.get("", response_model=HealthResponse)
def check_health(db: Session = Depends(get_db)):
    """
    Checks database connectivity, pgvector readiness, and returns
    knowledge base statistics.
    """
    try:
        # Test DB connection
        db.execute(text("SELECT 1")).scalar()

        total_drugs = db.execute(text("SELECT count(*) FROM drugs")).scalar() or 0
        total_chunks = db.execute(
            text("SELECT count(*) FROM medical_chunks WHERE embedding IS NOT NULL")
        ).scalar() or 0

        return HealthResponse(
            status="healthy",
            database="connected",
            total_drugs=total_drugs,
            total_chunks=total_chunks,
            embedding_dimension=settings.EMBEDDING_DIMENSION,
            embedding_model=settings.EMBEDDING_MODEL_NAME,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database or service unhealthy: {str(exc)}",
        )
