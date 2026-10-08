# Ingestion Management Endpoints for Medijinn

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy import text
from sqlalchemy.orm import Session

from db.db import get_db
from api.schemas import IngestRequest, IngestResponse, DrugSummary
from preprocess.pipeline import ingest_drug

router = APIRouter(prefix="/ingest", tags=["Data Ingestion"])


@router.post("", response_model=IngestResponse)
def trigger_ingestion(request: IngestRequest):
    """
    Ingests official drug label data from OpenFDA, DailyMed, or auto-selected source,
    generates 768-dim embeddings, and stores chunks in pgvector.
    """
    try:
        res = ingest_drug(
            drug_name=request.drug_name,
            source_choice=request.source,
            max_records=request.max_records,
        )

        if res.get("status") == "not_found":
            return IngestResponse(
                status="not_found",
                drug=request.drug_name,
                source=request.source,
                chunks_ingested=0,
                error=f"No FDA or DailyMed records found for {request.drug_name}",
            )

        return IngestResponse(
            status="success",
            drug=res.get("drug", request.drug_name),
            source=res.get("source"),
            brands_found=res.get("brands_found", 0),
            chunks_ingested=res.get("chunks_ingested", 0),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed for {request.drug_name}: {str(exc)}",
        )


@router.get("/drugs", response_model=list[DrugSummary])
def list_indexed_drugs(db: Session = Depends(get_db)):
    """
    Lists all medications currently indexed in the pgvector database.
    """
    try:
        stmt = text("""
            SELECT 
                d.id::text, 
                d.generic_name, 
                COALESCE(d.brand_names, '[]'::jsonb),
                count(c.id) as chunk_count
            FROM drugs d
            LEFT JOIN medical_chunks c ON c.drug_id = d.id
            GROUP BY d.id, d.generic_name, d.brand_names
            ORDER BY d.generic_name ASC;
        """)
        rows = db.execute(stmt).fetchall()

        results = []
        for r in rows:
            results.append(
                DrugSummary(
                    id=r[0],
                    generic_name=r[1],
                    brand_names=r[2] or [],
                    chunk_count=r[3],
                )
            )
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to query drug list: {str(exc)}")
