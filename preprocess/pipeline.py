
# Batch Ingestion and Embedding Pipeline for Medijinn
# Multi-source pipeline: RxNorm (entity resolution) + OpenFDA & DailyMed (clinical labels)
# Generates section-aware chunks, LangChain embeddings, and stores in PostgreSQL pgvector.

import argparse
import logging
from typing import Any
from db.db import SessionLocal
from db.models import (
    Drug,
    MedicalDocument,
    MedicalChunk,
)
from preprocess.medSources.openfda import OpenFDASource
from preprocess.medSources.dailymed import DailyMedSource
from preprocess.medSources.rxnorm import RxNormSource
from preprocess.normaliser import normalize_fda_record
from preprocess.chunker import ClinicalChunker
from rag.embeddings import get_embeddings_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def ingest_drug(
    drug_name: str,
    source_choice: str = "auto",
    max_records: int = 1,
    batch_size: int = 32,
) -> dict[str, Any]:
    """
    Ingests drug label data from OpenFDA or DailyMed, enriched with RxNorm.
    Computes LangChain embeddings and inserts into pgvector.
    
    source_choice: 'auto' (tries OpenFDA, falls back to DailyMed),
                   'openfda' (only OpenFDA),
                   'dailymed' (only DailyMed)
    """
    raw_query = drug_name.strip()
    logger.info("Initiating ingestion for: %s (source: %s)", raw_query, source_choice)

    # 1. Entity resolution with RxNorm
    rxnorm = RxNormSource()
    resolved = rxnorm.resolve_drug_name(raw_query)
    canonical_generic = resolved["generic_name"]
    discovered_brands = resolved["brand_names"]
    logger.info("RxNorm resolved '%s' -> Generic: %s | Brands: %d", raw_query, canonical_generic, len(discovered_brands))

    # 2. Fetch label records from selected source
    records: list[dict[str, Any]] = []
    active_source = "openfda"

    if source_choice in ("openfda", "auto"):
        try:
            fda_client = OpenFDASource()
            data = fda_client.fetch_drug(canonical_generic)
            records = data.get("results", []) or []
            if records:
                active_source = "openfda"
        except Exception as fda_exc:
            logger.warning("OpenFDA query failed for %s: %s", canonical_generic, fda_exc)

    if not records and source_choice in ("dailymed", "auto"):
        logger.info("Fetching authoritative SPL label from DailyMed for %s...", canonical_generic)
        try:
            dailymed_client = DailyMedSource()
            dm_data = dailymed_client.fetch_drug(canonical_generic, limit=max_records)
            records = dm_data.get("results", []) or []
            if records:
                active_source = "dailymed"
        except Exception as dm_exc:
            logger.warning("DailyMed query failed for %s: %s", canonical_generic, dm_exc)

    if not records:
        logger.warning("No records found in any source for %s", raw_query)
        return {"status": "not_found", "drug": raw_query, "chunks_ingested": 0}

    selected_records = records[:max_records]
    chunker = ClinicalChunker()
    embeddings_model = get_embeddings_model()

    db = SessionLocal()
    total_chunks_ingested = 0

    try:
        for record in selected_records:
            if active_source == "openfda":
                normalized = normalize_fda_record(record)
            else:
                # DailyMed records are already structured by DailyMedSource
                normalized = record

            generic_norm = normalized.get("generic_name", canonical_generic)
            record_brands = normalized.get("brand_names", []) or []
            all_brands = sorted(list(set(discovered_brands + record_brands)))

            # Upsert Drug
            drug = (
                db.query(Drug)
                .filter(Drug.generic_name == generic_norm)
                .first()
            )

            if not drug:
                drug = Drug(
                    generic_name=generic_norm,
                    brand_names=all_brands,
                    openfda_data=normalized.get("openfda"),
                )
                db.add(drug)
                db.flush()
            else:
                existing_brands = set(drug.brand_names or [])
                merged_brands = sorted(list(existing_brands.union(all_brands)))
                drug.brand_names = merged_brands
                db.flush()

            # Store or replace MedicalDocument
            set_id = normalized.get("set_id")
            existing_doc = (
                db.query(MedicalDocument)
                .filter(
                    MedicalDocument.drug_id == drug.id,
                    MedicalDocument.source_id == set_id,
                )
                .first()
            )

            if existing_doc:
                db.query(MedicalChunk).filter(MedicalChunk.document_id == existing_doc.id).delete()
                document = existing_doc
                document.raw_content = record
                document.source = active_source
            else:
                document = MedicalDocument(
                    drug_id=drug.id,
                    source=active_source,
                    source_id=set_id,
                    title=f"{generic_norm} {active_source.upper()} Label",
                    source_url="https://dailymed.nlm.nih.gov" if active_source == "dailymed" else "https://open.fda.gov",
                    raw_content=record,
                )
                db.add(document)
                db.flush()

            # Create Chunks
            chunks = chunker.chunk_normalized_drug(normalized)
            if not chunks:
                logger.warning("No chunks extracted for %s from %s", generic_norm, active_source)
                continue

            logger.info("Generated %d chunks from %s for %s. Generating embeddings...", len(chunks), active_source, generic_norm)

            # Generate Embeddings in Batches
            chunk_texts = [c["content"] for c in chunks]
            all_embeddings: list[list[float]] = []

            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i : i + batch_size]
                batch_embeddings = embeddings_model.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)

            # Insert Chunks with pgvector embeddings
            for idx, chunk_data in enumerate(chunks):
                db_chunk = MedicalChunk(
                    document_id=document.id,
                    drug_id=drug.id,
                    section=chunk_data["section"],
                    content=chunk_data["content"],
                    embedding=all_embeddings[idx],
                    metadata_=chunk_data["metadata"],
                    chunk_index=chunk_data["chunk_index"],
                    source=active_source,
                )
                db.add(db_chunk)

            total_chunks_ingested += len(chunks)

        db.commit()
        logger.info(
            "Successfully ingested %s from %s: %d documents, %d total chunks embedded.",
            canonical_generic,
            active_source,
            len(selected_records),
            total_chunks_ingested,
        )

        return {
            "status": "success",
            "drug": canonical_generic,
            "source": active_source,
            "brands_found": len(all_brands),
            "chunks_ingested": total_chunks_ingested,
        }

    except Exception as exc:
        db.rollback()
        logger.exception("Failed to ingest drug %s: %s", raw_query, exc)
        raise exc
    finally:
        db.close()


def ingest_drug_catalog(drugs: list[str], source_choice: str = "auto") -> list[dict[str, Any]]:
    """
    Ingests a catalog of common drugs.
    """
    results = []
    for d in drugs:
        try:
            res = ingest_drug(d, source_choice=source_choice)
            results.append(res)
        except Exception as e:
            results.append({"status": "error", "drug": d, "error": str(e)})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-source ingestion into pgvector (OpenFDA + DailyMed + RxNorm)")
    parser.add_argument(
        "--drugs",
        nargs="+",
        default=["METFORMIN", "LISINOPRIL", "IBUPROFEN"],
        help="List of drug names (generic or brand) to ingest",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "openfda", "dailymed"],
        default="auto",
        help="Data source to use: auto, openfda, or dailymed",
    )
    args = parser.parse_args()

    logger.info("Starting ingestion: drugs=%s source=%s", args.drugs, args.source)
    ingest_drug_catalog(args.drugs, source_choice=args.source)