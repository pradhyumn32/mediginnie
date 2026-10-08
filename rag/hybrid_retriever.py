"""
Production-grade Hybrid Retriever for Medijinn
Combines:
1. Dense Vector Search via pgvector HNSW cosine distance
2. Sparse Keyword & Trigram Search via PostgreSQL pg_trgm
3. Reciprocal Rank Fusion (RRF) for robust rank aggregation
4. LangChain BaseRetriever compliance
"""

from typing import Any, Optional
import uuid
from pydantic import Field
from sqlalchemy import func, select, or_
from sqlalchemy.orm import Session

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from db.db import SessionLocal
from db.models import Drug, MedicalChunk
from rag.embeddings import get_embeddings_model
from config.settings import settings


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining pgvector dense cosine search and
    Postgres trigram sparse search with Reciprocal Rank Fusion (RRF).
    """

    top_k: int = Field(default=settings.DENSE_TOP_K)
    rrf_k: int = Field(default=60)
    filter_drug_name: Optional[str] = Field(default=None)
    filter_section: Optional[str] = Field(default=None)

    def _dense_search(
        self,
        db: Session,
        query_vector: list[float],
        drug_id: Optional[uuid.UUID] = None,
        section: Optional[str] = None,
        limit: int = 20,
    ) -> list[tuple[MedicalChunk, float]]:
        """
        Cosine similarity search using pgvector index.
        """
        # cosine_distance is 0 for identical, 2 for opposite.
        # similarity = 1 - cosine_distance
        cosine_dist = MedicalChunk.embedding.cosine_distance(query_vector)

        stmt = select(
            MedicalChunk,
            cosine_dist.label("distance"),
        ).where(MedicalChunk.embedding.is_not(None))

        if drug_id:
            stmt = stmt.where(MedicalChunk.drug_id == drug_id)
        if section:
            stmt = stmt.where(MedicalChunk.section == section)

        stmt = stmt.order_by(cosine_dist.asc()).limit(limit)
        results = db.execute(stmt).all()

        return [(row[0], float(row[1])) for row in results]

    def _sparse_search(
        self,
        db: Session,
        query: str,
        drug_id: Optional[uuid.UUID] = None,
        section: Optional[str] = None,
        limit: int = 20,
    ) -> list[tuple[MedicalChunk, float]]:
        """
        Trigram similarity and keyword search in PostgreSQL.
        """
        # pg_trgm similarity function
        trgm_sim = func.similarity(MedicalChunk.content, query)

        # Break query into tokens for ILIKE fallback matches
        tokens = [t.strip() for t in query.split() if len(t.strip()) > 3]
        ilike_filters = [MedicalChunk.content.ilike(f"%{tok}%") for tok in tokens[:4]]

        stmt = select(
            MedicalChunk,
            trgm_sim.label("sim_score"),
        )

        conditions = []
        if ilike_filters:
            conditions.append(or_(*ilike_filters))
        conditions.append(trgm_sim > 0.05)

        stmt = stmt.where(or_(*conditions))

        if drug_id:
            stmt = stmt.where(MedicalChunk.drug_id == drug_id)
        if section:
            stmt = stmt.where(MedicalChunk.section == section)

        stmt = stmt.order_by(trgm_sim.desc()).limit(limit)
        results = db.execute(stmt).all()

        return [(row[0], float(row[1])) for row in results]

    def _find_drug_id(self, db: Session, drug_name: str) -> Optional[uuid.UUID]:
        """
        Lookup drug UUID by generic name or brand names.
        """
        clean_name = drug_name.strip().upper()
        # Direct generic match
        drug = db.query(Drug).filter(func.upper(Drug.generic_name) == clean_name).first()
        if drug:
            return drug.id

        # Substring / brand name match
        drug = db.query(Drug).filter(Drug.generic_name.ilike(f"%{clean_name}%")).first()
        if drug:
            return drug.id

        return None

    def retrieve_scored_chunks(
        self,
        query: str,
        drug_name: Optional[str] = None,
        section: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Executes hybrid search and merges results using Reciprocal Rank Fusion.
        Returns rich dictionary format with metadata and provenance.
        """
        k = top_k or self.top_k
        target_drug_name = drug_name or self.filter_drug_name
        target_section = section or self.filter_section

        # Generate query embedding
        embeddings_model = get_embeddings_model()
        query_vector = embeddings_model.embed_query(query)

        db = SessionLocal()
        try:
            drug_id: Optional[uuid.UUID] = None
            if target_drug_name:
                drug_id = self._find_drug_id(db, target_drug_name)

            # 1. Fetch dense candidates
            dense_results = self._dense_search(
                db=db,
                query_vector=query_vector,
                drug_id=drug_id,
                section=target_section,
                limit=k * 2,
            )

            # 2. Fetch sparse candidates
            sparse_results = self._sparse_search(
                db=db,
                query=query,
                drug_id=drug_id,
                section=target_section,
                limit=k * 2,
            )

            # 3. Reciprocal Rank Fusion
            rrf_scores: dict[uuid.UUID, float] = {}
            chunk_map: dict[uuid.UUID, MedicalChunk] = {}
            dense_ranks: dict[uuid.UUID, int] = {}
            sparse_ranks: dict[uuid.UUID, int] = {}

            for rank, (chunk, _dist) in enumerate(dense_results):
                chunk_id = chunk.id
                chunk_map[chunk_id] = chunk
                dense_ranks[chunk_id] = rank + 1
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))

            for rank, (chunk, _sim) in enumerate(sparse_results):
                chunk_id = chunk.id
                chunk_map[chunk_id] = chunk
                sparse_ranks[chunk_id] = rank + 1
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))

            # Sort by RRF score descending
            sorted_chunk_ids = sorted(
                rrf_scores.keys(),
                key=lambda cid: rrf_scores[cid],
                reverse=True,
            )

            final_results: list[dict[str, Any]] = []
            for cid in sorted_chunk_ids[:k]:
                ch = chunk_map[cid]
                meta = ch.metadata_ or {}
                final_results.append({
                    "id": str(ch.id),
                    "drug_id": str(ch.drug_id),
                    "generic_name": meta.get("generic_name", "UNKNOWN"),
                    "brand_names": meta.get("brand_names", []),
                    "section": ch.section,
                    "section_title": meta.get("section_title", ch.section.replace("_", " ").title()),
                    "severity": meta.get("severity", "INFO"),
                    "category": meta.get("category", "general"),
                    "content": ch.content,
                    "rrf_score": rrf_scores[cid],
                    "dense_rank": dense_ranks.get(cid),
                    "sparse_rank": sparse_ranks.get(cid),
                    "source": ch.source,
                })

            return final_results

        finally:
            db.close()

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> list[Document]:
        """
        LangChain BaseRetriever interface implementation.
        """
        scored_chunks = self.retrieve_scored_chunks(
            query=query,
            drug_name=self.filter_drug_name,
            section=self.filter_section,
            top_k=self.top_k,
        )

        documents: list[Document] = []
        for c in scored_chunks:
            doc = Document(
                page_content=c["content"],
                metadata={
                    "id": c["id"],
                    "drug_id": c["drug_id"],
                    "generic_name": c["generic_name"],
                    "brand_names": c["brand_names"],
                    "section": c["section"],
                    "section_title": c["section_title"],
                    "severity": c["severity"],
                    "category": c["category"],
                    "rrf_score": c["rrf_score"],
                    "source": c["source"],
                },
            )
            documents.append(doc)

        return documents
