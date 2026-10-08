import uuid

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pgvector.sqlalchemy import Vector

from db.db import Base


EMBEDDING_DIMENSION = 768


class Drug(Base):
    __tablename__ = "drugs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    generic_name: Mapped[str] = mapped_column(
        String,
        index=True,
        nullable=False,
    )

    brand_names: Mapped[list | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    openfda_data: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    documents: Mapped[list["MedicalDocument"]] = relationship(
        "MedicalDocument",
        back_populates="drug",
        cascade="all, delete-orphan",
    )

    chunks: Mapped[list["MedicalChunk"]] = relationship(
        "MedicalChunk",
        back_populates="drug",
        cascade="all, delete-orphan",
    )


class MedicalDocument(Base):
    __tablename__ = "medical_documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    drug_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("drugs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    source: Mapped[str] = mapped_column(
        String,
        index=True,
        default="openfda",
    )

    source_id: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
        index=True,
    )

    title: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    source_url: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    raw_content: Mapped[dict] = mapped_column(
        JSONB,
    )

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    drug: Mapped["Drug"] = relationship(
        "Drug",
        back_populates="documents",
    )

    chunks: Mapped[list["MedicalChunk"]] = relationship(
        "MedicalChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class MedicalChunk(Base):
    __tablename__ = "medical_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("medical_documents.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    drug_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("drugs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    section: Mapped[str] = mapped_column(
        String,
        index=True,
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    embedding: Mapped[list | None] = mapped_column(
        Vector(EMBEDDING_DIMENSION),
        nullable=True,
    )

    metadata_: Mapped[dict | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
    )

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    source: Mapped[str] = mapped_column(
        String,
        index=True,
        default="openfda",
    )

    drug: Mapped["Drug"] = relationship(
        "Drug",
        back_populates="chunks",
    )

    document: Mapped["MedicalDocument"] = relationship(
        "MedicalDocument",
        back_populates="chunks",
    )

    __table_args__ = (
        Index(
            "ix_medical_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index(
            "ix_medical_chunks_content_trgm",
            "content",
            postgresql_using="gin",
            postgresql_ops={"content": "gin_trgm_ops"},
        ),
        Index(
            "ix_medical_chunks_drug_section",
            "drug_id",
            "section",
        ),
    )