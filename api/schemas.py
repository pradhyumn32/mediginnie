# API Request and Response Schemas for Medijinn

from typing import Any, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="The user's query about a medicine or symptoms")
    session_id: str | None = Field(default=None, description="Optional conversational session ID")
    chat_history: list[ChatMessage] | None = Field(default=None, description="Previous messages in the conversation")


class SourceCitation(BaseModel):
    drug: str | None = None
    section: str | None = None
    severity: str | None = None
    source: str | None = None
    chunk_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    triage_level: Literal["NORMAL", "HIGH_RISK", "EMERGENCY"]
    is_emergency: bool
    emergency_message: str | None = None
    extracted_drugs: list[str] = []
    canonical_drugs: list[str] = []
    extracted_symptoms: list[str] = []
    intent: str
    sources: list[SourceCitation] = []
    disclaimer: str


class IngestRequest(BaseModel):
    drug_name: str = Field(..., min_length=2, description="Generic or brand name of the drug to ingest")
    source: Literal["auto", "openfda", "dailymed"] = Field(default="auto", description="Source provider")
    max_records: int = Field(default=1, ge=1, le=5, description="Max label records to parse")


class IngestResponse(BaseModel):
    status: str
    drug: str
    source: str | None = None
    brands_found: int | None = None
    chunks_ingested: int
    error: str | None = None


class DrugSummary(BaseModel):
    id: str
    generic_name: str
    brand_names: list[str]
    chunk_count: int


class HealthResponse(BaseModel):
    status: str
    database: str
    total_drugs: int
    total_chunks: int
    embedding_dimension: int
    embedding_model: str
