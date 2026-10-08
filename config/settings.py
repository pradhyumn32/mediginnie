import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    DATABASE_URL: str = Field(
        default="postgresql://postgres:medjinn-password-270903@localhost:5432/medjinn-db",
        description="PostgreSQL connection string with pgvector support"
    )
    HUGGINGFACEHUB_API_TOKEN: str | None = Field(
        default=None,
        description="Hugging Face API token"
    )
    GROQ_API_KEY: str | None = Field(
        default=None,
        description="Free Groq API key (from console.groq.com)"
    )
    OPENAI_API_KEY: str | None = Field(
        default=None,
        description="OpenAI or OpenRouter API key"
    )
    OPENAI_BASE_URL: str | None = Field(
        default=None,
        description="Custom OpenAI-compatible base URL (e.g. Ollama, Groq, OpenRouter)"
    )
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434/v1",
        description="Local Ollama endpoint"
    )

    # Embeddings setup
    EMBEDDING_MODEL_NAME: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="Default 768-dimensional embedding model"
    )
    EMBEDDING_DIMENSION: int = Field(
        default=768,
        description="Vector dimension matching the DB schema"
    )

    # Chunking
    CHUNK_SIZE: int = Field(
        default=800,
        description="Target characters per chunk"
    )
    CHUNK_OVERLAP: int = Field(
        default=150,
        description="Overlap characters between chunks"
    )

    # Retrieval setup
    DENSE_TOP_K: int = Field(
        default=15,
        description="Top-k vector similarity chunks"
    )
    FTS_TOP_K: int = Field(
        default=15,
        description="Top-k full text search chunks"
    )
    RERANK_TOP_K: int = Field(
        default=5,
        description="Final top-k chunks after cross-encoder / fusion"
    )

    # LLM & Reasoning setup
    LLM_REPO_ID: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="Default medical LLM repository or model ID"
    )
    LLM_PROVIDER: str = Field(
        default="auto",
        description="LLM provider: auto, groq, huggingface, openai, ollama, or fallback"
    )
    LLM_TEMPERATURE: float = Field(
        default=0.1,
        description="Sampling temperature (low for grounded medical accuracy)"
    )
    LLM_MAX_NEW_TOKENS: int = Field(
        default=768,
        description="Max tokens generated per response"
    )


settings = Settings()
