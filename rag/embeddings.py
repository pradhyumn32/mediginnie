
# LangChain Embeddings Provider for Medijinn, which
# supports local SentenceTransformers and HuggingFace Inference API.

import os
import logging
from langchain_core.embeddings import Embeddings
from config.settings import settings

logger = logging.getLogger(__name__)

# Ensure HuggingFace Hub uses configured token to avoid unauthenticated rate warnings
if settings.HUGGINGFACEHUB_API_TOKEN:
    os.environ["HF_TOKEN"] = settings.HUGGINGFACEHUB_API_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.HUGGINGFACEHUB_API_TOKEN

_EMBEDDINGS_INSTANCE: Embeddings | None = None


def get_embeddings_model() -> Embeddings:
    """
    Returns a cached LangChain Embeddings instance configured
    for the medical knowledge base.
    """
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is not None:
        return _EMBEDDINGS_INSTANCE

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Initializing local HuggingFaceEmbeddings: %s", settings.EMBEDDING_MODEL_NAME)
        _EMBEDDINGS_INSTANCE = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return _EMBEDDINGS_INSTANCE
    except Exception as exc:
        logger.warning("Local HuggingFaceEmbeddings initialization failed: %s", exc)

        if settings.HUGGINGFACEHUB_API_TOKEN:
            try:
                from langchain_huggingface import HuggingFaceEndpointEmbeddings

                logger.info("Falling back to HuggingFaceEndpointEmbeddings: %s", settings.EMBEDDING_MODEL_NAME)
                _EMBEDDINGS_INSTANCE = HuggingFaceEndpointEmbeddings(
                    model=settings.EMBEDDING_MODEL_NAME,
                    huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN,
                )
                return _EMBEDDINGS_INSTANCE
            except Exception as endpoint_exc:
                logger.error("HuggingFaceEndpointEmbeddings also failed: %s", endpoint_exc)
                raise endpoint_exc

        raise exc
