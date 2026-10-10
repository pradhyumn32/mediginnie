# Resilient Multi-Provider LLM Manager for Medijinn
# Supports Groq, OpenAI, Ollama, HuggingFace, and Local Extractive Fallback

import json
import logging
import re
from typing import Any
import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from config.settings import settings

logger = logging.getLogger(__name__)


class ClinicalFallbackChatModel(BaseChatModel):
    """
    100% offline, zero-cost clinical extraction model.
    Parses retrieved pgvector chunks and structures them into
    grounded clinical sections without external API dependencies.
    """

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_message = messages[-1].content if messages else ""

        # Extract drug name and inquiry
        drug_match = re.search(r"Identified Drug\(s\):\s*([^\n]+)", last_message)
        drug_name = drug_match.group(1).strip() if drug_match else "Medication"

        # Extract context
        context_str = ""
        if "Retrieved Official Drug Labeling Context:" in last_message:
            parts = last_message.split("Retrieved Official Drug Labeling Context:")
            if len(parts) > 1:
                context_str = parts[1].split("---")[1].strip() if "---" in parts[1] else parts[1]

        # Break into chunks
        raw_chunks = re.split(r"\[Chunk\s+\d+\s*\|", context_str)
        
        boxed_warnings = []
        adverse_reactions = []
        interactions = []
        contraindications = []
        general_notes = []

        for ch in raw_chunks:
            if not ch.strip():
                continue
            ch_clean = ch.strip()

            # Categorize chunk
            if "Boxed Warning" in ch_clean or "Severity: CRITICAL" in ch_clean:
                content = re.sub(r"^[^\]]*\]\s*", "", ch_clean).strip()
                boxed_warnings.append(content)
            elif "Adverse Reactions" in ch_clean or "Severity: SIDE_EFFECT" in ch_clean:
                content = re.sub(r"^[^\]]*\]\s*", "", ch_clean).strip()
                adverse_reactions.append(content)
            elif "Drug Interactions" in ch_clean or "Severity: INTERACTION" in ch_clean:
                content = re.sub(r"^[^\]]*\]\s*", "", ch_clean).strip()
                interactions.append(content)
            elif "Contraindications" in ch_clean or "Severity: HIGH" in ch_clean:
                content = re.sub(r"^[^\]]*\]\s*", "", ch_clean).strip()
                contraindications.append(content)
            else:
                content = re.sub(r"^[^\]]*\]\s*", "", ch_clean).strip()
                general_notes.append(content)
        # If context is empty or no valid chunks were found, strictly refuse
        if not context_str.strip() or not (boxed_warnings or adverse_reactions or interactions or contraindications or general_notes):
            response_text = (
                f"I do not have official FDA or DailyMed labeling information for **{drug_name}** in the database. "
                "I cannot provide medical information, side effects, or clinical guidance for medications not present "
                "in the official verified records. Please consult a qualified clinician or pharmacist."
            )
            generation = ChatGeneration(message=AIMessage(content=response_text))
            return ChatResult(generations=[generation])

        response_text = f"### Official Clinical Assessment for {drug_name}\n\n"
        response_text += "Synthesized directly from official FDA & DailyMed Structured Product Labeling (SPL):\n\n"

        if boxed_warnings:
            response_text += "#### 🚨 Boxed Warning (Highest Clinical Alert)\n"
            for bw in boxed_warnings[:2]:
                first_lines = [l.strip() for l in bw.split("\n") if l.strip() and not l.startswith("Drug:")]
                summary = " ".join(first_lines[:2])
                response_text += f"- **Boxed Alert**: {summary[:350]}... `[Source: FDA/DailyMed Label]`\n"
            response_text += "\n"

        if contraindications:
            response_text += "#### ⛔ Contraindications (When NOT to Take)\n"
            for ci in contraindications[:2]:
                first_lines = [l.strip() for l in ci.split("\n") if l.strip() and not l.startswith("Drug:")]
                summary = " ".join(first_lines[:2])
                response_text += f"- {summary[:300]} `[Source: FDA Label: Contraindications]`\n"
            response_text += "\n"

        if adverse_reactions:
            response_text += "#### ⚠️ Documented Adverse Reactions & Side Effects\n"
            for ar in adverse_reactions[:3]:
                first_lines = [l.strip() for l in ar.split("\n") if l.strip() and not l.startswith("Drug:")]
                summary = " ".join(first_lines[:2])
                response_text += f"- {summary[:320]}... `[Source: FDA/DailyMed: Adverse Reactions]`\n"
            response_text += "\n"

        if interactions:
            response_text += "#### 🔄 Notable Drug-Drug Interactions\n"
            for it in interactions[:2]:
                first_lines = [l.strip() for l in it.split("\n") if l.strip() and not l.startswith("Drug:")]
                summary = " ".join(first_lines[:2])
                response_text += f"- {summary[:320]}... `[Source: FDA/DailyMed: Drug Interactions]`\n"
            response_text += "\n"

        if not (boxed_warnings or adverse_reactions or interactions or contraindications) and general_notes:
            response_text += "#### Clinical Findings\n"
            for gn in general_notes[:2]:
                first_lines = [l.strip() for l in gn.split("\n") if l.strip() and not l.startswith("Drug:")]
                response_text += f"- {' '.join(first_lines[:2])[:300]}...\n"
            response_text += "\n"

        response_text += "#### 🩺 Discussion with Your Clinician\n"
        response_text += f"- Inform your prescribing healthcare provider about any unusual reactions to **{drug_name}**.\n"
        response_text += "- Never alter, discontinue, or double your prescribed medication dose without direct medical guidance."

        generation = ChatGeneration(message=AIMessage(content=response_text))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "clinical_fallback"


class OpenAICompatibleChatModel(BaseChatModel):
    """
    Universal client for Groq, Ollama, OpenRouter, and OpenAI.
    """

    api_url: str
    api_key: str
    model_name: str
    timeout: int = 25

    def __init__(self, api_url: str, api_key: str, model_name: str, timeout: int = 25):
        super().__init__(
            api_url=api_url,
            api_key=api_key,
            model_name=model_name,
            timeout=timeout,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Format messages
        payload_messages = []
        for m in messages:
            role = "user"
            if m.type == "system":
                role = "system"
            elif m.type in ("ai", "assistant"):
                role = "assistant"
            payload_messages.append({"role": role, "content": m.content})

        payload = {
            "model": self.model_name,
            "messages": payload_messages,
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_NEW_TOKENS,
        }

        resp = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "openai_compatible"


class ResilientChatModel(BaseChatModel):
    """
    Wraps primary chat model with seamless fallback to ClinicalFallbackChatModel.
    """

    primary_model: BaseChatModel | None = None
    fallback_model: BaseChatModel

    def __init__(self, primary_model: BaseChatModel | None = None):
        super().__init__(
            primary_model=primary_model,
            fallback_model=ClinicalFallbackChatModel(),
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.primary_model is not None:
            try:
                return self.primary_model._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as exc:
                logger.warning("Primary ChatModel error: %s. Seamlessly switching to ClinicalFallbackChatModel.", exc)

        return self.fallback_model._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "resilient_clinical_chat"


_CHAT_MODEL_INSTANCE: BaseChatModel | None = None


def get_chat_model() -> BaseChatModel:
    """
    Returns configured LangChain ChatModel.
    Auto-detects Groq, OpenAI, Ollama, HuggingFace, or Clinical Fallback.
    """
    global _CHAT_MODEL_INSTANCE
    if _CHAT_MODEL_INSTANCE is not None:
        return _CHAT_MODEL_INSTANCE

    primary: BaseChatModel | None = None

    # 1. Check for Groq (Fastest & Free Tier)
    if settings.GROQ_API_KEY:
        groq_key = settings.GROQ_API_KEY.strip("\"'")
        groq_model = "qwen/qwen3.8-27b"
        logger.info("Initializing Groq ChatModel (%s)...", groq_model)
        primary = OpenAICompatibleChatModel(
            api_url="https://api.groq.com/openai/v1/chat/completions",
            api_key=groq_key,
            model_name=groq_model,
        )
    # 2. Check for OpenAI / Custom Endpoint
    elif settings.OPENAI_API_KEY:
        api_url = settings.OPENAI_BASE_URL or "https://api.openai.com/v1/chat/completions"
        logger.info("Initializing OpenAI-compatible ChatModel at %s...", api_url)
        primary = OpenAICompatibleChatModel(
            api_url=api_url,
            api_key=settings.OPENAI_API_KEY,
            model_name=settings.LLM_REPO_ID or "gpt-4o-mini",
        )
    # 3. Check for HuggingFace Endpoint (if configured)
    elif settings.HUGGINGFACEHUB_API_TOKEN and settings.LLM_PROVIDER == "huggingface":
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

            logger.info("Configuring HuggingFace ChatModel: %s", settings.LLM_REPO_ID)
            hf_llm = HuggingFaceEndpoint(
                repo_id=settings.LLM_REPO_ID,
                task="text-generation",
                max_new_tokens=settings.LLM_MAX_NEW_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
                huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN,
                timeout=20,
            )
            primary = ChatHuggingFace(llm=hf_llm)
        except Exception as exc:
            logger.warning("Could not construct primary HuggingFace ChatModel: %s", exc)

    _CHAT_MODEL_INSTANCE = ResilientChatModel(primary_model=primary)
    return _CHAT_MODEL_INSTANCE
