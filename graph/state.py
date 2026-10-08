# LangGraph Clinical Agent State Definition for Medijinn

from typing import Annotated, Any, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State passed through the LangGraph reasoning workflow.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str

    # Safety & Triage
    triage_level: Literal["NORMAL", "HIGH_RISK", "EMERGENCY"]
    is_emergency: bool
    emergency_message: str | None

    # Clinical Entity Decomposition
    extracted_drugs: list[str]
    canonical_drugs: list[str]
    extracted_symptoms: list[str]
    intent: Literal["side_effects", "interaction", "contraindication", "dosage", "general", "emergency"]

    # Retrieval & Provenance
    retrieved_chunks: list[dict[str, Any]]
    sources: list[dict[str, Any]]

    # Synthesis Output
    final_response: str
    disclaimer: str
