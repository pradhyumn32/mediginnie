# Clinical Chatbot API Endpoints for Medijinn
# Supports synchronous JSON and real-time Server-Sent Events (SSE) streaming

import json
import logging
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage

from api.schemas import ChatRequest, ChatResponse, SourceCitation
from graph.workflow import clinical_agent
from graph.state import AgentState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Clinical Chatbot"])


def build_initial_state(request: ChatRequest) -> AgentState:
    """
    Constructs the initial LangGraph AgentState from the user's API request.
    """
    messages = []
    if request.chat_history:
        for m in request.chat_history:
            if m.role == "user":
                messages.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                messages.append(AIMessage(content=m.content))

    messages.append(HumanMessage(content=request.message))

    return {
        "messages": messages,
        "user_query": request.message,
        "triage_level": "NORMAL",
        "is_emergency": False,
        "emergency_message": None,
        "extracted_drugs": [],
        "canonical_drugs": [],
        "extracted_symptoms": [],
        "intent": "side_effects",
        "retrieved_chunks": [],
        "sources": [],
        "final_response": "",
        "disclaimer": "",
    }


@router.post("", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Executes clinical reasoning and grounded medical retrieval on the user's query.
    Returns structured clinical response with triage status and verified citations.
    """
    try:
        initial_state = build_initial_state(request)
        result = clinical_agent.invoke(initial_state)

        sources = [
            SourceCitation(
                drug=s.get("drug"),
                section=s.get("section"),
                severity=s.get("severity"),
                source=s.get("source"),
                chunk_id=str(s.get("chunk_id", "")),
            )
            for s in result.get("sources", [])
        ]

        return ChatResponse(
            response=result.get("final_response", ""),
            triage_level=result.get("triage_level", "NORMAL"),
            is_emergency=result.get("is_emergency", False),
            emergency_message=result.get("emergency_message"),
            extracted_drugs=result.get("extracted_drugs", []),
            canonical_drugs=result.get("canonical_drugs", []),
            extracted_symptoms=result.get("extracted_symptoms", []),
            intent=result.get("intent", "side_effects"),
            sources=sources,
            disclaimer=result.get("disclaimer", ""),
        )
    except Exception as exc:
        logger.exception("Error processing chat request: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Clinical analysis pipeline error: {str(exc)}",
        )


@router.post("/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streams LangGraph execution steps and synthesized clinical response via Server-Sent Events (SSE).
    Emits events: triage, entities, sources, and final response.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        initial_state = build_initial_state(request)

        try:
            # Stream graph updates step by step
            for output in clinical_agent.stream(initial_state):
                for node_name, node_state in output.items():
                    if node_name == "triage":
                        payload = {
                            "node": "triage",
                            "triage_level": node_state.get("triage_level", "NORMAL"),
                            "is_emergency": node_state.get("is_emergency", False),
                            "emergency_message": node_state.get("emergency_message"),
                        }
                        yield f"event: triage\ndata: {json.dumps(payload)}\n\n"

                    elif node_name == "extract_entities":
                        payload = {
                            "node": "entities",
                            "extracted_drugs": node_state.get("extracted_drugs", []),
                            "canonical_drugs": node_state.get("canonical_drugs", []),
                            "extracted_symptoms": node_state.get("extracted_symptoms", []),
                            "intent": node_state.get("intent", "side_effects"),
                        }
                        yield f"event: entities\ndata: {json.dumps(payload)}\n\n"

                    elif node_name == "retrieve":
                        payload = {
                            "node": "sources",
                            "count": len(node_state.get("sources", [])),
                            "sources": node_state.get("sources", []),
                        }
                        yield f"event: sources\ndata: {json.dumps(payload)}\n\n"

                    elif node_name in ("synthesize", "emergency"):
                        payload = {
                            "node": node_name,
                            "response": node_state.get("final_response", ""),
                            "disclaimer": node_state.get("disclaimer", ""),
                        }
                        yield f"event: message\ndata: {json.dumps(payload)}\n\n"

            yield "event: done\ndata: {}\n\n"

        except Exception as err:
            logger.exception("Streaming error: %s", err)
            err_payload = {"error": str(err)}
            yield f"event: error\ndata: {json.dumps(err_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
