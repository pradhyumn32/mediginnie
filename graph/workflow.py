# LangGraph Clinical Reasoning Workflow for Medijinn

import logging
from typing import Any
from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from graph.nodes import (
    triage_node,
    emergency_response_node,
    extract_entities_node,
    retrieve_node,
    synthesize_node,
)

logger = logging.getLogger(__name__)


def route_after_triage(state: AgentState) -> str:
    """
    Decides whether to route to immediate emergency directive or proceed to clinical RAG.
    """
    if state.get("is_emergency"):
        return "emergency"
    return "extract_entities"


def create_clinical_graph() -> Any:
    """
    Constructs and compiles the clinical reasoning StateGraph.
    """
    workflow = StateGraph(AgentState)

    # 1. Add nodes
    workflow.add_node("triage", triage_node)
    workflow.add_node("emergency", emergency_response_node)
    workflow.add_node("extract_entities", extract_entities_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("synthesize", synthesize_node)

    # 2. Add edges & conditional routing
    workflow.add_edge(START, "triage")

    workflow.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "emergency": "emergency",
            "extract_entities": "extract_entities",
        },
    )

    workflow.add_edge("emergency", END)
    workflow.add_edge("extract_entities", "retrieve")
    workflow.add_edge("retrieve", "synthesize")
    workflow.add_edge("synthesize", END)

    app = workflow.compile()
    return app


# Singleton compiled workflow instance
clinical_agent = create_clinical_graph()


def run_clinical_query(user_query: str) -> dict[str, Any]:
    """
    Convenience function to run a single query through the LangGraph pipeline.
    """
    initial_state: AgentState = {
        "messages": [],
        "user_query": user_query,
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

    result = clinical_agent.invoke(initial_state)
    return result


if __name__ == "__main__":
    print("Testing LangGraph Clinical Agent...")
    print("\n--- Test 1: Emergency Triage ---")
    res1 = run_clinical_query("I took Metformin and now my throat is swelling and I can't breathe!")
    print(f"Triage Level: {res1['triage_level']}")
    print(f"Response:\n{res1['final_response']}\n")

    print("\n--- Test 2: Brand Query (Glucophage) ---")
    res2 = run_clinical_query("What are common side effects of Glucophage?")
    print(f"Canonical Drug: {res2['canonical_drugs']}")
    print(f"Sources retrieved: {len(res2['sources'])}")
    print(f"Response:\n{res2['final_response']}\n")
