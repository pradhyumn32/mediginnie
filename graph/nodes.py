# LangGraph Nodes for Medijinn Clinical Workflow

import logging
import re
from typing import Any
from langchain_core.messages import AIMessage, HumanMessage

from graph.state import AgentState
from rag.guardrails import MedicalGuardrails
from rag.hybrid_retriever import HybridRetriever
from rag.prompts import SYNTHESIS_PROMPT
from rag.llm import get_chat_model
from preprocess.medSources.rxnorm import RxNormSource
from db.db import SessionLocal
from db.models import Drug

logger = logging.getLogger(__name__)

# Common non-drug query tokens to filter out from entity detection
NON_DRUG_WORDS = {
    "WHAT", "WHEN", "WHERE", "WHICH", "WHOSE", "ABOUT", "COULD", "WOULD", "SHOULD",
    "AFTER", "BEFORE", "DURING", "TAKING", "THERE", "CAUSE", "CAUSES", "CAUSED",
    "EFFECT", "EFFECTS", "COMMON", "SERIOUS", "SEVERE", "MILD", "MAJOR", "MINOR",
    "RISK", "RISKS", "SAFETY", "WARN", "WARNING", "WARNINGS", "MEDICATION", "MEDICINE",
    "PRESCRIPTION", "PATIENT", "PEOPLE", "DOCTOR", "PHYSICIAN", "PHARMACY",
    "EXPERIENCING", "SYMPTOM", "SYMPTOMS", "PROBLEM", "PROBLEMS", "COMPLICATION", "COMPLICATIONS",
    "ADVERSE", "REACTION", "REACTIONS", "INTERACT", "INTERACTION", "INTERACTIONS",
    "TREATMENT", "DOSAGE", "TABLET", "TABLETS", "CAPSULE", "CAPSULES", "PILL", "PILLS",
    "TELL", "KNOW", "MEAN", "DOES", "HAVE", "WITH", "TOGETHER", "PLEASE", "HELP"
}

# Pre-cached clinical drug candidates
KNOWN_DRUG_CANDIDATES = [
    "METFORMIN", "GLUCOPHAGE", "LISINOPRIL", "ZESTRIL", "PRINIVIL",
    "FINASTERIDE", "PROPECIA", "PROSCAR", "CEFIXIME", "SUPRAX",
    "IBUPROFEN", "ADVIL", "MOTRIN", "AMOXICILLIN", "AUGMENTIN",
    "ATORVASTATIN", "LIPITOR", "OMEPRAZOLE", "PRILOSEC", "ASPIRIN",
    "TYLENOL", "ACETAMINOPHEN", "METOPROLOL", "AMLODIPINE", "LOSARTAN",
    "GABAPENTIN", "SERTRALINE", "ZOLOFT", "LEVOTHYROXINE", "SYNTHROID"
]


# -------------------------------------------------------------
# Node 1: Emergency Triage Guardrail
# -------------------------------------------------------------
def triage_node(state: AgentState) -> dict[str, Any]:
    """
    Evaluates incoming query for acute medical emergencies.
    Bypasses retrieval if life-threatening conditions are detected.
    """
    query = state.get("user_query", "").strip()
    if not query and state.get("messages"):
        query = state["messages"][-1].content

    triage_result = MedicalGuardrails.evaluate_emergency(query)

    if triage_result["is_emergency"]:
        logger.warning("🚨 Acute emergency detected: %s", triage_result["emergency_type"])
        return {
            "user_query": query,
            "is_emergency": True,
            "triage_level": "EMERGENCY",
            "emergency_message": triage_result["emergency_message"],
            "intent": "emergency",
        }

    return {
        "user_query": query,
        "is_emergency": False,
        "triage_level": "NORMAL",
        "emergency_message": None,
    }


# -------------------------------------------------------------
# Node 2: Emergency Response Node
# -------------------------------------------------------------
def emergency_response_node(state: AgentState) -> dict[str, Any]:
    """
    Directly formats the urgent emergency directive without LLM hallucination risk.
    """
    msg = state.get("emergency_message") or (
        "🚨 **CRITICAL MEDICAL ALERT**: Please seek emergency medical care immediately. Call 911/112/108."
    )
    disclaimer = MedicalGuardrails.get_disclaimer()
    full_text = f"{msg}\n\n---\n{disclaimer}"

    return {
        "final_response": full_text,
        "disclaimer": disclaimer,
        "sources": [],
        "messages": [AIMessage(content=full_text)],
    }


# -------------------------------------------------------------
# Node 3: Entity Extraction & Intent Routing
# -------------------------------------------------------------
def extract_entities_node(state: AgentState) -> dict[str, Any]:
    """
    Extracts drug names and symptoms, and uses DB catalog / RxNorm to resolve brand names.
    """
    query = state.get("user_query", "")
    rxnorm = RxNormSource()

    detected_drugs: list[str] = []
    canonical_drugs: list[str] = []

    upper_query = query.upper()
    words = re.findall(r"\b[A-Za-z\-]{3,}\b", upper_query)

    # 1. Check pre-cached candidates
    for word in words:
        if word in KNOWN_DRUG_CANDIDATES and word not in detected_drugs:
            detected_drugs.append(word)

    # 2. Check local DB drugs for fast zero-latency matching
    try:
        with SessionLocal() as db:
            db_drugs = db.query(Drug.generic_name, Drug.brand_names).all()
            for d in db_drugs:
                gen_name = (d.generic_name or "").upper()
                brands = [b.upper() for b in (d.brand_names or [])]
                for word in words:
                    if word in NON_DRUG_WORDS:
                        continue
                    if (word in gen_name or any(word in b for b in brands)) and word not in detected_drugs:
                        detected_drugs.append(word)
                        if gen_name not in canonical_drugs:
                            canonical_drugs.append(gen_name)
    except Exception as db_err:
        logger.debug("DB catalog lookup non-fatal error: %s", db_err)

    # 3. If no known candidates matched, check RxNorm for remaining non-stopword terms
    if not detected_drugs:
        for word in words:
            if len(word) > 3 and word not in NON_DRUG_WORDS:
                resolved = rxnorm.resolve_drug_name(word)
                if resolved.get("rxcui"):
                    detected_drugs.append(word)
                    break

    # Canonicalize any detected drug not already canonicalized
    for d in detected_drugs:
        if d in canonical_drugs:
            continue
        res = rxnorm.resolve_drug_name(d)
        gen = res.get("generic_name", d)
        if gen not in canonical_drugs:
            canonical_drugs.append(gen)

    # Symptom extraction heuristic
    symptom_patterns = [
        "nausea", "vomiting", "diarrhea", "headache", "dizziness",
        "rash", "dry cough", "cough", "angioedema", "lactic acidosis",
        "stomach pain", "fatigue", "swelling", "constipation", "insomnia",
    ]
    detected_symptoms = [s for s in symptom_patterns if s in query.lower()]

    # Intent routing
    query_lower = query.lower()
    if len(canonical_drugs) > 1 or "with" in query_lower or "together" in query_lower or "interact" in query_lower:
        intent = "interaction"
    elif any(k in query_lower for k in ["pregnant", "pregnancy", "breastfeeding", "kidney", "liver", "contraindicated"]):
        intent = "contraindication"
    elif any(k in query_lower for k in ["dose", "dosage", "how much", "mg"]):
        intent = "dosage"
    else:
        intent = "side_effects"

    return {
        "extracted_drugs": detected_drugs,
        "canonical_drugs": canonical_drugs,
        "extracted_symptoms": detected_symptoms,
        "intent": intent,
    }


# -------------------------------------------------------------
# Node 4: Targeted Clinical Hybrid Retrieval
# -------------------------------------------------------------
def retrieve_node(state: AgentState) -> dict[str, Any]:
    """
    Retrieves section-aware chunks using pgvector dense + Postgres trigram sparse search.
    """
    query = state.get("user_query", "")
    canonical_drugs = state.get("canonical_drugs", [])
    intent = state.get("intent", "side_effects")

    retriever = HybridRetriever(top_k=5)

    target_drug = canonical_drugs[0] if canonical_drugs else None
    target_section = None

    if intent == "side_effects":
        target_section = "adverse_reactions"
    elif intent == "interaction":
        target_section = "drug_interactions"
    elif intent == "contraindication":
        target_section = "contraindications"

    chunks = retriever.retrieve_scored_chunks(
        query=query,
        drug_name=target_drug,
        section=target_section,
        top_k=5,
    )

    # If section-filtered search returned too few results, fall back to unfiltered hybrid search
    if len(chunks) < 2 and target_section is not None:
        fallback_chunks = retriever.retrieve_scored_chunks(
            query=query,
            drug_name=target_drug,
            section=None,
            top_k=5,
        )
        existing_ids = {c["id"] for c in chunks}
        for fc in fallback_chunks:
            if fc["id"] not in existing_ids:
                chunks.append(fc)

    sources = []
    for c in chunks:
        sources.append({
            "drug": c.get("generic_name"),
            "section": c.get("section_title"),
            "severity": c.get("severity"),
            "source": c.get("source"),
            "chunk_id": c.get("id"),
        })

    return {
        "retrieved_chunks": chunks,
        "sources": sources,
    }


# -------------------------------------------------------------
# Node 5: Grounded Clinical Synthesis
# -------------------------------------------------------------
def synthesize_node(state: AgentState) -> dict[str, Any]:
    """
    Synthesizes a grounded response citing retrieved FDA/DailyMed chunks.
    """
    query = state.get("user_query", "")
    chunks = state.get("retrieved_chunks", [])
    drugs = ", ".join(state.get("canonical_drugs", [])) or "Unspecified Drug"
    symptoms = ", ".join(state.get("extracted_symptoms", [])) or "General inquiry"

    if not chunks:
        no_info_msg = (
            f"I could not locate official FDA or DailyMed labeling records for **{drugs}** in the knowledge database. "
            "Please verify the medication spelling or check with a healthcare provider."
        )
        disclaimer = MedicalGuardrails.get_disclaimer()
        full_text = f"{no_info_msg}\n\n---\n{disclaimer}"
        return {
            "final_response": full_text,
            "disclaimer": disclaimer,
            "messages": [AIMessage(content=full_text)],
        }

    # Format context for prompt
    context_blocks = []
    for idx, c in enumerate(chunks, 1):
        block = f"[Chunk {idx} | Source: {c.get('source', 'FDA').upper()} | Section: {c.get('section_title')} | Severity: {c.get('severity')}]\n{c['content']}"
        context_blocks.append(block)

    context_str = "\n\n".join(context_blocks)

    chat_model = get_chat_model()
    prompt_input = {
        "user_query": query,
        "drugs": drugs,
        "symptoms": symptoms,
        "context": context_str,
        "chat_history": state.get("messages", [])[:-1],
    }

    chain = SYNTHESIS_PROMPT | chat_model
    response = chain.invoke(prompt_input)
    response_content = response.content if hasattr(response, "content") else str(response)

    disclaimer = MedicalGuardrails.get_disclaimer()
    final_output = f"{response_content}\n\n---\n{disclaimer}"

    return {
        "final_response": final_output,
        "disclaimer": disclaimer,
        "messages": [AIMessage(content=final_output)],
    }
