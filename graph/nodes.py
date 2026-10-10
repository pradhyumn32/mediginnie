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
    "TELL", "KNOW", "MEAN", "DOES", "HAVE", "WITH", "TOGETHER", "PLEASE", "HELP",
    "THE", "AND", "FOR", "ARE", "IS", "WAS", "WERE", "CAN", "MAY", "MIGHT", "WILL", "SHALL",
    "THAT", "THIS", "THESE", "THOSE", "FROM", "INTO", "ONTO", "SOME", "MANY", "MUCH", "MORE",
    "MOST", "ANY", "EACH", "EVERY", "ALL", "BOTH", "FEW", "LESS", "OVER", "UNDER", "BETWEEN",
    "LIKE", "ALSO", "EVEN", "JUST", "ONLY", "VERY", "HOW", "WHY", "WHO", "WHOM",
    "YOU", "YOUR", "YOURS", "THEY", "THEIR", "THEM", "HIM", "HER", "HERS", "HIS", "ITS",
    "OUR", "OURS", "MINE", "SAFE", "UNSAFE", "STOP", "START", "TAKE", "TOOK", "TAKEN",
    "GIVE", "GIVEN", "FEEL", "FEELING", "FELT", "FIND", "FOUND", "LOOK", "LOOKING",
    "DAILY", "WEEK", "WEEKS", "MONTH", "MONTHS", "DOSE", "DOSES", "INFO", "INFORMATION",
    "CARE", "BODY", "SKIN", "HEALTH", "LIFE", "WELL", "BETTER", "BEST", "WORST", "SUCH",
    "SIDE", "DRUG", "DRUGS", "CONTRAINDICATION", "CONTRAINDICATIONS",
    "BAKE", "BREAD", "COOK", "COOKING", "CAKE", "FOOD", "WATER", "MILK", "EAT", "EATING",
    "DRINK", "DRINKING", "MAKE", "MAKING", "PLAY", "WORK", "SLEEP", "HOME", "HOUSE", "WEATHER"
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
    Uses word-boundary matching and stopword protection to avoid false entity detection.
    """
    query = state.get("user_query", "")
    rxnorm = RxNormSource()

    detected_drugs: list[str] = []
    canonical_drugs: list[str] = []

    upper_query = query.upper()
    words = re.findall(r"\b[A-Za-z\-]{3,}\b", upper_query)

    # 1. Match against known drug candidates first (whole word match)
    for cand in KNOWN_DRUG_CANDIDATES:
        if re.search(rf"\b{re.escape(cand)}\b", upper_query) and cand not in detected_drugs:
            detected_drugs.append(cand)

    # 2. Match against local DB catalog using exact word/token boundaries
    try:
        with SessionLocal() as db:
            db_drugs = db.query(Drug.generic_name, Drug.brand_names).all()
            for d in db_drugs:
                gen_name = (d.generic_name or "").upper().strip()
                brands = [(b or "").upper().strip() for b in (d.brand_names or [])]

                # Check if exact brand appears in query
                for b in brands:
                    b_clean = re.sub(r"\[.*?\]", "", b).strip()
                    if len(b_clean) >= 3 and b_clean not in NON_DRUG_WORDS:
                        if re.search(rf"\b{re.escape(b_clean)}\b", upper_query):
                            if b_clean not in detected_drugs:
                                detected_drugs.append(b_clean)
                            if gen_name not in canonical_drugs:
                                canonical_drugs.append(gen_name)
                            break

                # Check active ingredient tokens from generic name (e.g. METFORMIN, LISINOPRIL, CEFIXIME)
                for tok in re.split(r"[\s,;/]+", gen_name):
                    if len(tok) >= 4 and tok not in NON_DRUG_WORDS:
                        if re.search(rf"\b{re.escape(tok)}\b", upper_query):
                            if tok not in detected_drugs:
                                detected_drugs.append(tok)
                            if gen_name not in canonical_drugs:
                                canonical_drugs.append(gen_name)
                            break
    except Exception as db_err:
        logger.debug("DB catalog lookup non-fatal error: %s", db_err)

    # 3. If no drugs detected yet, check remaining non-stopword tokens via RxNorm
    if not detected_drugs:
        candidate_words = [w for w in words if len(w) >= 4 and w not in NON_DRUG_WORDS]
        for word in candidate_words:
            resolved = rxnorm.resolve_drug_name(word)
            if resolved.get("rxcui"):
                detected_drugs.append(word)
                gen = resolved.get("generic_name", word)
                if gen not in canonical_drugs:
                    canonical_drugs.append(gen)
                break

        # If RxNorm did not match, check if query contains explicit clinical/medication intent
        # (e.g. "side effect", "warning", "taking", "dose", "reaction", "interact", "pill", "tablet")
        clinical_markers = [
            "side effect", "reaction", "warning", "taking", "took", "dose", "dosage",
            "interact", "pill", "tablet", "capsule", "drug", "medicine", "prescri",
            "allergic", "allergy", "symptom", "risk", "harm", "safe", "contraindicat"
        ]
        has_clinical_context = any(m in query.lower() for m in clinical_markers)

        if not detected_drugs and candidate_words and has_clinical_context:
            detected_drugs.append(candidate_words[0])
            canonical_drugs.append(candidate_words[0])

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
    If the requested drug is not in the DB, triggers automatic on-demand API ingestion.
    Guarantees that no chunks for unrelated drugs are returned.
    """
    query = state.get("user_query", "")
    canonical_drugs = state.get("canonical_drugs", [])
    extracted_drugs = state.get("extracted_drugs", [])
    intent = state.get("intent", "side_effects")

    retriever = HybridRetriever(top_k=5)

    target_drug = canonical_drugs[0] if canonical_drugs else (extracted_drugs[0] if extracted_drugs else None)
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

    # If section-filtered search returned too few results, fall back to general section
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

    # On-demand auto-ingestion: If the drug is not in the DB, pull directly from OpenFDA / DailyMed APIs
    if not chunks and target_drug:
        # Check if the target drug is recognized by RxNorm or in KNOWN_DRUG_CANDIDATES
        rxnorm = RxNormSource()
        rx_check = rxnorm.resolve_drug_name(target_drug)
        if rx_check.get("rxcui") or target_drug in KNOWN_DRUG_CANDIDATES:
            logger.info("Drug '%s' (RxCUI: %s) not present in local database. Attempting on-demand auto-ingestion from OpenFDA / DailyMed...", target_drug, rx_check.get("rxcui"))
            try:
                from preprocess.pipeline import ingest_drug
                candidates_to_ingest = [target_drug]
                if extracted_drugs and extracted_drugs[0].upper() not in [d.upper() for d in candidates_to_ingest]:
                    candidates_to_ingest.append(extracted_drugs[0])

                for cand in candidates_to_ingest:
                    ingest_res = ingest_drug(cand, source_choice="auto", max_records=1)
                    if ingest_res.get("status") == "success" and ingest_res.get("chunks_ingested", 0) > 0:
                        logger.info("Successfully auto-ingested '%s' (%d chunks). Re-running retrieval...", cand, ingest_res.get("chunks_ingested", 0))
                        chunks = retriever.retrieve_scored_chunks(
                            query=query,
                            drug_name=cand,
                            section=target_section,
                            top_k=5,
                        )
                        if len(chunks) < 2 and target_section is not None:
                            fallback_chunks = retriever.retrieve_scored_chunks(
                                query=query,
                                drug_name=cand,
                                section=None,
                                top_k=5,
                            )
                            existing_ids = {c["id"] for c in chunks}
                            for fc in fallback_chunks:
                                if fc["id"] not in existing_ids:
                                    chunks.append(fc)
                        if chunks:
                            target_drug = cand
                            break
            except Exception as ingest_err:
                logger.warning("On-demand ingestion error for '%s': %s", target_drug, ingest_err)

    # Filter chunks to ensure every chunk belongs strictly to the requested drug
    if target_drug:
        valid_chunks = []
        target_upper = target_drug.upper()
        target_tokens = [t for t in re.split(r"[\s,;/]+", target_upper) if len(t) >= 4 and t not in NON_DRUG_WORDS] or [target_upper]
        for c in chunks:
            c_gen = (c.get("generic_name") or "").upper()
            c_brands = [(b or "").upper() for b in (c.get("brand_names") or [])]
            c_content = (c.get("content") or "").upper()
            if (
                any(t in c_gen for t in target_tokens)
                or any(any(t in b for t in target_tokens) for b in c_brands)
                or any(re.search(rf"\b{re.escape(t)}\b", c_content) for t in target_tokens)
            ):
                valid_chunks.append(c)
        chunks = valid_chunks
    else:
        # If no specific drug was detected in query, do not return random drug chunks
        chunks = []

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
    Refuses cleanly if context is empty, missing, or irrelevant, without mentioning other drugs.
    """
    query = state.get("user_query", "")
    chunks = state.get("retrieved_chunks", [])
    canonical_drugs = state.get("canonical_drugs", [])
    extracted_drugs = state.get("extracted_drugs", [])
    drug_list = canonical_drugs or extracted_drugs
    drugs_display = ", ".join(drug_list) if drug_list else ""
    symptoms = ", ".join(state.get("extracted_symptoms", [])) or "General inquiry"

    # Strictly refuse if context is empty or irrelevant
    if not chunks:
        disclaimer = MedicalGuardrails.get_disclaimer()
        if drugs_display:
            refusal_msg = (
                f"I do not have official FDA or DailyMed labeling information for **{drugs_display}** "
                "in the knowledge database. I cannot provide medical information, side effects, or "
                "clinical guidance for medications not present in the official verified records. "
                "Please consult a qualified healthcare provider or pharmacist."
            )
        else:
            refusal_msg = (
                "I do not have relevant official FDA or DailyMed labeling records in the database to answer your inquiry. "
                "Please consult a qualified healthcare provider or pharmacist."
            )
        full_text = f"{refusal_msg}\n\n---\n{disclaimer}"
        return {
            "final_response": full_text,
            "disclaimer": disclaimer,
            "sources": [],
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
        "drugs": drugs_display or "Unspecified Drug",
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
