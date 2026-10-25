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
    "USE", "USES", "USED", "USAGE", "INDICATION", "INDICATIONS", "PURPOSE", "PURPOSES",
    "TREAT", "TREATING", "TREATS", "BENEFIT", "BENEFITS",
    "ACID", "ACIDS", "SODIUM", "POTASSIUM", "CALCIUM", "CHLORIDE", "SULFATE", "SULPHATE",
    "ACETATE", "CITRATE", "PHOSPHATE", "TARTRATE", "MALEATE", "SUCCINATE", "NITRATE",
    "HYDROCHLORIDE", "HYDROXIDE", "MONOHYDRATE", "DIHYDRATE", "TRIHYDRATE", "CARBON",
    "WATER", "ORAL", "INJECTION", "SOLUTION", "SUSPENSION", "CREAM", "OINTMENT", "GEL",
    "SPRAY", "EXTRACT", "POWDER", "SYRUP", "DROPS", "PATCH",
    "BAKE", "BREAD", "COOK", "COOKING", "CAKE", "FOOD", "MILK", "EAT", "EATING",
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


def extract_drugs_from_text(text: str, rxnorm: RxNormSource | None = None) -> tuple[list[str], list[str]]:
    """
    Extracts detected and canonical drug names from a text string using
    known candidates, local database catalog, and RxNorm fallback.
    """
    if not text:
        return [], []

    detected: list[str] = []
    canonical: list[str] = []
    upper_text = text.upper()
    words = re.findall(r"\b[A-Za-z\-]{3,}\b", upper_text)

    # 1. Match against known drug candidates first (whole word match)
    for cand in KNOWN_DRUG_CANDIDATES:
        if re.search(rf"\b{re.escape(cand)}\b", upper_text) and cand not in detected:
            detected.append(cand)

    # 2. Match against local DB catalog using exact word/token boundaries
    try:
        with SessionLocal() as db:
            db_drugs = db.query(Drug.generic_name, Drug.brand_names).all()
            for d in db_drugs:
                gen_name = (d.generic_name or "").upper().strip()
                brands = [(b or "").upper().strip() for b in (d.brand_names or [])]

                # Check if exact brand appears in text
                for b in brands:
                    b_clean = re.sub(r"\[.*?\]", "", b).strip()
                    if len(b_clean) >= 3 and b_clean not in NON_DRUG_WORDS:
                        if re.search(rf"\b{re.escape(b_clean)}\b", upper_text):
                            if b_clean not in detected:
                                detected.append(b_clean)
                            if gen_name not in canonical:
                                canonical.append(gen_name)
                            break

                # Check active ingredient tokens from generic name
                for tok in re.split(r"[\s,;/]+", gen_name):
                    if len(tok) >= 4 and tok not in NON_DRUG_WORDS:
                        if re.search(rf"\b{re.escape(tok)}\b", upper_text):
                            if tok not in detected:
                                detected.append(tok)
                            if gen_name not in canonical:
                                canonical.append(gen_name)
                            break
    except Exception as db_err:
        logger.debug("DB catalog lookup non-fatal error: %s", db_err)

    # 3. RxNorm fallback for remaining non-stopword tokens
    if not detected and rxnorm is not None:
        candidate_words = [w for w in words if len(w) >= 4 and w not in NON_DRUG_WORDS]
        for word in candidate_words:
            resolved = rxnorm.resolve_drug_name(word)
            if resolved.get("rxcui"):
                detected.append(word)
                gen = resolved.get("generic_name", word)
                if gen not in canonical:
                    canonical.append(gen)
                break

    # Canonicalize any remaining detected drugs
    if rxnorm is not None:
        for d in detected:
            if d in canonical:
                continue
            res = rxnorm.resolve_drug_name(d)
            gen = res.get("generic_name", d)
            if gen not in canonical:
                canonical.append(gen)

    return detected, canonical


# -------------------------------------------------------------
# Node 3: Entity Extraction & Intent Routing
# -------------------------------------------------------------
def extract_entities_node(state: AgentState) -> dict[str, Any]:
    """
    Extracts drug names and symptoms, resolves pronouns and follow-up queries
    across conversation history, and routes clinical intent.
    """
    query = state.get("user_query", "")
    rxnorm = RxNormSource()

    # 0. Early check for greetings, identity ("who are you"), capabilities ("what do you do"), and pleasantries
    query_lower = query.lower().strip()
    query_clean = re.sub(r"[^\w\s]", "", query_lower).strip()
    words_list = query_clean.split()

    is_greeting = (
        query_clean in [
            "hi", "hello", "hey", "hiya", "howdy", "greetings",
            "good morning", "good afternoon", "good evening", "good day", "sup", "yo"
        ]
        or (len(words_list) <= 2 and bool(words_list) and words_list[0] in ["hi", "hello", "hey"])
    )

    is_identity_or_help = (
        any(p in query_clean for p in [
            "who are you", "who r u", "who r you", "what are you", "what is medijinn", "who is medijinn",
            "what do you do", "what can you do", "what can u do", "how can you help", "how can u help",
            "tell me about yourself", "introduce yourself", "what is your name", "what is your purpose",
            "what do u do", "how does this work", "how do you work", "help me", "what features",
            "your capabilities", "what can i ask you", "what can i ask", "what are your capabilities",
            "who made you", "who created you"
        ])
        or query_clean in ["help", "info", "about", "about you", "menu"]
    )

    is_pleasantry = (
        query_clean in [
            "thanks", "thank you", "thank u", "thx", "many thanks", "thank you so much",
            "thanks a lot", "ok thanks", "okay thanks", "cool thanks", "great thanks",
            "bye", "goodbye", "see you", "see ya"
        ]
        or (len(words_list) <= 3 and any(w in words_list for w in ["thanks", "thank", "bye", "goodbye"]))
    )

    if is_greeting or is_identity_or_help or is_pleasantry:
        return {
            "extracted_drugs": [],
            "canonical_drugs": [],
            "extracted_symptoms": [],
            "intent": "greeting",
        }

    # 1. Extract drugs from current query
    detected_drugs, canonical_drugs = extract_drugs_from_text(query, rxnorm)

    # 2. Fallback check if query contains explicit clinical intent and a drug token
    words = re.findall(r"\b[A-Za-z\-]{3,}\b", query.upper())
    candidate_words = [w for w in words if len(w) >= 4 and w not in NON_DRUG_WORDS]
    clinical_markers = [
        "side effect", "reaction", "warning", "taking", "took", "dose", "dosage",
        "interact", "pill", "tablet", "capsule", "drug", "medicine", "prescri",
        "allergic", "allergy", "symptom", "risk", "harm", "safe", "contraindicat",
        "use", "uses", "used for", "indication", "indications", "treat", "treatment",
        "prescribed for", "purpose", "benefit", "cure", "help with"
    ]
    has_clinical_context = any(m in query.lower() for m in clinical_markers)

    if not detected_drugs and candidate_words and has_clinical_context:
        detected_drugs.append(candidate_words[0])
        canonical_drugs.append(candidate_words[0])

    # 3. Conversation History & Coreference Resolution
    # If the user is asking a follow-up ("What are its side effects?", "Can I take it with Ibuprofen?")
    messages = state.get("messages", [])
    prior_messages = messages[:-1] if len(messages) > 1 else []
    query_lower = query.lower()

    has_pronoun_or_reference = any(re.search(rf"\b{re.escape(w)}\b", query_lower) for w in [
        "it", "its", "this", "that", "these", "those", "the drug", "the medication", "the medicine",
        "them", "both", "together", "with this", "with it", "what about", "how about", "tell me more"
    ])

    if prior_messages and (not detected_drugs or has_pronoun_or_reference):
        prior_detected: list[str] = []
        prior_canonical: list[str] = []

        # 1. First, search previous user/human questions - they contain the exact drug name without clinical text noise
        for prev_msg in reversed(prior_messages):
            is_human = getattr(prev_msg, "type", "") in ("human", "user") or getattr(prev_msg, "role", "") == "user"
            if is_human:
                content = getattr(prev_msg, "content", "") or ""
                p_det, p_can = extract_drugs_from_text(content, rxnorm)
                if p_det:
                    prior_detected = p_det
                    prior_canonical = p_can
                    break

        # 2. If not found in human messages, search assistant messages
        if not prior_detected:
            for prev_msg in reversed(prior_messages):
                is_ai = getattr(prev_msg, "type", "") in ("ai", "assistant") or getattr(prev_msg, "role", "") == "assistant"
                if is_ai:
                    content = getattr(prev_msg, "content", "") or ""
                    header_match = re.search(r"(?:Assessment for|labeling information for \*\*|\*\*Summary\*\*\s*\n\s*)([A-Za-z0-9\s\-]+?)(?:\s+(?:Sodium|for Injection|Tablets|Capsules|is a|PPI|is an|\*\*|\n|$))", content, re.IGNORECASE)
                    if header_match:
                        cand = header_match.group(1).strip()
                        if cand and cand.upper() not in NON_DRUG_WORDS and len(cand) >= 3:
                            p_det, p_can = extract_drugs_from_text(cand, rxnorm)
                            if p_det:
                                prior_detected = p_det
                                prior_canonical = p_can
                                break

        if prior_detected:
            if not detected_drugs:
                # Follow-up question regarding the same drug
                detected_drugs = prior_detected
                canonical_drugs = prior_canonical
                logger.info("Carried forward drug from conversation history: %s", detected_drugs)
            elif has_pronoun_or_reference or any(k in query_lower for k in ["with", "together", "and", "mix", "combine"]):
                # Cross-turn drug-drug interaction inquiry (e.g. "Can I take it with Ibuprofen?")
                for d in prior_detected:
                    if d not in detected_drugs:
                        detected_drugs.append(d)
                for c in prior_canonical:
                    if c not in canonical_drugs:
                        canonical_drugs.append(c)
                logger.info("Combined current drug with chat history drug for interaction: %s", detected_drugs)

    # 4. Symptom extraction heuristic
    symptom_patterns = [
        "nausea", "vomiting", "diarrhea", "headache", "dizziness",
        "rash", "dry cough", "cough", "angioedema", "lactic acidosis",
        "stomach pain", "fatigue", "swelling", "constipation", "insomnia",
    ]
    detected_symptoms = [s for s in symptom_patterns if s in query_lower]

    # 5. Intent routing
    has_multi_drugs = len(detected_drugs) > 1
    has_interaction_kw = any(k in query_lower for k in [
        "interact", "interaction", "interactions", "taken together", "drug interaction",
        "mix", "mixing", "combine", "combining", "take with", "together with"
    ]) or (has_multi_drugs and any(k in query_lower for k in ["with", "together", "and", "+"]))

    is_conversational_meta = any(k in query_lower for k in [
        "what did i ask", "what did you say", "summarize our chat", "summarize our conversation",
        "repeat that", "what were we talking about", "what was the first medication",
        "what was the previous drug", "earlier", "previous message", "summarize what we discussed"
    ])

    if is_conversational_meta and prior_messages:
        intent = "conversational"
    elif has_interaction_kw:
        intent = "interaction"
    elif any(k in query_lower for k in [
        "use", "uses", "used for", "indication", "indications", "prescribed for",
        "purpose", "purposes", "treat", "treatment", "treats", "treating",
        "what does it do", "what is it for", "why take", "why is it used",
        "benefit", "benefits", "prescribed to", "what is this drug for", "what is this medication for"
    ]):
        intent = "indications"
    elif any(k in query_lower for k in [
        "side effect", "side effects", "adverse", "reaction", "reactions", "unwanted effect", "harmful effect"
    ]):
        intent = "side_effects"
    elif any(k in query_lower for k in ["pregnant", "pregnancy", "breastfeeding", "kidney", "liver", "contraindicated", "contraindication"]):
        intent = "contraindication"
    elif any(k in query_lower for k in ["dose", "dosage", "how much", "how often", "mg"]):
        intent = "dosage"
    elif any(k in query_lower for k in ["warning", "warnings", "precaution", "precautions", "boxed warning", "black box"]):
        intent = "warnings"
    elif has_multi_drugs:
        intent = "interaction"
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

    if intent in ("greeting", "conversational"):
        return {
            "retrieved_chunks": [],
            "sources": [],
        }
    elif intent == "side_effects":
        target_section = "adverse_reactions"
    elif intent == "indications":
        target_section = "indications_and_usage"
    elif intent == "interaction":
        target_section = "drug_interactions"
    elif intent == "contraindication":
        target_section = "contraindications"
    elif intent == "dosage":
        target_section = "dosage_and_administration"
    elif intent == "warnings":
        target_section = "warnings_and_cautions"

    chunks = retriever.retrieve_scored_chunks(
        query=query,
        drug_name=target_drug,
        section=target_section,
        top_k=5,
    )

    # For drug interactions with multiple drugs, retrieve interaction chunks for secondary drugs
    if intent == "interaction" and len(canonical_drugs) > 1:
        for other_d in canonical_drugs[1:3]:
            sec_chunks = retriever.retrieve_scored_chunks(
                query=query,
                drug_name=other_d,
                section="drug_interactions",
                top_k=3,
            )
            existing_ids = {c["id"] for c in chunks}
            for sc in sec_chunks:
                if sc["id"] not in existing_ids:
                    chunks.append(sc)

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

    # Filter chunks to ensure every chunk belongs strictly to the requested drugs
    if target_drug:
        valid_chunks = []
        all_target_drugs = [d.upper() for d in (canonical_drugs + extracted_drugs)]
        target_tokens = []
        for td in all_target_drugs:
            for t in re.split(r"[\s,;/]+", td):
                if len(t) >= 4 and t not in NON_DRUG_WORDS and t not in target_tokens:
                    target_tokens.append(t)
        if not target_tokens:
            target_tokens = [target_drug.upper()]

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
    Answers conversational queries from chat history, and refuses cleanly if
    context is empty, missing, or irrelevant without mentioning other drugs.
    """
    query = state.get("user_query", "")
    chunks = state.get("retrieved_chunks", [])
    canonical_drugs = state.get("canonical_drugs", [])
    extracted_drugs = state.get("extracted_drugs", [])
    intent = state.get("intent", "side_effects")
    prior_messages = state.get("messages", [])[:-1] if len(state.get("messages", [])) > 1 else []

    # Handle greetings, identity queries, and pleasantries cleanly without touching RAG
    if intent == "greeting":
        query_lower_s = query.lower().strip()
        query_clean_s = re.sub(r"[^\w\s]", "", query_lower_s).strip()

        # Farewell
        if any(w in query_clean_s for w in ["bye", "goodbye", "see you", "see ya"]):
            reply = "Goodbye! Stay healthy, and don't hesitate to return if you have any medication questions. 👋"
        # Thanks
        elif any(w in query_clean_s for w in ["thank", "thanks", "thx"]):
            reply = "You're very welcome! If you have any more questions about medications, I'm here to help. 😊"
        # Identity / capabilities
        elif any(p in query_clean_s for p in [
            "who are you", "who r u", "what are you", "what is medijinn", "who is medijinn",
            "tell me about yourself", "introduce yourself", "what is your name", "what is your purpose",
            "who made you", "who created you"
        ]):
            reply = (
                "👋 I'm **Medijinn**, your clinical pharmaceutical assistant.\n\n"
                "I provide accurate, FDA/DailyMed-grounded information about medications, including:\n"
                "- 💊 **Side effects & adverse reactions**\n"
                "- 📋 **Drug uses & approved indications**\n"
                "- ⚠️ **Warnings, precautions & contraindications**\n"
                "- 🔗 **Drug-drug interactions**\n"
                "- 📏 **Dosage & administration**\n\n"
                "Just ask me something like *\"What are the side effects of Metformin?\"* or "
                "*\"What is Lisinopril used for?\"* and I'll get you the information straight from official drug labeling.\n\n"
                "⚠️ *I am not a substitute for professional medical advice. Always consult your doctor or pharmacist.*"
            )
        # What can you do / help
        elif any(p in query_clean_s for p in [
            "what do you do", "what can you do", "how can you help", "what features",
            "your capabilities", "what can i ask", "how does this work", "how do you work",
            "help me", "help", "menu"
        ]):
            reply = (
                "Here's what I can help you with:\n\n"
                "- 💊 **Side effects** — *\"What are the side effects of Ibuprofen?\"*\n"
                "- 📋 **Drug uses** — *\"What is Metformin used for?\"*\n"
                "- ⚠️ **Warnings** — *\"What are the warnings for Lisinopril?\"*\n"
                "- 🔗 **Drug interactions** — *\"Can I take Aspirin with Ibuprofen?\"*\n"
                "- 📏 **Dosage info** — *\"What is the dosage of Amoxicillin?\"*\n"
                "- 🚫 **Contraindications** — *\"Is Metformin safe during pregnancy?\"*\n\n"
                "All information is sourced from official FDA and DailyMed drug labeling. "
                "For personalized medical advice, please consult your doctor or pharmacist."
            )
        # Simple greeting
        else:
            reply = (
                "👋 Hello! I'm **Medijinn**, your clinical pharmaceutical assistant.\n\n"
                "I can answer questions about drug side effects, uses, interactions, dosage, warnings, and more — "
                "all grounded in official FDA and DailyMed data.\n\n"
                "How can I help you today?"
            )

        disclaimer = MedicalGuardrails.get_disclaimer()
        full_text = f"{reply}\n\n---\n{disclaimer}"
        return {
            "final_response": full_text,
            "disclaimer": disclaimer,
            "sources": [],
            "messages": [AIMessage(content=full_text)],
        }

    # Handle conversational/meta history questions (e.g. "What did we talk about earlier?", "Can you summarize our conversation?")
    if intent == "conversational" and prior_messages:
        chat_model = get_chat_model()
        system_msg = (
            "You are Medijinn, a clinical pharmaceutical assistant. "
            "The user is asking a conversational question about your previous chat history. "
            "Answer concisely and accurately based strictly on the prior conversation messages."
        )
        prompt_msgs = [
            ("system", system_msg),
            *[(m.type if m.type in ("human", "ai") else "human", m.content) for m in prior_messages],
            ("human", query),
        ]
        response = chat_model.invoke(prompt_msgs)
        content = response.content if hasattr(response, "content") else str(response)
        disclaimer = MedicalGuardrails.get_disclaimer()
        full_text = f"{content}\n\n---\n{disclaimer}"
        return {
            "final_response": full_text,
            "disclaimer": disclaimer,
            "sources": [],
            "messages": [AIMessage(content=full_text)],
        }

    if extracted_drugs:
        drug_list = [d.title() for d in extracted_drugs]
    elif canonical_drugs:
        drug_list = [canonical_drugs[0].title()]
    else:
        drug_list = []
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
        "chat_history": prior_messages,
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
