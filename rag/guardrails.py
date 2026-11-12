# Medical Safety Guardrails & Emergency Triage for Medijinn
# Protects against acute clinical risks, life-threatening symptoms,
# and non-prescriptive boundaries.

import re
from typing import Any

STANDARD_MEDICAL_DISCLAIMER = (
    "⚠️ **MEDICAL DISCLAIMER**: Medijinn provides general health education "
    "and reference information retrieved directly from FDA and DailyMed approved drug labels. "
    "This tool does NOT provide clinical diagnoses, personalized treatment plans, or prescription alterations. "
    "Always consult a qualified clinician or pharmacist before modifying medication or if you experience adverse effects."
)

# Acute clinical patterns that demand immediate emergency triage
EMERGENCY_TRIGGERS: list[dict[str, Any]] = [
    {
        "type": "ANAPHYLAXIS",
        "patterns": [
            r"throat\s+(is\s+)?(closing|tightening|swelling|swollen)",
            r"(swollen|swelling)\s+(tongue|lips|face|throat)",
            r"(tongue|lips|face|throat)\s+(is\s+)?(swollen|swelling)",
            r"(can['’]?t|cannot|unable\s+to)\s+breathe",
            r"(severe\s+)?(difficulty|trouble|hard\s+time)\s+breathing",
            r"gasping\s+for\s+air",
            r"airway\s+(is\s+)?closing",
            r"lips\s+(are\s+)?(turning\s+)?blue",
            r"suffocat(ing|ion)",
            r"stridor",
            r"anaphylaxis|anaphylactic",
        ],
        "directive": (
            "🚨 **CRITICAL MEDICAL EMERGENCY DETECTED (Possible Anaphylaxis / Severe Allergic Reaction)**\n\n"
            "If you or someone nearby is experiencing difficulty breathing, swelling of the throat/lips, "
            "or severe allergic symptoms, **immediately call local emergency services (911 in the US/Canada, 112 in Europe, 108 in India)** "
            "or use an epinephrine auto-injector (EpiPen) if prescribed and proceed to the nearest emergency department."
        ),
    },
    {
        "type": "CARDIAC_ARREST_OR_INFARCTION",
        "patterns": [
            r"crushing\s+chest\s+pain",
            r"(severe|crushing|sharp)?\s*chest\s+(pressure|pain|tightness)",
            r"pain\s+(radiating|spreading)\s+to\s+(left\s+arm|jaw|back)",
            r"heart\s+attack",
            r"elephant\s+(is\s+)?(sitting\s+)?on\s+(my\s+)?chest",
            r"chest\s+(is\s+)?squeezing",
        ],
        "directive": (
            "🚨 **CRITICAL MEDICAL EMERGENCY DETECTED (Possible Acute Coronary Event)**\n\n"
            "Severe chest pain or pressure radiating to the arm or jaw may indicate a life-threatening cardiac event. "
            "**Immediately call emergency services (911/112/108)**. Do not drive yourself to the hospital."
        ),
    },
    {
        "type": "STROKE",
        "patterns": [
            r"face\s+(is\s+)?droop(ing)?",
            r"slurred\s+speech",
            r"sudden\s+numbness\s+on\s+one\s+side",
            r"sudden\s+paralysis",
            r"sudden\s+loss\s+of\s+vision",
            r"can['’]?t\s+(move|lift)\s+(my\s+)?arm",
            r"lost\s+consciousness",
            r"unresponsive",
            r"stroke",
        ],
        "directive": (
            "🚨 **CRITICAL MEDICAL EMERGENCY DETECTED (Possible Stroke - F.A.S.T.)**\n\n"
            "Sudden facial drooping, arm weakness, or slurred speech are acute signs of a stroke where every minute counts. "
            "**Call emergency medical services immediately (911/112/108)**."
        ),
    },
    {
        "type": "OVERDOSE_OR_POISONING",
        "patterns": [
            r"(took|swallowed)\s+(the\s+)?(whole|entire)\s+bottle",
            r"(swallowed|took|ate)\s+\d+\s+pills",
            r"(ate|took)\s+(a\s+)?handful\s+of\s+pills",
            r"(toddler|baby|child)\s+(ate|swallowed|took)\s+(my\s+)?(pills|medicine)",
            r"(swallowed|drank)\s+(bleach|cleaner|detergent|antifreeze|poison|chemical)",
            r"(intentional\s+)?overdose",
            r"drank\s+poison",
            r"tried\s+to\s+kill\s+myself",
        ],
        "directive": (
            "🚨 **ACUTE POISONING / OVERDOSE DETECTED**\n\n"
            "If you or someone else has ingested an excessive quantity of medication: "
            "**Immediately call Poison Control (1-800-222-1222 in the US) or emergency services (911/112/108)**. "
            "If you are feeling overwhelmed or having thoughts of self-harm, please call or text **988** (Suicide & Crisis Lifeline)."
        ),
    },
    {
        "type": "SEVERE_HEMORRHAGE",
        "patterns": [
            r"coughing\s+up\s+blood",
            r"vomiting\s+blood",
            r"black\s+tarry\s+stool",
            r"uncontrolled\s+bleeding",
        ],
        "directive": (
            "🚨 **CRITICAL MEDICAL EMERGENCY DETECTED (Severe Hemorrhage / Internal Bleeding)**\n\n"
            "Vomiting or coughing up blood, or passing black tarry stools indicates possible severe internal hemorrhage. "
            "**Seek immediate emergency medical attention at the nearest emergency room.**"
        ),
    },
    {
        "type": "ACUTE_UNRESPONSIVE_OR_APNEA",
        "patterns": [
            r"can['’]?t\s+wake\s+(him|her|them|my|the\s+person)\s+up",
            r"(he|she|they|person)\s+is\s+not\s+breathing",
            r"stopped\s+breathing",
            r"passed\s+out\s+and\s+won['’]?t\s+wake",
        ],
        "directive": (
            "🚨 **CRITICAL MEDICAL EMERGENCY DETECTED (Patient Unresponsive / Not Breathing)**\n\n"
            "If the person is unresponsive or not breathing normally: "
            "**Immediately call emergency medical services (911/112/108)** and begin CPR if trained. "
            "If an opioid overdose is suspected, administer Naloxone (Narcan) immediately."
        ),
    },
]

# Patterns requesting prescriptive decisions
PRESCRIBING_PATTERNS = [
    r"(can|could)\s+you\s+(please\s+)?prescribe",
    r"write\s+(me\s+)?a\s+prescription",
    r"should\s+i\s+(stop|quit|discontinue|pause)\s+taking",
    r"should\s+i\s+(double|halve|increase|decrease|change|reduce)\s+my\s+dose",
    r"(how\s+many\s+pills|what\s+dose)\s+should\s+i\s+take\s+to\s+(cure|treat)",
    r"adjust\s+my\s+(dose|medication|prescription)",
    r"recommend\s+a\s+dose\s+for\s+me",
]


class MedicalGuardrails:
    """
    Evaluates clinical queries for high-acuity emergencies, non-prescriptive
    boundaries, and disclaimer injection.
    """

    @staticmethod
    def evaluate_emergency(user_query: str) -> dict[str, Any]:
        """
        Scans user query for life-threatening acute medical emergencies.
        Returns triage status and emergency directive.
        """
        query_lower = user_query.lower()

        for trigger in EMERGENCY_TRIGGERS:
            for pattern in trigger["patterns"]:
                if re.search(pattern, query_lower):
                    return {
                        "is_emergency": True,
                        "emergency_type": trigger["type"],
                        "emergency_message": trigger["directive"],
                        "triage_level": "EMERGENCY",
                    }

        return {
            "is_emergency": False,
            "emergency_type": None,
            "emergency_message": None,
            "triage_level": "NORMAL",
        }

    @staticmethod
    def check_prescriptive_intent(user_query: str) -> bool:
        """
        Checks if the user is asking the AI to prescribe or unilaterally change prescription regimens.
        """
        query_lower = user_query.lower()
        return any(re.search(pat, query_lower) for pat in PRESCRIBING_PATTERNS)

    PRESCRIPTION_REFUSAL_DIRECTIVE: str = (
        "⛔ **NON-PRESCRIPTIVE CLINICAL BOUNDARY**\n\n"
        "Medijinn provides pharmaceutical reference information only. It cannot prescribe medications, "
        "recommend specific dosages, or authorize discontinuing or modifying prescribed therapies. "
        "Please consult your prescribing doctor or a licensed pharmacist to evaluate any dosage changes."
    )

    @classmethod
    def classify_safety(cls, user_query: str, chat_model: Any = None) -> dict[str, Any]:
        """
        Multi-stage safety classification:
        1. Fast deterministic regex guardrails (zero-latency).
        2. Prescriptive boundary check.
        3. Semantic LLM safety check for unpatterned high-acuity cues.
        """
        # Stage 1: Deterministic emergency regex
        emergency_res = cls.evaluate_emergency(user_query)
        if emergency_res["is_emergency"]:
            return emergency_res

        # Stage 2: Prescriptive boundary check
        if cls.check_prescriptive_intent(user_query):
            return {
                "is_emergency": False,
                "is_blocked": True,
                "emergency_type": "PRESCRIPTION_REQUEST",
                "emergency_message": cls.PRESCRIPTION_REFUSAL_DIRECTIVE,
                "triage_level": "BLOCKED",
            }

        # Stage 3: Semantic safety triage for acute cues not matched by regex
        ACUTE_CUES = (
            "chest", "breathe", "breathing", "swallowed", "bleeding", "blood",
            "collapsed", "unconscious", "poison", "suicide", "overdose", "seizure",
            "passed out", "dying", "fainted"
        )
        q_lower = user_query.lower()
        if chat_model and any(cue in q_lower for cue in ACUTE_CUES):
            try:
                prompt = (
                    "You are a clinical emergency triage classifier.\n"
                    "Determine whether the following patient query describes an acute, life-threatening medical emergency requiring immediate 911/ER attention.\n"
                    f"User Query: \"{user_query}\"\n"
                    "Reply ONLY with 'EMERGENCY' or 'NORMAL'."
                )
                res = chat_model.invoke(prompt)
                text = (res.content if hasattr(res, "content") else str(res)).strip().upper()
                if "EMERGENCY" in text:
                    return {
                        "is_emergency": True,
                        "is_blocked": False,
                        "emergency_type": "SEMANTIC_EMERGENCY_CLASSIFICATION",
                        "emergency_message": (
                            "🚨 **CRITICAL MEDICAL ALERT DETECTED**\n\n"
                            "Your description indicates potential acute medical distress. "
                            "**Immediately contact emergency services (911 in US/Canada, 112 in Europe, 108 in India)** "
                            "or go to the nearest emergency department."
                        ),
                        "triage_level": "EMERGENCY",
                    }
            except Exception:
                pass  # Fail open to NORMAL on classifier errors

        return {
            "is_emergency": False,
            "is_blocked": False,
            "emergency_type": None,
            "emergency_message": None,
            "triage_level": "NORMAL",
        }

    @staticmethod
    def get_disclaimer() -> str:
        return STANDARD_MEDICAL_DISCLAIMER
