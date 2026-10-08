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
            r"(swallowed|took)\s+\d+\s+pills",
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
]

# Patterns requesting prescriptive decisions
PRESCRIBING_PATTERNS = [
    r"how\s+many\s+pills\s+should\s+i\s+take\s+to\s+cure",
    r"can\s+you\s+prescribe",
    r"write\s+me\s+a\s+prescription",
    r"should\s+i\s+stop\s+taking\s+my\s+insulin",
    r"should\s+i\s+double\s+my\s+dose",
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

    @staticmethod
    def get_disclaimer() -> str:
        return STANDARD_MEDICAL_DISCLAIMER
