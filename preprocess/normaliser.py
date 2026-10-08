from typing import Any
from preprocess.chunker import ClinicalChunker

FDA_SECTIONS = [
    "boxed_warning",
    "adverse_reactions",
    "warnings_and_cautions",
    "contraindications",
    "drug_interactions",
    "overdosage",
    "use_in_specific_populations",
    "pregnancy",
    "nursing_mothers",
    "pediatric_use",
    "geriatric_use",
    "indications_and_usage",
    "dosage_and_administration",
    "clinical_pharmacology",
    "patient_medication_information",
    "information_for_patients",
]


def first_value(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def normalize_fda_record(record: dict[str, Any]) -> dict[str, Any]:
    openfda = record.get("openfda", {}) or {}

    generic_names = openfda.get("generic_name", []) or []
    brand_names = openfda.get("brand_name", []) or []

    generic_name = generic_names[0] if generic_names else "UNKNOWN"

    sections: dict[str, str] = {}

    for section in FDA_SECTIONS:
        value = record.get(section)
        if not value:
            continue

        if isinstance(value, list):
            value = "\n\n".join(str(v) for v in value)
        elif not isinstance(value, str):
            value = str(value)

        cleaned_value = value.strip()
        if cleaned_value:
            sections[section] = cleaned_value

    return {
        "generic_name": generic_name.upper(),
        "brand_names": [b.upper() for b in brand_names],
        "set_id": first_value(record.get("set_id")),
        "effective_date": first_value(record.get("effective_date")),
        "sections": sections,
        "openfda": openfda,
    }


def create_chunks(normalized_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Backward-compatible entry point delegating to ClinicalChunker.
    """
    chunker = ClinicalChunker()
    return chunker.chunk_normalized_drug(normalized_data)