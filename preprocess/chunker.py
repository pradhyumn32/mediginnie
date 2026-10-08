
# Clinical Section-Aware Chunker for Medijinn, which
# transforms raw FDA drug labels into semantic and metadata chunks.

# Using LangChain RecursiveCharacterTextSplitter.

from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings


# FDA Section mapping with clinical severity and friendly titles
SECTION_CONFIG: dict[str, dict[str, str]] = {
    "boxed_warning": {
        "title": "Boxed Warning (Black Box)",
        "severity": "CRITICAL",
        "category": "safety_alert",
    },
    "contraindications": {
        "title": "Contraindications",
        "severity": "HIGH",
        "category": "safety_alert",
    },
    "adverse_reactions": {
        "title": "Adverse Reactions & Side Effects",
        "severity": "SIDE_EFFECT",
        "category": "symptom_risk",
    },
    "warnings_and_cautions": {
        "title": "Warnings and Precautions",
        "severity": "MEDIUM",
        "category": "symptom_risk",
    },
    "drug_interactions": {
        "title": "Drug Interactions",
        "severity": "INTERACTION",
        "category": "interaction",
    },
    "overdosage": {
        "title": "Overdosage & Toxicity",
        "severity": "EMERGENCY",
        "category": "emergency",
    },
    "indications_and_usage": {
        "title": "Indications & Prescribed Usage",
        "severity": "INFO",
        "category": "indication",
    },
    "dosage_and_administration": {
        "title": "Dosage & Administration",
        "severity": "INFO",
        "category": "dosage",
    },
    "use_in_specific_populations": {
        "title": "Use in Specific Populations (Pregnancy/Pediatric/Geriatric)",
        "severity": "HIGH",
        "category": "population_risk",
    },
    "clinical_pharmacology": {
        "title": "Clinical Pharmacology",
        "severity": "INFO",
        "category": "pharmacology",
    },
}


class ClinicalChunker:
    """
    Splits normalized FDA drug records into context-preserved chunks.
    Prepends contextual headers to ensure dense vectors and full-text search
    retain the drug and clinical section identity.
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " "],
            keep_separator=True,
        )

    def chunk_normalized_drug(self, normalized_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Takes normalized FDA drug record and returns list of chunks with metadata.
        """
        generic_name = normalized_data.get("generic_name", "UNKNOWN").strip().upper()
        brand_names = normalized_data.get("brand_names", []) or []
        brand_str = ", ".join(brand_names[:5]) if brand_names else "N/A"
        set_id = normalized_data.get("set_id")
        sections = normalized_data.get("sections", {})

        chunks: list[dict[str, Any]] = []

        for section_key, raw_content in sections.items():
            if not raw_content or not raw_content.strip():
                continue

            sec_cfg = SECTION_CONFIG.get(
                section_key,
                {
                    "title": section_key.replace("_", " ").title(),
                    "severity": "INFO",
                    "category": "general",
                },
            )

            # Split content using LangChain RecursiveCharacterTextSplitter
            splits = self.text_splitter.split_text(raw_content)

            for idx, split_text in enumerate(splits):
                cleaned_text = split_text.strip()
                if not cleaned_text:
                    continue

                # Prepend contextual header for high-precision retrieval
                contextual_header = (
                    f"Drug: {generic_name} (Brands: {brand_str}) | "
                    f"Section: {sec_cfg['title']} | "
                    f"Severity: {sec_cfg['severity']}\n\n"
                )
                full_chunk_text = contextual_header + cleaned_text

                chunk_metadata = {
                    "source": "openfda",
                    "generic_name": generic_name,
                    "brand_names": brand_names,
                    "set_id": set_id,
                    "section": section_key,
                    "section_title": sec_cfg["title"],
                    "severity": sec_cfg["severity"],
                    "category": sec_cfg["category"],
                    "chunk_index": idx,
                    "total_chunks_in_section": len(splits),
                    "character_count": len(full_chunk_text),
                }

                chunks.append({
                    "section": section_key,
                    "chunk_index": idx,
                    "content": full_chunk_text,
                    "metadata": chunk_metadata,
                })

        return chunks
