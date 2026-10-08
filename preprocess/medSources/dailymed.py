# DailyMed (NLM/NIH) REST & SPL XML Adapter for Medijinn
# Authoritative repository for FDA Structured Product Labeling (SPL)

import logging
from typing import Any
import xml.etree.ElementTree as ET
import requests

logger = logging.getLogger(__name__)

DAILYMED_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

# LOINC standard ontology mappings to internal clinical section keys
LOINC_MAP: dict[str, str] = {
    "34066-1": "boxed_warning",
    "34084-4": "adverse_reactions",
    "34071-1": "warnings_and_cautions",
    "34070-3": "contraindications",
    "34073-7": "drug_interactions",
    "34067-9": "indications_and_usage",
    "34088-5": "overdosage",
    "34068-7": "dosage_and_administration",
    "34090-1": "clinical_pharmacology",
    "42228-7": "patient_medication_information",
    "43678-2": "use_in_specific_populations",
}

# Title heuristic mapping for sections with generic LOINC or missing codes
TITLE_FALLBACKS: dict[str, str] = {
    "BOXED WARNING": "boxed_warning",
    "BLACK BOX": "boxed_warning",
    "WARNING: LACTIC ACIDOSIS": "boxed_warning",
    "ADVERSE REACTIONS": "adverse_reactions",
    "SIDE EFFECTS": "adverse_reactions",
    "WARNINGS AND PRECAUTIONS": "warnings_and_cautions",
    "CONTRAINDICATIONS": "contraindications",
    "DRUG INTERACTIONS": "drug_interactions",
    "INDICATIONS AND USAGE": "indications_and_usage",
    "OVERDOSAGE": "overdosage",
    "DOSAGE AND ADMINISTRATION": "dosage_and_administration",
    "CLINICAL PHARMACOLOGY": "clinical_pharmacology",
    "PATIENT COUNSELING": "patient_medication_information",
    "USE IN SPECIFIC POPULATIONS": "use_in_specific_populations",
}


class DailyMedSource:
    """
    Client for National Library of Medicine (NLM) DailyMed API.
    Fetches official FDA Structured Product Labels (SPL) and parses clinical sections.
    """

    def __init__(self, timeout: int = 20):
        self.timeout = timeout

    def search_spls(self, drug_name: str, limit: int = 3) -> list[dict[str, Any]]:
        """
        Searches DailyMed for SPL records by drug name.
        """
        clean_name = drug_name.strip()
        url = f"{DAILYMED_BASE_URL}/spls.json"
        params = {"drug_name": clean_name, "page": 1, "pagesize": limit}

        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", []) or []
        except Exception as exc:
            logger.warning("DailyMed search failed for '%s': %s", clean_name, exc)
            return []

    def fetch_spl_xml(self, setid: str) -> str:
        """
        Fetches raw HL7 SPL XML for a given SPL setid.
        """
        url = f"{DAILYMED_BASE_URL}/spls/{setid}.xml"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.text

    def parse_spl_xml(self, xml_text: str, setid: str, title: str = "") -> dict[str, Any]:
        """
        Parses HL7 SPL XML into structured clinical sections using LOINC ontology.
        """
        root = ET.fromstring(xml_text)
        ns = {"hl7": "urn:hl7-org:v3"}

        sections: dict[str, list[str]] = {}

        for sec in root.iter("{urn:hl7-org:v3}section"):
            code_elem = sec.find("hl7:code", ns)
            title_elem = sec.find("hl7:title", ns)

            loinc_code = code_elem.attrib.get("code") if code_elem is not None else None
            sec_title = " ".join("".join(title_elem.itertext()).split()) if title_elem is not None else ""

            # 1. Match section key by LOINC code
            matched_key: str | None = LOINC_MAP.get(loinc_code or "")

            # 2. Fallback match by section title
            if not matched_key and sec_title:
                upper_title = sec_title.upper()
                for key_phrase, sec_key in TITLE_FALLBACKS.items():
                    if key_phrase in upper_title:
                        matched_key = sec_key
                        break

            if not matched_key:
                continue

            # Extract all textual content within this section (including lists, paragraphs, tables)
            text_chunks: list[str] = []
            for t_elem in sec.iter("{urn:hl7-org:v3}text"):
                t = " ".join("".join(t_elem.itertext()).split())
                if t:
                    text_chunks.append(t)

            if text_chunks:
                combined_sec_text = "\n\n".join(text_chunks)
                if matched_key not in sections:
                    sections[matched_key] = []
                sections[matched_key].append(combined_sec_text)

        # Merge duplicate subsections
        normalized_sections = {k: "\n\n".join(v) for k, v in sections.items() if v}

        return {
            "set_id": setid,
            "title": title,
            "sections": normalized_sections,
        }

    def fetch_drug(self, drug_name: str, limit: int = 1) -> dict[str, Any]:
        """
        End-to-end lookup: Searches DailyMed, fetches SPL XML, and parses
        into normalized dictionary format.
        """
        clean_name = drug_name.strip().upper()
        spls = self.search_spls(clean_name, limit=limit)

        if not spls:
            logger.info("No DailyMed SPLs found for '%s'", clean_name)
            return {"results": []}

        results: list[dict[str, Any]] = []

        for spl_meta in spls:
            setid = spl_meta.get("setid")
            title = spl_meta.get("title", "")
            published_date = spl_meta.get("published_date")

            if not setid:
                continue

            try:
                xml_text = self.fetch_spl_xml(setid)
                parsed = self.parse_spl_xml(xml_text, setid=setid, title=title)

                results.append({
                    "generic_name": clean_name,
                    "brand_names": [title] if title else [],
                    "set_id": setid,
                    "effective_date": published_date,
                    "sections": parsed["sections"],
                    "openfda": {},
                    "source": "dailymed",
                    "title": title,
                })
            except Exception as exc:
                logger.warning("Failed to fetch/parse DailyMed XML for setid %s: %s", setid, exc)

        return {"results": results}


if __name__ == "__main__":
    dm = DailyMedSource()
    print("Testing DailyMed lookup for METFORMIN:")
    res = dm.fetch_drug("METFORMIN", limit=1)
    if res["results"]:
        rec = res["results"][0]
        print(f"Title: {rec['title']}")
        print(f"Set ID: {rec['set_id']}")
        print(f"Sections extracted: {list(rec['sections'].keys())}")
        for s, content in rec["sections"].items():
            print(f"  - {s}: {len(content)} chars")
