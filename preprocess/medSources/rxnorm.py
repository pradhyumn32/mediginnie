# RxNorm (NLM/NIH) REST API Client for Medijinn
# Provides canonical drug entity resolution, brand-to-generic mapping,
# and active ingredient concept resolution.

import logging
from typing import Any
import requests

logger = logging.getLogger(__name__)

RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"


class RxNormSource:
    """
    Client for the National Library of Medicine (NLM) RxNav API.
    Resolves commercial brand names to active ingredients and finds synonyms.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def resolve_drug_name(self, name: str) -> dict[str, Any]:
        """
        Takes any drug name (brand or generic) and resolves:
        - generic_name (active ingredient)
        - rxcui (ingredient concept identifier)
        - brand_names (associated commercial brands)
        """
        clean_name = name.strip()
        logger.info("Resolving drug name '%s' via RxNorm...", clean_name)

        try:
            found_rxcui: str | None = None

            # 1a. Try exact concept lookup first
            try:
                exact_url = f"{RXNORM_BASE_URL}/rxcui.json"
                exact_resp = requests.get(exact_url, params={"name": clean_name}, timeout=self.timeout)
                if exact_resp.status_code == 200:
                    exact_ids = exact_resp.json().get("idGroup", {}).get("rxnormId", []) or []
                    if exact_ids:
                        found_rxcui = exact_ids[0]
            except Exception:
                pass

            # 1b. If not found, query drugs endpoint to find concept properties
            if not found_rxcui:
                url = f"{RXNORM_BASE_URL}/drugs.json"
                resp = requests.get(url, params={"name": clean_name}, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()

                concept_groups = data.get("drugGroup", {}).get("conceptGroup", []) or []

                # Prioritize single ingredients (IN, PIN, MIN) or exact name matches
                for group in concept_groups:
                    props = group.get("conceptProperties", []) or []
                    for prop in props:
                        if prop.get("name", "").strip().lower() == clean_name.lower():
                            found_rxcui = prop.get("rxcui")
                            break
                    if found_rxcui:
                        break

                if not found_rxcui:
                    for group in concept_groups:
                        props = group.get("conceptProperties", []) or []
                        if props:
                            found_rxcui = props[0].get("rxcui")
                            break

            if not found_rxcui:
                logger.info("No direct concept found for '%s' in RxNorm.", clean_name)
                return {
                    "query": clean_name,
                    "generic_name": clean_name.upper(),
                    "rxcui": None,
                    "brand_names": [],
                    "ingredients": [clean_name.upper()],
                }

            # 2. Find base ingredient (IN) for this RxCUI
            related_url = f"{RXNORM_BASE_URL}/rxcui/{found_rxcui}/related.json"
            ing_resp = requests.get(related_url, params={"tty": "IN"}, timeout=self.timeout)
            ing_resp.raise_for_status()
            ing_data = ing_resp.json()

            ing_groups = ing_data.get("relatedGroup", {}).get("conceptGroup", []) or []
            ingredients: list[str] = []
            ingredient_rxcui: str | None = None

            for group in ing_groups:
                for prop in group.get("conceptProperties", []):
                    ing_name = prop.get("name")
                    if ing_name:
                        ingredients.append(ing_name.upper())
                    if not ingredient_rxcui:
                        ingredient_rxcui = prop.get("rxcui")

            canonical_generic = " AND ".join(ingredients) if ingredients else clean_name.upper()

            # 3. Find brand names (BN) associated with this ingredient
            brand_names: list[str] = []
            if ingredient_rxcui:
                brand_resp = requests.get(
                    f"{RXNORM_BASE_URL}/rxcui/{ingredient_rxcui}/related.json",
                    params={"tty": "BN"},
                    timeout=self.timeout,
                )
                if brand_resp.status_code == 200:
                    brand_groups = brand_resp.json().get("relatedGroup", {}).get("conceptGroup", []) or []
                    for bg in brand_groups:
                        for bp in bg.get("conceptProperties", []):
                            b_name = bp.get("name")
                            if b_name:
                                brand_names.append(b_name.upper())

            # Ensure the queried name is in brand names if it was a brand
            if clean_name.upper() not in ingredients and clean_name.upper() not in brand_names:
                brand_names.insert(0, clean_name.upper())

            return {
                "query": clean_name,
                "generic_name": canonical_generic,
                "rxcui": ingredient_rxcui or found_rxcui,
                "brand_names": sorted(list(set(brand_names))),
                "ingredients": ingredients or [clean_name.upper()],
            }

        except Exception as exc:
            logger.warning("RxNorm resolution failed for '%s': %s", clean_name, exc)
            return {
                "query": clean_name,
                "generic_name": clean_name.upper(),
                "rxcui": None,
                "brand_names": [],
                "ingredients": [clean_name.upper()],
            }

    def get_brand_names(self, generic_name: str) -> list[str]:
        """
        Convenience helper to retrieve all known brand names for a generic active ingredient.
        """
        res = self.resolve_drug_name(generic_name)
        return res.get("brand_names", [])


if __name__ == "__main__":
    rx = RxNormSource()
    print("Testing brand lookup 'Glucophage':")
    print(rx.resolve_drug_name("Glucophage"))
    print("\nTesting generic lookup 'Ibuprofen':")
    print(rx.resolve_drug_name("Ibuprofen"))
