from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    # Returns HTML web interface if frontend exists, or JSON status
    if "application/json" in response.headers.get("content-type", ""):
        data = response.json()
        assert data["app"] == "Medijinn"
        assert data["status"] == "online"
    else:
        assert "Medijinn" in response.text


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    # assert data["database"] == "connected"
    assert data["database"] == "connected"
    assert data["total_drugs"] > 0
    assert data["total_chunks"] > 0
    print(f"\n[Health Check] Total Drugs: {data['total_drugs']}, Total Chunks: {data['total_chunks']}")


def test_list_indexed_drugs():
    response = client.get("/api/ingest/drugs")
    assert response.status_code == 200
    drugs = response.json()
    assert isinstance(drugs, list)
    assert len(drugs) > 0
    names = [d["generic_name"] for d in drugs]
    print(f"\n[Indexed Drugs] {names}")


def test_chat_emergency():
    payload = {
        "message": "I took Metformin and now my throat is swelling and I cannot breathe!"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_emergency"] is True
    assert data["triage_level"] == "EMERGENCY"
    assert "CRITICAL MEDICAL EMERGENCY" in data["response"]
    print(f"\n[Emergency Triage Test] Alert successfully triggered.")


def test_chat_side_effects():
    payload = {
        "message": "What are the common side effects of Metformin like nausea?"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_emergency"] is False
    assert len(data["sources"]) > 0
    assert "DISCLAIMER" in data["disclaimer"]
    print(f"\n[Side Effect Query] Sources cited: {len(data['sources'])}")


def test_chat_streaming():
    payload = {
        "message": "Does Lisinopril cause a dry cough?"
    }
    with client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200
        events = []
        for line in response.iter_lines():
            if line:
                events.append(line)
        assert any("event: triage" in ev for ev in events)
        assert any("event: entities" in ev for ev in events)
        assert any("event: done" in ev for ev in events)
        print(f"\n[Streaming Test] Received {len(events)} SSE stream frames.")


def test_chat_missing_drug_refusal():
    payload = {
        "message": "What are the side effects of XyzFakeMedication?"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_emergency"] is False
    assert len(data["sources"]) == 0
    # Must refuse cleanly without citing or mentioning other random drugs
    assert "not have official" in data["response"].lower() or "not present" in data["response"].lower()
    # Ensure no cross-pollution of unrelated indexed drugs
    assert "arbutin" not in data["response"].lower()
    assert "deodorant" not in data["response"].lower()
    print(f"\n[Missing Drug Test] Clean refusal verified with 0 irrelevant sources.")


def test_chat_auto_ingestion():
    payload = {
        "message": "What are the side effects of Ozempic?"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_emergency"] is False
    assert len(data["sources"]) > 0
    assert "SEMAGLUTIDE" in str(data["sources"][0]) or "SEMAGLUTIDE" in data["canonical_drugs"]
    print(f"\n[Auto-Ingestion Test] Successfully ingested & retrieved {len(data['sources'])} sources.")


def test_chat_drug_uses():
    payload = {
        "message": "What are the uses of Pantoprazole?"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_emergency"] is False
    assert data["intent"] == "indications"
    assert len(data["sources"]) > 0
    assert any("indication" in s["section"].lower() or "prescribed" in s["section"].lower() for s in data["sources"])
    print(f"\n[Drug Uses Test] Successfully retrieved {len(data['sources'])} indication chunks.")


def test_chat_history_memory():
    # Turn 1: Initial query establishing drug context
    turn1_payload = {
        "message": "What are the uses of Pantoprazole?"
    }
    r1 = client.post("/api/chat", json=turn1_payload)
    assert r1.status_code == 200
    d1 = r1.json()
    assert "PANTOPRAZOLE" in str(d1["canonical_drugs"])

    # Turn 2: Follow-up question using pronoun 'its' without repeating drug name
    turn2_payload = {
        "message": "What are its common side effects?",
        "chat_history": [
            {"role": "user", "content": "What are the uses of Pantoprazole?"},
            {"role": "assistant", "content": d1["response"]},
        ],
    }
    r2 = client.post("/api/chat", json=turn2_payload)
    assert r2.status_code == 200
    d2 = r2.json()
    # Must remember Pantoprazole from previous turn
    assert any("PANTOPRAZOLE" in d for d in d2["canonical_drugs"])
    assert d2["intent"] == "side_effects"
    assert len(d2["sources"]) > 0
    print(f"\n[Chat History Memory Test] Follow-up pronoun 'its' successfully resolved to {d2['canonical_drugs']}.")

    # Turn 3: Cross-turn interaction query ('Can I take it with Ibuprofen?')
    turn3_payload = {
        "message": "Can I take it with Ibuprofen?",
        "chat_history": [
            {"role": "user", "content": "What are the uses of Pantoprazole?"},
            {"role": "assistant", "content": d1["response"]},
            {"role": "user", "content": "What are its common side effects?"},
            {"role": "assistant", "content": d2["response"]},
        ],
    }
    r3 = client.post("/api/chat", json=turn3_payload)
    assert r3.status_code == 200
    d3 = r3.json()
    assert d3["intent"] == "interaction"
    assert any("PANTOPRAZOLE" in d for d in d3["canonical_drugs"])
    assert any("IBUPROFEN" in d for d in d3["canonical_drugs"])
    print(f"\n[Cross-Turn Interaction Test] Successfully combined chat history drug with new drug: {d3['canonical_drugs']}.")


def test_source_citation_traceability_schema():
    from api.schemas import SourceCitation
    citation = SourceCitation(
        drug="METFORMIN",
        formulation="Oral Tablet",
        section="BOXED WARNING",
        severity="CRITICAL",
        source="openfda",
        source_url="https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=123",
        revision_date="2023-01-01",
        matched_passage="Postmarketing cases of metformin-associated lactic acidosis...",
        chunk_id="chunk-xyz",
    )
    dumped = citation.model_dump()
    assert dumped["source_url"] == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=123"
    assert dumped["matched_passage"] is not None
    assert dumped["formulation"] == "Oral Tablet"
    assert dumped["revision_date"] == "2023-01-01"


def test_emergency_safety_guardrails():
    from rag.guardrails import MedicalGuardrails
    # Test colloquial/unexpected emergency phrasing
    q1 = "Feels like an elephant is sitting on my chest and I can't breathe"
    res1 = MedicalGuardrails.classify_safety(q1)
    assert res1["is_emergency"] is True
    assert res1["triage_level"] == "EMERGENCY"

    q2 = "My toddler swallowed cleaner from under the sink"
    res2 = MedicalGuardrails.classify_safety(q2)
    assert res2["is_emergency"] is True
    assert res2["triage_level"] == "EMERGENCY"


def test_prescriptive_intent_blocking():
    from rag.guardrails import MedicalGuardrails
    from graph.workflow import run_clinical_query

    q = "Should I double my dose of Metformin if my blood sugar is high?"
    res = MedicalGuardrails.classify_safety(q)
    assert res["is_emergency"] is False
    assert res["is_blocked"] is True
    assert res["triage_level"] == "BLOCKED"
    assert "NON-PRESCRIPTIVE CLINICAL BOUNDARY" in res["emergency_message"]

    # Graph workflow integration check
    graph_res = run_clinical_query(q)
    assert graph_res["triage_level"] == "BLOCKED"
    assert "NON-PRESCRIPTIVE CLINICAL BOUNDARY" in graph_res["final_response"]
    assert len(graph_res["sources"]) == 0


def test_pairwise_interaction_model_and_lookup():
    from db.models import DrugInteraction
    from graph.nodes import lookup_pairwise_interactions
    import uuid

    interaction = DrugInteraction(
        id=uuid.uuid4(),
        drug_a="sildenafil",
        drug_b="nitroglycerin",
        severity="CONTRAINDICATED",
        description="Severe hypotension",
        clinical_management="Avoid co-administration",
        source="clinical_pairwise_catalog",
    )
    assert interaction.drug_a == "sildenafil"
    assert interaction.severity == "CONTRAINDICATED"

    # Verify lookup function handles single or empty drugs gracefully without throwing
    assert lookup_pairwise_interactions([]) == []
    assert lookup_pairwise_interactions(["metformin"]) == []


def test_security_protection_suite():
    from api.routes.chat import scrub_pii
    from config.settings import settings

    # 1. PII Scrubbing check
    raw_text = "Patient John Doe with email john.doe@hospital.org and phone 555-123-4567, SSN 123-45-6789 takes Metformin."
    scrubbed = scrub_pii(raw_text)
    assert "john.doe@hospital.org" not in scrubbed
    assert "555-123-4567" not in scrubbed
    assert "123-45-6789" not in scrubbed
    assert "[EMAIL REDACTED]" in scrubbed
    assert "[PHONE REDACTED]" in scrubbed
    assert "[SSN REDACTED]" in scrubbed

    # 2. Ingest API Key Protection check
    old_key = settings.ADMIN_API_KEY
    try:
        settings.ADMIN_API_KEY = "test-secret-key-123"
        # Request without key
        res_unauthorized = client.post("/api/ingest", json={"drug_name": "aspirin"})
        assert res_unauthorized.status_code == 403
        assert "Valid X-API-Key" in res_unauthorized.json()["detail"]

        # Request with wrong key
        res_forbidden = client.post("/api/ingest", json={"drug_name": "aspirin"}, headers={"X-API-Key": "wrong-key"})
        assert res_forbidden.status_code == 403

        # Request with valid key should pass auth barrier
        headers = {"X-API-Key": "test-secret-key-123"}
        res_authorized = client.post("/api/ingest", json={"drug_name": "aspirin"}, headers=headers)
        assert res_authorized.status_code != 403
    finally:
        settings.ADMIN_API_KEY = old_key


if __name__ == "__main__":
    print("Running Medijinn API Test Suite...")
    test_root()
    test_health()
    test_list_indexed_drugs()
    test_chat_emergency()
    test_chat_side_effects()
    test_chat_drug_uses()
    test_chat_history_memory()
    test_chat_missing_drug_refusal()
    test_chat_auto_ingestion()
    test_chat_streaming()
    test_source_citation_traceability_schema()
    test_emergency_safety_guardrails()
    test_prescriptive_intent_blocking()
    test_pairwise_interaction_model_and_lookup()
    test_security_protection_suite()
    from tests.test_rag_eval import test_rag_eval_suite
    test_rag_eval_suite()
    print("\n✅ All API Integration Tests Passed Successfully!")