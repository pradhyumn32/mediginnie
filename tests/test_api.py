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


if __name__ == "__main__":
    print("Running Medijinn API Test Suite...")
    test_root()
    test_health()
    test_list_indexed_drugs()
    test_chat_emergency()
    test_chat_side_effects()
    test_chat_streaming()
    print("\n✅ All API Integration Tests Passed Successfully!")
