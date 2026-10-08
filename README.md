# Medijinn 🩺
### Clinical-Grade Drug Side-Effect & Safety Intelligence Platform

Medijinn is a production-level, retrieval-augmented generation (RAG) pharmaceutical platform designed to predict and explain medication side effects, adverse drug reactions, boxed warnings, and drug-drug interactions.

---

## 🏛 System Architecture

```
User / Web Client
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                       FastAPI Gateway                       │
│      (CORS, SSE Streaming, Schemas, Health Monitoring)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             LangGraph Clinical Reasoning Engine             │
│                                                             │
│  [Emergency Triage Guardrail] ──(Life Threat)──> [911 Directive]
│               │ (Normal)                                    │
│               ▼                                             │
│  [RxNorm Brand-to-Generic Entity Extraction]                │
│               │                                             │
│               ▼                                             │
│  [PostgreSQL pgvector Hybrid Dense + Trigram Sparse RRF]    │
│               │                                             │
│               ▼                                             │
│  [Grounded LLM Synthesis (Groq / HF / OpenAI) with Citations]│
└─────────────────────────────────────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌───────────────────────┐             ┌───────────────────────┐
│  Data Ingestion Plane │             │     Storage Layer     │
│  - OpenFDA JSON API   │             │  - PostgreSQL 18      │
│  - DailyMed SPL XML   │             │  - pgvector HNSW      │
│  - RxNav / RxNorm     │             │  - GIN Trigram Search │
└───────────────────────┘             └───────────────────────┘
```

---

## 🚀 Key Clinical & Engineering Features

1. **Section-Aware Clinical Chunking**:
   - Preserves FDA SPL sections: `boxed_warning` (Black Box), `adverse_reactions`, `warnings_and_cautions`, `contraindications`, `drug_interactions`, `overdosage`.
   - Tags each chunk with clinical severity tiers (`CRITICAL`, `HIGH`, `SIDE_EFFECT`, `INTERACTION`, `EMERGENCY`).
   - Prepends contextual headers to prevent cross-drug false positives.

2. **Multi-Source Clinical Ingestion**:
   - **OpenFDA API**: Fast structured drug label queries.
   - **DailyMed (NLM/NIH)**: Authoritative HL7 SPL XML parser with LOINC medical ontology codes.
   - **RxNorm (NLM/NIH)**: Automatic commercial brand-to-generic resolution (e.g. *Glucophage* $\rightarrow$ *Metformin*, *Propecia* $\rightarrow$ *Finasteride*).

3. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**:
   - Dense vector cosine similarity via `pgvector` HNSW index ($M=16, ef=64$).
   - Sparse keyword matching via PostgreSQL `pg_trgm` GIN index.
   - Combines scores using Reciprocal Rank Fusion:
     $$RRF\_score(d) = \sum_{m \in \{dense, sparse\}} \frac{1}{60 + rank_m(d)}$$

4. **Emergency Safety Triage**:
   - Scans queries for acute life-threats (anaphylaxis/throat swelling, cardiac infarction signs, acute stroke signs, overdose, severe hemorrhage).
   - Short-circuits generation immediately to an urgent emergency directive (Call 911 / 112 / 108 / Poison Control).
   - Enforces non-prescriptive, educational medical boundaries on every response.

5. **Real-Time Streaming Web UI**:
   - Built-in responsive dark-mode chat interface served directly by FastAPI.
   - Server-Sent Events (SSE) streaming with live entity pills and expandable clinical source citation cards.
   - Interactive drug knowledge catalog and on-demand label ingestion modal.

---

## 🛠 Local Setup & Running

### 1. Prerequisites
- Python 3.12+
- Docker & Docker Compose (for PostgreSQL + pgvector)

### 2. Start PostgreSQL with pgvector
```bash
docker compose up -d postgres
```

### 3. Install Python Dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure Environment
Create a `.env` file in the root directory:
```ini
DATABASE_URL=postgresql://postgres:medjinn-password-270903@localhost:5432/medjinn-db
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
EMBEDDING_MODEL_NAME=BAAI/bge-base-en-v1.5
```

### 5. Ingest Medications
```bash
# Auto mode resolves via RxNorm and queries OpenFDA / DailyMed
python -m preprocess.pipeline --drugs METFORMIN LISINOPRIL AMOXICILLIN IBUPROFEN ATORVASTATIN --source auto
```

### 6. Launch the Server
```bash
python main.py --port 8000
```
- **Web Chatbot UI**: Open [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Docker Deployment

To build and run the entire stack (PostgreSQL + FastAPI + Frontend) in isolated containers:

```bash
docker compose up --build -d
```

Check service status:
```bash
docker compose ps
curl http://localhost:8000/api/health
```

---

## 📡 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the web chatbot interface |
| `GET` | `/api/health` | PostgreSQL status and chunk count metrics |
| `POST` | `/api/chat` | Synchronous grounded clinical reasoning |
| `POST` | `/api/chat/stream` | Real-time Server-Sent Events (SSE) streaming |
| `POST` | `/api/ingest` | On-demand drug ingestion from FDA/DailyMed |
| `GET` | `/api/ingest/drugs` | List of all indexed medications & brand names |
| `GET` | `/docs` | OpenAPI / Swagger interactive documentation |

---

## 🧪 Running Integration Tests

```bash
PYTHONPATH=. pytest tests/test_api.py -v
# or
PYTHONPATH=. python tests/test_api.py
```
