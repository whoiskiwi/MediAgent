# MediAgent

An end-to-end LLM-powered multi-agent system for automated hospital appointment scheduling. The system takes a patient's natural language symptom description and returns a complete appointment with an assigned doctor, time slot, personalized pre-visit instructions, and RAG-enhanced possible causes — without human intervention.

---

## How It Works

```
Patient Text + Age (optional) + Gender (optional)
        ↓
[Agent 1: Symptom Classifier]      →  Department + Urgency
        ↓
[Agent 2: Appointment Retriever]   →  Doctor + Time Slot
        ↓
[Agent 3: Response Generator]      →  Confirmation + Instructions
        ↓
[Agent 4: Causes Generator]        →  Possible Causes (RAG-enhanced)
```

Four specialized agents are wired together in a sequential **LangGraph** pipeline, sharing an `AgentState` TypedDict that each agent enriches before passing to the next. Agents 1 and 3 run on **AWS SageMaker** endpoints; Agent 2 queries **AWS DynamoDB**; Agent 4 calls **DeepSeek** (with OpenAI fallback) augmented by **RAG over MedlinePlus**.

### Example

**Input:**
```json
{
  "symptom": "I have severe chest pain and shortness of breath for the past 2 hours.",
  "age": 55,
  "gender": "Male"
}
```

**Output:**
```json
{
  "agent1": { "department": "Cardiology", "urgency": "Emergency" },
  "agent2": { "doctor": "Dr. Chen Wei", "time_slot": "Monday at 08:00" },
  "agent3": {
    "confirmation": "Your appointment with Dr. Chen Wei in Cardiology has been confirmed for Monday at 08:00.",
    "instructions": "Avoid eating or drinking before your visit. Bring a list of current medications.",
    "possible_causes": [
      {
        "cause": "Acute Myocardial Infarction",
        "reason": "Chest pain with shortness of breath in a 55-year-old male is a classic presentation of heart attack.",
        "reference": { "title": "Heart Attack", "url": "https://medlineplus.gov/heartattack.html" }
      }
    ]
  }
}
```

> When urgency is `Emergency`, the frontend displays a prominent alert to call 911 immediately.

---

## Architecture

```
Browser / Streamlit (app.py)
        ↓  HTTP
FastAPI (main.py)
  ├── POST /api/v1/query             ← full pipeline (age/gender optional)
  ├── POST /api/v1/qa                ← medical Q&A via RAG
  ├── POST /api/v1/drug              ← drug lookup via RAG
  ├── POST /api/v1/auth/login
  ├── POST /api/v1/auth/register
  ├── GET  /api/v1/auth/google       ← Google OAuth
  └── GET  /api/v1/appointments

LangGraph Orchestrator (orchestrator/graph.py)
  ├── Agent 1 → SageMaker endpoint (medi-agent-classifier)
  ├── Agent 2 → DynamoDB (DoctorSchedule)
  ├── Agent 3 → SageMaker endpoint (medi-agent-generator)
  └── Agent 4 → DeepSeek API / OpenAI fallback + ChromaDB RAG

RAG Knowledge Base
  ├── Source:  MedlinePlus (NIH) — 1,015 English health topics
  ├── Index:   ChromaDB (local persistent, cosine similarity)
  └── Embed:   OpenAI text-embedding-3-small

AWS DynamoDB
  ├── DoctorSchedule          ← appointment slots
  └── medi-agent-appointments ← user appointment history

Auth
  ├── Email/password (JWT)
  └── Google OAuth 2.0
```

---

## Agents

### Agent 1 — Symptom Classifier

| | |
|---|---|
| **Inference** | AWS SageMaker endpoint (`medi-agent-classifier`) |
| **Model** | LLaMA-3.2-3B-Instruct + QLoRA adapter |
| **Input** | Patient symptom description (free text) |
| **Output** | Department + Urgency (`Routine` / `Urgent` / `Emergency`) |
| **Training Data** | 15,830 records from Kaggle Disease Symptom Prediction dataset |
| **GPU** | Tesla T4, 3 epochs |
| **Accuracy** | 96.5% overall · 99.7% department · 99.4% Emergency recall |

**Supported departments:** Cardiology, Neurology, Dermatology, Gastroenterology, Endocrinology, Pulmonology, Infectious Disease, Orthopedics, Urology, General Medicine

---

### Agent 2 — Appointment Retriever

| | |
|---|---|
| **Inference** | Deterministic (no ML) |
| **Input** | Department name |
| **Output** | Doctor name + earliest available time slot |
| **Backend** | AWS DynamoDB (`DoctorSchedule` table, `us-west-2`) |
| **Data** | 410 appointment slots across 30+ doctors, Monday–Friday |

---

### Agent 3 — Response Generator

| | |
|---|---|
| **Inference** | AWS SageMaker endpoint (`medi-agent-generator`) |
| **Model** | LLaMA-3.2-3B-Instruct + QLoRA adapter |
| **Input** | Patient text, department, doctor, time slot, urgency, age (optional), gender (optional) |
| **Output** | Appointment confirmation + pre-visit instructions |
| **Training Data** | 16,604 records from HuggingFace AI Medical Chatbot dataset |
| **GPU** | NVIDIA A100, 2 epochs |
| **Format Compliance** | 100% on evaluation set |

> `Emergency` urgency is normalised to `Urgent` before calling Agent 3, as the model was only trained on `Routine` / `Urgent`.

---

### Agent 4 — Causes Generator (RAG-enhanced)

| | |
|---|---|
| **Inference** | DeepSeek API (primary) / OpenAI GPT-4o-mini (fallback) |
| **Input** | Patient text, department, age (optional), gender (optional) |
| **Output** | 3–5 possible causes, each with a one-sentence reason and MedlinePlus reference |
| **RAG** | Retrieves top-4 relevant MedlinePlus documents via ChromaDB semantic search |
| **Knowledge Base** | 1,015 NIH MedlinePlus health topics, embedded with `text-embedding-3-small` |

---

## Frontend Features

| Tab | Description |
|---|---|
| **Symptom Query** | Submit symptoms with optional age/gender; shows department, doctor, urgency, medical advice, and RAG-enhanced possible causes with references |
| **Medical Q&A** | Ask any medical question; answered by DeepSeek using MedlinePlus as context, with source links |
| **Drug Lookup** | Enter a drug name; returns uses, side effects, and precautions from MedlinePlus |

**Emergency alert:** When urgency = `Emergency`, a red banner prompts the user to call 911 immediately.

---

## Project Structure

```
medi-agent/
├── agents/
│   ├── parsing.py                          # Regex-based output parsers
│   ├── symptom_classifier/agent.py         # Agent 1 — SageMaker
│   ├── appointment_retriever/agent.py      # Agent 2 — DynamoDB
│   ├── response_generator/agent.py         # Agent 3 — SageMaker
│   ├── causes_generator/agent.py           # Agent 4 — DeepSeek + RAG
│   └── rag/
│       ├── retriever.py                    # ChromaDB semantic search
│       └── qa.py                           # RAG-powered Q&A
│
├── orchestrator/
│   └── graph.py                            # LangGraph pipeline
│
├── api/
│   └── v1/
│       ├── router.py                       # /query, /qa, /drug, /appointments
│       └── auth.py                         # JWT + Google OAuth
│
├── auth/
│   └── dependencies.py                     # JWT validation
│
├── scripts/
│   ├── build_rag_index.py                  # Parse MedlinePlus XML → ChromaDB
│   ├── deploy_sagemaker.py
│   ├── merge_adapter.py
│   └── setup_infra.sh
│
├── data/
│   ├── raw/                                # Raw datasets (not tracked)
│   ├── processed/                          # Processed data & adapters (not tracked)
│   └── chroma_db/                          # Vector index (not tracked, rebuild locally)
│
├── colab/
│   ├── train_symptom_classifier.ipynb
│   ├── train_response_generator.ipynb
│   ├── eval_symptom_classifier.ipynb
│   └── eval_response_generator.ipynb
│
├── tests/
│
├── app.py                                  # Streamlit frontend
├── main.py                                 # FastAPI entry point
├── schemas.py                              # Pydantic models + AgentState
├── Dockerfile
└── requirements.txt
```

---

## Setup

### 1. Install Dependencies

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-west-2

AGENT1_ENDPOINT_NAME=medi-agent-classifier
AGENT3_ENDPOINT_NAME=medi-agent-generator
DYNAMODB_TABLE_NAME=DoctorSchedule
APPOINTMENTS_TABLE=medi-agent-appointments

JWT_SECRET=your_jwt_secret

GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=http://localhost:8000/api/v1/auth/google/callback
FRONTEND_URL=http://localhost:8501

DEEPSEEK_API_KEY=your_deepseek_key
OPENAI_API_KEY=your_openai_key
```

### 3. Build the RAG Index

Download the MedlinePlus XML from [medlineplus.gov](https://medlineplus.gov/xml.html) and place it in `data/`, then run:

```bash
python scripts/build_rag_index.py
```

This parses 1,015 English health topics and builds a persistent ChromaDB index at `data/chroma_db/`. Only needs to run once.

### 4. Run Locally

```bash
# Backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --reload-dir . --reload-exclude "venv/*"

# Frontend (separate terminal)
streamlit run app.py
```

Health check: `curl http://localhost:8000/api/v1/health`

---

## API

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/api/v1/health` | None | Health check |
| `POST` | `/api/v1/auth/register` | None | Register new user |
| `POST` | `/api/v1/auth/login` | None | Login, returns JWT |
| `GET` | `/api/v1/auth/google` | None | Google OAuth login |
| `POST` | `/api/v1/query` | Optional JWT | Run full pipeline |
| `GET` | `/api/v1/appointments` | Required JWT | Appointment history |
| `POST` | `/api/v1/qa` | None | Medical Q&A (RAG) |
| `POST` | `/api/v1/drug` | None | Drug lookup (RAG) |

**Query request:**
```json
{
  "symptom": "I have chest pain and shortness of breath",
  "age": 55,
  "gender": "Male"
}
```

---

## RAG Knowledge Base

| | |
|---|---|
| **Source** | NIH MedlinePlus (`mplus_topics_*.xml`) |
| **Coverage** | 1,015 English health topics across 30+ medical categories |
| **Vector Store** | ChromaDB (local persistent) |
| **Embedding Model** | OpenAI `text-embedding-3-small` |
| **Similarity** | Cosine |
| **Used by** | Agent 4 (possible causes), `/qa`, `/drug` endpoints |

Top covered categories: Infections (126), Brain & Nerves (102), Cardiology (93), Digestive System (79), Bones & Muscles (78), Oncology (66), Skin (63).

---

## CI/CD

Every push to `main` triggers the GitHub Actions pipeline:

```
push to main → Unit Tests (pytest) → Docker build → ECR push → ECS Fargate deploy
```

**Required GitHub Secrets:** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

---

## Training

Both LLaMA agents use **QLoRA** (4-bit quantization + Low-Rank Adaptation):

| | Agent 1 (Classifier) | Agent 3 (Generator) |
|---|---|---|
| Base Model | LLaMA-3.2-3B-Instruct | LLaMA-3.2-3B-Instruct |
| LoRA Rank | 8 | 16 |
| LoRA Alpha | 16 | 32 |
| Trainable Params | 2.29M (0.07%) | 24.31M (0.75%) |
| Quantization | 4-bit NF4, float16 | 4-bit NF4, bfloat16 |
| Epochs | 3 | 2 |
| GPU | Tesla T4 | NVIDIA A100 |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Fine-tuning | `transformers`, `peft`, `trl`, `bitsandbytes` |
| Orchestration | `langgraph` |
| API | `fastapi`, `uvicorn` |
| Frontend | `streamlit` |
| Auth | JWT (`python-jose`, `passlib`), Google OAuth 2.0 |
| Database | AWS DynamoDB (`boto3`) |
| Inference | AWS SageMaker (Agents 1 & 3), DeepSeek API / OpenAI (Agent 4) |
| RAG | ChromaDB, OpenAI Embeddings, MedlinePlus |
| Deployment | Docker, Amazon ECR, ECS Fargate |
| CI/CD | GitHub Actions |
| Testing | `pytest` |
| Training Platform | Google Colab (T4 / A100) |
