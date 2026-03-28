# MediAgent

An end-to-end LLM-powered multi-agent system for automated hospital appointment scheduling. The system takes a patient's natural language symptom description and returns a complete appointment with an assigned doctor, time slot, and personalized pre-visit instructions — without human intervention.

---

## How It Works

```
Patient Text (Natural Language)
        ↓
[Agent 1: Symptom Classifier]    →  Department + Urgency
        ↓
[Agent 2: Appointment Retriever] →  Doctor + Time Slot
        ↓
[Agent 3: Response Generator]    →  Confirmation + Instructions
```

Three specialized agents are wired together in a sequential **LangGraph** pipeline, sharing an `AgentState` TypedDict that each agent enriches before passing to the next. Agents 1 and 3 run on **AWS SageMaker** endpoints; Agent 2 queries **AWS DynamoDB** directly.

### Example

**Input:**
```
"I have severe chest pain and shortness of breath for the past 2 hours."
```

**Output:**
```json
{
  "agent1": { "department": "Cardiology", "urgency": "Emergency" },
  "agent2": { "doctor": "Dr. Chen Wei", "time_slot": "Monday at 08:00" },
  "agent3": {
    "confirmation": "Your appointment with Dr. Chen Wei in Cardiology has been confirmed for Monday at 08:00.",
    "instructions": "Please call emergency services if symptoms worsen. Avoid eating or drinking before your visit. Bring a list of current medications."
  }
}
```

---

## Architecture

```
Browser / Streamlit (app.py)
        ↓  HTTP
FastAPI (main.py)
  ├── POST /api/v1/query       ← calls run_pipeline()
  ├── POST /api/v1/auth/login
  ├── POST /api/v1/auth/register
  └── GET  /api/v1/appointments

LangGraph Orchestrator (orchestrator/graph.py)
  ├── Agent 1 → SageMaker endpoint (medi-agent-classifier)
  ├── Agent 2 → DynamoDB (DoctorSchedule)
  └── Agent 3 → SageMaker endpoint (medi-agent-generator)

AWS DynamoDB
  ├── DoctorSchedule          ← appointment slots
  └── medi-agent-appointments ← user appointment history
```

Deployed via **GitHub Actions → Docker → Amazon ECR → ECS Fargate**.

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

**DynamoDB Schema:**

| Attribute | Type | Role | Example |
|---|---|---|---|
| `department` | String | Partition Key | `"Cardiology"` |
| `sk` | String | Sort Key | `"Dr. Chen Wei#Monday#08:00"` |
| `doctor` | String | | `"Dr. Chen Wei"` |
| `day` | String | | `"Monday"` |
| `time_slot` | String | | `"08:00"` |
| `available` | Boolean | | `true` |

---

### Agent 3 — Response Generator

| | |
|---|---|
| **Inference** | AWS SageMaker endpoint (`medi-agent-generator`) |
| **Model** | LLaMA-3.2-3B-Instruct + QLoRA adapter |
| **Input** | Patient text, department, doctor, time slot, urgency |
| **Output** | Appointment confirmation + pre-visit instructions |
| **Training Data** | 16,604 records from HuggingFace AI Medical Chatbot dataset |
| **GPU** | NVIDIA A100, 2 epochs |
| **Format Compliance** | 100% on evaluation set |

> `Emergency` urgency is normalised to `Urgent` before calling Agent 3, as the model was only trained on `Routine` / `Urgent`.

---

## Project Structure

```
medi-agent/
├── agents/
│   ├── parsing.py                          # Regex-based output parsers + Q&A sign-off cleaner
│   ├── symptom_classifier/agent.py         # Agent 1 — calls SageMaker
│   ├── appointment_retriever/agent.py      # Agent 2 — queries DynamoDB
│   └── response_generator/agent.py         # Agent 3 — calls SageMaker
│
├── orchestrator/
│   └── graph.py                            # LangGraph pipeline (run_pipeline)
│
├── api/
│   └── v1/
│       ├── router.py                       # /query, /appointments endpoints
│       └── auth.py                         # /auth/login, /auth/register endpoints
│
├── auth/
│   └── dependencies.py                     # JWT token validation
│
├── infrastructure/
│   └── task-definition.json                # ECS task definition
│
├── sagemaker_code/
│   └── inference.py                        # SageMaker inference handler
│
├── scripts/
│   ├── deploy_sagemaker.py                 # Package adapters → S3 → SageMaker endpoints
│   ├── merge_adapter.py                    # Merge QLoRA adapter into base model
│   └── setup_infra.sh                      # AWS infra setup (ECR, ECS, DynamoDB)
│
├── data/
│   ├── process_symptom_classifier.py
│   ├── process_appointment_retriever.py
│   ├── process_response_generator.py
│   ├── upload_to_dynamodb.py
│   ├── raw/                                # Raw input datasets (not tracked)
│   └── processed/                          # Processed data & adapters (not tracked)
│
├── colab/
│   ├── train_symptom_classifier.ipynb      # Agent 1 fine-tuning (T4 GPU)
│   ├── train_response_generator.ipynb      # Agent 3 fine-tuning (A100 GPU)
│   ├── eval_symptom_classifier.ipynb
│   ├── eval_response_generator.ipynb
│   └── integration_test.ipynb
│
├── tests/
│   ├── test_schemas.py
│   ├── test_symptom_classifier.py
│   ├── test_appointment_retriever.py
│   ├── test_response_generator.py
│   ├── test_agents.py
│   ├── test_orchestrator.py
│   └── test_api.py
│
├── app.py                                  # Streamlit frontend
├── main.py                                 # FastAPI entry point
├── schemas.py                              # Pydantic models + AgentState
├── Dockerfile
├── start.sh
└── requirements.txt
```

---

## Setup

### 1. Install Dependencies

```bash
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
```

### 3. Run Locally

```bash
# FastAPI backend
uvicorn main:app --reload --port 8000

# Streamlit frontend (separate terminal)
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
| `POST` | `/api/v1/query` | Optional JWT | Run full pipeline |
| `GET` | `/api/v1/appointments` | Required JWT | Appointment history |

**Query request:**
```json
{ "symptom": "I have chest pain and shortness of breath" }
```

---

## CI/CD

Every push to `main` triggers the GitHub Actions pipeline (`.github/workflows/deploy.yml`):

```
push to main
    ↓
Unit Tests (pytest, no GPU/AWS needed)
    ↓
Build Docker image → push to Amazon ECR
    ↓
Deploy to ECS Fargate
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
| Target Modules | `q_proj`, `v_proj` | All 7 projection layers |
| Trainable Params | 2.29M (0.07%) | 24.31M (0.75%) |
| Quantization | 4-bit NF4, float16 | 4-bit NF4, bfloat16 |
| Epochs | 3 | 2 |
| GPU | Tesla T4 | NVIDIA A100 |

Use the Colab notebooks in `colab/` for training. After training, deploy to SageMaker:

```bash
python scripts/merge_adapter.py
python scripts/deploy_sagemaker.py
```

---

## Testing

```bash
# Unit tests (no GPU or AWS required)
pytest tests/ -v
```

For end-to-end testing with live SageMaker endpoints, use `colab/integration_test.ipynb`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Fine-tuning | `transformers`, `peft`, `trl`, `bitsandbytes` |
| Orchestration | `langgraph` |
| API | `fastapi`, `uvicorn` |
| Frontend | `streamlit` |
| Auth | JWT (`python-jose`, `passlib`) |
| Database | AWS DynamoDB (`boto3`) |
| Inference | AWS SageMaker |
| Deployment | Docker, Amazon ECR, ECS Fargate |
| CI/CD | GitHub Actions |
| Validation | `pydantic` |
| Testing | `pytest` |
| Training Platform | Google Colab (T4 / A100) |
