# MediAgent

An end-to-end LLM-powered multi-agent system for automated hospital appointment scheduling. The system takes a patient's natural language symptom description and returns a complete appointment with an assigned doctor, time slot, and personalized pre-visit instructions — without human intervention.

---

## How It Works

```
Patient Text (Natural Language)
        ↓
[Agent 1: Symptom Classifier]   →  Department + Urgency
        ↓
[Agent 2: Appointment Retriever] →  Doctor + Time Slot
        ↓
[Agent 3: Response Generator]   →  Confirmation + Instructions
```

Three specialized agents are wired together in a sequential **LangGraph** pipeline, sharing an `AgentState` TypedDict that each agent enriches before passing to the next.

### Example

**Input:**
```
"I have severe chest pain and shortness of breath for the past 2 hours."
```

**Output:**
```
Department:    Cardiology
Urgency:       Emergency
Doctor:        Dr. Chen Wei
Time Slot:     Monday at 08:00
Confirmation:  Your appointment with Dr. Chen Wei in Cardiology has been confirmed for Monday at 08:00. We're here to help with your concerns.
Instructions:  Please call emergency services if symptoms worsen. Avoid eating or drinking before your visit. Bring a list of current medications. Arrange for someone to drive you.
```

---

## Agents

### Agent 1 — Symptom Classifier

| | |
|---|---|
| **Model** | LLaMA-3.2-3B-Instruct + QLoRA adapter |
| **Input** | Patient symptom description (free text) |
| **Output** | Department + Urgency (`Routine` / `Urgent` / `Emergency`) |
| **Training Data** | 15,830 records from Kaggle Disease Symptom Prediction dataset |
| **GPU** | Tesla T4, 3 epochs |
| **Accuracy** | 96.5% overall · 99.7% department · 99.4% Emergency recall |

**Supported departments:** Cardiology, Neurology, Dermatology, Gastroenterology, Endocrinology, Pulmonology, Infectious Disease, Orthopedics, Urology, General Medicine

Urgency is determined algorithmically during training by summing per-symptom severity weights from the dataset. Scores ≥ 45 or high-risk diseases (e.g. heart attack, stroke) are labeled Emergency; ≥ 20 is Urgent; otherwise Routine.

---

### Agent 2 — Appointment Retriever

| | |
|---|---|
| **Model** | Deterministic (no ML) |
| **Input** | Department name |
| **Output** | Doctor name + earliest available time slot |
| **Backend** | AWS DynamoDB (`DoctorSchedule` table, `us-west-2`) |
| **Data** | 410 appointment slots across 30+ doctors, Monday–Friday |

Queries DynamoDB for available slots filtered by department, then sorts by `(day_of_week, time)` and returns the earliest one.

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
| **Model** | LLaMA-3.2-3B-Instruct + QLoRA adapter |
| **Input** | Patient text, department, doctor, time slot, urgency |
| **Output** | Appointment confirmation + pre-visit instructions |
| **Training Data** | 16,604 records from HuggingFace AI Medical Chatbot dataset |
| **GPU** | NVIDIA A100, 2 epochs |
| **Format Compliance** | 100% on evaluation set |

Generates a warm, professional confirmation message and 2–4 specific pre-visit instructions tailored to the patient's situation.

---

## Project Structure

```
medi-agent/
├── agents/
│   ├── parsing.py                          # Regex-based output parsers
│   ├── symptom_classifier/agent.py         # Agent 1
│   ├── appointment_retriever/agent.py      # Agent 2
│   └── response_generator/agent.py         # Agent 3
│
├── orchestrator/
│   └── graph.py                            # LangGraph pipeline
│
├── data/
│   ├── process_symptom_classifier.py       # Agent 1 data processing
│   ├── process_appointment_retriever.py    # Agent 2 data processing
│   ├── process_response_generator.py       # Agent 3 data processing
│   ├── upload_to_dynamodb.py               # Upload schedules to DynamoDB
│   ├── raw/                                # Raw input datasets (not tracked)
│   └── processed/                          # Processed data & adapters (not tracked)
│
├── colab/
│   ├── train_symptom_classifier.ipynb      # Agent 1 fine-tuning (T4 GPU)
│   ├── train_response_generator.ipynb      # Agent 3 fine-tuning (A100 GPU)
│   ├── eval_symptom_classifier.ipynb       # Agent 1 evaluation
│   ├── eval_response_generator.ipynb       # Agent 3 evaluation
│   ├── integration_test.ipynb              # End-to-end pipeline test
│   └── test_dept_fix.ipynb                 # Department mapping debug
│
├── tests/
│   ├── test_schemas.py
│   ├── test_symptom_classifier.py
│   ├── test_appointment_retriever.py
│   ├── test_response_generator.py
│   ├── test_data_processing.py
│   ├── eval_symptom_classifier.py
│   └── eval_response_generator.py
│
├── docs/
│   └── mediagent_report.md
│
├── schemas.py                              # Pydantic models + AgentState
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
AWS_REGION=us-west-2
DYNAMODB_TABLE_NAME=DoctorSchedule
```

Also configure AWS credentials via `~/.aws/credentials` or environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).

### 3. Prepare Data

Place the following raw data files:

```
data/raw/disease_symptom_prediction/
    dataset.csv
    Symptom-severity.csv
    symptom_Description.csv
    symptom_precaution.csv

data/raw/doctor_schedules/
    doctors.csv
```

Then run the data processing scripts in order:

```bash
# Agent 1: symptom classification data
python data/process_symptom_classifier.py

# Agent 2: doctor schedule data
python data/process_appointment_retriever.py

# Agent 3: response generation data (requires HuggingFace access)
python data/process_response_generator.py

# Upload doctor schedules to DynamoDB
python data/upload_to_dynamodb.py
```

### 4. Train the Models

Use the Google Colab notebooks:

| Notebook | GPU | Time |
|---|---|---|
| `colab/train_symptom_classifier.ipynb` | Tesla T4 | ~2 hours |
| `colab/train_response_generator.ipynb` | NVIDIA A100 | ~2–3 hours |

Trained adapters are saved to `data/processed/symptom_classifier_adapter/` and `data/processed/response_generator_adapter/`. These are excluded from git due to file size — store them on Google Drive or HuggingFace Hub.

---

## Running the Pipeline

```python
from orchestrator.graph import run

result = run("I have severe chest pain and shortness of breath")

print(result["department"])    # Cardiology
print(result["urgency"])       # Emergency
print(result["doctor"])        # Dr. Chen Wei
print(result["time_slot"])     # Monday at 08:00
print(result["confirmation"])  # Your appointment with...
print(result["instructions"])  # Please arrive 15 minutes early...
```

---

## Testing

```bash
# Run all unit tests (no GPU or AWS required)
pytest tests/ -v
```

Tests cover:
- Pydantic schema validation (`test_schemas.py`)
- Output parsing with edge cases (`test_symptom_classifier.py`, `test_response_generator.py`)
- DynamoDB query logic with mocks (`test_appointment_retriever.py`)
- Department mapping sanity checks (`test_data_processing.py`)

For full end-to-end testing with all three models loaded, use `colab/integration_test.ipynb`.

---

## Fine-tuning Details

Both LLaMA agents use **QLoRA** (4-bit quantization + Low-Rank Adaptation) for efficient fine-tuning:

| | Agent 1 (Classifier) | Agent 3 (Generator) |
|---|---|---|
| Base Model | LLaMA-3.2-3B-Instruct | LLaMA-3.2-3B-Instruct |
| LoRA Rank | 8 | 16 |
| LoRA Alpha | 16 | 32 |
| Target Modules | `q_proj`, `v_proj` | All 7 projection layers |
| Trainable Params | 2.29M (0.07%) | 24.31M (0.75%) |
| Quantization | 4-bit NF4, float16 | 4-bit NF4, bfloat16 |
| Epochs | 3 | 2 |
| Effective Batch Size | 8 | 16 |

---

## Known Limitations

- **Urgency accuracy on free-form text**: ~60% on natural language integration tests. Training urgency labels are derived from statistical symptom severity scores, not clinical triage standards (Manchester Triage System / ESI).
- **Department accuracy on free-form text**: ~80% on natural language vs. 99.7% on structured symptom lists. Distribution gap between training (structured) and inference (free-form) inputs.
- **Latency**: 21–87 seconds per request on shared Colab GPU. A dedicated inference endpoint is needed for production use.
- **No API or frontend** yet — FastAPI and Streamlit integration are planned.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Fine-tuning | `transformers`, `peft`, `trl`, `bitsandbytes` |
| Orchestration | `langgraph` |
| Database | AWS DynamoDB (`boto3`) |
| Validation | `pydantic` |
| Testing | `pytest` |
| Training Platform | Google Colab (T4 / A100) |
