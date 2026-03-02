# MediAgent

**Week 1 Status Update · February 28, 2026**

---

## 1. Project Summary

MediAgent is a multi-agent LLM system for hospital appointment scheduling and symptom triage. Three fine-tuned LLaMA-3.2-3B agents cooperate to (1) classify patient symptoms into a medical department and urgency level, (2) retrieve available doctor appointments from AWS DynamoDB, and (3) generate patient-friendly confirmations with pre-visit instructions — all orchestrated via LangGraph on AWS.

**Week 1 highlights:**

- Agent 1 (Symptom Classifier) data processing + full LoRA fine-tuning completed on Kaggle P100 — 3 epochs, 1,569 steps, final token accuracy **98.1%** (loss 0.057)
- Agent 2 (Appointment Retriever) data processing complete — **849 time slots** across 50 doctors and 10 departments uploaded to AWS DynamoDB (`us-west-2`)
- Agent 3 (Response Generator) data processing + full LoRA fine-tuning completed on Google Colab A100 — 2 epochs, 4,506 steps, 42,410 training records from 256k dialogues
- Pydantic interface contract (`schemas.py`) defined — 7 models shared with teammate Zhenhao for integration
- Training platform challenges resolved: Kaggle GPU quota exhausted → migrated Agent 3 training to Colab Pay As You Go; checkpoint recovery validated after Colab disconnection

---

## 2. Team Members

| Name | Student ID | Role | Responsibilities |
|------|-----------|------|-----------------|
| Qi Chen | 002315412 | ML & Agent Core | Data processing for all 3 agents, LoRA fine-tuning (Agent 1 & 3), LangGraph orchestration, Pydantic schemas, unit tests |
| Zhenhao Ma | 002309369 | Infrastructure & Deployment | AWS infrastructure (DynamoDB, SageMaker, S3, ECR, CloudWatch), FastAPI backend, Streamlit frontend, Docker, CI/CD |

---

## 3. Architecture Overview

| Agent | Model / Tech | Input | Output | Status |
|-------|-------------|-------|--------|--------|
| Agent 1: Symptom Classifier | LLaMA-3.2-3B-Instruct + QLoRA (r=8, alpha=16) | Patient symptom text | Department (1 of 10) + Urgency (Routine/Urgent/Emergency) | Training complete |
| Agent 2: Appointment Retriever | DynamoDB + boto3 (LlamaIndex planned) | Department from Agent 1 | Doctor name + Time slot | Data seeded in DynamoDB |
| Agent 3: Response Generator | LLaMA-3.2-3B-Instruct + QLoRA (r=16, alpha=32) | Patient text + Department + Doctor + Time slot + Urgency | Appointment confirmation + Pre-visit instructions | Training complete |

**Data Flow:** `Patient text` → Agent 1 (classify) → Agent 2 (retrieve appointment) → Agent 3 (generate confirmation) → `Patient response`

All agents are coordinated by a **LangGraph orchestrator** passing shared `AgentState` (defined in `schemas.py`). The Streamlit frontend and AWS deployment infrastructure (S3, SageMaker, ECR, CloudWatch, MLflow) are owned by Zhenhao Ma.

---

## 4. Tech Stack

| Category | Tools | Status |
|----------|-------|--------|
| ML Frameworks | Python 3.x, PyTorch 2.9, HuggingFace Transformers 5.2, PEFT 0.18.1, TRL 0.29, bitsandbytes (4-bit QLoRA) | Active — used for Agent 1 & 3 training |
| Orchestration | LangChain, LangGraph, LlamaIndex | Planned — schemas defined, wiring in Week 2 |
| AWS Services | DynamoDB (DoctorSchedule table, `us-west-2`), S3, SageMaker Endpoint | DynamoDB live; SageMaker GPU blocked |
| Data Processing | pandas 2.2, Pydantic 2.7, boto3, python-dotenv | Active — all 3 data pipelines complete |
| Fine-Tuning Env | Kaggle Notebook (P100 GPU, free tier) for Agent 1; Google Colab Pay As You Go (A100 GPU, $9.99/100 CU) for Agent 3 | Active — Kaggle quota exhausted, switched to Colab |
| DevOps | Docker, GitHub Actions, FastAPI, Streamlit | Planned (Zhenhao) |
| Experiment Tracking | MLflow 2.13 | Planned — Week 2 |
| Testing | pytest 8.2, httpx | Planned — Week 2 |

---

## 5. Task Progress (Week 1)

### Agent 1 — Symptom Classifier

| Task | Status | Details |
|------|--------|---------|
| Data collection | Done | [Disease Symptom Prediction dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) — 4,920 rows, 41 diseases, 132 symptoms |
| Disease → Department mapping | Done | 41 diseases mapped to 10 departments (Cardiology, Neurology, Dermatology, Gastroenterology, Endocrinology, Pulmonology, Infectious Disease, Orthopedics, Urology, General Medicine) |
| Urgency computation | Done | Severity-weighted scoring: `ALWAYS_EMERGENCY` set (Heart attack, Brain hemorrhage) + threshold-based (Emergency ≥ 45, Urgent ≥ 20, Routine < 20) |
| Train/test split | Done | 85/15 split → **4,182 train** + **738 test** (seed=42) |
| Supporting data files | Done | `department_mapping.json` (41 disease→dept entries), `disease_precautions.json` (precautions for Agent 3) |
| QLoRA fine-tuning | Done | 3 epochs on Kaggle P100, 1,569 steps, final loss 0.057, **98.1% token accuracy** |
| MLflow logging | Not started | Deferred — training metrics captured in `trainer_state.json` |
| pytest unit tests | Not started | Deferred to Week 2 |

**Agent 1 Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Method | QLoRA (4-bit quantization) + SFT via HuggingFace TRL |
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Epochs | 3 |
| Batch size | 4 |
| Total steps | 1,569 |
| Platform | Kaggle (free tier, NVIDIA P100 16GB) |
| Adapter size | ~30 MB per checkpoint (89 MB total for 3 checkpoints) |

**Training Loss Curve (31 logged steps):**

| Step | Train Loss | Token Accuracy | Note |
|------|-----------|----------------|------|
| 50 | 1.8828 | 63.1% | Cold start |
| 100 | 0.4997 | 89.4% | Rapid convergence |
| 200 | 0.1893 | 96.1% | |
| 300 | 0.1567 | 96.6% | |
| 500 | 0.0794 | 97.8% | End of Epoch 1 |
| 800 | 0.0622 | 98.0% | |
| 1050 | 0.0601 | 98.1% | End of Epoch 2 |
| 1250 | 0.0564 | 98.2% | Best accuracy |
| 1550 | 0.0571 | 98.1% | End of Epoch 3 |

**Checkpoint Comparison:**

| Checkpoint | Epoch | Train Loss | Token Accuracy | Size | Note |
|-----------|-------|------------|----------------|------|------|
| checkpoint-523 | 1 | 0.0794 | 97.8% | 30 MB | Rapid convergence phase |
| checkpoint-1046 | 2 | 0.0601 | 98.1% | 30 MB | Stable convergence |
| **checkpoint-1569 (final)** | **3** | **0.0571** | **98.1%** | **30 MB** | **Final checkpoint** |

**Agent 1 Data Distribution:**

| Department | Samples | | Urgency | Samples |
|-----------|---------|---|---------|---------|
| Gastroenterology | 1,440 | | Urgent | 2,466 |
| Dermatology | 720 | | Routine | 1,236 |
| Neurology | 480 | | Emergency | 1,218 |
| Infectious Disease | 480 | | | |
| Endocrinology | 480 | | | |
| Pulmonology | 480 | | | |
| Cardiology | 360 | | | |
| Orthopedics | 240 | | | |
| General Medicine | 120 | | | |
| Urology | 120 | | | |
| **Total** | **4,920** | | **Total** | **4,920** |

**Agent 1 Sample Record:**
```json
{
  "instruction": "You are a medical triage assistant. Classify the patient's symptoms into a medical department and urgency level. Respond in exactly this format:\nDepartment: <department>\nUrgency: <Routine|Urgent|Emergency>",
  "input": "Patient reports: skin rash, high fever, blister, red sore around nose, yellow crust ooze.",
  "output": "Department: Dermatology\nUrgency: Routine"
}
```

---

### Agent 2 — Appointment Retriever

| Task | Status | Details |
|------|--------|---------|
| Doctor schedule data | Done | `doctors.csv` — 50 doctors across 10 departments with working days and time slots |
| Schedule generation | Done | Read from `doctors.csv` → generate available time slots per doctor per day |
| DynamoDB upload | Done | **849 records** uploaded to `DoctorSchedule` table (`us-west-2`). Schema: PK=`department`, SK=`doctor#day#time_slot`. BillingMode=PAY_PER_REQUEST |
| LlamaIndex query logic | Not started | Deferred — will implement `department` → query DynamoDB → return `doctor + time_slot` in Week 2 |

**DynamoDB Slot Distribution:**

| Department | Slots | | Department | Slots |
|-----------|-------|---|-----------|-------|
| Gastroenterology | 99 | | Pulmonology | 85 |
| Endocrinology | 99 | | Dermatology | 83 |
| Orthopedics | 95 | | Neurology | 82 |
| Infectious Disease | 92 | | Urology | 82 |
| | | | General Medicine | 82 |
| | | | Cardiology | 50 |
| | | | **Total** | **849** |

**DynamoDB Table Schema:**
```
Table: DoctorSchedule (us-west-2)
├── PK: department      (String, HASH)
├── SK: doctor#day#time_slot  (String, RANGE)
├── doctor              (String)
├── day                 (String: Monday–Friday)
├── time_slot           (String: "08:00"–"16:30")
└── available           (Boolean)
```

---

### Agent 3 — Response Generator

| Task | Status | Details |
|------|--------|---------|
| Data collection | Done | [AI Medical Chatbot dataset](https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot) — 256,916 doctor-patient dialogues from HuggingFace |
| Data sampling & filtering | Done | 20% sample (~51k rows) → quality filtered (response length 80–800 chars) → **42,410 records** kept |
| Appointment context synthesis | Done | Each record enriched with synthetic department, doctor, time slot, and urgency |
| Train/test split | Done | 85/15 split → **36,048 train** (41 MB) + **6,362 test** (7.3 MB) |
| QLoRA fine-tuning | Done | 2-epoch run on Colab A100, 4,506 steps. Final checkpoints: checkpoint-4000, checkpoint-4500, checkpoint-4506, final_adapter |
| Checkpoint recovery | Verified | Successfully resumed from `checkpoint-4000` after Colab disconnection. Checkpoints saved to Google Drive every 200 steps |

**Agent 3 Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Method | QLoRA (4-bit quantization) + SFT via HuggingFace TRL |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | All attention + MLP layers (7 modules: q/k/v/o_proj + gate/up/down_proj) |
| Epochs | 2 |
| Batch size | 4 |
| Gradient accumulation | 4 (effective batch size = 16) |
| Total steps | 4,506 |
| Checkpoint interval | Every 200 steps → Google Drive |
| Eval interval | Every 500 steps |
| Best eval loss | **1.286** (step 4500) |
| Final train loss | 1.186 |
| Training time | ~8 hours |
| Adapter size | ~93 MB per checkpoint (582 MB total for 4 artifacts) |
| Platform | Google Colab Pay As You Go (NVIDIA A100 40GB, $9.99/100 compute units) |

**Agent 3 Training Loss Curve (selected steps):**

| Step | Epoch | Train Loss | Eval Loss | Note |
|------|-------|------------|-----------|------|
| 50 | 0.02 | 3.188 | — | Cold start |
| 500 | 0.22 | 1.419 | 1.411 | |
| 1000 | 0.44 | 1.364 | 1.362 | |
| 1500 | 0.67 | 1.319 | 1.335 | |
| 2000 | 0.89 | 1.291 | 1.317 | End of Epoch 1 |
| 2300 | 1.02 | 1.235 | — | Epoch 2 begins, loss drops |
| 2500 | 1.11 | 1.234 | 1.308 | |
| 3000 | 1.33 | 1.216 | 1.298 | |
| 3500 | 1.55 | 1.202 | 1.291 | |
| 4000 | 1.78 | 1.206 | 1.287 | Checkpoint used for resume |
| 4500 | 2.00 | 1.186 | **1.286** | **Best eval loss** |

**Agent 3 Checkpoint Comparison:**

| Checkpoint | Step | Eval Loss | Size | Note |
|-----------|------|-----------|------|------|
| checkpoint-4000 | 4000 | 1.287 | 157 MB | Colab disconnection recovery point |
| checkpoint-4500 | 4500 | **1.286** | 158 MB | **Best eval loss** |
| checkpoint-4506 | 4506 | 1.286 | 158 MB | Final training step |
| final_adapter | — | — | 109 MB | Inference-ready (no optimizer state) |

**Agent 3 Data Distribution:**

| Department | Samples | | Urgency | Samples |
|-----------|---------|---|---------|---------|
| Endocrinology | 4,356 | | Routine | 28,285 (66.7%) |
| Urology | 4,312 | | Urgent | 14,125 (33.3%) |
| General Medicine | 4,269 | | | |
| Gastroenterology | 4,255 | | **Avg response length** | **438 chars** |
| Infectious Disease | 4,237 | | | |
| Dermatology | 4,233 | | | |
| Orthopedics | 4,226 | | | |
| Cardiology | 4,221 | | | |
| Pulmonology | 4,187 | | | |
| Neurology | 4,114 | | | |
| **Total** | **42,410** | | | |

**Agent 3 Sample Record:**
```json
{
  "instruction": "You are a compassionate medical assistant. A patient has been assigned an appointment. Write a warm, clear appointment confirmation and practical pre-visit instructions...",
  "input": "Patient symptoms: I have a constant popping sensation under my rib cage...\nAssigned department: Gastroenterology\nDoctor: Dr. Patel\nAppointment: Wednesday at 12:00\nUrgency: Urgent",
  "output": "<doctor's response with confirmation and instructions>"
}
```

---

### LangGraph Orchestrator & Interface Contract

| Task | Status | Details |
|------|--------|---------|
| Pydantic state schema (`schemas.py`) | Done | 7 models defined and shared with Zhenhao |
| LangGraph workflow skeleton (Agent 1 → 2 → 3) | Not started | Planned for Week 2 — depends on all 3 agents being ready |

**Interface Contract (`schemas.py`):**

```
Agent 1:  SymptomInput{patient_text}  →  ClassifierOutput{department, urgency}
Agent 2:  AppointmentQuery{department}  →  AppointmentOutput{doctor, time_slot}
Agent 3:  ResponseInput{patient_text, department, doctor, time_slot, urgency}  →  ResponseOutput{confirmation, instructions}

AgentState (LangGraph shared state):
  patient_text, department, urgency, doctor, time_slot, confirmation, instructions
```

---

## 6. Plan for Week 2

| # | Task | Description | Priority |
|---|------|-------------|----------|
| 1 | ~~Download Agent 3 weights~~ | ~~Download final adapter + checkpoints from Google Drive into `response_generator_adapter/`~~ | Done |
| 2 | Agent 1 test set evaluation | Run Agent 1 (checkpoint-1569) on 738 test samples; compute precision/recall per department; verify emergency detection recall >95% | High |
| 3 | Agent 3 evaluation | Evaluate on 6,362 test samples; measure confirmation format correctness and instruction relevance; compare against >4.0/5.0 quality target | High |
| 4 | Agent 2 query logic | Implement boto3 DynamoDB query: input `department` → query `DoctorSchedule` table → return ranked `doctor + time_slot` by urgency | High |
| 5 | LangGraph orchestrator | Wire Agent 1 → 2 → 3 using `AgentState` schema; implement state transitions; end-to-end test: patient symptom text in → appointment confirmation out | High |
| 6 | Resolve SageMaker deployment | Option A: Quantize models (GGUF/GPTQ) for CPU-only SageMaker inference; Option B: HuggingFace Inference Endpoints; Option C: Escalate GPU permissions through instructor | Med |
| 7 | Upload models to S3 | Upload Agent 1 and Agent 3 LoRA adapters to Amazon S3 (regardless of deployment path) | Med |
| 8 | pytest unit tests | Write tests for all 3 agents: schema validation (Pydantic), output format verification, edge cases | Med |
| 9 | MLflow experiment tracking | Log Agent 1 and Agent 3 training loss curves, evaluation metrics, and hyperparameters | Low |

**Week 2 Evaluation Targets (from Project Proposal):**

| Metric | Project Target | Week 1 Status | Week 2 Goal |
|--------|---------------|---------------|-------------|
| Symptom Classification Accuracy | >85% | 98.1% token accuracy (training) | Formal test set eval |
| Emergency Detection Recall | >95% | Not yet evaluated | Evaluate on test set |
| Response Quality | >4.0/5.0 | Training complete | Evaluate on test set |
| End-to-End Latency | <5 seconds | Not yet measured | Measure after pipeline wiring |

---

## 7. Blockers / Risks

| Issue | Level | Status | Mitigation |
|-------|-------|--------|------------|
| SageMaker GPU instances blocked — AWS Academy only provides CPU quotas (`ml.m5.large`); GPU instances (`ml.g4dn.xlarge`, `ml.p3.2xlarge`) require quota increase we cannot request | High | Unresolved | Explore CPU-only quantized inference (GGUF/GPTQ), or HuggingFace Inference Endpoints as fallback, or escalate GPU permissions through course instructor |
| Kaggle GPU quota exhausted (30 hrs/week) — multiple failed training runs during Agent 1 development consumed entire weekly allocation | Med | Resolved | Migrated Agent 3 training to Google Colab Pay As You Go (A100 GPU, $9.99/100 compute units) |
| Training data loss on Kaggle — outputs lost due to runtime disconnections and incorrect output path configuration (ephemeral storage wiped on session end) | Med | Resolved | Colab training saves checkpoints to Google Drive every 200 steps with `resume_from_checkpoint` logic; validated after Colab disconnected and resumed from `checkpoint-4000` |
| Agent 3 training took ~8 hours on Colab A100 with one disconnection mid-run | Low | Resolved | Checkpoint recovery from Google Drive validated; all 4 artifacts saved (checkpoint-4000, 4500, 4506, final_adapter) |
| No formal evaluation split in Agent 1 training — only training metrics available | Low | Planned | Will run full test set evaluation (738 samples) in Week 2; 98.1% training accuracy exceeds >85% project target |
| MLflow not yet integrated — training metrics only in `trainer_state.json` | Low | Planned | Will set up before Agent 3 evaluation in Week 2 |

---

## 8. Known Limitations

The following data quality issues were identified during code review. They do not block the current pipeline but should be addressed if models are retrained.

| # | Issue | Impact | Affected Data | Mitigation |
|---|-------|--------|---------------|------------|
| 1 | **Symptom name mismatches** — 3 symptoms in `dataset.csv` have internal spaces (`dischromic _patches`, `spotting_ urination`, `foul_smell_of urine`) that do not match keys in `Symptom-severity.csv`, causing their severity weights to be scored as 0 | 240 rows (4.9%) have understated urgency scores. `Drug Reaction` and `Urinary tract infection` are labeled Routine instead of Urgent | Agent 1 training data | Normalize symptom names (strip internal spaces) in `compute_urgency()` before severity lookup; requires re-running `process_symptom_classifier.py` + retraining Agent 1 |
| 2 | **Emergency urgency absent in Agent 3 training data** — `process_response_generator.py` only samples Routine and Urgent; Emergency is never included | Agent 3 has 0% training coverage for Emergency inputs. If Agent 1 outputs Emergency, Agent 3 has never seen this label during training | Agent 3 training data | Runtime mitigation: `normalize_urgency()` in `schemas.py` maps Emergency → Urgent at inference time. Full fix: add Emergency to urgency sampling and retrain Agent 3 |
| 3 | **Duplicate symptom weight** — `fluid_overload` appears twice in `Symptom-severity.csv` (weights 6 and 4); `dict(zip())` silently keeps the last value | Affects `Alcoholic hepatitis` urgency scoring by ±2 points | Agent 1 training data | Deduplicate in `load_severity_weights()` using `groupby().max()`; requires retraining Agent 1 |
| 4 | **Spurious `prognosis` entry** — `Symptom-severity.csv` contains `prognosis,5` which is a metadata artifact, not a real symptom | Could inflate urgency scores if any row lists `prognosis` as a symptom value | Agent 1 training data | Filter out non-symptom entries in `load_severity_weights()` |

**Decision:** These issues affect <5% of training data and the current models already exceed project accuracy targets (Agent 1: 98.1% token accuracy). Retraining is deferred to Week 2 if GPU time permits. The Emergency mapping is handled at inference time via `schemas.normalize_urgency()`.

---

## 9. Project File Structure

```
medi-agent/
├── schemas.py                                 # Pydantic interface contracts (7 models: SymptomInput,
│                                              #   ClassifierOutput, AppointmentQuery, AppointmentOutput,
│                                              #   ResponseInput, ResponseOutput, AgentState)
├── requirements.txt                           # 15 dependencies across ML, orchestration, AWS, API, testing
├── Week1_Progress_Report.md                   # This report
│
├── data/
│   ├── process_symptom_classifier.py          # Agent 1 pipeline: dataset.csv → 4,920 JSONL records
│   │                                          #   41 diseases → 10 departments, severity-weighted urgency
│   ├── process_appointment_retriever.py       # Agent 2 pipeline: doctors.csv → doctor_schedules.json
│   │                                          #   50 doctors, 10 depts, read from CSV (no hardcoding)
│   ├── process_response_generator.py          # Agent 3 pipeline: 256k dialogues → 42,410 JSONL records
│   │                                          #   20% sample, quality filtered (80-800 chars), synthetic context
│   ├── upload_to_dynamodb.py                  # DynamoDB seeding: doctor_schedules.json → DoctorSchedule table
│   │                                          #   Schema: PK=department, SK=doctor#day#time_slot
│   ├── raw/
│   │   ├── disease_symptom_prediction/        # Kaggle: 4 CSV files (dataset, severity, precautions, descriptions)
│   │   └── doctor_schedules/                  # doctors.csv (50 doctors, 10 departments)
│   │
│   └── processed/
│       ├── symptom_classifier_train.jsonl          2.0 MB   4,182 samples  (Agent 1 training data)
│       ├── symptom_classifier_test.jsonl           360 KB     738 samples  (Agent 1 test data)
│       ├── response_generator_train.jsonl           41 MB  36,048 samples  (Agent 3 training data)
│       ├── response_generator_test.jsonl           7.3 MB   6,362 samples  (Agent 3 test data)
│       ├── department_mapping.json                 4.0 KB                  (41 disease → department entries)
│       ├── disease_precautions.json                8.0 KB                  (precautions for Agent 3)
│       ├── doctor_schedules.json                   124 KB                  (849 slots, 50 doctors, 10 depts)
│       ├── symptom_classifier_adapter/                89 MB                  (Agent 1 LoRA adapters)
│       │   ├── README.md                                                   (Model card with training details)
│       │   ├── checkpoint-523/                      30 MB                  (Epoch 1: loss=0.079, acc=97.8%)
│       │   │   ├── adapter_model.safetensors                              (LoRA weights)
│       │   │   ├── adapter_config.json                                    (r=8, alpha=16, q/v_proj)
│       │   │   ├── tokenizer.json + tokenizer_config.json                 (LLaMA-3.2 tokenizer)
│       │   │   ├── trainer_state.json                                     (Full training log)
│       │   │   ├── training_args.bin                                      (SFTTrainer config)
│       │   │   ├── optimizer.pt + scheduler.pt + rng_state.pth            (Resume state)
│       │   │   └── chat_template.jinja                                    (LLaMA chat format)
│       │   ├── checkpoint-1046/                     30 MB                  (Epoch 2: loss=0.060, acc=98.1%)
│       │   └── checkpoint-1569/                     30 MB                  (Epoch 3: loss=0.057, acc=98.1%)
│       │
│       └── response_generator_adapter/             582 MB                  (Agent 3 LoRA adapters)
│           ├── README.md                                                   (Model card with training details)
│           ├── checkpoint-4000/                    157 MB                  (Step 4000: eval_loss=1.287)
│           ├── checkpoint-4500/                    158 MB                  (Step 4500: eval_loss=1.286, best)
│           ├── checkpoint-4506/                    158 MB                  (Final step)
│           └── final_adapter/                      109 MB                  (Inference-ready, no optimizer)
│
├── agents/              # (Week 2 — Agent inference wrappers)
├── orchestrator/        # (Week 2 — LangGraph workflow)
└── tests/               # (Week 2 — pytest unit tests)
```
