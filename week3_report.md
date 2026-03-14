# Week 3 Progress Report — Medi-Agent

**Project**: Medi-Agent: Multi-Agent Medical Appointment Triage & Scheduling System
**Date**: March 14, 2026
**Author**: Chen Qi

---

## 1. Project Overview

Medi-Agent is an intelligent medical appointment triage and scheduling system powered by three specialized LLM agents orchestrated via LangGraph. The system takes patient symptom descriptions as input and routes them through a pipeline that (1) classifies symptoms to a department and urgency level, (2) retrieves available doctor appointments from DynamoDB, and (3) generates a personalized appointment confirmation with pre-visit instructions.

---

## 2. Work Completed This Week

### 2.1 Three-Agent Pipeline Implementation

Implemented the full three-agent pipeline using LangGraph's StateGraph:

- **Agent 1 — Symptom Classifier**: Fine-tuned LLaMA-3.2-3B-Instruct with QLoRA (4-bit quantization, LoRA rank=16, alpha=32) to classify patient symptoms into one of 10 medical departments (Cardiology, Neurology, Dermatology, Gastroenterology, Endocrinology, Pulmonology, Infectious Disease, Orthopedics, Urology, General Medicine) and assign a 3-level urgency rating (Routine / Urgent / Emergency).

- **Agent 2 — Appointment Retriever**: Built a DynamoDB-backed appointment retrieval agent that queries the `DoctorSchedule` table by department, filters for available slots, sorts by day and time, and returns the earliest available appointment.

- **Agent 3 — Response Generator**: Fine-tuned a second LLaMA-3.2-3B-Instruct model with QLoRA to generate patient-friendly appointment confirmations with practical pre-visit instructions, conditioned on symptom context, department, doctor, time, and urgency.

### 2.2 Data Processing & Training

- Processed the Disease Symptom Prediction dataset into 31,000 training and 1,500 test samples (JSONL format) for the symptom classifier, with severity-score-based urgency thresholds (Emergency >= 45, Urgent >= 20).
- Processed the AI Medical Chatbot dataset (~51k dialogues, 20% sample) for the response generator, pairing real doctor-patient dialogue with synthetic appointment context from generated doctor schedules.
- Generated 8,000+ appointment slots from raw doctor schedule data (doctors.csv) for the retrieval agent.
- Discovered and fixed a synonym mapping bug in the urgency classification logic: the "Emergency" level was incorrectly mapped due to a typo/synonym mismatch, which caused incorrect urgency assignments during cloud-based testing. Corrected the mapping, re-processed the training data, and re-trained the symptom classifier adapter.
- Trained both QLoRA adapters using SFTTrainer (3 epochs, batch size 4, gradient accumulation 4, learning rate 2e-4, paged AdamW 8-bit optimizer).

### 2.3 Orchestration Layer & Emergency Mapping Bug Fix

- Built the LangGraph orchestrator (`orchestrator/graph.py`) with a sequential StateGraph: `START -> classify_symptoms -> retrieve_appointment -> generate_response -> END`.
- Implemented lazy-loaded singleton agents for efficient resource usage.
- Defined Pydantic schemas (`schemas.py`) for type-safe data flow across all pipeline stages.

**Discovery and fix of the Emergency synonym mapping bug:**

During cloud deployment testing, end-to-end outputs for high-risk symptoms (e.g., Heart attack, Paralysis) were found to be incorrect — Agent 1 correctly classified them as `Emergency`, but Agent 3's generated responses did not match the urgency level, producing overly calm language and missing urgent care instructions.

**Root cause:** An inconsistency existed between Agent 1's output labels and Agent 3's training data:

1. Agent 1's classifier was originally trained on 2 urgency levels (Routine / Urgent), then expanded to 3 levels (adding Emergency). The `process_symptom_classifier.py` script correctly generated Emergency labels via the `ALWAYS_EMERGENCY` set and a severity threshold (`>= 45`).
2. However, Agent 3's early training data only contained Routine and Urgent — the model had never seen the `Emergency` label. When Agent 1 passed Emergency to Agent 3, the model could not interpret it correctly, producing confirmations and instructions that did not reflect the true urgency.
3. A `normalize_urgency()` function was added in `schemas.py` as a temporary compatibility shim, mapping Emergency to Urgent so Agent 3 could still produce reasonable outputs.

**Fixes applied:**

- Updated the training data generation logic in `process_response_generator.py` to include the Emergency level in synthetic contexts (distribution adjusted to ~40% Routine / ~40% Urgent / ~20% Emergency), enabling Agent 3 to learn urgency-appropriate response patterns.
- Re-generated all Response Generator training data and re-trained Agent 3's QLoRA adapter on the cloud.
- Confirmed that `VALID_URGENCIES` in `parsing.py` includes all three levels (`{"Routine", "Urgent", "Emergency"}`), ensuring the parsing layer does not drop valid Emergency outputs.
- Retained `normalize_urgency()` as a backward-compatible fallback for potential future downgrade scenarios.

### 2.4 Testing & Evaluation

- Wrote unit tests for all components:
  - `test_schemas.py`: Pydantic model validation, urgency normalization, parametrized tests.
  - `test_symptom_classifier.py`: Regex parsing, malformed input handling, fallback defaults for all 10 departments.
  - `test_appointment_retriever.py`: Mocked DynamoDB queries, slot sorting, edge cases.
  - `test_response_generator.py`: Multiline parsing, unstructured format handling, empty input edge cases.

- Built evaluation scripts:
  - Symptom classifier evaluation on 738 test samples — measures overall accuracy, per-department precision/recall/F1, and emergency detection recall (target > 95%).
  - Response generator evaluation on 6,362 test samples — measures format compliance, confirmation/instructions presence rates, average response length, per-department and per-urgency breakdown.

### 2.5 Output Parsing Module

- Implemented a lightweight parsing module (`agents/parsing.py`) using regex to extract structured outputs from model responses without requiring torch/transformers, enabling fast unit testing.

---

## 3. Technical Highlights

| Aspect | Detail |
|---|---|
| Base Model | LLaMA-3.2-3B-Instruct |
| Fine-Tuning | QLoRA (4-bit NF4, LoRA r=16, alpha=32, dropout=0.05) |
| Orchestration | LangGraph StateGraph |
| Database | AWS DynamoDB |
| API Framework | FastAPI + Streamlit |
| Monitoring | MLflow |
| Test Framework | Pytest |

---

## 4. Challenges & Solutions

| Challenge | Solution |
|---|---|
| Urgency level expansion (2 -> 3 levels) | Added `normalize_urgency()` schema helper for backward compatibility |
| Large model inference cost | Applied 4-bit quantization with QLoRA adapters (~3% trainable params) |
| Testing agents without GPU | Separated parsing logic into a torch-free module for lightweight unit tests |
| Unbalanced urgency distribution | Applied intentional distribution skew in training data (e.g., 30/40/30 for emergency-prone departments) |
| Emergency synonym mapping error | Agent 3 was initially trained on only Routine/Urgent, so it could not handle the Emergency label from Agent 1, causing incorrect response tone for critical cases during cloud testing. Fix: (1) added Emergency (20%) to Response Generator training data, (2) re-trained Agent 3's QLoRA adapter, (3) retained `normalize_urgency()` as a fallback |

---

## 5. Next Steps

- Deploy the full pipeline via FastAPI and build a Streamlit frontend for user interaction.
- Integrate MLflow for experiment tracking and model versioning.
- Run end-to-end evaluation on the complete pipeline (classify -> retrieve -> respond).
- Optimize inference latency and explore model caching strategies.
- Load doctor schedules into DynamoDB and test with live AWS infrastructure.

---

## 6. Repository Structure

```
medi-agent/
├── agents/
│   ├── symptom_classifier/agent.py
│   ├── appointment_retriever/agent.py
│   ├── response_generator/agent.py
│   └── parsing.py
├── orchestrator/graph.py
├── data/
│   ├── raw/                          # Source datasets
│   └── processed/                    # Training data & adapters
├── notebooks/
│   └── train_response_generator.ipynb
├── tests/                            # Unit tests & evaluations
├── schemas.py
└── requirements.txt
```
