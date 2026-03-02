# MediAgent

A multi-agent LLM system for hospital appointment scheduling and symptom triage, built with fine-tuned LLaMA-3.2-3B models orchestrated via LangGraph.

## Architecture

MediAgent uses three specialized agents working in sequence:

```
Patient Input → [Agent 1: Symptom Classifier] → [Agent 2: Appointment Retriever] → [Agent 3: Response Generator] → Confirmation
```

| Agent | Model | Function |
|-------|-------|----------|
| **Agent 1** — Symptom Classifier | LLaMA-3.2-3B + QLoRA (r=8) | Classifies symptoms → department + urgency |
| **Agent 2** — Appointment Retriever | DynamoDB + boto3 | Queries available doctor time slots by department |
| **Agent 3** — Response Generator | LLaMA-3.2-3B + QLoRA (r=16) | Generates patient-friendly confirmation with pre-visit instructions |

## Project Structure

```
MediAgent/
├── schemas.py                          # Pydantic interface contracts (7 models)
├── requirements.txt                    # Python dependencies
├── data/
│   ├── process_symptom_classifier.py   # Agent 1 data pipeline
│   ├── process_appointment_retriever.py# Agent 2 data pipeline
│   ├── process_response_generator.py   # Agent 3 data pipeline
│   ├── upload_to_dynamodb.py           # DynamoDB seeding script
│   ├── raw/                            # Source datasets
│   │   ├── disease_symptom_prediction/ # Kaggle symptom-disease dataset
│   │   └── doctor_schedules/           # Doctor roster (50 doctors, 10 depts)
│   └── processed/                      # Generated training data & adapters
│       ├── symptom_classifier_*.jsonl  # Agent 1 train/test splits
│       ├── response_generator_*.jsonl  # Agent 3 train/test splits
│       ├── department_mapping.json     # 41 diseases → 10 departments
│       ├── doctor_schedules.json       # 849 appointment slots
│       ├── symptom_classifier_adapter/ # Agent 1 LoRA checkpoints
│       └── response_generator_adapter/ # Agent 3 LoRA checkpoints
```

## Tech Stack

- **ML**: PyTorch, HuggingFace Transformers, PEFT (LoRA), TRL, bitsandbytes (4-bit quantization)
- **Orchestration**: LangChain, LangGraph
- **Backend**: FastAPI, Pydantic
- **Frontend**: Streamlit
- **Cloud**: AWS DynamoDB, S3, SageMaker

## Setup

```bash
pip install -r requirements.txt
```

## Data Pipeline

Each agent has a dedicated data processing script under `data/`:

```bash
# Generate Agent 1 training data (symptom → department + urgency)
python data/process_symptom_classifier.py

# Generate Agent 2 appointment slots
python data/process_appointment_retriever.py

# Generate Agent 3 training data (dialogue → confirmation)
python data/process_response_generator.py

# Seed DynamoDB with doctor schedules
python data/upload_to_dynamodb.py
```

## Training Results

| Agent | Platform | Performance | Training |
|-------|----------|-------------|----------|
| Agent 1 | Kaggle P100 | 98.1% token accuracy, loss 0.057 | 3 epochs, 1,569 steps |
| Agent 3 | Colab A100 | Best eval loss 1.286 | 2 epochs, 4,506 steps |

## Datasets

- **Agent 1**: 4,920 disease-symptom pairs from [Kaggle Disease Symptom Prediction](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) → 4,182 train / 738 test
- **Agent 2**: 50 doctors × 10 departments → 849 time slots (stored in DynamoDB)
- **Agent 3**: 256K doctor-patient dialogues from [HuggingFace ruslanmv/ai-medical-chatbot](https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot) → 36,048 train / 6,362 test

## Team

| Name | ID | Role |
|------|----|------|
| Qi Chen | 002315412 | ML & Agent Core |
| Zhenhao Ma | 002309369 | Infrastructure & Deployment |
