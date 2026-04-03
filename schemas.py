from pydantic import BaseModel
from typing import Literal, TypedDict

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


class SymptomInput(BaseModel):
    patient_text: str


class ClassifierOutput(BaseModel):
    department: str
    urgency: Literal["Routine", "Urgent", "Emergency"]


class AppointmentQuery(BaseModel):
    department: str


class AppointmentOutput(BaseModel):
    doctor: str
    time_slot: str


class ResponseInput(BaseModel):
    patient_text: str
    department: str
    doctor: str
    time_slot: str
    urgency: Literal["Routine", "Urgent", "Emergency"]


class ResponseOutput(BaseModel):
    confirmation: str
    instructions: str


class AgentState(TypedDict, total=False):
    """LangGraph state passed between all three agents."""
    patient_text: str
    age: Optional[int]
    gender: Optional[str]
    department: str
    urgency: Literal["Routine", "Urgent", "Emergency"]
    doctor: str
    time_slot: str
    confirmation: str
    instructions: str


def normalize_urgency(urgency: str) -> str:
    """Map Emergency → Urgent. Agent 3 only trained on Routine/Urgent."""
    if urgency == "Emergency":
        return "Urgent"
    return urgency


# ---------------------------------------------------------------------------
# HTTP API schemas (used by api/v1/router.py)
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    symptom: str
    age: Optional[int] = None
    gender: Optional[str] = None


class QueryResponse(BaseModel):
    agent1: ClassifierOutput
    agent2: AppointmentOutput
    agent3: ResponseOutput
