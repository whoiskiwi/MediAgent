from pydantic import BaseModel
from typing import Literal, Optional, TypedDict

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
    first_aid: Optional[str] = None
    possible_causes: Optional[list] = None


class PatientProfile(BaseModel):
    blood_type:  Optional[str]  = None   # e.g. "A+", "O-"
    allergies:   Optional[list] = None   # e.g. ["Penicillin", "Peanuts"]
    height_cm:   Optional[int]  = None
    weight_kg:   Optional[float] = None


class AgentState(TypedDict, total=False):
    """LangGraph state passed between all three agents."""
    patient_text: str
    user_age: Optional[int]
    user_gender: Optional[str]
    # patient profile fields
    blood_type: Optional[str]
    allergies: Optional[list]
    height_cm: Optional[int]
    weight_kg: Optional[float]
    department: str
    urgency: Literal["Routine", "Urgent", "Emergency"]
    doctor: str
    time_slot: str
    confirmation: str
    instructions: str
    first_aid: Optional[str]
    possible_causes: Optional[list]


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
    user_age: Optional[int] = None
    user_gender: Optional[str] = None
    # passed through when not logged in; ignored server-side when JWT is present
    blood_type: Optional[str]   = None
    allergies:  Optional[list]  = None
    height_cm:  Optional[int]   = None
    weight_kg:  Optional[float] = None


class QueryResponse(BaseModel):
    agent1: ClassifierOutput
    agent2: AppointmentOutput
    agent3: ResponseOutput
