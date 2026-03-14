from fastapi import APIRouter

from schemas import (
    ClassifierOutput,
    AppointmentOutput,
    ResponseOutput,
    QueryRequest,
    QueryResponse,
)
from orchestrator.graph import run_pipeline

api_router = APIRouter()


@api_router.get("/health")
def health():
    return {"status": "ok"}


@api_router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    state = run_pipeline(req.symptom)

    return QueryResponse(
        agent1=ClassifierOutput(
            department=state.get("department", "General Practice"),
            urgency=state.get("urgency", "Routine"),
        ),
        agent2=AppointmentOutput(
            doctor=state.get("doctor", ""),
            time_slot=state.get("time_slot", ""),
        ),
        agent3=ResponseOutput(
            confirmation=state.get("confirmation", ""),
            instructions=state.get("instructions", ""),
        ),
    )
