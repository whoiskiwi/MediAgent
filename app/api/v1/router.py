from fastapi import APIRouter
from app.schema import QueryRequest, QueryResponse
from app.services.agent_service import run_pipeline

api_router = APIRouter()


@api_router.get("/health")
def health():
    return {"status": "ok"}


@api_router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):

    agent1, agent2, agent3 = run_pipeline(req.symptom)

    return QueryResponse(
        agent1=agent1,
        agent2=agent2,
        agent3=agent3
    )