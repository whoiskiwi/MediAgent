from pydantic import BaseModel


class QueryRequest(BaseModel):
    symptom: str


class Agent1Output(BaseModel):
    department: str
    urgency: str


class Agent2Output(BaseModel):
    doctor: str
    time_slot: str


class Agent3Output(BaseModel):
    confirmation: str
    instructions: str


class QueryResponse(BaseModel):
    agent1: Agent1Output
    agent2: Agent2Output
    agent3: Agent3Output