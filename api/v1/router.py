import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import boto3
from boto3.dynamodb.conditions import Key as DDBKey
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from schemas import (
    ClassifierOutput,
    AppointmentOutput,
    ResponseOutput,
    QueryRequest,
    QueryResponse,
    PatientProfile,
)
from orchestrator.graph import run_pipeline, stream_pipeline
from auth.dependencies import get_current_user
from agents.rag.qa import answer_question, lookup_drug

api_router = APIRouter()

APPOINTMENTS_TABLE = os.getenv("APPOINTMENTS_TABLE", "medi-agent-appointments")
PROFILES_TABLE     = os.getenv("PROFILES_TABLE", "medi-agent-patient-profiles")
AWS_REGION         = os.getenv("AWS_REGION", "us-west-2")

_ddb      = boto3.resource("dynamodb", region_name=AWS_REGION)
_appts    = _ddb.Table(APPOINTMENTS_TABLE)
_profiles = _ddb.Table(PROFILES_TABLE)

_bearer = HTTPBearer(auto_error=False)

# In-memory conversation history: {conversation_id: [{symptom, department, urgency}]}
_conv_store: Dict[str, List[dict]] = {}


def _optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[dict]:
    if not credentials:
        return None
    try:
        return get_current_user(credentials)
    except Exception:
        return None


def _get_profile(user_id: str) -> dict:
    try:
        resp = _profiles.get_item(Key={"user_id": user_id})
        return resp.get("Item", {})
    except Exception:
        return {}


def _save_appointment(user: Optional[dict], symptom: str, state: dict):
    if not user:
        return
    item = {
        "user_id":      user["sub"],
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "appt_id":      str(uuid.uuid4()),
        "symptom":      symptom,
        "department":   state.get("department", ""),
        "urgency":      state.get("urgency", ""),
        "doctor":       state.get("doctor", ""),
        "time_slot":    state.get("time_slot", ""),
        "confirmation": state.get("confirmation", ""),
        "instructions": state.get("instructions", ""),
    }
    if state.get("first_aid"):
        item["first_aid"] = state["first_aid"]
    try:
        _appts.put_item(Item=item)
    except Exception as e:
        print(f"[router] DynamoDB save failed: {e}")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@api_router.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Query (synchronous)
# ---------------------------------------------------------------------------

@api_router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, user: Optional[dict] = Depends(_optional_user)):
    age    = user.get("age")    if user else req.user_age
    gender = user.get("gender") if user else req.user_gender
    profile = _get_profile(user["sub"]) if user else {}

    conv_id = req.conversation_id or str(uuid.uuid4())
    history = _conv_store.get(conv_id, [])

    if user and not history:
        try:
            resp = _appts.query(
                KeyConditionExpression=DDBKey("user_id").eq(user["sub"]),
                ScanIndexForward=False,
                Limit=5,
            )
            db_items = list(reversed(resp.get("Items", [])))
            history = [
                {
                    "symptom":    item.get("symptom", ""),
                    "department": item.get("department", ""),
                    "urgency":    item.get("urgency", ""),
                }
                for item in db_items
            ]
            if history:
                _conv_store[conv_id] = history
        except Exception:
            pass

    state = run_pipeline(
        req.symptom,
        user_age=age,
        user_gender=gender,
        blood_type=profile.get("blood_type") or req.blood_type,
        allergies=profile.get("allergies")   or req.allergies,
        height_cm=profile.get("height_cm")   or req.height_cm,
        weight_kg=profile.get("weight_kg")   or req.weight_kg,
        thread_id=f"{conv_id}_{len(history) + 1}",
        conversation_history=history,
    )

    _save_appointment(user, req.symptom, state)

    _conv_store[conv_id] = history + [{
        "symptom":    req.symptom,
        "department": state.get("department", ""),
        "urgency":    state.get("urgency", ""),
    }]

    return QueryResponse(
        agent1=ClassifierOutput(
            department=state.get("department", "General Medicine"),
            urgency=state.get("urgency", "Routine"),
        ),
        agent2=AppointmentOutput(
            doctor=state.get("doctor", ""),
            time_slot=state.get("time_slot", ""),
        ),
        agent3=ResponseOutput(
            confirmation=state.get("confirmation", ""),
            instructions=state.get("instructions", ""),
            first_aid=state.get("first_aid"),
            possible_causes=state.get("possible_causes", []),
        ),
        conversation_id=conv_id,
    )


# ---------------------------------------------------------------------------
# Query (streaming SSE)
# ---------------------------------------------------------------------------

@api_router.post("/query/stream")
def query_stream(req: QueryRequest, user: Optional[dict] = Depends(_optional_user)):
    """Stream pipeline results as Server-Sent Events, one event per agent node."""
    age    = user.get("age")    if user else req.user_age
    gender = user.get("gender") if user else req.user_gender
    profile = _get_profile(user["sub"]) if user else {}

    conv_id = req.conversation_id or str(uuid.uuid4())
    history = _conv_store.get(conv_id, [])

    # For logged-in users: if no in-memory history yet, seed from DynamoDB appointment history
    if user and not history:
        try:
            resp = _appts.query(
                KeyConditionExpression=DDBKey("user_id").eq(user["sub"]),
                ScanIndexForward=False,
                Limit=5,
            )
            db_items = list(reversed(resp.get("Items", [])))
            history = [
                {
                    "symptom":    item.get("symptom", ""),
                    "department": item.get("department", ""),
                    "urgency":    item.get("urgency", ""),
                }
                for item in db_items
            ]
            if history:
                _conv_store[conv_id] = history
        except Exception:
            pass

    _node_to_event = {
        "classify": "classifier",
        "retrieve": "retriever",
        "generate": "generator",
        "causes":   "causes",
    }

    def event_generator():
        final_state: dict = {}
        try:
            for update in stream_pipeline(
                req.symptom,
                user_age=age,
                user_gender=gender,
                blood_type=profile.get("blood_type") or req.blood_type,
                allergies=profile.get("allergies")   or req.allergies,
                height_cm=profile.get("height_cm")   or req.height_cm,
                weight_kg=profile.get("weight_kg")   or req.weight_kg,
                thread_id=f"{conv_id}_{len(history) + 1}",
                conversation_history=history,
            ):
                for node_name, node_updates in update.items():
                    event_name = _node_to_event.get(node_name, node_name)
                    payload = {"event": event_name, **node_updates}
                    yield f"data: {json.dumps(payload)}\n\n"
                    final_state.update(node_updates)
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'detail': str(e)})}\n\n"
            return

        _save_appointment(user, req.symptom, final_state)
        _conv_store[conv_id] = history + [{
            "symptom":    req.symptom,
            "department": final_state.get("department", ""),
            "urgency":    final_state.get("urgency", ""),
        }]

        yield f"data: {json.dumps({'event': 'done', 'conversation_id': conv_id})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Appointments
# ---------------------------------------------------------------------------

@api_router.get("/appointments")
def get_appointments(user: dict = Depends(get_current_user)):
    resp = _appts.query(
        KeyConditionExpression=DDBKey("user_id").eq(user["sub"]),
        ScanIndexForward=False,
        Limit=20,
    )
    return {"appointments": resp.get("Items", [])}


@api_router.delete("/appointments/{timestamp}")
def cancel_appointment(timestamp: str, user: dict = Depends(get_current_user)):
    resp = _appts.get_item(Key={"user_id": user["sub"], "timestamp": timestamp})
    if "Item" not in resp:
        raise HTTPException(status_code=404, detail="Appointment not found")
    _appts.delete_item(Key={"user_id": user["sub"], "timestamp": timestamp})
    return {"deleted": True}


# ---------------------------------------------------------------------------
# Patient Profile
# ---------------------------------------------------------------------------

@api_router.get("/profile")
def get_profile(user: dict = Depends(get_current_user)):
    item = _get_profile(user["sub"])
    return {
        "blood_type": item.get("blood_type"),
        "allergies":  item.get("allergies", []),
        "height_cm":  item.get("height_cm"),
        "weight_kg":  item.get("weight_kg"),
    }


@api_router.put("/profile")
def update_profile(body: PatientProfile, user: dict = Depends(get_current_user)):
    from decimal import Decimal
    item = {"user_id": user["sub"]}
    if body.blood_type is not None:
        item["blood_type"] = body.blood_type
    if body.allergies is not None:
        item["allergies"] = body.allergies
    if body.height_cm is not None:
        item["height_cm"] = body.height_cm
    if body.weight_kg is not None:
        item["weight_kg"] = Decimal(str(body.weight_kg))
    _profiles.put_item(Item=item)
    return {"saved": True}


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")

@api_router.delete("/admin/appointments")
def clear_all_appointments(secret: str):
    if not ADMIN_SECRET or secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    deleted = 0
    while True:
        resp = _appts.scan(
            ProjectionExpression="user_id, #ts",
            ExpressionAttributeNames={"#ts": "timestamp"},
        )
        items = resp.get("Items", [])
        if not items:
            break
        with _appts.batch_writer() as batch:
            for item in items:
                batch.delete_item(Key={"user_id": item["user_id"], "timestamp": item["timestamp"]})
        deleted += len(items)
        if "LastEvaluatedKey" not in resp:
            break
    return {"deleted": deleted}


# ---------------------------------------------------------------------------
# Medical Q&A (RAG + user docs)
# ---------------------------------------------------------------------------

class QARequest(BaseModel):
    question: str

@api_router.post("/qa")
def medical_qa(req: QARequest):
    try:
        return answer_question(req.question)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RAG service unavailable: {e}")


# ---------------------------------------------------------------------------
# Drug Lookup (FDA → RAG fallback)
# ---------------------------------------------------------------------------

class DrugRequest(BaseModel):
    drug_name: str

@api_router.post("/drug")
def drug_query(req: DrugRequest):
    try:
        return lookup_drug(req.drug_name)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Drug lookup failed: {e}")


# ---------------------------------------------------------------------------
# Chat Guide — generate follow-up questions and summary (one API call each)
# ---------------------------------------------------------------------------

class ChatQuestionsRequest(BaseModel):
    symptom: str

class ChatSummaryRequest(BaseModel):
    symptom: str
    qa_pairs: list

@api_router.post("/chat/questions")
def chat_questions(req: ChatQuestionsRequest):
    try:
        from agents.chat_guide.agent import generate_questions
        return {"questions": generate_questions(req.symptom)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@api_router.post("/chat/summary")
def chat_summary(req: ChatSummaryRequest):
    try:
        from agents.chat_guide.agent import generate_summary
        return {"summary": generate_summary(req.symptom, req.qa_pairs)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

