import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import boto3
from boto3.dynamodb.conditions import Key as DDBKey
from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from schemas import (
    ClassifierOutput,
    AppointmentOutput,
    ResponseOutput,
    QueryRequest,
    QueryResponse,
    PatientProfile,
)
from orchestrator.graph import run_pipeline
from auth.dependencies import get_current_user
from agents.rag.qa import answer_question

api_router = APIRouter()

APPOINTMENTS_TABLE = os.getenv("APPOINTMENTS_TABLE", "medi-agent-appointments")
PROFILES_TABLE     = os.getenv("PROFILES_TABLE", "medi-agent-patient-profiles")
AWS_REGION         = os.getenv("AWS_REGION", "us-west-2")

_ddb     = boto3.resource("dynamodb", region_name=AWS_REGION)
_appts   = _ddb.Table(APPOINTMENTS_TABLE)
_profiles = _ddb.Table(PROFILES_TABLE)

_bearer = HTTPBearer(auto_error=False)


def _optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[dict]:
    """Return JWT payload if a valid token is provided, else None."""
    if not credentials:
        return None
    try:
        return get_current_user(credentials)
    except Exception:
        return None


@api_router.get("/health")
def health():
    return {"status": "ok"}


def _get_profile(user_id: str) -> dict:
    """Fetch patient profile from DynamoDB; return empty dict if not found."""
    try:
        resp = _profiles.get_item(Key={"user_id": user_id})
        return resp.get("Item", {})
    except Exception:
        return {}


@api_router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, user: Optional[dict] = Depends(_optional_user)):
    age    = user.get("age")    if user else req.user_age
    gender = user.get("gender") if user else req.user_gender

    # Fetch patient profile when logged in
    profile: dict = {}
    if user:
        profile = _get_profile(user["sub"])

    state = run_pipeline(
        req.symptom,
        user_age=age,
        user_gender=gender,
        blood_type=profile.get("blood_type") or req.blood_type,
        allergies=profile.get("allergies")   or req.allergies,
        height_cm=profile.get("height_cm")   or req.height_cm,
        weight_kg=profile.get("weight_kg")   or req.weight_kg,
    )

    result = QueryResponse(
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
    )

    # Save to appointment history if user is logged in
    if user:
        item = {
            "user_id":      user["sub"],
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "appt_id":      str(uuid.uuid4()),
            "symptom":      req.symptom,
            "department":   result.agent1.department,
            "urgency":      result.agent1.urgency,
            "doctor":       result.agent2.doctor,
            "time_slot":    result.agent2.time_slot,
            "confirmation": result.agent3.confirmation,
            "instructions": result.agent3.instructions,
        }
        if result.agent3.first_aid:
            item["first_aid"] = result.agent3.first_aid
        _appts.put_item(Item=item)

    return result


@api_router.get("/appointments")
def get_appointments(user: dict = Depends(get_current_user)):
    """Return the current user's appointment history, newest first."""
    resp = _appts.query(
        KeyConditionExpression=DDBKey("user_id").eq(user["sub"]),
        ScanIndexForward=False,
        Limit=20,
    )
    return {"appointments": resp.get("Items", [])}


@api_router.delete("/appointments/{timestamp}")
def cancel_appointment(timestamp: str, user: dict = Depends(get_current_user)):
    """Cancel (delete) a specific appointment by timestamp."""
    from fastapi import HTTPException
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
    """Return the current user's patient profile."""
    item = _get_profile(user["sub"])
    return {
        "blood_type": item.get("blood_type"),
        "allergies":  item.get("allergies", []),
        "height_cm":  item.get("height_cm"),
        "weight_kg":  item.get("weight_kg"),
    }


@api_router.put("/profile")
def update_profile(body: PatientProfile, user: dict = Depends(get_current_user)):
    """Create or update the current user's patient profile."""
    item = {"user_id": user["sub"]}
    if body.blood_type is not None:
        item["blood_type"] = body.blood_type
    if body.allergies is not None:
        item["allergies"] = body.allergies
    if body.height_cm is not None:
        item["height_cm"] = body.height_cm
    if body.weight_kg is not None:
        item["weight_kg"] = body.weight_kg
    _profiles.put_item(Item=item)
    return {"saved": True}


ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")

@api_router.delete("/admin/appointments")
def clear_all_appointments(secret: str):
    """Delete all items in the appointments table. Requires ADMIN_SECRET query param."""
    if not ADMIN_SECRET or secret != ADMIN_SECRET:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Forbidden")

    # Scan and batch delete all items
    deleted = 0
    while True:
        resp = _appts.scan(ProjectionExpression="user_id, #ts",
                           ExpressionAttributeNames={"#ts": "timestamp"})
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
# Medical Q&A (RAG)
# ---------------------------------------------------------------------------
from pydantic import BaseModel

class QARequest(BaseModel):
    question: str

@api_router.post("/qa")
def medical_qa(req: QARequest):
    """Answer a medical question using RAG over MedlinePlus."""
    from fastapi import HTTPException
    try:
        result = answer_question(req.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RAG service unavailable: {e}")


# ---------------------------------------------------------------------------
# Drug Query (RAG)
# ---------------------------------------------------------------------------
class DrugRequest(BaseModel):
    drug_name: str

@api_router.post("/drug")
def drug_query(req: DrugRequest):
    """Look up drug information from MedlinePlus via RAG."""
    from fastapi import HTTPException
    try:
        result = answer_question(
            f"What is {req.drug_name}? What is it used for, what are the side effects, "
            f"and what precautions should patients know?"
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RAG service unavailable: {e}")
