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
)
from orchestrator.graph import run_pipeline
from auth.dependencies import get_current_user

api_router = APIRouter()

APPOINTMENTS_TABLE = os.getenv("APPOINTMENTS_TABLE", "medi-agent-appointments")
AWS_REGION         = os.getenv("AWS_REGION", "us-west-2")

_ddb   = boto3.resource("dynamodb", region_name=AWS_REGION)
_appts = _ddb.Table(APPOINTMENTS_TABLE)

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


@api_router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, user: Optional[dict] = Depends(_optional_user)):
    age    = user.get("age")    if user else req.user_age
    gender = user.get("gender") if user else req.user_gender
    state  = run_pipeline(req.symptom, user_age=age, user_gender=gender)

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
        ),
    )

    # Save to appointment history if user is logged in
    if user:
        _appts.put_item(Item={
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
        })

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
