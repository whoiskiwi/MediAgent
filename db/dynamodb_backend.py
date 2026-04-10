"""
DynamoDB backend — wraps existing AWS DynamoDB calls behind the same
interface as sqlite_backend.py.

Tables (configured via env vars):
    USERS_TABLE        → medi-agent-users
    APPOINTMENTS_TABLE → medi-agent-appointments
    PROFILES_TABLE     → medi-agent-patient-profiles
    DDB_TABLE          → DoctorSchedule
"""
import os
from datetime import datetime, timezone
from typing import Optional

import boto3
from boto3.dynamodb.conditions import Key as DDBKey, Attr

AWS_REGION         = os.getenv("AWS_REGION", "us-west-2")
USERS_TABLE        = os.getenv("USERS_TABLE", "medi-agent-users")
APPOINTMENTS_TABLE = os.getenv("APPOINTMENTS_TABLE", "medi-agent-appointments")
PROFILES_TABLE     = os.getenv("PROFILES_TABLE", "medi-agent-patient-profiles")
DDB_TABLE          = os.getenv("DDB_TABLE", "DoctorSchedule")

_ddb      = boto3.resource("dynamodb", region_name=AWS_REGION)
_users    = _ddb.Table(USERS_TABLE)
_appts    = _ddb.Table(APPOINTMENTS_TABLE)
_profiles = _ddb.Table(PROFILES_TABLE)
_doctors  = _ddb.Table(DDB_TABLE)


def init_db() -> None:
    """No-op for DynamoDB — tables are managed via AWS console / CDK."""
    pass


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def get_user_by_email(email: str) -> Optional[dict]:
    resp  = _users.query(
        IndexName="email-index",
        KeyConditionExpression=DDBKey("email").eq(email),
        Limit=1,
    )
    items = resp.get("Items", [])
    if not items:
        return None
    user = dict(items[0])
    if user.get("age") is not None:
        user["age"] = int(user["age"])
    return user


def put_user(item: dict) -> None:
    _users.put_item(Item=item)


def update_user_last_login(user_id: str, timestamp: str) -> None:
    _users.update_item(
        Key={"user_id": user_id},
        UpdateExpression="SET last_login = :t",
        ExpressionAttributeValues={":t": timestamp},
    )


def update_user_google(user_id: str, timestamp: str, google_id: str) -> None:
    _users.update_item(
        Key={"user_id": user_id},
        UpdateExpression="SET last_login = :t, google_id = :g",
        ExpressionAttributeValues={":t": timestamp, ":g": google_id},
    )


# ---------------------------------------------------------------------------
# Appointments
# ---------------------------------------------------------------------------

def put_appointment(item: dict) -> None:
    _appts.put_item(Item=item)


def get_appointments_by_user(user_id: str, limit: int = 20) -> list:
    resp = _appts.query(
        KeyConditionExpression=DDBKey("user_id").eq(user_id),
        ScanIndexForward=False,
        Limit=limit,
    )
    return resp.get("Items", [])


def get_appointment(user_id: str, timestamp: str) -> Optional[dict]:
    resp = _appts.get_item(Key={"user_id": user_id, "timestamp": timestamp})
    item = resp.get("Item")
    return dict(item) if item else None


def delete_appointment(user_id: str, timestamp: str) -> None:
    _appts.delete_item(Key={"user_id": user_id, "timestamp": timestamp})


def delete_all_appointments() -> int:
    deleted = 0
    while True:
        resp  = _appts.scan(ProjectionExpression="user_id, #ts",
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
    return deleted


def scan_appointments_by_day(day_name: str) -> list:
    """Return time_slot values for appointments whose time_slot starts with day_name."""
    resp  = _appts.scan(
        FilterExpression=Attr("time_slot").begins_with(day_name),
        ProjectionExpression="time_slot",
    )
    return [item.get("time_slot", "") for item in resp.get("Items", [])]


# ---------------------------------------------------------------------------
# Patient profiles
# ---------------------------------------------------------------------------

def get_profile(user_id: str) -> dict:
    resp = _profiles.get_item(Key={"user_id": user_id})
    item = resp.get("Item", {})
    return dict(item) if item else {}


def put_profile(item: dict) -> None:
    _profiles.put_item(Item=item)


# ---------------------------------------------------------------------------
# Doctor schedule
# ---------------------------------------------------------------------------

def get_doctor_by_department(department: str) -> Optional[str]:
    resp  = _doctors.query(
        KeyConditionExpression=DDBKey("department").eq(department),
        FilterExpression="available = :t",
        ExpressionAttributeValues={":t": True},
        Limit=1,
    )
    items = resp.get("Items", [])
    if not items:
        return None
    item   = items[0]
    doctor = item.get("doctor")
    if not doctor:
        sk = item.get("sk", item.get("SK", ""))
        doctor = sk.split("#")[0] if sk else None
    return doctor


# ---------------------------------------------------------------------------
# Metrics (no-op — DynamoDB env uses CloudWatch for monitoring)
# ---------------------------------------------------------------------------

def log_metric(timestamp: str, component: str, latency_ms: float,
               success: int = 1, meta: str = None) -> None:
    pass


def get_metrics(limit: int = 200, component: str = None) -> list:
    return []


def get_metrics_summary() -> list:
    return []
