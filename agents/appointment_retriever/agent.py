"""
Agent 2 — Appointment Retriever
No ML model. Queries AWS DynamoDB (DoctorSchedule table) for an available
doctor in the department produced by Agent 1.

DynamoDB schema:
  PK  = department        (e.g. "Cardiology")
  SK  = doctor#day#time   (e.g. "Dr. Smith#Monday#09:00")

Input : AgentState with 'department'
Output: AgentState updated with 'doctor' and 'time_slot'
"""
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import boto3
from boto3.dynamodb.conditions import Key
from dotenv import load_dotenv

# Load .env from project root (medi-agent/)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from schemas import AgentState

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
DDB_TABLE = os.getenv("DDB_TABLE", "DoctorSchedule")

# DynamoDB resource is module-level (one connection per process)
_table = boto3.resource("dynamodb", region_name=AWS_REGION).Table(DDB_TABLE)


def _query_department(department: str) -> Optional[dict]:
    """Return the first available item for the given department, or None."""
    resp = _table.query(
        KeyConditionExpression=Key("department").eq(department),
        Limit=1,
    )
    items = resp.get("Items", [])
    return items[0] if items else None


def _parse_item(item: dict) -> Tuple[str, str]:
    """
    Extract doctor and time_slot from a DynamoDB item.
    Supports two layouts:
      1. Flat attributes: doctor, day, time_slot as separate fields
      2. SK-encoded:  SK = 'doctor#day#time'
    """
    doctor = item.get("doctor")
    day = item.get("day", "")
    time_slot = item.get("time_slot", "")

    if doctor and day and time_slot:
        return doctor, f"{day} {time_slot}"

    if doctor and time_slot:
        return doctor, time_slot

    sk: str = item.get("sk", item.get("SK", ""))
    parts = sk.split("#", 2)
    if len(parts) == 3:
        doctor, day, time = parts
        return doctor, f"{day} {time}"

    return "Unknown Doctor", "Unknown Slot"


def run_retriever(state: AgentState) -> AgentState:
    """LangGraph node — writes 'doctor' and 'time_slot' into state."""
    department = state.get("department", "General Practice")

    item = _query_department(department)

    if item:
        doctor, time_slot = _parse_item(item)
        print(f"[Agent2] → doctor={doctor}, time_slot={time_slot}")
    else:
        doctor, time_slot = "No doctor available", "N/A"
        print(f"[Agent2] → No appointments found for department={department}")

    return {**state, "doctor": doctor, "time_slot": time_slot}
