"""
Agent 2 — Appointment Retriever
No ML model. Queries AWS DynamoDB (DoctorSchedule table) for an available
doctor in the department produced by Agent 1.

Time slot logic:
  - Base time = next half-hour boundary after server's current time
               (e.g. 14:23 → 14:30,  14:31 → 15:00)
  - If another appointment was already booked today, the next slot is
    max(base_time, latest_booked_slot + 30 min) to avoid collisions.
  - Doctor is picked from DoctorSchedule by department (first available).

DynamoDB schema (DoctorSchedule):
  PK  = department        (e.g. "Cardiology")
  SK  = doctor#day#time   (e.g. "Dr. Smith#Monday#09:00")
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from schemas import AgentState
from db.backend import scan_appointments_by_day, get_doctor_by_department

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _next_half_hour(dt: datetime) -> datetime:
    """Round dt up to the next :00 or :30 boundary."""
    minute = dt.minute
    if minute < 30:
        return dt.replace(minute=30, second=0, microsecond=0)
    else:
        return (dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)


def _latest_booked_slot() -> Optional[datetime]:
    """
    Scan the appointments table for the most recently booked time_slot today
    and return it as a datetime. Returns None if no appointments today.
    Time slots are stored as 'Monday 14:30' etc.
    """
    now = datetime.now(timezone.utc)
    today_name = _DAY_NAMES[now.weekday()]
    try:
        slots = scan_appointments_by_day(today_name)
        if not slots:
            return None
        # Parse "Monday 14:30" → datetime today at 14:30
        times = []
        for ts in slots:
            parts = ts.split(" ", 1)
            if len(parts) == 2:
                try:
                    t = datetime.strptime(parts[1], "%H:%M").replace(
                        year=now.year, month=now.month, day=now.day,
                        tzinfo=timezone.utc
                    )
                    times.append(t)
                except ValueError:
                    pass
        return max(times) if times else None
    except Exception as e:
        print(f"[Agent2] Warning: could not read latest booked slot: {e}")
        return None


def _compute_next_slot() -> Tuple[str, str]:
    """
    Return (day_name, 'HH:MM') for the next available appointment slot.
    Base = next half-hour from now.
    If a booking already exists at or after base, push to latest + 30 min.
    """
    now  = datetime.now(timezone.utc)
    base = _next_half_hour(now)

    latest = _latest_booked_slot()
    if latest is not None and latest >= base:
        base = latest + timedelta(minutes=30)

    day_name = _DAY_NAMES[base.weekday()]
    time_str = base.strftime("%H:%M")
    return day_name, time_str


# ---------------------------------------------------------------------------
# Doctor lookup (unchanged logic — pick any available doctor for department)
# ---------------------------------------------------------------------------

def _query_doctor(department: str) -> Optional[str]:
    """Return the first available doctor for the given department."""
    return get_doctor_by_department(department)


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def run_retriever(state: AgentState) -> AgentState:
    """LangGraph node — writes 'doctor' and 'time_slot' into state."""
    department = state.get("department", "General Practice")

    doctor = _query_doctor(department)
    if not doctor:
        doctor = "No doctor available"
        time_slot = "N/A"
        print(f"[Agent2] → No doctors found for department={department}")
    else:
        day_name, time_str = _compute_next_slot()
        time_slot = f"{day_name} {time_str}"
        print(f"[Agent2] → doctor={doctor}, time_slot={time_slot}")

    return {**state, "doctor": doctor, "time_slot": time_slot}
