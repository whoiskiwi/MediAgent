"""
SQLite backend — replaces AWS DynamoDB for local development.
Database file: medi-agent.db (project root, configurable via SQLITE_DB_PATH).

Tables:
    users               — auth accounts
    appointments        — appointment history (PK: user_id + timestamp)
    patient_profiles    — blood type, allergies, height, weight
    doctor_schedule     — department → doctor mapping (seeded from doctor_schedules.json)
"""
import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path(os.getenv(
    "SQLITE_DB_PATH",
    str(Path(__file__).resolve().parents[1] / "medi-agent.db"),
))

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id         TEXT PRIMARY KEY,
    email           TEXT UNIQUE NOT NULL,
    name            TEXT,
    hashed_password TEXT,
    age             INTEGER,
    gender          TEXT,
    google_id       TEXT,
    created_at      TEXT,
    last_login      TEXT
);
CREATE INDEX IF NOT EXISTS users_email ON users(email);

CREATE TABLE IF NOT EXISTS appointments (
    user_id      TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    appt_id      TEXT,
    symptom      TEXT,
    department   TEXT,
    urgency      TEXT,
    doctor       TEXT,
    time_slot    TEXT,
    confirmation TEXT,
    instructions TEXT,
    first_aid    TEXT,
    PRIMARY KEY (user_id, timestamp)
);

CREATE TABLE IF NOT EXISTS patient_profiles (
    user_id    TEXT PRIMARY KEY,
    blood_type TEXT,
    allergies  TEXT,   -- JSON array
    height_cm  INTEGER,
    weight_kg  REAL
);

CREATE TABLE IF NOT EXISTS doctor_schedule (
    department TEXT NOT NULL,
    doctor     TEXT NOT NULL,
    available  INTEGER DEFAULT 1,
    PRIMARY KEY (department, doctor)
);

CREATE TABLE IF NOT EXISTS metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    component   TEXT NOT NULL,
    latency_ms  REAL NOT NULL,
    success     INTEGER DEFAULT 1,
    meta        TEXT
);
CREATE INDEX IF NOT EXISTS metrics_ts   ON metrics(timestamp);
CREATE INDEX IF NOT EXISTS metrics_comp ON metrics(component);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _row_to_dict(row) -> dict:
    return dict(row) if row else {}


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables (idempotent) and seed doctor schedule if empty."""
    with _conn() as c:
        c.executescript(_CREATE_SQL)
        _seed_doctors(c)


def _seed_doctors(c: sqlite3.Connection) -> None:
    count = c.execute("SELECT COUNT(*) FROM doctor_schedule").fetchone()[0]
    if count > 0:
        return
    json_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "doctor_schedules.json"
    if not json_path.exists():
        print("[DB] doctor_schedules.json not found — doctor_schedule table stays empty")
        return
    with open(json_path) as f:
        records = json.load(f)
    seen: set = set()
    for rec in records:
        key = (rec["department"], rec["doctor"])
        if key not in seen:
            seen.add(key)
            c.execute(
                "INSERT OR IGNORE INTO doctor_schedule (department, doctor, available) VALUES (?, ?, 1)",
                (rec["department"], rec["doctor"]),
            )
    print(f"[DB] Seeded {len(seen)} doctor records into doctor_schedule")


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def get_user_by_email(email: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE email = ? LIMIT 1", (email,)).fetchone()
        return _row_to_dict(row) if row else None


def put_user(item: dict) -> None:
    with _conn() as c:
        c.execute(
            """INSERT INTO users
               (user_id, email, name, hashed_password, age, gender, google_id, created_at, last_login)
               VALUES (:user_id, :email, :name, :hashed_password, :age, :gender, :google_id, :created_at, :last_login)""",
            {
                "user_id":         item["user_id"],
                "email":           item["email"],
                "name":            item.get("name", ""),
                "hashed_password": item.get("hashed_password", ""),
                "age":             item.get("age"),
                "gender":          item.get("gender"),
                "google_id":       item.get("google_id"),
                "created_at":      item.get("created_at"),
                "last_login":      item.get("last_login"),
            },
        )


def update_user_last_login(user_id: str, timestamp: str) -> None:
    with _conn() as c:
        c.execute("UPDATE users SET last_login = ? WHERE user_id = ?", (timestamp, user_id))


def update_user_google(user_id: str, timestamp: str, google_id: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE users SET last_login = ?, google_id = ? WHERE user_id = ?",
            (timestamp, google_id, user_id),
        )


# ---------------------------------------------------------------------------
# Appointments
# ---------------------------------------------------------------------------

def put_appointment(item: dict) -> None:
    with _conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO appointments
               (user_id, timestamp, appt_id, symptom, department, urgency,
                doctor, time_slot, confirmation, instructions, first_aid)
               VALUES (:user_id, :timestamp, :appt_id, :symptom, :department, :urgency,
                       :doctor, :time_slot, :confirmation, :instructions, :first_aid)""",
            {
                "user_id":      item["user_id"],
                "timestamp":    item["timestamp"],
                "appt_id":      item.get("appt_id"),
                "symptom":      item.get("symptom"),
                "department":   item.get("department"),
                "urgency":      item.get("urgency"),
                "doctor":       item.get("doctor"),
                "time_slot":    item.get("time_slot"),
                "confirmation": item.get("confirmation"),
                "instructions": item.get("instructions"),
                "first_aid":    item.get("first_aid"),
            },
        )


def get_appointments_by_user(user_id: str, limit: int = 20) -> list:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM appointments WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_appointment(user_id: str, timestamp: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM appointments WHERE user_id = ? AND timestamp = ?",
            (user_id, timestamp),
        ).fetchone()
        return _row_to_dict(row) if row else None


def delete_appointment(user_id: str, timestamp: str) -> None:
    with _conn() as c:
        c.execute(
            "DELETE FROM appointments WHERE user_id = ? AND timestamp = ?",
            (user_id, timestamp),
        )


def delete_all_appointments() -> int:
    with _conn() as c:
        count = c.execute("SELECT COUNT(*) FROM appointments").fetchone()[0]
        c.execute("DELETE FROM appointments")
        return count


def scan_appointments_by_day(day_name: str) -> list:
    """Return time_slot values for appointments whose time_slot starts with day_name."""
    with _conn() as c:
        rows = c.execute(
            "SELECT time_slot FROM appointments WHERE time_slot LIKE ?",
            (f"{day_name}%",),
        ).fetchall()
        return [r["time_slot"] for r in rows]


# ---------------------------------------------------------------------------
# Patient profiles
# ---------------------------------------------------------------------------

def get_profile(user_id: str) -> dict:
    with _conn() as c:
        row = c.execute("SELECT * FROM patient_profiles WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            return {}
        d = _row_to_dict(row)
        # Deserialise allergies JSON string → list
        if d.get("allergies"):
            try:
                d["allergies"] = json.loads(d["allergies"])
            except Exception:
                d["allergies"] = []
        return d


def put_profile(item: dict) -> None:
    allergies = item.get("allergies")
    if isinstance(allergies, list):
        allergies = json.dumps(allergies)
    with _conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO patient_profiles
               (user_id, blood_type, allergies, height_cm, weight_kg)
               VALUES (?, ?, ?, ?, ?)""",
            (
                item["user_id"],
                item.get("blood_type"),
                allergies,
                item.get("height_cm"),
                item.get("weight_kg"),
            ),
        )


# ---------------------------------------------------------------------------
# Doctor schedule
# ---------------------------------------------------------------------------

def get_doctor_by_department(department: str) -> Optional[str]:
    with _conn() as c:
        row = c.execute(
            "SELECT doctor FROM doctor_schedule WHERE department = ? AND available = 1 LIMIT 1",
            (department,),
        ).fetchone()
        return row["doctor"] if row else None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def log_metric(timestamp: str, component: str, latency_ms: float,
               success: int = 1, meta: str = None) -> None:
    with _conn() as c:
        c.execute(
            """INSERT INTO metrics (timestamp, component, latency_ms, success, meta)
               VALUES (?, ?, ?, ?, ?)""",
            (timestamp, component, latency_ms, success, meta),
        )


def get_metrics(limit: int = 200, component: str = None) -> list:
    with _conn() as c:
        if component:
            rows = c.execute(
                "SELECT * FROM metrics WHERE component = ? ORDER BY timestamp DESC LIMIT ?",
                (component, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM metrics ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_metrics_summary() -> list:
    """Per-component aggregates: count, avg_latency, successes."""
    with _conn() as c:
        rows = c.execute(
            """SELECT component,
                      COUNT(*) AS count,
                      AVG(latency_ms) AS avg_latency,
                      SUM(success) AS successes
               FROM metrics
               GROUP BY component
               ORDER BY component""",
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
