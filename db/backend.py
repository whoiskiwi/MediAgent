"""
Database abstraction layer.

Backend is selected automatically:
  - AWS environment (Lambda/ECS) → DynamoDB
  - Local / anywhere else        → SQLite

Override by setting DB_BACKEND=sqlite|dynamodb explicitly.

All other modules import from here — never directly from sqlite_backend or
dynamodb_backend.
"""
import os


def _detect_backend() -> str:
    explicit = os.getenv("DB_BACKEND", "").lower()
    if explicit in ("sqlite", "dynamodb"):
        return explicit
    # AWS managed runtimes inject these variables automatically
    if os.getenv("AWS_EXECUTION_ENV") or os.getenv("ECS_CONTAINER_METADATA_URI"):
        return "dynamodb"
    return "sqlite"


_backend = _detect_backend()
print(f"[DB] backend = {_backend}")

if _backend == "dynamodb":
    from db.dynamodb_backend import (
        init_db,
        get_user_by_email,
        put_user,
        update_user_last_login,
        update_user_google,
        put_appointment,
        get_appointments_by_user,
        get_appointment,
        delete_appointment,
        delete_all_appointments,
        scan_appointments_by_day,
        get_profile,
        put_profile,
        get_doctor_by_department,
        log_metric,
        get_metrics,
        get_metrics_summary,
    )
else:
    from db.sqlite_backend import (
        init_db,
        get_user_by_email,
        put_user,
        update_user_last_login,
        update_user_google,
        put_appointment,
        get_appointments_by_user,
        get_appointment,
        delete_appointment,
        delete_all_appointments,
        scan_appointments_by_day,
        get_profile,
        put_profile,
        get_doctor_by_department,
        log_metric,
        get_metrics,
        get_metrics_summary,
    )

__all__ = [
    "init_db",
    "get_user_by_email",
    "put_user",
    "update_user_last_login",
    "update_user_google",
    "put_appointment",
    "get_appointments_by_user",
    "get_appointment",
    "delete_appointment",
    "delete_all_appointments",
    "scan_appointments_by_day",
    "get_profile",
    "put_profile",
    "get_doctor_by_department",
    "log_metric",
    "get_metrics",
    "get_metrics_summary",
]
