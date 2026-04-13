"""
Lightweight metrics tracker.

Usage:
    from monitoring.tracker import record

    record("classifier", latency_ms=142.3, success=True)
    record("pipeline", latency_ms=3200.0, success=True,
           urgency="Urgent", department="Cardiology")
"""
import json
from datetime import datetime, timezone


def record(component: str, latency_ms: float, success: bool = True, **meta) -> None:
    """Write one metrics row to the database. Silently swallows errors."""
    try:
        from db.backend import log_metric
        log_metric(
            timestamp=datetime.now(timezone.utc).isoformat(),
            component=component,
            latency_ms=round(latency_ms, 2),
            success=1 if success else 0,
            meta=json.dumps(meta) if meta else None,
        )
    except Exception as e:
        print(f"[Monitor] log_metric failed: {e}")
