"""
MediAgent Monitoring Dashboard
Run on a separate port:
    streamlit run monitoring_app.py --server.port 8502
"""
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from db.sqlite_backend import DB_PATH, get_metrics, get_metrics_summary

st.set_page_config(
    page_title="MediAgent Monitor",
    page_icon="📊",
    layout="wide",
)

st.title("MediAgent — System Monitor")
st.caption(f"Database: `{DB_PATH}`")

# ---------------------------------------------------------------------------
# Refresh controls
# ---------------------------------------------------------------------------
col_btn, col_auto = st.columns([1, 1])
with col_btn:
    if st.button("Refresh"):
        st.rerun()
with col_auto:
    auto_refresh = st.checkbox("Auto-refresh every 30 s", value=False)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
rows    = get_metrics(limit=500)
summary = get_metrics_summary()

if not rows:
    st.info("No metrics yet. Submit a query in the main app (port 8501) to start collecting data.")
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    st.stop()

df = pd.DataFrame(rows)
df["timestamp"]  = pd.to_datetime(df["timestamp"], utc=True)
df["latency_ms"] = df["latency_ms"].astype(float)
df["success"]    = df["success"].astype(int)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_components, tab_models, tab_log = st.tabs(
    ["Overview", "Components", "Model Metrics", "Activity Log"]
)

# ── Overview ────────────────────────────────────────────────────────────────
with tab_overview:
    now      = datetime.now(timezone.utc)
    last_24h = df[df["timestamp"] >= now - timedelta(hours=24)]
    pipe_df  = df[df["component"] == "pipeline"]

    total_24h   = int((last_24h["component"] == "pipeline").sum())
    avg_latency = pipe_df["latency_ms"].mean() if not pipe_df.empty else 0
    success_pct = df["success"].mean() * 100 if not df.empty else 0
    total_calls = len(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Requests (last 24 h)", total_24h)
    c2.metric("Avg Pipeline Latency", f"{avg_latency:,.0f} ms")
    c3.metric("Overall Success Rate", f"{success_pct:.1f}%")
    c4.metric("Total Calls (all time)", total_calls)

    st.divider()
    st.subheader("Request Volume — last 24 h (by hour)")
    pipe_24h = last_24h[last_24h["component"] == "pipeline"].copy()
    if not pipe_24h.empty:
        pipe_24h = pipe_24h.set_index("timestamp")
        hourly = pipe_24h.resample("1h").size().rename("requests")
        st.bar_chart(hourly)
    else:
        st.info("No pipeline calls in the last 24 h.")

# ── Components ───────────────────────────────────────────────────────────────
with tab_components:
    st.subheader("Per-Component Performance")
    if summary:
        sum_df = pd.DataFrame(summary)
        sum_df["avg_latency_ms"] = sum_df["avg_latency"].round(1)
        sum_df["success_rate"]   = (sum_df["successes"] / sum_df["count"] * 100).round(1)
        st.dataframe(
            sum_df[["component", "count", "avg_latency_ms", "success_rate"]].rename(columns={
                "component":       "Component",
                "count":           "Calls",
                "avg_latency_ms":  "Avg Latency (ms)",
                "success_rate":    "Success %",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.subheader("Avg Latency by Component")
        chart_df = sum_df.set_index("component")["avg_latency_ms"].sort_values(ascending=False)
        st.bar_chart(chart_df)
    else:
        st.info("No component data yet.")

# ── Model Metrics ────────────────────────────────────────────────────────────
with tab_models:
    meta_rows = df[df["meta"].notna()].copy()
    if not meta_rows.empty:
        meta_rows["parsed"] = meta_rows["meta"].apply(
            lambda x: json.loads(x) if x else {}
        )
        pipe_meta = meta_rows[meta_rows["component"] == "pipeline"]

        if not pipe_meta.empty:
            col_l, col_r = st.columns(2)

            urgencies = pipe_meta["parsed"].apply(lambda x: x.get("urgency") or "Unknown")
            with col_l:
                st.subheader("Urgency Distribution")
                st.bar_chart(urgencies.value_counts().rename("count"))

            departments = pipe_meta["parsed"].apply(lambda x: x.get("department") or "Unknown")
            with col_r:
                st.subheader("Department Distribution")
                st.bar_chart(departments.value_counts().rename("count"))

            # Recent requests table
            st.divider()
            st.subheader("Recent Pipeline Requests")
            recent = pipe_meta[["timestamp", "latency_ms", "success", "parsed"]].head(20).copy()
            recent["urgency"]    = recent["parsed"].apply(lambda x: x.get("urgency", ""))
            recent["department"] = recent["parsed"].apply(lambda x: x.get("department", ""))
            recent["timestamp"]  = recent["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            recent["success"]    = recent["success"].map({1: "OK", 0: "ERR"})
            recent["latency_ms"] = recent["latency_ms"].round(0).astype(int)
            st.dataframe(
                recent[["timestamp", "urgency", "department", "latency_ms", "success"]].rename(
                    columns={"latency_ms": "Latency (ms)", "success": "Status"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No pipeline metadata yet.")
    else:
        st.info("No model metadata yet.")

# ── Activity Log ─────────────────────────────────────────────────────────────
with tab_log:
    st.subheader("Recent Activity (last 50 calls)")
    log_df = df.head(50)[["timestamp", "component", "latency_ms", "success", "meta"]].copy()
    log_df["timestamp"]  = log_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    log_df["latency_ms"] = log_df["latency_ms"].round(1)
    log_df["success"]    = log_df["success"].map({1: "OK", 0: "ERR"})
    st.dataframe(
        log_df.rename(columns={
            "timestamp":  "Time",
            "component":  "Component",
            "latency_ms": "Latency (ms)",
            "success":    "Status",
            "meta":       "Meta",
        }),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------------
if auto_refresh:
    time.sleep(30)
    st.rerun()
