"""LangGraph orchestrator: Patient text → Agent 1 → Agent 2 → Agent 3 → Response

Wires the four agents into a sequential StateGraph with MemorySaver for
multi-turn conversation support.
"""
import sys
import time

try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    from langgraph.checkpoint import MemorySaver  # older langgraph versions

from langgraph.graph import END, START, StateGraph

from schemas import AgentState

from agents.symptom_classifier.agent import run_classifier
from agents.appointment_retriever.agent import run_retriever
from agents.response_generator.agent import run_generator
from agents.causes_generator.agent import run_causes_generator

# ---------------------------------------------------------------------------
# Instrumentation helper
# ---------------------------------------------------------------------------

def _timed(component: str, fn):
    """Wrap an agent function to record per-component latency."""
    def wrapper(state):
        start = time.time()
        success = True
        try:
            return fn(state)
        except Exception:
            success = False
            raise
        finally:
            from monitoring.tracker import record
            record(component, latency_ms=(time.time() - start) * 1000, success=success)
    return wrapper


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

_memory = MemorySaver()

_builder = StateGraph(AgentState)

# Use module self-reference for late binding so test patches work correctly.
# _timed() still wraps each call for monitoring.
_m = sys.modules[__name__]
_builder.add_node("classify", _timed("classifier", lambda s: _m.run_classifier(s)))
_builder.add_node("retrieve", _timed("retriever",  lambda s: _m.run_retriever(s)))
_builder.add_node("generate", _timed("generator",  lambda s: _m.run_generator(s)))
_builder.add_node("causes",   _timed("causes",     lambda s: _m.run_causes_generator(s)))

_builder.add_edge(START,      "classify")
_builder.add_edge("classify", "retrieve")
_builder.add_edge("retrieve", "generate")
_builder.add_edge("generate", "causes")
_builder.add_edge("causes",   END)

graph = _builder.compile(checkpointer=_memory)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_initial_state(
    patient_text: str,
    user_age,
    user_gender,
    blood_type,
    allergies,
    height_cm,
    weight_kg,
    conversation_history: list,
) -> AgentState:
    """Prepend conversation history to patient_text for multi-turn context."""
    if conversation_history:
        lines = [
            f"Turn {i + 1}: \"{h.get('symptom', '')}\" → {h.get('department', '')} ({h.get('urgency', '')})"
            for i, h in enumerate(conversation_history[-3:])
        ]
        context = "Previous consultations:\n" + "\n".join(lines)
        patient_text = f"{context}\n\nCurrent concern: {patient_text}"

    state: AgentState = {"patient_text": patient_text}
    if user_age is not None:
        state["user_age"] = user_age
    if user_gender:
        state["user_gender"] = user_gender
    if blood_type:
        state["blood_type"] = blood_type
    if allergies:
        state["allergies"] = allergies
    if height_cm is not None:
        state["height_cm"] = height_cm
    if weight_kg is not None:
        state["weight_kg"] = weight_kg
    return state


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    patient_text: str,
    user_age: int = None,
    user_gender: str = None,
    blood_type: str = None,
    allergies: list = None,
    height_cm: int = None,
    weight_kg: float = None,
    thread_id: str = None,
    conversation_history: list = None,
    logged_in: bool = False,
) -> AgentState:
    """Run the full pipeline synchronously. Supports multi-turn via thread_id."""
    state = _build_initial_state(
        patient_text, user_age, user_gender,
        blood_type, allergies, height_cm, weight_kg,
        conversation_history or [],
    )
    state["logged_in"] = logged_in
    config = {"configurable": {"thread_id": thread_id or "anonymous"}}

    start = time.time()
    success = True
    result = None
    try:
        result = graph.invoke(state, config=config)
        return result
    except Exception:
        success = False
        raise
    finally:
        from monitoring.tracker import record
        meta: dict = {}
        if result is not None:
            meta["urgency"]    = result.get("urgency", "")
            meta["department"] = result.get("department", "")
        record("pipeline", latency_ms=(time.time() - start) * 1000, success=success, **meta)


def stream_pipeline(
    patient_text: str,
    user_age: int = None,
    user_gender: str = None,
    blood_type: str = None,
    allergies: list = None,
    height_cm: int = None,
    weight_kg: float = None,
    thread_id: str = None,
    conversation_history: list = None,
    logged_in: bool = False,
):
    """Stream pipeline updates node-by-node. Yields {node_name: state_updates}."""
    state = _build_initial_state(
        patient_text, user_age, user_gender,
        blood_type, allergies, height_cm, weight_kg,
        conversation_history or [],
    )
    state["logged_in"] = logged_in
    config = {"configurable": {"thread_id": thread_id or "anonymous"}}
    yield from graph.stream(state, config=config, stream_mode="updates")
