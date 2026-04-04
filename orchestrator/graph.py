"""LangGraph orchestrator: Patient text → Agent 1 → Agent 2 → Agent 3 → Response

Wires the three agents into a sequential StateGraph using the shared
AgentState TypedDict defined in schemas.py.
"""

from langgraph.graph import END, START, StateGraph

from schemas import AgentState

from agents.symptom_classifier.agent import run_classifier
from agents.appointment_retriever.agent import run_retriever
from agents.response_generator.agent import run_generator
from agents.causes_generator.agent import run_causes_generator

# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

_builder = StateGraph(AgentState)

_builder.add_node("classify",        lambda state: run_classifier(state))
_builder.add_node("retrieve",        lambda state: run_retriever(state))
_builder.add_node("generate",        lambda state: run_generator(state))
_builder.add_node("causes",          lambda state: run_causes_generator(state))

_builder.add_edge(START,      "classify")
_builder.add_edge("classify", "retrieve")
_builder.add_edge("retrieve", "generate")
_builder.add_edge("generate", "causes")
_builder.add_edge("causes",   END)

graph = _builder.compile()


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
) -> AgentState:
    """Run the full three-agent pipeline synchronously."""
    initial_state: AgentState = {"patient_text": patient_text}
    if user_age is not None:
        initial_state["user_age"] = user_age
    if user_gender:
        initial_state["user_gender"] = user_gender
    if blood_type:
        initial_state["blood_type"] = blood_type
    if allergies:
        initial_state["allergies"] = allergies
    if height_cm is not None:
        initial_state["height_cm"] = height_cm
    if weight_kg is not None:
        initial_state["weight_kg"] = weight_kg
    return graph.invoke(initial_state)
