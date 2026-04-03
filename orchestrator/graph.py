"""LangGraph orchestrator: Patient text → Agent 1 → Agent 2 → Agent 3 → Response

Wires the three agents into a sequential StateGraph using the shared
AgentState TypedDict defined in schemas.py.
"""

from langgraph.graph import END, START, StateGraph

from schemas import AgentState

from agents.symptom_classifier.agent import run_classifier
from agents.appointment_retriever.agent import run_retriever
from agents.response_generator.agent import run_generator

# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

_builder = StateGraph(AgentState)

_builder.add_node("classify",  lambda state: run_classifier(state))
_builder.add_node("retrieve",  lambda state: run_retriever(state))
_builder.add_node("generate",  lambda state: run_generator(state))

_builder.add_edge(START,      "classify")
_builder.add_edge("classify", "retrieve")
_builder.add_edge("retrieve", "generate")
_builder.add_edge("generate", END)

graph = _builder.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(patient_text: str, age: int = None, gender: str = None) -> AgentState:
    """Run the full three-agent pipeline synchronously."""
    initial: AgentState = {"patient_text": patient_text}
    if age is not None:
        initial["age"] = age
    if gender is not None:
        initial["gender"] = gender
    return graph.invoke(initial)
