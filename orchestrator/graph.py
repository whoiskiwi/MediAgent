"""
LangGraph Orchestrator
Wires Agent 1 → Agent 2 → Agent 3 into a single compiled graph.

Usage:
    from orchestrator.graph import run_pipeline

    result = run_pipeline("I have severe chest pain and difficulty breathing")
    print(result["department"])    # "Cardiology"
    print(result["doctor"])        # "Dr. Smith"
    print(result["confirmation"])  # "Your appointment has been confirmed …"
"""
import sys
from pathlib import Path

from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schemas import AgentState

from agents.symptom_classifier.agent import run_classifier
from agents.appointment_retriever.agent import run_retriever
from agents.response_generator.agent import run_generator

# ---------------------------------------------------------------------------
# Build the graph (done once at import time)
# ---------------------------------------------------------------------------

_builder = StateGraph(AgentState)

_builder.add_node("classify", lambda state: run_classifier(state))
_builder.add_node("retrieve", lambda state: run_retriever(state))
_builder.add_node("generate", lambda state: run_generator(state))

_builder.add_edge(START, "classify")
_builder.add_edge("classify", "retrieve")
_builder.add_edge("retrieve", "generate")
_builder.add_edge("generate", END)

graph = _builder.compile()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(patient_text: str) -> AgentState:
    """
    Run the full three-agent pipeline synchronously.

    Args:
        patient_text: Raw symptom description from the patient.

    Returns:
        Final AgentState containing all intermediate and final outputs:
        {
            patient_text, department, urgency,   # from Agent 1
            doctor, time_slot,                   # from Agent 2
            confirmation, instructions           # from Agent 3
        }
    """
    initial_state: AgentState = {"patient_text": patient_text}
    final_state: AgentState = graph.invoke(initial_state)
    return final_state
