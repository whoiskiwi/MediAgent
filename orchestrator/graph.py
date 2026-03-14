"""LangGraph orchestrator: Patient text → Agent 1 → Agent 2 → Agent 3 → Response

Wires the three agents into a sequential StateGraph using the shared
AgentState TypedDict defined in schemas.py.
"""

from langgraph.graph import END, START, StateGraph

from schemas import (
    AgentState,
    AppointmentQuery,
    ResponseInput,
    SymptomInput,
)

from agents.symptom_classifier.agent import SymptomClassifierAgent
from agents.appointment_retriever.agent import AppointmentRetrieverAgent
from agents.response_generator.agent import ResponseGeneratorAgent

# ---------------------------------------------------------------------------
# Lazy-initialized singletons — models are loaded once on first invoke
# ---------------------------------------------------------------------------
_classifier: SymptomClassifierAgent | None = None
_retriever: AppointmentRetrieverAgent | None = None
_generator: ResponseGeneratorAgent | None = None


def _get_classifier() -> SymptomClassifierAgent:
    global _classifier
    if _classifier is None:
        _classifier = SymptomClassifierAgent()
    return _classifier


def _get_retriever() -> AppointmentRetrieverAgent:
    global _retriever
    if _retriever is None:
        _retriever = AppointmentRetrieverAgent()
    return _retriever


def _get_generator() -> ResponseGeneratorAgent:
    global _generator
    if _generator is None:
        _generator = ResponseGeneratorAgent()
    return _generator


# ---------------------------------------------------------------------------
# Node functions — each receives full state, returns partial update dict
# ---------------------------------------------------------------------------

def classify_symptoms(state: AgentState) -> dict:
    """Agent 1: classify patient symptoms → department + urgency."""
    agent = _get_classifier()
    result = agent.classify(SymptomInput(patient_text=state["patient_text"]))
    return {
        "department": result.department,
        "urgency": result.urgency,
    }


def retrieve_appointment(state: AgentState) -> dict:
    """Agent 2: query DynamoDB for an available doctor slot."""
    agent = _get_retriever()
    result = agent.retrieve(AppointmentQuery(department=state["department"]))
    return {
        "doctor": result.doctor,
        "time_slot": result.time_slot,
    }


def generate_response(state: AgentState) -> dict:
    """Agent 3: produce appointment confirmation + pre-visit instructions."""
    agent = _get_generator()
    result = agent.generate(
        ResponseInput(
            patient_text=state["patient_text"],
            department=state["department"],
            doctor=state["doctor"],
            time_slot=state["time_slot"],
            urgency=state["urgency"],
        )
    )
    return {
        "confirmation": result.confirmation,
        "instructions": result.instructions,
    }


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the 3-agent pipeline graph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_symptoms", classify_symptoms)
    workflow.add_node("retrieve_appointment", retrieve_appointment)
    workflow.add_node("generate_response", generate_response)

    workflow.add_edge(START, "classify_symptoms")
    workflow.add_edge("classify_symptoms", "retrieve_appointment")
    workflow.add_edge("retrieve_appointment", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()


# Pre-built compiled graph — import this to run the pipeline
app = build_graph()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run(patient_text: str) -> AgentState:
    """Run the full pipeline for a patient symptom description.

    Returns the final AgentState with all fields populated.
    """
    result = app.invoke({"patient_text": patient_text})
    return result
