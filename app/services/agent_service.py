from app.schema import Agent1Output, Agent2Output, Agent3Output

from app.agents.triage_service import run_triage
from app.agents.booking_agent import run_booking


def run_pipeline(symptom: str):
    # Agent1: triage
    agent1: Agent1Output = run_triage(symptom)

    # Agent2: booking (db-driven)
    booking = run_booking(agent1.department, agent1.urgency)

    agent2 = Agent2Output(
        doctor=booking.get("doctor") or "",
        time_slot=booking.get("slot") or ""
    )

    # Agent3: confirm
    if agent2.doctor and agent2.time_slot:
        agent3 = Agent3Output(
            confirmation="Appointment booked",
            instructions="Please arrive 15 minutes early."
        )
    else:
        agent3 = Agent3Output(
            confirmation="No appointment available",
            instructions="Please try another department or time later."
        )

    return agent1, agent2, agent3