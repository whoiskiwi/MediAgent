from app.schema import Agent1Output

def run_triage(symptom: str) -> Agent1Output:
    symptom_lower = symptom.lower()

    if any(keyword in symptom_lower for keyword in ["chest pain", "stroke", "unconscious"]):
        return Agent1Output(
            department="Emergency",
            urgency="High"
        )

    if any(keyword in symptom_lower for keyword in ["fever", "vomit", "rash"]):
        return Agent1Output(
            department="General Practice",
            urgency="Medium"
        )

    return Agent1Output(
        department="General Practice",
        urgency="Low"
    )