from app.services.doctor_service import get_doctors_by_department


def run_booking(department: str, urgency: str):
    doctors = get_doctors_by_department(department)

    if not doctors:
        return {"doctor": None, "slot": None, "message": "No doctors available"}

    doctor = doctors[0]
    slots = doctor.get("available_slots", [])
    slot = slots[0] if slots else None

    return {
        "doctor_id": doctor.get("doctor_id"),
        "doctor": doctor.get("name"),
        "department": doctor.get("department"),
        "slot": slot,
        "message": "OK" if slot else "Doctor has no available slots",
    }