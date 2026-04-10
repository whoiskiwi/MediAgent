"""
MediAgent Streamlit Frontend
"""
import json
import os
import requests
import streamlit as st

API_BASE    = os.getenv("API_BASE", "http://localhost:8000/api/v1")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="MediAgent", page_icon="🏥", layout="wide")

# ---------------------------------------------------------------------------
# Handle Google OAuth callback token in URL
# ---------------------------------------------------------------------------
params = st.query_params
if "token" in params and not st.session_state.get("jwt_token"):
    st.session_state["jwt_token"] = params["token"]
    st.session_state["user_info"] = {
        "name":  params.get("name", ""),
        "email": params.get("email", ""),
    }
    st.query_params.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def is_logged_in() -> bool:
    return bool(st.session_state.get("jwt_token"))

def logout():
    st.session_state.clear()
    st.rerun()

def auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state['jwt_token']}"}

# ---------------------------------------------------------------------------
# Auth sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏥 MediAgent")

    if is_logged_in():
        user = st.session_state["user_info"]
        st.success(f"Logged in as: {user.get('name', user.get('email', ''))}")
        age    = user.get("age")
        gender = user.get("gender")
        if age or gender:
            profile_parts = []
            if gender:
                profile_parts.append(gender)
            if age:
                profile_parts.append(f"Age {age}")
            st.caption(" · ".join(profile_parts))

        # Patient profile section
        st.divider()
        st.subheader("Patient Profile")
        with st.expander("Edit medical profile", expanded=False):
            prof = st.session_state.get("patient_profile", {})
            with st.form("profile_form"):
                blood_type_input = st.selectbox(
                    "Blood Type",
                    ["Not set", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
                    index=0 if not prof.get("blood_type") else
                          ["Not set","A+","A-","B+","B-","AB+","AB-","O+","O-"].index(prof["blood_type"])
                          if prof.get("blood_type") in ["A+","A-","B+","B-","AB+","AB-","O+","O-"] else 0,
                )
                allergies_input = st.text_input(
                    "Allergies (comma-separated)",
                    value=", ".join(prof.get("allergies") or []),
                    placeholder="e.g. Penicillin, Aspirin",
                )
                col_h, col_w = st.columns(2)
                with col_h:
                    height_input = st.number_input("Height (cm)", min_value=0, max_value=250,
                                                    value=int(prof.get("height_cm") or 0), step=1)
                with col_w:
                    weight_input = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0,
                                                    value=float(prof.get("weight_kg") or 0.0), step=0.5)
                save_profile = st.form_submit_button("Save Profile")

            if save_profile:
                payload = {
                    "blood_type": blood_type_input if blood_type_input != "Not set" else None,
                    "allergies":  [a.strip() for a in allergies_input.split(",") if a.strip()] or None,
                    "height_cm":  int(height_input) if height_input > 0 else None,
                    "weight_kg":  float(weight_input) if weight_input > 0 else None,
                }
                try:
                    resp = requests.put(f"{API_BASE}/profile", json=payload,
                                        headers=auth_headers(), timeout=10)
                    if resp.status_code == 200:
                        st.session_state["patient_profile"] = payload
                        st.success("Profile saved.")
                    else:
                        st.error("Failed to save profile.")
                except Exception as e:
                    st.error(f"Request failed: {e}")

        # Load profile once per session
        if "patient_profile" not in st.session_state:
            try:
                resp = requests.get(f"{API_BASE}/profile", headers=auth_headers(), timeout=10)
                if resp.status_code == 200:
                    st.session_state["patient_profile"] = resp.json()
            except Exception:
                st.session_state["patient_profile"] = {}

        if st.button("Log out"):
            logout()
    else:
        tab_login, tab_reg = st.tabs(["Login", "Register"])

        with tab_login:
            st.link_button("Sign in with Google", f"{BACKEND_URL}/api/v1/auth/google", use_container_width=True)
            st.divider()
            with st.form("login_form"):
                email    = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login", use_container_width=True):
                    try:
                        resp = requests.post(f"{API_BASE}/auth/login",
                                             json={"email": email, "password": password}, timeout=10)
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state["jwt_token"] = data["access_token"]
                            st.session_state["user_info"] = data["user"]
                            st.rerun()
                        else:
                            st.error(resp.json().get("detail", "Login failed"))
                    except Exception as e:
                        st.error(f"Request failed: {e}")

        with tab_reg:
            with st.form("reg_form"):
                reg_name     = st.text_input("Name")
                reg_email    = st.text_input("Email")
                reg_password = st.text_input("Password", type="password")
                reg_age      = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
                reg_gender   = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
                if st.form_submit_button("Register"):
                    try:
                        payload = {
                            "email":    reg_email,
                            "password": reg_password,
                            "name":     reg_name,
                            "age":      int(reg_age) if reg_age else None,
                            "gender":   reg_gender if reg_gender != "Prefer not to say" else None,
                        }
                        resp = requests.post(f"{API_BASE}/auth/register", json=payload, timeout=10)
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state["jwt_token"] = data["access_token"]
                            st.session_state["user_info"] = data["user"]
                            st.rerun()
                        else:
                            st.error(resp.json().get("detail", "Registration failed"))
                    except Exception as e:
                        st.error(f"Request failed: {e}")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_symptom, tab_qa, tab_drug = st.tabs(
    ["Symptom Query", "Medical Q&A", "Drug Lookup"]
)

# ---------------------------------------------------------------------------
# Tab: Drug Lookup
# ---------------------------------------------------------------------------
with tab_drug:
    st.header("Drug Lookup")
    st.caption("Search for drug information — uses FDA Drug Label database, with MedlinePlus RAG as fallback.")

    with st.form("drug_form"):
        drug_name   = st.text_input("Drug name", placeholder="e.g. Ibuprofen, Metformin, Amoxicillin")
        drug_submit = st.form_submit_button("Search")

    if drug_submit and drug_name.strip():
        with st.spinner("Looking up drug information..."):
            try:
                resp = requests.post(f"{API_BASE}/drug", json={"drug_name": drug_name}, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    source_badge = "FDA" if data.get("source") == "FDA" else "MedlinePlus RAG"
                    st.subheader(drug_name)
                    st.caption(f"Source: {source_badge}")

                    # If FDA returned structured fields, show them in sections
                    fda = data.get("fda")
                    if fda:
                        for label, key in [
                            ("Indications", "indications"),
                            ("Dosage", "dosage"),
                            ("Warnings", "warnings"),
                            ("Side Effects", "side_effects"),
                        ]:
                            val = fda.get(key, "")
                            if val and val != "N/A":
                                with st.expander(label, expanded=(label == "Indications")):
                                    st.write(val)
                    else:
                        st.write(data.get("answer", "-"))

                    sources = data.get("sources", [])
                    if sources:
                        st.divider()
                        st.caption("References")
                        for src in sources:
                            if src.get("url"):
                                st.markdown(f"- [{src['title']}]({src['url']})")
                            else:
                                st.markdown(f"- {src['title']}")
                else:
                    st.error(f"Request failed: {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# ---------------------------------------------------------------------------
# Tab: Medical Q&A
# ---------------------------------------------------------------------------
with tab_qa:
    st.header("Medical Q&A")
    st.caption("Ask any medical question — get practical advice on how to relieve symptoms, prevent recurrence, and reduce discomfort.")

    with st.form("qa_form"):
        question  = st.text_input("Your question", placeholder="e.g. What causes lower back pain?")
        qa_submit = st.form_submit_button("Ask")

    if qa_submit and question.strip():
        with st.spinner("Searching knowledge base..."):
            try:
                resp = requests.post(f"{API_BASE}/qa", json={"question": question}, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    st.subheader("Answer")
                    st.write(data.get("answer", "-"))
                    sources = data.get("sources", [])
                    if sources:
                        st.divider()
                        st.caption("Sources")
                        for src in sources:
                            if src.get("url"):
                                st.markdown(f"- [{src['title']}]({src['url']})")
                            else:
                                st.markdown(f"- {src['title']}")
                else:
                    st.error(f"Request failed: {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# ---------------------------------------------------------------------------
# Tab: Symptom Query (streaming)
# ---------------------------------------------------------------------------
with tab_symptom:

    # ── Chat state initialisation ─────────────────────────────────────────
    _chat_defaults = {
        "chat_messages":  [],           # [{role, content}]
        "chat_phase":     "init",       # init | questioning | summarising | confirming | booking | done
        "chat_questions": [],           # [{q, options}]
        "chat_q_idx":     0,
        "chat_symptom":   "",
        "chat_answers":   [],           # [{q, a}]
        "chat_conv_id":   None,
    }
    for _k, _v in _chat_defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ── Header ────────────────────────────────────────────────────────────
    col_title, col_reset = st.columns([5, 1])
    with col_title:
        st.header("Symptom Query")
    with col_reset:
        st.write("")
        if st.button("New Chat", use_container_width=True):
            for _k, _v in _chat_defaults.items():
                st.session_state[_k] = _v
            st.session_state.pop("appointments_cache", None)
            st.rerun()

    if not is_logged_in():
        st.info("Log in to save and view your appointment history.")

    # ── Render all past messages ──────────────────────────────────────────
    for _msg in st.session_state.chat_messages:
        with st.chat_message(_msg["role"]):
            st.markdown(_msg["content"])

    # ── Greeting (empty state) ────────────────────────────────────────────
    if not st.session_state.chat_messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Hello! I'm your medical assistant. 👋\n\n"
                "Please describe your symptoms and I'll help you book the right appointment."
            )

    # ── PHASE: init — wait for symptom ───────────────────────────────────
    if st.session_state.chat_phase == "init":
        if _symptom_input := st.chat_input("Describe your symptoms..."):
            st.session_state.chat_messages.append({"role": "user", "content": _symptom_input})
            st.session_state.chat_symptom = _symptom_input
            with st.spinner("Thinking..."):
                try:
                    _r = requests.post(
                        f"{API_BASE}/chat/questions",
                        json={"symptom": _symptom_input},
                        timeout=30,
                    )
                    _questions = _r.json()["questions"]
                except Exception:
                    _questions = [
                        {"q": "How long have you had this symptom?",
                         "options": ["Less than 1 day", "1–3 days", "4–7 days", "More than a week"]},
                        {"q": "How severe is it?",
                         "options": ["Mild", "Moderate", "Severe", "Extreme"]},
                        {"q": "Any other symptoms?",
                         "options": ["Fever", "Nausea/Vomiting", "Pain elsewhere", "None"]},
                    ]
            st.session_state.chat_questions = _questions
            st.session_state.chat_q_idx = 0
            st.session_state.chat_phase = "questioning"
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": _questions[0]["q"],
            })
            st.rerun()

    # ── PHASE: questioning — show options for current question ────────────
    elif st.session_state.chat_phase == "questioning":
        _idx  = st.session_state.chat_q_idx
        _cq   = st.session_state.chat_questions[_idx]

        def _submit_answer(ans: str):
            st.session_state.chat_messages.append({"role": "user", "content": ans})
            st.session_state.chat_answers.append({"q": _cq["q"], "a": ans})
            _nxt = _idx + 1
            if _nxt < len(st.session_state.chat_questions):
                st.session_state.chat_q_idx = _nxt
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": st.session_state.chat_questions[_nxt]["q"],
                })
            else:
                st.session_state.chat_phase = "summarising"

        # Option buttons
        _opts = _cq["options"]
        _cols = st.columns(len(_opts))
        for _i, _opt in enumerate(_opts):
            if _cols[_i].button(_opt, key=f"opt_{_idx}_{_i}", use_container_width=True):
                _submit_answer(_opt)
                st.rerun()

        # Free-text fallback
        if _custom := st.chat_input("Or describe in your own words..."):
            _submit_answer(_custom)
            st.rerun()

    # ── PHASE: summarising — one API call, then confirming ────────────────
    elif st.session_state.chat_phase == "summarising":
        with st.spinner("Summarising your condition..."):
            try:
                _r = requests.post(
                    f"{API_BASE}/chat/summary",
                    json={
                        "symptom":  st.session_state.chat_symptom,
                        "qa_pairs": st.session_state.chat_answers,
                    },
                    timeout=30,
                )
                _summary = _r.json()["summary"]
            except Exception:
                _summary = (
                    "Thank you for sharing those details. "
                    "Based on what you've described, I'd recommend seeing a doctor. "
                    "Would you like me to book an appointment for you?"
                )
        st.session_state.chat_messages.append({"role": "assistant", "content": _summary})
        st.session_state.chat_phase = "confirming"
        st.rerun()

    # ── PHASE: confirming — book or cancel ───────────────────────────────
    elif st.session_state.chat_phase == "confirming":
        _c1, _c2 = st.columns(2)
        if _c1.button("Book Appointment", use_container_width=True, type="primary"):
            st.session_state.chat_messages.append({"role": "user", "content": "Yes, please book the appointment."})
            st.session_state.chat_phase = "booking"
            st.rerun()
        if _c2.button("Not now", use_container_width=True):
            st.session_state.chat_messages.append({"role": "user", "content": "No, not now."})
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": "No problem! Feel free to come back anytime. Take care! 😊",
            })
            st.session_state.chat_phase = "done"
            st.rerun()

    # ── PHASE: booking — stream the pipeline, show thinking live ─────────
    elif st.session_state.chat_phase == "booking":
        # Build enriched symptom string from all collected answers
        _full_symptom = st.session_state.chat_symptom
        if st.session_state.chat_answers:
            _details = "; ".join(f"{qa['q']}: {qa['a']}" for qa in st.session_state.chat_answers)
            _full_symptom = f"{_full_symptom}. Additional details — {_details}"

        _headers   = auth_headers() if is_logged_in() else {}
        _user_info = st.session_state.get("user_info", {}) if is_logged_in() else {}
        _payload   = {
            "symptom":         _full_symptom,
            "conversation_id": st.session_state.chat_conv_id,
        }
        if is_logged_in():
            _payload["user_age"]    = _user_info.get("age")
            _payload["user_gender"] = _user_info.get("gender")

        with st.chat_message("assistant"):
            _status_ph = st.empty()
            _result_ph = st.empty()

        _urgency  = "Routine"
        _dept     = ""
        _results  = {}

        try:
            with requests.post(
                f"{API_BASE}/query/stream",
                json=_payload,
                headers=_headers,
                stream=True,
                timeout=120,
            ) as _resp:
                _resp.raise_for_status()

                for _raw in _resp.iter_lines():
                    if not _raw:
                        continue
                    _line = _raw.decode("utf-8") if isinstance(_raw, bytes) else _raw
                    if not _line.startswith("data: "):
                        continue
                    _data  = json.loads(_line[6:])
                    _event = _data.get("event")

                    if _event == "error":
                        _status_ph.error(f"Pipeline error: {_data.get('detail')}")
                        break

                    elif _event == "classifier":
                        _urgency = _data.get("urgency", "Routine")
                        _dept    = _data.get("department", "")
                        _results.update({"department": _dept, "urgency": _urgency})
                        if _urgency == "Emergency":
                            _status_ph.error("🚨 **EMERGENCY — Call 911 or go to ER immediately!**")
                        elif _urgency == "Urgent":
                            _status_ph.warning("⚠️ **Urgent — Please seek medical attention soon.**")
                        else:
                            _status_ph.caption(f"Step 1/4 — Department: **{_dept}** | Urgency: **{_urgency}**")

                    elif _event == "retriever":
                        _results.update({"doctor": _data.get("doctor", ""), "time_slot": _data.get("time_slot", "")})
                        _status_ph.caption(
                            f"Step 2/4 — Doctor: **{_results['doctor']}** | Slot: **{_results['time_slot']}**"
                        )

                    elif _event == "generator":
                        _results.update({
                            "confirmation": _data.get("confirmation", ""),
                            "instructions": _data.get("instructions", ""),
                            "first_aid":    _data.get("first_aid"),
                        })
                        _status_ph.caption("Step 3/4 — Generating medical advice...")

                    elif _event == "causes":
                        _results["causes"] = _data.get("possible_causes", [])
                        _status_ph.caption("Step 4/4 — Analyzing possible causes...")

                    elif _event == "done":
                        _status_ph.empty()
                        _new_conv_id = _data.get("conversation_id")
                        if _new_conv_id:
                            st.session_state.chat_conv_id = _new_conv_id

                        # Build final chat message
                        _lines = []
                        if _urgency == "Emergency":
                            _lines.append("🚨 **EMERGENCY — Call 911 immediately!**\n")
                        if _results.get("confirmation"):
                            _lines.append(_results["confirmation"])
                        if _results.get("instructions"):
                            _lines.append(f"\n**Instructions:** {_results['instructions']}")
                        if _results.get("first_aid"):
                            _label = "Immediate First Aid" if _urgency == "Emergency" else "While you wait"
                            _lines.append(f"\n**{_label}:** {_results['first_aid']}")
                        _causes = _results.get("causes", [])
                        if _causes:
                            _lines.append("\n**Possible causes:**")
                            for _i, _c in enumerate(_causes[:3], 1):
                                _lines.append(f"{_i}. **{_c.get('cause', '')}** — {_c.get('reason', '')}")

                        _final_msg = "\n".join(_lines)
                        st.session_state.chat_messages.append({"role": "assistant", "content": _final_msg})
                        st.session_state.chat_phase = "done"
                        st.session_state.pop("appointments_cache", None)

        except Exception as _e:
            _status_ph.empty()
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": f"Sorry, something went wrong: {_e}",
            })
            st.session_state.chat_phase = "done"

        st.rerun()

    # ── PHASE: done ───────────────────────────────────────────────────────
    elif st.session_state.chat_phase == "done":
        st.chat_input("Chat ended — click New Chat to start over.", disabled=True)

    # ── Appointment History ───────────────────────────────────────────────
    if is_logged_in():
        st.divider()
        col_hdr, col_refresh = st.columns([5, 1])
        with col_hdr:
            st.header("Appointment History")
        with col_refresh:
            st.write("")
            if st.button("Refresh", use_container_width=True):
                st.session_state.pop("appointments_cache", None)

        if "appointments_cache" not in st.session_state:
            try:
                resp = requests.get(f"{API_BASE}/appointments", headers=auth_headers(), timeout=10)
                if resp.status_code == 200:
                    st.session_state["appointments_cache"] = resp.json().get("appointments", [])
                else:
                    st.error("Failed to load appointment history.")
                    st.session_state["appointments_cache"] = []
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.session_state["appointments_cache"] = []

        appointments = st.session_state.get("appointments_cache", [])
        if not appointments:
            st.info("No appointment history yet.")
        else:
            for appt in appointments:
                ts         = appt.get("timestamp", "")
                ts_display = ts[:19].replace("T", " ")
                with st.expander(f"{ts_display} — {appt.get('department', '')} | {appt.get('doctor', '')}"):
                    st.write(f"**Symptom:** {appt.get('symptom', '-')}")
                    st.write(f"**Urgency:** {appt.get('urgency', '-')}")
                    st.write(f"**Time Slot:** {appt.get('time_slot', '-')}")
                    st.write(f"**Confirmation:** {appt.get('confirmation', '-')}")
                    st.write(f"**Instructions:** {appt.get('instructions', '-')}")
                    if appt.get("first_aid"):
                        st.write(f"**First Aid:** {appt.get('first_aid')}")
                    if st.button("Cancel Appointment", key=f"cancel_{ts}"):
                        try:
                            import urllib.parse
                            encoded_ts = urllib.parse.quote(ts, safe="")
                            del_resp = requests.delete(
                                f"{API_BASE}/appointments/{encoded_ts}",
                                headers=auth_headers(), timeout=10
                            )
                            if del_resp.status_code == 200:
                                st.session_state.pop("appointments_cache", None)
                                st.rerun()
                            else:
                                st.error("Failed to cancel appointment.")
                        except Exception as e:
                            st.error(f"Request failed: {e}")

