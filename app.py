"""
MediAgent Streamlit Frontend
"""
import os
import requests
import streamlit as st

API_BASE    = os.getenv("API_BASE", "http://localhost:8000/api/v1")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="MediAgent", page_icon="🏥", layout="centered")

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
# Main: symptom query form
# ---------------------------------------------------------------------------
st.header("Symptom Query")

if not is_logged_in():
    st.info("Log in to save and view your appointment history. You can also query without logging in.")

with st.form("query_form"):
    symptom   = st.text_area("Describe your symptoms", placeholder="e.g. I have had a headache and fever for three days...")
    submitted = st.form_submit_button("Submit")

if submitted and symptom.strip():
    headers = auth_headers() if is_logged_in() else {}
    user_info = st.session_state.get("user_info", {}) if is_logged_in() else {}
    query_payload = {
        "symptom":     symptom,
        "user_age":    user_info.get("age"),
        "user_gender": user_info.get("gender"),
    }
    with st.spinner("Analyzing..."):
        try:
            resp = requests.post(f"{API_BASE}/query",
                                 json=query_payload,
                                 headers=headers, timeout=60)
            if resp.status_code == 200:
                result  = resp.json()
                urgency = result["agent1"].get("urgency", "Routine")

                # Emergency / Urgent first aid alert
                first_aid = result["agent3"].get("first_aid")
                if urgency == "Emergency":
                    st.error("🚨 **EMERGENCY — Seek immediate medical attention or call 911.**")
                    if first_aid:
                        with st.expander("Immediate First Aid Steps", expanded=True):
                            st.write(first_aid)
                elif urgency == "Urgent":
                    st.warning("⚠️ **Urgent — Seek medical attention as soon as possible.**")
                    if first_aid:
                        with st.expander("While You Wait", expanded=True):
                            st.write(first_aid)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Department", result["agent1"].get("department", "-"))
                    st.metric("Urgency",    urgency)
                    st.metric("Doctor",     result["agent2"].get("doctor",     "-"))
                    st.metric("Time Slot",  result["agent2"].get("time_slot",  "-"))
                with col2:
                    st.subheader("Medical Advice")
                    st.write(result["agent3"].get("confirmation", "-"))
                    st.caption(result["agent3"].get("instructions", ""))
            else:
                st.error(f"Query failed: {resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

# ---------------------------------------------------------------------------
# Appointment history (logged-in users only)
# ---------------------------------------------------------------------------
if is_logged_in():
    st.divider()
    st.header("Appointment History")

    try:
        resp = requests.get(f"{API_BASE}/appointments", headers=auth_headers(), timeout=10)
        if resp.status_code == 200:
            appointments = resp.json().get("appointments", [])
            if not appointments:
                st.info("No appointment history yet.")
            else:
                for appt in appointments:
                    ts        = appt.get("timestamp", "")
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
                                    st.success("Appointment cancelled.")
                                    st.rerun()
                                else:
                                    st.error("Failed to cancel appointment.")
                            except Exception as e:
                                st.error(f"Request failed: {e}")
        else:
            st.error("Failed to load appointment history.")
    except Exception as e:
        st.error(f"Request failed: {e}")
