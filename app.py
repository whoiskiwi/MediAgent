"""
MediAgent Streamlit Frontend
"""
import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")

st.set_page_config(page_title="MediAgent", page_icon="🏥", layout="centered")

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
        st.success(f"已登录：{user.get('name', user.get('email', ''))}")
        if st.button("退出登录"):
            logout()
    else:
        tab_login, tab_reg = st.tabs(["登录", "注册"])

        with tab_login:
            with st.form("login_form"):
                email    = st.text_input("邮箱")
                password = st.text_input("密码", type="password")
                if st.form_submit_button("登录"):
                    try:
                        resp = requests.post(f"{API_BASE}/auth/login",
                                             json={"email": email, "password": password}, timeout=10)
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state["jwt_token"] = data["access_token"]
                            st.session_state["user_info"] = data["user"]
                            st.rerun()
                        else:
                            st.error(resp.json().get("detail", "登录失败"))
                    except Exception as e:
                        st.error(f"请求失败: {e}")

        with tab_reg:
            with st.form("reg_form"):
                reg_name     = st.text_input("姓名")
                reg_email    = st.text_input("邮箱")
                reg_password = st.text_input("密码", type="password")
                if st.form_submit_button("注册"):
                    try:
                        resp = requests.post(f"{API_BASE}/auth/register",
                                             json={"email": reg_email, "password": reg_password,
                                                   "name": reg_name}, timeout=10)
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state["jwt_token"] = data["access_token"]
                            st.session_state["user_info"] = data["user"]
                            st.rerun()
                        else:
                            st.error(resp.json().get("detail", "注册失败"))
                    except Exception as e:
                        st.error(f"请求失败: {e}")

# ---------------------------------------------------------------------------
# Main: query form
# ---------------------------------------------------------------------------
st.header("症状查询")

if not is_logged_in():
    st.info("登录后可查看历史预约记录。也可以直接查询症状（无需登录）。")

with st.form("query_form"):
    symptom   = st.text_area("描述你的症状", placeholder="例如：我最近头痛、发烧，持续了三天...")
    submitted = st.form_submit_button("提交查询")

if submitted and symptom.strip():
    headers = auth_headers() if is_logged_in() else {}
    with st.spinner("正在分析..."):
        try:
            resp = requests.post(f"{API_BASE}/query",
                                 json={"symptom": symptom},
                                 headers=headers, timeout=60)
            if resp.status_code == 200:
                result = resp.json()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("科室",    result.get("department", "-"))
                    st.metric("医生",    result.get("doctor",     "-"))
                    st.metric("预约时间", result.get("time_slot",  "-"))
                with col2:
                    st.subheader("医疗建议")
                    st.write(result.get("response", "-"))
            else:
                st.error(f"查询失败: {resp.text}")
        except Exception as e:
            st.error(f"请求失败: {e}")
