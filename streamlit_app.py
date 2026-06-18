"""
Streamlit chatbot UI for Chapter1 ERP Assistant.

Run backend:
    uvicorn app.main:app --reload --port 8000

Run UI:
    streamlit run streamlit_app.py
"""

import json
import os
import uuid

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Config
# -----------------------------
API_BASE = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
API_URL = f"{API_BASE}/chat/stream"

API_KEY = os.getenv("CHP1_API_TOKEN", "")
AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

SUGGESTED_PROMPTS = [
    "Show me top 10 products by sales",
    "Who are my top 5 customers this month?",
    "What's the GST summary for this quarter?",
    "Show overdue invoices",
    "What's the current stock level?",
]

st.set_page_config(
    page_title="Chapter1 ERP Assistant",
    page_icon="💼",
    layout="wide",
)

# -----------------------------
# CSS
# -----------------------------
st.markdown(
    """
<style>
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.2; }
}
.dot {
    display: inline-block;
    animation: blink 1.2s ease-in-out infinite;
    font-size: 1.4rem;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }

.session-card {
    padding: 0.4rem 0.2rem;
    border-radius: 0.4rem;
}
.small-caption {
    font-size: 0.8rem;
    opacity: 0.7;
}
</style>
""",
    unsafe_allow_html=True,
)

THINKING_HTML = '<span class="dot">●</span><span class="dot"> ●</span><span class="dot"> ●</span>'

# -----------------------------
# Backend helpers
# -----------------------------
def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return {}


def fetch_sessions():
    try:
        resp = requests.get(
            f"{API_BASE}/sessions",
            headers=AUTH_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        return safe_json(resp).get("sessions", [])
    except Exception as e:
        st.sidebar.error(f"Could not load sessions: {e}")
        return []


def create_session(name=""):
    try:
        resp = requests.post(
            f"{API_BASE}/sessions",
            json={"name": name},
            headers=AUTH_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        return safe_json(resp).get("session")
    except Exception as e:
        st.sidebar.error(f"Could not create session: {e}")
        return None


def delete_session(session_id):
    try:
        resp = requests.delete(
            f"{API_BASE}/sessions/{session_id}",
            headers=AUTH_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        st.sidebar.error(f"Could not delete session: {e}")
        return False


def rename_session(session_id, name):
    try:
        resp = requests.patch(
            f"{API_BASE}/sessions/{session_id}",
            json={"name": name},
            headers=AUTH_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        st.sidebar.error(f"Could not rename session: {e}")
        return False


def fetch_history(session_id):
    try:
        resp = requests.get(
            f"{API_BASE}/sessions/{session_id}/history",
            headers=AUTH_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        return safe_json(resp).get("messages", [])
    except Exception as e:
        st.error(f"Could not load history: {e}")
        return []


def chat_stream(session_id, query):
    """
    Expected backend SSE format:

        data: {"token": "Hello"}
        data: {"data": {...}}
        data: {"session_id": "..."}
        data: {"done": true}

    """
    collected_data = {}
    actual_sid = session_id

    try:
        with requests.post(
            API_URL,
            json={"query": query, "session_id": session_id},
            headers=AUTH_HEADERS,
            stream=True,
            timeout=600,
        ) as resp:
            resp.raise_for_status()

            buf = ""

            for raw in resp.iter_content(chunk_size=1, decode_unicode=True):
                if not raw:
                    continue

                buf += raw

                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()

                    if not line or not line.startswith("data: "):
                        continue

                    try:
                        payload = json.loads(line[len("data: "):])
                    except json.JSONDecodeError:
                        continue

                    if "token" in payload:
                        yield payload["token"]

                    elif "data" in payload and isinstance(payload["data"], dict):
                        collected_data.update(payload["data"])

                    elif "session_id" in payload:
                        actual_sid = payload["session_id"]

                    elif "error" in payload:
                        err = payload["error"]
                        if isinstance(err, dict):
                            yield f"\n\n⚠️ Error: {err.get('message', 'Unknown error')}"
                        else:
                            yield f"\n\n⚠️ Error: {err}"

                    elif payload.get("done"):
                        break

    except requests.exceptions.ConnectionError:
        yield f"⚠️ Backend not reachable. Check if FastAPI is running at `{API_BASE}`."

    except requests.exceptions.Timeout:
        yield "⚠️ The request timed out. Please try again."

    except requests.exceptions.RequestException as e:
        yield f"⚠️ Request failed: {e}"

    finally:
        st.session_state._stream_data = collected_data
        st.session_state._stream_session_id = actual_sid


# -----------------------------
# Record/table renderer
# -----------------------------
def render_records(data_dict):
    if not data_dict:
        return

    if not isinstance(data_dict, dict):
        return

    with st.expander("View structured data", expanded=False):
        for key, value in data_dict.items():
            st.markdown(f"#### {key}")

            if isinstance(value, list):
                if value and isinstance(value[0], dict):
                    df = pd.DataFrame(value)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.json(value)

            elif isinstance(value, dict):
                try:
                    df = pd.DataFrame([value])
                    st.dataframe(df, use_container_width=True)
                except Exception:
                    st.json(value)

            else:
                st.write(value)


# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None

if "editing_session_id" not in st.session_state:
    st.session_state.editing_session_id = None

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "delete_confirm_id" not in st.session_state:
    st.session_state.delete_confirm_id = None


# -----------------------------
# Restore session from URL
# -----------------------------
query_session_id = st.query_params.get("session_id")

if query_session_id and st.session_state.active_session_id != query_session_id:
    st.session_state.active_session_id = query_session_id
    st.session_state.messages = fetch_history(query_session_id)


# -----------------------------
# Header
# -----------------------------
st.title("💼 Chapter1 ERP Assistant")
st.caption(
    "Ask about sales, purchases, inventory, customers, vendors, outstanding invoices, GST, TDS, TCS, and more."
)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Sessions")

    sessions = fetch_sessions()

    if st.button("+ New Session", use_container_width=True, type="primary"):
        session = create_session()
        if session:
            st.session_state.active_session_id = session["id"]
            st.session_state.messages = []
            st.query_params["session_id"] = session["id"]
            st.rerun()

    st.divider()

    if not sessions:
        st.caption("No sessions yet.")
    else:
        for s in sessions:
            sid = s["id"]
            is_active = sid == st.session_state.active_session_id
            label = s.get("name") or sid[:20]
            msg_count = s.get("message_count", 0)

            row = st.columns([0.72, 0.14, 0.14])

            with row[0]:
                if is_active:
                    st.markdown(f"**● {label}**  \n`{msg_count} msgs`")
                else:
                    if st.button(
                        f"{label}  \n`{msg_count} msgs`",
                        key=f"switch_{sid}",
                        use_container_width=True,
                    ):
                        st.session_state.active_session_id = sid
                        st.session_state.messages = fetch_history(sid)
                        st.query_params["session_id"] = sid
                        st.rerun()

            with row[1]:
                if st.button("✎", key=f"rename_{sid}", help="Rename"):
                    st.session_state.editing_session_id = sid
                    st.rerun()

            with row[2]:
                if st.button("×", key=f"delete_{sid}", help="Delete"):
                    st.session_state.delete_confirm_id = sid
                    st.rerun()

            if st.session_state.editing_session_id == sid:
                new_name = st.text_input(
                    "Session name",
                    value=s.get("name", ""),
                    key=f"name_input_{sid}",
                    label_visibility="collapsed",
                )

                rename_cols = st.columns(2)

                with rename_cols[0]:
                    if st.button("Save", key=f"save_name_{sid}"):
                        if rename_session(sid, new_name):
                            st.session_state.editing_session_id = None
                            st.rerun()

                with rename_cols[1]:
                    if st.button("Cancel", key=f"cancel_name_{sid}"):
                        st.session_state.editing_session_id = None
                        st.rerun()

            if st.session_state.delete_confirm_id == sid:
                st.warning("Delete this session?")
                delete_cols = st.columns(2)

                with delete_cols[0]:
                    if st.button("Yes", key=f"confirm_delete_{sid}"):
                        if delete_session(sid):
                            if sid == st.session_state.active_session_id:
                                st.session_state.active_session_id = None
                                st.session_state.messages = []
                                st.query_params.clear()

                            st.session_state.delete_confirm_id = None
                            st.rerun()

                with delete_cols[1]:
                    if st.button("No", key=f"cancel_delete_{sid}"):
                        st.session_state.delete_confirm_id = None
                        st.rerun()

    st.divider()

    st.subheader("Quick actions")

    for suggestion in SUGGESTED_PROMPTS:
        if st.button(
            suggestion,
            use_container_width=True,
            key=f"suggestion_{suggestion}",
        ):
            st.session_state.pending_prompt = suggestion
            st.rerun()

    st.divider()
    st.caption("Chapter1 ERP Assistant v1.0")


# -----------------------------
# No active session
# -----------------------------
if not st.session_state.active_session_id:
    st.info("Create or select a session to get started.")
    st.stop()


session_id = st.session_state.active_session_id


# -----------------------------
# Chat history
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg.get("role", "assistant")):
        st.markdown(msg.get("content", ""))
        render_records(msg.get("data"))


# -----------------------------
# Prompt handling
# -----------------------------
prompt = st.session_state.pending_prompt or st.chat_input(
    "Ask about customers, stock, GST, invoices..."
)

if st.session_state.pending_prompt:
    st.session_state.pending_prompt = None


# -----------------------------
# Chat execution
# -----------------------------
if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_ph = st.empty()
        response_ph = st.empty()

        status_ph.markdown(THINKING_HTML, unsafe_allow_html=True)

        full_response = ""
        tokens_started = False

        for token in chat_stream(session_id, prompt):
            if not tokens_started:
                tokens_started = True
                status_ph.empty()

            full_response += token

            if full_response.strip():
                response_ph.markdown(full_response + "▌")

        status_ph.empty()
        response_ph.markdown(full_response or "_(no response)_")

        data_dict = st.session_state.pop("_stream_data", {})
        actual_session_id = st.session_state.pop("_stream_session_id", session_id)

        render_records(data_dict)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response or "_(no response)_",
            "data": data_dict,
        }
    )

    if actual_session_id != session_id:
        st.session_state.active_session_id = actual_session_id
        st.query_params["session_id"] = actual_session_id

    st.rerun()





# import streamlit as st
# import requests
# import pandas as pd
# import json
# from dotenv import load_dotenv
# import os
# load_dotenv()

# API_BASE = os.getenv("BACKEND_URL")

# st.set_page_config(page_title="Chapter1 ERP Assistant", page_icon="💼", layout="wide")


# def fetch_sessions():
#     try:
#         resp = requests.get(f"{API_BASE}/sessions", timeout=5)
#         return resp.json().get("sessions", [])
#     except Exception as e:
#         st.error(f"Could not load sessions: {e}")
#         return []


# def create_session(name=""):
#     try:
#         resp = requests.post(f"{API_BASE}/sessions", json={"name": name}, timeout=5)
#         return resp.json().get("session")
#     except Exception as e:
#         st.error(f"Could not create session: {e}")
#         return None


# def delete_session(session_id):
#     try:
#         requests.delete(f"{API_BASE}/sessions/{session_id}", timeout=5)
#     except Exception as e:
#         st.error(f"Could not delete session: {e}")


# def rename_session(session_id, name):
#     try:
#         requests.patch(f"{API_BASE}/sessions/{session_id}", json={"name": name}, timeout=5)
#     except Exception as e:
#         st.error(f"Could not rename session: {e}")


# def fetch_history(session_id):
#     try:
#         resp = requests.get(f"{API_BASE}/sessions/{session_id}/history", timeout=5)
#         return resp.json().get("messages", [])
#     except Exception as e:
#         st.error(f"Could not load history: {e}")
#         return []


# def chat_stream(session_id, query):
#     resp = requests.post(
#         f"{API_BASE}/chat/stream",
#         json={"query": query, "session_id": session_id},
#         stream=True, timeout=600,
#     )
#     collected_data = {}
#     actual_sid = session_id
#     for line in resp.iter_lines(decode_unicode=True):
#         if line.startswith("data: "):
#             payload = json.loads(line[6:])
#             if "token" in payload:
#                 yield payload["token"]
#             elif "data" in payload:
#                 collected_data.update(payload["data"])
#             elif "session_id" in payload:
#                 actual_sid = payload["session_id"]
#             elif payload.get("done"):
#                 break
#     st.session_state._stream_data = collected_data
#     st.session_state._stream_session_id = actual_sid


# def render_records(data_dict):
#     pass


# # ── Initialise session state ──
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "active_session_id" not in st.session_state:
#     st.session_state.active_session_id = None
# if "session_name" not in st.session_state:
#     st.session_state.session_name = ""
# if "editing_session_id" not in st.session_state:
#     st.session_state.editing_session_id = None

# # Restore active session from query params on page load
# query_session_id = st.query_params.get("session_id")
# if query_session_id and st.session_state.active_session_id != query_session_id:
#     st.session_state.active_session_id = query_session_id
#     st.session_state.messages = fetch_history(query_session_id)

# # ── Sidebar ──
# with st.sidebar:
#     st.title("Sessions")

#     sessions = fetch_sessions()

#     # New session button
#     if st.button("+ New Session", use_container_width=True, type="primary"):
#         session = create_session()
#         if session:
#             st.session_state.active_session_id = session["id"]
#             st.session_state.messages = []
#             st.session_state.session_name = ""
#             st.query_params["session_id"] = session["id"]
#             st.rerun()

#     st.divider()

#     # Session list
#     if not sessions:
#         st.caption("No sessions yet.")
#     else:
#         for s in sessions:
#             cols = st.columns([0.8, 0.2])
#             is_active = s["id"] == st.session_state.active_session_id
#             label = s.get("name") or s["id"][:20]
#             msg_count = s.get("message_count", 0)

#             with cols[0]:
#                 if is_active:
#                     st.markdown(f"**● {label}**  \n`{msg_count} msgs`")
#                 else:
#                     if st.button(
#                         f"{label}  \n`{msg_count} msgs`",
#                         key=f"switch_{s['id']}",
#                         use_container_width=True,
#                     ):
#                         st.session_state.active_session_id = s["id"]
#                         st.session_state.messages = fetch_history(s["id"])
#                         st.session_state.session_name = s.get("name", "")
#                         st.query_params["session_id"] = s["id"]
#                         st.rerun()

#             with cols[1]:
#                 # Rename button
#                 if st.button("✎", key=f"rename_{s['id']}", help="Rename"):
#                     st.session_state.editing_session_id = s["id"]
#                     st.rerun()

#             # Inline rename input
#             if st.session_state.editing_session_id == s["id"]:
#                 new_name = st.text_input(
#                     "Name",
#                     value=s.get("name", ""),
#                     key=f"name_input_{s['id']}",
#                     label_visibility="collapsed",
#                 )
#                 rename_cols = st.columns(2)
#                 with rename_cols[0]:
#                     if st.button("Save", key=f"save_name_{s['id']}"):
#                         rename_session(s["id"], new_name)
#                         st.session_state.editing_session_id = None
#                         st.rerun()
#                 with rename_cols[1]:
#                     if st.button("Cancel", key=f"cancel_name_{s['id']}"):
#                         st.session_state.editing_session_id = None
#                         st.rerun()

#             # Delete button (only if active or via expander)
#             with cols[1]:
#                 if st.button("×", key=f"delete_{s['id']}", help="Delete"):
#                     delete_session(s["id"])
#                     if s["id"] == st.session_state.active_session_id:
#                         st.session_state.active_session_id = None
#                         st.session_state.messages = []
#                         st.query_params.clear()
#                     st.rerun()

#     st.divider()
#     st.caption("Chapter1 ERP Assistant v1.0")

# # ── Chat area ──
# if not st.session_state.active_session_id:
#     st.info("Select a session or create a new one to get started.")
#     st.stop()

# session_id = st.session_state.active_session_id

# # Display messages
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
#         render_records(msg.get("data"))

# # Chat input
# if prompt := st.chat_input("Ask about customers, stock, GST..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             text = st.write_stream(chat_stream(session_id, prompt))
#         data_dict = st.session_state.pop("_stream_data", {})
#         actual_session_id = st.session_state.pop("_stream_session_id", session_id)
#         render_records(data_dict)

#     st.session_state.messages.append({"role": "assistant", "content": text, "data": data_dict})

#     # If the server returned a different session_id (auto-created), update it
#     if actual_session_id != session_id:
#         st.session_state.active_session_id = actual_session_id
#         st.query_params["session_id"] = actual_session_id

#     st.rerun()
