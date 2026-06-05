import streamlit as st
import requests
import pandas as pd

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Chapter1 ERP Assistant", page_icon="💼", layout="wide")


def fetch_sessions():
    try:
        resp = requests.get(f"{API_BASE}/sessions", timeout=5)
        return resp.json().get("sessions", [])
    except Exception as e:
        st.error(f"Could not load sessions: {e}")
        return []


def create_session(name=""):
    try:
        resp = requests.post(f"{API_BASE}/sessions", json={"name": name}, timeout=5)
        return resp.json().get("session")
    except Exception as e:
        st.error(f"Could not create session: {e}")
        return None


def delete_session(session_id):
    try:
        requests.delete(f"{API_BASE}/sessions/{session_id}", timeout=5)
    except Exception as e:
        st.error(f"Could not delete session: {e}")


def rename_session(session_id, name):
    try:
        requests.patch(f"{API_BASE}/sessions/{session_id}", json={"name": name}, timeout=5)
    except Exception as e:
        st.error(f"Could not rename session: {e}")


def fetch_history(session_id):
    try:
        resp = requests.get(f"{API_BASE}/sessions/{session_id}/history", timeout=5)
        return resp.json().get("messages", [])
    except Exception as e:
        st.error(f"Could not load history: {e}")
        return []


def send_query(session_id, query):
    try:
        resp = requests.post(
            f"{API_BASE}/chat?format=json",
            json={"query": query, "session_id": session_id},
            timeout=300,
        )
        data = resp.json()
        text = data.get("response_text") or (
            data.get("response", {}).get("summary", str(data))
        )
        data_dict = data.get("data", {})
        return text, data_dict, data.get("session_id", session_id)
    except Exception as e:
        return f"Error: {e}", {}, session_id


def render_records(data_dict):
    if not data_dict or not isinstance(data_dict, dict):
        return
    for tool_name, records in data_dict.items():
        if not records or not isinstance(records, list):
            continue
        clean = [r for r in records if isinstance(r, dict) and not r.get("__note")]
        if not clean:
            continue
        df = pd.DataFrame(clean)
        n = len(clean)
        with st.expander(f"📊 {tool_name}: {n} records", expanded=False):
            st.dataframe(df, height=300 if n > 10 else 35*n+40, use_container_width=True)


# ── Initialise session state ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "session_name" not in st.session_state:
    st.session_state.session_name = ""
if "editing_session_id" not in st.session_state:
    st.session_state.editing_session_id = None

# Restore active session from query params on page load
query_session_id = st.query_params.get("session_id")
if query_session_id and st.session_state.active_session_id != query_session_id:
    st.session_state.active_session_id = query_session_id
    st.session_state.messages = fetch_history(query_session_id)

# ── Sidebar ──
with st.sidebar:
    st.title("Sessions")

    sessions = fetch_sessions()

    # New session button
    if st.button("+ New Session", use_container_width=True, type="primary"):
        session = create_session()
        if session:
            st.session_state.active_session_id = session["id"]
            st.session_state.messages = []
            st.session_state.session_name = ""
            st.query_params["session_id"] = session["id"]
            st.rerun()

    st.divider()

    # Session list
    if not sessions:
        st.caption("No sessions yet.")
    else:
        for s in sessions:
            cols = st.columns([0.8, 0.2])
            is_active = s["id"] == st.session_state.active_session_id
            label = s.get("name") or s["id"][:20]
            msg_count = s.get("message_count", 0)

            with cols[0]:
                if is_active:
                    st.markdown(f"**● {label}**  \n`{msg_count} msgs`")
                else:
                    if st.button(
                        f"{label}  \n`{msg_count} msgs`",
                        key=f"switch_{s['id']}",
                        use_container_width=True,
                    ):
                        st.session_state.active_session_id = s["id"]
                        st.session_state.messages = fetch_history(s["id"])
                        st.session_state.session_name = s.get("name", "")
                        st.query_params["session_id"] = s["id"]
                        st.rerun()

            with cols[1]:
                # Rename button
                if st.button("✎", key=f"rename_{s['id']}", help="Rename"):
                    st.session_state.editing_session_id = s["id"]
                    st.rerun()

            # Inline rename input
            if st.session_state.editing_session_id == s["id"]:
                new_name = st.text_input(
                    "Name",
                    value=s.get("name", ""),
                    key=f"name_input_{s['id']}",
                    label_visibility="collapsed",
                )
                rename_cols = st.columns(2)
                with rename_cols[0]:
                    if st.button("Save", key=f"save_name_{s['id']}"):
                        rename_session(s["id"], new_name)
                        st.session_state.editing_session_id = None
                        st.rerun()
                with rename_cols[1]:
                    if st.button("Cancel", key=f"cancel_name_{s['id']}"):
                        st.session_state.editing_session_id = None
                        st.rerun()

            # Delete button (only if active or via expander)
            with cols[1]:
                if st.button("×", key=f"delete_{s['id']}", help="Delete"):
                    delete_session(s["id"])
                    if s["id"] == st.session_state.active_session_id:
                        st.session_state.active_session_id = None
                        st.session_state.messages = []
                        st.query_params.clear()
                    st.rerun()

    st.divider()
    st.caption("Chapter1 ERP Assistant v1.0")

# ── Chat area ──
if not st.session_state.active_session_id:
    st.info("Select a session or create a new one to get started.")
    st.stop()

session_id = st.session_state.active_session_id

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_records(msg.get("data"))

# Chat input
if prompt := st.chat_input("Ask about customers, stock, GST..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            text, data_dict, actual_session_id = send_query(session_id, prompt)

        st.markdown(text)
        render_records(data_dict)
        st.session_state.messages.append({"role": "assistant", "content": text, "data": data_dict})

        # If the server returned a different session_id (auto-created), update it
        if actual_session_id != session_id:
            st.session_state.active_session_id = actual_session_id
            st.query_params["session_id"] = actual_session_id

        st.rerun()
