import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000/chat?format=json"

st.set_page_config(page_title="Chapter1 ERP Assistant", page_icon="💼")

if "session_id" not in st.session_state:
    st.session_state.session_id = "sess_" + uuid.uuid4().hex[:12]
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Session")
    st.text_input("Session ID", value=st.session_state.session_id, disabled=True)
    if st.button("🔄 New Session"):
        st.session_state.session_id = "sess_" + uuid.uuid4().hex[:12]
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Chapter1 ERP Assistant v1.0")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about customers, stock, GST..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    API_URL,
                    json={"query": prompt, "session_id": st.session_state.session_id},
                    timeout=300,
                )
                data = resp.json()
                text = data.get("response_text") or (
                    data.get("response", {}).get("summary", str(data))
                )
                st.markdown(text)
                st.session_state.messages.append({"role": "assistant", "content": text})
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
