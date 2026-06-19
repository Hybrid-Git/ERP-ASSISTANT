from langchain_core.messages import AIMessage

import session_store


# ── Shared chat helpers ──

async def _prepare_session(session_id: str):
    session = session_store.get_or_create_session(session_id)
    past_messages = session_store.load_messages(session_id)
    past_summary = (session or {}).get("summary", "")
    past_context, past_last_tool = session_store.load_session_context(session_id)[1:]
    return session, past_messages, past_summary, past_context, past_last_tool


def _save_chat_result(session_id: str, result: dict):
    updated_messages = list(result.get("updated_messages", []))
    response_text = result.get("response_text")
    if response_text:
        updated_messages.append(AIMessage(content=response_text))
    session_store.save_session(
        session_id,
        updated_messages,
        result.get("summary", ""),
        result.get("conversation_context"),
        result.get("last_tool_call"),
    )
    return result