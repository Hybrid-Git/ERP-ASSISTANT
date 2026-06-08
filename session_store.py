# import json
# import os
import uuid
import threading
from datetime import datetime, timezone
from typing import Optional
from langchain_core.messages import message_to_dict, messages_from_dict

# _sessions_file = "sessions.json"
_sessions: dict[str, dict] = {}
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# def _save():
#     with _lock:
#         with open(_sessions_file, "w") as f:
#             json.dump(_sessions, f, default=str)


# def _load():
#     global _sessions
#     if os.path.exists(_sessions_file):
#         with open(_sessions_file) as f:
#             _sessions = json.load(f)


# def init_db():
#     _load()


def create_session(name: str = "", session_id: str = "") -> dict:
    sid = session_id or ("sess_" + uuid.uuid4().hex[:12])
    now = _now()
    data = {
        "id": sid,
        "name": name,
        "summary": "",
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }
    with _lock:
        _sessions[sid] = data
    # _save()
    return dict(data)


def get_or_create_session(session_id: str) -> dict:
    """Atomically get an existing session or create a new one."""
    with _lock:
        data = _sessions.get(session_id)
        if data is not None:
            return dict(data)
        now = _now()
        _sessions[session_id] = {
            "id": session_id,
            "name": "",
            "summary": "",
            "messages": [],
            "created_at": now,
            "updated_at": now,
        }
        result = dict(_sessions[session_id])
    # _save()
    return result


def list_sessions() -> list[dict]:
    with _lock:
        items = sorted(
            _sessions.values(),
            key=lambda x: x.get("updated_at", ""),
            reverse=True,
        )
    return [
        {
            "id": s["id"],
            "name": s.get("name", ""),
            "summary": s.get("summary", ""),
            "created_at": s.get("created_at", ""),
            "updated_at": s.get("updated_at", ""),
            "message_count": len(s.get("messages", [])),
        }
        for s in items
    ]


def get_session(session_id: str) -> Optional[dict]:
    with _lock:
        data = _sessions.get(session_id)
        if data is None:
            return None
        return dict(data)


def save_session(session_id: str, messages: list, summary: str = ""):
    serialized = [message_to_dict(m) for m in messages]
    now = _now()
    with _lock:
        if session_id in _sessions:
            _sessions[session_id]["messages"] = serialized
            _sessions[session_id]["summary"] = summary
            _sessions[session_id]["updated_at"] = now
        else:
            _sessions[session_id] = {
                "id": session_id,
                "name": "",
                "summary": summary,
                "messages": serialized,
                "created_at": now,
                "updated_at": now,
            }
    # _save()


def delete_session(session_id: str) -> bool:
    with _lock:
        if session_id in _sessions:
            del _sessions[session_id]
            deleted = True
        else:
            deleted = False
    # if deleted:
    #     _save()
    return deleted


def rename_session(session_id: str, name: str) -> bool:
    now = _now()
    with _lock:
        data = _sessions.get(session_id)
        if data is None:
            return False
        data["name"] = name
        data["updated_at"] = now
    # _save()
    
    return True


def load_messages(session_id: str) -> list:
    with _lock:
        data = _sessions.get(session_id)
        if data is None:
            return []
        dicts = data.get("messages", [])
    if not dicts:
        return []
    return messages_from_dict(dicts) if dicts else []
