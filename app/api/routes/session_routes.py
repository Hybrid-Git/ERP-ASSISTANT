from fastapi import APIRouter, Depends, HTTPException, Request
import hmac

import session_store
from app.core.config import APP_API_KEY
from app.schemas.api import CreateSessionRequest, RenameSessionRequest

router = APIRouter()


def bearer_token_valid(authorization_header: str) -> bool:
    if not APP_API_KEY:
        return True

    parts = (authorization_header or "").strip().split(None, 1)

    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False

    return hmac.compare_digest(parts[1].strip(), APP_API_KEY)


def require_api_key(request: Request):
    if not bearer_token_valid(request.headers.get("authorization", "")):
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.get("/sessions", dependencies=[Depends(require_api_key)])
async def api_list_sessions():
    sessions = session_store.list_sessions()
    return {"sessions": sessions}


@router.post("/sessions", dependencies=[Depends(require_api_key)])
async def api_create_session(body: CreateSessionRequest):
    session = session_store.create_session(name=body.name)
    return {"session": session}


@router.delete("/sessions/{session_id}", dependencies=[Depends(require_api_key)])
async def api_delete_session(session_id: str):
    deleted = session_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}


@router.patch("/sessions/{session_id}", dependencies=[Depends(require_api_key)])
async def api_rename_session(session_id: str, body: RenameSessionRequest):
    updated = session_store.rename_session(session_id, body.name)
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"updated": True}


@router.get("/sessions/{session_id}/history", dependencies=[Depends(require_api_key)])
async def api_session_history(session_id: str):
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = session_store.load_messages(session_id)
    history = []

    for msg in messages:
        if msg.type == "tool":
            continue
        if msg.type == "ai" and not msg.content and getattr(msg, "tool_calls", None):
            continue

        role = msg.type
        content = msg.content

        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )

        history.append({"role": role, "content": content})

    return {"session_id": session_id, "messages": history}