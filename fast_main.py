from dotenv import load_dotenv
load_dotenv(override=True)
import hmac
from fastapi import Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from src.config import APP_API_KEY, CORS_ORIGINS
import os
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse
from langchain_core.messages import  SystemMessage, HumanMessage, AIMessage
from src.config import llm
import session_store
from src.settings_validator import validate_settings
from src.logging_config import setup_logging
from fastapi.responses import JSONResponse
from src.erp_client import erp_client
from src.response_utils import make_error_response
from src.exceptions import ERPAssistantError
from src.formatters import format_response_as_chat_text
from src.session_helper import _prepare_session, _save_chat_result
from src.langsmit_utils import _build_langsmith_config
from src.graph_runner import run_graph_query,graph
from src.streaming import generate_stream_events
from src.schema import ChatRequest, CreateSessionRequest, RenameSessionRequest
logger = setup_logging()
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Validating configuration")
    validate_settings()
    logger.info("Configuration validation completed")
    logger.info("FastAPI started. Warming up worker LLM...")
    start = time.perf_counter()
    try:
        await llm.ainvoke([SystemMessage(content="Return only: OK\n/no_think"),
                           HumanMessage(content="ping")])
        elapsed_time = time.perf_counter() - start
        logger.info(
                    "Worker LLM warmup completed",
                    extra={"duration_sec": round(elapsed_time, 3)}
                    )
    except Exception as e:
        logger.exception("LLM warmup failed; will load on first query")
    yield
    await erp_client.close()
    logger.info("ERP client connection pool closed")
def get_cors_origins() -> list[str]:
    return [
        origin.strip()
        for origin in CORS_ORIGINS.split(",")
        if origin.strip()
    ]

app = FastAPI(
    title="CHAPTER-1-ASSIST",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)
#Exception handlers
@app.exception_handler(ERPAssistantError)
async def erp_error_handler(request, exc: ERPAssistantError):
    logger.error(
        "ERP assistant error",
        extra={"error_code": exc.error_code},
    )

    return JSONResponse(
        status_code=400,
        content=make_error_response(
            status=exc.error_code,
            summary=exc.user_message,
            errors=[exc.user_message],
        ),
    )


@app.exception_handler(Exception)
async def generic_error_handler(request, exc: Exception):
    logger.exception("Unhandled server error")

    return JSONResponse(
        status_code=500,
        content=make_error_response(
            status="internal_error",
            summary="An unexpected error occurred. Please try again later.",
            errors=["internal_error"],
        ),
    )



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


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "chapter1-erp-assistant",
    }

# existing routes below
@app.get("/")
async def root():
    return {"message": "ERP Assistant API is running"}





def get_output_format() -> str:
    return os.getenv("OUTPUT_FORMAT", "text")













# @app.get("/")
# async def root():
#     return {"message": "ERP Assistant API is running"}


@app.post("/chat")
async def chat(request: ChatRequest, fmt: Optional[str] = Query(None, alias="format")):
    request_id = str(uuid.uuid4())
    session_id = request.session_id or "default_session"
    _, past_messages, past_summary, past_context, past_last_tool = await _prepare_session(session_id)
    langsmith_config = _build_langsmith_config("CHAPTER1_ASSIST_CHAT", request_id, request.query, session_id)

    try:
        result = await run_graph_query(
            user_query=request.query,
            past_messages=past_messages,
            past_summary=past_summary,
            past_conversation_context=past_context,
            past_last_tool_call=past_last_tool,
            langsmith_config=langsmith_config,
        )

        _save_chat_result(session_id, result)
        result["session_id"] = session_id
        result.pop("updated_messages", None)

        output_format = fmt or get_output_format()

        if output_format == "text":
            text = result.get("response_text") or result.get("response")
            if not isinstance(text, str):
                text = await format_response_as_chat_text(text)
            return PlainTextResponse(text)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=make_error_response(
                user_query=request.query,
                status="server_error",
                summary="Server error while processing the query.",
                errors=[str(e)],
            ),
        )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    request_id = str(uuid.uuid4())
    session_id = request.session_id or "default_session"
    _, past_messages, past_summary, past_context, past_last_tool = await _prepare_session(session_id)
    langsmith_config = _build_langsmith_config(
        "CHAPTER1_ASSIST_CHAT_STREAM", request_id, request.query, session_id,
        tags=["fastapi", "langgraph", "erp-assistant", "stream"],
    )

    initial_state = {
        "user_query": request.query,
        "messages": past_messages or [],
        "retrieved_tools": [],
        "selected_tools": [],
        "query_parts": [],
        "router_decision": {},
        "skip_router": False,
        "loop_count": 0,
        "final_response": "",
        "tools_utilized": [],
        "step_timings": [],
        "document_type": "",
        "unsupported_parts": [],
        "summary": past_summary or "",
        "conversation_context": past_context or {},
        "last_tool_call": past_last_tool or {},
    }

    config = {
        **langsmith_config,
        "configurable": {"thread_id": session_id},
    }
    return StreamingResponse(
    generate_stream_events(
        graph=graph,
        initial_state=initial_state,
        config=config,
        session_id=session_id,
        past_messages=past_messages,
        past_summary=past_summary,
        past_context=past_context,
        past_last_tool=past_last_tool,
    ),
    media_type="text/event-stream",
)
    

@app.post("/chat-text")
async def chat_text(request: ChatRequest):
    request_id = str(uuid.uuid4())
    session_id = request.session_id or "default_session"
    _, past_messages, past_summary, past_context, past_last_tool = await _prepare_session(session_id)
    langsmith_config = _build_langsmith_config(
        "CHAPTER1_ASSIST_CHAT_TEXT", request_id, request.query, session_id,
        tags=["fastapi", "langgraph", "erp-assistant", "text-response"],
    )

    try:
        result = await run_graph_query(
            user_query=request.query,
            past_messages=past_messages,
            past_summary=past_summary,
            past_conversation_context=past_context,
            past_last_tool_call=past_last_tool,
            langsmith_config=langsmith_config,
        )

        _save_chat_result(session_id, result)
        result.pop("updated_messages", None)
        text = result.get("response_text") or result.get("response")
        if not isinstance(text, str):
            text = await format_response_as_chat_text(text)
        return PlainTextResponse(text)

    except Exception as e:
        error_response = make_error_response(
            user_query=request.query,
            status="server_error",
            summary="Server error while processing the query.",
            errors=[str(e)],
        )
        return PlainTextResponse(
            await format_response_as_chat_text(error_response),
            status_code=500,
        )


# ==============================
# Session Management Endpoints
# ==============================


@app.get("/sessions", dependencies=[Depends(require_api_key)])
async def api_list_sessions():
    sessions = session_store.list_sessions()
    return {"sessions": sessions}


@app.post("/sessions", dependencies=[Depends(require_api_key)])
async def api_create_session(body: CreateSessionRequest):
    session = session_store.create_session(name=body.name)
    return {"session": session}


@app.delete("/sessions/{session_id}", dependencies=[Depends(require_api_key)])
async def api_delete_session(session_id: str):
    deleted = session_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}


@app.patch("/sessions/{session_id}", dependencies=[Depends(require_api_key)])
async def api_rename_session(session_id: str, body: RenameSessionRequest):
    updated = session_store.rename_session(session_id, body.name)
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"updated": True}


@app.get("/sessions/{session_id}/history", dependencies=[Depends(require_api_key)])
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


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)