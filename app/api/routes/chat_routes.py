import os
import uuid

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse

from src.formatters import format_response_as_chat_text
from src.graph_runner import graph, run_graph_query
from src.langsmit_utils import _build_langsmith_config
from src.response_utils import make_error_response
from src.schema import ChatRequest
from src.session_helper import _prepare_session, _save_chat_result
from src.streaming import generate_stream_events

router = APIRouter()


def get_output_format() -> str:
    return os.getenv("OUTPUT_FORMAT", "text")


@router.post("/chat")
async def chat(request: ChatRequest, fmt: str | None = Query(None, alias="format")):
    request_id = str(uuid.uuid4())
    session_id = request.session_id or "default_session"

    _, past_messages, past_summary, past_context, past_last_tool = await _prepare_session(session_id)

    langsmith_config = _build_langsmith_config(
        "CHAPTER1_ASSIST_CHAT",
        request_id,
        request.query,
        session_id,
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
                query=request.query,
                status="server_error",
                summary="Server error while processing the query.",
                errors=[str(e)],
            ),
        )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    request_id = str(uuid.uuid4())
    session_id = request.session_id or "default_session"

    _, past_messages, past_summary, past_context, past_last_tool = await _prepare_session(session_id)

    langsmith_config = _build_langsmith_config(
        "CHAPTER1_ASSIST_CHAT_STREAM",
        request_id,
        request.query,
        session_id,
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


@router.post("/chat-text")
async def chat_text(request: ChatRequest):
    request_id = str(uuid.uuid4())
    session_id = request.session_id or "default_session"

    _, past_messages, past_summary, past_context, past_last_tool = await _prepare_session(session_id)

    langsmith_config = _build_langsmith_config(
        "CHAPTER1_ASSIST_CHAT_TEXT",
        request_id,
        request.query,
        session_id,
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
            query=request.query,
            status="server_error",
            summary="Server error while processing the query.",
            errors=[str(e)],
        )

        return PlainTextResponse(
            await format_response_as_chat_text(error_response),
            status_code=500,
        )