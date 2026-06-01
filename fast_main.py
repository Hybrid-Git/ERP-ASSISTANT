from dotenv import load_dotenv
load_dotenv()

import os
import json
import re
import time
import asyncio
import copy
import uuid
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from src.graph import graph_builder
from src.config import llm, normalizer_llm


app = FastAPI(
    title="CHAPTER-1-ASSIST",
    version="1.0.0",
)

graph = graph_builder()
GRAPH_TIMEOUT_SECONDS = 300

# ==============================
# FINAL RESPONSE CACHE & SESSION STORAGE
# ==============================
FINAL_RESPONSE_CACHE = {}
FINAL_RESPONSE_CACHE_TTL_SECONDS = 300  # 5 minutes

# Server-side session memory.
# This now stores only messages. No rolling summary is stored.
SESSION_MEMORY = {}


def normalize_query_for_cache(query: str) -> str:
    return " ".join((query or "").lower().strip().split())


def should_cache_final_response(result: dict) -> bool:
    if not isinstance(result, dict):
        return False

    response = result.get("response")
    if not isinstance(response, dict):
        return False

    success = response.get("success")
    status = response.get("status")
    tools_used = response.get("tools_used", [])

    if not tools_used:
        return False

    return success is True and status == "success"


def get_cached_final_response(query: str):
    key = normalize_query_for_cache(query)
    cached = FINAL_RESPONSE_CACHE.get(key)

    if not cached:
        print(f"[FINAL CACHE MISS] {key}")
        return None

    age = time.time() - cached.get("cached_at", 0)
    if age > FINAL_RESPONSE_CACHE_TTL_SECONDS:
        print(f"[FINAL CACHE EXPIRED] {key}")
        FINAL_RESPONSE_CACHE.pop(key, None)
        return None

    result = cached.get("result")
    if not isinstance(result, dict) or "response" not in result:
        print(f"[FINAL CACHE INVALID] {key}")
        FINAL_RESPONSE_CACHE.pop(key, None)
        return None

    print(f"[FINAL CACHE HIT] {key}")

    # Use deepcopy instead of JSON serialization because LangChain objects may not serialize cleanly.
    result = copy.deepcopy(result)
    result["timings"] = [{"node": "final_response_cache", "duration_sec": 0.001}]
    result["total_time_sec"] = 0.001

    return result


def set_cached_final_response(query: str, result: dict):
    if not should_cache_final_response(result):
        return

    key = normalize_query_for_cache(query)

    # Cache only API output payload, never session-specific LangChain messages.
    cacheable_result = {
        "response": result.get("response"),
        "timings": result.get("timings", []),
        "total_time_sec": result.get("total_time_sec", 0.0),
    }

    FINAL_RESPONSE_CACHE[key] = {
        "cached_at": time.time(),
        "result": copy.deepcopy(cacheable_result),
    }
    print(f"[FINAL CACHE SET] {key}")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = "default_session"


def make_error_response(
    user_query: str,
    status: str,
    summary: str,
    errors: List[str],
    tools_used: List[str] | None = None,
    data: Dict[str, Any] | None = None,
):
    return {
        "success": False,
        "status": status,
        "query": user_query,
        "tools_used": tools_used or [],
        "data": data or {},
        "summary": summary,
        "errors": errors,
    }


DEFAULT_OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "text")


def pretty_field_name(key: str) -> str:
    replacements = {
        "id": "ID",
        "name": "Name",
        "hsnCode": "HSN Code",
        "closingQty": "Closing Quantity",
        "closingRate": "Closing Rate",
        "closingValue": "Closing Value",
        "openingBalance": "Opening Balance",
        "openingType": "Opening Type",
        "voucherCount": "Voucher Count",
        "taxableAmount": "Taxable Amount",
        "invoiceAmount": "Invoice Amount",
        "igst": "IGST",
        "cgst": "CGST",
        "sgst": "SGST",
        "cess": "CESS",
        "tax": "Total Tax",
        "category": "Category",
    }

    if key in replacements:
        return replacements[key]

    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", key)
    return spaced.replace("_", " ").title()


async def format_response_as_chat_text(
    response_data: dict,
    timings: list = None,
    total_time: float = None,
    **kwargs,
) -> str:
    """
    Converts deterministic JSON into a clean conversational sentence.
    This is response formatting only. It is not conversation summarization.
    """
    status = response_data.get("status", "")
    summary = response_data.get("summary", "")
    query = response_data.get("query", "")
    data = response_data.get("data", {})

    if status == "needs_clarification":
        return f"ℹ️ {summary if summary else 'Could you please clarify your request with a specific name or ID?'}"

    if status == "no_matching_records":
        return "I checked your ERP records but couldn't find any matching data for that description."

    if not data:
        return "I encountered an issue retrieving those records right now."

    # Flatten data: remove tool name wrapper, just pass the records
    flat_records = []
    for tool_name, records in data.items():
        if isinstance(records, list):
            flat_records.extend(records)
        elif isinstance(records, dict):
            flat_records.append(records)

    payload_to_process = {
        "user_query": query,
        "data": flat_records,
    }

    system_instruction = (
        "You are a helpful ERP assistant. Answer the user's question directly using the provided data. "
        "Reply in a natural, conversational tone like ChatGPT or Gemini. "
        "Do NOT mention tool names, field names, JSON structure, or technical details. "
        "Do NOT say 'retrieved_data' or 'get_customer' or 'get_stock_levels'. "
        "Just give the answer the user wants — like a knowledgeable assistant would. "
        "Use plain text only. Keep it concise."
    )

    try:
        response = await normalizer_llm.ainvoke([
            SystemMessage(content=system_instruction),
            HumanMessage(content=json.dumps(payload_to_process, ensure_ascii=False)),
        ])

        return str(response.content).strip()

    except Exception:
        return f"Here is the data found: {json.dumps(data, ensure_ascii=False)}"


async def run_graph_query(
    user_query: str,
    past_messages: list = None,
    langsmith_config: dict | None = None,
):
    cached_result = get_cached_final_response(user_query)
    if cached_result is not None:
        # Keep session memory unchanged on cache hits.
        cached_result["updated_messages"] = past_messages or []
        return cached_result

    start_time = time.perf_counter()

    initial_state = {
        "user_query": user_query,
        "canonical_query": "",
        "translator_used": False,
        "translator_confidence": "",
        "detected_language": "",
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
    }

    final_response = None
    timings = []
    tools_requested = []
    messages_tracker = list(initial_state["messages"])

    try:
        async with asyncio.timeout(GRAPH_TIMEOUT_SECONDS):
            async for chunks in graph.astream(
                initial_state,
                config=langsmith_config or {},
                stream_mode="updates",
            ):
                for node_name, state_update in chunks.items():
                    print(f"Finished running: {node_name}")

                    if "step_timings" in state_update:
                        timings.extend(state_update["step_timings"])
                        for timing in state_update["step_timings"]:
                            print(f"[STEP TIME] {timing['node']} = {timing['duration_sec']}s")

                    # Save all returned messages. No summarization or trimming is applied.
                    if "messages" in state_update:
                        for msg in state_update["messages"]:
                            if msg not in messages_tracker:
                                messages_tracker.append(msg)

                    if node_name == "chat_model":
                        messages = state_update.get("messages", [])
                        if not messages:
                            final_response = make_error_response(
                                user_query=user_query,
                                status="no_chat_model_message",
                                summary="chat_model completed but returned no messages.",
                                errors=["chat_model state_update did not contain messages."],
                                tools_used=tools_requested,
                            )
                            continue

                        last_message = messages[-1]
                        tool_calls = getattr(last_message, "tool_calls", None)

                        if tool_calls:
                            for tool_call in tool_calls:
                                tool_name = tool_call.get("name")
                                if tool_name and tool_name not in tools_requested:
                                    tools_requested.append(tool_name)
                            continue

                        content = getattr(last_message, "content", None)
                        if content:
                            content_lower = content.lower()
                            status_type = (
                                "needs_clarification"
                                if "specify" in content_lower or "please tell me" in content_lower
                                else "unsupported"
                            )
                            final_response = make_error_response(
                                user_query=user_query,
                                status=status_type,
                                summary=content,
                                errors=[],
                                tools_used=tools_requested,
                            )

                    if node_name == "deterministic_final":
                        final_response_raw = state_update.get("final_response")
                        tools_utilized = state_update.get("tools_utilized", [])

                        if isinstance(final_response_raw, dict):
                            final_response = final_response_raw
                        else:
                            final_response = make_error_response(
                                user_query=user_query,
                                status="missing_final_response",
                                summary="deterministic_final did not return a valid response.",
                                errors=["No final_response dict found in deterministic_final node output."],
                                tools_used=tools_utilized,
                            )

    except TimeoutError:
        total_time = round(time.perf_counter() - start_time, 3)
        return {
            "response": make_error_response(
                user_query=user_query,
                status="graph_timeout",
                summary="The graph exceeded the timeout limit.",
                errors=[f"Graph execution timed out after {GRAPH_TIMEOUT_SECONDS} seconds."],
                tools_used=tools_requested,
            ),
            "timings": timings,
            "total_time_sec": total_time,
            "updated_messages": messages_tracker,
        }

    except Exception as e:
        total_time = round(time.perf_counter() - start_time, 3)
        return {
            "response": make_error_response(
                user_query=user_query,
                status="graph_error",
                summary="Error while running the graph.",
                errors=[str(e)],
                tools_used=tools_requested,
            ),
            "timings": timings,
            "total_time_sec": total_time,
            "updated_messages": messages_tracker,
        }

    total_time = round(time.perf_counter() - start_time, 3)

    if final_response is None:
        final_response = make_error_response(
            user_query=user_query,
            status="no_final_response",
            summary="The graph completed without producing a final response.",
            errors=["No deterministic_final response found. Check graph.py flow."],
            tools_used=tools_requested,
        )

    result = {
        "response": final_response,
        "timings": timings,
        "total_time_sec": total_time,
        "updated_messages": messages_tracker,
    }

    set_cached_final_response(user_query, result)
    return result


@app.on_event("startup")
async def startup_event():
    try:
        print("FastAPI started. Graph already built. Warming up worker LLM...")
        start = time.perf_counter()
        await llm.ainvoke("Return only: OK")
        print(f"Worker LLM warmup completed in {round(time.perf_counter() - start, 3)}s")
    except Exception as e:
        print("Worker LLM warmup failed:", e)


@app.get("/")
async def root():
    return {"message": "ERP Assistant API is running"}


@app.post("/chat")
async def chat(request: ChatRequest, fmt: Optional[str] = Query(None, alias="format")):
    request_id = str(uuid.uuid4())
    langsmith_config = {
        "run_name": "CHAPTER1_ASSIST_CHAT",
        "tags": ["fastapi", "langgraph", "erp-assistant"],
        "metadata": {
            "request_id": request_id,
            "query": request.query,
            "session_id": request.session_id,
        },
    }

    try:
        session_id = request.session_id or "default_session"
        session_data = SESSION_MEMORY.get(session_id, {"messages": []})

        result = await run_graph_query(
            user_query=request.query,
            past_messages=session_data.get("messages", []),
            langsmith_config=langsmith_config,
        )

        SESSION_MEMORY[session_id] = {
            "messages": result.get("updated_messages", []),
        }

        output_format = fmt or DEFAULT_OUTPUT_FORMAT

        if output_format == "text":
            text = await format_response_as_chat_text(
                response_data=result["response"],
                timings=result.get("timings", []),
                total_time=result.get("total_time_sec", 0.0),
            )
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


@app.post("/chat-text")
async def chat_text(request: ChatRequest):
    request_id = str(uuid.uuid4())
    langsmith_config = {
        "run_name": "CHAPTER1_ASSIST_CHAT_TEXT",
        "tags": ["fastapi", "langgraph", "erp-assistant", "text-response"],
        "metadata": {
            "request_id": request_id,
            "query": request.query,
            "session_id": request.session_id,
        },
    }

    try:
        session_id = request.session_id or "default_session"
        session_data = SESSION_MEMORY.get(session_id, {"messages": []})

        result = await run_graph_query(
            user_query=request.query,
            past_messages=session_data.get("messages", []),
            langsmith_config=langsmith_config,
        )

        SESSION_MEMORY[session_id] = {
            "messages": result.get("updated_messages", []),
        }

        text = await format_response_as_chat_text(
            response_data=result["response"],
            timings=result.get("timings", []),
            total_time=result.get("total_time_sec", 0.0),
        )
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


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)