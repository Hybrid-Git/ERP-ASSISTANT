from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from src.graph import graph_builder
from src.config import llm

import json
import re
import time
import asyncio
import copy
from typing import Any, Dict, List, Optional

import uvicorn
import uuid

app = FastAPI(
    title="CHAPTER-1-ASSIST",
    version="1.0.0"
)
graph = graph_builder()
GRAPH_TIMEOUT_SECONDS = 300
# ==============================
# FINAL RESPONSE CACHE
# ==============================

FINAL_RESPONSE_CACHE = {}
FINAL_RESPONSE_CACHE_TTL_SECONDS = 300  # 5 minutes


def normalize_query_for_cache(query: str) -> str:
    return " ".join((query or "").lower().strip().split())


def should_cache_final_response(result: dict) -> bool:
    """
    Only cache complete FastAPI-shaped successful responses.

    Expected shape:
    {
        "response": {...},
        "timings": [...],
        "total_time_sec": 12.34
    }
    """

    if not isinstance(result, dict):
        return False

    response = result.get("response")

    if not isinstance(response, dict):
        print("[FINAL CACHE SKIP] Missing response wrapper")
        return False

    success = response.get("success")
    status = response.get("status")
    tools_used = response.get("tools_used", [])

    if not tools_used:
        print("[FINAL CACHE SKIP] No tools used")
        return False

    if success is True and status == "success":
        return True

    print("[FINAL CACHE SKIP] Response not safe to cache")
    return False


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

    result = json.loads(json.dumps(result, ensure_ascii=False))

    result["timings"] = [
        {
            "node": "final_response_cache",
            "duration_sec": 0.001,
        }
    ]

    result["total_time_sec"] = 0.001

    return result


def set_cached_final_response(query: str, result: dict):
    if not should_cache_final_response(result):
        return

    key = normalize_query_for_cache(query)

    FINAL_RESPONSE_CACHE[key] = {
        "cached_at": time.time(),
        "result": json.loads(json.dumps(result, ensure_ascii=False)),
    }

    print(f"[FINAL CACHE SET] {key}")

    
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)





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


def ns_to_sec(value):
    if value is None:
        return None

    try:
        return round(value / 1_000_000_000, 3)
    except Exception:
        return value


def print_ollama_metadata(message):
    metadata = getattr(message, "response_metadata", {}) or {}

    if not metadata:
        print("[OLLAMA METADATA] No response_metadata found")
        return

    print("\n========== OLLAMA METADATA ==========")
    print("model:", metadata.get("model"))
    print("done_reason:", metadata.get("done_reason"))

    print("total_duration:", ns_to_sec(metadata.get("total_duration")), "sec")
    print("load_duration:", ns_to_sec(metadata.get("load_duration")), "sec")
    print("prompt_eval_duration:", ns_to_sec(metadata.get("prompt_eval_duration")), "sec")
    print("eval_duration:", ns_to_sec(metadata.get("eval_duration")), "sec")

    print("prompt_eval_count:", metadata.get("prompt_eval_count"))
    print("eval_count:", metadata.get("eval_count"))
    print("=====================================\n")
# ==============================
# FORMAT NEGOTIATION
# ==============================
# DEFAULT_OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "json")
DEFAULT_OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "text")


import re
from typing import Any

def clean_key_dynamically(key: str) -> str:
    """
    Automatically converts camelCase and snake_case keys into Title Case.
    Converts standard ERP acronyms (like GST, HSN, ID) to full uppercase.
    """
    # Split camelCase and snake_case into separate words
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", key)
    spaced = spaced.replace("_", " ")
    words = spaced.split()
    
    # Common ERP / Financial terms that look best fully capitalized
    acronyms = {"id", "hsn", "gst", "tds", "tcs", "sku", "igst", "cgst", "sgst", "uom", "b2b", "b2c", "refid", "tx"}
    
    cleaned_words = [
        word.upper() if word.lower() in acronyms else word.capitalize()
        for word in words
    ]
    return " ".join(cleaned_words)


def format_value_dynamically(value: Any, indent_level: int = 1) -> str:
    """
    Recursively dissects any nested data types (dicts, lists, timestamps) 
    and layouts them cleanly without hardcoded field definitions.
    """
    indent = "  " * indent_level
    
    # 1. Clean up Date-Range maps like {'from': '...', 'to': '...'}
    if isinstance(value, dict) and "from" in value and "to" in value:
        return f"{value['from']} to {value['to']}"
        
    # 2. Handle standard nested dictionaries
    if isinstance(value, dict):
        if not value:
            return "None"
        dict_lines = []
        for k, v in value.items():
            dict_lines.append(f"\n{indent}• {clean_key_dynamically(k)}: {format_value_dynamically(v, indent_level + 1)}")
        return "".join(dict_lines)
        
    # 3. Handle arrays and lists (like transactions)
    if isinstance(value, list):
        if not value:
            return "None"
        list_lines = []
        for index, item in enumerate(value, start=1):
            if isinstance(item, dict):
                # Format sub-records cleanly with dynamic list pointers
                list_lines.append(f"\n{indent} Record Line #{index}:")
                for k, v in item.items():
                    list_lines.append(f"\n{indent}  ▪ {clean_key_dynamically(k)}: {format_value_dynamically(v, indent_level + 2)}")
            else:
                list_lines.append(f"\n{indent}• {format_value_dynamically(item, indent_level + 1)}")
        return "".join(list_lines)
        
    # 4. Clean up ISO Datetime strings (e.g., 2024-04-03T00:00:00.000Z -> 2024-04-03)
    if isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}T", value):
        return value.split("T")[0]
        
    return str(value)


# def format_response_as_text(response_data: dict) -> str:
#     """
#     Replaced static presentation loop with dynamic parsing layout engine.
#     """
#     status = response_data.get("status", "")
#     summary = response_data.get("summary", "")
#     query = response_data.get("query", "")
#     tools_used = response_data.get("tools_used", [])
#     data = response_data.get("data", {})
#     errors = response_data.get("errors", [])
#     unsupported = response_data.get("unsupported_parts", [])

#     lines = []
#     lines.append("=" * 70)
#     lines.append("                      ERP ASSISTANT REPORT")
#     lines.append("=" * 70)

#     lines.append(f"\n USER QUERY: {query}")
#     lines.append(f" RUN STATUS: {status.upper()}")

#     if summary:
#         lines.append(f" SUMMARY:    {summary}")

#     if tools_used:
#         # Format the technical tool names visually cleaner
#         readable_tools = [t.replace("get_", "").replace("_", " ").upper() for t in tools_used]
#         lines.append(f" ENGINES:    {', '.join(readable_tools)}")

#     if data:
#         for tool_name, records in data.items():
#             if not records:
#                 continue
            
#             section_header = tool_name.replace("get_", "").replace("_", " ").upper()
#             lines.append(f"\n▶ [{section_header}]")
#             lines.append("─" * 40)
            
#             for record in records:
#                 if isinstance(record, dict):
#                     for key, value in record.items():
#                         cleaned_key = clean_key_dynamically(key)
#                         formatted_val = format_value_dynamically(value, indent_level=2)
#                         lines.append(f"  ▪ {cleaned_key}: {formatted_val}")
#                     lines.append("  " + "┈" * 20)

#     if errors:
#         lines.append(f"\n ERRORS ({len(errors)}):")
#         for err in errors:
#             if isinstance(err, dict):
#                 lines.append(f"  - {err.get('error', err)}")
#             else:
#                 lines.append(f"  - {err}")

#     if unsupported:
#         lines.append(f"\n UNSUPPORTED REQS: {', '.join(unsupported)}")

#     lines.append("\n" + "=" * 70)
#     return "\n".join(lines)

def format_response_as_text(response_data: dict, timings: list = None, total_time: float = None) -> str:
    """
    Completely dynamic text presentation engine.
    Accepts graph metrics to output node-by-node execution logs.
    """
    status = response_data.get("status", "")
    summary = response_data.get("summary", "")
    query = response_data.get("query", "")
    tools_used = response_data.get("tools_used", [])
    data = response_data.get("data", {})
    errors = response_data.get("errors", [])
    unsupported = response_data.get("unsupported_parts", [])

    lines = []
    lines.append("=" * 70)
    lines.append("                      📢 ERP ASSISTANT REPORT")
    lines.append("=" * 70)

    lines.append(f"\n📝 USER QUERY: {query}")
    lines.append(f"🚦 RUN STATUS: {status.upper()}")

    if summary:
        lines.append(f"📊 SUMMARY:    {summary}")

    if tools_used:
        readable_tools = [t.replace("get_", "").replace("_", " ").upper() for t in tools_used]
        lines.append(f"⚙️ ENGINES:    {', '.join(readable_tools)}")

    if data:
        for tool_name, records in data.items():
            if not records:
                continue
            
            section_header = tool_name.replace("get_", "").replace("_", " ").upper()
            lines.append(f"\n▶ [{section_header}]")
            lines.append("─" * 40)
            
            for record in records:
                if isinstance(record, dict):
                    for key, value in record.items():
                        cleaned_key = clean_key_dynamically(key)
                        formatted_val = format_value_dynamically(value, indent_level=2)
                        lines.append(f"  ▪ {cleaned_key}: {formatted_val}")
                    lines.append("  " + "┈" * 20)

    if errors:
        lines.append(f"\n❌ ERRORS ({len(errors)}):")
        for err in errors:
            if isinstance(err, dict):
                lines.append(f"  - {err.get('error', err)}")
            else:
                lines.append(f"  - {err}")

    if unsupported:
        lines.append(f"\n⚠️ UNSUPPORTED REQS: {', '.join(unsupported)}")

    # 🚀 NEW: DYNAMIC PERFORMANCE LOGGING (Zero Hardcoding)
    if timings:
        lines.append("\n⏱️ PERFORMANCE LOGS")
        lines.append("─" * 40)
        for timing in timings:
            # Dynamically pull and clean the node identity and duration string
            node_label = clean_key_dynamically(timing.get("node", "Unknown Node"))
            duration = timing.get("duration_sec", 0.0)
            lines.append(f"  ▪ {node_label}: {duration}s")
            
        if total_time:
            lines.append(f"  ▪ Total Turnaround Time: {total_time}s")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)



# def format_response_as_text(response_data: dict) -> str:
#     status = response_data.get("status", "")
#     summary = response_data.get("summary", "")
#     query = response_data.get("query", "")
#     tools_used = response_data.get("tools_used", [])
#     data = response_data.get("data", {})
#     errors = response_data.get("errors", [])
#     unsupported = response_data.get("unsupported_parts", [])

#     lines = []
#     lines.append("=" * 60)
#     lines.append("  ERP ASSISTANT RESPONSE")
#     lines.append("=" * 60)

#     lines.append(f"\nQuery:    {query}")
#     lines.append(f"Status:   {status}")

#     if summary:
#         lines.append(f"Summary:  {summary}")

#     if tools_used:
#         lines.append(f"\nTools Used: {', '.join(tools_used)}")

#     if data:
#         for tool_name, records in data.items():
#             if not records:
#                 continue
#             lines.append(f"\n--- {tool_name} ---")
#             for record in records:
#                 if isinstance(record, dict):
#                     for key, value in record.items():
#                         lines.append(f"  {key}: {value}")
#                     lines.append("  " + "-" * 40)

#     if errors:
#         lines.append(f"\nErrors ({len(errors)}):")
#         for err in errors:
#             if isinstance(err, dict):
#                 lines.append(f"  - {err.get('error', err)}")
#             else:
#                 lines.append(f"  - {err}")

#     if unsupported:
#         lines.append(f"\nUnsupported Parts: {', '.join(unsupported)}")

#     lines.append("\n" + "=" * 60)
#     return "\n".join(lines)

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


def format_response_as_chat_text(response_data: dict) -> str:
    status = response_data.get("status", "")
    summary = response_data.get("summary", "")
    query = response_data.get("query", "")
    data = response_data.get("data", {})
    errors = response_data.get("errors", [])
    unsupported_parts = response_data.get("unsupported_parts", [])

    lines = []

    if status == "success":
        lines.append("Here’s what I found:")
    elif status == "partial_success":
        lines.append("I found some of the requested information:")
    elif status == "no_matching_records":
        lines.append("I could not find any matching records for this query.")
    else:
        lines.append("I could not complete the request properly.")

    if summary:
        lines.append("")
        lines.append(summary)

    if data:
        for tool_name, records in data.items():
            if not records:
                continue

            readable_section = tool_name.replace("get_", "").replace("_", " ").title()
            lines.append("")
            lines.append(f"{readable_section}:")

            for index, record in enumerate(records, start=1):
                if isinstance(record, dict):
                    if len(records) > 1:
                        lines.append(f"\n{index}. Record")

                    for key, value in record.items():
                        lines.append(f"- {pretty_field_name(key)}: {value}")

                else:
                    lines.append(f"- {record}")

    if unsupported_parts:
        lines.append("")
        lines.append("Some parts of your query are not supported yet:")
        for part in unsupported_parts:
            lines.append(f"- {part}")

    if errors:
        lines.append("")
        lines.append("Errors:")
        for err in errors:
            if isinstance(err, dict):
                lines.append(f"- {err.get('error', err)}")
            else:
                lines.append(f"- {err}")

    return "\n".join(lines)

async def run_graph_query(
    user_query: str,
    langsmith_config: dict | None = None,
):
    cached_result = get_cached_final_response(user_query)

    if cached_result is not None:
        return cached_result

    start_time = time.perf_counter()

    initial_state = {
        "user_query": user_query,
        "canonical_query": "",
        "translator_used": False,
        "translator_confidence": "",
        "detected_language": "",
        "messages": [],
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
    }

    final_response = None
    timings = []
    tools_requested = []

    try:
        async with asyncio.timeout(GRAPH_TIMEOUT_SECONDS):
            async for chunks in graph.astream(
                initial_state,
                config=langsmith_config or {},
                stream_mode="updates",
            ):
                for node_name, state_update in chunks.items():
                    print(f"Finished running: {node_name}")

                    # Collect timings from timed_node wrapper
                    if "step_timings" in state_update:
                        timings.extend(state_update["step_timings"])

                        for timing in state_update["step_timings"]:
                            print(
                                f"[STEP TIME] {timing['node']} = {timing['duration_sec']}s"
                            )

                    # First LLM call: tool planning/tool calling only
                    if node_name == "chat_model":
                        messages = state_update.get("messages", [])

                        if not messages:
                            final_response = make_error_response(
                                user_query=user_query,
                                status="no_chat_model_message",
                                summary="chat_model completed but returned no messages.",
                                errors=[
                                    "chat_model state_update did not contain messages."
                                ],
                                tools_used=tools_requested,
                            )
                            continue

                        last_message = messages[-1]

                        # Print Ollama metadata for debugging latency
                        tool_calls = getattr(last_message, "tool_calls", None)

                        if tool_calls:
                            print("Tool calls requested:")

                            for tool_call in tool_calls:
                                tool_name = tool_call.get("name")
                                tool_args = tool_call.get("args", {})

                                if tool_name and tool_name not in tools_requested:
                                    tools_requested.append(tool_name)

                                print(f"- {tool_name} args={tool_args}")

                            continue

                        # If chat_model gives no tool calls, graph will likely end.
                        # So capture the real issue here instead of returning no_final_response.
                        content = getattr(last_message, "content", None)

                        print("[CHAT_MODEL] No tool calls returned.")
                        print("[CHAT_MODEL] Content:", repr(content))

                        if content:
                            final_response = make_error_response(
                                user_query=user_query,
                                status="unsupported",
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
                                errors=[
                                    "No final_response dict found in deterministic_final node output."
                                ],
                                tools_used=tools_utilized,
                            )

    except TimeoutError:
        total_time = round(time.perf_counter() - start_time, 3)

        return {
            "response": make_error_response(
                user_query=user_query,
                status="graph_timeout",
                summary="The graph exceeded the timeout limit.",
                errors=[
                    f"Graph execution timed out after {GRAPH_TIMEOUT_SECONDS} seconds."
                ],
                tools_used=tools_requested,
            ),
            "timings": timings,
            "total_time_sec": total_time,
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
        }

    total_time = round(time.perf_counter() - start_time, 3)

    if final_response is None:
        final_response = make_error_response(
            user_query=user_query,
            status="no_final_response",
            summary="The graph completed without producing a final response.",
            errors=[
                "No deterministic_final response found. Check graph.py flow."
            ],
            tools_used=tools_requested,
        )

    result = {
        "response": final_response,
        "timings": timings,
        "total_time_sec": total_time,
    }

    set_cached_final_response(user_query, result)

    return result


@app.on_event("startup")
async def startup_event():
    try:
        print("FastAPI started. Graph already built.")
        print("Warming up worker LLM...")

        start = time.perf_counter()
        await llm.ainvoke("Return only: OK")
        print(f"Worker LLM warmup completed in {round(time.perf_counter() - start, 3)}s")

    except Exception as e:
        print("Worker LLM warmup failed:", e)


@app.get("/")
async def root():
    return {
        "message": "ERP Assistant API is running"
    }


@app.post("/chat")
async def chat(request: ChatRequest, fmt: Optional[str] = Query(None, alias="format")):
    request_id = str(uuid.uuid4())

    langsmith_config = {
        "run_name": "CHAPTER1_ASSIST_CHAT",
        "tags": [
            "fastapi",
            "langgraph",
            "erp-assistant",
        ],
        "metadata": {
            "request_id": request_id,
            "query": request.query,
        },
    }

    try:
        result = await run_graph_query(
            request.query,
            langsmith_config=langsmith_config,
        )

        output_format = fmt or DEFAULT_OUTPUT_FORMAT

        if output_format == "text":
            text = format_response_as_text(
                response_data=result["response"],
                timings=result.get("timings", []),
                total_time=result.get("total_time_sec",0.0),
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
async def chat_text(request: ChatRequest,fmt: Optional[str] = Query(None, alias="format")):
    request_id = str(uuid.uuid4())

    langsmith_config = {
        "run_name": "CHAPTER1_ASSIST_CHAT_TEXT",
        "tags": [
            "fastapi",
            "langgraph",
            "erp-assistant",
            "text-response",
        ],
        "metadata": {
            "request_id": request_id,
            "query": request.query,
        },
    }

    try:
        result = await run_graph_query(
            request.query,
            langsmith_config=langsmith_config,
        )
        output_format = fmt or DEFAULT_OUTPUT_FORMAT
        if output_format == "text":
            text = format_response_as_chat_text(result["response"])
            return PlainTextResponse(text)

    except Exception as e:
        error_response = make_error_response(
            user_query=request.query,
            status="server_error",
            summary="Server error while processing the query.",
            errors=[str(e)],
        )

        return PlainTextResponse(
            format_response_as_chat_text(error_response),
            status_code=500,
        )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

