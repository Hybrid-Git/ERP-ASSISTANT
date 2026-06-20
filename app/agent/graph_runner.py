
import asyncio
import time
import traceback
from src.graph import graph_builder
from langchain_core.messages import RemoveMessage
from src.response_utils import make_error_response
from src.cache import get_cached_final_response, set_cached_final_response


GRAPH_TIMEOUT_SECONDS = 300
graph = graph_builder()

async def _timeout_iterate(agen, timeout):
    """Iterate over an async generator with a per-step timeout.

    Compatible with Python < 3.11 (unlike asyncio.timeout()).
    """
    try:
        while True:
            try:
                item = await asyncio.wait_for(agen.__anext__(), timeout=timeout)
                yield item
            except StopAsyncIteration:
                return
    except asyncio.TimeoutError:
        raise TimeoutError()



async def run_graph_query(
    user_query: str,
    past_messages: list = None,
    langsmith_config: dict | None = None,
    past_summary: str | None = None,
    past_conversation_context: dict | None = None,
    past_last_tool_call: dict | None = None,
):
    cached_result = await get_cached_final_response(user_query)
    if cached_result is not None:
        cached_result["updated_messages"] = past_messages or []
        # Don't update context on cache hit — keep previous session state
        cached_result["conversation_context"] = past_conversation_context
        cached_result["last_tool_call"] = past_last_tool_call
        return cached_result

    start_time = time.perf_counter()

    initial_state = {
        "user_query": user_query,
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
        "conversation_context": past_conversation_context or {},
        "last_tool_call": past_last_tool_call or {},
    }

    final_response = None
    timings = []
    tools_requested = []
    messages_tracker = list(initial_state["messages"])
    config = langsmith_config or {}
    session_id = config.get("metadata", {}).get("session_id", "default_session")
    config = {**config, "configurable": {"thread_id": session_id}}
    try:
        summary_tracker = past_summary or ""
        context_tracker = dict(past_conversation_context or {})
        last_tool_tracker = dict(past_last_tool_call or {})
        response_text = None
        async for chunks in _timeout_iterate(
            graph.astream(
                initial_state,
                config=config,
                stream_mode="updates",
            ),
            GRAPH_TIMEOUT_SECONDS,
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
                            if isinstance(msg, RemoveMessage):
                                messages_tracker = [m for m in messages_tracker if m.id != msg.id]
                            elif msg not in messages_tracker:
                                messages_tracker.append(msg)
                    if "summary" in state_update and state_update["summary"]:
                        summary_tracker = state_update["summary"]
                    if "conversation_context" in state_update:
                        context_tracker = dict(state_update["conversation_context"])
                    if "last_tool_call" in state_update:
                        last_tool_tracker = dict(state_update["last_tool_call"])
                    if node_name == "chat_model":
                        messages = state_update.get("messages", [])
                        if not messages:
                            final_response = make_error_response(
                                query=user_query,
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

                        memory_answer = state_update.get("memory_answer", "")
                        if memory_answer:
                            print("Memory answer detected — not treating as error.")
                        else:
                            content = getattr(last_message, "content", None)
                            if content:
                                content_lower = content.lower()
                                status_type = (
                                    "needs_clarification"
                                    if "specify" in content_lower or "please tell me" in content_lower
                                    else "unsupported"
                                )
                                final_response = make_error_response(
                                    query=user_query,
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
                                query=user_query,
                                status="missing_final_response",
                                summary="deterministic_final did not return a valid response.",
                                errors=["No final_response dict found in deterministic_final node output."],
                                tools_used=tools_utilized,
                            )
                    if node_name == "response_generation":
                        new_text = state_update.get("response_text")
                        if new_text:
                            response_text = new_text

    except TimeoutError:
        total_time = round(time.perf_counter() - start_time, 3)
        return {
            "response": make_error_response(
                query=user_query,
                status="graph_timeout",
                summary="The graph exceeded the timeout limit.",
                errors=[f"Graph execution timed out after {GRAPH_TIMEOUT_SECONDS} seconds."],
                tools_used=tools_requested,
            ),
            "timings": timings,
            "total_time_sec": total_time,
            "updated_messages": messages_tracker,
            "summary": summary_tracker or "",
            "conversation_context": context_tracker,
            "last_tool_call": last_tool_tracker,
        }

    except Exception as e:
        import traceback
        total_time = round(time.perf_counter() - start_time, 3)
        print(f"[GRAPH ERROR] {traceback.format_exc()}")
        return {
            "response": make_error_response(
                query=user_query,
                status="graph_error",
                summary="Error while running the graph.",
                errors=[str(e)],
                tools_used=tools_requested,
            ),
            "timings": timings,
            "total_time_sec": total_time,
            "updated_messages": messages_tracker,
            "summary": summary_tracker or "",
            "conversation_context": context_tracker,
            "last_tool_call": last_tool_tracker,
        }
    print(f"[Remove Messages Result] total messages tracked:{len(messages_tracker)}. Final summary: {summary_tracker}")
    total_time = round(time.perf_counter() - start_time, 3)

    if final_response is None:
        final_response = make_error_response(
            query=user_query,
            status="no_final_response",
            summary="The graph completed without producing a final response.",
            errors=["No deterministic_final response found. Check graph.py flow."],
            tools_used=tools_requested,
        )

    result = {
        "response": response_text if response_text else final_response,
        "response_text": response_text,
        "data":final_response.get("data", {}) if isinstance(final_response, dict) else {},
        "timings": timings,
        "total_time_sec": total_time,
        "updated_messages": messages_tracker,
        "summary": summary_tracker,
        "conversation_context": context_tracker,
        "last_tool_call": last_tool_tracker,
    }

    await set_cached_final_response(user_query, result)
    return result
