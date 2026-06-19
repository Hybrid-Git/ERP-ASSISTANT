import asyncio
import json
import logging

import session_store
from langchain_core.messages import AIMessage, AIMessageChunk, RemoveMessage

from src.graph_runner import GRAPH_TIMEOUT_SECONDS, _timeout_iterate

logger = logging.getLogger("erp_assistant.streaming")


async def generate_stream_events(
    graph,
    initial_state: dict,
    config: dict,
    session_id: str,
    past_messages: list,
    past_summary: str,
    past_context: dict | None,
    past_last_tool: dict | None,
):
    messages_tracker = list(past_messages)
    summary_tracker = past_summary or ""
    context_tracker = dict(past_context or {})
    last_tool_tracker = dict(past_last_tool or {})
    response_text = None
    stream_data = {}
    tokens_emitted = False
    in_think_block = False

    try:
        try:
            async for event in _timeout_iterate(
                graph.astream_events(
                    initial_state,
                    config=config,
                    version="v2",
                ),
                GRAPH_TIMEOUT_SECONDS,
            ):
                kind = event["event"]
                tags = event.get("tags", [])

                if kind == "on_chat_model_stream" and "response_stream" in tags:
                    chunk = event["data"]["chunk"]

                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        tokens_emitted = True
                        token_text = chunk.content

                        if isinstance(token_text, str):
                            while token_text:
                                if not in_think_block:
                                    idx = token_text.find("<think>")

                                    if idx == -1:
                                        yield f"data: {json.dumps({'token': token_text})}\n\n"
                                        break

                                    if idx > 0:
                                        yield f"data: {json.dumps({'token': token_text[:idx]})}\n\n"

                                    token_text = token_text[idx + 7:]
                                    in_think_block = True

                                if in_think_block:
                                    idx = token_text.find("</think>")

                                    if idx == -1:
                                        break

                                    token_text = token_text[idx + 8:]
                                    in_think_block = False

                elif kind == "on_chain_end":
                    output = event["data"].get("output", {})

                    if isinstance(output, dict):
                        if "response_text" in output and output["response_text"]:
                            response_text = output["response_text"]

                        if "messages" in output:
                            for msg in output["messages"]:
                                if isinstance(msg, RemoveMessage):
                                    messages_tracker = [
                                        m for m in messages_tracker if m.id != msg.id
                                    ]
                                elif msg not in messages_tracker:
                                    messages_tracker.append(msg)

                        if "summary" in output and output["summary"]:
                            summary_tracker = output["summary"]

                        if "conversation_context" in output:
                            context_tracker = dict(output["conversation_context"])

                        if "last_tool_call" in output:
                            last_tool_tracker = dict(output["last_tool_call"])

                        if "final_response" in output and isinstance(output["final_response"], dict):
                            data = output["final_response"].get("data", {})
                            if data:
                                stream_data = data

        except TimeoutError:
            yield f"data: {json.dumps({'error': 'Request timed out'})}\n\n"

        except asyncio.CancelledError:
            logger.info("SSE client disconnected; saving partial session")
            raise

        except Exception as e:
            logger.exception("Stream error")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    finally:
        if response_text:
            try:
                updated = list(messages_tracker)
                updated.append(AIMessage(content=response_text))

                session_store.save_session(
                    session_id,
                    updated,
                    summary_tracker,
                    context_tracker,
                    last_tool_tracker,
                )

            except Exception:
                logger.exception("Failed to save streamed session")

    if response_text and not tokens_emitted:
        yield f"data: {json.dumps({'token': response_text})}\n\n"

    yield f"data: {json.dumps({'data': stream_data})}\n\n"
    yield f"data: {json.dumps({'session_id': session_id, 'done': True})}\n\n"