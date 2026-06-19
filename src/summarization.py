from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage
from src.schema import MainState
from src.config import summary_llm
from src.utils import log_token_usage, _get_tokenizer
import re
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger("erp_assistant.summarization")

load_dotenv()
limit = int(os.getenv("summary_limit"))


def _message_to_text(msg) -> str:
    if isinstance(msg, HumanMessage):
        return f"User: {msg.content}"
    if isinstance(msg, AIMessage):
        if getattr(msg, "tool_calls", None):
            calls = [
                f"{tc.get('name')}({tc.get('args', {})})"
                for tc in msg.tool_calls
            ]
            return f"Assistant called tools: {'; '.join(calls)}"
        return f"Assistant: {msg.content}"
    if isinstance(msg, ToolMessage):
        return f"Tool {msg.name} result: {str(msg.content)[:1000]}"
    return str(msg)


@traceable(name="summarization_node", run_type="chain")
async def summarization_node(state: MainState):
    logger.info("Summarization started", extra={"node": "summarization"})
    messages = state.get("messages", [])
    current_summary = state.get("summary", "")
    try:
        human_indices = [i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage)]
        if len(human_indices) < limit:
            print(f"Summary skipped............... Only {len(human_indices)} human messages so far.")
            return {}

        cutoff_index = human_indices[-1]
        messages_to_summarize = [m for m in messages[:cutoff_index] if not isinstance(m, SystemMessage)]

        if not messages_to_summarize:
            return {}

        print(f"Summary triggered after and now removing {len(human_indices)} old messages...............")

        summary_prompt = (
            f"You are an ERP assistant memory manager.\n"
            f"TASK: Write a concise summary of the conversation below. "
            f"Include key facts: which tools were called, what data was requested, "
            f"and any important results or conclusions. "
            f"Do NOT include raw data dumps — just the gist.\n\n"
            f"CONVERSATION:\n"
            f"/no_think\n"
        )
        conversation_text = "\n".join(_message_to_text(m) for m in messages_to_summarize)
        summary_input = [
            SystemMessage(content=summary_prompt),
            HumanMessage(content=conversation_text),
        ]
        response = await summary_llm.ainvoke(summary_input)
        log_token_usage(response, "summarization", input_text=str(summary_input), output_text=response.content)
        new_summary = response.content
        if not new_summary:
            new_summary = ""

        MAX_SUMMARY_CHARS = 16000
        if len(new_summary) > MAX_SUMMARY_CHARS:
            tail = new_summary[-MAX_SUMMARY_CHARS:]
            idx = tail.find("\n\n")
            if idx != -1:
                tail = tail[idx + 2:]
            new_summary = "... (truncated) ...\n\n" + tail

        delete_messages = [RemoveMessage(id=msg.id) for msg in messages_to_summarize if msg.id]

        # Compress ToolMessages in the remaining last turn
        remaining = [m for m in messages[cutoff_index:] if not isinstance(m, SystemMessage)]
        for msg in remaining:
            if isinstance(msg, ToolMessage) and len(msg.content) > 200:
                record_count = msg.content.count('"id":') or msg.content.count('"name":') or 0
                limit_match = re.search(r'"limit":\s*(\d+)', msg.content)
                summary_text = f"Tool {msg.name} returned {record_count} record(s)"
                if limit_match:
                    summary_text += f" (limited to {limit_match.group(1)})"
                delete_messages.append(RemoveMessage(id=msg.id))
                delete_messages.append(ToolMessage(
                    content=summary_text,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                ))

        print(f"Deleting and compressing: {len(delete_messages)} operations")
        return {
            "summary": new_summary,
            "messages": delete_messages,
        }

    except Exception as e:
        print(f"Error while removing messages and updating summary: {e}")
        return {"summary": current_summary or ""}
