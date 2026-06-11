from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, RemoveMessage
from src.schema import MainState
from src.config import summary_llm


@traceable(name="summarization_node", run_type="chain")
async def summarization_node(state: MainState):
    print("Summarization node activated............")
    messages = state.get("messages", [])
    current_summary = state.get("summary", "")
    try:
        human_indices = [i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage)]
        if len(human_indices) < 6:
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
        summary_input = [SystemMessage(content=summary_prompt)] + messages_to_summarize
        response = await summary_llm.ainvoke(summary_input)
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
        print(f"Deleting {len(delete_messages)} old messages")
        return {
            "summary": new_summary,
            "messages": delete_messages,
        }

    except Exception as e:
        print(f"Error while removing messages and updating summary: {e}")
        return {"summary": current_summary or ""}
