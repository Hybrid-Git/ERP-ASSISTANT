import json
import re
import time
import uuid
import traceback
from langchain_core.utils.function_calling import convert_to_openai_tool
from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage,ToolMessage
from app.schemas.state import MainState
from app.services.tools_api import tools_dict
from app.prompts.tool_doc import TOOL_INTENT_REGISTRY, TOOL_NAME_ALIASES
from app.core.config import llm, summary_llm
from app.utils.utils import (
    now, sec, log_token_usage, sanitize_tool_filters,
    strip_think_tags,
)
from app.prompts.prompts import META_QUESTION_PATTERNS_GLOBAL
import logging

logger = logging.getLogger("erp_assistant.chat_model")


def _strip_schema_descriptions(tool_schemas: list[dict]) -> list[dict]:
    for schema in tool_schemas:
        func = schema.get("function", {})
        params = func.get("parameters", {})
        props = params.get("properties", {})
        for p_name, p_schema in props.items():
            p_schema.pop("description", None)
    return tool_schemas


def _get_recent_tool_calls(messages: list, max_calls: int = 3) -> list[dict]:
    calls = []
    seen = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                name = tc.get("name")
                args = tc.get("args", {})
                if name and args and name not in seen:
                    calls.append({"name": name, "args": args})
                    seen.add(name)
                    if len(calls) >= max_calls:
                        return calls
    return calls


def _summarize_tool_result(content: str) -> str:
    try:
        parsed = json.loads(content) if isinstance(content, str) else content
        if not isinstance(parsed, dict):
            return "Data found"
        if not parsed.get("success", True):
            return f"Error: {parsed.get('error', 'Unknown')}"
        records = parsed.get("data", [])
        if not isinstance(records, list):
            records = [records] if records else []
        if not records:
            return "No results"
        names = []
        for r in records[:3]:
            if isinstance(r, dict):
                n = r.get("name") or r.get("productName") or r.get("customerName") or r.get("partyName") or ""
                if n:
                    names.append(str(n))
        count = len(records)
        if names:
            name_list = ", ".join(names)
            return f"Found {count}: {name_list}" + ("..." if count > 3 else "")
        return f"Found {count} records"
    except (json.JSONDecodeError, TypeError):
        return "Data found"


def _build_memory_context(messages: list, max_exchanges: int = 3) -> str:
    exchanges = []
    i = len(messages) - 1
    while i >= 0 and len(exchanges) < max_exchanges:
        msg = messages[i]
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            user_query = ""
            for k in range(i - 1, -1, -1):
                if isinstance(messages[k], HumanMessage):
                    user_query = getattr(messages[k], "content", "") or ""
                    break
            tc_ids = {tc["id"] for tc in msg.tool_calls if tc.get("id")}
            result_summaries = []
            for j in range(i + 1, len(messages)):
                tm = messages[j]
                if isinstance(tm, ToolMessage) and getattr(tm, "tool_call_id", None) in tc_ids:
                    result_summaries.append(_summarize_tool_result(tm.content))
                elif isinstance(tm, AIMessage) and getattr(tm, "tool_calls", None):
                    break
            parts = []
            for tc in msg.tool_calls:
                name = tc.get("name", "?")
                args = tc.get("args", {})
                nice_args = ", ".join(f"{k}={v}" for k, v in args.items() if k != "fields")
                parts.append(f"{name}({nice_args})")
            tool_desc = "; ".join(parts)
            line = f'  Asked: "{user_query}" → {tool_desc}'
            if result_summaries:
                results_str = "; ".join(result_summaries)
                if len(results_str) > 250:
                    results_str = results_str[:250] + "..."
                line += f" → {results_str}"
            exchanges.append(line)
        elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = getattr(msg, "content", "") or ""
            if content.strip():
                user_query = ""
                for k in range(i - 1, -1, -1):
                    if isinstance(messages[k], HumanMessage):
                        user_query = getattr(messages[k], "content", "") or ""
                        break
                if user_query:
                    exchanges.append(f'  Asked: "{user_query}" → Answered: {content[:250]}')
        i -= 1
    exchanges.reverse()
    if exchanges:
        return "Earlier in this conversation:\n" + "\n".join(exchanges) + "\n"
    return ""


CATEGORY_SUMMARIES: dict[str, tuple[str, str]] = {
    "customer": ("Customers", "Search customers/parties, view ledgers, find top buyers."),
    "stock": ("Stock & Inventory", "Check stock levels, low stock, out-of-stock by product/SKU/HSN."),
    "gst_report": ("GST", "View GST summary — B2B, B2C, exports, nil-rated, exempt."),
    "tds_report": ("TDS/TCS", "Check TDS and TCS outstanding/payable reports."),
    "analytics": ("Reports & Analytics", "Top/popular/slow-moving products, sales summary & trends."),
}



def _build_capability_text() -> str:
    seen_categories = set()
    lines = []
    for _tname, meta in TOOL_INTENT_REGISTRY.items():
        cat = meta.get("category")
        if not cat or cat in seen_categories:
            continue
        seen_categories.add(cat)
        entry = CATEGORY_SUMMARIES.get(cat)
        if entry:
            heading, summary = entry
            lines.append(f"- **{heading}**: {summary}")
    extra_cats = seen_categories - set(CATEGORY_SUMMARIES.keys())
    if extra_cats:
        lines.append("- **And more**: Vendors, purchases, ledgers, and invoices.")
    if not lines:
        return "I can help with ERP data — customers, stock, GST, invoices, reports, and more."
    return "\n".join(lines)

def build_system_prompt(
    user_query: str,
    selected_tools: list[str],
    query_parts: list[str] | None = None,
    summary: str | None = None,
    messages: list | None = None,
    last_tool_call: dict | None = None,
    conversation_context: dict | None = None,
    original_query: str = "",
) -> str:
    lines = [
        "You are an ERP assistant. Be literal — no reinterpretation.",
        "Never invent IDs, names, dates, or amounts.",
        "Must call a tool. Never answer in prose. No thinking output.",
        "",
        "STRICT ANTI-HALLUCINATION RULES:",
        "- Never pass scope words (sab/saare/all/every/poore/saari) as search/term. They describe scope ('all'), not a name. For 'all records', call list without a search filter.",
        "- Never invent IDs/names. Use ONLY identifiers from prior tool results. If missing, call a list tool first — never guess.",
        "- When query has resolved entity names (e.g. from pronoun resolution), use ONLY those exact names as search/term. Do not pull unrelated names from other tools like top_customers.",
        "",
        "FOLLOW-UP: Reuse prior results when possible. "
        "For extremes (top/highest/least): sort_order 'desc'/'asc'. "
        "CRITICAL: Clear ALL params when switching tools — reuse only within same tool.",
    ]

    # Short queries are likely follow-ups; always include context if available
    is_follow_up = len(user_query.split()) <= 3
    if messages and is_follow_up:
        recent_calls = _get_recent_tool_calls(messages)
        if recent_calls:
            lines.append("")
            lines.append("--- RECENT TOOL CALLS (for follow-up context) ---")
            for call in recent_calls:
                compact = json.dumps(call['args'], separators=(',',':'))
                lines.append(f"  {call['name']}({compact})")
            lines.append("For follow-up queries, reuse these same parameters. Only change what the user explicitly asks about.")
            lines.append("--------------------------------------------------")
        if summary:
            lines.append("")
            lines.append("--- PREVIOUS CONVERSATION CONTEXT ---")
            lines.append(summary[:800])
            lines.append("--------------------------------------------------")
        if conversation_context:
            entities = conversation_context.get("entities", [])
            if entities:
                lines.append("")
                lines.append("--- KNOWN ENTITIES ---")
                names = []
                for e in entities[-1:]:
                    name = e.get("name", "")
                    id_ = e.get("id")
                    if id_ is not None:
                        names.append(f"{name} (ID {id_})")
                    elif name:
                        names.append(name)
                lines.append("Recently mentioned: " + ", ".join(names))
                lines.append("--------------------------------------------------")
        if last_tool_call:
            recent_names = {c["name"] for c in recent_calls} if recent_calls else set()
            extra = {k: v for k, v in last_tool_call.items() if k not in recent_names}
            capped_extra = dict(list(extra.items())[-2:])
            if capped_extra:
                lines.append("")
                lines.append("--- PREVIOUS TOOL CALLS (from earlier in conversation) ---")
                lines.append(json.dumps(capped_extra))
                lines.append("For follow-up queries, reuse these same parameters. Only change what the user explicitly asks about.")
                lines.append("--------------------------------------------------")
    if query_parts and len(query_parts) > 1:
        lines.append("")
        lines.append("--- MULTI-INTENT QUERY ---")
        lines.append(f"The user's query has {len(query_parts)} separate intents:")
        for i, part in enumerate(query_parts, 1):
            lines.append(f"  {i}. {part}")
        lines.append("You MUST call a separate tool for each distinct intent. Do NOT combine different intents into one tool call.")
        lines.append("Use the same parameters for follow-up parts, but call a new tool for each independent sub-query.")
        lines.append("--------------------------")
    if selected_tools:
        lines.append("")
        lines.append(f"Available tools: {', '.join(selected_tools)}")
        lines.append("Call the tool(s) that are relevant to the query. You do NOT need to call every tool — only those that actually address the user's request.")
        lines.append("")
        lines.append("Tool rules:")
        lines.append("  You may call the SAME tool MULTIPLE TIMES with different sort/filter arguments for different sub-requests.")
        for tool_name in selected_tools:
            meta = TOOL_INTENT_REGISTRY.get(tool_name)
            if meta and meta.get("prompt_tips"):
                lines.append(f"  {tool_name}: {meta['prompt_tips']}")
    lines.append("")
    lines.append("FAILURE-AWARE:")
    lines.append("  If empty → say 'No records for X'. Acknowledge prior failures if any.")
    lines.append("")
    lines.append("PARAMETER RULE:")
    lines.append("  NEVER copy params between different tools.")
    return "\n".join(lines)


@traceable(name="chat_model_node", run_type="llm")
async def chat_model_node(state: MainState):
    node_start = now()
    try:
        logger.info("Chat model started", extra={"node": "chat_model"})
        step = now()
        original_query = state.get("original_query") or state.get("user_query", "")
        user_query = state.get("canonical_query") or state.get("user_query", "")
        selected_tools = state.get("selected_tools", [])
        query_parts = state.get("query_parts", [user_query])
        loop_count = state.get("loop_count", 0)
        summary = state.get("summary", "")
        previous_messages = [
            msg for msg in state.get("messages", [])
            if not isinstance(msg, SystemMessage)
        ]

        step = now()
        available_tools = [
            tools_dict[name]
            for name in selected_tools
            if name in tools_dict
        ]
        _LANG_EXAMPLES = {
            "gujarati": "Like: 'hu tamari business data ma help karu chu. customers, stock, GST ni details apvi shaku chu.'",
            "marathi": "Like: 'mi tumhalya business data madat karu shakto. customers, stock, GST chi mahiti deu shakto.'",
            "hinglish": "Like: 'Main sirf business data aur ERP systems ke baare mein bata sakta hoon. Thanos jaise fictional topics ke baare mein main nahi jaanta.'",
            "hindi": "Like: 'Main keval business data aur ERP ki jankari de sakta hoon. Thanos jaise fictional characters ke baare mein nahi bata sakta.'",
        }
        if not available_tools:
            query_type = (state.get("query_type") or "").strip()
            original = state.get("original_query", "")
            canonical = state.get("canonical_query", "")
            detected_language = (state.get("detected_language") or "").strip().lower()
            lang_label = f" in {detected_language}" if detected_language not in ("unknown", "auto", "") else ""
            example_suffix = _LANG_EXAMPLES.get(detected_language, "")

            if detected_language == "english":
                lang_warning = "Respond in natural English."
            elif detected_language == "hinglish":
                lang_warning = (
                    "Respond in Hinglish (Hindi mixed with English, written in a romanized script). "
                    "Do NOT respond in plain English, formal Hindi, or native Devanagari script."
                )
            elif detected_language == "hindi":
                lang_warning = (
                    "Respond in Hindi written in a romanized/transliterated script (using Latin characters). "
                    "Do NOT respond in plain English or native Devanagari script."
                )
            elif detected_language in ("unknown", "auto", ""):
                lang_warning = "Match the user's language, script, and register. If they write in a romanized script, respond in romanized form."
            else:
                lang_warning = (
                    f"Respond in {detected_language} written in a romanized script (using Roman/Latin alphabet). "
                    f"Do NOT respond in plain English, Hindi, or native script characters (like Devanagari or Gujarati script)."
                )

            if query_type == "greeting":
                greeting_prompt = (
                    "You are an ERP assistant. The user just greeted you.\n"
                    f"The user said: \"{original}\"\n"
                    f"This means: \"{canonical}\"\n\n"
                    "Respond warmly in 1-2 sentences. Vary your greeting each time. "
                    "End with a follow-up question.\n"
                    f"CRITICAL — The user asked{lang_label}: \"{original}\". "
                    f"You MUST respond{lang_label}, matching the user's language, script, and register. "
                    f"If the user writes in a romanized script, you MUST respond in romanized form. "
                    f"{example_suffix} "
                    f"{lang_warning}\n"
                    "/no_think"
                )
                try:
                    resp = await summary_llm.ainvoke([
                        SystemMessage(content=greeting_prompt),
                        HumanMessage(content=user_query),
                    ])
                    reason = strip_think_tags((getattr(resp, "content", "") or "").strip())
                except Exception:
                    reason = "Hello! How can I help you with your ERP data today?"
                print(f"[CHAT MODEL] Greeting response: {reason}")
                return {
                    "messages": [HumanMessage(content=user_query), AIMessage(content=reason)],
                    "memory_answer": reason,
                    "loop_count": loop_count + 1,
                }

            if query_type == "capability":
                caps_text = _build_capability_text()
                cap_prompt = (
                    "The user asked what you can do. These are your capabilities:\n\n"
                    f"{caps_text}\n\n"
                    f"The user said: \"{original}\"\n"
                    f"This means: \"{canonical}\"\n\n"
                    "Present these capabilities naturally in 2-3 sentences. "
                    "End with a follow-up question.\n"
                    f"CRITICAL — The user asked{lang_label}: \"{original}\". "
                    f"You MUST respond{lang_label}, matching the user's language, script, and register. "
                    f"If the user writes in a romanized script, you MUST respond in romanized form. "
                    f"{example_suffix} "
                    f"{lang_warning}\n"
                    "/no_think"
                )
                try:
                    resp = await summary_llm.ainvoke([
                        SystemMessage(content=cap_prompt),
                        HumanMessage(content=user_query),
                    ])
                    reason = strip_think_tags((getattr(resp, "content", "") or "").strip())
                except Exception:
                    reason = "I can help you with customer details, stock levels, GST reports, TDS/TCS, sales summaries, invoices, and more. Just ask!"
                print(f"[CHAT MODEL] Capability response: {reason}")
                return {
                    "messages": [HumanMessage(content=user_query), AIMessage(content=reason)],
                    "memory_answer": reason,
                    "loop_count": loop_count + 1,
                }

            if query_type == "ambiguous":
                caps_text = _build_capability_text()
                ambig_prompt = (
                    "The user was unclear. Describe what you CAN help with "
                    "in 2-3 friendly sentences with specific examples:\n"
                    f"{caps_text}\n\n"
                    f"The user said: \"{original}\"\n"
                    f"This means: \"{canonical}\"\n\n"
                    "End by asking what they want to look up.\n"
                    f"CRITICAL — The user asked{lang_label}: \"{original}\". "
                    f"You MUST respond{lang_label}, matching the user's language, script, and register. "
                    f"If the user writes in a romanized script, you MUST respond in romanized form. "
                    f"{example_suffix} "
                    f"{lang_warning}\n"
                    "/no_think"
                )
                try:
                    resp = await summary_llm.ainvoke([
                        SystemMessage(content=ambig_prompt),
                        HumanMessage(content=user_query),
                    ])
                    reason = strip_think_tags((getattr(resp, "content", "") or "").strip())
                except Exception:
                    reason = ("I can help you with customers, stock levels, GST reports, TDS/TCS, "
                              "sales summaries, invoices, and more. What would you like me to look up?")
                print(f"[CHAT MODEL] Ambiguous response: {reason}")
                return {
                    "messages": [HumanMessage(content=user_query), AIMessage(content=reason)],
                    "memory_answer": reason,
                    "loop_count": loop_count + 1,
                }

            if query_type == "ood":
                caps_text = _build_capability_text()
                ood_prompt = (
                    "You are an ERP assistant. The user asked about something outside your domain.\n"
                    "You MUST refuse to answer. "
                    "Say you only handle business data and list what you can do. "
                    "End with a follow-up question.\n"
                    f"The user said: \"{original}\"\n"
                    f"This means: \"{canonical}\"\n\n"
                    f"CRITICAL — The user asked{lang_label}: \"{original}\". "
                    f"You MUST respond{lang_label}, matching the user's language, script, and register. "
                    f"If the user writes in a romanized script, you MUST respond in romanized form. "
                    f"{example_suffix} "
                    f"{lang_warning}\n"
                    "/no_think"
                )
                try:
                    resp = await summary_llm.ainvoke([
                        SystemMessage(content=ood_prompt),
                        HumanMessage(content=user_query),
                    ])
                    reason = strip_think_tags((getattr(resp, "content", "") or "").strip())
                except Exception:
                    reason = "I only work with business data. I can help you search customers, check stock, view GST reports, find invoices, and more. What would you like to look up?"
                print(f"[CHAT MODEL] OOD response: {reason}")
                return {
                    "messages": [HumanMessage(content=user_query), AIMessage(content=reason)],
                    "memory_answer": reason,
                    "loop_count": loop_count + 1,
                }

            if query_type == "conversational":
                history_lines = []
                if summary:
                    history_lines.append(f"- Previous conversation summary: {summary}")
                narrative = _build_memory_context(state.get("messages", []), max_exchanges=5)
                if narrative:
                    history_lines.append(f"- Recent conversation context:\n{narrative}")
                history_context = "\n".join(history_lines) if history_lines else ""

                conv_prompt = (
                    "You are a friendly ERP assistant having a casual chat. "
                    f"The user said: \"{original}\"\n"
                    f"This means: \"{canonical}\"\n\n"
                )
                if history_context:
                    conv_prompt += (
                        "Use the following conversation context to answer if the user asks about the history, "
                        "what they asked, what you answered, or references past details. Do not guess or make up details:\n"
                        f"{history_context}\n\n"
                    )
                conv_prompt += (
                    "Respond naturally in 1-2 sentences. Vary your responses. "
                    "End with a natural invitation to help.\n"
                    f"CRITICAL — The user asked{lang_label}: \"{original}\". "
                    f"You MUST respond{lang_label}, matching the user's language, script, and register. "
                    f"If the user writes in a romanized script, you MUST respond in romanized form. "
                    f"{example_suffix} "
                    f"{lang_warning}\n"
                    "/no_think"
                )
                resp = await summary_llm.ainvoke([
                    SystemMessage(content=conv_prompt),
                    HumanMessage(content=user_query),
                ])
                reason = strip_think_tags((getattr(resp, "content", "") or "").strip())
                print(f"[CHAT MODEL] Conversational response: {reason}")
                return {
                    "messages": [HumanMessage(content=user_query), AIMessage(content=reason)],
                    "memory_answer": reason,
                    "loop_count": loop_count + 1,
                }

            existing_memory_answer = state.get("memory_answer", "")
            if existing_memory_answer:
                print(f"[CHAT MODEL] Using pre-set memory_answer: {existing_memory_answer}")
                return {
                    "messages": [HumanMessage(content=user_query), AIMessage(content=existing_memory_answer)],
                    "memory_answer": existing_memory_answer,
                    "loop_count": loop_count + 1,
                }

            unsupported_reason = state.get("unsupported_reason")
            if unsupported_reason:
                reason = unsupported_reason
                print(f"[CHAT MODEL] Query unsupported, using fallback: {reason}")
                return {
                    "messages": [HumanMessage(content=user_query), AIMessage(content=reason)],
                    "memory_answer": reason,
                    "loop_count": loop_count + 1,
                }

            print("[CHAT MODEL] No available tools. Trying conversation memory...")
            previous_summary = state.get("summary", "") or ""
            conversation_context = state.get("conversation_context", {})
            has_messages = bool(state.get("messages"))
            if previous_summary or conversation_context or has_messages:
                detected_language = (state.get("detected_language") or "").strip().lower()
                if detected_language == "english":
                    lang_instruction = "Reply in natural English."
                elif detected_language == "hinglish":
                    lang_instruction = "Reply in Hinglish (Hindi mixed with English, written in Latin script/alphabets) like the user."
                elif detected_language == "hindi":
                    lang_instruction = "Reply in Hindi written in Latin script (e.g. Hinglish) like the user."
                elif detected_language in ("unknown", "auto", ""):
                    lang_instruction = "Reply in a friendly, conversational tone matching the user's language."
                else:
                    lang_instruction = f"Reply in {detected_language} (written in Latin/English alphabets — no native script characters) like the user."

                mem_prompt = (
                    "You are an ERP assistant. Answer based ONLY on the conversation history below. "
                    "Do not make up information. If the answer is not in the history, say so plainly. "
                    f"{lang_instruction} "
                    "NEVER mention tool names, API calls, or technical details. Use only Latin characters (a-z A-Z 0-9).\n\n"
                )
                if previous_summary:
                    mem_prompt += f"Conversation History:\n{previous_summary}\n\n"
                elif has_messages:
                    narrative = _build_memory_context(state.get("messages", []), max_exchanges=5)
                    if narrative:
                        mem_prompt += narrative
                if conversation_context:
                    entities = conversation_context.get("entities", [])
                    if entities:
                        mem_prompt += f"Known Entities:\n{json.dumps(entities, indent=2, ensure_ascii=False)}\n"
                mem_prompt += "\n/no_think"
                try:
                    mem_resp = await summary_llm.ainvoke([
                        SystemMessage(content=mem_prompt),
                        HumanMessage(content=user_query),
                    ])
                    reason = strip_think_tags((getattr(mem_resp, "content", "") or "").strip())
                except Exception as e:
                    print(f"[CHAT MODEL] Memory LLM error: {e}")
                    reason = state.get("unsupported_reason", "I can only answer ERP-related queries about customers, stock, GST, TDS, and TCS. Please ask a relevant business question.")
            else:
                reason = state.get("unsupported_reason", "I can only answer ERP-related queries about customers, stock, GST, TDS, and TCS. Please ask a relevant business question.")
            return {
                "messages": [HumanMessage(content=user_query), AIMessage(content=reason)],
                "memory_answer": reason,
                "loop_count": loop_count + 1,
            }

        step = now()
        prompt_start = time.perf_counter()
        system_prompt_text = build_system_prompt(
            user_query=user_query,
            original_query=state.get("original_query", ""),
            selected_tools=selected_tools,
            query_parts=query_parts,
            summary=summary,
            messages=state.get("messages", []),
            last_tool_call=state.get("last_tool_call"),
            conversation_context=state.get("conversation_context"),
        )
        prompt_duration = time.perf_counter() - prompt_start
        print(f"[4] Built system prompt: {prompt_duration:.3f}s")
        print("system_prompt_chars:", len(system_prompt_text))

        chat_history = [
            msg for msg in state.get("messages", [])
            if not isinstance(msg, SystemMessage)
        ]
        # keeps last 4 messages for conversation context
        if len(chat_history) > 4:
            print(f"[TRIM] Truncating chat history from {len(chat_history)} to last 4 messages")
            # Keep a summary context message if available
            if summary:
                chat_history = [SystemMessage(content=f"Previous conversation summary: {summary[:500]}")] + chat_history[-4:]
            else:
                chat_history = chat_history[-4:]
        for i, msg in enumerate(chat_history):
            if msg.type == "tool" and len(msg.content)>200:
                tool_name = msg.name or "tool"
                record_count = msg.content.count('"id":') or msg.content.count('"name":')
                limit_match = re.search(r'"limit":\s*(\d+)',msg.content)
                summary_text = f"Tool {tool_name} returned {record_count} record(s)"
                if limit_match:
                    summary_text += f" (limited to {limit_match.group(1)})"
                chat_history[i] = ToolMessage(content=summary_text,tool_call_id = msg.tool_call_id,name=tool_name)
                print(f"[COMPRESS] ToolMessage {tool_name}:{len(msg.content)} chars -> '{summary_text}'")
        system_prompt = SystemMessage(content=system_prompt_text + "\n\n/no_think")
        llm_input = (
            [system_prompt]
            + chat_history
            + [HumanMessage(content=user_query)]
        )


        all_raw_calls = []
        called_names = set()
        remaining_names = list(selected_tools)
        loop_input = llm_input
        retry_count = 0
        llm_bound_tool_names = set()

        while remaining_names:
            if retry_count >= 3:
                break
            if retry_count >= 2 and called_names:
                break
            retry_count += 1
            remaining_tools = [
                t for t in available_tools if t.name in remaining_names
            ]
            if not remaining_tools:
                break

            llm_bound_tool_names = {t.name for t in remaining_tools}

            step = now()
            print(f"[6] Invoking LLM with bind_tools (round {retry_count}, tools: {[t.name for t in remaining_tools]})...")

            # Debug: log the tool schemas being sent
            try:
                tool_schemas = _strip_schema_descriptions(
                    [convert_to_openai_tool(t, strict=False) for t in remaining_tools]
                )
                print(f"[BIND_TOOLS] model={llm.model}, tool_count={len(tool_schemas)}")
                print(f"[BIND_TOOLS] schemas={json.dumps(tool_schemas, indent=2)[:3000]}")
            except Exception as schema_e:
                print(f"[BIND_TOOLS] Error building schema preview: {schema_e}")

            try:
                response = await llm.bind_tools(remaining_tools, strict=False).ainvoke(loop_input)
            except Exception as e:
                print(f"[RETRY ERROR] {e} — preserving round {retry_count - 1} results")
                print(f"[RETRY ERROR TRACEBACK] {traceback.format_exc()}")

                break
            log_token_usage(response,"chat_model")

            raw_tool_calls = getattr(response, "tool_calls", None) or []
            for call in raw_tool_calls:
                name = call.get("name", "")
                if name:
                    called_names.add(name)
                if not any(
                    c.get("name") == name and c.get("args") == call.get("args")
                    for c in all_raw_calls
                ):
                    all_raw_calls.append(call)

            remaining_names = [
                n for n in selected_tools if n not in called_names
            ]
            remaining_names = [
                n for n in remaining_names
                if n in TOOL_INTENT_REGISTRY and (
                    not (set(TOOL_INTENT_REGISTRY[n].get("keywords", []))
                          | set(TOOL_INTENT_REGISTRY[n].get("aliases", [])))
                    or any(kw in user_query.lower() for kw in
                           set(TOOL_INTENT_REGISTRY[n].get("keywords", []))
                           | set(TOOL_INTENT_REGISTRY[n].get("aliases", [])))
                )
            ]
            if not remaining_names:
                break

            query_type = (state.get("query_type") or "").strip()
            is_meta = query_type == "conversational" or any(
                re.search(p, user_query, re.IGNORECASE) for p in META_QUESTION_PATTERNS_GLOBAL
            )
            if is_meta:
                print(f"[RETRY] Skipping retry — conversational query: {user_query}")
                break

            print(f"[RETRY] Missing tool calls for: {remaining_names}")
            loop_input = (
                [llm_input[0]]
                + [HumanMessage(content=user_query)]
                + [response]
                + [HumanMessage(content=f"Call remaining tools: {', '.join(remaining_names)}. Do NOT repeat already-called tools.")]
            )

        if not all_raw_calls:
            print("[FALLBACK] No tool calls after retries — forcing LLM to pick closest tool")
            remaining_tools = available_tools
            llm_bound_tool_names = {t.name for t in remaining_tools}
            fallback_msg = HumanMessage(
                content=f"The user asked: {user_query}\n\n"
                        f"Available tools: {', '.join(t.name for t in remaining_tools)}\n\n"
                        f"None of these tools perfectly match, but you MUST pick the MOST RELEVANT one and call it. "
                        f"Do NOT refuse. Choose the tool whose purpose best aligns with: {user_query}"
            )
            try:
                print(f"[FALLBACK BIND_TOOLS] tools={[t.name for t in remaining_tools]}")
                try:
                    fb_schemas = [convert_to_openai_tool(t, strict=False) for t in remaining_tools]
                    print(f"[FALLBACK BIND_TOOLS] schemas={json.dumps(fb_schemas, indent=2)[:2000]}")
                except Exception as fb_schema_e:
                    print(f"[FALLBACK BIND_TOOLS] schema preview error: {fb_schema_e}")
                fallback_response = await llm.bind_tools(remaining_tools, strict=False).ainvoke([
                    llm_input[0], HumanMessage(content=user_query), fallback_msg
                ])
                fallback_calls = getattr(fallback_response, "tool_calls", None) or []
                for call in fallback_calls:
                    name = call.get("name", "")
                    if name:
                        called_names.add(name)
                    all_raw_calls.append(call)
            except Exception as e:
                print(f"[FALLBACK ERROR] {e}")
                print(f"[FALLBACK ERROR TRACEBACK] {traceback.format_exc()}")
                print("[FALLBACK PROBE] Sending without bind_tools...")
                try:
                    fb_probe = await llm.ainvoke([llm_input[0], HumanMessage(content=user_query), fallback_msg])
                    print(f"[FALLBACK PROBE] OK: type={type(fb_probe).__name__}, content={repr(str(getattr(fb_probe, 'content', ''))[:200])}")
                except Exception as fb_probe_e:
                    print(f"[FALLBACK PROBE] FAILED: {fb_probe_e}")
                fallback_calls = []
            if fallback_calls:
                print(f"[FALLBACK] LLM produced {len(fallback_calls)} tool call(s): {[c.get('name') for c in fallback_calls]}")
                response = fallback_response
            else:
                print("[FALLBACK] LLM still refused — will return empty")
                last_tc = state.get("last_tool_call", {})
                for tool_name, tool_args in last_tc.items():
                    if tool_name in {t.name for t in available_tools}:
                        all_raw_calls.append({
                            "name": tool_name,
                            "args": tool_args,
                            "id": f"call_lr_{tool_name}_{uuid.uuid4().hex[:8]}",
                            "type": "tool_call",
                        })
                        called_names.add(tool_name)
                        print(f"[FALLBACK] Last resort: reused last tool call: {tool_name}")
                        break
                response = AIMessage(content="", tool_calls=list(all_raw_calls))

        # --- SIMPLIFIED FORCE-INJECT (zero-regex migration) ---
        # Just call remaining selected tools with empty args if LLM didn't call them
        for tn in selected_tools:
            if tn in called_names:
                continue
            if tn in llm_bound_tool_names:
                print(f"[FORCE-INJECT] Skipping {tn} — was available to LLM but not called")
                continue
            all_raw_calls.append({
                "name": tn,
                "args": {},
                "id": f"call_force_{tn}_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            })
            called_names.add(tn)
            print(f"[FORCE-INJECT] Adding missing tool: {tn}")

        raw_tool_calls = all_raw_calls
        tool_calls = []

        def _apply_repair(name, args, user_query):
            meta = TOOL_INTENT_REGISTRY.get(name, {})
            repair = meta.get("repair")
            if not repair:
                return {"name": name, "args": args}
            combined_q = f"{original_query or ''} {state.get('canonical_query', '') or ''}".lower()
            worker_extra = {}
            if args:
                for k, v in args.items():
                    if v is not None:
                        worker_extra[k] = v
            if repair.get("overwrite"):
                new_args = dict(repair.get("base_args", {}))
                if args:
                    for k, v in args.items():
                        if v not in (None, ""):
                            new_args[k] = v
            else:
                new_args = dict(args or {})
            for kw, kwar in repair.get("keyword_args", {}).items():
                if kw.lower() in combined_q:
                    new_args.update(kwar)
            param_aliases = repair.get("param_aliases", {})
            for llm_arg, real_param in param_aliases.items():
                if llm_arg in new_args and real_param not in new_args:
                    new_args[real_param] = new_args.pop(llm_arg)
            for k, v in worker_extra.items():
                if k not in new_args or new_args.get(k) in (None, "", []):
                    new_args[k] = v
            if "limit" in new_args and isinstance(new_args.get("limit"), int):
                intent = state.get("query_intent", "sample")
                intent_limits = {
                    "count": 10000,
                    "aggregate": 10000,
                    "list_all": 10000,
                    "comparison": 1000,
                    "detail": 200,
                    "sample": 100,
                    "extreme": 1,
                }
                target = intent_limits.get(intent, 50)
                if new_args["limit"] < target:
                    new_args["limit"] = target
                    print(f"[LIMIT] {name}: {intent} intent — raised limit to {target}")
            if name == "get_gst_summary" or name in ("get_tds_outstanding", "get_tcs_outstanding"):
                print(f"[{name.upper()} FINAL ARGS] {json.dumps(new_args, default=str)}")
            return {"name": name, "args": new_args}

        def _strip_unknown_params(tool_name: str, tool_args: dict) -> dict:
            t = tools_dict[tool_name]
            schema = t.args_schema
            valid = set(schema.model_fields.keys()) if schema and hasattr(schema, 'model_fields') else set()
            cleaned = {}
            for k, v in tool_args.items():
                if k in valid:
                    if v is None and schema and k in schema.model_fields:
                        field = schema.model_fields[k]
                        if not field.is_required():
                            cleaned[k] = field.default
                        else:
                            cleaned[k] = v
                    else:
                        cleaned[k] = v
                else:
                    print(f"[STRIP] {tool_name}: removing unknown param '{k}'")
            return cleaned

        def _repair_tool_call(name: str, args: dict) -> dict | None:
            # Unwrap tool/parameters wrapper format (Qwen non-standard tool call format)
            if "parameters" in args and isinstance(args["parameters"], dict):
                if "tool" in args and isinstance(args["tool"], str):
                    name = TOOL_NAME_ALIASES.get(args["tool"], args["tool"])
                args = args["parameters"]
            elif "arguments" in args and isinstance(args["arguments"], dict):
                if "tool" in args and isinstance(args["tool"], str):
                    name = TOOL_NAME_ALIASES.get(args["tool"], args["tool"])
                args = args["arguments"]

            name = TOOL_NAME_ALIASES.get(name, name)
            if name not in tools_dict:
                return None
            for alias, canonical in [
                ("date_from", "from_date"), ("date_to", "to_date"),
                ("startDate", "from_date"), ("endDate", "to_date"),
                ("fromDate", "from_date"), ("toDate", "to_date"),
                ("start_date", "from_date"), ("end_date", "to_date"),
            ]:
                if alias in args and canonical not in args:
                    args[canonical] = args.pop(alias)
            result = _apply_repair(name, args, original_query)
            if result:
                # Move unknown params into filters (LLM often passes filter fields at top level)
                rargs = result["args"]
                schema = tools_dict[name].args_schema
                valid_params = set(schema.model_fields.keys()) if schema and hasattr(schema, 'model_fields') else set()
                unknown_params = {k: v for k, v in list(rargs.items()) if k not in valid_params}
                if unknown_params:
                    for k in unknown_params:
                        rargs.pop(k, None)
                    existing_filters = rargs.get("filters", {}) or {}
                    existing_filters.update(unknown_params)
                    rargs["filters"] = existing_filters
                    print(f"[PARAM-TO-FILTERS] {name}: moved unknown params into filters: {unknown_params}")
                result["args"] = _strip_unknown_params(name, result["args"])
                rargs = result["args"]
                filters_dict = rargs.get("filters")
                if isinstance(filters_dict, dict):
                    for fk, fv in filters_dict.items():
                        if isinstance(fv, str):
                            filters_dict[fk] = fv.strip()
            return result

        for call in raw_tool_calls:
            name = call.get("name", "")
            args = dict(call.get("args", {}))
            repaired = _repair_tool_call(name, args)
            if repaired:
                tool_calls.append({
                    "name": repaired["name"],
                    "args": repaired["args"],
                    "id": f"call_{repaired['name']}_{uuid.uuid4().hex[:12]}",
                    "type": "tool_call",
                })

        deduped = {}
        for call in tool_calls:
            n = call["name"]
            a = call["args"]
            key = json.dumps({"name": n, "args": a}, sort_keys=True, default=str)
            if key not in deduped:
                deduped[key] = call
            else:
                existing = deduped[key]["args"]
                existing_filled = sum(1 for v in existing.values() if v not in ("", None, [], {}))
                new_filled = sum(1 for v in a.values() if v not in ("", None, [], {}))
                if new_filled > existing_filled:
                    deduped[key] = call
        if len(deduped) < len(tool_calls):
            tool_calls = list(deduped.values())

        sanitized = [call for call in tool_calls if sanitize_tool_filters(call["name"], call["args"]) is not None]
        tool_calls = sanitized

        if tool_calls:
            seen = set()
            unique_calls = []
            for call in tool_calls:
                key = json.dumps(
                    {"name": call["name"], "args": call["args"]},
                    sort_keys=True,
                    default=str,
                )
                if key not in seen:
                    seen.add(key)
                    unique_calls.append(call)
            final_calls = []
            seen_name_args = set()
            for call in unique_calls:
                n = call["name"]
                meta = TOOL_INTENT_REGISTRY.get(n, {})
                a_key = json.dumps(call["args"], sort_keys=True, default=str)
                call_key = (n, a_key)
                if meta.get("multi_call_ok"):
                    final_calls.append(call)
                elif call_key not in seen_name_args:
                    seen_name_args.add(call_key)
                    final_calls.append(call)
            tool_calls = final_calls
            response = AIMessage(
                content=response.content or "",
                tool_calls=tool_calls,
                additional_kwargs=response.additional_kwargs,
                response_metadata=response.response_metadata,
                id=response.id,
            )


        memory_answer = ""

        return {
            "messages": [HumanMessage(content=user_query), response],
            "memory_answer": memory_answer,
            "loop_count": loop_count + 1,
        }

    except Exception as e:
        print(f"[CHAT MODEL ERROR]: {e}")
        print(f"[TOTAL chat_model_node before error]: {sec(node_start)}s")
        return {
            "messages": [
                HumanMessage(content=state.get("user_query", "")),
                AIMessage(content="Chat model error: The model encountered an issue while processing your request. Please try again."),
            ],
            "memory_answer": "",
            "loop_count": state.get("loop_count", 0) + 1,
        }
