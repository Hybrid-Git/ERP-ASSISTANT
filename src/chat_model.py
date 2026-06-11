import json
import re
import time
import uuid
from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.schema import MainState
from src.tools_api import tools_dict
from src.tool_doc import TOOL_INTENT_REGISTRY, TOOL_NAME_ALIASES
from src.config import llm, summary_llm
from src.utils import (
    now, sec, log_token_usage, print_ollama_metadata, sanitize_tool_filters,
    TOOL_DOMAINS, INVOICE_NO_PATTERNS, INVOICE_TOOLS,
    extract_date_range_for_tool,
)
from src.semantic_search import classify_domains
from src.prompts import META_QUESTION_PATTERNS_GLOBAL, _REFERENCE_PATTERN


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
        "You are an ERP assistant. Use the available tools to answer the user.",
        'Preserve all query text literally. Do not reinterpret or assume intent. If user says "mars" use "mars", not March.',
        "Never invent IDs, names, dates, or amounts.",
        "You MUST call at least one tool. Never answer in prose without a tool call.",
        "Do NOT output any thinking or reasoning — call the tool directly.",
        "",
        "STRICT ANTI-HALLUCINATION RULES:",
        "- CRITICAL: Do NOT search for quantity/scope words. Never pass words like 'sara', 'sab', 'all', 'every', 'each', 'saare', 'poora', 'pure', 'sabhi', 'saari', 'sare', 'sari' as the 'search' or 'term' parameter. These describe scope ('all'/'every'), NOT entity names. If the user asks for 'all records', call the list tool without a search filter.",
        "- CRITICAL: Never invent entity identifiers. Never pass an ID, name, or search term that you made up or inferred from non-entity words. Only use IDs/names that you have actually seen in prior tool results. If you lack a real ID, call a list/search tool first and extract the ID from its results — do not guess.",
        "- CRITICAL: When the user's query has been updated to include specific entity names (e.g., resolved from pronouns like 'dono'), use ONLY those exact names as your search/term parameters. Do NOT use entity names from other previous tool results (e.g., top_customers, top_vendors) unless they match the resolved names. For example, if the resolved query says 'NYKAA... aur Nykaa Warehouse...', search for THOSE names only — NOT for B2CTELANGANA or any other name from get_top_customer.",
        "",
        "FOLLOW-UP RULES:",
        "- If the answer is already in previous tool results, use it directly without a new API call.",
        "- When a tool has sort_field/sort_order parameters and the user asks for extreme/comparative values (highest, most, least, top, bottom, etc.), ALWAYS set sort_field to the field being compared and sort_order accordingly: 'desc' for highest/most/top, 'asc' for lowest/least/bottom.",
        "- CRITICAL: When the current query requires a DIFFERENT tool than the previous one (e.g. switching from get_customer to get_stock_levels), you MUST clear ALL old search terms and parameters. Reuse parameters ONLY within the same tool.",
    ]

    check_query = f"{original_query} {user_query}" if original_query else user_query
    is_follow_up = bool(_REFERENCE_PATTERN.search(check_query)) or len(user_query.split()) <= 3
    if messages and is_follow_up:
        recent_calls = _get_recent_tool_calls(messages)
        if recent_calls:
            lines.append("")
            lines.append("--- RECENT TOOL CALLS (for follow-up context) ---")
            for call in recent_calls:
                lines.append(f"Tool: {call['name']}")
                lines.append(f"Args: {json.dumps(call['args'], indent=2)}")
            lines.append("For follow-up queries, reuse these same parameters. Only change what the user explicitly asks about.")
            lines.append("--------------------------------------------------")
        if summary:
            lines.append("")
            lines.append("--- PREVIOUS CONVERSATION CONTEXT ---")
            lines.append(summary)
            lines.append("--------------------------------------------------")
        if conversation_context:
            entities = conversation_context.get("entities", [])
            if entities:
                lines.append("")
                lines.append("--- KNOWN ENTITIES ---")
                names = []
                for e in entities[-3:]:
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
                lines.append(json.dumps(capped_extra, indent=2))
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
    lines.append("FAILURE-AWARE RESPONSE:")
    lines.append("  If a tool call returns empty results, do NOT hallucinate data. Report clearly: 'No records found for X.'")
    lines.append("  If the user asks about something from earlier that returned no data, acknowledge the prior failure.")
    lines.append("")
    lines.append("PARAMETER RULE:")
    lines.append("  NEVER copy parameters between different tools. Each tool has its own unique set of valid parameters.")
    return "\n".join(lines)


@traceable(name="chat_model_node", run_type="llm")
async def chat_model_node(state: MainState):
    node_start = now()
    try:
        print("\n========== CHAT MODEL NODE START ==========")
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
        print(f"[1] Read state: {sec(step)}s")
        print("user_query:", user_query)
        print("selected_tools:", selected_tools)
        print("query_parts:", query_parts)
        print("loop_count:", loop_count)

        step = now()
        available_tools = [
            tools_dict[name]
            for name in selected_tools
            if name in tools_dict
        ]
        print(f"[2] Loaded available tools: {sec(step)}s")
        print("available_tool_names:", [tool.name for tool in available_tools])

        if not available_tools:
            query_type = (state.get("query_type") or "").strip()
            if query_type == "greeting":
                greeting_prompt = (
                    "You are an ERP assistant. The user just greeted you.\n"
                    "Respond warmly and naturally like a friendly human. "
                    "Vary your greeting each time — don't repeat the same words. "
                    "You can say hi hello namaste, welcome them, and briefly offer help. "
                    "Keep it to 1-2 short sentences. Be warm, not robotic.\n"
                    "Do NOT mention tools, APIs, or technical details. "
                    "Speak in the same language the user used (English or Hinglish).\n"
                    "/no_think"
                )
                try:
                    resp = await summary_llm.ainvoke([
                        SystemMessage(content=greeting_prompt),
                        HumanMessage(content=state.get("original_query", "")),
                    ])
                    reason = (getattr(resp, "content", "") or "").strip()
                except Exception:
                    reason = "Hello! How can I help you with your ERP data today?"
                print(f"[CHAT MODEL] Greeting response: {reason}")
                return {
                    "messages": [HumanMessage(content=user_query), AIMessage(content=reason)],
                    "memory_answer": reason,
                    "loop_count": loop_count + 1,
                }

            if query_type == "capability":
                tool_descriptions = []
                for tname, tmeta in TOOL_INTENT_REGISTRY.items():
                    desc = tmeta.get("description", "")
                    aliases = tmeta.get("aliases", [])
                    alias_str = ", ".join(aliases[:3])
                    tool_descriptions.append(f"- {alias_str}: {desc}")
                tools_text = "\n".join(tool_descriptions)
                cap_prompt = (
                    "You are an ERP assistant. The user asked about what you can do.\n"
                    "Describe your capabilities conversationally, like a helpful human.\n"
                    "Here are the tools/features available to you:\n"
                    f"{tools_text}\n\n"
                    "Explain in a natural, friendly way — not as a list of technical tools. "
                    "Say something like 'I can help you look up customers, check stock levels, "
                    "view GST reports, find outstanding invoices, and more.' "
                    "Keep it to 2-4 sentences. Be inviting and conversational. "
                    "Speak in the same language as the user (English or Hinglish).\n"
                    "Do NOT mention tool names, APIs, or technical details.\n"
                    "/no_think"
                )
                try:
                    resp = await summary_llm.ainvoke([
                        SystemMessage(content=cap_prompt),
                        HumanMessage(content=state.get("original_query", "")),
                    ])
                    reason = (getattr(resp, "content", "") or "").strip()
                except Exception:
                    reason = "I can help you with customer details, stock levels, GST reports, TDS/TCS, sales summaries, invoices, and more. Just ask!"
                print(f"[CHAT MODEL] Capability response: {reason}")
                return {
                    "messages": [HumanMessage(content=user_query), AIMessage(content=reason)],
                    "memory_answer": reason,
                    "loop_count": loop_count + 1,
                }

            if query_type == "ood":
                ood_prompt = (
                    "You are an ERP assistant. The user asked something OUTSIDE your domain.\n"
                    "CRITICAL: Do NOT answer the user's question. You do NOT have this information.\n"
                    "Instead, politely refuse and say you can only help with ERP-related "
                    "business queries (customers, stock, GST, TDS, TCS, invoices, sales, etc.).\n"
                    "Example: 'Sorry, I can only assist with ERP-related queries like customers, stock, GST, and invoices.'\n"
                    "Be friendly — don't sound robotic or defensive. "
                    "Suggest what you CAN help with. "
                    "Keep it to 1-2 short sentences. "
                    "Speak in the same language as the user (English or Hinglish).\n"
                    "Do NOT answer the question. Do NOT provide any information about the topic.\n"
                    "/no_think"
                )
                try:
                    resp = await summary_llm.ainvoke([
                        SystemMessage(content=ood_prompt),
                        HumanMessage(content=state.get("original_query", "")),
                    ])
                    reason = (getattr(resp, "content", "") or "").strip()
                except Exception:
                    reason = "I'm an ERP assistant — I can help with customers, stock, GST, TDS, invoices, and sales data. Could you ask about any of these?"
                print(f"[CHAT MODEL] OOD response: {reason}")
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
                mem_prompt = (
                    "You are an ERP assistant. Answer based ONLY on the conversation history below. "
                    "Do not make up information. If the answer is not in the history, say so plainly. "
                    "Reply in natural Hinglish (Hindi+English) like the user. NEVER mention tool names, API calls, or technical details.\n\n"
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
                    reason = (getattr(mem_resp, "content", "") or "").strip()
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
        print("[3] Using bind_tools")
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
        system_prompt = SystemMessage(content=system_prompt_text + "\n\n/no_think")
        llm_input = (
            [system_prompt]
            + chat_history
            + [HumanMessage(content=user_query)]
        )
        print(f"[5] Built LLM input messages: {sec(prompt_start)}s")
        print("message_count:", len(llm_input))
        print("message_types:", [type(m).__name__ for m in llm_input])

        all_raw_calls = []
        called_names = set()
        remaining_names = list(selected_tools)
        loop_input = llm_input
        retry_count = 0

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

            step = now()
            print(f"[6] Invoking LLM with bind_tools (round {retry_count}, tools: {[t.name for t in remaining_tools]})...")
            response = await llm.bind_tools(remaining_tools).ainvoke(loop_input)
            print(f"[6] LLM invoke completed: {sec(step)}s")
            log_token_usage(response, "chat_model")

            print("\n========== RAW WORKER RESPONSE DEBUG ==========")
            print("response_type:", type(response).__name__)
            print("content:", repr(getattr(response, "content", "")))
            print("tool_calls:", getattr(response, "tool_calls", None))
            print("additional_kwargs:", getattr(response, "additional_kwargs", {}))
            print("response_metadata:", getattr(response, "response_metadata", {}))
            print("==============================================\n")
            print_ollama_metadata(response)

            raw_tool_calls = getattr(response, "tool_calls", None) or []
            for call in raw_tool_calls:
                name = call.get("name", "")
                if name:
                    called_names.add(name)
                all_raw_calls.append(call)

            remaining_names = [
                n for n in selected_tools if n not in called_names
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
                + llm_input[1:-1]
                + [HumanMessage(content=user_query)]
                + [response]
                + [HumanMessage(content=f"You still need to call the following tool(s): {', '.join(remaining_names)}. Call them now.")]
            )

        if not all_raw_calls:
            print("[FALLBACK] No tool calls after retries — forcing LLM to pick closest tool")
            remaining_tools = available_tools
            fallback_msg = HumanMessage(
                content=f"The user asked: {user_query}\n\n"
                        f"Available tools: {', '.join(t.name for t in remaining_tools)}\n\n"
                        f"None of these tools perfectly match, but you MUST pick the MOST RELEVANT one and call it. "
                        f"Do NOT refuse. Choose the tool whose purpose best aligns with: {user_query}"
            )
            fallback_response = await llm.bind_tools(remaining_tools).ainvoke([
                llm_input[0], HumanMessage(content=user_query), fallback_msg
            ])
            fallback_calls = getattr(fallback_response, "tool_calls", None) or []
            for call in fallback_calls:
                name = call.get("name", "")
                if name:
                    called_names.add(name)
                all_raw_calls.append(call)
            if fallback_calls:
                print(f"[FALLBACK] LLM produced {len(fallback_calls)} tool call(s): {[c.get('name') for c in fallback_calls]}")
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

        query_parts = state.get("query_parts", [original_query])
        combined_q = (f"{original_query or ''} {state.get('canonical_query', '') or ''}").lower()
        doc_override = state.get("document_type", "").replace("_invoice", "").replace("_", "")
        query_invoice_no = None
        for pat in INVOICE_NO_PATTERNS:
            m = re.search(pat, combined_q, re.IGNORECASE)
            if m:
                query_invoice_no = m.group(0)
                break

        for qp in query_parts:
            qp_domains, _ = classify_domains(qp, state.get("resolved_entities"))
            if doc_override:
                qp_domains.add(doc_override)
            for tn in selected_tools:
                if tn in called_names:
                    continue
                td = set(TOOL_DOMAINS.get(tn, []))
                if td and qp_domains and not td & qp_domains:
                    continue
                if not qp_domains:
                    meta = TOOL_INTENT_REGISTRY.get(tn, {})
                    all_kw = set(meta.get("keywords", [])) | set(meta.get("aliases", []))
                    if not any(kw in qp.lower() for kw in all_kw):
                        continue
                inject_args = {}
                if tn in INVOICE_TOOLS and query_invoice_no:
                    inject_args = {"filters": {"invoiceNo": query_invoice_no}}
                all_raw_calls.append({
                    "name": tn,
                    "args": inject_args,
                    "id": f"call_force_{tn}_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                })
                called_names.add(tn)
                print(f"[FORCE-INJECT] Adding missing tool: {tn} for query part '{qp}' (args: {inject_args})")

        raw_tool_calls = all_raw_calls
        tool_calls = []

        def _apply_repair(name, args, user_query):
            meta = TOOL_INTENT_REGISTRY.get(name, {})
            repair = meta.get("repair")
            if not repair:
                return {"name": name, "args": args}
            combined_q = f"{original_query or ''} {state.get('canonical_query', '') or ''}".lower()
            worker_has = {}
            if args:
                for dk in ("from_date", "to_date"):
                    v = args.get(dk)
                    if v and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(v)):
                        worker_has[dk] = v
            summary_text = state.get("summary", "") or ""
            messages_list = state.get("messages", [])
            recent_tool_dates = ""
            for msg in reversed(messages_list[:-1]):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                        for dk in ("from_date", "to_date"):
                            dv = tc_args.get(dk, "") if isinstance(tc_args, dict) else ""
                            if dv:
                                recent_tool_dates += " " + str(dv)
                    if recent_tool_dates.strip():
                        break
            if worker_has and not re.search(r"\d{4}-\d{2}-\d{2}|\b\d{4}\b", combined_q + " " + summary_text + recent_tool_dates):
                worker_has = {}
            worker_extra = {}
            if args:
                for k, v in args.items():
                    if k not in ("from_date", "to_date") and v is not None:
                        worker_extra[k] = v
            if repair.get("overwrite"):
                new_args = dict(repair.get("base_args", {}))
                if args:
                    for k, v in args.items():
                        if v not in (None, ""):
                            new_args[k] = v
            else:
                new_args = dict(args or {})
            for dk, dv in worker_has.items():
                new_args[dk] = dv
            for kw, kwar in repair.get("keyword_args", {}).items():
                if kw.lower() in combined_q:
                    new_args.update(kwar)
            if repair.get("extract_customer_id"):
                cm = re.search(r"(?:customer|party|client)\s*(?:id|number|no|#)?\s*[:#-]?\s*(\d+)", combined_q)
                if cm:
                    new_args["customer_id"] = int(cm.group(1))
            date_kws = repair.get("date_keywords")
            if date_kws and (not new_args.get("from_date") or not new_args.get("to_date")):
                f, t = extract_date_range_for_tool(combined_q, date_kws)
                if f:
                    new_args["from_date"] = f
                    new_args["to_date"] = t
            DATE_REQUIRED_TOOLS = {"get_gst_summary", "get_tds_outstanding", "get_tcs_outstanding"}
            if name in DATE_REQUIRED_TOOLS:
                has_dates = bool(new_args.get("from_date")) and bool(new_args.get("to_date"))
                if not has_dates:
                    print(f"[{name.upper()}] No date range found — signalling clarification needed")
                    return {"name": name, "args": new_args, "_needs_date_range": True}
            low_stock_kws = repair.get("low_stock_only_keywords")
            if low_stock_kws and new_args.get("low_stock_only") is True:
                if not any(kw in combined_q for kw in low_stock_kws):
                    new_args["low_stock_only"] = False
            if name == "get_customer":
                cust_num = re.search(r"(?:customer|party|client)\s*(?:number|no|#)?\s*:?\s*(\d+)", combined_q)
                if cust_num and not new_args.get("search"):
                    new_args["search"] = cust_num.group(1)
            if name in INVOICE_TOOLS:
                inv_no = new_args.pop("invoice_no", None)
                if not inv_no:
                    for pat in INVOICE_NO_PATTERNS:
                        m = re.search(pat, combined_q, re.IGNORECASE)
                        if m:
                            inv_no = m.group(0)
                            print(f"[REPAIR] {name}: filled invoice_no='{inv_no}' from query")
                            break
                if inv_no:
                    existing_filters = new_args.get("filters", {}) or {}
                    existing_filters["invoiceNo"] = inv_no
                    new_args["filters"] = existing_filters
                    new_args["limit"] = 5000
                    new_args["sort_by"] = "invoiceDate"
                    new_args["sort_order"] = "desc"
                    if new_args.get("customer_id") or new_args.get("party_id"):
                        if not re.search(r'\b(customer|party|client|vendor|ledger)\b', combined_q, re.IGNORECASE):
                            new_args.pop("customer_id", None)
                            new_args.pop("party_id", None)
                            print(f"[CROSS-VALIDATE] {name}: stripped customer/party ID (invoice_no present, query doesn't mention customer)")
            param_aliases = repair.get("param_aliases", {})
            for llm_arg, real_param in param_aliases.items():
                if llm_arg in new_args and real_param not in new_args:
                    new_args[real_param] = new_args.pop(llm_arg)
            for k, v in worker_extra.items():
                if k not in new_args or new_args.get(k) in (None, "", []):
                    new_args[k] = v
            if re.search(r'\b(sabse\s+kam|sabse\s+jya[dz]a|sabse\s+zyada|least|most|lowest|highest|minimum|maximum)\b', combined_q) and not re.search(r'\b(top\s+\d+|first\s+\d+|last\s+\d+)\b', combined_q):
                if "limit" not in new_args or new_args.get("limit", 10) > 5:
                    new_args["limit"] = 1
            if "limit" in new_args and isinstance(new_args.get("limit"), int) and new_args["limit"] <= 10:
                if not re.search(r'\b(sabse\s+kam|sabse\s+jya[dz]a|sabse\s+zyada|least|most|lowest|highest|minimum|maximum|top\s+\d+|first\s+\d+|last\s+\d+)\b', combined_q):
                    new_args["limit"] = 50
                    print(f"[LIMIT] {name}: raised limit from 10 to 50")
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

        needs_date_clarification = []

        def _repair_tool_call(name: str, args: dict) -> dict | None:
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
                if result.get("_needs_date_range"):
                    needs_date_clarification.append(result["name"])
                    return None
                result["args"] = _strip_unknown_params(name, result["args"])
                # Normalize filter values and search params
                rargs = result["args"]
                for nk in ("search", "search_term", "term"):
                    if nk in rargs and isinstance(rargs[nk], str):
                        rargs[nk] = rargs[nk].strip()
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
            print(f"[DEDUP] tool_calls: {len(tool_calls)} -> {len(deduped)}")
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
            print(f"[FIX] Extracted {len(tool_calls)} tool call(s) from bind_tools")

        print("\n========== WORKER LLM RESPONSE ==========")
        print("response_type:", type(response).__name__)
        print("tool_call_count:", len(tool_calls))
        print("tool_calls:", tool_calls)
        print("========================================\n")
        for i, call in enumerate(tool_calls, start=1):
            print(f"\n--- Tool Call {i} ---")
            print("name:", call.get("name"))
            print("args:")
            print(json.dumps(call.get("args", {}), indent=2, ensure_ascii=False))

        print(f"[TOTAL chat_model_node]: {sec(node_start)}s")
        print("========== CHAT MODEL NODE END ==========\n")

        memory_answer = ""
        if needs_date_clarification and tool_calls:
            non_date_tools = [tc for tc in tool_calls if tc["name"] not in needs_date_clarification]
            if not non_date_tools:
                tools_str = ", ".join(needs_date_clarification)
                memory_answer = (
                    f"Main aapki madad ke liye [**{needs_date_clarification[0]}**] tool use karna chahta hoon, "
                    f"lekin iske liye date range (from_date / to_date) chahiye. "
                    f"Kya aap kripya karke starting aur ending date bata sakte hain?\n"
                    f"Jaise: '1 April 2025 se 31 March 2026 ka data chahiye'"
                )
                print(f"[DATE NEAR] All tools need date range: {tools_str}")
            else:
                print(f"[DATE NEAR] Skipping memory_answer — {len(non_date_tools)} tool(s) have data: {[t['name'] for t in non_date_tools]}")

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
