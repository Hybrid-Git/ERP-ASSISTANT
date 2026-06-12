import json
import re
from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage
from src.schema import MainState
from src.config import summary_llm
from src.utils import LIST_WORDS, TOOL_DOMAINS, strip_think_tags, log_token_usage
from src.prompts import HINGLISH_PRONOUNS
from src.deterministic_final import make_summary


def _clean_llm_response(text: str) -> str:
    text = re.sub(r'```(?:json)?\s*\n.*?\n```', '', text, flags=re.DOTALL)
    text = text.strip()
    meta_prefixes = (
        "based on the provided", "based on the data", "based on the tool",
        "here are the key", "here is the summary", "here is the detail",
        "from the tool results", "from the provided", "according to the tool",
        "the tool results show", "the data shows that", "the following are",
        "below are the", "i have analyzed the", "after reviewing the",
        "in response to your",
    )
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip().lower()
        if any(stripped.startswith(p) for p in meta_prefixes) and len(stripped) < 80:
            continue
        if stripped in ("json", "```json", "```", "````"):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned).strip()
    text = re.sub(r'```\s*$', '', text).strip()
    text = re.sub(r'\{[^}]*"[^}]*"[^}]*\}', '', text).strip()
    if not text or not text.strip("` \n\r"):
        return ""
    return text


def _strip_hallucinated_values(text: str, tool_data: dict, original_query: str = "") -> str:
    actual_fields = set()
    for _tool_name, records in tool_data.items():
        if isinstance(records, list):
            for rec in records:
                if isinstance(rec, dict):
                    for k in rec:
                        actual_fields.add(k.lower())
    if not actual_fields:
        return text
    query_tokens = set(re.findall(r'\b\w+\b', original_query.lower()))
    field_like = re.compile(r'\b[a-z]+[A-Z][a-zA-Z]*\b|\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b|\b[a-zA-Z]+_[a-zA-Z]+\b')
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    cleaned = []
    for sent in sentences:
        if not sent.strip():
            continue
        words = set(m.group(0).lower() for m in field_like.finditer(sent))
        hallucinated = any(
            w not in actual_fields and w not in query_tokens
            for w in words
        )
        if hallucinated:
            print(f"[HALLUCINATION] Stripped sentence containing invented field(s): {words - actual_fields} | sentence: {sent.strip()[:100]}")
        else:
            cleaned.append(sent)
    result = " ".join(cleaned).strip()
    if not result:
        print(f"[HALLUCINATION] All sentences stripped — returning empty")
    return result


@traceable(name="response_generation_node", run_type="chain")
async def response_generation_node(state: MainState):
    memory_answer = state.get("memory_answer", "")
    final_response = state.get("final_response", {})
    if memory_answer:
        query_type = state.get("query_type", "")
        if query_type in ("capability", "ambiguous", "greeting", "ood", "conversational"):
            return {"response_text": memory_answer, "memory_answer": ""}
        tool_data = final_response.get("data", {}) if isinstance(final_response, dict) else {}
        has_real_data = any(
            isinstance(recs, list) and len(recs) > 0
            for recs in tool_data.values()
        )
        if not has_real_data:
            return {"response_text": memory_answer, "memory_answer": ""}
        print(f"[RESPONSE_GEN] Skipping memory_answer — real tool data exists")
        memory_answer = ""
    messages = state.get("messages", [])

    original_query = (
        state.get("original_query", "")
        or state.get("user_query", "")
        or ""
    )
    if not original_query:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                original_query = getattr(msg, "content", "") or ""
                break
    detected_language = state.get("detected_language") or "auto"

    if detected_language not in ("hinglish", "hindi"):
        hinglish_words = {"batao", "chaia", "wale", "ka", "ki", "kya", "hai", "kitne", "konse", "konsa", "karli", "hua", "hue"}
        tokens = re.findall(r"\w+", original_query.lower())
        if any(w in tokens for w in hinglish_words):
            detected_language = "hinglish"

    previous_summary = state.get("summary", "") or ""
    conversation_context = state.get("conversation_context", {})

    list_mode = bool(re.search(
        r'\b(' + '|'.join(re.escape(w) for w in LIST_WORDS) + r')\b',
        original_query, re.IGNORECASE
    ))

    system_prompt = (
        "You are an ERP assistant. Reply short, natural, in the user's language. Use ONLY TOOL RESULTS below.\n"
        "Personality: warm, conversational. Mirror user's language. Never say 'As an AI'. "
        "End with '.' unless user used '?'.\n"
        "If multiple tools returned data, answer only from the tool(s) relevant to the user's question.\n"
        "NEVER output JSON/code. NEVER invent fields or values.\n"
        "NO headings ('Summary:', 'Details:', etc.), NO meta-framing ('Based on...', 'Here are...'), "
        "NO bullet points unless user asked for a list. "
        "If tool results are empty, say 'data nahi mila' / 'kuch nahi mila'. "
        "Answer only what was asked, 1-3 sentences.\n"
    )
    if previous_summary:
        system_prompt += f"Background conversation:\n{previous_summary}\n\n"
    if conversation_context:
        entities = conversation_context.get("entities", [])
        if entities:
            system_prompt += f"KNOWN ENTITIES:\n{json.dumps(entities[-3:], indent=2, ensure_ascii=False)}\n\n"

    lang_rule = (
        "LANGUAGE: Reply in {mode}. "
        "Use ONLY a-z A-Z 0-9 and basic punctuation. "
        "No Devanagari or non-Latin scripts. "
        "Write Hindi words with English letters (e.g. 'aap', 'hai', 'nahi', 'se', 'ka'). "
        "Mirror the user's words.\n"
        "WRONG: आपके ग्राहक रोहन हैं\nCORRECT: aapke customer Rohan hai\n"
    )
    if detected_language == "hinglish":
        system_prompt += lang_rule.replace("{mode}", "Hinglish (Hindi words in English letters)")
    elif detected_language == "hindi":
        system_prompt += lang_rule.replace("{mode}", "Hinglish")
    else:
        system_prompt += "LANGUAGE: Reply in English.\n"

    system_prompt += (
        "End with '.' not '?' (facts are not questions). "
        "Never invent fields/values — TOOL RESULTS are the ONLY truth. "
        "WRONG: 'productId F12 ka qty 5 se jada hai'\nCORRECT: 'is baare mein data nahi mila'\n"
        "No headings, no meta-framing ('Based on...', 'Here are...', 'In summary...'). "
        "WRONG: 'Sales Invoices Summary: AI/0324/0010 ka netAmount 29315.7'\nCORRECT: 'AI/0324/0010 ka netAmount 29315.7'\n"
        "If data is empty, say 'data/kuch nahi mila'. "
        "Report ALL matching records, never omit duplicates. "
        "WRONG: 'PR-269 ka net 3292.7'\nCORRECT: 'PR-269 ke 2 records: Bigfoot 3292.7, Amazon 1215.95'\n"
        "For multi-part queries, answer each part. "
        "When one tool has the exact record, ignore other tool outputs. "
        "WRONG: 'Purchase has 684 records but Sales has AI/0324/0010'\nCORRECT: 'AI/0324/0010 ka netAmount 29315.7'\n"
        "For detail requests: include EVERY field from the record. "
        "For truncation: just say total count and ask if they want more. No 'Add a filter'. "
        "WRONG: 'Showing first 10 of 854. Add a filter.'\nCORRECT: 'Aapke 27 records hain. Dikhaun?'\n"
        "Show max 5 fields per record for list queries, all fields for detail queries. "
        "Only show data from TOOL RESULTS — no summaries/aggregates of hidden records. "
        "End with a natural follow-up unless query was a yes/no/command.\n"
    )
    if list_mode:
        system_prompt += (
            "LIST MODE: Format each record as '- ' bullet. "
            "STRICT: headings or numbered lists instead of '- ' bullets → response DISCARDED. "
            "Show 1-2 fields per record. Mention total count, ask if they want more.\n"
            "Example:\n"
            "  - Masjid To Churchgate\n"
            "  - Vasai-Dahisar\n\n"
            "  Ye 100 records mein se 10 hain. Aur dikhaun?\n"
        )

    final_response_prompt = dict(final_response)
    truncation_info = final_response_prompt.pop("truncation_info", {}) or {}
    summary_text = final_response_prompt.pop("summary", "") or ""
    final_response_prompt.pop("tools_used", None)
    summary_text = re.sub(r'^[^:]+:\s*', '', summary_text)
    summary_text = re.sub(r'; [^:]+:\s*', '; ', summary_text)
    data = final_response_prompt.get("data", {})

    doc_type = (state.get("document_type", "") or "").lower()
    if doc_type == "general":
        orig_q = (state.get("original_query", "") or "").lower()
        is_follow_up = any(re.search(rf'\b{re.escape(p)}\b', orig_q) for p in HINGLISH_PRONOUNS)
        if is_follow_up:
            customer_vendor_domains = {"customer", "vendor"}
            filtered = {}
            for tool_name, records in data.items():
                td = set(TOOL_DOMAINS.get(tool_name, []))
                if td & customer_vendor_domains:
                    filtered[tool_name] = records
                else:
                    print(f"[DOMAIN-FILTER] {tool_name}: {len(records) if isinstance(records, list) else 'non-list'} records removed (domain {td} not in customer/vendor)")
            if filtered:
                final_response_prompt["data"] = filtered

    detail_mode = bool(re.search(
        r'\b(all details?|sabhi detail|saari detail|full info|complete info|sara detail|poora detail|saare detail)\b',
        original_query, re.IGNORECASE
    ))

    invoice_match = final_response_prompt.pop("_invoice_match", None)
    if invoice_match and isinstance(invoice_match, dict):
        matched_tool = invoice_match.get("tool_name", "")
        if matched_tool and matched_tool in final_response_prompt.get("data", {}):
            filtered_data = {matched_tool: final_response_prompt["data"][matched_tool]}
            final_response_prompt["data"] = filtered_data
            summary_text = make_summary(filtered_data, [])
            summary_text = re.sub(r'^[^:]+:\s*', '', summary_text)
            summary_text = re.sub(r'; [^:]+:\s*', '; ', summary_text)
    final_response_prompt.pop("summary", None)
    final_response_prompt.pop("_invoice_match", None)
    data = final_response_prompt.get("data", {})

    canonical = state.get("canonical_query", "") or ""
    has_range_filter = bool(re.search(
        r'(?:between|range)\s+(-?\d+(?:\.\d+)?)\s+(?:and|to|–|-)\s+(-?\d+(?:\.\d+)?)',
        original_query, re.IGNORECASE
    )) or bool(re.search(
        r'\b(\d+(?:\.\d+)?)\s+se\s+(\d+(?:\.\d+)?)\s+ke?\s+bich',
        original_query, re.IGNORECASE
    )) or bool(re.search(
        r'(?:between|range)\s+(-?\d+(?:\.\d+)?)\s+(?:and|to|–|-)\s+(-?\d+(?:\.\d+)?)',
        canonical, re.IGNORECASE
    ))

    MAX_SAMPLE = 5 if has_range_filter else (20 if detail_mode else (10 if list_mode else 5))
    for tool_name, records in data.items():
        if isinstance(records, list) and len(records) > MAX_SAMPLE:
            data[tool_name] = records[:MAX_SAMPLE]

    # Sync truncation_info shown count with actual data after further truncation
    if truncation_info:
        for tool_name in truncation_info:
            tool_records = data.get(tool_name, [])
            if isinstance(tool_records, list):
                truncation_info[tool_name]["shown"] = len(tool_records)
    if has_range_filter:
        final_response_prompt.pop("total_rows", None)
        truncation_info = {}
    truncation_note = ""
    if truncation_info:
        total = sum(v.get("total", 0) for v in truncation_info.values())
        shown = sum(v.get("shown", 0) for v in truncation_info.values())
        remaining = total - shown
        truncation_note = f"\nMORE RECORDS AVAILABLE: {shown} shown out of {total}. {remaining} more available. Ask conversationally if user wants more.\n"

    TOOL_KEY_MAP = {
        "get_customer": "customers",
        "get_sales_invoice": "sales_invoices",
        "get_purchase_invoice": "purchase_invoices",
        "get_stock_levels": "stock_levels",
        "get_gst_summary": "gst_summary",
        "get_sales_summary": "sales_summary",
        "get_purchase_summary": "purchase_summary",
        "get_sales_trend": "sales_trend",
        "get_trial_balance": "trial_balance",
        "get_top_customer": "top_customers",
        "get_item_details": "items",
        "search_ledger": "ledger_matches",
    }
    tool_data = final_response_prompt.get("data", {})
    if isinstance(tool_data, dict):
        renamed = {}
        for tool_name, records in tool_data.items():
            domain_key = TOOL_KEY_MAP.get(tool_name, tool_name.replace("get_", "", 1))
            renamed[domain_key] = records
        final_response_prompt["data"] = renamed

    if not detail_mode and isinstance(final_response_prompt.get("data"), dict):
        TOOL_FIELD_PRIORITY = {
            "customers": ["name"],
            "stock_levels": ["name", "closingQty"],
            "sales_invoices": ["invoiceNo", "netAmount", "outstanding"],
            "purchase_invoices": ["invoiceNo", "netAmount", "outstanding"],
            "gst_summary": ["category", "totalTaxableValue", "totalTax"],
            "sales_summary": ["category", "total"],
            "purchase_summary": ["category", "total"],
            "sales_trend": ["period", "total"],
            "trial_balance": ["ledgerName", "closingBalance"],
            "top_customers": ["name", "totalRevenue"],
            "items": ["name", "closingQty"],
            "ledger_matches": ["name", "ledgerType"],
            "customer_ledger": ["ledgerName", "debit", "credit"],
            "slow_moving_products": ["name", "closingQty"],
            "search_vendors": ["name"],
            "tds_outstanding": ["category", "amount"],
            "tcs_outstanding": ["category", "amount"],
            "overdue_invoices": ["invoiceNo", "netAmount", "outstanding"],
        }
        pruned_data = {}
        for key, records in final_response_prompt["data"].items():
            if not isinstance(records, list):
                pruned_data[key] = records
                continue
            keep = TOOL_FIELD_PRIORITY.get(key, ["name"])
            pruned_records = []
            for rec in records:
                if not isinstance(rec, dict):
                    pruned_records.append(rec)
                    continue
                new_rec = {k: rec[k] for k in keep if k in rec and rec[k] not in (None, "", [])}
                if not new_rec:
                    first_key = next(iter(rec), None)
                    if first_key is not None:
                        new_rec[first_key] = rec[first_key]
                pruned_records.append(new_rec)
            pruned_data[key] = pruned_records
        final_response_prompt["data"] = pruned_data

    detail_note = "\nDETAIL MODE: Show EVERY field of each record.\n" if detail_mode else ""

    human_prompt = (
        f"USER QUERY:\n{original_query}\n\n"
        f"CRITICAL — Only use field names and values that are present in TOOL RESULTS below. "
        f"Do NOT invent any field name, ID, or value.\n"
        f"TOOL RESULTS (JSON):\n{json.dumps(final_response_prompt, indent=2, ensure_ascii=False)}\n\n"
    )
    if summary_text:
        human_prompt += f"For context (do not repeat this verbatim): {summary_text}\n\n"
    human_prompt += truncation_note + detail_note

    try:
        full_content = ""
        async for chunk in summary_llm.with_config({"tags": ["response_stream"]}).astream([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]):
            if hasattr(chunk, "content") and chunk.content:
                full_content += chunk.content
        response_text = strip_think_tags(full_content.strip())
        input_chars = len(human_prompt) + len(system_prompt)
        output_chars = len(full_content)
        print(f"[TOKENS] response_gen | input_est={input_chars // 4} | output_est={output_chars // 4} | total_est={(input_chars + output_chars) // 4}")
        if not response_text:
            raise ValueError("Empty response from LLM")
        response_text = _clean_llm_response(response_text)
        response_text = _strip_hallucinated_values(
            response_text,
            final_response_prompt.get("data", {}),
            original_query=original_query,
        )
        if not response_text.strip():
            raise ValueError("Response had only meta-framing/JSON after cleaning")
        if list_mode and response_text.strip():
            lines = response_text.strip().split('\n')
            has_bullets = any(line.strip().startswith('- ') for line in lines)
            if not has_bullets:
                print(f"[LIST-MODE-FALLBACK] LLM didn't use bullets for list query. Replacing with deterministic bullet list.")
                fallback_parts = []
                for recs in final_response_prompt.get("data", {}).values():
                    if isinstance(recs, list):
                        for rec in recs[:10]:
                            if isinstance(rec, dict):
                                keys = list(rec.keys())[:2]
                                parts = [f"{k}: {rec[k]}" for k in keys if k in rec]
                                fallback_parts.append("- " + " | ".join(parts))
                total_count = 0
                for v in truncation_info.values() if isinstance(truncation_info, dict) else []:
                    total_count += v.get("total", 0) if isinstance(v, dict) else 0
                if fallback_parts:
                    response_text = "\n".join(fallback_parts)
                    if total_count > len(fallback_parts):
                        response_text += f"\n\nYe {total_count} records mein se {len(fallback_parts)} hain. Aur dikhaun?"
                    else:
                        response_text += f"\n\nYe {len(fallback_parts)} records hain."
                else:
                    response_text = "kuch nahi mila"
        data = final_response.get("data", {})
        has_any_data = any(
            isinstance(recs, list) and len(recs) > 0
            for recs in data.values()
        )
        if not has_any_data:
            empty_indicators = ["data nahi mila", "kuch nahi mila", "no data found",
                                "no records found", "nahi mila", "kuch nahi"]
            if not any(ind in response_text.lower() for ind in empty_indicators):
                print(f"[HALLUCINATION GUARD] Empty data but response didn't acknowledge: '{response_text[:150]}...'")
                response_text = "data nahi mila"
    except Exception as e:
        print(f"Error in response generation node: {e}")
        summary_text = final_response.get("summary", "") if isinstance(final_response, dict) else ""
        if summary_text and len(summary_text) > 20:
            response_text = summary_text
        else:
            data = final_response.get("data", {}) if isinstance(final_response, dict) else {}
            parts = []
            for tool_name, records in data.items() if isinstance(data, dict) else []:
                if records and isinstance(records, list):
                    count = len(records)
                    parts.append(f"{tool_name}: {count} record(s)")
            response_text = (
                "; ".join(parts)
                if parts
                else (str(final_response)[:500] if isinstance(final_response, dict) else str(final_response)[:500])
            )
    return {"response_text": response_text}
