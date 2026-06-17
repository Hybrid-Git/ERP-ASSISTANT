import json
import re
from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage
from src.schema import MainState
from src.config import summary_llm
from src.utils import LIST_WORDS, TOOL_DOMAINS, strip_think_tags, _get_tokenizer
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
    known_field_like_values = {
        "b2b", "b2cLarge", "b2cSmall", "b2cs", "b2cl", "exports",
        "nillRated", "nilRated", "nillrated", "nilrated",
        "creditNotesRegistered", "creditNotesUnregistered",
        "creditnotesregistered", "creditnotesunregistered",
        "grandTotal", "grandtotal", "exempted", "sez",
    }
    field_like = re.compile(r'\b[a-z]+[A-Z][a-zA-Z]*\b|\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b|\b[a-zA-Z]+_[a-zA-Z]+\b')
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    cleaned = []
    for sent in sentences:
        if not sent.strip():
            continue
        words = set(m.group(0).lower() for m in field_like.finditer(sent))
        hallucinated = any(
            w not in actual_fields and w not in query_tokens and w not in known_field_like_values
            for w in words
        )
        if hallucinated:
            print(f"[HALLUCINATION] Stripped sentence containing invented field(s): {words - actual_fields - known_field_like_values} | sentence: {sent.strip()[:100]}")
        else:
            cleaned.append(sent)
    result = " ".join(cleaned).strip()
    if not result:
        print(f"[HALLUCINATION] All sentences stripped — returning empty")
    return result


@traceable(name="response_generation_node", run_type="chain")
async def response_generation_node(state: MainState):
    print("→ response_gen")
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
        "You are an ERP assistant. Reply using ONLY the tool results below.\n"
        "Vary your tone. Mirror the user's language. "
        "Never say 'As an AI'. No JSON/code/headings. "
        "If empty → 'data nahi mila'. Be conversational.\n"
    )
    if previous_summary:
        system_prompt += f"Background conversation:\n{previous_summary[:400]}\n\n"
    if conversation_context:
        entities = conversation_context.get("entities", [])
        if entities:
            system_prompt += f"KNOWN ENTITIES:\n{json.dumps(entities[-3:], indent=2, ensure_ascii=False)}\n\n"

    mode_map = {"hinglish": "Hinglish (Hindi words in English letters)", "hindi": "Hinglish"}
    lang_mode = mode_map.get(detected_language, "English")
    system_prompt += (
        "LANGUAGE: Reply in " + lang_mode + ". "
        "Use ONLY a-z A-Z 0-9. No Devanagari. "
        "Write Hindi with English letters (aap/hai/nahi). "
        "Mirror the user's words.\n"
    )

    system_prompt += (
        "TOOL RESULTS are the ONLY truth — never invent fields/values. "
        "Multi-part → answer each. "
        "Never use the word 'sample'.\n"
    )

    intent = state.get("query_intent", "sample")
    if intent == "count":
        system_prompt += (
            "If truncated, mention total count. "
            "COUNT ONLY: Report the total number from 'MORE RECORDS AVAILABLE'. "
            "Say 'aapke paas X records hain' (or 'you have X records' in English). "
            "Do NOT mention truncation, shown count, or list records. "
            "Do NOT offer to show more records. "
            "Answer just the count and nothing else.\n"
        )
    elif list_mode:
        system_prompt += (
            "If truncated, mention total count. For list queries, ask if they want more. "
            "LIST MODE - STRICT:\n"
            "First line MUST say 'showing X of Y total records' "
            "(X=records shown below, Y=total from MORE RECORDS AVAILABLE).\n"
            "Then each record as '- ' bullet, 1-2 fields.\n"
            "Never use headings.\n"
            "No numbered lists (only '- ' bullets).\n"
            "End by asking if they want to see more.\n"
        )
    elif intent == "aggregate":
        system_prompt += (
            "If truncated, mention total count. "
            "AGGREGATE: Use ALL records. If truncated, mention total and "
            "note values are based on shown records only.\n"
        )
    else:
        system_prompt += (
            "If truncated, mention total count. "
            "End with a natural follow-up unless yes/no/command.\n"
        )

    final_response_prompt = dict(final_response)
    truncation_info = final_response_prompt.pop("truncation_info", {}) or {}
    summary_text = final_response_prompt.pop("summary", "") or ""
    final_response_prompt.pop("tools_used", None)
    summary_text = re.sub(r'^[^:]+:\s*', '', summary_text)
    summary_text = re.sub(r';\s*[^:]+:\s*', '; ', summary_text)
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
        r'\b(all details?|detailed|sabhi detail|saari detail|full info|complete info|sara detail|poora detail|saare detail|pura detail|puri detail|pura detailed|puri detailed|poora detailed)\b',
        original_query, re.IGNORECASE
    ))

    invoice_match = final_response_prompt.pop("_invoice_match", None)
    if invoice_match and isinstance(invoice_match, dict):
        detail_mode = True
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

    # truncation already done in deterministic_final.py per intent

    # Sync truncation_info shown count with actual data after further truncation
    if truncation_info:
        for tool_name in truncation_info:
            tool_records = data.get(tool_name, [])
            if isinstance(tool_records, list):
                truncation_info[tool_name]["shown"] = len(tool_records)
    if has_range_filter:
        final_response_prompt.pop("total_rows", None)
        truncation_info = {}
    TOOL_KEY_MAP = {
        "get_customer": "customers",
        "get_customer_ledger": "customer_ledger",
        "get_stock_levels": "stock_levels",
        "get_gst_summary": "gst_summary",
        "get_tds_outstanding": "tds_outstanding",
        "get_tcs_outstanding": "tcs_outstanding",
        "get_top_products": "top_products",
        "get_popular_products": "popular_products",
        "get_slow_moving_products": "slow_moving_products",
        "get_sales_summary": "sales_summary",
        "get_sales_trend": "sales_trend",
        "get_top_customer": "top_customers",
        "get_top_vendor": "top_vendors",
        "get_purchase_summary": "purchase_summary",
        "get_search_ledgers": "ledger_matches",
        "get_search_vendors": "vendor_matches",
        "get_outstanding_sales_invoices": "outstanding_sales_invoices",
        "get_outstanding_purchase_invoices": "outstanding_purchase_invoices",
        "get_overdue_invoices": "overdue_invoices",
    }
    truncation_note = ""
    if truncation_info:
        notes = []
        for tn, info in truncation_info.items():
            nice_name = TOOL_KEY_MAP.get(tn, tn.replace("get_", "", 1))
            notes.append(
                f"{nice_name}: {info['shown']} total"
                if info['shown'] >= info['total']
                else f"{nice_name}: {info['shown']} out of {info['total']} shown"
            )
        if intent == "count":
            truncation_note = f"\nMORE RECORDS AVAILABLE: Total: {'; '.join(notes)}. Report ONLY the total count.\n"
        else:
            truncation_note = f"\nMORE RECORDS AVAILABLE: {'; '.join(notes)}. Ask if user wants more.\n"
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
            "outstanding_purchase_invoices": ["invoiceNo", "netAmount", "outstanding"],
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
        f"TOOL RESULTS:\n{json.dumps(final_response_prompt, separators=(',',':'), ensure_ascii=False)}\n\n"
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
        enc = _get_tokenizer()
        input_tokens = len(enc.encode(system_prompt + human_prompt))
        output_tokens = len(enc.encode(full_content))
        print(f"[TOKENS] response_gen | input={input_tokens} | output={output_tokens} | total={input_tokens + output_tokens}")
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
            print(f"[LIST-MODE] LLM generated {len(response_text.strip().split(chr(10)))} lines for list query.")
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
        query_intent = state.get("query_intent", "")
        if query_intent == "count" and isinstance(final_response, dict):
            trunc_info = final_response.get("truncation_info", {})
            if trunc_info:
                total_strs = []
                for tn, info in trunc_info.items():
                    total = info.get("total", "unknown")
                    nice_name = tn.replace("get_", "").replace("_", " ")
                    total_strs.append(f"{nice_name}: {total}")
                response_text = "aapke paas " + " aur ".join(total_strs) + " hain"
            else:
                data = final_response.get("data", {})
                counts = [f"{tn}: {len(recs)}" for tn, recs in data.items() if isinstance(recs, list)]
                response_text = "; ".join(counts) if counts else "data nahi mila"
        elif isinstance(final_response, dict):
            summary_text = final_response.get("summary", "")
            if summary_text and len(summary_text) > 20:
                response_text = summary_text
            else:
                data = final_response.get("data", {})
                parts = [f"{tn}: {len(recs)} record(s)" for tn, recs in data.items() if isinstance(recs, list)]
                response_text = "; ".join(parts) if parts else "data nahi mila"
        else:
            response_text = "data nahi mila"
    return {"response_text": response_text}
