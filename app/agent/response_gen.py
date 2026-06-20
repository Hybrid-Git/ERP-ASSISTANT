import json
import re
from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage
from src.schema import MainState
from src.config import summary_llm
from src.utils import strip_think_tags, _get_tokenizer
from src.deterministic_final import make_summary
import logging
import os

logger = logging.getLogger("erp_assistant.response_gen")
logger_token = logging.getLogger("erp_assistant.tokens")
model = os.getenv("LLM_MODEL")
provider = "sarvamai"
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
    logger.info("Response generation started", extra={"node": "response_generation"})
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
    detected_language = (state.get("detected_language") or "").strip().lower()
    _LANG_EXAMPLES = {
        "gujarati": "Like: 'hu tamari business data ma help karu chu. customers, stock, GST ni details apvi shaku chu.'",
        "marathi": "Like: 'mi tumhalya business data madat karu shakto. customers, stock, GST chi mahiti deu shakto.'",
        "hinglish": "Like: 'Main aapki business data aur ERP systems mein madad kar sakta hoon. Main doosre topics ke baare mein nahi jaanta.'",
        "hindi": "Like: 'Main keval business data aur ERP ki jankari de sakta hoon. Main anya vishayo ke baare mein nahi bata sakta.'",
    }
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
    previous_summary = state.get("summary", "") or ""
    conversation_context = state.get("conversation_context", {})

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

    system_prompt += (
        "TOOL RESULTS are the ONLY truth — never invent fields/values. "
        "Multi-part → answer each. "
        "Never use the word 'sample'.\n"
        "If the TOOL RESULTS don't contain data matching what the user asked about, "
        "say 'data nahi mila'. Do NOT pretend records from one category "
        "are records from another category.\n"
    )

    intent = state.get("query_intent", "sample")

    final_response_prompt = dict(final_response)
    truncation_info = final_response_prompt.pop("truncation_info", {}) or {}
    summary_text = final_response_prompt.pop("summary", "") or ""
    final_response_prompt.pop("tools_used", None)
    final_response_prompt.pop("success", None)
    final_response_prompt.pop("status", None)
    final_response_prompt.pop("errors", None)
    summary_text = re.sub(r'^[^:]+:\s*', '', summary_text)
    summary_text = re.sub(r';\s*[^:]+:\s*', '; ', summary_text)
    data = final_response_prompt.get("data", {})

    detail_mode = state.get("query_intent") == "detail"

    invoice_match = final_response_prompt.pop("_invoice_match", None)
    if invoice_match and isinstance(invoice_match, dict):
        detail_mode = True
    final_response_prompt.pop("summary", None)
    final_response_prompt.pop("_invoice_match", None)
    data = final_response_prompt.get("data", {})
    tool_count = sum(1 for v in data.values() if isinstance(v, list) and len(v) > 0)

    lang_ref = "in the same language as the user"

    # --- Intent-specific response format ---
    if intent == "count":
        system_prompt += (
            "COUNT ONLY: Report the total number. "
            f"Say the count {lang_ref}. "
            "Do NOT mention truncation, shown count, or list records. "
            "Do NOT offer to show more records. "
            "Answer just the count and nothing else.\n"
        )
    elif intent == "aggregate":
        system_prompt += (
            "AGGREGATE ONLY: Report the total or sum. "
            "Just the number — no list, no details. "
            "Do NOT mention truncation or offer to show more. "
            f"Answer {lang_ref}.\n"
        )
    elif intent == "list_all":
        system_prompt += (
            "LIST ALL MODE — show every record the user asked for:\n"
            "- One per line, starting with '- '\n"
            "- Just the values — no 'name:' labels\n"
            "- No intro — start directly with bullet list\n"
        )
        if tool_count > 1:
            system_prompt += (
                "- MULTIPLE TOOLS HAVE RESULTS — group by section:\n"
                "  Section header on its own line, then its bullet list.\n"
                "  Blank line between sections. End with 'Wanna see more of any?'\n"
            )
        else:
            system_prompt += (
                f"- End with a follow-up question {lang_ref}\n"
                "- No JSON, code blocks, or headings\n"
            )
    elif intent == "extreme":
        system_prompt += (
            "EXTREME MODE — top or bottom result only:\n"
            "State the name and value in a single clear sentence. "
            "No bullets, no lists. "
            f"Answer {lang_ref}.\n"
        )
    elif intent == "detail":
        system_prompt += (
            "DETAIL MODE — show EVERY field of each record:\n"
            "- key: value on each line\n"
            "- Blank line between records\n"
            "- Do NOT skip or truncate any field\n"
        )
    elif intent == "comparison":
        system_prompt += (
            "COMPARISON MODE — compare both sides clearly:\n"
            "Show what differs and by how much. "
            f"Use {lang_ref}. "
            "No bullets — plain explanation.\n"
        )
    else:  # sample (default)
        system_prompt += (
            "SAMPLE MODE — show a few example records:\n"
            "- One per line, starting with '- '\n"
            "- Just the values — no 'name:' labels\n"
            "- No intro — start directly with bullet list\n"
        )
        if tool_count > 1:
            system_prompt += (
                "- MULTIPLE TOOLS HAVE RESULTS — group by section:\n"
                "  Section header on its own line, then its bullet list.\n"
                "  Blank line between sections. End with 'Wanna see more of any?'\n"
            )
        else:
            system_prompt += (
                f"- End with a follow-up question {lang_ref}\n"
                "- No JSON, code blocks, or headings\n"
            )

    canonical = state.get("canonical_query", "") or ""
    has_range_filter = False

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
    human_prompt += detail_note
    if original_query:
        lang_label = f" in {detected_language}" if detected_language not in ("unknown", "auto", "") else ""
        human_prompt += (
            f"\nCRITICAL — The user asked{lang_label}: \"{original_query}\". "
            f"You MUST respond{lang_label}, matching the user's language, script, and register. "
            f"If the user writes in a romanized script, you MUST respond in romanized form. "
            f"{example_suffix} "
            f"{lang_warning}\n"
        )

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
        logger.info(
                    "Token usage",
                    extra={
                        "node": "response_gen",
                        "provider": provider,
                        "model": model,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens+output_tokens,
                    },
                )
        if not response_text:
            raise ValueError("Empty response from LLM")
        response_text = _clean_llm_response(response_text)
        data_has_values = any(
            isinstance(recs, list) and len(recs) > 0
            for recs in (final_response_prompt.get("data", {}) or {}).values()
        )
        list_intents = ("sample", "list_all", "detail")
        if not (data_has_values and intent in list_intents):
            response_text = _strip_hallucinated_values(
                response_text,
                final_response_prompt.get("data", {}),
                original_query=original_query,
            )
        if not response_text.strip():
            raise ValueError("Response had only meta-framing/JSON after cleaning")
        if response_text.strip() and original_query:
            print(f"[RESPONSE] {len(response_text.strip().split(chr(10)))} lines for: {original_query[:60]}")
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
