import json
import re
from langsmith import traceable
from langchain_core.messages import AIMessage, ToolMessage
from src.schema import MainState
from src.tool_doc import TOOL_INTENT_REGISTRY
from src.utils import parse_planner_json_blocks, normalize_text
# --- COMMENTED OUT (zero-regex migration): word-list imports ---
# from src.tool_doc import CITY_WORDS
# from src.utils import TOOL_DOMAINS, PARTY_WORDS, NAME_WORDS, LIST_WORDS, INVOICE_TOOLS, INVOICE_NO_PATTERNS
# from src.prompts import GST_CATEGORY_KEYWORDS
from src.config import get_cfg


# --- COMMENTED OUT (zero-regex migration): identifier/city/name config ---
# IDENTIFIER_CONFIG = get_cfg("identifier_patterns", default={})
# CITY_ALIASES = get_cfg("city_aliases", default={})
# CITY_WORDS_SET = {w.lower() for w in CITY_WORDS}
# NAME_STOPWORDS = {w.lower() for w in get_cfg("name_extraction", "stopwords", default=[])}
ENTITY_SKIP_TOOLS = {"get_top_customer", "get_sales_trend", "get_stock_levels"}


def parse_tool_output(content):
    try:
        if isinstance(content, dict):
            return content
        if isinstance(content, list):
            return {"success": True, "data": content, "count": len(content), "error": None}
        return json.loads(content)
    except Exception as e:
        return {"success": False, "data": [], "count": 0, "error": f"Could not parse tool output: {str(e)}"}


def get_tool_name(tool_message, messages):
    tool_name = getattr(tool_message, "name", None)
    if tool_name:
        return tool_name
    tool_call_id = getattr(tool_message, "tool_call_id", None)
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                if call.get("id") == tool_call_id:
                    return call.get("name")
    return "unknown_tool"


def make_summary(data: dict, errors: list, unsupported_parts: list | None = None, total_rows: int = 0) -> str:
    parts = []
    FIELD_DISPLAY_ORDER = ("ledgerName", "name", "invoiceNo", "netAmount", "outstanding", "taxableAmount", "totalInvoices", "totalOutstanding", "category")
    def _format_record_fields(record: dict) -> str:
        return ", ".join(
            f"{k}={record[k]}" for k in FIELD_DISPLAY_ORDER
            if k in record and record[k] is not None
        )
    for tool_name, records in data.items():
        count = len(records) if isinstance(records, list) else 0
        if count == 0:
            parts.append(f"{tool_name}: no records found")
        elif count == 1:
            summary = f"{tool_name}: found 1 record"
            record = records[0] if isinstance(records, list) else records
            if isinstance(record, dict):
                formatted = _format_record_fields(record)
                if formatted:
                    summary += " | " + formatted
            parts.append(summary)
        else:
            summary = f"{tool_name}: found {count} records"
            record = records[0] if isinstance(records, list) else records
            if isinstance(record, dict):
                formatted = _format_record_fields(record)
                if formatted:
                    summary += " (e.g., " + formatted + ")"
            parts.append(summary)
    if total_rows > 0:
        parts.append(f"total_rows: {total_rows}")
    if errors:
        parts.append(f"{len(errors)} error(s)")
    if unsupported_parts:
        parts.append(f"{len(unsupported_parts)} unsupported part(s)")
    return "; ".join(parts)


def dedupe_records_by_field(records: list[dict], field: str) -> list[dict]:
    if not isinstance(records, list):
        return records
    seen = set()
    deduped = []
    for record in records:
        if not isinstance(record, dict):
            continue
        value = record.get(field)
        if value is None or str(value).strip() == "":
            continue
        key = str(value).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


# --- COMMENTED OUT (zero-regex migration): word-list based functions ---
# def wants_unique_party_names(query: str) -> bool:
#     q = (query or "").lower()
#     has_party_word = any(word in q for word in PARTY_WORDS)
#     has_name_word = any(word in q for word in NAME_WORDS)
#     has_list_word = any(word in q for word in LIST_WORDS)
#     return has_party_word and has_name_word and has_list_word


def apply_final_postprocessing(final_data: dict, original_query: str, canonical_query: str = "") -> dict:
    if not isinstance(final_data, dict):
        return final_data
    for tool_name, records in final_data.items():
        if isinstance(records, list):
            seen = set()
            deduped = []
            for r in records:
                key = json.dumps(r, sort_keys=True) if isinstance(r, dict) else str(r)
                if key not in seen:
                    seen.add(key)
                    deduped.append(r)
            final_data[tool_name] = deduped
    return final_data


def compact_transactions(records: list[dict]) -> list[dict]:
    compacted_records = []
    for record in records:
        if not isinstance(record, dict):
            continue
        new_record = dict(record)
        transactions = new_record.get("transactions")
        if isinstance(transactions, list):
            compacted_txns = []
            for txn in transactions:
                if not isinstance(txn, dict):
                    continue
                txn_copy = dict(txn)
                items = txn_copy.pop("items", [])
                txn_copy["itemCount"] = len(items) if isinstance(items, list) else 0
                compacted_txns.append(txn_copy)
            new_record["transactions"] = compacted_txns
        compacted_records.append(new_record)
    return compacted_records


# --- COMMENTED OUT (zero-regex migration): GST category filtering ---
# def requested_gst_categories(query: str) -> list[str]:
#     q = normalize_text(query)
#     categories: list[str] = []
#     def add(category: str):
#         if category not in categories:
#             categories.append(category)
#     for cat_val, kws in GST_CATEGORY_KEYWORDS.items():
#         for kw in kws:
#             if kw in q:
#                 add(cat_val)
#                 break
#     if "b2c" in q and "b2cSmall" not in categories and "b2cLarge" not in categories:
#         if not any(c.lower() in ("b2csmall", "b2clarge") for c in categories):
#             add("b2cSmall")
#             add("b2cLarge")
#     return categories

# def filter_gst_records_by_query(records: list[dict], query: str) -> list[dict]:
#     requested = requested_gst_categories(query)
#     if not requested:
#         return records
#     has_category = any(isinstance(r, dict) and "category" in r for r in records)
#     if not has_category:
#         return records
#     requested_set = set(requested)
#     return [
#         record for record in records
#         if isinstance(record, dict) and record.get("category") in requested_set
#     ]


# --- COMMENTED OUT (zero-regex migration): identifier/city extraction ---
# def extract_query_identifiers(query: str, last_tool_call: dict | None = None) -> dict:
#     identifiers: dict = {}
#     q_lower = query.lower() if query else ""
#     for id_key, patterns in IDENTIFIER_CONFIG.items():
#         for pat in patterns:
#             m = re.search(pat, q_lower, re.IGNORECASE)
#             if m:
#                 raw = m.group(1) if m.lastindex else m.group(0)
#                 if id_key in ("party_id", "vendor_id") and raw.isdigit():
#                     identifiers[id_key] = int(raw)
#                 else:
#                     identifiers[id_key] = raw
#                 break
#     for city_name in CITY_WORDS_SET:
#         if re.search(rf'\b{re.escape(city_name.lower())}\b', q_lower):
#             if re.search(rf'\b(B2[Cc]|B2[Bb])\s*[-_]?{re.escape(city_name.lower())}\b', q_lower):
#                 continue
#             identifiers["city"] = city_name.upper()
#             break
#     if "city" not in identifiers:
#         for alias, canonical in CITY_ALIASES.items():
#             if alias in q_lower:
#                 identifiers["city"] = canonical
#                 break
#     return identifiers

# def apply_identifier_filter(data: dict, identifiers: dict) -> dict:
#     if not identifiers:
#         return data
#     result = {}
#     for tool_name, records in data.items():
#         if not isinstance(records, list):
#             result[tool_name] = records
#             continue
#         tool_filters = TOOL_INTENT_REGISTRY.get(tool_name, {}).get("record_filters", [])
#         if not tool_filters:
#             result[tool_name] = records
#             continue
#         kept_summaries = []
#         kept_details = []
#         for record in records:
#             if not isinstance(record, dict):
#                 kept_details.append(record)
#                 continue
#             if record.get("recordType") == "summary":
#                 kept_summaries.append(record)
#                 continue
#             matched = False
#             for f in tool_filters:
#                 id_val = identifiers.get(f["id_key"])
#                 if id_val is None:
#                     continue
#                 rec_val = record.get(f["match_field"])
#                 if rec_val is None:
#                     continue
#                 mt = f.get("match_type", "exact")
#                 if mt == "exact":
#                     if str(id_val) == str(rec_val):
#                         matched = True
#                         break
#                 elif mt == "icontains":
#                     if str(id_val).lower() in str(rec_val).lower():
#                         matched = True
#                         break
#                 elif mt == "exact_case_insensitive":
#                     if str(id_val).lower() == str(rec_val).lower():
#                         matched = True
#                         break
#             if matched:
#                 kept_details.append(record)
#         kept = kept_summaries + kept_details
#         if kept_details:
#             result[tool_name] = kept
#             if len(kept) < len(records):
#                 print(f"[IDENT-FILTER] {tool_name}: kept {len(kept)}/{len(records)} records matching {identifiers}")
#         else:
#             result[tool_name] = records
#             print(f"[IDENT-FILTER] {tool_name}: no match for {identifiers}, kept all {len(records)} records (fallback)")
#     return result


@traceable(name="deterministic_final_node", run_type="chain")
async def deterministic_final_node(state: MainState):
    print("→ deterministic_final")
    user_query = state.get("user_query", "")
    canonical_query = state.get("canonical_query", "")
    messages = state.get("messages", [])
    data = {}
    tools_used = []
    errors = []
    total_rows = 0
    total_rows_per_tool = {}
    current_tool_call_ids = set()
    last_tool_call = {}
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            current_tool_call_ids = {tc.get("id") for tc in msg.tool_calls if tc.get("id")}
            for tc in msg.tool_calls:
                name = tc.get("name")
                args = tc.get("args")
                if name and args:
                    last_tool_call[name] = args
            break
    tool_messages = [
        msg for msg in messages
        if isinstance(msg, ToolMessage)
        and (not current_tool_call_ids or msg.tool_call_id in current_tool_call_ids)
    ]
    if not tool_messages:
        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if last_ai_msg:
            blocks = parse_planner_json_blocks(last_ai_msg.content)
            for block in blocks:
                if isinstance(block, dict) and block.get("status") == "needs_clarification":
                    return {
                        "final_response": {
                            "success": False,
                            "status": "needs_clarification",
                            "query": user_query,
                            "tools_used": [],
                            "data": {},
                            "summary": block.get("summary", "Please specify the customer name or customer id and date range."),
                            "errors": [],
                        },
                        "tools_utilized": [],
                    }
    ctx = dict(state.get("conversation_context", {}))
    for tool_msg in tool_messages:
        tool_name = get_tool_name(tool_msg, messages)
        if tool_name not in tools_used:
            tools_used.append(tool_name)
        parsed = parse_tool_output(tool_msg.content)
        if not parsed.get("success"):
            data.setdefault(tool_name, [])
            error_text = parsed.get("error", "Unknown tool error")
            errors.append({"tool": tool_name, "error": error_text})
            continue
        records = parsed.get("data", [])
        if records is None:
            records = []
        if not isinstance(records, list):
            records = [records]
        if isinstance(parsed, dict):
            tr = parsed.get("total_rows", 0) or 0
            total_rows += tr
            if tr > 0:
                total_rows_per_tool[tool_name] = max(total_rows_per_tool.get(tool_name, 0), tr)
        if tool_name == "get_customer_ledger":
            records = compact_transactions(records)
        data.setdefault(tool_name, [])
        existing_ids = {r.get("id") for r in data[tool_name] if isinstance(r, dict) and r.get("id") is not None}
        records = [r for r in records if not (isinstance(r, dict) and r.get("id") is not None and r["id"] in existing_ids)]
        data[tool_name].extend(records)

    entities = list(ctx.get("entities", []))
    seen_names = {e["name"] for e in entities if "name" in e}
    for tool_msg in tool_messages:
        tool_name = getattr(tool_msg, "name", "")
        if tool_name in ENTITY_SKIP_TOOLS:
            continue
        parsed = parse_tool_output(tool_msg.content)
        recs = parsed.get("data", []) if isinstance(parsed, dict) else []
        if not isinstance(recs, list):
            recs = [recs]
        if len(recs) > 10:
            continue
        for rec in recs:
            if isinstance(rec, dict):
                if rec.get("recordType") == "summary":
                    continue
                name = rec.get("name") or rec.get("ledgerName") or rec.get("customerName") or ""
                if name and name.lower() in ("grand total", "summary", "total"):
                    continue
                id_ = rec.get("id") or rec.get("ledgerId") or rec.get("customerId")
                if name and name not in seen_names:
                    seen_names.add(name)
                    entry = {"name": name}
                    if id_ is not None:
                        entry["id"] = id_
                    # --- COMMENTED OUT (zero-regex migration): domain from TOOL_DOMAINS ---
                    # source_domains = TOOL_DOMAINS.get(tool_name, [])
                    # if source_domains:
                    #     entry["domain"] = source_domains[0]
                    entities.append(entry)
    ctx["entities"] = entities

    # --- COMMENTED OUT (zero-regex migration): cross-reference customer→invoice filtering ---
    # customer_names = []
    # for tn in ("get_customer", "get_search_vendors"):
    #     for rec in data.get(tn, []):
    #         if isinstance(rec, dict):
    #             n = rec.get("name") or rec.get("ledgerName") or rec.get("customerName") or ""
    #             if n:
    #                 customer_names.append(n)
    # if customer_names:
    #     combined_q = f"{user_query or ''} {canonical_query or ''}".lower()
    #     mentions_customer = any(re.search(rf'\b{re.escape(cn.lower())}\b', combined_q) for cn in customer_names)
    #     if not mentions_customer:
    #         party_words_in_q = any(re.search(rf'\b{re.escape(w)}\b', combined_q) for w in PARTY_WORDS)
    #         name_words_in_q = any(re.search(rf'\b{re.escape(w)}\b', combined_q) for w in NAME_WORDS)
    #         mentions_customer = party_words_in_q and name_words_in_q
    #     if mentions_customer:
    #         for tin in INVOICE_TOOLS:
    #             invoice_records = data.get(tin, [])
    #             if not invoice_records:
    #                 continue
    #             filtered = [
    #                 r for r in invoice_records
    #                 if isinstance(r, dict) and any(
    #                     re.search(rf'\b{re.escape(cn.lower())}\b', (r.get("ledgerName") or r.get("partyName") or "").lower())
    #                     for cn in customer_names
    #                 )
    #             ]
    #             if filtered:
    #                 print(f"[CROSS-REF] {tin}: {len(invoice_records)} → {len(filtered)} records (matched customer: {customer_names})")
    #                 data[tin] = filtered

    # --- COMMENTED OUT (zero-regex migration): identifier/city filtering ---
    # combined_query = f"{user_query or ''} {canonical_query or ''}"
    # identifiers = extract_query_identifiers(combined_query, last_tool_call)
    # if identifiers:
    #     data = apply_identifier_filter(data, identifiers)
    data = apply_final_postprocessing(data, user_query, canonical_query)

    # --- COMMENTED OUT (zero-regex migration): show_all_keywords set ---
    # combined_q = f"{user_query or ''} {canonical_query or ''}".lower()
    # show_all_keywords = {"show all", "show more", "sab dikhao", ...}
    # wants_all = any(kw in combined_q for kw in show_all_keywords)

    intent = state.get("query_intent", "sample")
    intent_max = {"count": 500, "aggregate": 500, "list_all": 10, "comparison": 200, "detail": 200, "sample": 100, "extreme": 1}
    MAX_RECORDS = intent_max.get(intent, 10)
    truncation_info = {}
    for tool_name, records in data.items():
        if isinstance(records, list):
            actual_count = len(records)
            total = total_rows_per_tool.get(tool_name, actual_count)
            truncation_info[tool_name] = {"total": total, "shown": min(actual_count, MAX_RECORDS)}
            if total > MAX_RECORDS:
                data[tool_name] = records[:MAX_RECORDS]

    # Strip null/empty values to reduce response gen tokens
    for tool_name, records in data.items():
        if isinstance(records, list):
            data[tool_name] = [
                {k: v for k, v in r.items() if v not in (None, "", [])}
                if isinstance(r, dict) else r
                for r in records
            ]

    if not ctx.get("focus_entity"):
        total_records = sum(len(v) for v in data.values() if isinstance(v, list))
        if total_records == 1:
            for tool_name, records in data.items():
                if isinstance(records, list) and len(records) == 1:
                    rec = records[0]
                    if isinstance(rec, dict) and rec.get("recordType") != "summary":
                        name = rec.get("ledgerName") or rec.get("customerName") or rec.get("name") or ""
                        if name:
                            ctx["focus_entity"] = {"name": name}
                            break

    unsupported_parts = state.get("unsupported_parts", [])
    has_any_data = any(isinstance(records, list) and len(records) > 0 for records in data.values())
    has_empty_requested_sections = any(isinstance(records, list) and len(records) == 0 for records in data.values())

    if unsupported_parts:
        status = "partial_success"
        success = bool(has_any_data)
    elif errors and has_any_data:
        status = "partial_success"
        success = True
    elif errors and not has_any_data:
        status = "error"
        success = False
    elif has_any_data and has_empty_requested_sections:
        status = "partial_success"
        success = True
    elif has_any_data:
        status = "success"
        success = True
    else:
        status = "no_matching_records"
        success = False

    final_response = {
        "success": success,
        "status": status,
        "query": user_query,
        "tools_used": tools_used,
        "data": data,
        "summary": make_summary(data, errors, unsupported_parts, total_rows),
        "errors": errors,
        "truncation_info": truncation_info,
    }

    if total_rows > 0:
        final_response["total_rows"] = total_rows
    if unsupported_parts:
        final_response["unsupported_parts"] = unsupported_parts

    for m in state.get("messages", []):
        if isinstance(m, ToolMessage):
            try:
                content = json.loads(m.content) if isinstance(m.content, str) else m.content
                if isinstance(content, dict) and "raw_response" in content:
                    del content["raw_response"]
                    m.content = json.dumps(content, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                pass

    return {
        "final_response": final_response,
        "tools_utilized": tools_used,
        "last_tool_call": last_tool_call,
        "conversation_context": ctx,
    }
