import json
import re
from langsmith import traceable
from langchain_core.messages import AIMessage, ToolMessage
from src.schema import MainState
from src.tool_doc import TOOL_INTENT_REGISTRY
from src.utils import parse_planner_json_blocks, normalize_text
from src.config import get_cfg
import logging

logger = logging.getLogger("erp_assistant.deterministic_final")


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




@traceable(name="deterministic_final_node", run_type="chain")
async def deterministic_final_node(state: MainState):
    logger.info("Deterministic final started", extra={"node": "deterministic_final"})
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
             
                    entities.append(entry)
    ctx["entities"] = entities


    intent = state.get("query_intent", "sample")
    intent_max = {"count": 500, "aggregate": 500, "list_all": 10, "comparison": 200, "detail": 200, "sample": 10, "extreme": 1}
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
