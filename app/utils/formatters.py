import re

from app.core.config import get_cfg


_pretty_field_names = get_cfg("pretty_field_names", default={})

def pretty_field_name(key: str) -> str:
    if key in _pretty_field_names:
        return _pretty_field_names[key]
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", key)
    return spaced.replace("_", " ").title()


_TOOL_DISPLAY_NAMES = {
    "get_customer": "Customers",
    "get_customer_ledger": "Customer Ledger",
    "get_stock_levels": "Products / Stock",
    "get_gst_summary": "GST Summary",
    "get_tds_outstanding": "TDS Outstanding",
    "get_tcs_outstanding": "TCS Outstanding",
}


def get_tool_display_name(tool_name: str) -> str:
    return _TOOL_DISPLAY_NAMES.get(tool_name, tool_name.replace("_", " ").title())


async def format_response_as_chat_text(
    response_data: dict,
    timings: list = None,
    total_time: float = None,
    **kwargs,
) -> str:
    """
    Converts deterministic JSON into a clean conversational sentence.
    This is response formatting only. It is not conversation summarization.
    """
    status = response_data.get("status", "")
    summary = response_data.get("summary", "")
    query = response_data.get("query", "")
    data = response_data.get("data", {})

    if status == "needs_clarification":
        return f"[INFO] {summary if summary else 'Could you please clarify your request with a specific name or ID?'}"

    if status == "no_matching_records":
        return "I checked your ERP records but couldn't find any matching data for that description."

    if not data:
        return "I encountered an issue retrieving those records right now."

    lines = []
    for tool_name, records in data.items():
        if not isinstance(records, list) or not records:
            continue
        label = get_tool_display_name(tool_name)
        lines.append(f"\n--- {label} ---")
        for i, record in enumerate(records, 1):
            if not isinstance(record, dict):
                lines.append(f"{i}. {record}")
                continue
            parts = [pretty_field_name(k) + ": " + str(v) for k, v in record.items() if v is not None]
            if parts:
                lines.append(f"{i}. " + ", ".join(parts))
            else:
                lines.append(f"{i}. (empty record)")

    return "\n".join(lines) if lines else "No data found."
