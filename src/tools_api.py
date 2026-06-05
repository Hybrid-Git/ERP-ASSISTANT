from langchain.tools import tool
import json
import re
from typing import Optional, Any
from src.api_client import api_post
from src.config import COMPANY_ID
import time
import copy
from collections import OrderedDict

CUSTOMER_ENDPOINT = "/customers"
CUSTOMER_LEDGER_ENDPOINT = "/customers/ledger"
STOCK_LEVELS_ENDPOINT = "/inventory/stock"
GST_SUMMARY_ENDPOINT = "/reports/gst-summary"
TDS_OUTSTANDING_ENDPOINT = "/reports/tds-outstanding"
TCS_OUTSTANDING_ENDPOINT = "/reports/tcs-outstanding"

api_cache_maxsize = 100  # Max number of cached entries
api_cache_ttl_secs = 600  # 10 minutes
api_cache = OrderedDict()  # Cache dict to store API responses with insertion order


def make_cache_key(endpoint: str, body: dict) -> str:
    return f"{endpoint}::{json.dumps(body, sort_keys=True, ensure_ascii=False)}"


def cached_api_post(endpoint: str, body: dict) -> dict:
    """
    Caches API responses by endpoint + body.
    Important: returns deep copies because project_result mutates result["data"].
    """

    cache_key = make_cache_key(endpoint, body)
    now = time.monotonic()

    cached = api_cache.get(cache_key)

    if cached:
        cached_at = cached["cached_at"]
        age = now - cached_at

        if age <= api_cache_ttl_secs:
            print(f"[CACHE HIT] {endpoint}")
            return copy.deepcopy(cached["result"])

        print(f"[CACHE EXPIRED] {endpoint}")
        api_cache.pop(cache_key, None)

    print(f"[CACHE MISS] {endpoint}")
    result = api_post(endpoint, body=body)

    api_cache[cache_key] = {
        "cached_at": now,
        "result": copy.deepcopy(result),
    }
    while len(api_cache) > api_cache_maxsize:
        api_cache.popitem(last=False)
    return copy.deepcopy(result)

def normalize_fields(fields):
    if fields is None or fields == "":
        return []

    if isinstance(fields, list):
        return [
            str(field).strip()
            for field in fields
            if str(field).strip()
        ]

    if isinstance(fields, dict):
        return [
            str(field).strip()
            for field, enabled in fields.items()
            if enabled in [1, True, "1", "true", "True"]
        ]

    if isinstance(fields, str):
        return [
            field.strip()
            for field in fields.split(",")
            if field.strip()
        ]

    return []


def normalize_value(value):
    if value is None:
        return ""

    return str(value).strip().lower()


def to_number(value):
    try:
        if value is None or value == "":
            return None

        return float(value)
    except (TypeError, ValueError):
        return None


def match_filter(actual, expected):
    if isinstance(expected, dict):
        for operator, value in expected.items():
            operator = str(operator).lower().strip()

            if operator == "eq":
                if normalize_value(actual) != normalize_value(value):
                    return False

            elif operator == "contains":
                if normalize_value(value) not in normalize_value(actual):
                    return False

            elif operator == "in":
                if not isinstance(value, list):
                    return False

                allowed_values = [normalize_value(v) for v in value]

                if normalize_value(actual) not in allowed_values:
                    return False

            elif operator in ["gt", "gte", "lt", "lte"]:
                actual_num = to_number(actual)
                expected_num = to_number(value)

                if actual_num is None or expected_num is None:
                    return False

                if operator == "gt" and not actual_num > expected_num:
                    return False

                if operator == "gte" and not actual_num >= expected_num:
                    return False

                if operator == "lt" and not actual_num < expected_num:
                    return False

                if operator == "lte" and not actual_num <= expected_num:
                    return False

            else:
                return False

        return True

    if isinstance(expected, list):
        return normalize_value(actual) in [normalize_value(v) for v in expected]

    act_str = normalize_value(actual)
    exp_str = normalize_value(expected)

    # Numeric comparison when both are numbers (prevents "0" matching "-5" via substring).
    try:
        return float(act_str) == float(exp_str)
    except (ValueError, TypeError):
        pass

    # return exp_str == act_str or (exp_str and exp_str in act_str)
    return exp_str == act_str


def apply_filters(records, filters=None):
    if not filters:
        return records

    record_fields = set()
    for record in records:
        if isinstance(record, dict):
            record_fields.update(record.keys())

    # Check for invented filter fields that match no record data
    invented_fields = []
    for field in filters:
        if "." in field or " " in field or any(op in field for op in ["$", ">", "<", "="]):
            continue
        if record_fields and field not in record_fields:
            invented_fields.append(field)
    if invented_fields and record_fields:
        print(f"[WARN] Filter fields not found in records — returning empty: {invented_fields}")
        return []

    filtered_records = []

    for record in records:
        if not isinstance(record, dict):
            continue

        matched = True

        for field, expected in filters.items():
            # Skip operator-style filter keys (e.g., "name.contains", "closingQty gt").
            # These are LLM-generated filters meant for the API, not for local matching.
            if "." in field or " " in field or any(op in field for op in ["$", ">", "<", "="]):
                continue

            actual = record.get(field)

            if not match_filter(actual, expected):
                matched = False
                break

        if matched:
            filtered_records.append(record)

    return filtered_records


def project_fields(records, fields=None):
    selected_fields = normalize_fields(fields)

    if not selected_fields:
        return records

    # Treat "*" as "include all original fields" (wildcard).
    if "*" in selected_fields:
        return records

    projected_records = []

    for record in records:
        if not isinstance(record, dict):
            continue

        projected = {
            field: record.get(field)
            for field in selected_fields
            if field in record
        }

        # Prevent fake records like [{}] from being counted as valid data.
        if projected:
            projected_records.append(projected)

    # If ALL records were stripped due to field name mismatch
    # (e.g. LLM asked for "customer_id" but API field is "id"),
    # return original records instead of silently returning empty data.
    if not projected_records and records:
        # Check if the requested fields actually exist in any record
        actual_fields = set()
        for r in records:
            if isinstance(r, dict):
                actual_fields.update(r.keys())
        if not any(f in actual_fields for f in selected_fields):
            # LLM invented field names — return full records instead of empty
            return records
        return []


    return projected_records


def project_result(result: dict, fields=None, filters=None) -> dict:
    if not isinstance(result, dict):
        return {
            "success": False,
            "status_code": None,
            "data": [],
            "count": 0,
            "error": "API result is not a dictionary",
            "raw_response": result,
        }

    if not result.get("success", False):
        return result

    data = result.get("data", [])

    if data is None:
        data = []

    if not isinstance(data, list):
        data = [data]

    data = apply_filters(data, filters)
    data = project_fields(data, fields)

    result["data"] = data
    result["count"] = len(data)

    return result

def flatten_gst_summary_result(result: dict) -> dict:
    """
    Converts GST summary object into list rows so deterministic_final can handle it.
    """

    if not result.get("success"):
        return result

    raw = result.get("raw_response", {}) or {}
    gst_data = raw.get("data") or result.get("data") or {}

    rows = []

    if isinstance(gst_data, dict):
        for category_key, values in gst_data.items():
            if isinstance(values, dict):
                rows.append({
                    "category": category_key,
                    **values,
                })

    grand_total = raw.get("grandTotal")

    if isinstance(grand_total, dict):
        rows.append({
            "category": "grandTotal",
            "name": "Grand Total",
            **grand_total,
        })

    result["data"] = rows
    result["count"] = len(rows)
    result["period"] = raw.get("period")

    return result


def append_report_summary_row(result: dict, report_type: str) -> dict:
    """
    Adds summary row for TDS/TCS reports so totals are available in final data.
    """

    if not result.get("success"):
        return result

    raw = result.get("raw_response", {}) or {}
    records = result.get("data", [])

    if records is None:
        records = []

    if not isinstance(records, list):
        records = [records]

    normalized_records = []

    for record in records:
        if isinstance(record, dict):
            normalized_records.append({
                "recordType": report_type,
                **record,
            })

    summary = raw.get("summary")

    if isinstance(summary, dict):
        normalized_records.append({
            "recordType": "summary",
            "name": "Summary",
            **summary,
            "total_rows": raw.get("total_rows"),
            "total_pages": raw.get("total_pages"),
            "period": raw.get("period"),
        })

    result["data"] = normalized_records
    result["count"] = len(normalized_records)
    result["period"] = raw.get("period")

    return result
# ============================================================
# TOOLS
# ============================================================
@tool
def get_gst_summary(
    from_date: str,
    to_date: str,
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Fetch GST summary report for a date range.
    """

    body = {
        "companyId": COMPANY_ID,
        "from": from_date,
        "to": to_date,
    }

    if filters:
        body["filters"] = filters

    result = cached_api_post(GST_SUMMARY_ENDPOINT, body=body)
    result = flatten_gst_summary_result(result)
    result = project_result(result, fields=fields, filters=None)

    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
def get_tds_outstanding(
    from_date: str = "",
    to_date: str = "",
    page: int = 1,
    limit: int = 10,
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Fetch TDS outstanding report for a date range.
    """

    body = {
        "companyId": COMPANY_ID,
        "from": from_date or "",
        "to": to_date or "",
        "page": page,
        "limit": limit,
    }

    if filters:
        body["filters"] = filters

    result = cached_api_post(TDS_OUTSTANDING_ENDPOINT, body=body)
    result = append_report_summary_row(result, "tdsOutstanding")
    result = project_result(result, fields=fields, filters=None)

    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)

@tool
def get_tcs_outstanding(
    from_date: str = "",
    to_date: str = "",
    page: int = 1,
    limit: int = 10,
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Fetch TCS outstanding report for a date range.
    """

    body = {
        "companyId": COMPANY_ID,
        "from": from_date or "",
        "to": to_date or "",
        "page": page,
        "limit": limit,
    }

    if filters:
        body["filters"] = filters

    result = cached_api_post(TCS_OUTSTANDING_ENDPOINT, body=body)
    result = append_report_summary_row(result, "tcsOutstanding")
    result = project_result(result, fields=fields, filters=None)

    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)
@tool
def get_customer(
    search: Optional[str] = "",
    limit: int = 10,
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Search and retrieve customers/parties/ledgers from Chapter-1 API.

    Use this tool when the user asks about customers, customer name,
    customer ID, ledger party, customer opening balance, or wants to find
    a customer before fetching customer ledger.

    Args:
        search: Customer name or party name (substring match). Use empty string "" to return ALL customers.
            Do NOT pass concepts like "outside india" or city names as search; those are not customer names.
        limit: Number of records to fetch. Default is 10.
        fields: Optional output columns.
        filters: Optional exact filters.
    """

    # ── Sanitize search parameter ──
    # The LLM often treats `search` as a semantic filter instead of a name lookup.
    # These guards convert bad search terms into empty search (return all customers).
    raw_search = search
    if search:
        stripped = search.strip().lower()
        # 1. Fetch-all keywords → empty search
        non_specific = {"all", "all customers", "all parties", "all ledgers", "everyone", "every customer"}
        # 2. Comma-separated list of names (e.g. "kolkata, punjab, bangalore") → not a real name
        comma_separated_names = bool(re.search(r'\w+,\s*\w+', search))
        if stripped in non_specific or comma_separated_names:
            search = ""

    body = {
        "companyId": COMPANY_ID,
        "search": search or "",
        "limit": limit,
    }

    if filters:
        body["filters"] = filters

    result = cached_api_post(CUSTOMER_ENDPOINT, body=body)
    result = project_result(result, fields=fields, filters=None)

    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
def get_customer_ledger(
    customer_id: int,
    from_date: str = "",
    to_date: str = "",
    page: int = 1,
    limit: int = 10,
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Get ledger/account statement details for a specific customer.

    Use this tool when the user asks about customer ledger, account statement,
    ledger entries, transactions, opening balance, current balance,
    closing balance, debit/credit history, or customer account statement.

    Important:
        customer_id is required.
        If the user gives only customer name, call get_customer first.

    Args:
        customer_id: Numeric customer ID.
        from_date: Start date in YYYY-MM-DD. Empty string if not provided.
        to_date: End date in YYYY-MM-DD. Empty string if not provided.
        page: Page number. Default is 1.
        limit: Number of ledger rows. Default is 10.
        fields: Optional output columns.
        filters: Optional exact filters.
    """

    body = {
        "companyId": COMPANY_ID,
        "customerId": customer_id,
        "from": from_date or "",
        "to": to_date or "",
        "page": page,
        "limit": limit,
    }

    if filters:
        body["filters"] = filters

    result = cached_api_post(CUSTOMER_LEDGER_ENDPOINT, body=body)

    if not isinstance(result, dict) or not result.get("success", False):
        print("[TOOL OUTPUT]", result)
        return json.dumps(result, ensure_ascii=False)

    raw = result.get("raw_response", {}) or {}

    ledger_record = dict(raw)
    ledger_record.pop("data", None)
    ledger_record["transactions"] = raw.get("data", [])
    # Ensure required top-level keys exist even if API changes
    ledger_record.setdefault("ledgerName", raw.get("ledgerName"))
    ledger_record.setdefault("period", raw.get("period"))

    records = [ledger_record]
    records = project_fields(records, fields)

    final_result = {
        "success": True,
        "status_code": result.get("status_code"),
        "data": records,
        "count": len(records),
        "error": None,
        "raw_response": raw,
    }

    print("[TOOL OUTPUT]", final_result)
    return json.dumps(final_result, ensure_ascii=False)


def _safe_sort_key(record: dict, field: str):
    val = record.get(field)
    if val is None:
        return float("-inf")
    try:
        return float(val)
    except (ValueError, TypeError):
        return str(val)


@tool
def get_stock_levels(
    from_date: Optional[str] = "",
    to_date: Optional[str] = "",
    low_stock_only: bool = False,
    page: int = 1,
    limit: int = 10,
    term: Optional[str] = "",
    sort_field: str = "name",
    sort_order: str = "asc",
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Get stock/inventory levels from Chapter-1 API.

    Use this tool when the user asks about stock levels, inventory,
    product stock, low stock, out of stock, closing quantity, HSN code,
    SKU, inward quantity, outward quantity, closing value, or item stock.

    Args:
        from_date: Start date. Empty string if not given.
        to_date: End date. Empty string if not given.
        low_stock_only: True when user asks for low stock only.
        page: Page number. Default is 1.
        limit: Number of records. Default is 10.
        term: Product name, HSN code, SKU, or search keyword.
        sort_field: Sort field. Default is "name".
        sort_order: "asc" or "desc". Default is "asc".
        fields: Optional output columns.
        filters: Optional exact filters.
    """

    needs_local_sort = bool(sort_field) and sort_field != "name"
    if needs_local_sort:
        needed = page * limit
        fetch_limit = min(max(needed, 200), 1000)
        fetch_page = 1
    else:
        fetch_limit = limit
        fetch_page = page
    body = {
        "companyId": COMPANY_ID,
        "from": from_date or "",
        "to": to_date or "",
        "lowStockOnly": low_stock_only,
        "page": fetch_page,
        "limit": fetch_limit,
        "term": term or "",
    }

    if filters:
        body["filters"] = filters

    result = cached_api_post(STOCK_LEVELS_ENDPOINT, body=body)

    raw_len = len(result.get("data", []) or []) if isinstance(result.get("data"), list) else 0
    print(f"[STOCK DEBUG] After cached_api_post: raw_data_len={raw_len}, fields={fields}")

    if low_stock_only and fields is not None:
        if isinstance(fields, list) and "isLowStock" not in fields:
            fields = fields + ["isLowStock"]
        elif isinstance(fields, dict):
            fields = {**fields, "isLowStock": True}
    result = project_result(result, fields=fields, filters=None)
    projected_len = len(result.get("data", []) or []) if isinstance(result.get("data"), list) else 0
    print(f"[STOCK DEBUG] After project_result: data_len={projected_len}, count={result.get('count')}")

    # Preserve total_rows from raw response so count/aggregate queries still work
    raw = result.get("raw_response", {})
    if isinstance(raw, dict) and "total_rows" in raw:
        result["total_rows"] = raw["total_rows"]
    if low_stock_only and result.get("success"):
        records = result.get("data", [])

        records = [
            record for record in records
            if isinstance(record, dict) and record.get("isLowStock") is True
        ]

        result["data"] = records
        result["count"] = len(records)

    records = result.get("data", [])
    if needs_local_sort and isinstance(result.get("data"), list):

        def _sort_key(r):
            val = r.get(sort_field)
            if val is None:
                return (1, float('-inf'))
            try:
                return (0, float(val))
            except (TypeError, ValueError):
                return (0, 0)

        sorted_data = sorted(
            records,
            key=_sort_key,
            reverse=(sort_order or "").lower() == "desc",
        )
        start = (page - 1) * limit
        end = start + limit
        result["data"] = sorted_data[start:end]
        result["count"] = len(result["data"])
        print(f"[STOCK DEBUG] After local sort: data_len={len(result['data'])}, count={result['count']}, sort_field={sort_field}, sort_order={sort_order}, limit={limit}")

    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)

tools = [
    get_customer,
    get_customer_ledger,
    get_stock_levels,
    get_gst_summary,
    get_tds_outstanding,
    get_tcs_outstanding,
]

tools_dict = {tool.name: tool for tool in tools}