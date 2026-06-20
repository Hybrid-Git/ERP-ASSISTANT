from langchain.tools import tool
import json
import re  # noqa: F401
from typing import Optional, Any
from app.services.api import api_post
from app.core.config import COMPANY_ID
import time
import copy
from collections import OrderedDict
import asyncio  
import logging

logger = logging.getLogger("erp_assistant.tools_api")


CUSTOMER_ENDPOINT = "/aiAnalytics/customers"
CUSTOMER_LEDGER_ENDPOINT = "/aiAnalytics/customers/ledger"
STOCK_LEVELS_ENDPOINT = "/aiAnalytics/inventory/stock"
GST_SUMMARY_ENDPOINT = "/aiAnalytics/reports/gst-summary"
TDS_OUTSTANDING_ENDPOINT = "/aiAnalytics/reports/tds-outstanding"
TCS_OUTSTANDING_ENDPOINT = "/aiAnalytics/reports/tcs-outstanding"
TOP_PRODUCTS_ENDPOINT = "/aiAnalytics/top-products"
POPULAR_PRODUCTS_ENDPOINT = "/aiAnalytics/popular-products"
SLOW_MOVING_PRODUCTS_ENDPOINT = "/aiAnalytics/slow-moving-products"
SALES_SUMMARY_ENDPOINT = "/aiAnalytics/sales-summary"
SALES_TRENDS_ENDPOINT = "/aiAnalytics/sales-trends"
TOP_CUSTOMER_ENDPOINT = "/aiAnalytics/top-customers"
TOP_VENDOR_ENDPOINT = "/aiAnalytics/top-vendors"
PURCHASE_SUMMARY_ENDPOINT = "/aiAnalytics/purchase-summary"
OUTSTANDING_SALES_INVOICES_ENDPOINT = "/aiAnalytics/reports/outstanding-sales-invoices"
OUTSTANDING_PURCHASE_INVOICES_ENDPOINT = "/aiAnalytics/reports/outstanding-purchase-invoices"
OVERDUE_INVOICES_ENDPOINT = "/aiAnalytics/reports/overdue-invoices"
LEDGERS_SEARCH_ENDPOINT = "/aiAnalytics/ledgers/search"
VENDORS_SEARCH_ENDPOINT = "/aiAnalytics/vendors"

api_cache_maxsize = 100  # Max number of cached entries
api_cache_ttl_secs = 600  # 10 minutes
api_cache = OrderedDict()  # Cache dict to store API responses with insertion order


def make_cache_key(endpoint: str, body: dict) -> str:
    return f"{endpoint}::{json.dumps(body, sort_keys=True, ensure_ascii=False)}"


async def cached_api_post(endpoint: str, body: dict) -> dict:
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
            logger.info(
                        "ERP API cache hit",
                        extra={
                            "tool_endpoint": endpoint,
                            "cache_status": "hit",
                        },
                        )
            return copy.deepcopy(cached["result"])

        logger.info(
                    "ERP API cache expired",
                    extra={
                        "tool_endpoint": endpoint,
                        "cache_status": "expired",
                    },
                    )
        api_cache.pop(cache_key, None)

    logger.info(
                "ERP API cache miss",
                extra={
                    "tool_endpoint": endpoint,
                    "cache_status": "miss",
                },
                )
    result = await api_post(endpoint, body=body)

    logger.info(
                "ERP API tool result received",
                extra={
                    "tool_endpoint": endpoint,
                    "status": "success" if result.get("success") else "failed",
                    "status_code": result.get("status_code"),
                    "record_count": result.get("count"),
                },
                )
    api_cache[cache_key] = {
        "cached_at": now,
        "result": copy.deepcopy(result),
    }
    while len(api_cache) >= api_cache_maxsize:
        api_cache.popitem(last=False)
    return copy.deepcopy(result)

def _snake_to_camel(name: str) -> str:
    if "_" not in name:
        return name
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _field_in_set(field_set: set, field: str) -> bool:
    if field in field_set:
        return True
    camel = _snake_to_camel(field)
    if camel != field and camel in field_set:
        return True
    return False


def _field_in_record(record: dict, field: str) -> bool:
    if field in record:
        return True
    camel = _snake_to_camel(field)
    if camel != field and camel in record:
        return True
    return False


def _get_field(record: dict, field: str):
    val = record.get(field)
    if val is not None:
        return val
    camel = _snake_to_camel(field)
    if camel != field:
        val = record.get(camel)
        if val is not None:
            return val
        if camel in record:
            return record[camel]
    if field in record:
        return record[field]
    return None


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

    s = str(value).strip().lower()
    s = re.sub(r"[_]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


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

    # Comma-separated expected values → OR match (e.g. "b2cSmall,b2cLarge")
    if isinstance(expected, str) and "," in expected:
        parts = [p.strip() for p in expected.split(",")]
        return normalize_value(actual) in [normalize_value(p) for p in parts]

    return exp_str == act_str


def apply_filters(records, filters=None):
    if not filters:
        return records

    record_fields = set()
    for record in records:
        if isinstance(record, dict):
            record_fields.update(record.keys())

    usable_filters = {}
    skipped_fields = []
    for field, expected in filters.items():
        if "." in field or " " in field or any(op in field for op in ["$", ">", "<", "="]):
            skipped_fields.append(field)
            continue
        if record_fields and not _field_in_set(record_fields, field):
            skipped_fields.append(field)
            continue
        usable_filters[field] = expected

    if skipped_fields:
        print(f"[FILTER] Skipped fields not found in response data: {skipped_fields}")
        if record_fields:
            print(f"[FILTER] Available response fields: {sorted(record_fields)}")

    if not usable_filters:
        if skipped_fields:
            print(f"[FILTER] No usable filter fields — returning unfiltered data")
        return records

    filtered_records = []

    for record in records:
        if not isinstance(record, dict):
            continue

        matched = True

        for field, expected in usable_filters.items():
            actual = _get_field(record, field)

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
            field: _get_field(record, field)
            for field in selected_fields
            if _field_in_record(record, field)
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

    # Ensure fields used in filters survive projection (e.g. ledgerName)
    # so filtered records retain context for the LLM
    if fields is not None and filters is not None:
        filter_keys = list(filters.keys())
        norm_fields = normalize_fields(fields)
        if norm_fields and "*" not in norm_fields:
            for k in filter_keys:
                if k not in norm_fields:
                    if isinstance(fields, list):
                        fields = list(fields) + [k]
                    elif isinstance(fields, dict):
                        fields = dict(fields)
                        fields[k] = True
                    elif isinstance(fields, str):
                        fields = fields + "," + k
                    break

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
                **record,
                "recordType": report_type,
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


def flatten_sales_summary_result(result: dict) -> dict:
    """
    Converts sales summary object into list rows by category.
    """

    if not result.get("success"):
        return result

    raw = result.get("raw_response", {}) or {}
    data = raw.get("data") or result.get("data") or {}

    rows = []

    if isinstance(data, dict):
        for category_key, values in data.items():
            if isinstance(values, dict):
                rows.append({
                    "category": category_key,
                    **values,
                })

    result["data"] = rows
    result["count"] = len(rows)
    result["period"] = raw.get("period")

    return result


def flatten_purchase_summary_result(result: dict) -> dict:
    """
    Converts purchase summary object into list rows by category.
    """

    if not result.get("success"):
        return result

    raw = result.get("raw_response", {}) or {}
    data = raw.get("data") or result.get("data") or {}

    rows = []

    if isinstance(data, dict):
        for category_key, values in data.items():
            if isinstance(values, dict):
                rows.append({
                    "category": category_key,
                    **values,
                })

    result["data"] = rows
    result["count"] = len(rows)
    result["period"] = raw.get("period")

    return result



def flatten_sales_trend_result(result: dict) -> dict:
    """
    Converts sales trend object (current/previous/growth with categories)
    into list rows.
    """

    if not result.get("success"):
        return result

    raw = result.get("raw_response", {}) or {}
    data = raw.get("data") or result.get("data") or {}

    rows = []

    if isinstance(data, dict):
        for period_key, categories in data.items():
            if isinstance(categories, dict):
                for category_key, values in categories.items():
                    if isinstance(values, dict):
                        rows.append({
                            "period": period_key,
                            "category": category_key,
                            **values,
                        })

    result["data"] = rows
    result["count"] = len(rows)
    result["period"] = raw.get("period")

    return result


# ============================================================
# TOOLS
# ============================================================
@tool
async def get_gst_summary(
    from_date: str = "",
    to_date: str = "",
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

    result = await cached_api_post(GST_SUMMARY_ENDPOINT, body=body)
    result = flatten_gst_summary_result(result)
    result = project_result(result, fields=fields, filters=filters)

        # print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_tds_outstanding(
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

    result = await cached_api_post(TDS_OUTSTANDING_ENDPOINT, body=body)
    result = append_report_summary_row(result, "tdsOutstanding")
    result = project_result(result, fields=fields, filters=filters)

        # print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)

@tool
async def get_tcs_outstanding(
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

    result = await cached_api_post(TCS_OUTSTANDING_ENDPOINT, body=body)
    result = append_report_summary_row(result, "tcsOutstanding")
    result = project_result(result, fields=fields, filters=filters)

        # print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_top_products(
    from_date: str = "",
    to_date: str = "",
    sort_by: str = "revenue",
    limit: int = 10,
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Fetch top/best-selling products by revenue, quantity, or other metrics.

    Use this tool when the user asks about top products, best-selling products,
    highest revenue products, most sold items, popular products, etc.

    Args:
        from_date: Start date in YYYY-MM-DD. Empty string if not provided.
        to_date: End date in YYYY-MM-DD. Empty string if not provided.
        sort_by: Sort metric ("revenue", "quantity", "profit", etc.). Default "revenue".
        limit: Number of top products to return. Default 10.
        fields: Optional output columns.
        filters: Optional exact filters.
    """

    body = {
        "companyId": COMPANY_ID,
        "from": from_date or "",
        "to": to_date or "",
        "sortBy": sort_by,
        "limit": limit,
    }

    if filters:
        body["filters"] = filters

    result = await cached_api_post(TOP_PRODUCTS_ENDPOINT, body=body)
    result = project_result(result, fields=fields, filters=filters)

        # print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_popular_products(
    period: str = "this_month",
    limit: int = 5,
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Fetch popular/trending products for a given period.

    Use this tool when the user asks about popular products, trending products,
    most viewed products, popular items, what's popular, current trends, etc.

    Args:
        period: Time period ("this_month", "last_month", "this_quarter", "last_quarter", "this_year", "last_year"). Default "this_month".
        limit: Number of popular products to return. Default 5.
        fields: Optional output columns.
        filters: Optional exact filters.
    """

    body = {
        "companyId": COMPANY_ID,
        "period": period,
        "limit": limit,
    }

    if filters:
        body["filters"] = filters

    result = await cached_api_post(POPULAR_PRODUCTS_ENDPOINT, body=body)
    result = project_result(result, fields=fields, filters=filters)

        # print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_slow_moving_products(
    period: str = "current_fy",
    limit: int = 10,
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Fetch slow-moving products with low turnover for a given period.

    Use this tool when the user asks about slow-moving products, slow selling items,
    dead stock, non-moving items, low turnover products, inventory that isn't selling, etc.

    Args:
        period: Time period ("current_fy", "current_month", "last_month", "last_fy", etc.). Default "current_fy".
        limit: Number of products to return. Default 10.
        fields: Optional output columns.
        filters: Optional exact filters.
    """

    body = {
        "companyId": COMPANY_ID,
        "period": period,
        "limit": limit,
    }

    if filters:
        body["filters"] = filters

    result = await cached_api_post(SLOW_MOVING_PRODUCTS_ENDPOINT, body=body)
    result = project_result(result, fields=fields, filters=filters)

        # print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_sales_summary(
    from_date: str = "",
    to_date: str = "",
    group_by: str = "day",
    vendor_id: Optional[int] = None,
    product_id: Optional[int] = None,
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Fetch sales summary report for a date range, grouped by day/week/month.

    Use this tool when the user asks about sales summary, total sales,
    sales report, overall sales, revenue summary, sales breakdown,
    invoice summary, paid/unpaid sales, etc.

    Args:
        from_date: Start date in YYYY-MM-DD. Empty string if not provided.
        to_date: End date in YYYY-MM-DD. Empty string if not provided.
        group_by: Grouping period ("day", "week", "month", "year"). Default "day".
        vendor_id: Filter by vendor ID. None if not provided.
        product_id: Filter by product ID. None if not provided.
        fields: Optional output columns.
        filters: Optional exact filters.
    """

    body = {
        "companyId": COMPANY_ID,
        "from": from_date or "",
        "to": to_date or "",
        "groupBy": group_by,
        "vendorId": vendor_id,
        "productId": product_id,
    }

    if filters:
        body["filters"] = filters

    result = await cached_api_post(SALES_SUMMARY_ENDPOINT, body=body)
    result = flatten_sales_summary_result(result)
    result = project_result(result, fields=fields, filters=filters)

        # print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_sales_trend(
    period: str = "this_month",
    compare_with: str = "last_year",
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Fetch sales trend report comparing current period with a previous period.

    Use this tool when the user asks about sales trends, growth comparison,
    sales comparison, month-over-month sales, year-over-year sales,
    how sales have changed, trend analysis, etc.

    Args:
        period: Current period ("this_month", "last_month", "this_quarter", "last_quarter", "this_year", "last_year"). Default "this_month".
        compare_with: Period to compare against ("last_year", "last_month", "last_quarter"). Default "last_year".
        fields: Optional output columns.
        filters: Optional exact filters.
    """

    body = {
        "companyId": COMPANY_ID,
        "period": period,
        "compareWith": compare_with,
    }

    if filters:
        body["filters"] = filters

    result = await cached_api_post(SALES_TRENDS_ENDPOINT, body=body)
    result = flatten_sales_trend_result(result)
    result = project_result(result, fields=fields, filters=filters)

        # print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_top_customer(
    period: str = "current_fy",
    sort_by: str = "revenue",
    limit: int = 10,
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Fetch top customers by revenue, order count, or other metrics.
    IMPORTANT: Returns only revenue-ranked aggregate data (not full customer details).
    Do NOT use this to search for a specific customer by name or city — use get_customer instead.

    Use this tool when the user asks about top customers, best customers,
    highest spending customers, top buyers, top clients, etc.
    Do NOT use for: customer search by name/city/location.

    Args:
        period: Time period ("current_fy", "current_month", "last_month", "last_fy", etc.). Default "current_fy".
        sort_by: Sort metric ("revenue", "orderCount", etc.). Default "revenue".
        limit: Number of top customers to return. Default 10.
        fields: Optional output columns.
        filters: Optional exact filters.
    """

    body = {
        "companyId": COMPANY_ID,
        "period": period,
        "sortBy": sort_by,
        "limit": limit,
    }

    if filters:
        body["filters"] = filters

    result = await cached_api_post(TOP_CUSTOMER_ENDPOINT, body=body)
    result = project_result(result, fields=fields, filters=filters)

        # print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_top_vendor(
    period: str = "current_fy",
    limit: int = 10,
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Fetch top vendors by purchase amount or bill count.

    Use this tool when the user asks about top vendors, best vendors,
    top suppliers, highest purchase vendors, vendor ranking, etc.

    Args:
        period: Time period ("current_fy", "current_month", "last_month", "last_fy", etc.). Default "current_fy".
        limit: Number of top vendors to return. Default 10.
        fields: Optional output columns.
        filters: Optional exact filters.
    """

    body = {
        "companyId": COMPANY_ID,
        "period": period,
        "limit": limit,
    }

    if filters:
        body["filters"] = filters

    result = await cached_api_post(TOP_VENDOR_ENDPOINT, body=body)
    result = project_result(result, fields=fields, filters=filters)

        # print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_purchase_summary(
    from_date: str = "",
    to_date: str = "",
    vendor_id: Optional[int] = None,
    product_id: Optional[int] = None,
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Fetch purchase summary report for a date range.

    Use this tool when the user asks about purchase summary, total purchases,
    purchase report, expense summary, bill summary, what was purchased, etc.

    Args:
        from_date: Start date in YYYY-MM-DD. Empty string if not provided.
        to_date: End date in YYYY-MM-DD. Empty string if not provided.
        vendor_id: Filter by vendor ID. None if not provided.
        product_id: Filter by product ID. None if not provided.
        fields: Optional output columns.
        filters: Optional exact filters.
    """

    body = {
        "companyId": COMPANY_ID,
        "from": from_date or "",
        "to": to_date or "",
        "vendorId": vendor_id,
        "productId": product_id,
    }

    if filters:
        body["filters"] = filters

    result = await cached_api_post(PURCHASE_SUMMARY_ENDPOINT, body=body)
    result = flatten_purchase_summary_result(result)
    result = project_result(result, fields=fields, filters=filters)

        # print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_search_ledgers(
    search_term: str = "",
    group_type: Optional[str] = None,
    page: int = 1,
    limit: int = 10,
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Search ledgers by name or group type (expense, income, etc.).

    Use this tool when the user asks about ledgers, search ledgers,
    find ledgers, ledger groups, expense ledgers, income ledgers, etc.

    Args:
        search_term: Keyword to search ledger names by. Empty string returns all.
        group_type: Filter by group type ("expense", "income", "liability", "asset", etc.). None if not provided.
        page: Page number. Default 1.
        limit: Number of records per page. Default 10.
        fields: Optional output columns.
        filters: Optional exact filters.
    """

    body = {
        "companyId": COMPANY_ID,
        "searchTerm": search_term or "",
        "groupType": group_type,
        "page": page,
        "limit": limit,
    }

    if filters:
        body["filters"] = filters

    result = await cached_api_post(LEDGERS_SEARCH_ENDPOINT, body=body)
    result = project_result(result, fields=fields, filters=filters)

        # print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_search_vendors(
    search: str = "",
    limit: int = 10,
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Search vendors/suppliers by name.

    Use this tool when the user asks about vendors, suppliers,
    search vendors, find vendors, vendor list, supplier list, etc.

    Args:
        search: Vendor/supplier name to search for (substring match). Empty string returns all.
        limit: Number of records to return. Default 10.
        fields: Optional output columns.
        filters: Optional exact filters.
    """

    body = {
        "companyId": COMPANY_ID,
        "search": search or "",
        "limit": limit,
    }

    if filters:
        body["filters"] = filters

    result = await cached_api_post(VENDORS_SEARCH_ENDPOINT, body=body)
    result = project_result(result, fields=fields, filters=filters)

        # print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_customer(
    search: str = "",
    limit: int = 100,
):
    """
    Search and retrieve customers/parties/ledgers from Chapter-1 API.

    Use this tool when the user asks about customers, customer name,
    customer ID, ledger party, customer opening balance, or wants to find
    a customer before fetching customer ledger.

    Args:
        search: Customer name or party name (substring match). Use empty string "" to return ALL customers.
            Do NOT pass concepts like "outside india" or city names as search; those are not customer names.
        limit: Number of records to fetch. Default is 100.
    """

    # ── Sanitize search parameter ──
    # The LLM often treats `search` as a semantic filter instead of a name lookup.
    # These guards convert bad search terms into empty search (return all customers).
    raw_search = search  # noqa: F841
    if search:
        stripped = search.strip().lower()
        # 1. Fetch-all keywords → empty search
        non_specific = {"all", "all customers", "all parties", "all ledgers", "everyone", "every customer"}
        # 2. Comma-separated list of names (e.g. "kolkata, punjab, bangalore") → not a real name
        comma_separated_names = search.count(",") >= 2
        if stripped in non_specific or comma_separated_names:
            search = ""

    body = {
        "companyId": COMPANY_ID,
        "search": search or "",
        "limit": limit,
    }

    result = await cached_api_post(CUSTOMER_ENDPOINT, body=body)
    result = project_result(result, fields=None, filters=None)

        # print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_customer_ledger(
    customer_id: int,
    from_date: str = "",
    to_date: str = "",
    page: int = 1,
    limit: int = 50,
):
    """
    Get ledger/account statement details for a specific customer.

    Use this tool when the user asks about customer ledger, account statement,
    ledger entries, transactions, opening balance, current balance,
    closing balance, debit/credit history, or customer account statement.

    Important:
        customer_id is required and MUST come from a prior get_customer call result.
        If the user gives only customer name, call get_customer first to find the customer_id.

    Args:
        customer_id: Numeric customer ID (from get_customer result).
        from_date: Start date in YYYY-MM-DD. Empty string if not provided.
        to_date: End date in YYYY-MM-DD. Empty string if not provided.
        page: Page number. Default is 1.
        limit: Number of ledger rows. Default is 50.
    """

    body = {
        "companyId": COMPANY_ID,
        "customerId": customer_id,
        "from": from_date or "",
        "to": to_date or "",
        "page": page,
        "limit": limit,
    }

    result = await cached_api_post(CUSTOMER_LEDGER_ENDPOINT, body=body)

    if not isinstance(result, dict) or not result.get("success", False):
            # print("[TOOL OUTPUT]", result)
        return json.dumps(result, ensure_ascii=False)

    raw = result.get("raw_response", {}) or {}

    ledger_record = dict(raw)
    ledger_record.pop("data", None)
    ledger_record["transactions"] = raw.get("data", [])
    # Ensure required top-level keys exist even if API changes
    ledger_record.setdefault("ledgerName", raw.get("ledgerName"))
    ledger_record.setdefault("period", raw.get("period"))

    records = [ledger_record]
    records = project_fields(records, None)

    final_result = {
        "success": True,
        "status_code": result.get("status_code"),
        "data": records,
        "count": len(records),
        "error": None,
        "raw_response": raw,
    }

        # print("[TOOL OUTPUT]", final_result)
    return json.dumps(final_result, ensure_ascii=False)




@tool
async def get_stock_levels(
    from_date: str = "",
    to_date: str = "",
    low_stock_only: bool = False,
    page: int = 1,
    limit: int = 10,
    term: str = "",
    sort_field: str = "name",
    sort_order: str = "asc",
    filters: Optional[dict] = None,
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

    result = await cached_api_post(STOCK_LEVELS_ENDPOINT, body=body)

    result = project_result(result, fields=None, filters=filters)

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
        page = max(page,1)
        limit = max(limit,1)
        start = (page - 1) * limit
        end = start + limit
        result["data"] = sorted_data[start:end]
        result["count"] = len(result["data"])
        print(f"[STOCK DEBUG] After local sort: data_len={len(result['data'])}, count={result['count']}, sort_field={sort_field}, sort_order={sort_order}, limit={limit}")

        # print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


async def _fetch_invoices_with_pagination(
    endpoint: str,
    body: dict,
    fields,
    filters: Optional[dict],
    report_type: str,
) -> dict:
    """
    Fetches invoice data with automatic pagination when local filters
    return empty but total_rows > limit (API didn't filter server-side).

    Hybrid approach:
    1. Try high-limit single call (limit = min(total_rows, 10000))
    2. Fallback: paginate up to 10 pages with limit=500 each
    """
    MAX_HIGH_LIMIT = 10000
    MAX_FALLBACK_PAGES = 10
    FALLBACK_PAGE_LIMIT = 500

    result = await cached_api_post(endpoint, body=body)
    raw = result.get("raw_response", {}) or {}
    api_data = list(result.get("data", []) or []) if isinstance(result.get("data"), list) else []
    total_rows = raw.get("total_rows", 0) or 0
    orig_limit = body.get("limit", 50)

    needs_more = (
        filters
        and total_rows > orig_limit
        and len(api_data) > 0
        and len(apply_filters(list(api_data), filters)) == 0
    )

    if needs_more:
        total_pages = raw.get("total_pages", 0) or 1
        high_limit = min(total_rows, MAX_HIGH_LIMIT)
        if high_limit > orig_limit:
            high_body = {**body, "limit": high_limit, "page": 1}
            high_result = await cached_api_post(endpoint, body=high_body)
            high_data = high_result.get("data", [])
            if isinstance(high_data, list) and len(high_data) > len(api_data):
                api_data = list(high_data)

        if len(apply_filters(list(api_data), filters)) == 0:
            max_pages = min(total_pages + 1, MAX_FALLBACK_PAGES + 1)
            for p in range(2, max_pages + 1):
                page_body = {**body, "page": p, "limit": FALLBACK_PAGE_LIMIT}
                page_result = await cached_api_post(endpoint, body=page_body)
                page_data = page_result.get("data", [])
                if isinstance(page_data, list):
                    api_data.extend(page_data)

    combined = {
        "success": True,
        "status_code": 200,
        "data": api_data,
        "count": len(api_data),
        "raw_response": raw,
    }
    combined = append_report_summary_row(combined, report_type)
    combined = project_result(combined, fields=fields, filters=filters)
    print(f"[PAGINATION] Collected {len(api_data)} records → {len(combined.get('data', []))} after filter")

    return combined


@tool
async def get_outstanding_sales_invoices(
    from_date: str = "",
    to_date: str = "",
    as_of_date: str = "",
    page: int = 1,
    limit: int = 50,
    sort_by: str = "daysOverdue",
    sort_order: str = "desc",
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Fetch outstanding/unpaid sales invoices with aging details.

    Use this tool when the user asks about outstanding invoices, pending payments,
    unpaid sales invoices, due invoices, overdue invoices, invoice aging, etc.

    Args:
        from_date: Start date YYYY-MM-DD. Empty if not provided.
        to_date: End date YYYY-MM-DD. Empty if not provided.
        as_of_date: Snapshot date YYYY-MM-DD. Empty for API default.
        page: Page number. Default 1.
        limit: Records per page. Default 50.
        sort_by: Sort field ("daysOverdue", "invoiceDate", "outstanding", etc.). Default "daysOverdue".
        sort_order: Sort order ("asc", "desc"). Default "desc".
        fields: Optional output columns.
        filters: Optional exact filters.
    """

    body = {
        "companyId": COMPANY_ID,
        "from": from_date or "",
        "to": to_date or "",
        "asOfDate": as_of_date or "",
        "page": page,
        "limit": limit,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    if filters:
        body["filters"] = filters

    result = await _fetch_invoices_with_pagination(
        OUTSTANDING_SALES_INVOICES_ENDPOINT, body, fields, filters, "outstandingSalesInvoices"
    )

        # print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_outstanding_purchase_invoices(
    from_date: str = "",
    to_date: str = "",
    as_of_date: str = "",
    page: int = 1,
    limit: int = 50,
    sort_by: str = "daysOverdue",
    sort_order: str = "desc",
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Fetch outstanding/unpaid purchase invoices with aging details.

    Use this tool when the user asks about outstanding purchases, pending bills,
    unpaid purchase invoices, bills payable, creditor outstanding, etc.

    Args:
        from_date: Start date YYYY-MM-DD. Empty if not provided.
        to_date: End date YYYY-MM-DD. Empty if not provided.
        as_of_date: Snapshot date YYYY-MM-DD. Empty for API default.
        page: Page number. Default 1.
        limit: Records per page. Default 50.
        sort_by: Sort field ("daysOverdue", "invoiceDate", "outstanding", etc.). Default "daysOverdue".
        sort_order: Sort order ("asc", "desc"). Default "desc".
        fields: Optional output columns.
        filters: Optional exact filters.
    """

    body = {
        "companyId": COMPANY_ID,
        "from": from_date or "",
        "to": to_date or "",
        "asOfDate": as_of_date or "",
        "page": page,
        "limit": limit,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    if filters:
        body["filters"] = filters

    result = await _fetch_invoices_with_pagination(
        OUTSTANDING_PURCHASE_INVOICES_ENDPOINT, body, fields, filters, "outstandingPurchaseInvoices"
    )

        # print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_overdue_invoices(
    invoice_type: str = "BOTH",
    as_of_date: str = "",
    page: int = 1,
    limit: int = 50,
    sort_by: str = "daysOverdue",
    sort_order: str = "desc",
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Fetch overdue invoices (both sales and purchase) beyond their due date.

    Use this tool when the user asks about overdue invoices, past due bills,
    delayed payments, overdue receivables, overdue payables, etc.

    Args:
        invoice_type: Filter by type ("SALES", "PURCHASE", or "BOTH"). Default "BOTH".
        as_of_date: Snapshot date YYYY-MM-DD. Empty for API default.
        page: Page number. Default 1.
        limit: Records per page. Default 50.
        sort_by: Sort field ("daysOverdue", "invoiceDate", "outstanding", etc.). Default "daysOverdue".
        sort_order: Sort order ("asc", "desc"). Default "desc".
        fields: Optional output columns.
        filters: Optional exact filters.
    """

    body = {
        "companyId": COMPANY_ID,
        "invoiceType": invoice_type or "BOTH",
        "asOfDate": as_of_date or "",
        "page": page,
        "limit": limit,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    if filters:
        body["filters"] = filters

    result = await _fetch_invoices_with_pagination(
        OVERDUE_INVOICES_ENDPOINT, body, fields, filters, "overdueInvoices"
    )

        # print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


tools = [
    get_customer,
    get_customer_ledger,
    get_stock_levels,
    get_gst_summary,
    get_tds_outstanding,
    get_tcs_outstanding,
    get_top_products,
    get_popular_products,
    get_slow_moving_products,
    get_sales_summary,
    get_sales_trend,
    get_top_customer,
    get_top_vendor,
    get_purchase_summary,
    get_search_ledgers,
    get_search_vendors,
    get_outstanding_sales_invoices,
    get_outstanding_purchase_invoices,
    get_overdue_invoices
]

tools_dict = {tool.name: tool for tool in tools}