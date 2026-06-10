from langchain.tools import tool
import json
import re
from typing import Optional
from src.api import api_post
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
TOP_PRODUCTS_ENDPOINT = "/top-products"
POPULAR_PRODUCTS_ENDPOINT = "/popular-products"
SLOW_MOVING_PRODUCTS_ENDPOINT = "/slow-moving-products"
SALES_SUMMARY_ENDPOINT = "/sales-summary"
SALES_TRENDS_ENDPOINT = "/sales-trends"
TOP_CUSTOMER_ENDPOINT = "/top-customers"
TOP_VENDOR_ENDPOINT = "/top-vendors"
PURCHASE_SUMMARY_ENDPOINT = "/purchase-summary"
OUTSTANDING_SALES_INVOICES_ENDPOINT = "/reports/outstanding-sales-invoices"
OUTSTANDING_PURCHASE_INVOICES_ENDPOINT = "/reports/outstanding-purchase-invoices"
OVERDUE_INVOICES_ENDPOINT = "/reports/overdue-invoices"
LEDGERS_SEARCH_ENDPOINT = "/ledgers/search"
VENDORS_SEARCH_ENDPOINT = "/vendors"

api_cache_maxsize = 100
api_cache_ttl_secs = 600
api_cache = OrderedDict()


def make_cache_key(endpoint: str, body: dict) -> str:
    return f"{endpoint}::{json.dumps(body, sort_keys=True, ensure_ascii=False)}"


async def cached_api_post(endpoint: str, body: dict) -> dict:
    cache_key = make_cache_key(endpoint, body)
    now = time.monotonic()
    cached = api_cache.get(cache_key)
    if cached:
        age = now - cached["cached_at"]
        if age <= api_cache_ttl_secs:
            print(f"[CACHE HIT] {endpoint}")
            return copy.deepcopy(cached["result"])
        print(f"[CACHE EXPIRED] {endpoint}")
        api_cache.pop(cache_key, None)
    print(f"[CACHE MISS] {endpoint}")
    result = await api_post(endpoint, body=body)
    api_cache[cache_key] = {"cached_at": now, "result": copy.deepcopy(result)}
    while len(api_cache) >= api_cache_maxsize:
        api_cache.popitem(last=False)
    return copy.deepcopy(result)


def flatten_gst_summary_result(result: dict) -> dict:
    if not result.get("success"):
        return result
    raw = result.get("raw_response", {}) or {}
    gst_data = raw.get("data") or result.get("data") or {}
    rows = []
    if isinstance(gst_data, dict):
        for category_key, values in gst_data.items():
            if isinstance(values, dict):
                rows.append({"category": category_key, **values})
    grand_total = raw.get("grandTotal")
    if isinstance(grand_total, dict):
        rows.append({"category": "grandTotal", "name": "Grand Total", **grand_total})
    result["data"] = rows
    result["count"] = len(rows)
    result["period"] = raw.get("period")
    return result


def append_report_summary_row(result: dict, report_type: str) -> dict:
    if not result.get("success"):
        return result
    raw = result.get("raw_response", {}) or {}
    records = result.get("data", [])
    if records is None:
        records = []
    if not isinstance(records, list):
        records = [records]
    normalized = []
    for record in records:
        if isinstance(record, dict):
            normalized.append({**record, "recordType": report_type})
    summary = raw.get("summary")
    if isinstance(summary, dict):
        normalized.append({
            "recordType": "summary", "name": "Summary", **summary,
            "total_rows": raw.get("total_rows"), "total_pages": raw.get("total_pages"),
            "period": raw.get("period"),
        })
    result["data"] = normalized
    result["count"] = len(normalized)
    result["period"] = raw.get("period")
    return result


def flatten_sales_summary_result(result: dict) -> dict:
    if not result.get("success"):
        return result
    raw = result.get("raw_response", {}) or {}
    data = raw.get("data") or result.get("data") or {}
    rows = []
    if isinstance(data, dict):
        for category_key, values in data.items():
            if isinstance(values, dict):
                rows.append({"category": category_key, **values})
    result["data"] = rows
    result["count"] = len(rows)
    result["period"] = raw.get("period")
    return result


def flatten_purchase_summary_result(result: dict) -> dict:
    if not result.get("success"):
        return result
    raw = result.get("raw_response", {}) or {}
    data = raw.get("data") or result.get("data") or {}
    rows = []
    if isinstance(data, dict):
        for category_key, values in data.items():
            if isinstance(values, dict):
                rows.append({"category": category_key, **values})
    result["data"] = rows
    result["count"] = len(rows)
    result["period"] = raw.get("period")
    return result


def flatten_sales_trend_result(result: dict) -> dict:
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
                        rows.append({"period": period_key, "category": category_key, **values})
    result["data"] = rows
    result["count"] = len(rows)
    result["period"] = raw.get("period")
    return result


# ============================================================
# TOOLS — no fields/filters params. Each sends only native API params.
# ============================================================

@tool
async def get_gst_summary(from_date: str, to_date: str):
    """Fetch GST summary report for a date range."""
    body = {"companyId": COMPANY_ID, "from": from_date, "to": to_date}
    result = await cached_api_post(GST_SUMMARY_ENDPOINT, body=body)
    result = flatten_gst_summary_result(result)
    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_tds_outstanding(from_date: str = "", to_date: str = "", page: int = 1, limit: int = 10):
    """Fetch TDS outstanding report for a date range."""
    body = {"companyId": COMPANY_ID, "from": from_date or "", "to": to_date or "", "page": page, "limit": limit}
    result = await cached_api_post(TDS_OUTSTANDING_ENDPOINT, body=body)
    result = append_report_summary_row(result, "tdsOutstanding")
    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_tcs_outstanding(from_date: str = "", to_date: str = "", page: int = 1, limit: int = 10):
    """Fetch TCS outstanding report for a date range."""
    body = {"companyId": COMPANY_ID, "from": from_date or "", "to": to_date or "", "page": page, "limit": limit}
    result = await cached_api_post(TCS_OUTSTANDING_ENDPOINT, body=body)
    result = append_report_summary_row(result, "tcsOutstanding")
    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_top_products(from_date: str = "", to_date: str = "", sort_by: str = "revenue", limit: int = 10):
    """Fetch top/best-selling products by revenue, quantity, or profit."""
    body = {"companyId": COMPANY_ID, "from": from_date or "", "to": to_date or "", "sortBy": sort_by, "limit": limit}
    result = await cached_api_post(TOP_PRODUCTS_ENDPOINT, body=body)
    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_popular_products(period: str = "this_month", limit: int = 5):
    """Fetch popular/trending products for a given period."""
    body = {"companyId": COMPANY_ID, "period": period, "limit": limit}
    result = await cached_api_post(POPULAR_PRODUCTS_ENDPOINT, body=body)
    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_slow_moving_products(period: str = "current_fy", limit: int = 10):
    """Fetch slow-moving products with low turnover for a given period."""
    body = {"companyId": COMPANY_ID, "period": period, "limit": limit}
    result = await cached_api_post(SLOW_MOVING_PRODUCTS_ENDPOINT, body=body)
    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_sales_summary(from_date: str = "", to_date: str = "", group_by: str = "day",
                             vendor_id: Optional[int] = None, product_id: Optional[int] = None):
    """Fetch sales summary report for a date range, grouped by day/week/month."""
    body = {"companyId": COMPANY_ID, "from": from_date or "", "to": to_date or "",
            "groupBy": group_by, "vendorId": vendor_id, "productId": product_id}
    result = await cached_api_post(SALES_SUMMARY_ENDPOINT, body=body)
    result = flatten_sales_summary_result(result)
    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_sales_trend(period: str = "this_month", compare_with: str = "last_year"):
    """Fetch sales trend report comparing current period with a previous period."""
    body = {"companyId": COMPANY_ID, "period": period, "compareWith": compare_with}
    result = await cached_api_post(SALES_TRENDS_ENDPOINT, body=body)
    result = flatten_sales_trend_result(result)
    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_top_customer(period: str = "current_fy", sort_by: str = "revenue", limit: int = 10):
    """Fetch top customers by revenue, order count, or other metrics."""
    body = {"companyId": COMPANY_ID, "period": period, "sortBy": sort_by, "limit": limit}
    result = await cached_api_post(TOP_CUSTOMER_ENDPOINT, body=body)
    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_top_vendor(period: str = "current_fy", limit: int = 10):
    """Fetch top vendors by purchase amount or bill count."""
    body = {"companyId": COMPANY_ID, "period": period, "limit": limit}
    result = await cached_api_post(TOP_VENDOR_ENDPOINT, body=body)
    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_purchase_summary(from_date: str = "", to_date: str = "",
                                vendor_id: Optional[int] = None, product_id: Optional[int] = None):
    """Fetch purchase summary report for a date range."""
    body = {"companyId": COMPANY_ID, "from": from_date or "", "to": to_date or "",
            "vendorId": vendor_id, "productId": product_id}
    result = await cached_api_post(PURCHASE_SUMMARY_ENDPOINT, body=body)
    result = flatten_purchase_summary_result(result)
    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_search_ledgers(search_term: str = "", group_type: Optional[str] = None,
                              page: int = 1, limit: int = 10):
    """Search ledgers by name or group type (expense, income, etc.)."""
    body = {"companyId": COMPANY_ID, "searchTerm": search_term or "",
            "groupType": group_type, "page": page, "limit": limit}
    result = await cached_api_post(LEDGERS_SEARCH_ENDPOINT, body=body)
    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_search_vendors(search: str = "", limit: int = 10):
    """Search vendors/suppliers by name."""
    body = {"companyId": COMPANY_ID, "search": search or "", "limit": limit}
    result = await cached_api_post(VENDORS_SEARCH_ENDPOINT, body=body)
    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_customer(search: Optional[str] = "", limit: int = 10):
    """Search and retrieve customers/parties from Chapter-1 API.

    Args:
        search: Customer name or party name (substring match).
        limit: Number of records to fetch. Default is 10.
    """
    if search:
        stripped = search.strip().lower()
        non_specific = {"all", "all customers", "all parties", "all ledgers", "everyone", "every customer"}
        comma_separated = search.count(",") >= 2
        if stripped in non_specific or comma_separated:
            search = ""
    body = {"companyId": COMPANY_ID, "search": search or "", "limit": limit}
    result = await cached_api_post(CUSTOMER_ENDPOINT, body=body)
    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_customer_ledger(customer_id: int, from_date: str = "", to_date: str = "",
                               page: int = 1, limit: int = 10):
    """Get ledger/account statement details for a specific customer.

    Args:
        customer_id: Numeric customer ID.
        from_date: Start date in YYYY-MM-DD.
        to_date: End date in YYYY-MM-DD.
        page: Page number. Default 1.
        limit: Number of ledger rows. Default 10.
    """
    body = {"companyId": COMPANY_ID, "customerId": customer_id,
            "from": from_date or "", "to": to_date or "", "page": page, "limit": limit}
    result = await cached_api_post(CUSTOMER_LEDGER_ENDPOINT, body=body)
    if not isinstance(result, dict) or not result.get("success", False):
        print("[TOOL OUTPUT]", result)
        return json.dumps(result, ensure_ascii=False)
    raw = result.get("raw_response", {}) or {}
    ledger_record = dict(raw)
    ledger_record.pop("data", None)
    ledger_record["transactions"] = raw.get("data", [])
    ledger_record.setdefault("ledgerName", raw.get("ledgerName"))
    ledger_record.setdefault("period", raw.get("period"))
    final_result = {
        "success": True, "status_code": result.get("status_code"),
        "data": [ledger_record], "count": 1, "error": None, "raw_response": raw,
    }
    print("[TOOL OUTPUT]", final_result)
    return json.dumps(final_result, ensure_ascii=False)


@tool
async def get_stock_levels(from_date: Optional[str] = "", to_date: Optional[str] = "",
                            low_stock_only: bool = False, page: int = 1, limit: int = 10,
                            term: Optional[str] = "", sort_field: str = "name", sort_order: str = "asc"):
    """Get stock/inventory levels from Chapter-1 API.

    Args:
        from_date: Start date. Empty string if not given.
        to_date: End date. Empty string if not given.
        low_stock_only: True when user asks for low stock only.
        page: Page number. Default 1.
        limit: Number of records. Default 10.
        term: Product name, HSN code, SKU, or search keyword.
        sort_field: Sort field. Default "name".
        sort_order: "asc" or "desc". Default "asc".
    """
    needs_local_sort = bool(sort_field) and sort_field != "name"
    if needs_local_sort:
        needed = page * limit
        fetch_limit = min(max(needed, 200), 1000)
        fetch_page = 1
    else:
        fetch_limit = limit
        fetch_page = page
    body = {"companyId": COMPANY_ID, "from": from_date or "", "to": to_date or "",
            "lowStockOnly": low_stock_only, "page": fetch_page, "limit": fetch_limit, "term": term or ""}
    result = await cached_api_post(STOCK_LEVELS_ENDPOINT, body=body)

    raw = result.get("raw_response", {})
    if isinstance(raw, dict) and "total_rows" in raw:
        result["total_rows"] = raw["total_rows"]

    if low_stock_only and result.get("success"):
        records = result.get("data", [])
        if isinstance(records, list):
            records = [r for r in records if isinstance(r, dict) and r.get("isLowStock") is True]
            result["data"] = records
            result["count"] = len(records)

    records = result.get("data", [])
    if needs_local_sort and isinstance(records, list):
        def _sort_key(r):
            val = r.get(sort_field)
            if val is None:
                return (1, float('-inf'))
            try:
                return (0, float(val))
            except (TypeError, ValueError):
                return (0, 0)
        sorted_data = sorted(records, key=_sort_key, reverse=(sort_order or "").lower() == "desc")
        page = max(page, 1)
        limit = max(limit, 1)
        start = (page - 1) * limit
        end = start + limit
        result["data"] = sorted_data[start:end]
        result["count"] = len(result["data"])

    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_outstanding_sales_invoices(from_date: str = "", to_date: str = "",
                                          as_of_date: str = "", page: int = 1, limit: int = 50,
                                          sort_by: str = "daysOverdue", sort_order: str = "desc"):
    """Fetch outstanding/unpaid sales invoices with aging details."""
    body = {"companyId": COMPANY_ID, "from": from_date or "", "to": to_date or "",
            "asOfDate": as_of_date or "", "page": page, "limit": limit,
            "sortBy": sort_by, "sortOrder": sort_order}
    result = await cached_api_post(OUTSTANDING_SALES_INVOICES_ENDPOINT, body=body)
    result = append_report_summary_row(result, "outstandingSalesInvoices")
    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_outstanding_purchase_invoices(from_date: str = "", to_date: str = "",
                                             as_of_date: str = "", page: int = 1, limit: int = 50,
                                             sort_by: str = "daysOverdue", sort_order: str = "desc"):
    """Fetch outstanding/unpaid purchase invoices with aging details."""
    body = {"companyId": COMPANY_ID, "from": from_date or "", "to": to_date or "",
            "asOfDate": as_of_date or "", "page": page, "limit": limit,
            "sortBy": sort_by, "sortOrder": sort_order}
    result = await cached_api_post(OUTSTANDING_PURCHASE_INVOICES_ENDPOINT, body=body)
    result = append_report_summary_row(result, "outstandingPurchaseInvoices")
    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


@tool
async def get_overdue_invoices(invoice_type: str = "BOTH", as_of_date: str = "",
                                page: int = 1, limit: int = 50,
                                sort_by: str = "daysOverdue", sort_order: str = "desc"):
    """Fetch overdue invoices (both sales and purchase) beyond their due date."""
    body = {"companyId": COMPANY_ID, "invoiceType": invoice_type or "BOTH",
            "asOfDate": as_of_date or "", "page": page, "limit": limit,
            "sortBy": sort_by, "sortOrder": sort_order}
    result = await cached_api_post(OVERDUE_INVOICES_ENDPOINT, body=body)
    result = append_report_summary_row(result, "overdueInvoices")
    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)


tools = [
    get_customer, get_customer_ledger, get_stock_levels,
    get_gst_summary, get_tds_outstanding, get_tcs_outstanding,
    get_top_products, get_popular_products, get_slow_moving_products,
    get_sales_summary, get_sales_trend, get_top_customer, get_top_vendor,
    get_purchase_summary, get_search_ledgers, get_search_vendors,
    get_outstanding_sales_invoices, get_outstanding_purchase_invoices, get_overdue_invoices,
]

tools_dict = {tool.name: tool for tool in tools}
