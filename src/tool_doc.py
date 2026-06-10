import re

# ============================================================
# TOOL INTENT REGISTRY
# This is routing/tool metadata only. It does NOT contain business data.
# Business data always comes from the Chapter-1 API.
# ============================================================

from src.config import get_cfg

CITY_WORDS = get_cfg("cities", default=[
    "BANGALORE", "KOLKATA", "MUMBAI", "DELHI", "SURAT",
    "AHMEDABAD", "PUNE", "CHENNAI", "HYDERABAD",
    "BHIWANDI", "TAURU", "GUWAHATI", "PUNJAB",
])

TOOL_INTENT_REGISTRY = {
    "get_customer": {
        "category": "customer",
        "multi_call_ok": True,
        "description": "Search customers or parties and return id, name, opening balance and opening type.",
        "prompt_tips": "Customer name or ID (e.g. 'customer 76'): use search=customer_id_number. City/location: use search=city_name. Brand+city: search=brand_name.",
        "aliases": [
            "customer", "customers", "party", "parties", "client", "buyer", "grahak",
            "customer_report",
        ],
        "keywords": [
            "customer id", "party id", "customer name", "party name",
            "find customer", "search customer", "contact info", "customer list",
        ],
        "repair": {
            "keyword_args": {"nykaa": {"search": "Nykaa"}},
            "param_aliases": {"name": "search"},
        },
        "record_filters": [
            {"id_key": "party_id", "match_field": "id", "match_type": "exact"},
            {"id_key": "name", "match_field": "name", "match_type": "icontains"},
            {"id_key": "city", "match_field": "city", "match_type": "icontains"},
        ],
    },
    "get_customer_ledger": {
        "category": "customer_ledger",
        "multi_call_ok": True,
        "description": "Fetch customer ledger or account statement by customer_id; returns opening, current, closing balance and transactions.",
        "prompt_tips": "customer_id=int, dates YYYY-MM-DD.",
        "aliases": [
            "ledger", "account statement", "statement", "khata", "hisab",
            "customer_ledger",
        ],
        "keywords": [
            "customer ledger", "ledger balance", "closing balance", "current balance",
            "opening balance", "transactions", "transaction", "entries", "debit", "credit",
        ],
        "repair": {
            "extract_customer_id": True,
            "date_keywords": ["customer", "ledger", "closing", "opening", "balance"],
            "param_aliases": {"customer": "customer_id", "party": "customer_id", "id": "customer_id"},
        },
        "record_filters": [
            {"id_key": "party_id", "match_field": "customerId", "match_type": "exact"},
        ],
    },
    "get_stock_levels": {
        "category": "stock",
        "multi_call_ok": True,
        "description": "Fetch stock and inventory levels using product name, SKU or HSN; returns closing quantity/value, low stock and out-of-stock details.",
        "prompt_tips": "HSN: term=HSN_code. Low stock: low_stock_only=true.",
        "aliases": [
            "stock", "inventory", "product", "products", "item", "items",
            "maal", "jaththo", "satha", "stock_levels", "stock_report",
        ],
        "keywords": [
            "hsn", "hsn code", "sku", "closing quantity", "closing qty",
            "closing stock", "closing value", "closing rate", "low stock", "out of stock",
        ],
        "repair": {
            "param_aliases": {"name": "term"},
            "low_stock_only_keywords": ["low stock", "out of stock", "out-of-stock", "stock out", "low inventory"],
        },
        "record_filters": [
            {"id_key": "hsn_code", "match_field": "hsnCode", "match_type": "exact"},
            {"id_key": "sku", "match_field": "name", "match_type": "icontains"},
            {"id_key": "name", "match_field": "name", "match_type": "icontains"},
        ],
    },
    "get_gst_summary": {
        "category": "gst_report",
        "multi_call_ok": True,
        "description": "Fetch GST summary/report by date range. B2B / B2C Small / B2C Large / Exports / Nil-Rated / Exempt are GST categories. Use this tool (NOT get_sales_summary) when the user asks about B2B, B2C (Small/Large), exports, nil-rated, exempt, credit-notes, or grand-total data.",
        "prompt_tips": "Categories: B2B, B2C Large, B2C Small, exports, Nill-Rated, credit notes. Returns all categories in one call — no filtering needed.",
        "aliases": ["gst", "gstr", "gst summary", "gst report", "gstsummary", "b2csmall", "b2clarge", "b2c", "b2b"],
        "keywords": [
            "b2b", "b2c", "b2c invoices", "b2c count", "b2b invoices", "b2b count",
            "b2b b2c split", "b2c large", "b2clarge", "b2c small", "b2csmall",
            "exports", "export", "nil rated", "nilrated", "nillrated", "exempt",
            "igst", "cgst", "sgst", "cess",
            "taxable amount", "taxableamount", "invoice amount", "invoiceamount",
            "voucher count", "vouchercount",
            "grand total", "grandtotal", "total gst", "totalgst",
            "credit note", "creditnotes", "credit notes",
            "credit note registered", "creditnoteregistered",
            "credit note unregistered", "creditnoteunregistered",
        ],
        "repair": {
            "overwrite": True,
            "base_args": {"from_date": "", "to_date": ""},
            "date_keywords": ["gst", "b2b", "grand total", "b2c", "exports", "nil", "exempt"],
            "param_aliases": {"start": "from_date", "end": "to_date"},
        },
        "record_filters": [
            {"id_key": "gst_category", "match_field": "category", "match_type": "exact_case_insensitive"},
        ],
    },
    "get_tds_outstanding": {
        "category": "tds_report",
        "multi_call_ok": True,
        "description": "Fetch TDS outstanding/payable report by date range; supports section filters like 194C, 194J and 194I.",
        "prompt_tips": "Dates YYYY-MM-DD.",
        "aliases": ["tds", "tds outstanding", "tds payable", "tds report"],
        "keywords": ["tds", "outstanding", "payable", "pending", "section", "194c", "194j", "194i"],
        "repair": {
            "overwrite": True,
            "base_args": {"from_date": "", "to_date": ""},
            "date_keywords": ["tds"],
        },
        "record_filters": [
            {"id_key": "tds_section", "match_field": "section", "match_type": "exact"},
        ],
    },
    "get_tcs_outstanding": {
        "category": "tcs_report",
        "multi_call_ok": True,
        "description": "Fetch TCS outstanding/payable report by date range; supports section filters like 206C.",
        "prompt_tips": "Dates YYYY-MM-DD.",
        "aliases": ["tcs", "tcs outstanding", "tcs payable", "tcs report"],
        "keywords": ["tcs", "outstanding", "payable", "pending", "section", "206c"],
        "repair": {
            "overwrite": True,
            "base_args": {"from_date": "", "to_date": ""},
            "date_keywords": ["tcs"],
        },
        "record_filters": [
            {"id_key": "tds_section", "match_field": "section", "match_type": "exact"},
        ],
    },
    "get_top_products": {
        "category": "analytics",
        "multi_call_ok": True,
        "description": "Fetch top/best-selling products by revenue, quantity, profit, or other metrics for a date range.",
        "prompt_tips": "sort_by=revenue|quantity|profit, dates YYYY-MM-DD, limit=N. Default sort_by=revenue, limit=10.",
        "aliases": [
            "top products", "top product", "best selling", "bestseller",
            "top selling", "top items",
        ],
        "keywords": [
            "top products", "best selling", "bestseller",
            "top revenue", "top selling products", "top items",
        ],
        "repair": {
            "param_aliases": {"revenue": "sort_by", "sales": "sort_by", "quantity": "sort_by", "profit": "sort_by", "orders": "sort_by", "time": "period", "duration": "period"},
        },
    },
    "get_popular_products": {
        "category": "analytics",
        "multi_call_ok": True,
        "description": "Fetch popular/trending products for a given time period (this_month, last_month, etc.).",
        "prompt_tips": "period=this_month|last_month|this_quarter|last_quarter|this_year|last_year, limit=N.",
        "aliases": [
            "popular products", "popular product", "trending", "trending products",
            "whats popular", "what's popular", "in demand", "trending items",
        ],
        "keywords": [
            "popular", "trending", "trend", "in demand", "liked", "favorite",
        ],
        "repair": {
            "param_aliases": {"time": "period", "duration": "period"},
        },
    },
    "get_slow_moving_products": {
        "category": "analytics",
        "multi_call_ok": True,
        "description": "Fetch slow-moving products with low turnover for a given time period (current_fy, current_month, etc.).",
        "prompt_tips": "period=current_fy|current_month|last_month|last_fy, limit=N.",
        "aliases": [
            "slow moving products", "slow moving", "slow selling", "dead stock",
            "non moving", "low turnover", "not selling", "inventory slow",
        ],
        "keywords": [
            "slow moving", "slow selling", "dead stock", "non moving",
            "low turnover", "not selling", "low sales",
        ],
        "repair": {
            "param_aliases": {"time": "period", "duration": "period"},
        },
    },
    "get_sales_summary": {
        "category": "analytics",
        "multi_call_ok": True,
        "description": "Fetch sales summary report showing overall sales, item sales, and income breakdown for a date range.",
        "prompt_tips": "from_date/to_date YYYY-MM-DD, group_by=day|week|month|year.",
        "aliases": [
            "sales summary", "sales report", "total sales", "revenue summary",
            "overall sales", "sales breakdown", "bikri", "vikri",
        ],
        "keywords": [
            "sales summary", "sales report", "total sales", "overall sales",
            "revenue summary", "sales breakdown", "invoice summary",
        ],
        "repair": {
            "date_keywords": ["sales", "summary", "overall", "total", "bikri", "vikri"],
            "param_aliases": {"start": "from_date", "end": "to_date"},
        },
    },
    "get_sales_trend": {
        "category": "analytics",
        "multi_call_ok": True,
        "description": "Fetch sales trend report comparing current period sales with a previous period (e.g. this_month vs last_year).",
        "prompt_tips": "period=this_month|last_month|this_quarter|last_quarter|this_year|last_year, compare_with=last_year|last_month|last_quarter.",
        "aliases": [
            "sales trend", "sales trends", "growth comparison", "sales comparison",
            "month over month", "year over year", "trend analysis",
            "how sales changed",
        ],
        "keywords": [
            "sales trend", "trend", "growth", "comparison", "compare",
            "month over month", "year over year", "mom", "yoy",
            "increase", "decrease", "change",
        ],
        "repair": {
            "date_keywords": ["trend", "comparison", "growth", "change", "month over month", "year over year"],
            "param_aliases": {"time": "period", "compare": "compare_with"},
        },
    },
    "get_top_customer": {
        "category": "customer",
        "multi_call_ok": True,
        "description": "Fetch top customers by revenue, order count, or other metrics for a given period.",
        "prompt_tips": "period=current_fy|current_month|last_month|last_fy, sort_by=revenue|orderCount, limit=N.",
        "aliases": [
            "top customer", "top customers", "best customer", "best customers",
            "top buyer", "top buyers", "top client", "top clients",
            "highest spending",
        ],
        "keywords": [
            "top customer", "best customer", "top buyer", "top client",
            "highest spending", "most orders",
        ],
        "repair": {
            "param_aliases": {"revenue": "sort_by", "orders": "sort_by", "spending": "sort_by", "time": "period", "duration": "period"},
        },
    },
    "get_top_vendor": {
        "category": "vendor",
        "multi_call_ok": True,
        "description": "Fetch top vendors by purchase amount or bill count for a given period.",
        "prompt_tips": "period=current_fy|current_month|last_month|last_fy, limit=N.",
        "aliases": [
            "top vendor", "top vendors", "best vendor", "best vendors",
            "top supplier", "top suppliers",
        ],
        "keywords": [
            "top vendor", "best vendor", "top supplier",
            "highest purchase", "most bills",
        ],
        "repair": {
            "param_aliases": {"purchases": "sort_by", "bills": "sort_by", "spending": "sort_by", "time": "period", "duration": "period"},
        },
    },
    "get_purchase_summary": {
        "category": "purchase",
        "multi_call_ok": True,
        "description": "Fetch purchase summary report showing overall purchases, item purchases, and expenses for a date range.",
        "prompt_tips": "from_date/to_date YYYY-MM-DD.",
        "aliases": [
            "purchase summary", "purchase report", "total purchases",
            "expense summary", "bill summary", "kharidi", "khareedari",
        ],
        "keywords": [
            "purchase summary", "purchase report", "total purchases",
            "expense summary", "bill summary", "expenses",
            "kharidi",
        ],
        "repair": {
            "date_keywords": ["purchase", "kharidi", "khareedari", "expense", "bill"],
            "param_aliases": {"start": "from_date", "end": "to_date"},
        },
    },
    "get_search_ledgers": {
        "category": "ledger",
        "multi_call_ok": True,
        "description": "Search ledgers (party, expense, income, asset) by name + optional groupType. Has `city` field in response for location-based lookups.",
        "prompt_tips": "searchTerm is free-text for ledger name. group_type=expense|income|liability|asset|party.",
        "aliases": [
            "search ledger", "search ledgers", "find ledger", "find ledgers",
            "ledger search", "ledger group", "ledger groups",
            "find id", "ledger id", "expense id", "expense ledger",
            "office expenses id", "salary id", "rent id",
        ],
        "keywords": [
            "search ledger", "find ledger", "ledger group",
            "expense ledger", "income ledger",
            "office expenses", "expense", "expenses", "salary", "rent", "utility",
            "expense id", "ledger id", "find id",
            "gl group", "group type",
        ],
        "repair": {
            "param_aliases": {"name": "search_term"},
        },
        "record_filters": [
            {"id_key": "name", "match_field": "name", "match_type": "icontains"},
            {"id_key": "city", "match_field": "city", "match_type": "icontains"},
        ],
    },
    "get_search_vendors": {
        "category": "vendor",
        "multi_call_ok": True,
        "description": "Search vendors/suppliers by name.",
        "prompt_tips": "search=name (substring match), limit=N.",
        "aliases": [
            "search vendor", "search vendors", "find vendor", "find vendors",
            "vendor search", "supplier search",
        ],
        "keywords": [
            "search vendor", "find vendor", "vendor list", "supplier list",
            "vendors", "suppliers",
        ],
        "repair": {
            "param_aliases": {"name": "search"},
        },
        "record_filters": [
            {"id_key": "name", "match_field": "name", "match_type": "icontains"},
            {"id_key": "city", "match_field": "city", "match_type": "icontains"},
        ],
    },
    "get_outstanding_sales_invoices": {
        "category": "sales",
        "multi_call_ok": True,
        "description": "Fetch outstanding/unpaid sales invoices with aging details, invoice amounts, due dates, and summary totals.",
        "prompt_tips": "Lists invoices by DATE RANGE. ledgerName field contains customer names (e.g. 'B2C_MAHARASHTRA', 'B2C_ANDHRA PRADESH'). When invoice_no is given, scan results for matching invoiceNo — the invoice may be anywhere in the list.",
        "aliases": [
            "outstanding sales", "outstanding invoices", "pending invoices",
            "unpaid invoices", "sales due", "invoice outstanding",
            "due invoices", "invoice aging",
            "pending payments", "outstanding amount",
        ],
        "keywords": [
            "outstanding", "pending", "unpaid",
            "sales invoices", "all invoices", "invoice list",
            "receivable",
        ],
        "repair": {
            "date_keywords": ["outstanding", "invoice", "sales", "due", "pending", "unpaid", "receivable"],
            "param_aliases": {},
        },
        "record_filters": [
            {"id_key": "invoice_no", "match_field": "invoiceNo", "match_type": "exact"},
            {"id_key": "party_id", "match_field": "ledgerId", "match_type": "exact"},
        ],
    },
    "get_outstanding_purchase_invoices": {
        "category": "purchase",
        "multi_call_ok": True,
        "description": "Fetch outstanding/unpaid purchase invoices with aging details, bill amounts, due dates, and summary totals.",
        "prompt_tips": "Lists bills by DATE RANGE. When invoice_no is given, scan results for matching invoiceNo — the bill may be anywhere in the list.",
        "aliases": [
            "outstanding purchases", "outstanding purchase invoices", "pending purchase invoices",
            "unpaid purchase invoices", "bills payable", "creditors",
            "payable invoices", "vendor outstanding", "pending bills",
        ],
        "keywords": [
            "outstanding", "pending", "unpaid",
            "payable", "creditor", "bills payable",
        ],
        "repair": {
            "date_keywords": ["outstanding", "purchase", "invoice", "payable", "creditor", "vendor", "bill"],
            "param_aliases": {},
        },
        "record_filters": [
            {"id_key": "invoice_no", "match_field": "invoiceNo", "match_type": "exact"},
            {"id_key": "party_id", "match_field": "ledgerId", "match_type": "exact"},
        ],
    },
    "get_overdue_invoices": {
        "category": "sales",
        "multi_call_ok": True,
        "description": "Fetch overdue invoices (both sales receivables and purchase payables) past their due date, with aging details and summary totals.",
        "prompt_tips": "invoice_type=SALES|PURCHASE|BOTH, as_of_date YYYY-MM-DD. When invoice_no is given, scan results for matching invoiceNo — the invoice may be anywhere in the list.",
        "aliases": [
            "overdue invoices", "overdue bills", "overdue payments",
            "past due", "past due invoices", "delayed payments",
            "overdue receivables", "overdue payables",
            "overdue sales invoices", "overdue purchase invoices",
        ],
        "keywords": [
            "overdue", "past due", "delayed", "late", "expired",
            "overdue invoice", "overdue bills", "overdue payment",
            "invoice number lookup", "find by invoice",
        ],
        "repair": {
            "date_keywords": ["overdue", "invoice", "past due", "delayed", "late"],
            "param_aliases": {"type": "invoice_type", "sales": "invoice_type", "purchase": "invoice_type", "receivable": "invoice_type", "payable": "invoice_type"},
        },
        "record_filters": [
            {"id_key": "invoice_no", "match_field": "invoiceNo", "match_type": "exact"},
            {"id_key": "party_id", "match_field": "ledgerId", "match_type": "exact"},
        ],
    },
}

# Build alias→tool_name map for semantic search
TOOL_NAME_ALIASES = {}
for _tn, _meta in TOOL_INTENT_REGISTRY.items():
    for _alias in _meta.get("aliases", []):
        _alias_key = _alias.replace(" ", "_")
        if _alias_key not in TOOL_NAME_ALIASES:
            TOOL_NAME_ALIASES[_alias_key] = _tn
