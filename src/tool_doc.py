import re
from langchain_core.documents import Document

# ============================================================
# SIMPLE 6-TOOL REGISTRY
# This is routing/tool metadata only. It does NOT contain business data.
# Business data always comes from the Chapter-1 API.
#
# Each tool now includes:
#   repair: metadata annotation for generic arg repair (no if/elif needed)
#   prompt_tips: short instruction injected into the system prompt by build_system_prompt()
# ============================================================

from src.config import get_cfg

CITY_WORDS = get_cfg("cities", default=[
    "BANGALORE", "KOLKATA", "MUMBAI", "DELHI", "SURAT",
    "AHMEDABAD", "PUNE", "CHENNAI", "HYDERABAD",
    "BHIWANDI", "TAURU", "GUWAHATI", "PUNJAB",
])


def _build_field_triggers(field_aliases: dict) -> dict:
    """Invert field_aliases into keyword->[fields] mapping for field projection."""
    triggers: dict[str, list[str]] = {}
    for field, aliases in field_aliases.items():
        for alias in aliases:
            alias_l = alias.lower().strip()
            if alias_l:
                if alias_l not in triggers:
                    triggers[alias_l] = []
                if field not in triggers[alias_l]:
                    triggers[alias_l].append(field)
    return triggers


TOOL_INTENT_REGISTRY = {
    "get_customer": {
        "category": "customer",
        "multi_call_ok": True,
        "description": "Search customers or parties and return id, name, opening balance and opening type.",
        "prompt_tips": "Brand+city (e.g. Nykaa Bangalore): search=brand, filters=name.contains city. City/location alone also works: search=city_name.",
        "aliases": [
            "customer", "customers", "party", "parties", "client", "buyer", "grahak",
            "customer_report",
        ],
        "keywords": [
            "customer id", "party id", "customer name", "party name",
            "opening balance", "opening type", "find customer", "search customer",
        ],
        "fields": ["id", "name", "openingBalance", "openingType"],
        "default_fields": ["id", "name"],
        "field_aliases": {
            "id": ["customer id", "party id", "id"],
            "name": ["name", "customer name", "party name"],
            "openingBalance": ["opening balance", "opening"],
            "openingType": ["opening type", "opening"],
        },
        "repair": {
            "overwrite": True,
            "base_args": {"search": "", "fields": ["id", "name"]},
            "keyword_args": {"nykaa": {"search": "Nykaa"}},
            "city_filter": {"key": "name"},
            "field_triggers": {"opening balance": "openingBalance", "opening": "openingBalance"},
            "param_aliases": {"name": "search"},
        },
    },

    "get_customer_ledger": {
        "category": "customer_ledger",
        "description": "Fetch customer ledger or account statement by customer_id; returns opening, current, closing balance and transactions.",
        "prompt_tips": "customer_id=int, dates YYYY-MM-DD, fields=ledgerName,opening,current,closing,period[,transactions].",
        "aliases": [
            "ledger", "account statement", "statement", "khata", "hisab",
            "customer_ledger",
        ],
        "keywords": [
            "customer ledger", "ledger balance", "closing balance", "current balance",
            "opening balance", "transactions", "transaction", "entries", "debit", "credit",
        ],
        "fields": [
            "ledgerName", "glName", "opening", "current", "closing", "period",
            "total_rows", "total_pages", "transactions",
        ],
        "default_fields": ["ledgerName", "opening", "current", "closing", "period"],
        "field_aliases": {
            "ledgerName": ["ledger name"],
            "glName": ["gl name"],
            "opening": ["opening", "opening balance"],
            "current": ["current", "current balance"],
            "closing": ["closing", "closing balance"],
            "period": ["period", "from", "to"],
            "transactions": ["transaction", "transactions", "statement", "entry", "entries"],
        },
        "repair": {
            "overwrite": False,
            "extract_customer_id": True,
            "date_keywords": ["customer", "ledger", "closing", "opening", "balance"],
            "strict_field_keywords": {"sirf": ["closing"], "only": ["closing"]},
            "field_triggers": {"transactions": "transactions", "transaction": "transactions"},
            "fixed_fields": ["ledgerName", "opening", "current", "closing", "period"],
        },
    },

    "get_stock_levels": {
        "category": "stock",
        "multi_call_ok": True,
        "description": "Fetch stock and inventory levels using product name, SKU or HSN; returns closing quantity/value, low stock and out-of-stock details.",
            "prompt_tips": "HSN: term=HSN, filters=hsnCode. Low stock: low_stock_only=true. Qty compare: closingQty lt/gt. Can call w/ sort descending for max, ascending for min.",
        "aliases": [
            "stock", "inventory", "product", "products", "item", "items",
            "maal", "jaththo", "satha", "stock_levels", "stock_report",
        ],
        "keywords": [
            "hsn", "hsn code", "sku", "closing quantity", "closing qty",
            "closing stock", "closing value", "closing rate", "low stock", "out of stock",
        ],
        "fields": [
            "id", "name", "sku", "hsnCode", "group", "uom", "openingQty", "openingRate",
            "openingValue", "inwardQty", "inwardValue", "outwardQty", "outwardValue",
            "closingQty", "closingRate", "closingValue", "isLowStock", "isOutOfStock",
        ],
        "default_fields": ["name"],
        "field_aliases": {
            "name": ["name", "product name", "item name"],
            "id": ["id", "product id", "item id"],
            "sku": ["sku"],
            "hsnCode": ["hsn", "hsn code"],
            "closingQty": ["closing quantity", "closing qty", "quantity", "qty", "closing stock", "closing", "stock", "jaththo", "satha"],
            "closingValue": ["closing value", "value"],
            "closingRate": ["closing rate", "rate"],
            "isLowStock": ["low stock"],
            "isOutOfStock": ["out of stock"],
        },
        "repair": {
            "overwrite": False,
            "hsn_extract": True,
            "default_fields": ["name"],
            "field_triggers": {"value": "closingValue", "quantity": "closingQty", "qty": "closingQty"},
            "param_aliases": {"name": "term"},
            "low_stock_only_keywords": ["low stock", "out of stock", "out-of-stock", "stock out", "low inventory"],
        },
    },

    "get_gst_summary": {
        "category": "gst_report",
        "description": "Fetch GST summary/report by date range; supports B2B, B2C, exports, nil/exempt, credit notes and grand total rows.",
        "prompt_tips": "Categories: B2B=b2b, B2C Large=b2cLarge, B2C Small=b2cSmall, exports=exports, nil=nillRated, grandTotal=grandTotal. Single cat=>filter, multi=>no filter.",
        "aliases": ["gst", "gstr", "gst summary", "gst report", "gstsummary", "b2csmall", "b2clarge"],
        "keywords": [
            "b2b", "b2c", "b2c large", "b2clarge", "b2c small", "b2csmall",
            "exports", "export",
            "nil rated", "nilrated", "exempt",
            "igst", "cgst", "sgst", "cess",
            "taxable amount", "taxableamount", "invoice amount", "invoiceamount",
            "voucher count", "vouchercount",
            "grand total", "grandtotal", "total gst", "totalgst",
            "credit note", "creditnotes", "credit notes",
            "credit note registered", "creditnoteregistered",
            "credit note unregistered", "creditnoteunregistered",
        ],
        "fields": [
            "category", "name", "voucherCount", "taxableAmount", "igst", "cgst",
            "sgst", "cess", "tax", "invoiceAmount",
        ],
        "default_fields": ["category", "name", "voucherCount", "taxableAmount", "igst", "cgst", "sgst", "cess", "tax", "invoiceAmount"],
        "include_all_on_no_trigger": True,
        "field_aliases": {
            "category": ["category", "b2b", "b2c", "exports", "nil", "grand total"],
            "name": ["name"],
            "voucherCount": ["voucher count", "voucher"],
            "taxableAmount": ["taxable amount", "taxable"],
            "igst": ["igst"],
            "cgst": ["cgst"],
            "sgst": ["sgst"],
            "cess": ["cess"],
            "tax": ["total tax", "tax"],
            "invoiceAmount": ["invoice amount"],
        },
        "repair": {
            "overwrite": True,
            "base_args": {
                "from_date": "",
                "to_date": "",
                "fields": ["category", "name", "voucherCount", "taxableAmount", "igst", "cgst", "sgst", "cess", "tax", "invoiceAmount"],
            },
            "date_keywords": ["gst", "b2b", "grand total", "b2c", "exports", "nil", "exempt", "igst", "cgst", "sgst", "cess", "taxable", "invoice"],
            "remove_filters": True,
            "category_to_filter": True,
            "category_map": {
                "b2b": "b2b",
                "grand total": "grandTotal",
                "b2c small": "b2cSmall",
                "b2c large": "b2cLarge",
                "b2c": "b2cLarge",
                "exports": "exports",
                "nil rated": "nillRated",
                "nil": "nillRated",
                "exempt": "nillRated",
                "exempted": "nillRated",
            },
            "field_triggers": {
                "taxable amount": "taxableAmount",
                "invoice amount": "invoiceAmount",
                "igst": "igst",
                "cgst": "cgst",
                "sgst": "sgst",
                "cess": "cess",
            },
        },
    },

    "get_tds_outstanding": {
        "category": "tds_report",
        "description": "Fetch TDS outstanding/payable report by date range; supports section filters like 194C, 194J and 194I.",
        "prompt_tips": "Section filter (e.g. 194C): filters.section=194C. Dates YYYY-MM-DD.",
        "aliases": ["tds", "tds outstanding", "tds payable", "tds report"],
        "keywords": ["tds", "outstanding", "payable", "pending", "section", "194c", "194j", "194i"],
        "fields": [
            "recordType", "name", "section", "amount", "outstanding",
            "totalAmount", "totalOutstanding", "total_rows", "total_pages", "period",
        ],
        "default_fields": ["recordType", "name"],
        "always_include_fields": ["period"],
        "field_aliases": {
            "recordType": ["record type"],
            "section": ["section", "194c", "194j", "194i"],
            "amount": ["amount", "total amount"],
            "totalAmount": ["total amount", "amount", "total", "summary"],
            "outstanding": ["outstanding", "pending", "payable", "total outstanding"],
            "totalOutstanding": ["total outstanding", "outstanding", "pending", "payable", "total", "summary"],
            "total_rows": ["total rows", "total", "summary"],
            "total_pages": ["total pages", "total", "summary"],
            "period": ["period", "from", "to"],
        },
        "repair": {
            "overwrite": True,
            "date_keywords": ["tds"],
            "base_args": {
                "from_date": "",
                "to_date": "",
                "fields": ["recordType", "name", "totalAmount", "totalOutstanding", "period"],
            },
        },
    },

    "get_tcs_outstanding": {
        "category": "tcs_report",
        "description": "Fetch TCS outstanding/payable report by date range; supports section filters like 206C.",
        "prompt_tips": "Section filter (e.g. 206C): filters.section=206C. Dates YYYY-MM-DD.",
        "aliases": ["tcs", "tcs outstanding", "tcs payable", "tcs report"],
        "keywords": ["tcs", "outstanding", "payable", "pending", "section", "206c"],
        "fields": [
            "recordType", "name", "section", "amount", "outstanding",
            "totalAmount", "totalOutstanding", "total_rows", "total_pages", "period",
        ],
        "default_fields": ["recordType", "name"],
        "always_include_fields": ["period"],
        "field_aliases": {
            "recordType": ["record type"],
            "section": ["section", "206c"],
            "amount": ["amount", "total amount"],
            "totalAmount": ["total amount", "amount", "total", "summary"],
            "outstanding": ["outstanding", "pending", "payable", "total outstanding"],
            "totalOutstanding": ["total outstanding", "outstanding", "pending", "payable", "total", "summary"],
            "total_rows": ["total rows", "total", "summary"],
            "total_pages": ["total pages", "total", "summary"],
            "period": ["period", "from", "to"],
        },
        "repair": {
            "overwrite": True,
            "date_keywords": ["tcs"],
            "base_args": {
                "from_date": "",
                "to_date": "",
                "fields": ["recordType", "name", "totalAmount", "totalOutstanding", "period"],
            },
        },
    },
}


# Auto-generate no-space variants for category_map keys
# e.g., "b2c small" → also add "b2csmall"
for _tool_name, _meta in TOOL_INTENT_REGISTRY.items():
    _repair = _meta.get("repair", {})
    _cat_map = _repair.get("category_map")
    if _cat_map and isinstance(_cat_map, dict):
        _no_space_variants = {}
        for _kw, _val in _cat_map.items():
            if " " in _kw:
                _compact = _kw.replace(" ", "")
                if _compact not in _cat_map:
                    _no_space_variants[_compact] = _val
        _cat_map.update(_no_space_variants)


def get_field_triggers(tool_name: str) -> dict[str, list[str]]:
    """Build keyword->[fields] mapping from field_aliases for a tool."""
    meta = TOOL_INTENT_REGISTRY.get(tool_name, {})
    return _build_field_triggers(meta.get("field_aliases", {}))


def _query_matches(q: str, keyword: str) -> bool:
    """Match keyword in query using word boundaries for single words."""
    if " " in keyword:
        return keyword in q
    return bool(re.search(rf"\b{re.escape(keyword)}\b", q))


def infer_requested_fields_from_registry(user_query: str, tool_name: str) -> list[str]:
    """
    Generic field inference using TOOL_INTENT_REGISTRY.
    Replaces the old if/elif chain in infer_requested_fields().
    """
    meta = TOOL_INTENT_REGISTRY.get(tool_name)
    if not meta:
        return []

    q = (user_query or "").lower()
    fields = list(meta.get("default_fields", []))
    triggers = _build_field_triggers(meta.get("field_aliases", {}))
    any_trigger_matched = False

    for keyword, triggered_fields in triggers.items():
        if _query_matches(q, keyword):
            for f in triggered_fields:
                if f not in fields:
                    fields.append(f)
            any_trigger_matched = True

    # GST special: if no specific field asked, include all non-default fields
    if not any_trigger_matched and meta.get("include_all_on_no_trigger"):
        for f in meta.get("fields", []):
            if f not in fields:
                fields.append(f)

    # TDS/TCS special: always include certain fields
    for f in meta.get("always_include_fields", []):
        if f not in fields:
            fields.append(f)

    return list(dict.fromkeys(fields))


# Build alias→tool_name map from registry (replaces hardcoded TOOL_NAME_ALIASES in nodes.py)
TOOL_NAME_ALIASES = {}
for _tn, _meta in TOOL_INTENT_REGISTRY.items():
    for _alias in _meta.get("aliases", []):
        _alias_key = _alias.replace(" ", "_")
        if _alias_key not in TOOL_NAME_ALIASES:
            TOOL_NAME_ALIASES[_alias_key] = _tn

TOOL_DOCUMENTS = []

for tool_name, meta in TOOL_INTENT_REGISTRY.items():
    TOOL_DOCUMENTS.append(
        Document(
            page_content=f"""
Tool: {tool_name}
Category: {meta.get('category', '')}
Description: {meta.get('description', '')}
Aliases: {', '.join(meta.get('aliases', []))}
Keywords: {', '.join(meta.get('keywords', []))}
Fields: {', '.join(meta.get('fields', []))}
""".strip(),
            metadata={
                "tool_name": tool_name,
                "category": meta.get("category", ""),
            },
        )
    )
