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
        "prompt_tips": "City/location alone: use search=city_name, NOT filters.city. Brand+city (e.g. Nykaa Bangalore): search=brand, filters=name.contains city. Customer name: search=name.",
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
            "field_triggers": {"opening balance": "openingBalance",                 "opening": "openingBalance",
                "opening type": "openingType"},
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
        "prompt_tips": "HSN: term=HSN, filters=hsnCode. Low stock: low_stock_only=true. Qty compare: closingQty lt/gt. Can call w/ sort descending for max, ascending for min. VALID filters ONLY: hsnCode, lowStockOnly. DO NOT copy filters from GST/customer tools.",
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
        "prompt_tips": "Categories: B2B=b2b, B2C Large=b2cLarge, B2C Small=b2cSmall, exports=exports, nil=nilRated, grandTotal=grandTotal. Single cat=>filter, multi=>no filter. NO search/term/limit params — these do NOT exist for this tool.",
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
            "date_keywords": ["gst", "b2b", "grand total", "b2c", "exports", "nil", "exempt"],
            "remove_filters": True,
            "category_to_filter": True,
            "category_map": {
                "b2b": "b2b",
                "grand total": "grandTotal",
                "b2c small": "b2cSmall",
                "b2c large": "b2cLarge",
                "b2c": "b2cLarge",
                "exports": "exports",
                "nil rated": "nilRated",
                "nil": "nilRated",
                "exempt": "nilRated",
                "exempted": "nilRated",
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
            "name": ["name", "party name", "customer name", "customerName"],
            "section": ["section", "194c", "194j", "194i"],
            "amount": ["amount", "total amount"],
            "totalAmount": ["total amount", "amount", "total", "summary", "tdsAmount"],
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
            "name": ["name", "party name", "customer name", "customerName"],
            "section": ["section", "206c"],
            "amount": ["amount", "total amount"],
            "totalAmount": ["total amount", "amount", "total", "summary", "tcsAmount"],
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
    "get_top_products": {
        "category": "analytics",
        "multi_call_ok": True,
        "description": "Fetch top/best-selling products by revenue, quantity, profit, or other metrics for a date range.",
        "prompt_tips": "sort_by=revenue|quantity|profit, dates YYYY-MM-DD, limit=N. Default sort_by=revenue, limit=10.",
        "aliases": [
            "top products", "top product", "best selling", "bestseller",
            "top selling", "popular products", "top items",
        ],
        "keywords": [
            "top products", "best selling", "bestseller", "popular",
            "top revenue", "top selling products", "top items",
        ],
        "fields": [
            "id", "name", "sku", "totalQty", "totalRevenue",
            "totalTaxableAmount", "totalTaxAmount", "totalDiscount",
            "totalProfit", "orderCount", "avgRate",
        ],
        "default_fields": ["name", "totalRevenue", "totalQty"],
        "field_aliases": {
            "id": ["product id", "id"],
            "name": ["name", "product name", "item name"],
            "sku": ["sku"],
            "totalQty": ["total quantity", "quantity", "qty", "total qty", "sold"],
            "totalRevenue": ["total revenue", "revenue", "sales", "income"],
            "totalTaxableAmount": ["taxable amount", "taxable"],
            "totalTaxAmount": ["tax amount", "tax"],
            "totalDiscount": ["discount", "total discount"],
            "totalProfit": ["profit", "total profit"],
            "orderCount": ["order count", "orders", "total orders"],
            "avgRate": ["average rate", "avg rate", "rate"],
        },
        "repair": {
            "overwrite": False,
            "default_fields": ["name", "totalRevenue", "totalQty"],
            "field_triggers": {
                "revenue": "totalRevenue",
                "sales": "totalRevenue",
                "quantity": "totalQty",
                "qty": "totalQty",
                "profit": "totalProfit",
                "orders": "orderCount",
            },
        },
    },
    "get_popular_products": {
        "category": "analytics",
        "multi_call_ok": True,
        "description": "Fetch popular/trending products for a given time period (this_month, last_month, etc.).",
        "prompt_tips": "period=this_month|last_month|this_quarter|last_quarter|this_year|last_year, limit=N. Default period=this_month, limit=5.",
        "aliases": [
            "popular products", "popular product", "trending", "trending products",
            "whats popular", "what's popular", "in demand", "trending items",
        ],
        "keywords": [
            "popular", "trending", "trend", "in demand", "liked", "favorite",
        ],
        "fields": [
            "id", "name", "sku", "totalQty", "totalRevenue",
            "totalTaxableAmount", "totalTaxAmount", "totalDiscount",
            "totalProfit", "orderCount", "avgRate",
        ],
        "default_fields": ["name", "totalQty"],
        "field_aliases": {
            "id": ["product id", "id"],
            "name": ["name", "product name", "item name"],
            "sku": ["sku"],
            "totalQty": ["total quantity", "quantity", "qty", "total qty", "sold", "popular"],
            "totalRevenue": ["total revenue", "revenue", "sales"],
            "totalTaxableAmount": ["taxable amount", "taxable"],
            "totalTaxAmount": ["tax amount", "tax"],
            "totalDiscount": ["discount", "total discount"],
            "totalProfit": ["profit", "total profit"],
            "orderCount": ["order count", "orders", "total orders"],
            "avgRate": ["average rate", "avg rate", "rate"],
        },
        "repair": {
            "overwrite": False,
            "default_fields": ["name", "totalQty"],
        },
    },
    "get_slow_moving_products": {
        "category": "analytics",
        "multi_call_ok": True,
        "description": "Fetch slow-moving products with low turnover for a given time period (current_fy, current_month, etc.).",
        "prompt_tips": "period=current_fy|current_month|last_month|last_fy, limit=N. Default period=current_fy, limit=10.",
        "aliases": [
            "slow moving products", "slow moving", "slow selling", "dead stock",
            "non moving", "low turnover", "not selling", "inventory slow",
        ],
        "keywords": [
            "slow moving", "slow selling", "dead stock", "non moving",
            "low turnover", "not selling", "low sales",
        ],
        "fields": [
            "id", "name", "sku", "totalQty", "totalRevenue",
            "totalTaxableAmount", "totalTaxAmount", "totalDiscount",
            "totalProfit", "orderCount", "avgRate",
        ],
        "default_fields": ["name", "totalQty"],
        "field_aliases": {
            "id": ["product id", "id"],
            "name": ["name", "product name", "item name"],
            "sku": ["sku"],
            "totalQty": ["total quantity", "quantity", "qty", "total qty", "sold", "stock"],
            "totalRevenue": ["total revenue", "revenue", "sales"],
            "totalTaxableAmount": ["taxable amount", "taxable"],
            "totalTaxAmount": ["tax amount", "tax"],
            "totalDiscount": ["discount", "total discount"],
            "totalProfit": ["profit", "total profit"],
            "orderCount": ["order count", "orders", "total orders"],
            "avgRate": ["average rate", "avg rate", "rate"],
        },
        "repair": {
            "overwrite": False,
            "default_fields": ["name", "totalQty"],
        },
    },
    "get_sales_summary": {
        "category": "analytics",
        "multi_call_ok": True,
        "description": "Fetch sales summary report showing overall sales, item sales, and income breakdown for a date range.",
        "prompt_tips": "from_date/to_date YYYY-MM-DD, group_by=day|week|month|year. Categories: overall(items)->totalAmount/totalSales, totalTaxableAmount, totalTax, invoiceCount, etc.",
        "aliases": [
            "sales summary", "sales report", "total sales", "revenue summary",
            "overall sales", "sales breakdown", "bikri", "vikri",
        ],
        "keywords": [
            "sales summary", "sales report", "total sales", "overall sales",
            "revenue summary", "sales breakdown", "invoice summary",
            "paid", "unpaid", "outstanding sales",
        ],
        "fields": [
            "category", "totalAmount", "totalSales", "totalTaxableAmount",
            "totalTax", "totalOutstanding", "totalPaid", "totalIncome",
            "totalQuantity", "invoiceCount", "averageInvoiceValue",
        ],
        "default_fields": ["category", "totalAmount", "totalSales", "totalTaxableAmount", "totalTax", "totalOutstanding", "totalPaid", "invoiceCount", "averageInvoiceValue"],
        "include_all_on_no_trigger": True,
        "field_aliases": {
            "category": ["category", "overall", "items", "income"],
            "totalAmount": ["total amount", "sales", "overall sales", "bikri"],
            "totalSales": ["total sales", "item sales", "items total"],
            "totalTaxableAmount": ["taxable amount", "taxable"],
            "totalTax": ["total tax", "tax", "gst"],
            "totalOutstanding": ["outstanding", "unpaid", "pending"],
            "totalPaid": ["paid", "total paid", "payment"],
            "totalIncome": ["income", "total income"],
            "totalQuantity": ["total quantity", "quantity", "qty"],
            "invoiceCount": ["invoice count", "invoices", "vouchers"],
            "averageInvoiceValue": ["average invoice value", "avg invoice", "average"],
        },
        "repair": {
            "overwrite": False,
            "date_keywords": ["sales", "summary", "overall", "total", "bikri", "vikri"],
            "default_fields": ["category", "totalAmount", "totalSales", "totalTaxableAmount", "totalTax", "totalOutstanding", "totalPaid", "invoiceCount", "averageInvoiceValue"],
        },
    },
    "get_sales_trend": {
        "category": "analytics",
        "multi_call_ok": True,
        "description": "Fetch sales trend report comparing current period sales with a previous period (e.g. this_month vs last_year).",
        "prompt_tips": "period=this_month|last_month|this_quarter|last_quarter|this_year|last_year, compare_with=last_year|last_month|last_quarter. Hierarchical: period(current/previous/growth) > category(overall/items/income) > values. Fields vary by period: growth has percentage/absolute.",
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
        "fields": [
            "period", "category", "totalAmount", "totalSales", "totalTaxableAmount",
            "totalTax", "totalOutstanding", "totalPaid", "totalIncome",
            "totalQuantity", "invoiceCount", "averageInvoiceValue",
            "percentage", "absolute",
        ],
        "default_fields": ["period", "category", "totalAmount", "totalSales", "totalTaxableAmount", "totalTax", "totalOutstanding", "totalPaid", "invoiceCount", "percentage", "absolute"],
        "include_all_on_no_trigger": True,
        "field_aliases": {
            "period": ["period", "current", "previous", "growth"],
            "category": ["category", "overall", "items", "income"],
            "totalAmount": ["total amount", "sales", "amount"],
            "totalSales": ["total sales", "item sales"],
            "totalTaxableAmount": ["taxable amount", "taxable"],
            "totalTax": ["total tax", "tax"],
            "totalOutstanding": ["outstanding", "unpaid"],
            "totalPaid": ["paid", "total paid"],
            "totalIncome": ["income", "total income"],
            "totalQuantity": ["total quantity", "quantity", "qty"],
            "invoiceCount": ["invoice count", "invoices"],
            "averageInvoiceValue": ["average invoice value", "avg invoice"],
            "percentage": ["percentage", "growth %", "percent", "%"],
            "absolute": ["absolute", "absolute change", "difference"],
        },
        "repair": {
            "overwrite": False,
            "date_keywords": ["trend", "comparison", "growth", "change", "month over month", "year over year"],
            "default_fields": ["period", "category", "totalAmount", "totalSales", "totalTaxableAmount", "totalTax", "totalOutstanding", "totalPaid", "invoiceCount", "percentage", "absolute"],
        },
    },
    "get_top_customer": {
        "category": "customer",
        "multi_call_ok": True,
        "description": "Fetch top customers by revenue, order count, or other metrics for a given period.",
        "prompt_tips": "period=current_fy|current_month|last_month|last_fy, sort_by=revenue|orderCount, limit=N. Default period=current_fy, sort_by=revenue, limit=10.",
        "aliases": [
            "top customer", "top customers", "best customer", "best customers",
            "top buyer", "top buyers", "top client", "top clients",
            "highest spending",
        ],
        "keywords": [
            "top customer", "best customer", "top buyer", "top client",
            "highest spending", "most orders",
        ],
        "fields": [
            "id", "name", "totalRevenue", "orderCount",
        ],
        "default_fields": ["name", "totalRevenue", "orderCount"],
        "field_aliases": {
            "id": ["customer id", "id"],
            "name": ["name", "customer name", "party name"],
            "totalRevenue": ["total revenue", "revenue", "sales", "spending"],
            "orderCount": ["order count", "orders", "total orders"],
        },
        "repair": {
            "overwrite": False,
            "default_fields": ["name", "totalRevenue", "orderCount"],
            "param_aliases": {"revenue": "sort_by"},
        },
    },
    "get_top_vendor": {
        "category": "vendor",
        "multi_call_ok": True,
        "description": "Fetch top vendors by purchase amount or bill count for a given period.",
        "prompt_tips": "period=current_fy|current_month|last_month|last_fy, limit=N. Default period=current_fy, limit=10. Fields: name, totalPurchases, billCount.",
        "aliases": [
            "top vendor", "top vendors", "best vendor", "best vendors",
            "top supplier", "top suppliers",
        ],
        "keywords": [
            "top vendor", "best vendor", "top supplier",
            "highest purchase", "most bills",
        ],
        "fields": [
            "id", "name", "totalPurchases", "billCount",
        ],
        "default_fields": ["name", "totalPurchases", "billCount"],
        "field_aliases": {
            "id": ["vendor id", "supplier id", "id"],
            "name": ["name", "vendor name", "supplier name"],
            "totalPurchases": ["total purchases", "purchases", "purchase amount", "spending"],
            "billCount": ["bill count", "bills", "total bills"],
        },
        "repair": {
            "overwrite": False,
            "default_fields": ["name", "totalPurchases", "billCount"],
        },
    },
    "get_purchase_summary": {
        "category": "purchase",
        "multi_call_ok": True,
        "description": "Fetch purchase summary report showing overall purchases, item purchases, and expenses for a date range.",
        "prompt_tips": "from_date/to_date YYYY-MM-DD. Categories: overall(items/expenses)->totalAmount/totalPurchases/totalExpenses, totalTaxableAmount, totalTax, billCount, etc.",
        "aliases": [
            "purchase summary", "purchase report", "total purchases",
            "expense summary", "bill summary", "kharidi", "khareedari",
        ],
        "keywords": [
            "purchase summary", "purchase report", "total purchases",
            "expense summary", "bill summary", "expenses",
            "kharidi",
        ],
        "fields": [
            "category", "totalAmount", "totalPurchases", "totalExpenses",
            "totalTaxableAmount", "totalTax", "totalOutstanding",
            "totalPaid", "totalQuantity", "billCount", "averageBillValue",
        ],
        "default_fields": ["category", "totalAmount", "totalPurchases", "totalExpenses", "totalTaxableAmount", "totalTax", "totalOutstanding", "totalPaid", "billCount", "averageBillValue"],
        "include_all_on_no_trigger": True,
        "field_aliases": {
            "category": ["category", "overall", "items", "expenses"],
            "totalAmount": ["total amount", "purchases", "overall purchases", "kharidi"],
            "totalPurchases": ["total purchases", "item purchases"],
            "totalExpenses": ["total expenses", "expenses", "expense"],
            "totalTaxableAmount": ["taxable amount", "taxable"],
            "totalTax": ["total tax", "tax", "gst"],
            "totalOutstanding": ["outstanding", "unpaid", "pending"],
            "totalPaid": ["paid", "total paid"],
            "totalQuantity": ["total quantity", "quantity", "qty"],
            "billCount": ["bill count", "bills", "invoices"],
            "averageBillValue": ["average bill value", "avg bill", "average"],
        },
        "repair": {
            "overwrite": False,
            "date_keywords": ["purchase", "kharidi", "khareedari", "expense", "bill"],
            "default_fields": ["category", "totalAmount", "totalPurchases", "totalExpenses", "totalTaxableAmount", "totalTax", "totalOutstanding", "totalPaid", "billCount", "averageBillValue"],
        },
    },
    "get_search_ledgers": {
        "category": "ledger",
        "multi_call_ok": True,
        "description": "Search ledgers by name or filter by group type (expense, income, liability, asset).",
        "prompt_tips": "search_term=keyword, group_type=expense|income|liability|asset. Fields: id, name, openingBalance, openingType, glGroup (nested), email, phone, address, city, state, gstNumber.",
        "aliases": [
            "search ledger", "search ledgers", "find ledger", "find ledgers",
            "ledger search", "ledger group", "ledger groups",
        ],
        "keywords": [
            "search ledger", "find ledger", "ledger group",
            "expense ledger", "income ledger",
        ],
        "fields": [
            "id", "name", "email", "phoneNumber", "address",
            "city", "state", "country", "pincode",
            "openingBalance", "openingType", "gstNumber", "glGroup",
        ],
        "default_fields": ["id", "name", "openingBalance", "openingType", "glGroup"],
        "field_aliases": {
            "id": ["ledger id", "id"],
            "name": ["name", "ledger name"],
            "email": ["email"],
            "phoneNumber": ["phone", "phone number", "mobile"],
            "address": ["address"],
            "city": ["city"],
            "state": ["state"],
            "country": ["country"],
            "pincode": ["pincode", "pin code"],
            "openingBalance": ["opening balance", "opening"],
            "openingType": ["opening type"],
            "gstNumber": ["gst", "gst number", "gstin"],
            "glGroup": ["group", "gl group", "ledger group", "parent group"],
        },
        "repair": {
            "overwrite": False,
            "default_fields": ["id", "name", "openingBalance", "openingType", "glGroup"],
            "param_aliases": {"name": "search_term"},
        },
    },
    "get_search_vendors": {
        "category": "vendor",
        "multi_call_ok": True,
        "description": "Search vendors/suppliers by name.",
        "prompt_tips": "search=name (substring match), limit=N. Fields: id, name, email, phone, address, gstNumber, openingBalance, openingType.",
        "aliases": [
            "search vendor", "search vendors", "find vendor", "find vendors",
            "vendor search", "supplier search",
        ],
        "keywords": [
            "search vendor", "find vendor", "vendor list", "supplier list",
            "vendors", "suppliers",
        ],
        "fields": [
            "id", "name", "email", "phoneNumber", "address",
            "city", "state", "gstNumber", "openingBalance", "openingType",
        ],
        "default_fields": ["id", "name"],
        "field_aliases": {
            "id": ["vendor id", "supplier id", "id"],
            "name": ["name", "vendor name", "supplier name"],
            "email": ["email"],
            "phoneNumber": ["phone", "phone number", "mobile"],
            "address": ["address"],
            "city": ["city"],
            "state": ["state"],
            "gstNumber": ["gst", "gst number", "gstin"],
            "openingBalance": ["opening balance", "opening"],
            "openingType": ["opening type"],
        },
        "repair": {
            "overwrite": False,
            "default_fields": ["id", "name"],
            "param_aliases": {"name": "search"},
        },
    },
    "get_outstanding_sales_invoices": {
        "category": "sales",
        "multi_call_ok": True,
        "description": "Fetch outstanding/unpaid sales invoices with aging details, invoice amounts, due dates, and summary totals.",
        "prompt_tips": "from_date/to_date YYYY-MM-DD, sort_by=daysOverdue|invoiceDate|outstanding, sort_order=asc|desc. Does NOT support searching by invoice number — this lists invoices by date range/aging only. Summary row (recordType=outstandingSalesInvoices, name=Summary) has totalInvoices, totalOutstanding, totalOverdue, totalCurrent.",
        "aliases": [
            "outstanding sales", "outstanding invoices", "pending invoices",
            "unpaid invoices", "sales due", "invoice outstanding",
            "due invoices", "overdue invoices", "invoice aging",
            "pending payments", "outstanding amount",
        ],
        "keywords": [
            "outstanding", "pending", "unpaid", "due", "overdue",
            "aging", "invoice", "invoices", "receivable",
        ],
        "fields": [
            "recordType", "invoiceId", "invoiceNo", "invoiceDate", "dueDate",
            "ledgerId", "ledgerName", "country", "state",
            "txModeType", "invoiceType",
            "netAmount", "taxableAmount", "outstanding", "paidAmount",
            "daysOverdue", "isOverdue", "agingBucket",
            "totalInvoices", "totalOutstanding", "totalOverdue", "totalCurrent",
            "total_rows", "total_pages", "period",
        ],
        "default_fields": ["invoiceNo", "invoiceDate", "ledgerName", "netAmount", "outstanding", "daysOverdue", "agingBucket"],
        "always_include_fields": ["period"],
        "field_aliases": {
            "recordType": ["record type"],
            "invoiceId": ["invoice id"],
            "invoiceNo": ["invoice no", "invoice number", "invoice", "invoiceNumber"],
            "invoiceDate": ["invoice date", "date", "bill date"],
            "dueDate": ["due date", "due"],
            "ledgerId": ["customer id", "ledger id"],
            "ledgerName": ["customer", "customer name", "party", "ledger", "ledger name", "customerName"],
            "country": ["country"],
            "state": ["state"],
            "txModeType": ["transaction type", "tx type", "mode"],
            "invoiceType": ["invoice type", "type"],
            "netAmount": ["net amount", "amount", "invoice amount", "total"],
            "taxableAmount": ["taxable amount", "taxable"],
            "outstanding": ["outstanding", "pending", "due amount", "balance", "remaining", "outstandingAmount"],
            "paidAmount": ["paid amount", "paid", "payment"],
            "daysOverdue": ["days overdue", "overdue days", "days", "delay"],
            "isOverdue": ["is overdue", "overdue"],
            "agingBucket": ["aging bucket", "bucket", "aging", "current", "30-60", "60-90", "90+"],
            "totalInvoices": ["total invoices", "total invoice count"],
            "totalOutstanding": ["total outstanding", "grand total outstanding"],
            "totalOverdue": ["total overdue", "overdue total"],
            "totalCurrent": ["total current", "current total"],
        },
        "repair": {
            "overwrite": False,
            "date_keywords": ["outstanding", "invoice", "sales", "due", "pending", "unpaid", "receivable"],
            "default_fields": ["invoiceNo", "invoiceDate", "ledgerName", "netAmount", "outstanding", "daysOverdue", "agingBucket"],
            "param_aliases": {"invoiceNumber": "invoiceNo", "customerName": "ledgerName", "outstandingAmount": "outstanding"},
        },
    },
    "get_outstanding_purchase_invoices": {
        "category": "purchase",
        "multi_call_ok": True,
        "description": "Fetch outstanding/unpaid purchase invoices with aging details, bill amounts, due dates, and summary totals.",
        "prompt_tips": "from_date/to_date YYYY-MM-DD, sort_by=daysOverdue|invoiceDate|outstanding, sort_order=asc|desc. Does NOT support searching by invoice number — this lists invoices by date range/aging only. Summary row (recordType=outstandingPurchaseInvoices, name=Summary) has totalInvoices, totalOutstanding, totalOverdue, totalCurrent.",
        "aliases": [
            "outstanding purchases", "outstanding purchase invoices", "pending purchase invoices",
            "unpaid purchase invoices", "bills payable", "creditors",
            "payable invoices", "vendor outstanding", "pending bills",
        ],
        "keywords": [
            "outstanding", "pending", "unpaid", "due", "overdue",
            "payable", "creditor", "bill", "bills",
        ],
        "fields": [
            "recordType", "invoiceId", "invoiceNo", "invoiceDate", "dueDate",
            "ledgerId", "ledgerName", "country", "state",
            "txModeType", "invoiceType",
            "netAmount", "taxableAmount", "outstanding", "paidAmount",
            "daysOverdue", "isOverdue", "agingBucket",
            "totalInvoices", "totalOutstanding", "totalOverdue", "totalCurrent",
            "total_rows", "total_pages", "period",
        ],
        "default_fields": ["invoiceNo", "invoiceDate", "ledgerName", "netAmount", "outstanding", "daysOverdue", "agingBucket"],
        "always_include_fields": ["period"],
        "field_aliases": {
            "recordType": ["record type"],
            "invoiceId": ["invoice id"],
            "invoiceNo": ["invoice no", "invoice number", "invoice", "bill no", "bill number", "invoiceNumber"],
            "invoiceDate": ["invoice date", "date", "bill date"],
            "dueDate": ["due date", "due"],
            "ledgerId": ["vendor id", "supplier id", "ledger id"],
            "ledgerName": ["vendor", "vendor name", "supplier", "supplier name", "ledger", "ledger name", "customerName"],
            "country": ["country"],
            "state": ["state"],
            "txModeType": ["transaction type", "tx type", "mode"],
            "invoiceType": ["invoice type", "type"],
            "netAmount": ["net amount", "amount", "bill amount", "total"],
            "taxableAmount": ["taxable amount", "taxable"],
            "outstanding": ["outstanding", "pending", "due amount", "balance", "remaining", "payable", "outstandingAmount"],
            "paidAmount": ["paid amount", "paid", "payment"],
            "daysOverdue": ["days overdue", "overdue days", "days", "delay"],
            "isOverdue": ["is overdue", "overdue"],
            "agingBucket": ["aging bucket", "bucket", "aging", "current", "30-60", "60-90", "90+"],
            "totalInvoices": ["total invoices", "total invoice count", "total bills"],
            "totalOutstanding": ["total outstanding", "grand total outstanding", "total payable"],
            "totalOverdue": ["total overdue", "overdue total"],
            "totalCurrent": ["total current", "current total"],
        },
        "repair": {
            "overwrite": False,
            "date_keywords": ["outstanding", "purchase", "invoice", "payable", "creditor", "vendor", "bill"],
            "default_fields": ["invoiceNo", "invoiceDate", "ledgerName", "netAmount", "outstanding", "daysOverdue", "agingBucket"],
            "param_aliases": {"invoiceNumber": "invoiceNo", "customerName": "ledgerName", "outstandingAmount": "outstanding"},
        },
    },
    "get_overdue_invoices": {
        "category": "sales",
        "multi_call_ok": True,
        "description": "Fetch overdue invoices (both sales receivables and purchase payables) past their due date, with aging details and summary totals.",
        "prompt_tips": "invoice_type=SALES|PURCHASE|BOTH, as_of_date YYYY-MM-DD, sort_by=daysOverdue|invoiceDate|outstanding, sort_order=asc|desc. Default invoice_type=BOTH. Does NOT support searching by invoice number. Filter by invoiceCategory=RECEIVABLE|PAYABLE to separate sales vs purchase. Summary row (recordType=overdueInvoices, name=Summary) has totalInvoices, totalOverdue, totalReceivables, totalPayables, receivablesCount, payablesCount.",
        "aliases": [
            "overdue invoices", "overdue bills", "overdue payments",
            "past due", "past due invoices", "delayed payments",
            "overdue receivables", "overdue payables",
            "overdue sales invoices", "overdue purchase invoices",
        ],
        "keywords": [
            "overdue", "past due", "delayed", "late",
            "overdue invoice", "overdue bills", "overdue payment",
        ],
        "fields": [
            "recordType", "invoiceId", "invoiceNo", "invoiceDate", "dueDate",
            "ledgerId", "ledgerName", "country",
            "txModeType", "invoiceCategory", "invoiceType",
            "netAmount", "taxableAmount", "outstanding", "paidAmount",
            "daysOverdue", "agingBucket",
            "totalInvoices", "totalOverdue", "totalReceivables", "totalPayables",
            "receivablesCount", "payablesCount",
            "total_rows", "total_pages",
        ],
        "default_fields": ["invoiceNo", "invoiceDate", "dueDate", "ledgerName", "invoiceCategory", "netAmount", "outstanding", "daysOverdue", "agingBucket"],
        "field_aliases": {
            "recordType": ["record type"],
            "invoiceId": ["invoice id"],
            "invoiceNo": ["invoice no", "invoice number", "invoice", "bill no", "invoiceNumber"],
            "invoiceDate": ["invoice date", "date", "bill date"],
            "dueDate": ["due date", "due"],
            "ledgerId": ["customer id", "vendor id", "ledger id"],
            "ledgerName": ["customer", "customer name", "vendor", "vendor name", "party", "ledger", "ledger name", "customerName"],
            "country": ["country"],
            "txModeType": ["transaction type", "tx type", "mode"],
            "invoiceCategory": ["category", "invoice category", "receivable", "payable", "type"],
            "invoiceType": ["invoice type", "type"],
            "netAmount": ["net amount", "amount", "invoice amount", "total"],
            "taxableAmount": ["taxable amount", "taxable"],
            "outstanding": ["outstanding", "pending", "due amount", "balance", "remaining", "outstandingAmount", "overdueAmount"],
            "paidAmount": ["paid amount", "paid", "payment"],
            "daysOverdue": ["days overdue", "overdue days", "days", "delay", "late"],
            "agingBucket": ["aging bucket", "bucket", "aging", "90+ days", "60-90", "30-60", "1-30"],
            "totalInvoices": ["total invoices", "total invoice count"],
            "totalOverdue": ["total overdue", "grand total overdue", "total"],
            "totalReceivables": ["total receivables", "receivables total", "sales overdue"],
            "totalPayables": ["total payables", "payables total", "purchase overdue"],
            "receivablesCount": ["receivable invoices", "sales count", "receivable count"],
            "payablesCount": ["payable invoices", "purchase count", "payable count"],
        },
        "repair": {
            "overwrite": False,
            "default_fields": ["invoiceNo", "invoiceDate", "dueDate", "ledgerName", "invoiceCategory", "netAmount", "outstanding", "daysOverdue", "agingBucket"],
            "param_aliases": {"type": "invoice_type", "invoiceNumber": "invoiceNo", "customerName": "ledgerName", "outstandingAmount": "outstanding"},
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
