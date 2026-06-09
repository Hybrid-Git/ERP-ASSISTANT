# Adding a New Tool

## Files to modify

| # | File | What to add |
|---|------|-------------|
| 1 | `src/tools_api.py` | Endpoint constant + `@tool` function + register in `tools` list |
| 2 | `src/tool_doc.py` | Entry in `TOOL_INTENT_REGISTRY` |
| 3 | `config.yaml` | `route_keywords` + `segment_next_keywords` + `pretty_field_names` |

---

## 1. `src/tools_api.py`

### a) Add endpoint constant (near top of file)

```python
MY_ENDPOINT = "/some/path"
```

### b) Add `@tool` async function (before the `tools` list)

```python
@tool
async def get_my_new_tool(
    from_date: str = "",
    to_date: str = "",
    page: int = 1,
    limit: int = 10,
    fields: Optional[Any] = None,
    filters: Optional[dict[str, Any]] = None,
):
    """
    Description of what this tool does — used by the LLM to understand when to call it.

    Args:
        from_date: Start date in YYYY-MM-DD. Empty string if not provided.
        to_date: End date in YYYY-MM-DD. Empty string if not provided.
        fields: Optional output columns.
        filters: Optional exact filters.
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

    result = await cached_api_post(MY_ENDPOINT, body=body)

    # If response is nested object → flatten first (see flatten helpers)
    # If response has a "summary" object → append summary row:

    result = append_report_summary_row(result, "myRecordType")

    result = project_result(result, fields=fields, filters=filters)

    print("[TOOL OUTPUT]", result)
    return json.dumps(result, ensure_ascii=False)
```

### c) Register in the `tools` list

```python
tools = [
    ...
    get_my_new_tool,
]
```

`tools_dict` on line 1224 auto-includes it via `tool.name`.

---

## 2. `src/tool_doc.py`

Add entry to `TOOL_INTENT_REGISTRY` (before the closing `}`):

```python
"get_my_new_tool": {
    "category": "my_category",
    "multi_call_ok": True,
    "description": "Human-readable description of the tool.",
    "prompt_tips": "Usage hints injected into system prompt for the LLM.",
    "aliases": [
        "alias1", "alias2",  # User-facing names for route matching
    ],
    "keywords": [
        "keyword1", "keyword2",  # Search keywords for embedding retrieval
    ],
    "fields": [
        "field1", "field2",  # All possible fields the API can return
    ],
    "default_fields": ["field1"],  # Fields shown by default
    "always_include_fields": ["period"],  # Always included (optional)
    "field_aliases": {
        "apiFieldName": ["alias", "alias2"],  # User terms → API field mapping
    },
    "repair": {
        "overwrite": False,  # True = replace args, False = merge
        "date_keywords": ["keyword1"],  # Words that indicate date context
        "default_fields": ["field1"],  # Fallback fields
        # Optional:
        "param_aliases": {"userParam": "apiParam"},
        "field_triggers": {"keyword": "fieldName"},
        "category_map": {"userCat": "apiCat"},
    },
},
```

---

## 3. `config.yaml`

### a) `route_keywords`

```yaml
  - keyword1
  - keyword2
```

### b) `segment_next_keywords`

```yaml
  - keyword1
```

### c) `pretty_field_names`

```yaml
  apiFieldName: Display Name
```

---

## Existing Tool Patterns

### Flat data (no transform needed)
`get_top_products`, `get_popular_products`, `get_slow_moving_products`, `get_top_customer`, `get_top_vendor`, `get_search_ledgers`, `get_search_vendors`, `get_customer`

→ Just call `cached_api_post` + `project_result`.

### Nested object → flatten first
`get_gst_summary`, `get_sales_summary`, `get_purchase_summary`, `get_sales_trend`

→ Call flatten helper (e.g. `flatten_gst_summary_result`) before `project_result`.

### Summary row from API response
`get_tds_outstanding`, `get_tcs_outstanding`, `get_outstanding_sales_invoices`, `get_outstanding_purchase_invoices`

→ Call `append_report_summary_row(result, "recordType")` before `project_result`.

---

## Tools currently registered (as of June 2026)

| Tool | Endpoint | Category |
|------|----------|----------|
| `get_customer` | `/customers` | customer |
| `get_customer_ledger` | `/customers/ledger` | customer_ledger |
| `get_stock_levels` | `/inventory/stock` | stock |
| `get_gst_summary` | `/reports/gst-summary` | gst_report |
| `get_tds_outstanding` | `/reports/tds-outstanding` | tds_report |
| `get_tcs_outstanding` | `/reports/tcs-outstanding` | tcs_report |
| `get_top_products` | `/aiAnalytics/top-products` | analytics |
| `get_popular_products` | `/aiAnalytics/popular-products` | analytics |
| `get_slow_moving_products` | `/slow-moving-products` | analytics |
| `get_sales_summary` | `/sales-summary` | analytics |
| `get_sales_trend` | `/sales-trends` | analytics |
| `get_top_customer` | `/top-customers` | customer |
| `get_top_vendor` | `/top-vendors` | vendor |
| `get_purchase_summary` | `/purchase-summary` | purchase |
| `get_search_ledgers` | `/aiAnalytics/ledgers/search` | ledger |
| `get_search_vendors` | `/aiAnalytics/vendors` | vendor |
| `get_outstanding_sales_invoices` | `/aiAnalytics/reports/outstanding-sales-invoices` | sales |
| `get_outstanding_purchase_invoices` | `/aiAnalytics/reports/outstanding-purchase-invoices` | purchase |
