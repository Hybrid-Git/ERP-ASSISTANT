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

---

## Known Bugs & Issues (Session Log - June 2026)

### 1. Invoice-number filter silently dropped

**Symptom:** Query `"AI/22-23/018 iska kitna overdue hai"` returns ALL overdue invoices instead of just the matching one.

**Root cause chain:**
1. LLM emits `filters: {"invoiceNumber": "AI/22-23/018"}` for `get_overdue_invoices`
2. `sanitize_tool_filters` (`nodes.py:1012`) checks `invoiceNumber` against `meta.fields` — valid field is `invoiceNo`, not `invoiceNumber` → flagged invalid
3. Tries to move value to `search`/`term` param — neither exists on invoice tools → **value silently dropped** (`nodes.py:1031-1037`)
4. Tool returns unfiltered API response (all overdue invoices)
5. Client-side `apply_filters` (`tools_api.py:296`) never receives the filter → no post-filtering either

**Fix locations:**
- `src/nodes.py` ~line 1012 in `sanitize_tool_filters`: build reverse `field_aliases` lookup (same pattern as `_apply_repair` line 1676-1683) and remap filter keys before validity check
- `src/tools_api.py` ~line 193 in `apply_filters`: add same reverse alias remap before checking `record_fields`
- Currently unaffected: `param_aliases` in `_apply_repair` (line 1691-1694) only remaps top-level args, not `filters` sub-dict — and runs *after* `sanitize_tool_filters` has already stripped the data

**Related:** The same bug affects `filters: {"customerName": "..."}` — `customerName` → `ledgerName`, `outstandingAmount` → `outstanding`. Any LLM-invented filter key gets silently dropped for tools without `search`/`term` params.

---

### 2. Semantic search / cross-encoder too slow

**File:** `config.yaml:299` | `nodes.py:26`

**Measured time:** 10.99s for one query

**Calculation:** 19 tools × 4 query parts = **76 pairwise cross-encoder scores** using `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Fix:** Reduce `reranker_top_k: 5` → `reranker_top_k: 3`. Cuts ~40% of cross-encoder work. Or switch to `MiniLM-L-4-v2` for ~2x speedup at minor accuracy cost.

**Alternative:** The first query (1 part × 19 tools = 19 scores) took only 2.7s. The 10s queries have 3-4 parts. Can we limit parts before cross-encoder, or deduplicate overlapping parts?

---

### 3. Unnecessary tool selection via keyword fallback

**Symptom:** A query like `"AI/22-23/018 overdue amount and name"` selects 4 tools when only `get_overdue_invoices` is sufficient.

**Selected:** `get_outstanding_sales_invoices`, `get_outstanding_purchase_invoices`, `get_overdue_invoices`, `get_customer`

**Why:** Query part "overdue" matches keyword lists for all 3 invoice tools. "Customer name" hits `get_customer`.

**Impact:** qwen3:8b emits only 1-2 tool calls per LLM round → 2 rounds (30s + 8.5s) instead of 1. Two of those tool calls (`get_outstanding_*`) then timeout (10s wasted).

**Fix:** Add invoice-number regex check (`\w{2}/\d{2}-\d{2}/\d{3}`) in semantic search or keyword fallback — when detected, skip `get_customer` and opposite-category invoice tools.

---

### 4. Outstanding invoice API timeout

**Symptom:** Both `get_outstanding_sales_invoices` and `get_outstanding_purchase_invoices` timed out (API 504 / no response in 30s).

**Observation:** These endpoints are slower than `get_overdue_invoices` (which returned in 0.7s). The outstanding endpoints might compute totals on-the-fly or fetch larger datasets.

**Status:** Not reproducible on demand — likely load-dependent. Increase `cached_api_post` timeout or implement retry with backoff as a mitigation.

---

### 5. `_apply_repair` field remap covers `fields` but not `filters`

**File:** `src/nodes.py:1676-1683` (field remap) vs `src/nodes.py:1691-1694` (param_aliases)

The field alias reverse remap (`nodes.py:1676-1683`) only applies to the `fields` list in args, not to `filters` keys. So:
- `fields: ["customerName"]` → corrected to `fields: ["ledgerName"]` ✅
- `filters: {"customerName": "..."}` → NOT corrected ❌

**Fix:** Either extend the field remap to also mutate `filters` keys, or add reverse alias lookup in `sanitize_tool_filters` (see Bug #1).

---

### 6. Latency breakdown (reference)

| Stage | Time | % of total |
|-------|------|-----------|
| translator | 4.7s | 6% |
| semantic_search | 11.0s | 14% |
| chat_model (round 1) | 29.9s | 39% |
| chat_model (round 2) | 8.6s | 11% |
| tools | 10.1s | 13% |
| response_generation | 7.0s | 9% |
| **Total** | **~76s** | |

**Key insight:** Multi-round LLM calls (38.5s total) dominate. Reducing unnecessary tools (Bug #3) eliminates round 2. Faster cross-encoder (Bug #2) saves ~4-5s.
