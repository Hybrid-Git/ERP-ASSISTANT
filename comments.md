# Session Changes Log — June 2026

## Changes Made

### 1. Semantic search re-enabled (embedding recall only)
- **File:** `src/nodes.py:427-590`
- **What:** Activated the preserved `semantic_search` body. Switched model from `qwen3:8b` → `qwen2.5:7b-instruct` and embedding model from `qwen3-embedding:0.6b` → `bge-m3:latest`.
- **Why:** All-19-tools approach caused unnecessary API calls and noise. Embedding recall at ~50ms is negligible vs 28s LLM time. bge-m3 handles multilingual terms better.
- **Cleanup:** Removed dead cross-encoder code (`CrossEncoder` import, `get_cross_encoder()`, `TH_RERANKER_MIN`, `CROSS_ENCODER_MODEL` config).

### 2. Cross-encoder code fully removed
- **File:** `src/nodes.py`, `fast_main.py`, `config.yaml`
- **What:** Deleted `CrossEncoder` import, `_cross_encoder` variable, `get_cross_encoder()` function, `CROSS_ENCODER_MODEL` config, and `TH_RERANKER_MIN`. Removed commented-out warmup call in `fast_main.py`.
- **Why:** Dead code since embedding recall replaced cross-encoder for tool selection.

### 3. `sanitize_tool_filters` — filter key remap via `field_aliases`
- **File:** `src/nodes.py:1016-1026`
- **What:** Added reverse alias lookup before stripping invalid filter keys.
- **Why:** LLM sends `filters: {"invoiceNumber": "AI/22-23/018"}` but API fields use `invoiceNo`. The remap converts `invoiceNumber` → `invoiceNo` so the filter survives sanitization and works with client-side `apply_filters`.
- **Affects:** `invoiceNumber`→`invoiceNo`, `customerName`→`ledgerName`, `outstandingAmount`→`outstanding`, and all other `field_aliases` entries.

### 4. `_apply_repair` — filter key remap safety net
- **File:** `src/nodes.py:1691-1696`
- **What:** Added the same reverse alias remap for `filters` keys inside `_apply_repair`.
- **Why:** Safety net if `sanitize_tool_filters` misses a key. Catches LLM-invented filter names during repair phase.

### 5. `overdueAmount` alias added
- **File:** `src/tool_doc.py:813`
- **What:** Added `"overdueAmount"` to the `outstanding` field_aliases list.
- **Why:** LLM frequently requests `overdueAmount` as a field name but the API field is `outstanding`. Without this alias, the field is stripped and the amount doesn't appear in output.

### 6. Model migration: qwen3:8b → qwen2.5:7b-instruct
- **File:** `.env`
- **What:** Changed `LLM_MODEL`, `TRANS_LLM_MODEL`, `SUMMARY_LLM_MODEL` from `qwen3:8b` to `qwen2.5:7b-instruct`.
- **Why:** Faster inference (~2x), same tool-calling support, already available in Ollama.

### 7. Embedding model: qwen3-embedding → bge-m3
- **File:** `.env`
- **What:** Changed `EMB_MODEL` from `qwen3-embedding:0.6b` to `bge-m3:latest`.
- **Why:** Better multilingual support for Hinglish/Hindi/Gujarati ERP queries.

## Known Remaining Issues

### A. No invoice-number lookup endpoint
The API has no endpoint that accepts a specific invoice number (`/invoices/{no}`). Current workaround: `get_overdue_invoices` returns all overdue invoices, and client-side `apply_filters` post-filters by `invoiceNo`. This works but fetches unnecessary data.

### B. LLM still invents field names
qwen2.5 occasionally requests `overdueAmount`, `customerName`, etc. instead of the real API field names. The `field_aliases` system handles most of these, but any new invented name needs a manual entry.

### C. Latency still dominated by LLM
qwen2.5:7b-instruct faster than qwen3:8b but still ~10-15s on CPU/Ollama. A smaller quantized model or GPU inference would further reduce this.

## Future Considerations

- If embedding recall consistently picks wrong tools, increase `reranker_top_k` or adjust `embedding_recall_min` threshold.
- If the API adds an invoice-number lookup endpoint, create a dedicated tool.
- `tools.md` has the full tool reference and architecture documentation.
