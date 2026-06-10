# Changes Log

All fixes made in a single commit on `staging` branch. Files changed: `src/nodes.py`, `fast_main.py`, `session_store.py`, `src/config.py`.

---

## Bug Fixes

### Bug #1a — Tool count cap per intent type
**File:** `src/nodes.py:876-882`

**Problem:** Single-intent queries got the same 5-tool budget as multi-intent queries,
causing tool-calling noise for targeted lookups (e.g. "konsa customer..." → 5 tools
instead of the 1-2 that make sense).

**Fix:** Dynamic cap based on intent count and query phrasing:
- Single-intent + targeted keywords (`konsa`, `which`, `find`, etc.) → **2 tools max**
- Single-intent non-targeted → **4 tools max**
- Multi-intent (2+ query parts) → **5 tools max**

### Bug #1b — Force-inject per query part (not global domain pass)
**File:** `src/nodes.py:1783-1812`

**Problem:** The force-inject loop checked tool-domain overlap against a union of all
query-part domains. This meant a tool matching part A's domain could be force-injected
for part B, or tools for part B were skipped if part A's domains already "covered" them.

**Fix:** Rewrote the loop to iterate `query_parts` individually. For each part, domains
are classified from that part alone, and only tools matching that part's domain are
injected. Each force-inject log now says `"for query part '{qp}'"`.

### Bug #2 — GST/TDS/TCS date refusal
**File:** `src/nodes.py` (in `_apply_repair` at line ~1885 and `chat_model_node` return at line ~2074)

**Problem:** When the user asked for GST/TDS/TCS data without specifying a date range,
the tool was called anyway with empty `from_date`/`to_date`, returning all data or
garbage.

**Fix:** In `_apply_repair`, `DATE_REQUIRED_TOOLS` check detects missing dates and
returns `{"_needs_date_range": True}`. The repair loop collects these and injects a
`memory_answer` in Hinglish asking for the date range (e.g. "Kya aap kripya karke
starting aur ending date bata sakte hain?"). The routing node sees `memory_answer` and
routes to `response_generation` to present the question to the user.

The `overwrite: True` repair mode now merges worker-provided non-empty args under the
tool's `base_args` (from `tool_doc.py`), so worker-filled values like `category` are
preserved while dates remain blank.

**Config change:** `src/tool_doc.py` — `get_gst_summary.repair` has `overwrite: True,
base_args: {from_date:"", to_date:""}`.

### Bug #3a — List-intent filter
**File:** `src/nodes.py:686-698`

**Problem:** Broad list queries like `"sab customer batao"` were triggering
`get_customer_ledger` (a single-record lookup that expects a specific customer name),
resulting in no data or wrong results.

**Fix:** New `_filter_for_list_intent()` function detects list-all keywords (`sab`,
`sare`, `all`, `sabhi`, etc.) and drops `get_customer_ledger` from the selected tool
list for that query part.

### Bug #4 — Entity-aware domain classification
**File:** `src/nodes.py:76-99`

**Problem:** When a resolved entity name contained a tax/GST keyword (e.g. company
named "GST Suppliers"), `classify_domains` would incorrectly flag the query as
tax/GST domain, selecting wrong tools.

**Fix:** `classify_domains()` now accepts an optional `resolved_entities` list. It
builds a token set from entity names and skips `gst`/`tax` domain matches when the
matching keyword appears inside a resolved entity name. All call sites pass
`state.get("resolved_entities")`.

### Bug #5 — Ollama metadata not logged (None keys crash)
**File:** `src/nodes.py:1259-1284`

**Problem:** `print_ollama_metadata()` and `log_token_usage()` crashed on None values
or missing keys from Ollama's response metadata.

**Fix:** All field accesses guarded with `.get()` and `or "N/A"` / `or 0` fallbacks.
Duration fields check for None before formatting.

### Bug #6 — Token/usage_metadata access in log_token_usage
**File:** `src/nodes.py:1263-1274`

**Problem:** `log_token_usage()` tried to read `response_metadata.token_usage` which
could be absent depending on the LLM provider.

**Fix:** Reads `usage_metadata` (LangChain's standard field) first, falls back to
`response_metadata` for token counts and `model_name` for the model identifier.

### Bug #7 — dead `tools_node` (investigated, not a bug)
**File:** `src/graph.py:77`

`tools_node` is imported from `src.nodes` and added to the graph as `"tools"`.
It is actively used by the conditional edge from `chat_model` → `"tools"`. This is
**not dead code**.

### Bug #8 — FastAPI lifespan deprecation
**File:** `fast_main.py:39-48`

**Problem:** `@app.on_event("startup")` is deprecated in FastAPI ≥ 0.93.

**Fix:** Replaced with modern `lifespan` async context manager (`@asynccontextmanager`).
The warmup logic (LLM ping) runs inside `lifespan()` and the old `startup_event`
function was removed.

### Config env var mismatch
**File:** `src/config.py:30-31`

**Problem:** `llm` was configured with `BASE_URL` and `MODEL_API_KEY` env vars, but
`.env` and `.env.example` defined `LLM_BASE_URL` and `LLM_MODEL_API_KEY`. The LLM
was silently using default values (OpenAI), making it point at the wrong endpoint.

**Fix:** Changed to `LLM_BASE_URL` and `LLM_MODEL_API_KEY` to match `.env`.

---

## Discovered & Fixed During Testing

### Context persistence across turns
**Files:** `fast_main.py`, `session_store.py`

**Problem:** `conversation_context` (entity references) and `last_tool_call` (last
tool args per name) were computed inside `deterministic_final_node` on every turn
but *not saved* to the session store. On the next turn, `initial_state` got empty
defaults, so pronoun resolution (`_resolve_pronouns`) had no entity history to work
with — follow-up queries like "iski ledger" failed.

**Fix:**
- `session_store.save_session()` now accepts optional `conversation_context` and
  `last_tool_call` dicts and persists them alongside messages/summary.
- New `session_store.load_session_context()` returns `(summary, context, last_tool)`
  for a session.
- All three endpoint paths (`/chat`, `/chat/stream`, `/chat-text`) call
  `load_session_context()` and pass the values through `run_graph_query`'s new params
  (`past_conversation_context`, `past_last_tool_call`).
- The `run_graph_query` function tracks these as `context_tracker` / `last_tool_tracker`
  through the `astream` loop, updates them from state updates, and returns them in the
  result dict.
- On cache hits, the persisted context is preserved (not overwritten with empty data).

### Entity memory pollution from aggregation tools
**File:** `src/nodes.py:2639-2640`

**Problem:** `get_top_customer`, `get_sales_trend`, and `get_stock_levels` returned
aggregated data whose entity names were added to `conversation_context.entities`.
These polluted pronoun resolution — a follow-up like "baki dono" might resolve to
aggregation entities instead of the customers/products the user actually asked about.

**Fix:** `ENTITY_SKIP_TOOLS` set defined; entity extraction skips these tools in
`deterministic_final_node`.

### Dedup collapsing different-arg calls to same tool
**File:** `src/nodes.py:2001-2010`

**Problem:** The name-based dedup (`seen_names: dict[str, int]`) collapsed
`get_customer(search=Bangalore)` and `get_customer(search=Hyderabad)` into a single
call, losing the multi-city resolution.

**Fix:** Dedup key changed from tool `name` to `(name, args_json)`. The second dedup
pass (`multi_call_ok` gate) also uses `(name, args_key)` instead of `name` alone.

### Plural pronoun resolution bypassing focus_entity
**File:** `src/nodes.py:135-142`

**Problem:** `_resolve_pronouns` always applied `focus_entity` first. For plural
pronouns like `dono` / `in dono`, this replaced the pronoun with a single entity name
when the user clearly wanted two entities ("baki dono" → "baki [single entity]").

**Fix:** `DONO_PRONOUNS` set detected at the top of `_resolve_pronouns`. When a
plural pronoun is found, the `focus_entity` shortcut is skipped entirely, falling
through to the full `entities` list for resolution.

---

## Response Generation Improvements (Latest Fix)
**File:** `src/nodes.py:2943-3179`

**Problem:** The `summary_llm` (qwen2.5:7b) sometimes returned raw JSON dumps or
empty responses instead of natural-language summaries.

**Fixes:**
1. **Rule 0** added to system prompt: "CRITICAL — NEVER output raw JSON, JSON
   blocks, or any data dump. Your ENTIRE reply MUST be a plain conversational
   paragraph."
2. **`_clean_llm_response()`** post-processor strips:
   - Lines starting with meta-framing phrases (`based on`, `here are`, `from the
     data`, `according to`, etc.)
   - JSON fence markers (` ```json `, ` ``` `)
   - Trailing code fences
3. **Improved fallback**: on LLM failure, uses the pre-built `summary` text (from
   `make_summary()`) instead of raw `key=value` dumps. If summary is unavailable,
   outputs a minimal `"tool: N record(s)"` message.
4. **Summary separation**: `summary` field extracted from the JSON blob sent to LLM
   and passed as a plain-text note instead, reducing prompt size and preventing the
   LLM from echoing it back as raw data.

---

## Summary of Files Changed

| File | Changes |
|---|---|
| `src/nodes.py` | Bug #1a, #1b, #2, #3a, #4, #5, #6; entity memory pollution; dedup collapse; plural pronoun fix; response generation improvements |
| `fast_main.py` | Bug #8 (lifespan); context persistence across all 3 endpoints |
| `session_store.py` | `save_session` accepts context/last_tool; new `load_session_context()` |
| `src/config.py` | env var name fix (`BASE_URL` → `LLM_BASE_URL`, `MODEL_API_KEY` → `LLM_MODEL_API_KEY`) |
