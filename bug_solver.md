# Bug Solver & Implementation Plan

## Bugs Found

### Bug 1: Cross-Query Data Leak via Session Memory

**Severity:** Critical
**Files:** `src/nodes.py:1725-1728`, `fast_main.py:266-269`

#### Root Cause

Two interacting issues:

**A. `deterministic_final_node` processes ALL historical ToolMessages**

```python
# src/nodes.py:1725-1728
tool_messages = [
    msg for msg in messages        # ← Every message ever in the conversation
    if isinstance(msg, ToolMessage)
]
```

On query 2+, the `messages` list contains ToolMessage objects from ALL previous turns. So customer data from query 1 leaks into the response for query 2.

**B. `fast_main.py` discards `RemoveMessage` deletions**

```python
# fast_main.py:267-269
for msg in state_update["messages"]:
    if msg not in messages_tracker and not isinstance(msg, RemoveMessage):
        messages_tracker.append(msg)   # ← RemoveMessage is IGNORED, not applied
```

When summarization fires and returns `RemoveMessage` to delete old messages, the tracker silently discards them. `SESSION_MEMORY` never shrinks — it grows unbounded and feeds the entire history into every subsequent request.

#### Fix

**Fix A — Scope ToolMessages to current turn (`src/nodes.py:1725-1728`):**

```python
# Find the current turn's tool call IDs from the most recent AIMessage
current_tool_call_ids = set()
for msg in reversed(messages):
    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
        current_tool_call_ids = {tc.get("id") for tc in msg.tool_calls if tc.get("id")}
        break

tool_messages = [
    msg for msg in messages
    if isinstance(msg, ToolMessage)
    and (not current_tool_call_ids or msg.tool_call_id in current_tool_call_ids)
]
```

**Fix B — Apply deletions to `messages_tracker` (`fast_main.py:266-269`):**

```python
if "messages" in state_update:
    for msg in state_update["messages"]:
        if isinstance(msg, RemoveMessage):
            messages_tracker = [m for m in messages_tracker if m.id != msg.id]
        elif msg not in messages_tracker:
            messages_tracker.append(msg)
```

---

### Bug 2: Unbounded Memory Growth in Session & Caches

**Severity:** High (production reliability)
**Files:** `fast_main.py:34-39`, `src/tools_api.py:17-18`

- `SESSION_MEMORY` — no eviction, grows with every unique session_id
- `FINAL_RESPONSE_CACHE` — no size limit, grows with every unique query
- `api_cache` — TTL-based but no size cap

**Fix (post-MVP):** Add `lru_cache`-style eviction or max-size caps. Use `collections.OrderedDict` or `cachetools.TTLCache`.

---

### Bug 3: `match_filter` Substring False Positives

**Severity:** Medium
**File:** `src/tools_api.py:160`

```python
return exp_str == act_str or (exp_str and exp_str in act_str)
```

A filter like `{"name": "am"}` matches "Rohan Sharma" because `"am" in "rohan sharma"` is True. The substring fallback was meant for partial-name matching but creates false positives for short/common substrings.

**Fix:** Remove the `in` fallback. `exp_str == act_str` is sufficient for exact matching. Use explicit `contains` operator for substring searches.

---

### Bug 4: Import-Time Crash on Model Failure

**Severity:** Medium
**File:** `src/nodes.py:43,46`

```python
_build_tool_embeddings()          # runs at import
_cross_encoder = CrossEncoder(...)  # runs at import
```

If Ollama is down or the CrossEncoder model is missing, the **entire server fails to start**.

**Fix:** Lazy-init with a simple `if not _tool_embeddings: _build_tool_embeddings()` guard inside `score_tools_via_reranker`.

---

### Bug 5: `project_fields` Silent Fallback on Mismatch

**Severity:** Medium
**File:** `src/tools_api.py:232-233`

```python
if not projected_records and records:
    return records   # ← silently returns ALL fields instead of empty
```

If user requests fields that don't exist in any record, the function returns all data unfiltered instead of empty.

**Fix:** Only return fallback if `fields` were explicitly API-level (not user-requested). Or better: just return `[]` when no fields match — the user asked for specific columns.

---

### Bug 6: `get_stock_levels` Local Sort Defeats Pagination

**Severity:** Low
**File:** `src/tools_api.py:613-615`

```python
"page": 1 if needs_local_sort else page,
"limit": 200 if needs_local_sort else limit,
```

Sorting by a non-default field forces page=1 and limit=200, fetches 200 records, sorts locally, then trims. If user asks "page 5 sorted by closingQty", they get wrong results (page 1 data sorted differently).

**Fix:** Accept the API's sort limitation and document it, or implement proper offset-based local pagination.

---

## Feature: Language-Mirroring LLM Response Generation

### Goal

Replace the deterministic `format_response_as_chat_text` formatting with an LLM-generated natural language response that mirrors the user's exact wording/language (Hinglish, English, Hindi, etc.).

### Changes Required

#### 1. New node: `response_generation_node` (`src/nodes.py`)

A new async node placed between `deterministic_final` and `summarization` in the graph.

**Input from state:**
- `final_response` — structured data dict from `deterministic_final_node`
- `original_query` — the raw user input (preserved language)
- `detected_language` — from translator node

**Logic:**
```python
async def response_generation_node(state: MainState) -> MainState:
    final_response = state.get("final_response", {})
    original_query = state.get("original_query", "")
    detected_language = state.get("detected_language", "")

    prompt = f"""You are an ERP assistant. Below are the tool results for the user's query.

Tool results: {json.dumps(final_response.get('data', {}), indent=2)}
Summary: {final_response.get('summary', '')}

Original user query: "{original_query}"
Language detected: {detected_language}

Task: Write a natural conversational response using the tool results above.
CRITICAL RULE: Mirror the user's exact language and wording style.
- If they used Hinglish (e.g., "muje customer details chaia"), respond in Hinglish with the same vocabulary ("batao", "chaia", "ka", "kaunsa").
- If they used English, respond in English.
- If they used Hindi, respond in Hindi.
- Match their level of formality and sentence structure.

Respond conversationally. Do NOT use bullet points or numbering unless the user asked for a list."""
    
    response = await summary_llm.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content=original_query),
    ])
    
    return {"response_text": response.content}
```

#### 2. Update graph (`src/graph.py`)

```python
from src.nodes import response_generation_node  # add import

builder.add_node("response_generation", timed_node("response_generation", response_generation_node))
builder.add_edge("deterministic_final", "response_generation")
# keep existing edge:
builder.add_edge("response_generation", "summarization")
```

#### 3. Update `fast_main.py`

**In `run_graph_query`** — capture `response_text` from the new node:
```python
if "response_text" in state_update:
    response_text = state_update["response_text"]
```

Return it in the result:
```python
result = {
    "response": response_text or final_response,  # LLM text preferred, fallback to structured
    "data": final_response.get("data", {}),
    "tools_utilized": final_response.get("tools_used", []),
    "timings": timings,
    "total_time_sec": total_time,
    "updated_messages": messages_tracker,
    "summary": summary_tracker,
}
```

**`/chat` endpoint** — return JSON with `response_text`:
```python
return {
    "response": result["response"],
    "tools_used": result.get("tools_utilized", []),
    "data": result.get("data", {}),
    "timings": result.get("timings", []),
    "total_time_sec": result.get("total_time_sec", 0.0),
}
```

**`/chat-text` endpoint** — return `result["response"]` as `PlainTextResponse`:
```python
text = result.get("response", "No response generated.")
return PlainTextResponse(text)
```

#### 4. Remove dead code

- `format_response_as_chat_text()` 
- `pretty_field_name()`
- `_TOOL_DISPLAY_NAMES`
- `get_tool_display_name()`
- `_pretty_field_names` config lookup

---

## Execution Order (Tomorrow)

| Step | File | Change | Est. Time |
|------|------|--------|-----------|
| 1 | `src/nodes.py:1725-1728` | Fix Bug 1A — scope ToolMessages to current turn | 5min |
| 2 | `fast_main.py:266-269` | Fix Bug 1B — apply RemoveMessage to tracker | 5min |
| 3 | `src/nodes.py` | Add `response_generation_node` | 20min |
| 4 | `src/graph.py` | Add node + edge to graph | 5min |
| 5 | `fast_main.py` | Update run_graph_query + both endpoints | 15min |
| 6 | `fast_main.py` | Remove dead formatting code | 5min |
| 7 | Test | Run server, test cross-query leak fix | 10min |
| 8 | Test | Test language mirroring (Hinglish, English, Hindi) | 10min |

**Total:** ~75 min
