# Chapter1-Assist

A FastAPI + LangGraph based ERP/accounting assistant that answers business-data queries by selecting the correct ERP tool, calling the backend API, and returning structured JSON or natural-language responses.

**Author:** Yash Sheth

---

## Overview

Chapter1-Assist connects to the Chapter-1 ERP backend and answers natural language queries about:

- Customer lookup by brand and city
- Customer ledger balance and transactions
- Stock levels by product, HSN, SKU, or low-stock status
- GST summary reports (B2B, B2C, exports, nil-rated, credit notes, grand total)
- TDS outstanding reports
- TCS outstanding reports

The system uses three local LLMs: a small translator (4B, 1024 ctx) for optional Hinglish→English normalization, a full-size worker (8B, 8192 ctx) for tool-call generation, and a summary LLM (8B, 8192 ctx) for natural-language responses and conversation summarization. No external API costs.

Sessions are stored in-memory (dict + threading.Lock) and do NOT survive a server restart. The in-memory store replaced an earlier SQLite-backed implementation for simplicity — no file I/O, no schema migrations, but sessions are lost on crash or restart. Switch to SQLite if you need persistent "time travel" across restarts.

---

## Architecture

```
START
  -> translator            (Ollama qwen3:4b — optional Hinglish→English, language detection with fast-path shortcuts)
  -> semantic_search       (bge-m3 embedding + cross-encoder rerank + keyword fallback)
  -> chat_model            (Ollama qwen3:latest with bind_tools — native tool calling + retry loop + repair layer)
  -> routing_node          (conditional: tool_calls found? → tools | memory_answer? → response_generation | else → END)
  -> tools                 (executes backend APIs)
  -> deterministic_final   (Python dict builder — scopes results to current turn, builds conversation_context)
  -> response_generation   (LLM natural-language response mirroring user's language)
  -> summarization         (optional after 6+ human messages — context trim via RemoveMessage)
  -> END
```

### Node Responsibilities

| Node | Responsibility |
|---|---|
| `translator` | Detects Hinglish/Hindi/Gujarati via Unicode script detection and keyword lists. Only invokes LLM when needed — has 3 fast-path shortcuts for plain English, routeable queries, and no-normalization-needed cases. Model: `qwen3:4b`. Stores both `original_query` and `canonical_query`. |
| `semantic_search` | Selects relevant ERP tools via embedding recall (bge-m3) + cross-encoder reranking + keyword fallback with word-boundary regex. No LLM used — purely deterministic scoring. Splits multi-intent queries by connector words. Detects meta-questions ("what did we ask?") and returns empty tool list. |
| `chat_model` | Generates tool calls using Ollama native `bind_tools()` with internal retry loop (up to 3 rounds) when the LLM emits fewer calls than requested tools. Followed by `_apply_repair` — a registry-driven arg fixup layer handling param aliases, date hallucination detection, category mapping, filter normalization, HSN extraction, cross-tool corrections, and multi-identifier expansion. Falls back to `memory_answer` via summary LLM when no tools are selected. |
| `routing_node` | Routes to `tools` if the last message has tool_calls; to `response_generation` if `memory_answer` is set; otherwise ends the graph. Falls through after `loop_count > 5`. |
| `tools` | Executes the 6 ERP tool functions against the Chapter-1 backend API via `ToolNode`. |
| `deterministic_final` | Builds final JSON response from tool output using pure Python. Scopes `ToolMessage` content to current turn's `tool_call_ids` (prevents cross-query data leak). Aggregates errors, deduplicates records, applies GST category filtering, builds `conversation_context` entities, persists `last_tool_call` across summarization. |
| `response_generation` | Generates natural-language response mirroring the user's language (Hinglish/Hindi/English). Uses summary LLM with language-aware system prompt. Falls back to inline text builder on error. Directly returns `memory_answer` when no tools were called. |
| `summarization` | After 6+ human messages (3+ exchanges), summarizes old conversation context using summary LLM and deletes past messages via `RemoveMessage`. Strips `raw_response` from ToolMessage content before summary input. Caps summary at 16000 characters. |

---

## Supported Tools

| Tool | Purpose |
|---|---|
| `get_customer` | Search customers by brand/city, return ID, name, opening balance |
| `get_customer_ledger` | Fetch ledger opening, current, closing balance, and transactions |
| `get_stock_levels` | Fetch stock levels by product, HSN, SKU, quantity, low/out-of-stock; local sort when API ignores sort params |
| `get_gst_summary` | Fetch GST summary (B2B, B2C, export, nil-rated, credit notes, grand total) |
| `get_tds_outstanding` | Fetch TDS outstanding summary and section-wise details |
| `get_tcs_outstanding` | Fetch TCS outstanding summary and section-wise details |

---

## Key Features

### Local LLMs, No External API

Worker and summary LLM use `qwen3:latest` (8B); translator uses `qwen3:4b` (4B) — all running locally via Ollama with `reasoning=False`. No Groq/OpenAI keys, no rate limits, no per-query cost.

### Fast-Path Translator

The translator node avoids an LLM call on most queries:
- **Plain English** → skip (script detection + non-English word check)
- **Routeable by keyword** → skip (ERP domain keywords like "stock", "gst", "customer" found in query)
- **No multilingual words** → skip (no Hinglish/Hindi/Gujarati tokens detected)
- Only queries with actual non-English words trigger the LLM normalization

### Embedding + Cross-Encoder Tool Routing

Tool selection uses a 3-stage pipeline:
1. **Embedding recall** (bge-m3) over pre-computed tool embeddings
2. **Cross-encoder reranking** (cross-encoder/ms-marco-MiniLM-L-6-v2)
3. **Keyword fallback** with word-boundary regex against the registry's keywords/aliases

### Native Tool Calling (No JSON Parsing)

Worker LLM uses `llm.bind_tools().ainvoke()`. Ollama's tool-calling API handles schema validation. Internal retry loop re-invokes when multiple tools are needed (qwen3 emits 1 call per response).

### Deterministic Final Response (No Hallucination)

The `deterministic_final_node` builds responses using pure Python. Tool output is parsed, filtered, projected, and formatted without any LLM. This guarantees accurate amounts, names, and counts. Results are scoped to the current turn's `tool_call_ids` to prevent cross-query data leak.

### Config-Driven Repair System

Tool argument repair uses `TOOL_INTENT_REGISTRY` in `src/tool_doc.py`:

- **`param_aliases`** — maps LLM param names to real API params
- **`category_map`** — maps category keywords with auto-generated no-space variants and prefix expansion
- **`date hallucination detection`** — discards dates the LLM invented when query has no date reference
- **`field_triggers`** — adds fields user asked for but LLM missed; `sirf`/`only`/`just`/`bas` disables them
- **`hsn_extract`** — extracts 8-digit HSN codes with optional product name from labeled format
- **`category_to_filter`** — converts category keywords to API filter params
- **`strict_field_keywords`** — locks fields to a narrow set on exact keyword match
- **`extract_customer_id`** — regex-based customer ID extraction
- **`city_filter`** — applies city-based filtering from query keywords
- **`low_stock_only_keywords`** — detects low-stock intent from query phrases

### Generic Cross-Tool Corrections

Applied after all tool-specific repairs:

- **Min/max → limit=1**: "sabse kam/jyada", "least/most" → forces `limit=1`
- **"Which entity?" → ensure name**: "kis product ka", "kaunsa customer" → prepends `name` to fields
- **Value-comparison filters**: "N se jyada/kam", "greater/less than N" → injects filter params
- **Malformed filter key normalization**: "closingQty gt: 2" → `{"closingQty": {"gt": 2}}`

### Conversation Memory & Context

- **`memory_answer`** — when no tools are relevant, the summary LLM answers from conversation history
- **`conversation_context`** — entity tracking (customer/product names, IDs) across turns
- **`last_tool_call`** — persists tool arguments across summarization for follow-up queries

### Natural-Language Response Generation

The `response_generation_node` uses the summary LLM to craft responses that mirror the user's language (Hinglish, Hindi, or English). Language detection with Hinglish word-list override ensures appropriate tone.

### Multi-Identifier Queries

Queries like "49090090 aur id 349 dono ka stock status" are split into separate tool calls — one per identifier. Controlled by `multi_call_ok` safety (only customer and stock tools allow duplicates).

### GST Category Filtering

GST summary results are filtered deterministically based on the user's query keywords (B2B, B2C, exports, nil-rated, grand total, credit notes).

### Field Projection Fallback

When LLM-requested fields don't match actual API field names, returns original records instead of silently dropping data.

### Search Sanitization

`get_customer` auto-converts bad search terms ("all", "everyone", comma-separated lists) to empty string (returns all customers).

### Local Sort for Stock

Stock API ignores sort params. `get_stock_levels` fetches 200 records, sorts locally by requested field, truncates to limit.

### API-Level Caching

`cached_api_post` uses `OrderedDict` with TTL (600s) and maxsize 100 — uses `time.monotonic()` for drift-free expiry.

---

## Tech Stack

| Component | Technology |
|---|---|
| Component | Technology |
|---|---|---|
| Framework | FastAPI |
| Graph Engine | LangGraph |
| Worker LLM | `qwen3:latest` (8B) via Ollama (`reasoning=False`, `num_ctx=8192`) |
| Translator LLM | `qwen3:4b` (4B) via Ollama (`reasoning=False`, `num_ctx=1024`) |
| Summary LLM | `qwen3:latest` (8B) via Ollama (`reasoning=False`, `num_ctx=8192`) |
| Embeddings | `bge-m3` via Ollama |
| Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Backend | Chapter-1 ERP API |
| HTTP client | `requests` (sync) |
| Session Store | In-memory dict + threading.Lock (lost on restart) |
| UI (optional) | Streamlit |

---

## Project Structure

```text
CHAPTER1-ASSIST/
├── fast_main.py              # FastAPI entry point, cache, session management, response formatting
├── streamlit_app.py          # Streamlit chat UI (optional client)
├── session_store.py          # In-memory session CRUD (dict + threading.Lock)
├── config.yaml               # Pipeline config (cities, thresholds, keywords, field names)
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not committed)
│
├── src/
│   ├── config.py             # LLM setup, API env vars, YAML config loader
│   ├── schema.py             # State/schema definitions (MainState, InputState, OutputState)
│   ├── api_client.py         # HTTP client for Chapter-1 ERP API
│   ├── tools_api.py          # ERP tool functions (API calls, caching, filtering, projection)
│   ├── tool_doc.py           # Tool registry (TOOL_INTENT_REGISTRY), repair configs, field aliases, category maps
│   ├── nodes.py              # All LangGraph nodes (translator, semantic search, chat model, routing, deterministic final, response generation, summarization)
│   └── graph.py              # StateGraph builder with conditional routing and timed_node wrapper
│
└── README.md
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
CHP1_API_BASE_URL=https://dev.chapter1.finance/aiAnalytics/
CHP1_API_TOKEN=your_api_token_here
COMPANY_ID=355
CHP1_API_TIMEOUT=10
```

---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Pull Ollama models:

```bash
ollama pull qwen3:latest
ollama pull qwen3:4b
ollama pull bge-m3
```

The cross-encoder downloads on first startup.

---

## Running

### FastAPI Server

```bash
python fast_main.py
```

Server: `http://127.0.0.1:8000`

First startup takes ~5-10s for cross-encoder model load.

### Streamlit UI (Optional)

```bash
streamlit run streamlit_app.py
```

UI: `http://127.0.0.1:8501`

The Streamlit client sends queries to the FastAPI `/chat` endpoint and displays natural-language responses.

---

## API Usage

### POST /chat

```json
{
  "query": "HSN 48211090 ka stock name and closing quantity dikhao"
}
```

Returns structured JSON with `response_text` (natural-language):

```json
{
  "response": {
    "success": true,
    "status": "success",
    "query": "HSN 48211090 ka stock name and closing quantity dikhao",
    "tools_used": ["get_stock_levels"],
    "data": {
      "get_stock_levels": [
        {
          "name": "Office Products 48211090 @ 18",
          "hsnCode": "48211090",
          "closingQty": -43
        }
      ]
    },
    "summary": "get_stock_levels: found 1 record",
    "response_text": "HSN 48211090 ke liye stock mil gaya: Office Products 48211090 @ 18 jiska closing quantity -43 hai.",
    "errors": []
  }
}
```

### POST /chat-text

Same input, returns plain text response (the `response_text` field).

---

## Test Queries

```json
{"query": "Nykaa Bangalore customer id, name and opening balance batao"}
{"query": "jo sabse kam closing quantity wala product hai uska value aur name chaia"}
{"query": "Customer id 814 ka opening, current and closing balance from 2024-04-01 to 2024-12-31"}
{"query": "Show stock levels for HSN 48211090"}
{"query": "49090090 aur id 349 dono ka stock status kya hai?"}
{"query": "Show B2B GST from 2024-04-01 to 2024-04-30"}
{"query": "Show TDS outstanding and TCS outstanding from 2024-04-01 to 2024-12-31"}
{"query": "negative closing quantity wale products dikhao"}
{"query": "sirf closing quantity batao"}
{"query": "humne abhi tak kya poocha hai?"}
```

---

## Performance

Typical local timings (qwen3:8b on RTX 3070):

| Stage | Approx Time |
|---|---|
| translator_node (cold) | ~7s |
| translator_node (warm) | ~3s |
| semantic_search | ~2.5s |
| chat_model_node | ~3.5s |
| tools_node (1 call) | ~0.5s |
| deterministic_final | <5ms |
| response_generation | ~1.5s |
| **Total single-tool (cold)** | **~14s** |
| **Total single-tool (warm)** | **~8-9s** |

---

## Security

Before pushing to GitHub, run:

```bash
grep -R "Authorization\|API_TOKEN\|SECRET\|KEY" . --exclude-dir=.git
```

Do not commit `.env`, `venv/`, `__pycache__/`, or `chroma_db/`.

---

## Known Bugs & Limitations

- **API pagination** — `get_stock_levels` defaults to `limit=10`, so count queries may see only 10 records despite `total_rows` indicating hundreds or thousands. Fix: increase limit or paginate in the tool.
- **In-memory sessions** — sessions are lost on server restart. Switch to SQLite (`pip install aiosqlite`) and re-implement `session_store.py` with a persistent backend for "time travel" across restarts.
- **Summarization trim** — after 6+ human messages, old context is summarized via LLM; summary quality depends on the 8B model.

---

## Recent Updates (2026-06-05)

**Session Persistence:**
- **Replaced SQLite with in-memory store** (`session_store.py`) — sessions stored in `dict` + `threading.Lock`. No file I/O, no schema migrations, simplifies deployment. Sessions are lost on restart — switch back to SQLite if persistence is needed.
- **History display fix** (`fast_main.py`) — appends `AIMessage(content=response_text)` before `save_session` so the conversation response appears correctly in history. History endpoint filters out tool/empty-content messages.
- **In-memory session CRUD** — `init_db` is a no-op; `create_session`, `get_session`, `save_session`, `delete_session`, `rename_session`, `load_messages`, `list_sessions` all operate on the in-memory dict.

**Tool & Response Fixes:**
- **Client-side filter bug** (`tools_api.py`) — all 6 tools previously re-applied filters on project_result output after the API already filtered. Removed re-application; API response is authoritative.
- **Response truncation fix** (`nodes.py`) — `MAX_SAMPLE_RECORDS` raised from 10 → 50, so the LLM sees more records before generating a response. Added anti-hallucination rule 7: *"If truncated data is shown with a `__note`, do not claim you have shown all records."*
- **API pagination awareness** — `get_stock_levels` defaults to `limit=10`, meaning count queries may see only a fraction of total records despite `total_rows` indicating the true count.

**UI Improvements:**
- **Scrollable tool records** (`streamlit_app.py`) — new `render_records()` helper wraps tool dataframes inside collapsed `st.expander` panels (📊 tool_name: N records). Each expander contains a scrollable `st.dataframe` (height 300px if >10 rows, auto-height otherwise).
- **Triple return from send_query** — now returns `(text, data_dict, session_id)`. `data_dict` is passed to `render_records` and stored in `st.session_state` for display across rerenders.

## Previous Updates (2026-06-04)

**Critical (P0):**
- **Summarization 40s blowup** — stripped `raw_response` from ToolMessage content before summary LLM input (dropped latency to ~1.3s)
- **Meta-question hallucination** — `semantic_search` detects conversational queries ("what did we ask?") and returns empty `selected_tools` instead of random tool picks
- **Tool call ID collision** — added `uuid.uuid4().hex[:12]` suffix to `call_{name}` IDs, preventing LangGraph silent dedup across turns
- **Missing import crash** — replaced `await format_response_as_chat_text` (never imported) with inline text fallback in `response_generation_node`
- **Cross-query data leak** — scoped ToolMessage content to current turn's `tool_call_ids` in `deterministic_final_node`

**High (P1):**
- Summary changed from APPEND to regenerate-from-scratch, eliminating accumulated hallucinations
- Sort key `TypeError` risk fixed with `_sort_key` helper (`float('-inf')` fallback)
- `fields` dict mutation fixed (`{**fields, "isLowStock": True}` instead of in-place assignment)
- Silent filter drop fixed — `apply_filters` validates field names upfront, returns empty with `[WARN]` log
- Node crash no longer corrupts state — exception handler merges into `dict(state)` instead of replacing all state
- `match_filter` substring false positives removed — substring fallback deleted

**Medium (P2):**
- Auth token reverted to hardcoded `"ROHANVAJA007"` for testing (was broken by env var import)
- Cache clock fixed — `time.time()` → `time.monotonic()` in `cached_api_post`
- Tool timing lost fixed — `timed_node` handles `list` return from `ToolNode`
- Comma-search crash fixed — blanket `"," in search` replaced with `re.search(r'\w+,\s*\w+', search)`
- Hinglish detection gap closed — fallback checks query words when translator says English
- Tool name alias spaces fixed — `normalize_tool_name` converts spaces to underscores before alias lookup
- Missing `import re` added to `tools_api.py`
- Conversation memory routing — `memory_answer` path routes directly to `response_generation` node, skipping tools

**Files affected:** `session_store.py`, `fast_main.py`, `streamlit_app.py`, `src/nodes.py`, `src/tools_api.py`, `src/graph.py`, `src/api_client.py`

---

**Author:** Yash Sheth
