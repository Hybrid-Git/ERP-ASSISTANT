# Chapter1-Assist

A FastAPI + LangGraph based ERP/accounting assistant that answers business-data queries by selecting the correct ERP tool, calling the Chapter-1 backend API, and returning structured JSON or natural-language responses. Detects greetings and responds appropriately; silently rejects out-of-context queries.

**Author:** Yash Sheth

---

## Overview

Chapter1-Assist connects to the Chapter-1 ERP backend and answers natural language queries about:

- Customer lookup by brand and city
- Customer ledger balance and transactions
- Stock levels by product, HSN, SKU, or low-stock status
- GST summary reports (B2B, B2C, exports, nil-rated, credit notes, grand total)
- TDS / TCS outstanding reports
- Sales analytics (top products, popular products, slow-moving products, sales summary, sales trends)
- Top customers and top vendors
- Purchase summary
- Ledger and vendor search
- Outstanding sales/purchase invoices with aging
- Overdue invoices (receivables & payables)

All LLMs run locally via an OpenAI-compatible endpoint (Ollama, vLLM, etc.). Model selection, temperature, and context length are configured via environment variables — no hardcoded model names.

---

## Architecture

```
START
  -> translator            (optional Hinglish→English normalization — LLM; fast-path shortcuts for plain English)
  -> semantic_search       (embedding recall + cross-encoder rerank + keyword fallback; detects greetings & meta-questions)
  -> chat_model            (LLM with bind_tools — native tool calling + retry loop + registry-driven repair layer;
                            falls back to memory_answer when no tools selected)
  -> routing_node          (conditional: tool_calls? → tools | memory_answer? → response_generation | else → END)
  -> tools                 (executes backend APIs via ToolNode)
  -> deterministic_final   (Python dict builder — scopes results to current turn, builds conversation_context)
  -> response_generation   (LLM natural-language response mirroring user's language)
  -> summarization         (optional after 6+ human messages — context trim via RemoveMessage)
  -> END
```

### Node Responsibilities

| Node | Responsibility |
|---|---|
| `translator` | Detects Hinglish/Hindi/Gujarati via Unicode script detection and keyword lists. Only invokes LLM when needed — 3 fast-path shortcuts. Stores both `original_query` and `canonical_query`. |
| `semantic_search` | Selects relevant ERP tools via embedding recall (bge-m3) + cross-encoder reranking + keyword fallback with word-boundary regex. Splits multi-intent queries. Detects meta-questions, greetings (returns welcome message), and unsupported queries (returns refusal). |
| `chat_model` | Generates tool calls using `llm.bind_tools().ainvoke()` with internal retry loop (up to 3 rounds). Followed by `_apply_repair` — registry-driven arg fixup handling param aliases, date hallucination detection, category mapping, filter normalization, HSN extraction, and multi-identifier expansion. Falls back to `memory_answer` via summary LLM when no tools are selected. |
| `routing_node` | Routes to `tools` if last message has tool_calls; to `response_generation` if `memory_answer` is set; otherwise ends. Falls through after `loop_count > 5`. |
| `tools` | Executes tool functions against the Chapter-1 backend API via `ToolNode`. |
| `deterministic_final` | Builds final JSON response from tool output using pure Python. Scopes `ToolMessage` content to current turn's `tool_call_ids`. Aggregates errors, deduplicates records, applies GST category filtering, builds `conversation_context` entities. |
| `response_generation` | Generates natural-language response mirroring the user's language (Hinglish/Hindi/English). Uses summary LLM with language-aware system prompt. Falls back to inline text builder on error. |
| `summarization` | After 6+ human messages, summarizes old context using summary LLM and deletes past messages via `RemoveMessage`. Strips `raw_response` from ToolMessage content before summary input. Caps summary at 16,000 characters. |

---

## Supported Tools (19)

| Tool | Purpose |
|---|---|
| `get_customer` | Search customers by brand/city, return ID, name, opening balance |
| `get_customer_ledger` | Fetch ledger opening, current, closing balance, and transactions |
| `get_stock_levels` | Fetch stock levels by product, HSN, SKU, low/out-of-stock; local sort when API ignores sort params |
| `get_gst_summary` | Fetch GST summary (B2B, B2C, export, nil-rated, credit notes, grand total) |
| `get_tds_outstanding` | Fetch TDS outstanding summary and section-wise details |
| `get_tcs_outstanding` | Fetch TCS outstanding summary and section-wise details |
| `get_top_products` | Top/best-selling products by revenue, quantity, or profit |
| `get_popular_products` | Popular/trending products for a given period |
| `get_slow_moving_products` | Slow-moving products with low turnover |
| `get_sales_summary` | Sales summary report grouped by day/week/month |
| `get_sales_trend` | Sales trend comparison (current vs previous period) |
| `get_top_customer` | Top customers by revenue or order count |
| `get_top_vendor` | Top vendors by purchase amount or bill count |
| `get_purchase_summary` | Purchase summary report with expense breakdown |
| `get_search_ledgers` | Search ledgers by name or group type |
| `get_search_vendors` | Search vendors/suppliers by name |
| `get_outstanding_sales_invoices` | Outstanding/unpaid sales invoices with aging |
| `get_outstanding_purchase_invoices` | Outstanding/unpaid purchase invoices with aging |
| `get_overdue_invoices` | Overdue invoices (receivables & payables) past due date |

---

## Key Features

### Local LLMs, Configurable via Env Vars

Three LLM instances (`normalizer_llm`, `llm`, `summary_llm`) are created from environment variables (`TRANS_LLM_MODEL`, `LLM_MODEL`, `SUMMARY_LLM_MODEL`). Works with any OpenAI-compatible endpoint — Ollama, vLLM, Groq, or OpenAI itself. `/no_think` directive is appended to all system prompts to suppress chain-of-thought reasoning.

### Greeting Detection & Out-of-Context Refusal

The `semantic_search_node` recognizes common greetings via regex patterns. Pure greetings return a welcome message immediately. Non-ERP queries receive a clear refusal message.

### Fast-Path Translator

The translator node avoids an LLM call on most queries:
- **Plain English** → skip (script detection + non-English word check)
- **Routeable by keyword** → skip (ERP domain keywords found in query)
- **No multilingual words** → skip (no Hinglish/Hindi/Gujarati tokens)
- Only queries with actual non-English words trigger the LLM normalization

### Embedding + Cross-Encoder Tool Routing

3-stage pipeline: embedding recall (bge-m3) → cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2) → keyword fallback with word-boundary regex.

### Native Tool Calling (No JSON Parsing)

Worker LLM uses `llm.bind_tools().ainvoke()`. Ollama's tool-calling API handles schema validation. Internal retry loop re-invokes when multiple tools are needed.

### Deterministic Final Response (No Hallucination)

The `deterministic_final_node` builds responses using pure Python. Tool output is parsed, filtered, projected, and formatted without any LLM.

### Config-Driven Repair System

Tool argument repair uses `TOOL_INTENT_REGISTRY` in `src/tool_doc.py`:

- **`param_aliases`** — maps LLM param names to real API params
- **`category_map`** — maps category keywords with auto-generated no-space variants
- **`date hallucination detection`** — discards dates the LLM invented when query has no date reference
- **`field_triggers`** — adds fields user asked for but LLM missed; `sirf`/`only` disables them
- **`hsn_extract`** — extracts 8-digit HSN codes with optional product name
- **`category_to_filter`** — converts category keywords to API filter params
- **`strict_field_keywords`** — locks fields to a narrow set on exact keyword match
- **`extract_customer_id`** — regex-based customer ID extraction
- **`city_filter`** — applies city-based filtering from query keywords
- **`low_stock_only_keywords`** — detects low-stock intent from query phrases

### Generic Cross-Tool Corrections

- **Min/max → limit=1**: "sabse kam/jyada", "least/most" → forces `limit=1`
- **"Which entity?" → ensure name**: prepends `name` to fields
- **Value-comparison filters**: "N se jyada/kam" → injects filter params
- **Malformed filter key normalization**: "closingQty gt: 2" → `{"closingQty": {"gt": 2}}`

### Additional Features

- **Conversation memory** with entity tracking and `memory_answer` fallback
- **Multi-identifier queries** — split into separate tool calls
- **GST category filtering** — deterministic filtering based on query keywords
- **Field projection fallback** — returns original records when LLM-invented field names don't match
- **Search sanitization** — auto-converts bad search terms to empty string
- **Local sort for stock** — fetches 200 records, sorts locally, truncates to limit
- **API-level caching** — `OrderedDict` with TTL (600s) and maxsize 100

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | FastAPI |
| Graph Engine | LangGraph |
| LLMs | OpenAI-compatible (Ollama/vLLM/Groq) — model per env var |
| Embeddings | `bge-m3` via Ollama |
| Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Backend | Chapter-1 ERP API |
| HTTP client | `httpx` (async) |
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
├── tools.md                  # New tool addition guide
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not committed)
│
├── src/
│   ├── config.py             # LLM setup, API env vars, YAML config loader
│   ├── schema.py             # State/schema definitions (MainState, InputState, OutputState)
│   ├── api.py                # HTTP client for Chapter-1 ERP API
│   ├── tools_api.py          # 19 ERP tool functions (API calls, caching, filtering, projection)
│   ├── tool_doc.py           # Tool registry (TOOL_INTENT_REGISTRY), repair configs, field aliases
│   ├── nodes.py              # All LangGraph nodes (translator, semantic search, chat model, routing, deterministic final, response generation, summarization)
│   └── graph.py              # StateGraph builder with conditional routing and timed_node wrapper
│
└── README.md
```

---

## Prerequisites

- **Python 3.11+**
- **OpenAI-compatible endpoint** (Ollama recommended — download from [ollama.com](https://ollama.com))
- **Chapter-1 ERP API credentials**

### Verify Ollama

```bash
ollama list          # should show installed models
ollama serve         # ensure the server is running (default: http://localhost:11434)
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
CHP1_API_BASE_URL=https://dev.chapter1.finance/aiAnalytics/
CHP1_API_TOKEN=your_api_token_here
COMPANY_ID=355
CHP1_API_TIMEOUT=10

# LLM Configuration (OpenAI-compatible)
OPENAI_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3:latest
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=2048
LLM_KEEP_ALIVE=30m
LLM_THINK=false

# Optional: LangSmith tracing
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

| Variable | Required | Description |
|---|---|---|
| `CHP1_API_BASE_URL` | Yes | Chapter-1 ERP API base URL |
| `CHP1_API_TOKEN` | Yes | API authentication token |
| `COMPANY_ID` | Yes | Company identifier for API requests |
| `CHP1_API_TIMEOUT` | No | API request timeout in seconds (default: 10) |
| `OPENAI_BASE_URL` | Yes | LLM endpoint URL |
| `LLM_MODEL` | Yes | Worker + summary LLM model name |
| `LLM_TEMPERATURE` | No | LLM temperature (default: 0.0) |
| `LLM_MAX_TOKENS` | No | Max tokens (default: 2048) |
| `LLM_KEEP_ALIVE` | No | Ollama keep-alive duration |
| `LLM_THINK` | No | Enable/disable reasoning |
| `LANGSMITH_*` | No | LangSmith tracing (leave empty to disable) |

---

## Installation

### 1. Clone & Setup Virtual Environment

```bash
git clone <repo-url> chapter1-assist
cd chapter1-assist
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull Ollama Models (if using Ollama)

```bash
ollama pull qwen3:latest    # worker + summary LLM
ollama pull bge-m3          # embedding model for tool routing
```

The cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) downloads automatically on first startup (~500 MB).

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials and model choice
```

---

## Running

### Start the FastAPI Server

```bash
python fast_main.py
```

Expected output:
```
LLM and embedding model initialised!
Building graph...
FastAPI started. Graph already built. Warming up worker LLM...
Worker LLM warmup completed in 1.3s
Cross-encoder model loaded successfully during startup.
Application startup complete.
```

Server: `http://127.0.0.1:8000`

### Health Check

```bash
curl http://127.0.0.1:8000/
# {"message":"ERP Assistant API is running"}
```

> **Order matters:** Start the FastAPI server **first**, then launch the Streamlit UI.

### Streamlit UI (Optional, run in a second terminal)

```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

UI: `http://127.0.0.1:8501`

### Quick Test

```bash
# Greeting
curl -X POST http://127.0.0.1:8000/chat?format=text \
  -H "Content-Type: application/json" \
  -d '{"query": "hello"}'

# ERP query
curl -X POST http://127.0.0.1:8000/chat?format=text \
  -H "Content-Type: application/json" \
  -d '{"query": "Show stock for HSN 48211090"}'
```

---

## API Usage

### POST /chat

```json
{"query": "HSN 48211090 ka stock name and closing quantity dikhao"}
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
        {"name": "Office Products 48211090 @ 18", "hsnCode": "48211090", "closingQty": -43}
      ]
    },
    "summary": "get_stock_levels: found 1 record",
    "response_text": "HSN 48211090 ke liye stock mil gaya: Office Products 48211090 @ 18 jiska closing quantity -43 hai.",
    "errors": []
  }
}
```

### POST /chat/stream

Server-Sent Events (SSE) stream with token, data, and completion events:

```
data: {"token": "Here"}
data: {"token": " are"}
data: {"data": {"get_stock_levels": [...]}}
data: {"session_id": "abc123", "done": true}
```

### POST /chat-text

Same input, returns plain text response only.

---

## Test Queries

### ERP Queries
```json
{"query": "Nykaa Bangalore customer id, name and opening balance batao"}
{"query": "jo sabse kam closing quantity wala product hai uska value aur name chaia"}
{"query": "Customer id 814 ka opening, current and closing balance from 2024-04-01 to 2024-12-31"}
{"query": "Show stock levels for HSN 48211090"}
{"query": "Show B2B GST from 2024-04-01 to 2024-04-30"}
{"query": "Show TDS outstanding and TCS outstanding from 2024-04-01 to 2024-12-31"}
{"query": "Top 5 best selling products this month"}
{"query": "Sales summary for last month"}
{"query": "Outstanding sales invoices dikhao"}
{"query": "Overdue invoices for purchase"}
{"query": "pending bills dikhao"}
```

### Greetings
```json
{"query": "hello"}, {"query": "hi"}, {"query": "good morning"}, {"query": "namaste"}
```

### Out-of-Context Queries (responds with refusal)
```json
{"query": "what is the meaning of life?"}, {"query": "tell me a joke"}
```

### Meta Questions (from conversation memory)
```json
{"query": "humne abhi tak kya poocha hai?"}
```

---

## Performance

Typical local timings (8B model on RTX 3070):

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

- **API pagination** — Some tools default to `limit=10`; count queries may see only partial records despite `total_rows` indicating hundreds.
- **In-memory sessions** — sessions are lost on server restart. Switch to SQLite for persistence.
- **Summarization trim** — after 6+ human messages, old context is summarized via LLM; summary quality depends on the model.
- **Multi-intent splitting** — queries with multiple tools may produce sequential (not parallel) tool calls due to single-call-per-response LLM behavior.

---

**Author:** Yash Sheth
