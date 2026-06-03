# Chapter1-Assist

A FastAPI + LangGraph based ERP/accounting assistant that answers business-data queries by selecting the correct ERP tool, calling the backend API, and returning structured JSON or plain-text responses.

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

The system uses two local LLMs: a small translator (1024 ctx) for optional Hinglish→English normalization and a full-size worker (8192 ctx) for tool-call generation. No external API costs.

---

## Architecture

```
START
  -> translator            (Ollama qwen3:latest — optional Hinglish→English, language detection with fast-path shortcuts)
  -> semantic_search       (bge-m3 embedding + cross-encoder rerank + keyword fallback)
  -> chat_model            (Ollama qwen3:latest with bind_tools — native tool calling + retry loop + repair layer)
  -> routing_node          (conditional: tool_calls found? → tools, else → END)
  -> tools                 (executes backend APIs)
  -> deterministic_final   (Python dict builder — no LLM call, no hallucination)
  -> summarization         (optional conversation summarization after 3+ human messages)
  -> END
```

### Node Responsibilities

| Node | Responsibility |
|---|---|
| `translator` | Detects Hinglish/Hindi/Gujarati via script detection and keyword lists. Only invokes LLM when needed — has 3 fast-path shortcuts for plain English, routeable queries, and no-normalization-needed cases. Stores both `original_query` and `canonical_query`. |
| `semantic_search` | Selects relevant ERP tools via embedding recall (bge-m3) + cross-encoder reranking + keyword fallback with word-boundary regex. No LLM used — purely deterministic scoring. Splits multi-intent queries by connector words. |
| `chat_model` | Generates tool calls using Ollama native `bind_tools()` with internal retry loop (up to 3 rounds) when the LLM emits fewer calls than requested tools. Followed by `_apply_repair` — a registry-driven arg fixup layer that handles param aliases, date hallucination detection, category mapping, filter normalization, and generic cross-tool corrections. |
| `routing_node` | Routes to `tools` if the last message has tool_calls, otherwise ends the graph. Falls through after `loop_count > 5`. |
| `tools` | Executes the 6 ERP tool functions against the Chapter-1 backend API via `ToolNode`. |
| `deterministic_final` | Builds final JSON response from tool output using pure Python. No LLM involved — prevents hallucination of names, amounts, IDs, or quantities. Aggregates errors, deduplicates records, applies GST category filtering. |
| `summarization` | After 3+ human messages, summarizes old conversation context and deletes past messages using `RemoveMessage`. Keeps the most recent query intact. |

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

Worker and translator are both `qwen3:latest` (8B) running locally via Ollama with `reasoning=False`. No Groq/OpenAI keys, no rate limits, no per-query cost.

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

The `deterministic_final_node` builds responses using pure Python. Tool output is parsed, filtered, projected, and formatted without any LLM. This guarantees accurate amounts, names, and counts.

### Config-Driven Repair System

Tool argument repair uses `TOOL_INTENT_REGISTRY` in `src/tool_doc.py`:

- **`param_aliases`** — maps LLM param names to real API params
- **`category_map`** — maps category keywords with auto-generated no-space variants and prefix expansion
- **`date hallucination detection`** — discards dates the LLM invented when query has no date reference
- **`field_triggers`** — adds fields user asked for but LLM missed; `sirf`/`only`/`just`/`bas` disables them
- **`hsn_extract`** — extracts 8-digit HSN codes with optional product name
- **`category_to_filter`** — converts category keywords to API filter params
- **`strict_field_keywords`** — locks fields to a narrow set on exact keyword match

### Generic Cross-Tool Corrections

Applied after all tool-specific repairs:

- **Min/max → limit=1**: "sabse kam/jyada", "least/most" → forces `limit=1`
- **"Which entity?" → ensure name**: "kis product ka", "kaunsa customer" → prepends `name` to fields

### Positive/Negative Value Filters

Automatically injects value-comparison filters for queries mentioning negative/positive values like "negative closing qty", "0 se kam", "greater than 0".

### Multi-Identifier Queries

Queries like "49090090 aur id 349 dono ka stock status" are split into separate tool calls — one per identifier.

### GST Category Filtering

GST summary results are filtered deterministically based on the user's query keywords (B2B, B2C, exports, nil-rated, grand total, credit notes).

### Field Projection Fallback

When LLM-requested fields don't match actual API field names, returns original records instead of silently dropping data.

### Search Sanitization

`get_customer` auto-converts bad search terms ("all", "everyone", comma-separated lists) to empty string (returns all customers).

### Local Sort for Stock

Stock API ignores sort params. `get_stock_levels` fetches 200 records, sorts locally by requested field, truncates to limit.

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | FastAPI |
| Graph Engine | LangGraph |
| Worker LLM | `qwen3:latest` (8B) via Ollama (`reasoning=False`, `num_ctx=8192`) |
| Translator LLM | `qwen3:latest` (8B) via Ollama (`reasoning=False`, `num_ctx=1024`) |
| Summary LLM | `qwen3:latest` (8B) via Ollama (`reasoning=False`, `num_ctx=8192`) |
| Embeddings | `bge-m3` via Ollama |
| Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Backend | Chapter-1 ERP API |
| HTTP client | `requests` (sync) |

---

## Project Structure

```text
CHAPTER1-ASSIST/
├── fast_main.py              # FastAPI entry point, cache, session management, response formatting
├── config.yaml               # Pipeline config (cities, thresholds, keywords, field names)
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not committed)
├── bug_solver.md             # Known bugs and upcoming implementation plan
│
├── src/
│   ├── config.py             # LLM setup, API env vars, YAML config loader
│   ├── schema.py             # State/schema definitions (MainState, InputState, OutputState)
│   ├── api_client.py         # HTTP client for Chapter-1 ERP API
│   ├── tools_api.py          # ERP tool functions (API calls, caching, filtering, projection)
│   ├── tool_doc.py           # Tool registry, repair configs, field aliases, category maps
│   ├── nodes.py              # All LangGraph nodes (translator, semantic search, chat model, routing, deterministic final, summarization)
│   └── graph.py              # StateGraph builder with conditional routing
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
ollama pull bge-m3
```

The cross-encoder downloads on first startup.

---

## Running

```bash
python fast_main.py
```

Server: `http://127.0.0.1:8000`

First startup takes ~5-10s for cross-encoder model load.

---

## API Usage

### POST /chat

```json
{
  "query": "HSN 48211090 ka stock name and closing quantity dikhao"
}
```

Returns structured JSON:

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
    "errors": []
  }
}
```

### POST /chat-text

Same input, returns plain text response.

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
| **Total single-tool (cold)** | **~12s** |
| **Total single-tool (warm)** | **~6-7s** |

---

## Security

Before pushing to GitHub, run:

```bash
grep -R "Authorization\|API_TOKEN\|SECRET\|KEY" . --exclude-dir=.git
```

Do not commit `.env`, `venv/`, `__pycache__/`, or `chroma_db/`.

---

## Bug Tracker

See [bug_solver.md](./bug_solver.md) for known bugs and planned improvements.

---

**Author:** Yash Sheth
