# Chapter1-Assist

A FastAPI + LangGraph based ERP/accounting assistant that answers business-data queries by selecting the correct ERP tool, calling the backend API, and returning clean conversational responses.

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

The system uses an LLM for tool-call generation while the final response is built deterministically from real API output, reducing hallucination and keeping responses predictable.

---

## Architecture

```text
START
  -> translator_node       (Ollama granite4.1:8b — Hinglish→English)
  -> semantic_search       (keyword + metadata tool routing)
  -> chat_model_node       (Groq llama-3.3-70b — generates tool calls as JSON text)
  -> routing_node          (conditional: tool_calls? → tools : → end)
  -> ToolNode              (executes backend APIs)
  -> deterministic_final   (builds JSON from API data)
  -> format_response       (conversational text output)
  -> END
```

### Node Responsibilities

| Node | Responsibility |
|---|---|
| `translator_node` | Translates Hinglish/Hindi queries to English (granite4.1:8b) |
| `semantic_search` | Selects relevant ERP tools using keyword + metadata rules |
| `chat_model_node` | Generates tool calls from query + tools (Groq llama-3.3-70b) |
| `routing_node` | Routes to tools if tool calls exist, else ends |
| `ToolNode` | Executes backend API tools |
| `deterministic_final_node` | Builds final JSON from tool output |
| `format_response` | Converts structured JSON into conversational text |

---

## Supported Tools

| Tool | Purpose |
|---|---|
| `get_customer` | Search customers by brand/city, return ID, name, opening balance |
| `get_customer_ledger` | Fetch ledger opening, current, closing balance, and transactions |
| `get_stock_levels` | Fetch stock levels by product, HSN, SKU, quantity, low/out-of-stock |
| `get_gst_summary` | Fetch GST summary (B2B, B2C, export, nil-rated, credit notes, grand total) |
| `get_tds_outstanding` | Fetch TDS outstanding summary and section-wise details |
| `get_tcs_outstanding` | Fetch TCS outstanding summary and section-wise details |

---

## Key Features

### Deterministic Final Response

The LLM does not write the final business answer directly. Instead:

```text
LLM chooses tool calls → Tools fetch real ERP data → Python builds final response
```

This prevents hallucinated records, amounts, customer IDs, GST values, stock quantities, or ledger balances.

### Conversational Output

Responses are formatted as natural language (like ChatGPT/Gemini) rather than raw JSON. Tool names and field labels are stripped from the output.

### Config-Driven Repair System

Tool argument repair uses config from `TOOL_INTENT_REGISTRY` rather than hardcoded if/elif chains:

- **`param_aliases`** — maps LLM param names to real params (e.g., `"name"` → `"term"`)
- **`category_to_filter`** — converts GST category keywords to API filters
- **`category_map`** — maps category aliases (e.g., `"b2csmall"` → `"b2cSmall"`)
- Auto-generated no-space variants for category matching

### Multi-Identifier Queries

Queries with multiple identifiers (e.g., "49090090 aur id 349") are split into separate tool calls:

```text
"49090090 aur id 349 dono ka stock status kya hai?"
  -> Call 1: get_stock_levels(filters={"hsnCode": "49090090"})
  -> Call 2: get_stock_levels(filters={"id": 349})
```

### GST Category Filtering

GST summary API returns all categories, but the system filters based on the user query:

| User asks | Returned categories |
|---|---|
| B2B GST | `b2b` only |
| Grand total GST | `grandTotal` only |
| B2B + grand total | `b2b`, `grandTotal` |
| Full GST summary | All categories |

### Deterministic Repair Layer

Groq generates tool calls as JSON text. A repair layer normalizes and corrects arguments:

- **Tool name aliases**: `tds_report`→`get_tds_outstanding`, `stock_report`→`get_stock_levels`
- **Date alias normalization**: `startDate`/`endDate` → `from_date`/`to_date`
- **Worker date preservation**: Captures correct dates from worker output before overwrite
- **HSN strict override**: Discards bad worker filters, uses `{term: HSN, filters: {hsnCode: HSN}}`
- **Customer multi-city expansion**: Creates one call per city via `expand_customer_city_calls()`
- **Multi-query splitting**: Detects multiple identifiers and creates separate tool calls

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | FastAPI |
| Graph Engine | LangGraph |
| Worker LLM | `llama-3.3-70b-versatile` (Groq API) |
| Translator LLM | `granite4.1:8b` (Ollama) |
| Embeddings | `bge-m3` (Ollama) |
| Backend | Chapter-1 ERP API |

---

## Project Structure

```text
CHAPTER1-ASSIST/
├── fast_main.py              # FastAPI entry point, cache, session management
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not committed)
│
├── src/
│   ├── config.py             # LLM config, API credentials, model initialization
│   ├── schema.py             # State/schema definitions (MainState, InputState)
│   ├── api_client.py         # HTTP client with timeout/error handling
│   ├── tools_api.py          # API-backed ERP tool functions
│   ├── tool_doc.py           # Tool descriptions, intent registry, repair configs
│   ├── nodes.py              # LangGraph nodes, repair logic, query processing
│   ├── graph.py              # LangGraph graph builder
│   ├── retriever.py          # Tool retriever logic
│   └── vector_store.py       # Vector store handling
│
└── README.md
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
CHP1_API_BASE_URL=https://dev.chapter1.finance/aiAnalytics/
COMPANY_ID=355
CHP1_API_TOKEN=your_api_token_here
GROQ_API_KEY=your_groq_api_key_here
```

Do not hardcode private API tokens before pushing to GitHub.

---

## Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd CHAPTER1-ASSIST
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure Ollama is running and required models are available:

```bash
ollama list
```

Pull models if needed:

```bash
ollama pull granite4.1:8b
ollama pull bge-m3
```

You also need a [Groq API key](https://console.groq.com) set as `GROQ_API_KEY` in `.env`.

---

## Running

Start the FastAPI server:

```bash
python fast_main.py
```

Server starts at:

```text
http://127.0.0.1:8000
```

---

## API Usage

**Endpoint:** `POST /chat`

**Example Request:**

```json
{
  "query": "HSN 48211090 ka stock name and closing quantity dikhao"
}
```

**Example Response:**

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

---

## Test Queries

### Customer Lookup

```json
{
  "query": "Nykaa Bangalore customer id, name and opening balance batao"
}
```

### Customer Ledger

```json
{
  "query": "Customer id 814 ka opening, current and closing balance bata from 2024-04-01 to 2024-12-31"
}
```

### Stock by HSN

```json
{
  "query": "Show stock levels for HSN 48211090"
}
```

### Multi-Product Stock Query

```json
{
  "query": "49090090 aur id 349 dono ka stock status kya hai?"
}
```

### GST B2B + Grand Total

```json
{
  "query": "Show B2B GST taxable amount, IGST, CGST, SGST and invoice amount, also show grand total GST from 2024-04-01 to 2024-04-30"
}
```

### TDS + TCS Outstanding

```json
{
  "query": "Show TDS outstanding and TCS outstanding from 2024-04-01 to 2024-12-31"
}
```

### Multi-tool Query

```json
{
  "query": "Nykaa Bangalore customer id batao, HSN 48211090 ka stock name and closing quantity dikhao, aur B2B GST taxable amount and invoice amount dikhao from 2024-04-01 to 2024-04-30"
}
```

---

## Performance

Typical local timings:

| Query Type | Approx Time |
|---|---|
| Customer lookup | 1.2s - 1.5s |
| Ledger balance | 1.3s - 1.5s |
| Stock HSN lookup | 1.2s - 1.4s |
| GST summary | 2.0s - 2.3s |
| TDS + TCS | 1.7s - 2.0s |
| Multi-identifier stock | 1.5s - 2.0s |

Backend API latency may vary. Cached API responses are faster.

---

## Security

Before pushing to GitHub, ensure secrets are not committed:

```bash
grep -R "Authorization\|API_TOKEN\|SECRET\|KEY" .
```

Do not commit `.env`, `venv/`, `__pycache__/`, or `chroma_db/`.

---

## License

This project is currently a prototype/portfolio ERP assistant. Add a license file before public distribution if required.

---

**Author:** Yash Sheth