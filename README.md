# ERP Assistant

A multilingual AI-powered ERP assistant built using LangGraph, FastAPI, vector search, and Large Language Models (LLMs). The system enables users to interact with ERP data using natural language queries while supporting multiple Indian languages, intelligent tool selection, contextual understanding, and structured responses.

## Features

* 🌐 Multilingual query support

  * English
  * Hindi
  * Gujarati
  * Hinglish
  * Mixed-language queries

* 🤖 AI-driven query understanding and routing

* 🔍 Semantic search-based capability discovery

* 🧠 Context-aware conversation handling

* ⚡ FastAPI-powered backend

* 📊 Structured JSON responses

* 🔗 Dynamic tool orchestration

* 📈 Built-in performance and timing tracking

* 🛡️ Hallucination reduction through deterministic response construction

* 💾 Session-based conversation management

---

## High-Level Architecture

```text
User Query
    │
    ▼
Language Normalization
    │
    ▼
Semantic Search
    │
    ▼
Query Routing
    │
    ▼
Tool Selection
    │
    ▼
Data Retrieval
    │
    ▼
Response Generation
    │
    ▼
Structured JSON Response
```

The assistant follows a graph-based workflow that dynamically determines how a query should be processed, which capabilities should be used, and how the final response should be generated.

---

## Tech Stack

### Backend

* Python
* FastAPI
* Uvicorn

### AI & Orchestration

* LangGraph
* LangChain
* Large Language Models (LLMs)

### Vector Search

* ChromaDB
* Embedding Models

### Data Handling

* Pydantic
* Typed State Management

---

## Core Capabilities

* Natural language ERP interactions
* Multilingual query processing
* Intelligent query routing
* Semantic capability discovery
* Context-aware conversations
* Structured response generation
* Session memory management
* Performance monitoring and timing analysis
* Extensible tool-based architecture

---

## Project Structure

```text
ERP-ASSISTANT/
│
├── api/
├── graph/
├── tools/
├── prompts/
├── embeddings/
├── sessions/
├── configs/
├── logs/
├── data/
│
├── main.py
├── requirements.txt
└── README.md
```

> Actual structure may vary depending on the active branch and deployment configuration.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/Hybrid-Git/ERP-ASSISTANT.git
cd ERP-ASSISTANT
```

### Create a Virtual Environment

Linux/macOS

```bash
python -m venv venv
source venv/bin/activate
```

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file and provide the required configuration values.

```env
# LLM Configuration
MODEL_NAME=

# Embedding Configuration
EMBEDDING_MODEL=

# Application Configuration
APP_CONFIG=

# Additional Settings
...
```

---

## Running the Application

```bash
python main.py
```

or

```bash
uvicorn main:app --reload
```

---

## Example Workflow

1. User submits a natural language query.
2. Query is analyzed and normalized.
3. Semantic search identifies relevant capabilities.
4. The router determines the optimal execution path.
5. Required tools are executed.
6. Results are aggregated.
7. A structured response is generated and returned.

---

## Response Format

Responses are returned in a structured JSON format suitable for frontend integration and downstream processing.

Example:

```json
{
  "success": true,
  "status": "success",
  "query": "user query",
  "summary": "response summary",
  "data": {},
  "errors": [],
  "timings": {}
}
```

---

## Security & Privacy

This repository intentionally excludes:

* Internal API specifications
* Private endpoints and URLs
* Production credentials
* Business-specific logic
* Deployment secrets
* Infrastructure configuration details

Sensitive configuration should be managed through environment variables or a secure secret management solution.

---

## Development Goals

* Improve multilingual understanding
* Reduce response latency
* Enhance routing accuracy
* Expand ERP capability coverage
* Improve response reliability
* Optimize token consumption
* Improve scalability and maintainability

---

## Author

### Yash Sheth

AI Engineer focused on building multilingual AI assistants, agentic workflows, LLM-powered applications, semantic search systems, and enterprise automation solutions.

**Skills & Areas of Interest**

* AI Agents & Agentic Workflows
* LangGraph & LangChain
* FastAPI & Python
* Retrieval-Augmented Generation (RAG)
* Multilingual LLM Applications
* Enterprise Automation Systems

GitHub: https://github.com/Hybrid-Git

---

## License

This project is intended for educational, research, and development purposes unless otherwise specified.
