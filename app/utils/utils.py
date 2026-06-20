import json
import re
import time
from types import MappingProxyType
import numpy as np
import tiktoken
from app.core.config import embedding_model, get_cfg
from app.prompts.tool_doc import TOOL_INTENT_REGISTRY, TOOL_NAME_ALIASES
import re
import logging
import os
model = os.getenv("LLM_MODEL")
provider = "sarvamai"
logger = logging.getLogger("erp_assistant.tokens")


NON_ENGLISH_HINTS = []
MULTILINGUAL_WORDS = []
ROUTE_KEYWORDS = []
CONNECTORS = []
STOP_TOKENS = set()
SEGMENT_NEXT_KEYWORDS = []
TH_EMBEDDING_RECALL_MIN = get_cfg("thresholds", "embedding_recall_min", default=0.3)
TH_RERANKER_TOP_K = get_cfg("thresholds", "reranker_top_k", default=5)

PARTY_WORDS = []
NAME_WORDS = []
LIST_WORDS = []

DOMAIN_KEYWORDS = MappingProxyType({})

INVOICE_PATTERNS = {}

INVOICE_NO_PATTERNS = []

TOOL_DOMAINS = {}

INVOICE_TOOLS = set()

_tool_embeddings: dict[str, list[float]] = {}

def now():
    return time.perf_counter()

def sec(start):
    return round(time.perf_counter() - start, 3)

def ns_to_sec(value):
    if value is None:
        return None
    try:
        return round(value / 1_000_000_000, 3)
    except Exception:
        return value

def extract_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()

def add_unique(items: list[str], value: str):
    if value and value not in items:
        items.append(value)

def _cosine_sim(a: list[float], b: list[float]) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _build_tool_embeddings():
    if _tool_embeddings:
        return
    try:
        for tool_name, meta in TOOL_INTENT_REGISTRY.items():
            text_parts = [meta.get("description", "")]
            text_parts.extend(meta.get("aliases", []))
            text_parts.extend(meta.get("keywords", []))
            text = " ".join(str(p) for p in text_parts if p)
            _tool_embeddings[tool_name] = embedding_model.embed_query(text)
    except Exception as e:
        print(f"[WARN] Failed to build tool embeddings: {e}")
        _tool_embeddings.clear()

ERP_AMBIGUOUS_THRESHOLD = 0.0
def max_erp_similarity(query: str) -> float:
    return 0.0

_TOKENIZER = None

def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return _TOKENIZER

def log_token_usage(response, node_name: str, input_text: str = "", output_text: str = ""):
    response_metadata = getattr(response, "response_metadata", {}) or {}

    usage = (
        getattr(response, "usage_metadata", None)
        or response_metadata.get("token_usage", {})
        or response_metadata.get("usage", {})
        or {}
    )

    input_tokens = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or 0
    )

    output_tokens = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or 0
    )

    total_tokens = (
        usage.get("total_tokens")
        or input_tokens + output_tokens
    )
    logger.info(
                    "Token usage",
                    extra={
                        "node": node_name,
                        "provider": provider,
                        "model": model,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                    },
                )
def print_ollama_metadata(response):
    metadata = getattr(response, "response_metadata", {}) or {}
    print("\n========== OLLAMA METADATA ==========")
    model = metadata.get("model") or metadata.get("model_name", "unknown")
    print("model:", model)
    print("done_reason:", metadata.get("done_reason", "N/A"))
    total_dur = metadata.get("total_duration")
    print("total_duration:", f"{ns_to_sec(total_dur):.2f}s" if total_dur else "N/A")
    load_dur = metadata.get("load_duration")
    print("load_duration:", f"{ns_to_sec(load_dur):.2f}s" if load_dur else "N/A")
    prompt_eval_dur = metadata.get("prompt_eval_duration")
    print("prompt_eval_duration:", f"{ns_to_sec(prompt_eval_dur):.2f}s" if prompt_eval_dur else "N/A")
    eval_dur = metadata.get("eval_duration")
    print("eval_duration:", f"{ns_to_sec(eval_dur):.2f}s" if eval_dur else "N/A")
    print("prompt_eval_count:", metadata.get("prompt_eval_count", "N/A"))
    print("eval_count:", metadata.get("eval_count", "N/A"))
    print("=====================================\n")

def parse_planner_json_blocks(text: str) -> list:
    if not text:
        return []
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        return [parsed]
    except Exception:
        pass
    blocks = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(cleaned):
        while idx < len(cleaned) and cleaned[idx] not in "[{":
            idx += 1
        if idx >= len(cleaned):
            break
        try:
            obj, end = decoder.raw_decode(cleaned[idx:])
            blocks.append(obj)
            idx += end
        except Exception:
            idx += 1
    return blocks

def normalize_tool_name(name: str) -> str:
    if not name:
        return ""
    name = str(name).strip()
    name = name.replace(" ", "_")
    if "=" in name:
        name = name.split("=", 1)[0].strip()
    return TOOL_NAME_ALIASES.get(name, name)

def extract_date_ranges_with_positions(query: str) -> list[dict]:
    return []
def nearest_date_range_to_keyword(query: str, keywords: list[str]) -> tuple[str, str]:
    return "", ""
def get_segment_for_tool(query: str, date_keywords: list[str]) -> str:
    return query or ""
def extract_date_range_for_tool(query: str, date_keywords: list[str]) -> tuple[str, str]:
    return "", ""

def sanitize_tool_filters(name: str, args: dict) -> dict:
    return args

def strip_think_tags(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def is_plain_english_query(query: str) -> bool:
    return False
