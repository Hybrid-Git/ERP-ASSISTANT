import json
import re
import time
from types import MappingProxyType
import numpy as np
from src.config import embedding_model, get_cfg
from src.tool_doc import TOOL_INTENT_REGISTRY, TOOL_NAME_ALIASES
import re


NON_ENGLISH_HINTS = get_cfg("hinglish", "non_english_hints", default=[])
MULTILINGUAL_WORDS = get_cfg("hinglish", "multilingual_words", default=[])
ROUTE_KEYWORDS = get_cfg("route_keywords", default=[])
CONNECTORS = get_cfg("connectors", default=[])
STOP_TOKENS = set(get_cfg("stop_tokens", default=[]))
SEGMENT_NEXT_KEYWORDS = get_cfg("segment_next_keywords", default=[])
TH_EMBEDDING_RECALL_MIN = get_cfg("thresholds", "embedding_recall_min", default=0.3)
TH_RERANKER_TOP_K = get_cfg("thresholds", "reranker_top_k", default=5)
PARTY_WORDS = get_cfg("party_words", default=[])
NAME_WORDS = get_cfg("name_words", default=[])
LIST_WORDS = get_cfg("list_words", default=[])
DOMAIN_KEYWORDS = MappingProxyType({
    "sales": ["sales", "sale", "sell", "sold", "overdue", "receivable", "debtor", "billing"],
    "purchase": ["purchase", "kharidi", "buy", "bought", "payable", "creditor", "bills payable"],
    "customer": ["customer", "client", "party", "ledger", "customer name", "customer list", "customer code"],
    "vendor": ["vendor", "supplier", "vendor list"],
    "gst": ["gst", "gst summary", "gst detail", "gstr", "taxable", "igst", "cgst", "sgst", "b2b", "b2c"],
    "tax": ["tds", "tcs", "tax deducted", "tax collected", "tax outstanding"],
    "stock": ["stock", "inventory", "quantity", "hsn", "product", "slow moving", "stock level"],
    "analytics": ["top", "popular", "trend", "summary", "analytics", "report"],
})
INVOICE_PATTERNS = {
    "sales": [r"\bA/\d{4}/C\d{4}\b", r"\bAI/\d{4}/\d{4}\b", r"\bSI-?\d+\b", r"\bOUT-?\d+\b"],
    "purchase": [r"\bPR-?\d+\b"],
}
INVOICE_NO_PATTERNS = get_cfg("identifier_patterns", "invoice_no", default=[
    r'\bPR[-/]?\d+\b', r'\bA/\d{4}/C\d{4}\b', r'\bAI/\d{4}/\d{4}\b',
    r'\bAI/\d{2}-\d{2}/\d{3,4}\b', r'\bSI[-/]?\d+\b', r'\bOUT[-/]?\d+\b',
])
TOOL_DOMAINS = {
    "get_customer": ["customer"],
    "get_customer_ledger": ["customer"],
    "get_search_ledgers": ["ledger"],
    "get_stock_levels": ["stock"],
    "get_gst_summary": ["gst"],
    "get_tds_outstanding": ["tax"],
    "get_tcs_outstanding": ["tax"],
    "get_top_products": ["analytics"],
    "get_popular_products": ["analytics"],
    "get_slow_moving_products": ["analytics", "stock"],
    "get_sales_summary": ["analytics", "sales"],
    "get_sales_trend": ["analytics", "sales"],
    "get_top_customer": ["customer", "analytics"],
    "get_top_vendor": ["vendor", "analytics"],
    "get_purchase_summary": ["analytics", "purchase"],
    "get_search_vendors": ["vendor"],
    "get_outstanding_sales_invoices": ["sales"],
    "get_outstanding_purchase_invoices": ["purchase"],
    "get_overdue_invoices": ["sales", "purchase"],
}
INVOICE_TOOLS = {"get_outstanding_sales_invoices", "get_outstanding_purchase_invoices", "get_overdue_invoices"}

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

ERP_AMBIGUOUS_THRESHOLD = 0.30
_erp_domain_embedding: list[float] | None = None

def _build_erp_domain_embedding() -> list[float]:
    combined = " ".join(
        f"{meta.get('description', '')} {' '.join(meta.get('aliases', []))} {' '.join(meta.get('keywords', []))}"
        for meta in TOOL_INTENT_REGISTRY.values()
    )
    return embedding_model.embed_query(combined)

def max_erp_similarity(query: str) -> float:
    global _erp_domain_embedding
    if _erp_domain_embedding is None:
        try:
            _erp_domain_embedding = _build_erp_domain_embedding()
        except Exception:
            return 0.0
    try:
        query_emb = embedding_model.embed_query(query)
        return _cosine_sim(query_emb, _erp_domain_embedding)
    except Exception:
        return 0.0

def log_token_usage(response, label: str):
    meta = getattr(response, "response_metadata", {}) or {}
    um = getattr(response, "usage_metadata", None) or {}
    token_usage = meta.get("token_usage", {}) or {}
    prompt_tokens = (um.get("input_tokens")
                     or meta.get("prompt_eval_count")
                     or token_usage.get("prompt_tokens", 0))
    output_tokens = (um.get("output_tokens")
                     or meta.get("eval_count")
                     or token_usage.get("completion_tokens", 0))
    model = meta.get("model") or meta.get("model_name", "unknown")
    model_provider = meta.get("model_provider", "")
    tag = f"[TOKENS] {label}"
    if model_provider:
        tag += f" | provider={model_provider}"
    total = (prompt_tokens or 0) + (output_tokens or 0)
    print(f"{tag} | model={model} | input={prompt_tokens or 0} | output={output_tokens or 0} | total={total}")

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
    pattern = r"\b\d{4}-\d{2}-\d{2}\b"
    matches = list(re.finditer(pattern, query or ""))
    ranges = []
    for i in range(0, len(matches) - 1, 2):
        ranges.append({
            "from": matches[i].group(),
            "to": matches[i + 1].group(),
            "pos": matches[i].start(),
        })
    if len(matches) % 2:
        print(f"[WARN] Dropped unpaired date: {matches[-1].group()}")
    return ranges

def nearest_date_range_to_keyword(query: str, keywords: list[str]) -> tuple[str, str]:
    q_lower = (query or "").lower()
    ranges = extract_date_ranges_with_positions(query)
    if not ranges:
        return "", ""
    keyword_positions = []
    for kw in keywords:
        idx = q_lower.find(kw.lower())
        if idx != -1:
            keyword_positions.append(idx)
    if not keyword_positions:
        return ranges[0]["from"], ranges[0]["to"]
    key_pos = min(keyword_positions)
    selected = min(ranges, key=lambda r: abs(r["pos"] - key_pos))
    return selected["from"], selected["to"]

def get_segment_for_tool(query: str, date_keywords: list[str]) -> str:
    q = query or ""
    q_lower = q.lower()
    positions = [q_lower.find(k) for k in date_keywords if q_lower.find(k) != -1]
    if not positions:
        return q
    start = min(positions)
    ends = []
    for k in SEGMENT_NEXT_KEYWORDS:
        idx = q_lower.find(k, start + 1)
        if idx != -1 and idx > start:
            ends.append(idx)
    end = min(ends) if ends else len(q)
    return q[start:end]

def extract_date_range_for_tool(query: str, date_keywords: list[str]) -> tuple[str, str]:
    segment = get_segment_for_tool(query, date_keywords)
    ranges = extract_date_ranges_with_positions(segment)
    if ranges:
        return ranges[0]["from"], ranges[0]["to"]
    return nearest_date_range_to_keyword(query, date_keywords)

def sanitize_tool_filters(name: str, args: dict) -> dict:
    return args



def strip_think_tags(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
