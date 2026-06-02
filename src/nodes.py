from src.schema import MainState
from src.tools_api import tools_dict, tools
from src.tool_doc import TOOL_INTENT_REGISTRY, TOOL_NAME_ALIASES, get_field_triggers, infer_requested_fields_from_registry, CITY_WORDS
from src.config import llm, normalizer_llm, get_cfg
import time
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
import json
import re
from langsmith import traceable

import numpy as np
from sentence_transformers import CrossEncoder # type: ignore
from src.config import embedding_model

# ── Pipeline config from config.yaml ──
NON_ENGLISH_HINTS = get_cfg("hinglish", "non_english_hints", default=[])
MULTILINGUAL_WORDS = get_cfg("hinglish", "multilingual_words", default=[])
ROUTE_KEYWORDS = get_cfg("route_keywords", default=[])
CONNECTORS = get_cfg("connectors", default=[])
STOP_TOKENS = set(get_cfg("stop_tokens", default=[]))
SEGMENT_NEXT_KEYWORDS = get_cfg("segment_next_keywords", default=[])
TH_EMBEDDING_RECALL_MIN = get_cfg("thresholds", "embedding_recall_min", default=0.3)
TH_RERANKER_MIN = get_cfg("thresholds", "reranker_min", default=0.5)
TH_RERANKER_TOP_K = get_cfg("thresholds", "reranker_top_k", default=5)
CROSS_ENCODER_MODEL = get_cfg("cross_encoder_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
PARTY_WORDS = get_cfg("party_words", default=[])
NAME_WORDS = get_cfg("name_words", default=[])
LIST_WORDS = get_cfg("list_words", default=[])

# ── Embedding recall + cross-encoder reranker for tool routing ──
_tool_embeddings: dict[str, list[float]] = {}
_cross_encoder: CrossEncoder

def _cosine_sim(a: list[float], b: list[float]) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _build_tool_embeddings():
    for tool_name, meta in TOOL_INTENT_REGISTRY.items():
        text = f"{meta['description']} {' '.join(meta.get('aliases', []))} {' '.join(meta.get('keywords', []))}"
        _tool_embeddings[tool_name] = embedding_model.embed_query(text)

_build_tool_embeddings()

# Eager load cross-encoder at import time (not lazy — avoids cold-start latency on first query)
_cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

def now():
    return time.perf_counter()

TRANSLATOR_PROMPT = """
You are an ERP query normalizer. Convert Hinglish/Hindi/Gujarati queries to canonical English JSON.

CRITICAL: NEVER translate standalone words. If user types "mars" output "mars", not March. Preserve words literally. Hinglish hints below are for ERP terms only (bill→invoice, sale→sales), NOT for converting ambiguous words.

Schema: {"canonical_query":"...","document_type":"...","language":"...","confidence":"high|medium|low"}

ERP hints: bill/invoice=invoice, sale/bikri=sales, purchase/kharidi=purchase, customer/grahak= customer, vendor/vikreta=vendor, amount/rakam=net amount, baki/due=outstanding amount, stock/qty=closing quantity, kam/zyada=less/greater, dikhao/batao=show, aur/ane=and

Rules: Preserve IDs, HSN, dates, names exactly. Keep all intents in multi-part queries. If query is already clean English, return as-is with language "english". document_type: sales_invoice / purchase_invoice / unknown_invoice / product / mixed. bill/invoice alone→unknown_invoice.

Examples:
Q: A/0326/C0077 sales bill ka customer name, amount aur status batao
A: {"canonical_query": "Show customer name, net amount and status for sales invoice A/0326/C0077", "document_type": "sales_invoice", "language": "hinglish", "confidence": "high"}

Q: muje mars ke customer ka detail chaia
A: {"canonical_query": "Show customer details for mars", "document_type": "customer", "language": "hinglish", "confidence": "medium"}
"""
def is_plain_english_query(query: str) -> bool:
    """
    Returns True when the query looks like normal English.
    Mixed Hindi/Gujarati/Marathi slang should return False
    so translator can normalize it.
    """

    q = query.lower().strip()

    if not q:
        return True

    # Detect Devanagari/Gujarati script characters
    for char in q:
        code = ord(char)

        # Devanagari block: Hindi/Marathi
        if 0x0900 <= code <= 0x097F:
            return False

        # Gujarati block
        if 0x0A80 <= code <= 0x0AFF:
            return False

    words = set(q.replace(",", " ").replace("?", " ").split())

    return not any(word in words for word in NON_ENGLISH_HINTS)

def needs_translation(query: str) -> bool:
    q = query.lower()
    words = set(re.sub(r"[^\w/.-]+", " ", q).split())
    return bool(words & set(MULTILINGUAL_WORDS))


def is_routeable_without_translator(query: str) -> bool:
    """
    Returns True when the query already contains enough ERP/tool-domain
    keywords for semantic_search to route it without normalizer_llm.

    This is tool-level routing only. It does not hardcode output columns.
    """
    q = re.sub(r"\s+", " ", (query or "").lower()).strip()

    return any(keyword in q for keyword in ROUTE_KEYWORDS)


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

@traceable(name="translator_node", run_type="chain")
async def translator_node(state: MainState) -> MainState:
    """
    Translates only when needed.

    Fast path:
    - Plain English queries skip translator.
    - ERP queries routeable by keyword/domain (and no Hinglish words) skip translator.

    Slow path:
    - Any query with Hinglish/Hindi/Gujarati words gets normalized by normalizer_llm.
    """
    try:
        print("Translator node triggered")

        user_query = state.get("user_query", "") or ""

        if not user_query:
            return {
                "original_query": "",
                "canonical_query": "",
                "user_query": "",
                "translator_used": False,
                "translator_confidence": "low",
                "detected_language": "unknown",
                "document_type": "unknown",
            }

        if is_plain_english_query(user_query):
            print("Translator skipped: query looks English")
            return {
                "original_query": user_query,
                "canonical_query": user_query,
                "user_query": user_query,
                "translator_used": False,
                "translator_confidence": "skipped_english",
                "detected_language": "english",
                "document_type": "routeable",
            }

        if needs_translation(user_query):
            print("Translator: query needs Hinglish/Hindi/Gujarati normalization")
            response = await normalizer_llm.ainvoke([
                SystemMessage(content=TRANSLATOR_PROMPT),
                HumanMessage(content=user_query),
            ])
            log_token_usage(response, "translator")

            data = extract_json_object(response.content)

            canonical_query = data.get("canonical_query") or user_query
            language = data.get("language", "mixed")
            confidence = data.get("confidence", "medium")

            print("Original query:", user_query)
            print("Canonical query:", canonical_query)
            print("Detected language:", language)
            print("Translator confidence:", confidence)

            return {
                "original_query": user_query,
                "canonical_query": canonical_query,
                "user_query": canonical_query,
                "translator_used": True,
                "translator_confidence": confidence,
                "detected_language": language,
                "document_type": data.get("document_type", "unknown"),
            }

        if is_routeable_without_translator(user_query):
            print("Translator skipped: query is directly routeable by ERP keywords")
            return {
                "original_query": user_query,
                "canonical_query": user_query,
                "user_query": user_query,
                "translator_used": False,
                "translator_confidence": "skipped_routeable",
                "detected_language": "mixed_or_english",
                "document_type": "routeable",
            }

        print("Translator skipped: no multilingual normalization needed")
        return {
            "original_query": user_query,
            "canonical_query": user_query,
            "user_query": user_query,
            "translator_used": False,
            "translator_confidence": "skipped_no_normalization_needed",
            "detected_language": "english_or_mixed",
            "document_type": "unknown",
        }
    except Exception as e:
        print(f"Translator failed: {e}")
        user_query = state.get("user_query", "") or ""
        return {
            "original_query": user_query,
            "canonical_query": user_query,
            "user_query": user_query,
            "translator_used": False,
            "translator_confidence": "low",
            "detected_language": "unknown",
            "document_type": "unknown",
        }

def score_tools_via_reranker(query_part: str, registry: dict) -> list[str]:
    """Embedding recall + cross-encoder reranker for tool selection."""
    if not _tool_embeddings:
        return []

    try:
        query_emb = embedding_model.embed_query(query_part)
    except Exception:
        return []

    scores = []
    for tool_name, tool_emb in _tool_embeddings.items():
        sim = _cosine_sim(query_emb, tool_emb)
        scores.append((tool_name, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    if not scores or scores[0][1] < TH_EMBEDDING_RECALL_MIN:
        return []

    top_k = scores[:TH_RERANKER_TOP_K]
    pairs = []
    for tool_name, _ in top_k:
        meta = registry.get(tool_name, {})
        desc = f"{tool_name}: {meta.get('description', '')}. Aliases: {', '.join(meta.get('aliases', []))}"
        pairs.append((query_part, desc))

    rerank_scores = _cross_encoder.predict(pairs)
    reranked = [(top_k[i][0], float(rerank_scores[i])) for i in range(len(top_k))]
    reranked.sort(key=lambda x: x[1], reverse=True)

    result = []
    for tool_name, score in reranked:
        if score > TH_RERANKER_MIN:
            result.append(tool_name)
    return result

def add_unique(items: list[str], value: str):
    """Append a tool name only once while preserving order."""
    if value and value not in items:
        items.append(value)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()



def split_query_parts(query: str) -> list[str]:
    """
    Splits one query into intent parts.
    Kept for backward compatibility, but semantic_search now uses
    split_query_for_tools() so original and canonical queries are not
    concatenated before splitting.
    """
    if not query:
        return []

    split_pattern = "|".join(
        rf"\s+{re.escape(c)}\s+" for c in CONNECTORS
    ) + "|;\\s*"

    return [
        part.strip()
        for part in re.split(split_pattern, query, flags=re.IGNORECASE)
        if part.strip()
    ]


def split_query_for_tools(original_query: str, canonical_query: str = "") -> list[str]:
    """
    Split original and canonical query separately.

    Important: do NOT concatenate original + canonical before splitting.
    Concatenation created broken parts such as:
    'closing quantity dikhao Show Nykaa Bangalore customer ID'
    """
    parts: list[str] = []

    for query in [original_query, canonical_query]:
        for part in split_query_parts(query):
            if part and part not in parts:
                parts.append(part)

    return parts or [original_query or canonical_query]


def _keyword_fallback(part: str) -> list[str]:
    """Fallback: match registry keywords/aliases against the query part.
    Uses word-boundary matching to prevent false positives
    (e.g. "credit" inside "creditNotesUnregistered")."""
    q = normalize_text(part)
    matched = []
    for tool_name, meta in TOOL_INTENT_REGISTRY.items():
        for kw in meta.get("keywords", []):
            if re.search(rf"(?<!\w){re.escape(kw.lower())}(?!\w)", q):
                add_unique(matched, tool_name)
                break
        if tool_name not in matched:
            for alias in meta.get("aliases", []):
                if re.search(rf"(?<!\w){re.escape(alias.lower())}(?!\w)", q):
                    add_unique(matched, tool_name)
                    break
    return matched


def merge_unique_tools(tool_lists: list[list[str]]) -> list[str]:
    merged: list[str] = []

    for tool_group in tool_lists:
        for tool_name in tool_group:
            add_unique(merged, tool_name)

    return merged


def is_multi_intent_query(original_query: str, canonical_query: str, query_parts: list[str]) -> bool:
    combined = f"{original_query or ''} {canonical_query or ''}".lower()

    if any(c in combined for c in CONNECTORS):
        return True

    return len(query_parts) > 1
# ============================================
# SEMANTIC SEARCH NODE
# ============================================

@traceable(name="semantic_search_node", run_type="retriever")
async def semantic_search(state: MainState) -> MainState:
    try:
        print("Semantic search node triggered")

        original_query = state.get("original_query") or state.get("user_query", "") or ""
        canonical_query = state.get("canonical_query", "") or ""
        document_type = (state.get("document_type", "") or "").lower().strip()

        user_query = canonical_query or original_query

        if not user_query:
            return {
                "retrieved_tools": [],
                "selected_tools": [],
                "query_parts": [],
                "skip_router": True,
            }

        print(f"Original query: {original_query}")
        print(f"Canonical query: {canonical_query}")
        print(f"Document type: {document_type}")

        query_parts = split_query_for_tools(
            original_query=original_query,
            canonical_query=canonical_query,
        )

        if not query_parts:
            query_parts = [user_query]

        print(f"Query parts for metadata matching: {query_parts}")

        selected_tool_groups: list[list[str]] = []

        for part in query_parts:
            tools_for_part = score_tools_via_reranker(part, TOOL_INTENT_REGISTRY)

            if tools_for_part:
                print(f"Reranker tools for part '{part}': {tools_for_part}")
                selected_tool_groups.append(tools_for_part)
                continue

            keyword_tools = _keyword_fallback(part)
            if keyword_tools:
                print(f"Keyword fallback tools for part '{part}': {keyword_tools}")
                selected_tool_groups.append(keyword_tools)

        # Optional document_type hint from translator (additive only).
        if document_type in {"product", "inventory", "stock"}:
            selected_tool_groups.append(["get_stock_levels"])
        elif document_type in {"customer", "party"}:
            selected_tool_groups.append(["get_customer"])
        elif document_type in {"customer_ledger", "ledger"}:
            selected_tool_groups.append(["get_customer_ledger"])

        selected_tools = merge_unique_tools(selected_tool_groups)

        selected_tools = [
            tool_name for tool_name in selected_tools
            if tool_name in tools_dict
        ]

        if selected_tools:
            print(f"Final selected tools: {selected_tools}")

            return {
                "retrieved_tools": selected_tools,
                "selected_tools": selected_tools,
                "query_parts": query_parts,
                "skip_router": True,
            }

        print("No confident tool match. Marking query unsupported.")

        return {
            "retrieved_tools": [],
            "selected_tools": [],
            "query_parts": query_parts,
            "skip_router": True,
            "unsupported": True,
            "unsupported_reason": "No supported ERP tool matched this query.",
        }

    except Exception as e:
        print(f"Error in semantic search node: {e}")

        return {
            "retrieved_tools": [],
            "selected_tools": [],
            "query_parts": [],
            "skip_router": True,
        }

# ============================================
# SYSTEM PROMPT
# ============================================
def _build_tool_desc(tool_name: str, meta: dict) -> str:
    """One-line tool description for the system prompt."""
    fields = meta.get("fields", [])
    field_str = ",".join(fields[:5])
    if len(fields) > 5:
        field_str += "..."
    return f"{tool_name}={meta.get('category', '')}: {meta.get('description', '')} Fields: [{field_str}]."


def _build_field_examples(tool_name: str, meta: dict) -> list[str]:
    """Generate field-usage examples from the registry for the system prompt."""
    examples = []
    triggers = get_field_triggers(tool_name)

    for keyword, triggered_fields in triggers.items():
        if keyword in ["name", "id", "category"]:
            continue
        if len(triggered_fields) <= 2:
            default = meta.get("default_fields", [])
            all_fields = list(dict.fromkeys(default + triggered_fields))
            examples.append(
                f"{keyword}=>{tool_name}(fields={json.dumps(all_fields, ensure_ascii=False)})"
            )

    return examples

def build_system_prompt(
    user_query: str,
    selected_tools: list[str],
    query_parts: list[str] | None = None
) -> str:
    lines = [
        "You are an ERP assistant. Use the available tools to answer the user.",
        'Preserve all query text literally. Do not reinterpret or assume intent. If user says "mars" use "mars", not March.',
        "Never invent IDs, names, dates, or amounts.",
        "You MUST call at least one tool. Never answer in prose without a tool call.",
        "Do NOT output any thinking or reasoning — call the tool directly.",
        "Set `fields` to only the columns the user explicitly asks for. "
        'E.g. "sirf name" → fields=["name"]; "name and cgst" → fields=["name","cgst"]. '
        "Omit `fields` if user doesn't specify any columns.",
    ]

    if selected_tools:
        lines.append("")
        lines.append(f"You MUST call ALL {len(selected_tools)} tools: {', '.join(selected_tools)}")
        lines.append("Do NOT skip any. Each tool addresses a different part of the user's request.")
        lines.append("")
        lines.append("Tool rules:")
        for tool_name in selected_tools:
            meta = TOOL_INTENT_REGISTRY.get(tool_name)
            if meta and meta.get("prompt_tips"):
                lines.append(f"  {tool_name}: {meta['prompt_tips']}")

    lines.append("")
    lines.append("Example:")
    lines.append("  user: show me b2b invoices for april 2024")
    lines.append("  assistant: (tool call with from_date=2024-04-01, to_date=2024-04-30, categories=[b2b])")

    return "\n".join(lines)


def sec(start):
    return round(time.perf_counter() - start, 3)


def ns_to_sec(value):
    if value is None:
        return None
    try:
        return round(value / 1_000_000_000, 3)
    except Exception:
        return value


def log_token_usage(response, label: str):
    meta = getattr(response, "response_metadata", {}) or {}
    tu = meta.get("token_usage", {}) or {}
    prompt_tokens = tu.get("prompt_tokens") or meta.get("prompt_eval_count", 0)
    output_tokens = tu.get("completion_tokens") or meta.get("eval_count", 0)
    model = tu.get("model") or meta.get("model", "unknown")
    model_provider = meta.get("model_provider", "")
    tag = f"[TOKENS] {label}"
    if model_provider:
        tag += f" | provider={model_provider}"
    print(f"{tag} | model={model} | input={prompt_tokens} | output={output_tokens} | total={prompt_tokens + output_tokens}")


def print_ollama_metadata(response):
    metadata = getattr(response, "response_metadata", {}) or {}

    print("\n========== OLLAMA METADATA ==========")
    print("model:", metadata.get("model"))
    print("done_reason:", metadata.get("done_reason"))

    print("total_duration:", ns_to_sec(metadata.get("total_duration")), "sec")
    print("load_duration:", ns_to_sec(metadata.get("load_duration")), "sec")
    print("prompt_eval_duration:", ns_to_sec(metadata.get("prompt_eval_duration")), "sec")
    print("eval_duration:", ns_to_sec(metadata.get("eval_duration")), "sec")

    print("prompt_eval_count:", metadata.get("prompt_eval_count"))
    print("eval_count:", metadata.get("eval_count"))
    print("=====================================\n")


# ============================================
# TOLERANT JSON PLANNER PARSER
# ============================================
def parse_planner_json_blocks(text: str) -> list:
    """
    Handles:
    - single JSON object
    - single JSON array
    - markdown JSON fences
    - multiple JSON arrays/objects in one response
    """
    if not text:
        return []

    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # First try full JSON parse
    try:
        parsed = json.loads(cleaned)
        return [parsed]
    except Exception:
        pass

    # Extract multiple JSON arrays/objects
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
    """Return the substring of query most relevant to a tool, bounded by
    the next major tool keyword after this tool's first keyword match."""
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
    """Find the first date range within the tool's query segment,
    falling back to nearest-keyword on the full query."""
    segment = get_segment_for_tool(query, date_keywords)
    ranges = extract_date_ranges_with_positions(segment)
    if ranges:
        return ranges[0]["from"], ranges[0]["to"]
    return nearest_date_range_to_keyword(query, date_keywords)


def expand_customer_city_calls(base_name: str, base_args: dict, user_query: str) -> list[dict]:
    """Create per-city get_customer calls when multiple known cities, or
    filter by unknown location token when Nykaa + <unknown> is present."""
    q_upper = (user_query or "").upper()
    q_lower = (user_query or "").lower()
    extra: list[dict] = []

    if base_name != "get_customer":
        return extra

    if "NYKAA" not in q_upper:
        return extra

    matched_cities = [c for c in CITY_WORDS if c in q_upper]

    if len(matched_cities) > 1:
        for city in matched_cities:
            city_args = dict(base_args)
            city_args["filters"] = {"name": {"contains": city}}
            extra.append({
                "name": "get_customer",
                "args": city_args,
                "id": f"call_get_customer_{city.lower()}",
                "type": "tool_call",
            })
        return extra

    if not matched_cities and "NYKAA" in q_upper:
        # Nykaa <unknown location> — prevent broad dump
        after = q_upper.split("NYKAA", 1)[1]
        tokens = re.findall(r"\b[A-Z]+\b", after)
        unknown = next((t for t in tokens if t not in STOP_TOKENS), None)
        if unknown:
            filtered_args = dict(base_args)
            filtered_args["filters"] = {"name": {"contains": unknown}}
            extra.append({
                "name": "get_customer",
                "args": filtered_args,
                "id": f"call_get_customer_{unknown.lower()}",
                "type": "tool_call",
            })
            return extra

    return extra


def expand_multi_identifier_calls(base_name: str, base_args: dict, user_query: str) -> list[dict]:
    """When a stock query has HSN filter but the query also mentions id <number>,
    create a second call for the id filter. E.g. '49090090 aur id 349' needs both."""
    extra: list[dict] = []

    if base_name != "get_stock_levels":
        return extra

    q_lower = (user_query or "").lower()

    # Check if the original call already has an HSN filter
    filters = base_args.get("filters") or {}
    has_hsn = "hsnCode" in filters

    # Check if query also mentions an id
    id_match = re.search(r"\bid\s*[:#-]?\s*(\d+)\b", q_lower)

    if has_hsn and id_match:
        product_id = int(id_match.group(1))
        # Create a separate call for the id
        id_args = dict(base_args)
        id_args["filters"] = {"id": product_id}
        id_args.pop("term", None)
        extra.append({
            "name": "get_stock_levels",
            "args": id_args,
            "id": "call_get_stock_levels_by_id",
            "type": "tool_call",
        })

    return extra


@traceable(name="chat_model_node", run_type="llm")
async def chat_model_node(state: MainState):
    node_start = now()

    try:
        print("\n========== CHAT MODEL NODE START ==========")

        step = now()
        original_query = state.get("original_query") or state.get("user_query", "")
        user_query = state.get("canonical_query") or state.get("user_query", "")
        selected_tools = state.get("selected_tools", [])
        query_parts = state.get("query_parts", [user_query])
        loop_count = state.get("loop_count", 0)

        previous_messages = [
            msg for msg in state.get("messages", [])
            if not isinstance(msg, SystemMessage)
        ]

        print(f"[1] Read state: {sec(step)}s")
        print("user_query:", user_query)
        print("selected_tools:", selected_tools)
        print("query_parts:", query_parts)
        print("loop_count:", loop_count)

        step = now()
        available_tools = [
            tools_dict[name]
            for name in selected_tools
            if name in tools_dict
        ]

        print(f"[2] Loaded available tools: {sec(step)}s")
        print("available_tool_names:", [tool.name for tool in available_tools])

        if not available_tools:
            print("[CHAT MODEL] No available tools. Ending safely.")
            reason = state.get("unsupported_reason", "No available tools for this query.")

            return {
                "messages": [
                    HumanMessage(content=user_query),
                    AIMessage(content=reason),
                ],
                "loop_count": loop_count + 1,
            }

        step = now()
        print("[3] Using bind_tools")

        prompt_start = time.perf_counter()

        system_prompt_text = build_system_prompt(
            user_query=user_query,
            selected_tools=selected_tools,
            query_parts=query_parts,
        )

        prompt_duration = time.perf_counter() - prompt_start

        print(f"[4] Built system prompt: {prompt_duration:.3f}s")
        print("system_prompt_chars:", len(system_prompt_text))

        chat_history = [
            msg for msg in state.get("messages", [])
            if not isinstance(msg, SystemMessage)
        ]

        system_prompt = SystemMessage(content=system_prompt_text)

        llm_input = (
            [system_prompt]
            + chat_history
            + [HumanMessage(content=user_query)]
        )

        print(f"[5] Built LLM input messages: {sec(prompt_start)}s")
        print("message_count:", len(llm_input))
        print("message_types:", [type(m).__name__ for m in llm_input])

        all_raw_calls = []
        called_names = set()
        remaining_names = list(selected_tools)
        loop_input = llm_input
        retry_count = 0

        while remaining_names and retry_count < 3:
            retry_count += 1
            remaining_tools = [
                t for t in available_tools if t.name in remaining_names
            ]
            if not remaining_tools:
                break

            step = now()
            print(f"[6] Invoking LLM with bind_tools (round {retry_count}, tools: {[t.name for t in remaining_tools]})...")
            response = await llm.bind_tools(remaining_tools).ainvoke(loop_input)
            print(f"[6] LLM invoke completed: {sec(step)}s")
            log_token_usage(response, "chat_model")

            print("\n========== RAW WORKER RESPONSE DEBUG ==========")
            print("response_type:", type(response).__name__)
            print("content:", repr(getattr(response, "content", "")))
            print("tool_calls:", getattr(response, "tool_calls", None))
            print("additional_kwargs:", getattr(response, "additional_kwargs", {}))
            print("response_metadata:", getattr(response, "response_metadata", {}))
            print("==============================================\n")

            print_ollama_metadata(response)

            raw_tool_calls = getattr(response, "tool_calls", None) or []
            for call in raw_tool_calls:
                name = call.get("name", "")
                if name and name not in called_names:
                    called_names.add(name)
                    all_raw_calls.append(call)

            remaining_names = [
                n for n in selected_tools if n not in called_names
            ]

            if not remaining_names:
                break

            print(f"[RETRY] Missing tool calls for: {remaining_names}")
            loop_input = (
                [llm_input[0]]  # system prompt
                + llm_input[1:-1]  # chat history (exclude prev human/ai messages from this node)
                + [HumanMessage(content=user_query)]
                + [response]
                + [HumanMessage(content=f"You still need to call the following tool(s): {', '.join(remaining_names)}. Call them now.")]
            )

        raw_tool_calls = all_raw_calls
        tool_calls = []

        # ---------- deterministic repair helpers ----------
        def _apply_repair(name, args, user_query):
            meta = TOOL_INTENT_REGISTRY.get(name, {})
            repair = meta.get("repair")

            if not repair:
                return {
                    "name": name,
                    "args": args,
                }

            combined_q = f"{original_query or ''} {state.get('canonical_query', '') or ''}".lower()

            if args:
                flds = args.get("fields")
                if isinstance(flds, dict):
                    from src.tools_api import normalize_fields
                    args["fields"] = normalize_fields(flds)
                elif isinstance(flds, str):
                    args["fields"] = [f.strip() for f in flds.split(",") if f.strip()]

                # If LLM sent no fields, build from query triggers only (no force-added extras)
                if "fields" not in args or not args.get("fields"):
                    args["fields"] = list(repair.get("default_fields", ["name"]))

                # Clear term if it looks like a filter expression, not a product name
                term = args.get("term")
                if term and isinstance(term, str):
                    if re.search(r"\b(lt|gt|lte|gte|eq|ne|in|\$lt|\$gt|<=|>=|!=)\b", term):
                        args["term"] = ""

            worker_has = {}

            if args:
                for dk in ("from_date", "to_date"):
                    v = args.get(dk)

                    if v and re.match(r"\d{4}-\d{2}-\d{2}", str(v)):
                        worker_has[dk] = v

            # Discard hallucinated dates: if LLM provided dates but the query
            # has no date reference at all (neither YYYY-MM-DD nor a 4-digit year),
            # they are invented — let defaults apply.
            if worker_has and not re.search(r"\d{4}-\d{2}-\d{2}|\b\d{4}\b", combined_q):
                worker_has = {}

            # Preserve worker's explicit non-standard params before overwrite.
            # Exclude category/categories — category_map + category_to_filter handle them.
            worker_extra = {}
            if repair.get("overwrite") and args:
                for k, v in args.items():
                    if k not in ("from_date", "to_date", "fields", "category", "categories") and v is not None:
                        worker_extra[k] = v

            new_args = (
                dict(repair.get("base_args", {}))
                if repair.get("overwrite")
                else dict(args or {})
            )

            # Preserve worker's valid dates
            for dk, dv in worker_has.items():
                new_args[dk] = dv

            for kw, kwar in repair.get("keyword_args", {}).items():
                if kw.lower() in combined_q:
                    new_args.update(kwar)

            city_cfg = repair.get("city_filter")

            if city_cfg:
                matched = [
                    c for c in CITY_WORDS
                    if c in combined_q.upper()
                ]

                if len(matched) == 1:
                    new_args["filters"] = {
                        city_cfg.get("key", "name"): {
                            "contains": matched[0]
                        }
                    }

            if repair.get("hsn_extract"):
                hsn_match = re.search(r"\b(\d{8})\b", combined_q)

                if hsn_match:
                    hsn = hsn_match.group(1)

                    # Try labeled format first: "name: XYZ" or "product: XYZ"
                    name_match = re.search(
                        r"(?:name|product|item|naam)[:\s]+(.+?)(?:\s+(?:hsn|closing|stock|qty|quantity)|\s*$)",
                        combined_q,
                        re.IGNORECASE
                    )
                    product_name = name_match.group(1).strip() if name_match else None

                    # Only use product_name if it looks clean (no question words, max ~5 words)
                    if product_name and (
                        re.search(r"\b(kya|hai|batao|dikhao|show|what|is|the)\b", product_name, re.IGNORECASE)
                        or len(product_name.split()) > 5
                    ):
                        product_name = None

                    new_args["term"] = product_name or hsn
                    new_args["filters"] = {
                        "hsnCode": hsn
                    }

                    if product_name:
                        new_args["filters"]["name"] = {"contains": product_name}

                    fields = list(
                        repair.get(
                            "default_fields",
                            ["name", "id", "hsnCode", "closingQty"]
                        )
                    )

                    all_triggers = get_field_triggers(name)

                    for kw, triggered_fields in all_triggers.items():
                        match_found = (
                            kw in combined_q
                            if " " in kw
                            else bool(re.search(rf"\b{re.escape(kw)}\b", combined_q))
                        )

                        if match_found:
                            for fld in triggered_fields:
                                if fld not in fields:
                                    fields.append(fld)

                    new_args["fields"] = fields

                    return {
                        "name": name,
                        "args": new_args,
                    }

            cat_map = repair.get("category_map", {})

            if cat_map:
                matched = []

                for kw, val in cat_map.items():
                    if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", combined_q):
                        matched.append(val)

                # Remove generic parent when a specific child also matched.
                # E.g., "b2c" + "b2c small" → keep only "b2c small".
                # Only remove when parent val differs from child val
                # (e.g., "b2c"→"b2cLarge" + "b2c large"→"b2cLarge": same val, keep it).
                for kw in list(cat_map):
                    if " " in kw:
                        parent = kw.split(" ")[0]
                        pv = cat_map.get(parent)
                        cv = cat_map[kw]
                        if pv and pv in matched and cv in matched and pv != cv:
                            matched.remove(pv)

                # Generic prefix expansion: if a single-word keyword is a prefix of
                # multi-word keywords (e.g., "b2c" → "b2c small" + "b2c large"),
                # include all sub-variants when query matches only the parent word.
                prefix_kws = {}
                for kw in cat_map:
                    if " " in kw:
                        first = kw.split(" ")[0]
                        prefix_kws.setdefault(first, []).append(kw)
                for parent, children in prefix_kws.items():
                    if parent not in cat_map:
                        continue
                    parent_val = cat_map[parent]
                    if parent_val not in matched:
                        continue
                    has_specific = any(c in combined_q for c in children)
                    if not has_specific:
                        matched.remove(parent_val)
                        for child in children:
                            cv = cat_map[child]
                            if cv not in matched:
                                matched.append(cv)

                if len(matched) == 1:
                    new_args["category"] = matched[0]
                    new_args.pop("categories", None)

                elif len(matched) > 1:
                    new_args["categories"] = matched
                    new_args.pop("category", None)

            if repair.get("extract_customer_id"):
                cm = re.search(r"customer\s*id\s*[:#-]?\s*(\d+)", combined_q)

                if cm:
                    new_args["customer_id"] = int(cm.group(1))

            date_kws = repair.get("date_keywords")

            if date_kws and (
                not new_args.get("from_date")
                or not new_args.get("to_date")
            ):
                f, t = extract_date_range_for_tool(combined_q, date_kws)

                if f:
                    new_args["from_date"] = f
                    new_args["to_date"] = t

            if repair.get("remove_filters"):
                new_args.pop("filters", None)

            low_stock_kws = repair.get("low_stock_only_keywords")
            if low_stock_kws and new_args.get("low_stock_only") is True:
                if not any(kw in combined_q for kw in low_stock_kws):
                    new_args["low_stock_only"] = False

            # Generic value-comparison filter: detect "negative <field>",
            # "less than 0" / "0 se kam" / "< 0" with a field keyword → {field: {lt: 0}}.
            # And "positive" / "greater than 0" / "more than 0" / "0 se jyada" → {field: {gt: 0}}.
            # Applies to fields that look numeric (contain Qty/Value/Rate/Amount/Balance/Count/Gst/St).
            if "filters" not in new_args:
                is_lt_zero = bool(re.search(
                    r"\bnegative\b|\bless\s+th[ae]n\s+0\b|<\s*0\b|\bbelow\s+0\b|0\s+se\s+kam\b",
                    combined_q,
                ))
                is_gt_zero = bool(re.search(
                    r"\bpositive\b|\bgreater\s+th[ae]n\s+0\b|\bmore\s+th[ae]n\s+0\b|>\s*0\b|\babove\s+0\b|0\s+se\s+jyada\b|0\s+se\s+upar\b",
                    combined_q,
                ))
                if is_lt_zero or is_gt_zero:
                    operator = "gt" if is_gt_zero else "lt"
                    for field, aliases in TOOL_INTENT_REGISTRY.get(name, {}).get("field_aliases", {}).items():
                        if not re.search(r"(Qty|Value|Rate|Amount|Balance|Count|gst|igst|cgst|sgst|cess)", field, re.IGNORECASE):
                            continue
                        for alias in aliases:
                            if (" " in alias and alias in combined_q) or re.search(rf"\b{re.escape(alias)}\b", combined_q):
                                new_args.setdefault("filters", {})[field] = {operator: 0}
                                break

            # Normalize malformed filter keys: LLM sometimes sends
            # "closingQty gt": "2" (space) or "name.contains": "Bangalore" (dot)
            # instead of the correct nested format {"closingQty": {"gt": 2}}.
            norm_filters = new_args.get("filters")
            if norm_filters and isinstance(norm_filters, dict):
                _ops = {"gt", "gte", "lt", "lte", "eq", "ne", "contains", "in"}
                for raw_key in list(norm_filters):
                    norm_key = raw_key.replace(".", " ").strip()
                    if " " in norm_key:
                        parts = norm_key.rsplit(" ", 1)
                        if len(parts) == 2 and parts[1] in _ops:
                            field_name, operator = parts
                            raw_val = norm_filters.pop(raw_key)
                            try:
                                raw_val = float(raw_val)
                            except (TypeError, ValueError):
                                pass
                            existing = norm_filters.setdefault(field_name, {})
                            if isinstance(existing, dict):
                                existing[operator] = raw_val
                            # If existing is a primitive, that's a conflict — skip.

            for f in repair.get("prepend_fields", []):
                fields = new_args.setdefault("fields", [])

                if f not in fields:
                    fields.insert(0, f)

            strip = repair.get("strip_fields")

            if strip:
                fields = list(new_args.get("fields") or [])
                new_args["fields"] = [
                    f for f in fields
                    if f not in strip
                ]

            fixed = repair.get("fixed_fields")

            if fixed is not None:
                new_args["fields"] = list(
                    repair.get("default_fields", fixed)
                )

            for f in repair.get("ensure_fields", []):
                if f not in new_args.get("fields", []):
                    new_args.setdefault("fields", []).append(f)

            # Build the field list.
            # When LLM explicitly sent fields, use those as starting point.
            # Otherwise, fall back to default_fields.
            # Then apply curated field_triggers as a safety net for fields the
            # user asked for but LLM missed. Skip triggers when query has
            # "sirf"/"only"/"just"/"bas" (user wants ONLY those fields).
            llm_sent_fields = "fields" in args
            fields = list(
                (args.get("fields") or [])
                if llm_sent_fields
                else (repair.get("default_fields") or [])
            )
            has_strict_marker = bool(re.search(r'\b(sirf|only|just|bas)\b', combined_q))
            if not has_strict_marker:
                for kw, fld in repair.get("field_triggers", {}).items():
                    match = kw in combined_q if " " in kw else bool(re.search(rf'\b{re.escape(kw)}\b', combined_q))
                    if match and fld not in fields:
                        fields.append(fld)
            if fields:
                new_args["fields"] = fields

            strict_kws = repair.get("strict_field_keywords", {})

            if strict_kws:
                for kw_exact, narrow_fields in strict_kws.items():
                    if kw_exact in combined_q:
                        new_args["fields"] = list(narrow_fields)
                        break

            if "fields" in new_args and isinstance(new_args["fields"], list):
                flds = new_args["fields"]

                if "closingQuantity" in flds:
                    flds[flds.index("closingQuantity")] = "closingQty"

            # Merge back worker's explicit non-standard args that repair didn't set
            for k, v in worker_extra.items():
                if k not in new_args or new_args.get(k) in (None, "", []):
                    new_args[k] = v

            # Apply param_aliases from repair config (e.g., "name" → "term", "name" → "search")
            param_aliases = repair.get("param_aliases", {})
            for llm_arg, real_param in param_aliases.items():
                if llm_arg in new_args and real_param not in new_args:
                    new_args[real_param] = new_args.pop(llm_arg)

            # Convert category to filters when repair config says so.
            # Use "category" (singular) as the filter key to match flattened
            # GST-summary record fields — the API receives this verbatim but
            # doesn't actually filter server-side for GST summary.
            if repair.get("category_to_filter"):
                cat_val = None
                for cat_key in ("category", "categories"):
                    if cat_key in new_args:
                        cat_val = new_args.pop(cat_key)
                        break
                if cat_val:
                    new_args.setdefault("filters", {})["category"] = cat_val

            if name == "get_gst_summary" or name in (
                "get_tds_outstanding",
                "get_tcs_outstanding",
            ):
                print(f"[{name.upper()} FINAL ARGS] {json.dumps(new_args, default=str)}")

            # ──────────────────────────────────────────────────
            # Generic cross-tool corrections (applied to all tools)
            # ──────────────────────────────────────────────────

            # 1. Singular min/max query → limit=1.
            # "sabse kam/jyada X", "least/most/lowest/highest X".
            # Does not fire when query has an explicit count ("top 3", "first 5").
            if (
                re.search(r'\b(sabse\s+kam|sabse\s+jya[dz]a|sabse\s+zyada|least|most|lowest|highest|minimum|maximum)\b', combined_q)
                and not re.search(r'\b(top\s+\d+|first\s+\d+|last\s+\d+)\b', combined_q)
            ):
                if "limit" not in new_args or new_args.get("limit", 10) > 5:
                    new_args["limit"] = 1

            # 2. "Which entity?" query → ensure name field in results.
            # Patterns: "kis product ka", "kaunsa customer", "which party", etc.
            if re.search(r'\b(kis|kaunsa|kaun\s*sa|which)\s+(product|customer|party|item|entity)s?\b', combined_q):
                flds = new_args.get("fields", [])
                if flds and "name" not in flds:
                    flds.insert(0, "name")
                    new_args["fields"] = flds

            return {
                "name": name,
                "args": new_args,
            }

        def _repair_tool_call(name: str, args: dict) -> dict | None:
            name = TOOL_NAME_ALIASES.get(name, name)

            if name not in tools_dict:
                return None

            for alias, canonical in [
                ("date_from", "from_date"),
                ("date_to", "to_date"),
                ("startDate", "from_date"),
                ("endDate", "to_date"),
                ("fromDate", "from_date"),
                ("toDate", "to_date"),
                ("start_date", "from_date"),
                ("end_date", "to_date"),
            ]:
                if alias in args and canonical not in args:
                    args[canonical] = args.pop(alias)

            return _apply_repair(name, args, original_query)

        # Process bind_tools output — raw_tool_calls from bind_tools is already structured
        for call in raw_tool_calls:
            name = call.get("name", "")
            args = dict(call.get("args", {}))

            repaired = _repair_tool_call(name, args)

            if repaired:
                tool_calls.append({
                    "name": repaired["name"],
                    "args": repaired["args"],
                    "id": f"call_{repaired['name']}",
                    "type": "tool_call",
                })

        # Expand multi-city customer calls and unknown-location filters
        expanded = []

        for call in tool_calls:
            extra = expand_customer_city_calls(
                call["name"],
                call["args"],
                original_query,
            )

            if extra:
                expanded.extend(extra)
            else:
                expanded.append(call)
                multi_id_extra = expand_multi_identifier_calls(
                    call["name"],
                    call["args"],
                    original_query,
                )
                expanded.extend(multi_id_extra)

        tool_calls = expanded

        if tool_calls:
            seen = set()
            unique_calls = []

            for call in tool_calls:
                key = json.dumps(
                    {"name": call["name"], "args": call["args"]},
                    sort_keys=True,
                    default=str,
                )
                if key not in seen:
                    seen.add(key)
                    unique_calls.append(call)

            final_calls = []
            seen_names: dict[str, int] = {}

            for call in unique_calls:
                n = call["name"]
                meta = TOOL_INTENT_REGISTRY.get(n, {})
                if meta.get("multi_call_ok"):
                    final_calls.append(call)
                elif n not in seen_names:
                    seen_names[n] = 1
                    final_calls.append(call)

            tool_calls = final_calls
            response.__dict__["tool_calls"] = tool_calls

            print(f"[FIX] Extracted {len(tool_calls)} tool call(s) from bind_tools")

        print("\n========== WORKER LLM RESPONSE ==========")
        print("response_type:", type(response).__name__)
        print("tool_call_count:", len(tool_calls))
        print("tool_calls:", tool_calls)
        print("========================================\n")

        for i, call in enumerate(tool_calls, start=1):
            print(f"\n--- Tool Call {i} ---")
            print("name:", call.get("name"))
            print("args:")
            print(json.dumps(call.get("args", {}), indent=2, ensure_ascii=False))

        print(f"[TOTAL chat_model_node]: {sec(node_start)}s")
        print("========== CHAT MODEL NODE END ==========\n")

        return {
            "messages": [
                HumanMessage(content=user_query),
                response,
            ],
            "loop_count": loop_count + 1,
        }

    except Exception as e:
        print(f"[CHAT MODEL ERROR]: {e}")
        print(f"[TOTAL chat_model_node before error]: {sec(node_start)}s")

        return {
            "messages": [
                HumanMessage(content=state.get("user_query", "")),
                AIMessage(content=f"Chat model error: {str(e)}"),
            ],
            "loop_count": state.get("loop_count", 0) + 1,
        }
# ============================================
# ROUTING NODE
# ============================================
async def routing_node(state: MainState):
    """
    Routes to tools if the LLM requested tool calls.
    Otherwise ends the graph.
    """

    try:
        print("Routing node activated............")

        messages = state.get("messages", [])

        if not messages:
            return "__end__"

        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        loop_count = state.get("loop_count", 0)

        if loop_count > 5:
            return "__end__"

        print("No tool call is detected, ending the graph...")
        return "__end__"

    except Exception as e:
        print(f"Error in routing node: {e}")
        return "__end__"


# ============================================
# DETERMINISTIC FINAL NODE HELPERS
# ============================================
def parse_tool_output(content):
    """
    Converts ToolMessage content into Python dict.
    Tool output usually comes as JSON string.
    """

    try:
        if isinstance(content, dict):
            return content

        if isinstance(content, list):
            return {
                "success": True,
                "data": content,
                "count": len(content),
                "error": None,
            }

        return json.loads(content)

    except Exception as e:
        return {
            "success": False,
            "data": [],
            "count": 0,
            "error": f"Could not parse tool output: {str(e)}",
        }


def get_tool_name(tool_message, messages):
    """
    Gets tool name from ToolMessage.
    Fallback: match ToolMessage.tool_call_id with AIMessage.tool_calls.
    """

    tool_name = getattr(tool_message, "name", None)

    if tool_name:
        return tool_name

    tool_call_id = getattr(tool_message, "tool_call_id", None)

    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                if call.get("id") == tool_call_id:
                    return call.get("name")

    return "unknown_tool"


def make_summary(data: dict, errors: list, unsupported_parts: list | None = None) -> str:
    parts = []

    for tool_name, records in data.items():
        count = len(records) if isinstance(records, list) else 0

        if count == 0:
            parts.append(f"{tool_name}: no records found")
        elif count == 1:
            parts.append(f"{tool_name}: found 1 record")
        else:
            parts.append(f"{tool_name}: found {count} records")

    if errors:
        parts.append(f"{len(errors)} error(s)")

    if unsupported_parts:
        parts.append(f"{len(unsupported_parts)} unsupported part(s)")

    return "; ".join(parts)


def dedupe_records_by_field(records: list[dict], field: str) -> list[dict]:
    """
    Removes duplicate records based on one field.
    Example: billToName for customer/vendor name list queries.
    """
    if not isinstance(records, list):
        return records

    seen = set()
    deduped = []

    for record in records:
        if not isinstance(record, dict):
            continue

        value = record.get(field)

        if value is None or str(value).strip() == "":
            continue

        key = str(value).strip().lower()

        if key in seen:
            continue

        seen.add(key)
        deduped.append(record)

    return deduped


def wants_unique_party_names(query: str) -> bool:
    """
    Detects user intent like:
    - all customer names
    - all vendors
    - jinse kharidi ki hai un sab ka name
    - jo jo customers ko sell kia hai
    """
    q = (query or "").lower()

    has_party_word = any(word in q for word in PARTY_WORDS)
    has_name_word = any(word in q for word in NAME_WORDS)
    has_list_word = any(word in q for word in LIST_WORDS)

    return has_party_word and has_name_word and has_list_word


def apply_final_postprocessing(
    final_data: dict,
    original_query: str,
    canonical_query: str = "",
) -> dict:
    """
    Final deterministic cleanup after tools return data.
    This should not invent data.
    It only cleans/organizes existing tool results.
    """
    if not isinstance(final_data, dict):
        return final_data

    combined_query = f"{original_query or ''} {canonical_query or ''}".strip()

    return final_data
def infer_requested_fields(user_query: str, tool_name: str) -> list[str]:
    """
    Fallback projection only. Normal projection should come from tool args.
    Uses TOOL_INTENT_REGISTRY as single source of truth.
    """
    return infer_requested_fields_from_registry(user_query, tool_name)


def compact_transactions(records: list[dict]) -> list[dict]:
    """
    Keep ledger transactions useful but prevent huge nested item dumps.
    Preserves ALL transaction fields dynamically; only replaces `items` with itemCount.
    New API fields are automatically passed through.
    """
    compacted_records = []

    for record in records:
        if not isinstance(record, dict):
            continue

        new_record = dict(record)
        transactions = new_record.get("transactions")

        if isinstance(transactions, list):
            compacted_txns = []

            for txn in transactions:
                if not isinstance(txn, dict):
                    continue

                txn_copy = dict(txn)
                items = txn_copy.pop("items", [])
                txn_copy["itemCount"] = len(items) if isinstance(items, list) else 0
                compacted_txns.append(txn_copy)

            new_record["transactions"] = compacted_txns

        compacted_records.append(new_record)

    return compacted_records

def project_records_by_fields(records: list, fields: list[str]) -> list:
    if not fields:
        return records

    projected = []

    for record in records:
        if not isinstance(record, dict):
            continue

        row = {}
        for field in fields:
            if field in record:
                row[field] = record[field]

        if row:
            projected.append(row)

    return projected


def requested_gst_categories(query: str) -> list[str]:
    """
    Infer requested GST rows from the user query.
    Uses the GST category_map from TOOL_INTENT_REGISTRY — no hardcoded category names.
    """
    q = normalize_text(query)
    categories: list[str] = []

    def add(category: str):
        if category not in categories:
            categories.append(category)

    gst_meta = TOOL_INTENT_REGISTRY.get("get_gst_summary", {})
    cat_map = gst_meta.get("repair", {}).get("category_map", {})

    # Match each category_map keyword against the query
    for kw, val in cat_map.items():
        if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", q):
            add(val)

    # Generic prefix expansion: if a single-word keyword is a prefix of
    # multi-word keywords (e.g., "b2c" → "b2c small" + "b2c large"),
    # include all sub-variants when query matches only the parent word.
    # This mirrors the same logic in _apply_repair — no hardcoded category names.
    prefix_kws = {}
    for kw in cat_map:
        if " " in kw:
            first = kw.split(" ")[0]
            prefix_kws.setdefault(first, []).append(kw)
    for parent, children in prefix_kws.items():
        if parent not in cat_map:
            continue
        parent_val = cat_map[parent]
        if parent_val not in categories:
            continue
        has_specific = any(c in q for c in children)
        if not has_specific:
            for child in children:
                cv = cat_map[child]
                if cv not in categories:
                    categories.append(cv)

    # Additional keyword checks not in category_map
    if "export" in q or "exports" in q:
        add("exports")

    if "creditnotesregistered" in q or ("credit" in q and "registered" in q):
        add("creditNotesRegistered")

    if "creditnotesunregistered" in q or ("credit" in q and "unregistered" in q):
        add("creditNotesUnregistered")

    if "grand total" in q or "total gst" in q or "gst total" in q:
        add("grandTotal")

    return categories


def filter_gst_records_by_query(records: list[dict], query: str) -> list[dict]:
    """Filter GST rows deterministically based on category words in query."""
    requested = requested_gst_categories(query)

    if not requested:
        return records

    requested_set = set(requested)

    return [
        record for record in records
        if isinstance(record, dict) and record.get("category") in requested_set
    ]
# ============================================
# DETERMINISTIC FINAL NODE
# ============================================
@traceable(name="deterministic_final_node", run_type="chain")
async def deterministic_final_node(state: MainState):
    """
    Builds final JSON using Python, not LLM.
    This removes the second LLM call.
    """

    user_query = state.get("user_query", "")
    canonical_query = state.get("canonical_query", "")
    messages = state.get("messages", [])

    data = {}
    tools_used = []
    errors = []

    tool_messages = [
        msg for msg in messages
        if isinstance(msg, ToolMessage)
    ]
    # 🚀 DYNAMIC AMBIGUITY EVALUATOR (Zero Hardcoding)
    if not tool_messages:
        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if last_ai_msg:
            blocks = parse_planner_json_blocks(last_ai_msg.content)
            for block in blocks:
                if isinstance(block, dict) and block.get("status") == "needs_clarification":
                    return {
                        "final_response": {
                            "success": False,
                            "status": "needs_clarification",
                            "query": user_query,
                            "tools_used": [],
                            "data": {},
                            "summary": block.get("summary", "Please specify the customer name or customer id and date range."),
                            "errors": [],
                        },
                        "tools_utilized": [],
                    }
    for tool_msg in tool_messages:
        tool_name = get_tool_name(tool_msg, messages)

        if tool_name not in tools_used:
            tools_used.append(tool_name)

        parsed = parse_tool_output(tool_msg.content)

        if not parsed.get("success"):
            data.setdefault(tool_name, [])

            errors.append({
                "tool": tool_name,
                "error": parsed.get("error", "Unknown tool error"),
            })

            continue

        records = parsed.get("data", [])

        if records is None:
            records = []

        if not isinstance(records, list):
            records = [records]
        if tool_name == "get_gst_summary":
            records = filter_gst_records_by_query(
                records,
                f"{user_query or ''} {canonical_query or ''}",
            )

        if tool_name == "get_customer_ledger":
            records = compact_transactions(records)

        data.setdefault(tool_name, [])

        # Deduplicate records by id or full content
        existing_ids = {r.get("id") for r in data[tool_name] if isinstance(r, dict) and r.get("id") is not None}
        records = [r for r in records if not (isinstance(r, dict) and r.get("id") is not None and r["id"] in existing_ids)]
        data[tool_name].extend(records)

    # -------------------------------------------------
    # NEW: final deterministic cleanup
    # Example: dedupe customer/vendor names for list queries
    data = apply_final_postprocessing(
    data,
    user_query,
    canonical_query,
)

    unsupported_parts = state.get("unsupported_parts", [])

    has_any_data = any(
        isinstance(records, list) and len(records) > 0
        for records in data.values()
    )

    has_empty_requested_sections = any(
        isinstance(records, list) and len(records) == 0
        for records in data.values()
    )

    if unsupported_parts:
        status = "partial_success"
        success = bool(has_any_data)

    elif errors and has_any_data:
        status = "partial_success"
        success = True

    elif errors and not has_any_data:
        status = "error"
        success = False

    elif has_any_data and has_empty_requested_sections:
        status = "partial_success"
        success = True

    elif has_any_data:
        status = "success"
        success = True

    else:
        status = "no_matching_records"
        success = False

    final_response = {
        "success": success,
        "status": status,
        "query": user_query,
        "tools_used": tools_used,
        "data": data,
        "summary": make_summary(data, errors, unsupported_parts),
        "errors": errors,
    }

    if unsupported_parts:
        final_response["unsupported_parts"] = unsupported_parts

    return {
        "final_response": final_response,
        "tools_utilized": tools_used,
    }
# ============================================
# TOOL NODE
# ============================================
tools_node = ToolNode(tools)