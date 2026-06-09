from src.schema import MainState
from src.tools_api import tools_dict, tools
from src.tool_doc import TOOL_INTENT_REGISTRY, TOOL_NAME_ALIASES, get_field_triggers, infer_requested_fields_from_registry, CITY_WORDS
from collections import Counter
from src.config import llm, normalizer_llm, get_cfg,summary_llm
import time
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage,RemoveMessage
from langgraph.prebuilt import ToolNode
import json
import re
import uuid
from langsmith import traceable

import numpy as np
from src.config import embedding_model

# ── Pipeline config from config.yaml ──
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

# ── Domain pre-filtering constants ──
DOMAIN_KEYWORDS = {
    "sales": ["sales", "sale", "sell", "sold", "overdue", "receivable", "debtor", "billing"],
    "purchase": ["purchase", "kharidi", "buy", "bought", "payable", "creditor", "bills payable"],
    "customer": ["customer", "client", "party", "ledger", "customer name", "customer list", "customer code"],
    "vendor": ["vendor", "supplier", "vendor list"],
    "gst": ["gst", "gst summary", "gst detail", "gstr", "taxable", "igst", "cgst", "sgst", "b2b", "b2c"],
    "tax": ["tds", "tcs", "tax deducted", "tax collected", "tax outstanding"],
    "stock": ["stock", "inventory", "quantity", "hsn", "product", "slow moving", "stock level"],
    "analytics": ["top", "popular", "trend", "summary", "analytics", "report"],
}

INVOICE_PATTERNS = {
    "sales": [r"\bA/\d{4}/C\d{4}\b", r"\bSI-?\d+\b", r"\bOUT-?\d+\b"],
    "purchase": [r"\bPR-?\d+\b"],
}

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
    "get_overdue_invoices": ["sales"],
}


def classify_domains(query: str) -> tuple[set[str], set[str]]:
    query_lower = query.lower()
    hard = set()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            hard.add(domain)
    soft = set()
    for domain, patterns in INVOICE_PATTERNS.items():
        if any(re.search(p, query, re.IGNORECASE) for p in patterns):
            soft.add(domain)
    return hard, soft


def _filter_registry_by_domain(domains: set[str]) -> dict:
    if not domains:
        return TOOL_INTENT_REGISTRY
    filtered = {}
    for tname, meta in TOOL_INTENT_REGISTRY.items():
        td = TOOL_DOMAINS.get(tname, [])
        if not td or any(d in domains for d in td):
            filtered[tname] = meta
    return filtered


# ── Pronoun resolution ──
HINGLISH_PRONOUNS = ["uska", "iska", "unka", "iski", "inki", "uski", "woh", "uss", "in sab"]


def _resolve_pronouns(query: str, conv_ctx: dict | None, last_tool: dict | None) -> tuple[str, list[dict]]:
    q_lower = query.lower()
    found = [p for p in HINGLISH_PRONOUNS if re.search(rf'\b{re.escape(p)}\b', q_lower)]
    if not found:
        return query, []

    entities = (conv_ctx or {}).get("entities", [])
    if entities:
        name = entities[-1].get("name", "")
        if name:
            resolved = query
            for p in found:
                resolved = re.sub(rf'\b{re.escape(p)}\b', name, resolved, flags=re.IGNORECASE)
            return resolved, [{"original": p, "resolved": name, "type": "any"} for p in found]

    if last_tool:
        for tname, targs in last_tool.items():
            filters = targs.get("filters", {}) if isinstance(targs, dict) else {}
            for key in ("invoiceNo", "invoiceNumber", "invoice_number"):
                val = filters.get(key) or (targs.get(key) if isinstance(targs, dict) else None)
                if val:
                    resolved = query
                    for p in found:
                        resolved = re.sub(rf'\b{re.escape(p)}\b', str(val), resolved, flags=re.IGNORECASE)
                    return resolved, [{"original": p, "resolved": str(val), "type": "invoice"} for p in found]

    return query, []


def _is_specific_lookup(query: str) -> bool:
    """Detects if the user is asking about a specific record (not a broad list query)."""
    q = (query or "").lower()
    if re.search(r'\b(pr|lr|si|out)[\s-]?\d+\b', q, re.IGNORECASE):
        return True
    if re.search(r'\ba/\d{4}/c\d{4}\b', q, re.IGNORECASE):
        return True
    if any(w in q for w in ["for", "of ", "detail of", "detail for", "specific", "particular"]):
        if re.search(r'\b(pr|invoice|bill|customer|party|vendor|product|hsn)\b', q):
            return True
    return False


# ── Embedding recall + cross-encoder reranker for tool routing ──
_tool_embeddings: dict[str, list[float]] = {}
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
            fields = meta.get("fields", [])
            falias = meta.get("field_aliases", {})
            for fname in fields:
                text_parts.append(fname)
                if fname in falias:
                    text_parts.extend(falias[fname][:2])
            text = " ".join(str(p) for p in text_parts if p)
            _tool_embeddings[tool_name] = embedding_model.embed_query(text)
    except Exception as e:
        print(f"[WARN] Failed to build tool embeddings: {e}")
        _tool_embeddings.clear()

def now():
    return time.perf_counter()

TRANSLATOR_PROMPT_BASE = """Normalize Hinglish/Hindi/Gujarati → clean English JSON.

SCHEMA: {"canonical_query":"...","document_type":"sales_invoice|purchase_invoice|customer|product|general","language":"...","confidence":"high|medium|low","query_type":"erp_query|conversational|mixed","query_parts":["..."],"resolved_entities":[{"original":"...","resolved":"...","type":"..."}]}

WORD MAP: bill=sales_invoice, bikri=sales, kharidi=purchase, grahak=customer, rakam=amount, baki=outstanding, kam=less, zyada=greater, dikhao/batao=show, aur=and, kitne/kitna=how_many/much, hai/ho=is_are, kya=what, konse/konsa/jiska=which, kyu=why, chaia/chahiye=need, nahi=not, hamare/mera/uska/uski=our/my/his, wala/wale=with, sari/saari=all

RULES:
- query_type: "conversational" if asking about conversation history (what we discussed, what was asked, recap, etc.), "erp_query" if asking about ERP data (customers/stock/GST/invoices), "mixed" if asking about both history AND data.
- Preserve IDs/HSN/dates/names. Clean English → language="english", query unchanged. Bare number/name → treat as lookup.

EXAMPLES:
Q: A/0326/C0077 sales bill ka customer name batao
A: {"canonical_query":"Show customer name for sales invoice A/0326/C0077","document_type":"sales_invoice","language":"hinglish","confidence":"high","query_type":"erp_query"}
Q: kitne products hai inventory mai
A: {"canonical_query":"How many products in inventory","document_type":"product","language":"hinglish","confidence":"high","query_type":"erp_query"}
Q: kyu nahi mila
A: {"canonical_query":"Why no results found","document_type":"general","language":"hinglish","confidence":"high","query_type":"erp_query"}
Q: hamne sabse pehle kya pucha tha
A: {"canonical_query":"What was asked first by us","document_type":"general","language":"hinglish","confidence":"high","query_type":"conversational"}
/no_think"""


def _build_translator_prompt(
    conversation_context: dict | None = None,
    last_tool_call: dict | None = None,
    summary: str | None = None,
) -> str:
    lines = [TRANSLATOR_PROMPT_BASE.rstrip()]
    ctx_lines = []
    if summary:
        ctx_lines.append(f"- Conversation summary: {summary}")
    if last_tool_call:
        for tool_name, args in last_tool_call.items():
            safe_args = {k: v for k, v in args.items() if k in ("filters", "search", "term", "invoiceNo")}
            ctx_lines.append(f"- Last tool: {tool_name} with {json.dumps(safe_args)}")
    if conversation_context:
        entities = conversation_context.get("entities", [])
        names = [e.get("name", "") for e in entities if e.get("name")]
        if names:
            ctx_lines.append(f"- Known entities: {', '.join(names[-5:])}")
        failures = conversation_context.get("tool_failures", [])
        for f in failures[-3:]:
            ctx_lines.append(f"- {f.get('tool')} returned no data for '{f.get('entity')}'")
    if ctx_lines:
        lines.append("")
        lines.append("CONVERSATION CONTEXT (resolve pronouns using this):")
        lines.extend(ctx_lines)
        lines.append("")
        lines.append("PRONOUN RESOLUTION RULES (CRITICAL):")
        lines.append("- If the current query has pronouns (uska, iska, unka, iski, inki, uska, woh, uss, is, es, in sab, its, this, that, these, those), replace them with the actual entity name from the CONTEXT above.")
        lines.append("- If the query has multiple independent intents, output each as a separate item in query_parts[].")
        lines.append("- Example: context has PR-269, query='to uska taxable amount kitna hai? aur april ka gst'")
        lines.append("  → query_parts: ['Show taxable amount of PR-269', 'Show GST details for April']")
    return "\n".join(lines)
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


META_QUESTION_PATTERNS_GLOBAL = [
    r"what (have|did|was|were|is|are) we (discussed?|talked?|said?|done|covered|asked)",
    r"what (have|did|was|were|is|are) (i|you|we|the) (discussed?|talked?|said?|done|covered|asked).*\b(first|previous|last|pichl|pehle)",
    r"which (products|items|customers) (have|were) (discussed|talked|mentioned)",
    r"what (was|were) (discussed|talked|mentioned|said)",
    r"(summarize|summary|recap) (the |our |this )?(conversation|chat|discussion)",
    r"conversation (history|so far|till now)",
    r"kya (baat|discuss|hua|kaha)",
    r"humne kya (baat|discuss|kiya|kaha|kari)",
    r"aur\s+usse?\s+pehle",
    r"(es?|is|us)\s+se?\s+pehle",
    r"(kiska|kiski|kiske)\s+(id|name|number|details|baat)\s+manga",
    r"(baat|bat)\s+(hua|hui|kiya|kia|kari|karke?\b)",
    r"(pichl[ei])\s+(baat|baar|query|sawal|question)",
    r"\b(shayad|thana|thahi)\b",
    r"(maine|hamne|humne)\s+(sabse\s+)?(pehle|pahle)\s+kya\s+(pucha|kaha|manga|poocha)",
    r"what\s+(was|were|did|have)\s+.*?\b(first|pehle|pahle|previous)\b",
    r"(usse?|is|es)\s+bhi\s+pehle",
    r"(first|pehle|pahle)\s+(query|question|sawal|baat)",
    r"(sabse\s+)?(pehle|pahle)\s+(kya\s+)?(pucha|kaha|manga|question|query)",
]


def _classify_query_type(query: str) -> str:
    if not query:
        return "unknown"
    if any(re.search(p, query, re.IGNORECASE) for p in META_QUESTION_PATTERNS_GLOBAL):
        return "conversational"
    return "unknown"


_INVOICE_DOC_MAP = {
    "purchase": "purchase_invoice",
    "sales": "sales_invoice",
}


def _override_document_type(original: str, canonical: str, doc_type: str) -> str:
    combined = f"{original or ''} {canonical or ''}"
    for domain, patterns in INVOICE_PATTERNS.items():
        if any(re.search(p, combined, re.IGNORECASE) for p in patterns):
            mapped = _INVOICE_DOC_MAP.get(domain)
            if mapped and mapped != doc_type:
                print(f"[OVERRIDE] document_type: {doc_type} -> {mapped} (matched {domain} pattern)")
                return mapped
    return doc_type


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
                "query_type": "unknown",
            }

        if is_plain_english_query(user_query):
            print("Translator skipped: query looks English")
            doc_type = _override_document_type(user_query, user_query, "routeable")
            return {
                "original_query": user_query,
                "canonical_query": user_query,
                "user_query": user_query,
                "translator_used": False,
                "translator_confidence": "skipped_english",
                "detected_language": "english",
                "document_type": doc_type,
                "query_type": _classify_query_type(user_query),
                "query_parts": [user_query],
                "resolved_entities": [],
            }

        if needs_translation(user_query):
            print("Translator: query needs Hinglish/Hindi/Gujarati normalization")

            # Pronoun pre-processing (deterministic, before LLM)
            resolved_query, pre_resolved_entities = _resolve_pronouns(
                user_query,
                state.get("conversation_context"),
                state.get("last_tool_call"),
            )
            if pre_resolved_entities:
                print(f"[PRONOUN] Resolved pronouns: {pre_resolved_entities}")
                print(f"[PRONOUN] Original: {user_query} → Resolved: {resolved_query}")
                user_query = resolved_query

            ctx = state.get("conversation_context") or {}
            ltc = state.get("last_tool_call") or {}
            summary = state.get("summary") or ""
            prompt = _build_translator_prompt(
                conversation_context=ctx,
                last_tool_call=ltc,
                summary=summary,
            )
            response = await normalizer_llm.ainvoke([
                SystemMessage(content=prompt),
                HumanMessage(content=user_query),
            ])
            log_token_usage(response, "translator")

            data = extract_json_object(response.content)

            canonical_query = data.get("canonical_query") or user_query
            language = data.get("language", "mixed")
            confidence = data.get("confidence", "medium")
            query_type = data.get("query_type", "")
            query_parts = data.get("query_parts") or []
            llm_resolved = data.get("resolved_entities") or []
            resolved_entities = (pre_resolved_entities or []) + (llm_resolved or [])

            print("Original query:", user_query)
            print("Canonical query:", canonical_query)
            print("Detected language:", language)
            print("Translator confidence:", confidence)
            print("Query type:", query_type)
            if query_parts:
                print("Query parts:", query_parts)
            if resolved_entities:
                print("Resolved entities:", resolved_entities)

            # For conversational queries, the canonical_query may hallucinate
            # entity names. Keep the original query instead.
            final_canonical = user_query if query_type == "conversational" else canonical_query
            doc_type = _override_document_type(user_query, final_canonical, data.get("document_type", "unknown"))
            return {
                "original_query": user_query,
                "canonical_query": final_canonical,
                "user_query": final_canonical,
                "translator_used": True,
                "translator_confidence": confidence,
                "detected_language": language,
                "document_type": doc_type,
                "query_type": query_type,
                "query_parts": query_parts,
                "resolved_entities": resolved_entities,
            }

        if is_routeable_without_translator(user_query):
            print("Translator skipped: query is directly routeable by ERP keywords")
            doc_type = _override_document_type(user_query, user_query, "routeable")
            return {
                "original_query": user_query,
                "canonical_query": user_query,
                "user_query": user_query,
                "translator_used": False,
                "translator_confidence": "skipped_routeable",
                "detected_language": "mixed_or_english",
                "document_type": doc_type,
                "query_type": _classify_query_type(user_query),
                "query_parts": [user_query],
                "resolved_entities": [],
            }

        print("Translator skipped: no multilingual normalization needed")
        doc_type = _override_document_type(user_query, user_query, "unknown")
        return {
            "original_query": user_query,
            "canonical_query": user_query,
            "user_query": user_query,
            "translator_used": False,
            "translator_confidence": "skipped_no_normalization_needed",
            "detected_language": "english_or_mixed",
            "document_type": doc_type,
            "query_type": _classify_query_type(user_query),
            "query_parts": [user_query],
            "resolved_entities": [],
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
            "query_type": "unknown",
        }

def score_tools_via_reranker(query_part: str, registry: dict) -> list[str]:
    """Hybrid: embedding recall (dense) + Jaccard word overlap (sparse)."""
    if not _tool_embeddings:
        _build_tool_embeddings()
    if not _tool_embeddings:
        return []

    try:
        query_emb = embedding_model.embed_query(query_part)
    except Exception:
        return []

    query_tokens = set(query_part.lower().split())

    scores = []
    for tool_name, tool_emb in _tool_embeddings.items():
        emb_sim = _cosine_sim(query_emb, tool_emb)

        meta = registry.get(tool_name, {})
        tool_text = f"{meta.get('description', '')} {' '.join(meta.get('aliases', []))} {' '.join(meta.get('keywords', []))}"
        tool_tokens = set(tool_text.lower().split())
        containment = len(query_tokens & tool_tokens) / max(len(query_tokens), 1) if tool_tokens else 0.0

        combined = 0.7 * emb_sim + 0.3 * containment
        scores.append((tool_name, combined, emb_sim, containment))

    scores.sort(key=lambda x: x[1], reverse=True)

    if not scores:
        return []

    top_k = scores[:TH_RERANKER_TOP_K]
    result = []
    for tool_name, combined, emb_sim, containment in top_k:
        if tool_name not in result and (emb_sim >= TH_EMBEDDING_RECALL_MIN or containment >= 0.1):
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


def _keyword_fallback(part: str, registry: dict | None = None) -> list[str]:
    """Fallback: match registry keywords/aliases against the query part.
    Uses word-boundary matching to prevent false positives
    (e.g. "credit" inside "creditNotesUnregistered")."""
    if registry is None:
        registry = TOOL_INTENT_REGISTRY
    q = normalize_text(part)
    matched = []
    for tool_name, meta in registry.items():
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
    freq: Counter = Counter()
    seen_order: dict[str, int] = {}
    rank = 0
    for group in tool_lists:
        for t in group:
            freq[t] += 1
            if t not in seen_order:
                seen_order[t] = rank
                rank += 1

    return sorted(freq.keys(), key=lambda t: (-freq[t], seen_order[t]))


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

        pre_resolved = state.get("query_parts") or []
        has_entities = bool(state.get("resolved_entities"))
        translator_used = state.get("translator_used", False)
        use_pre_resolved = translator_used or has_entities or len(pre_resolved) > 1

        if use_pre_resolved and pre_resolved:
            query_parts = pre_resolved
        else:
            query_parts = split_query_for_tools(
                original_query=original_query,
                canonical_query=canonical_query,
            )

        if not query_parts:
            query_parts = [user_query]

        print(f"Query parts for metadata matching: {query_parts}")

        query_type = (state.get("query_type") or "").strip()
        if query_type == "conversational":
            print(f"Translator flagged as conversational — no tool needed: {user_query}")
            return {
                "retrieved_tools": [],
                "selected_tools": [],
                "query_parts": query_parts,
                "skip_router": True,
            }

        full_meta = any(
            any(re.search(p, q, re.IGNORECASE) for p in META_QUESTION_PATTERNS_GLOBAL)
            for q in [original_query, canonical_query] if q
        )
        is_pure_meta = full_meta or (
            len(query_parts) > 0 and all(
                any(re.search(p, part, re.IGNORECASE) for p in META_QUESTION_PATTERNS_GLOBAL)
                for part in query_parts
            )
        )
        if is_pure_meta:
            print(f"Meta-question detected — no tool needed: {user_query}")
            return {
                "retrieved_tools": [],
                "selected_tools": [],
                "query_parts": query_parts,
                "skip_router": True,
            }

        GREETING_PATTERNS = [
            r"^(hello|hi|hey|hii|hiii|heyy|holla|namaste|namaskar|vanakkam|howdy|greetings|salam)\\s*[!?.]*$",
            r"^(good\\s*morning|good\\s*afternoon|good\\s*evening|good\\s*night|gm|gn)\\s*[!?.]*$",
            r"^(hey\\s+there|hi\\s+there|hello\\s+there)\\s*[!?.]*$",
            r"^(how\\s+are\\s+(you|u)|how\\s+are\\s+you\\s+doing|how\\'s\\s+it\\s+going|what\\'s\\s+up|wassup|sup)\\s*[!?.]*$",
            r"^(kaise\\s+ho|kya\\s+haal|kya\\s+kar\\s+rahe|kya\\s+kar\\s+raha|kya\\s+kar\\s+rahi)\\s*[!?.]*$",
        ]
        full_query = original_query.strip().lower()
        is_greeting = any(re.match(p, full_query) for p in GREETING_PATTERNS)
        if is_greeting:
            has_erp_keywords = any(kw in full_query for kw in ROUTE_KEYWORDS)
            if not has_erp_keywords:
                print(f"Greeting detected — responding with welcome message: {user_query}")
                return {
                    "retrieved_tools": [],
                    "selected_tools": [],
                    "query_parts": query_parts,
                    "skip_router": True,
                    "memory_answer": "Hello! I am your Chapter1 ERP assistant...",
                }

        selected_tool_groups: list[list[str]] = []

        for part in query_parts:
            hard_domains, soft_domains = classify_domains(part)
            if hard_domains:
                filtered_registry = _filter_registry_by_domain(hard_domains)
                print(f"Part '{part}': hard domains {hard_domains}, using filtered registry ({len(filtered_registry)} tools)")
            else:
                filtered_registry = TOOL_INTENT_REGISTRY
                if soft_domains:
                    print(f"Part '{part}': no hard domains, soft domains {soft_domains}, using full registry")
                else:
                    print(f"Part '{part}': no domains matched, using full registry")

            tools_for_part = score_tools_via_reranker(part, filtered_registry)
            if tools_for_part:
                print(f"Reranker tools for part '{part}': {tools_for_part}")
                selected_tool_groups.append(tools_for_part)
                continue

            kw_tools = _keyword_fallback(part, filtered_registry)
            if kw_tools:
                print(f"Keyword fallback tools for part '{part}': {kw_tools}")
                selected_tool_groups.append(kw_tools)
            else:
                print(f"No tools found for part '{part}' via reranker or keyword fallback")

        if document_type in {"product", "inventory", "stock"}:
            selected_tool_groups.append(["get_stock_levels"])
        elif document_type in {"customer", "party"}:
            selected_tool_groups.append(["get_customer"])
        elif document_type in {"customer_ledger", "ledger"}:
            selected_tool_groups.append(["get_customer_ledger"])
        elif document_type in {"purchase_invoice", "purchase"}:
            selected_tool_groups.append(["get_outstanding_purchase_invoices", "get_purchase_summary"])
        elif document_type in {"sales_invoice", "sales"}:
            selected_tool_groups.append(["get_outstanding_sales_invoices", "get_overdue_invoices", "get_sales_summary"])
        elif document_type in {"gst", "gst_report", "gst_summary"}:
            selected_tool_groups.append(["get_gst_summary"])

        selected_tools = merge_unique_tools(selected_tool_groups)
        selected_tools = [t for t in selected_tools if t in tools_dict]
        MAX_TOOLS_FOR_LLM = 8
        if len(selected_tools) > MAX_TOOLS_FOR_LLM:
            print(f"Trimming selected_tools from {len(selected_tools)} to {MAX_TOOLS_FOR_LLM}")
            preserved_order = []
            seen = set()
            for group in selected_tool_groups:
                for t in group:
                    if t in selected_tools and t not in seen:
                        preserved_order.append(t)
                        seen.add(t)
                        break
            remaining = [t for t in selected_tools if t not in seen]
            selected_tools = (preserved_order + remaining)[:MAX_TOOLS_FOR_LLM]
            print(f"Trimmed selected_tools: {selected_tools}")

        if not selected_tools:
            has_erp_kw = any(kw in (original_query or "").lower() for kw in ROUTE_KEYWORDS)
            if has_erp_kw or document_type:
                fallback_tools = []
                for q in [original_query, canonical_query]:
                    if q:
                        fallback_tools = _keyword_fallback(q)
                        if fallback_tools:
                            break
                if not fallback_tools:
                    full_query = canonical_query or original_query
                    if full_query:
                        fallback_tools = score_tools_via_reranker(full_query, TOOL_INTENT_REGISTRY)
                    if not fallback_tools:
                        fallback_tools = list(tools_dict.keys())[:min(8, len(tools_dict))]
                selected_tools = fallback_tools
                print(f"Fallback selected tools: {selected_tools}")

        combined = f"{original_query or ''} {canonical_query or ''}"
        if re.search(r'[A-Z]+/\d{2}-\d{2}/\d{3}', combined):
            selected_tools = [t for t in selected_tools if t not in ('get_customer',)]

        # Exclude get_customer_ledger when query contains an invoice pattern
        if combined and any(re.search(p, combined, re.IGNORECASE) for pats in INVOICE_PATTERNS.values() for p in pats):
            selected_tools = [t for t in selected_tools if t != 'get_customer_ledger']

        if selected_tools:
            print(f"Final selected tools: {selected_tools}")
            return {
                "retrieved_tools": selected_tools,
                "selected_tools": selected_tools,
                "query_parts": query_parts,
                "skip_router": True,
            }

        OOD_PATTERNS = [
            r"^(who|what|why|when|where|how)\\s+(is|are|was|were|does|do|did|can|could|will|would|shall|should)\\s+",
            r"(tell me about|explain|describe|define)\\s",
        ]
        raw_queries = [q for q in [original_query, canonical_query] if q]
        has_erp_kw = any(kw in (original_query or "").lower() for kw in ROUTE_KEYWORDS)
        is_ood = (not has_erp_kw and any(
            any(re.search(p, q.strip().lower()) for p in OOD_PATTERNS) for q in raw_queries
        ))
        if is_ood:
            print(f"Out-of-domain question detected: {user_query}")
        else:
            messages = state.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        tool_name = tc.get("name")
                        if tool_name and tool_name in tools_dict:
                            print(f"Using tool from conversation history: {tool_name}")
                            selected_tools = [tool_name]
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
            "unsupported_reason": "I am an ERP assistant...",
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

def _get_recent_tool_calls(messages: list, max_calls: int = 3) -> list[dict]:
    """Return the most recent distinct tool calls from AIMessages, newest first."""
    calls = []
    seen = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                name = tc.get("name")
                args = tc.get("args", {})
                if name and args and name not in seen:
                    calls.append({"name": name, "args": args})
                    seen.add(name)
                    if len(calls) >= max_calls:
                        return calls
    return calls


def _summarize_tool_result(content: str) -> str:
    """Convert tool result JSON to a brief human-readable summary."""
    try:
        parsed = json.loads(content) if isinstance(content, str) else content
        if not isinstance(parsed, dict):
            return "Data found"
        if not parsed.get("success", True):
            return f"Error: {parsed.get('error', 'Unknown')}"
        records = parsed.get("data", [])
        if not isinstance(records, list):
            records = [records] if records else []
        if not records:
            return "No results"
        names = []
        for r in records[:3]:
            if isinstance(r, dict):
                n = r.get("name") or r.get("productName") or r.get("customerName") or r.get("partyName") or ""
                if n:
                    names.append(str(n))
        count = len(records)
        if names:
            name_list = ", ".join(names)
            return f"Found {count}: {name_list}" + ("..." if count > 3 else "")
        return f"Found {count} records"
    except (json.JSONDecodeError, TypeError):
        return "Data found"


def _build_memory_context(messages: list, max_exchanges: int = 3) -> str:
    """Build a readable conversation narrative from past exchanges.
    
    Walks messages in reverse, pairing each AIMessage (with tool_calls)
    with its preceding HumanMessage and following ToolMessages.
    Returns a natural-language string for the memory LLM prompt.
    """
    exchanges = []
    i = len(messages) - 1

    while i >= 0 and len(exchanges) < max_exchanges:
        msg = messages[i]
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            # Find the preceding HumanMessage
            user_query = ""
            for k in range(i - 1, -1, -1):
                if isinstance(messages[k], HumanMessage):
                    user_query = getattr(messages[k], "content", "") or ""
                    break

            # Collect tool call IDs for this exchange
            tc_ids = {tc["id"] for tc in msg.tool_calls if tc.get("id")}

            # Find corresponding ToolMessages (follow the AIMessage)
            result_summaries = []
            for j in range(i + 1, len(messages)):
                tm = messages[j]
                if isinstance(tm, ToolMessage) and getattr(tm, "tool_call_id", None) in tc_ids:
                    result_summaries.append(_summarize_tool_result(tm.content))
                elif isinstance(tm, AIMessage) and getattr(tm, "tool_calls", None):
                    break

            # Format tool calls in this exchange
            parts = []
            for tc in msg.tool_calls:
                name = tc.get("name", "?")
                args = tc.get("args", {})
                nice_args = ", ".join(f"{k}={v}" for k, v in args.items() if k != "fields")
                parts.append(f"{name}({nice_args})")
            tool_desc = "; ".join(parts)

            line = f'  Asked: "{user_query}" → {tool_desc}'
            if result_summaries:
                results_str = "; ".join(result_summaries)
                if len(results_str) > 250:
                    results_str = results_str[:250] + "..."
                line += f" → {results_str}"
            exchanges.append(line)

        elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = getattr(msg, "content", "") or ""
            if content.strip():
                user_query = ""
                for k in range(i - 1, -1, -1):
                    if isinstance(messages[k], HumanMessage):
                        user_query = getattr(messages[k], "content", "") or ""
                        break
                if user_query:
                    exchanges.append(f'  Asked: "{user_query}" → Answered: {content[:250]}')

        i -= 1

    exchanges.reverse()
    if exchanges:
        return "Earlier in this conversation:\n" + "\n".join(exchanges) + "\n"
    return ""


def build_system_prompt(
    user_query: str,
    selected_tools: list[str],
    query_parts: list[str] | None = None,
    summary: str | None = None,
    messages: list | None = None,
    last_tool_call: dict | None = None,
    conversation_context: dict | None = None,
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
        "",
        "FOLLOW-UP RULES:",
        "- Reuse the search term and filters only if the new query is a follow-up about the same specific entity. If the topic or scope changes (e.g. switching to a different search or asking a general/broad question), clear them.",
        "- When the user asks for additional fields (e.g. 'id' after previously asking for 'name'),",
        "  KEEP the previous fields and ADD the new ones. Never remove previously requested fields.",
        "- If the answer is already in previous tool results, use it directly without a new API call.",
        "- When a tool has sort_field/sort_order parameters and the user asks for extreme/comparative values (highest, most, least, top, bottom, etc.), ALWAYS set sort_field to the field being compared and sort_order accordingly: 'desc' for highest/most/top, 'asc' for lowest/least/bottom.",
        "- CRITICAL: When the current query requires a DIFFERENT tool than the previous one (e.g. switching from get_customer to get_stock_levels), you MUST clear ALL old search terms, filters, and parameters. Reuse parameters ONLY within the same tool.",
    ]

    if messages:
        recent_calls = _get_recent_tool_calls(messages)
        if recent_calls:
            lines.append("")
            lines.append("--- RECENT TOOL CALLS (for follow-up context) ---")
            for call in recent_calls:
                lines.append(f"Tool: {call['name']}")
                lines.append(f"Args: {json.dumps(call['args'], indent=2)}")
            lines.append("For follow-up queries, reuse these same parameters. Only change what the user explicitly asks about.")
            lines.append("--------------------------------------------------")
        if summary:
            lines.append("")
            lines.append("--- PREVIOUS CONVERSATION CONTEXT ---")
            lines.append(summary)
            lines.append("--------------------------------------------------")
        if conversation_context:
            entities = conversation_context.get("entities", [])
            if entities:
                lines.append("")
                lines.append("--- KNOWN ENTITIES ---")
                names = []
                for e in entities:
                    name = e.get("name", "")
                    id_ = e.get("id")
                    if id_ is not None:
                        names.append(f"{name} (ID {id_})")
                    elif name:
                        names.append(name)
                lines.append("Previously mentioned: " + ", ".join(names))
                lines.append("--------------------------------------------------")
        # Inject last_tool_call for tools not already covered by recent_calls
        if last_tool_call:
            recent_names = {c["name"] for c in recent_calls} if recent_calls else set()
            extra = {k: v for k, v in last_tool_call.items() if k not in recent_names}
            if extra:
                lines.append("")
                lines.append("--- PREVIOUS TOOL CALLS (from earlier in conversation) ---")
                lines.append(json.dumps(extra, indent=2))
                lines.append("For follow-up queries, reuse these same parameters. Only change what the user explicitly asks about.")
                lines.append("--------------------------------------------------")
    if query_parts and len(query_parts) > 1:
        lines.append("")
        lines.append("--- MULTI-INTENT QUERY ---")
        lines.append(f"The user's query has {len(query_parts)} separate intents:")
        for i, part in enumerate(query_parts, 1):
            lines.append(f"  {i}. {part}")
        lines.append("You MUST call a separate tool for each distinct intent. Do NOT combine different intents into one tool call.")
        lines.append("Use the same parameters for follow-up parts, but call a new tool for each independent sub-query.")
        lines.append("--------------------------")
    if selected_tools:
        lines.append("")
        lines.append(f"Available tools: {', '.join(selected_tools)}")
        lines.append("Call the tool(s) that are relevant to the query. You do NOT need to call every tool — only those that actually address the user's request.")
        lines.append("")
        lines.append("Tool rules:")
        lines.append("  You may call the SAME tool MULTIPLE TIMES with different sort/filter arguments for different sub-requests.")
        for tool_name in selected_tools:
            meta = TOOL_INTENT_REGISTRY.get(tool_name)
            if meta and meta.get("prompt_tips"):
                lines.append(f"  {tool_name}: {meta['prompt_tips']}")
    lines.append("")
    lines.append("FAILURE-AWARE RESPONSE:")
    lines.append("  If a tool call returns empty results, do NOT hallucinate data. Report clearly: 'No records found for X.'")
    lines.append("  If the user asks about something from earlier that returned no data, acknowledge the prior failure.")
    lines.append("")
    lines.append("PARAMETER RULES:")
    lines.append("  1. Tools with `search`/`term`: put name/city/product lookups there, NEVER in `filters`.")
    lines.append("  2. `filters` is for exact matches only: hsnCode, category, lowStockOnly, etc.")
    lines.append("  3. Tools without `search`/`term` (gst_summary, tds, tcs): `filters` is correct usage.")
    lines.append("  4. CRITICAL — NEVER copy parameters between different tools. Each tool has its own unique set of valid parameters. What works for one tool will NOT work for another.")

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
    prompt_tokens = tu.get("prompt_tokens") if tu.get("prompt_tokens") is not None else meta.get("prompt_eval_count", 0)
    output_tokens = tu.get("completion_tokens") if tu.get("completion_tokens") is not None else meta.get("eval_count", 0)
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


def sanitize_tool_filters(name: str, args: dict) -> dict:
    """Strip invalid filter keys from any tool call and move values to search/term param.
    Uses TOOL_INTENT_REGISTRY fields as the set of valid filter keys.
    Generic across all tools — zero config needed."""
    meta = TOOL_INTENT_REGISTRY.get(name)
    if not meta:
        return args

    filters = args.get("filters")
    if not filters:
        return args

    valid_keys = set(meta.get("fields", []))
    if not valid_keys:
        return args

    # ── Remap LLM-invented filter keys to real API fields using field_aliases ──
    field_aliases = meta.get("field_aliases", {})
    alias_to_real = {}
    for real_field, aliases in field_aliases.items():
        for alias in aliases:
            alias_to_real[alias] = real_field
    for k in list(filters.keys()):
        if k not in valid_keys and k in alias_to_real:
            filters[alias_to_real[k]] = filters.pop(k)

    invalid = {}
    for k in list(filters.keys()):
        if k not in valid_keys:
            invalid[k] = filters.pop(k)

    if not invalid:
        return args

    # ── Entity salvage: inject stripped values into name-like filters ──
    remaining = {}
    for k, v in invalid.items():
        if isinstance(v, str) and v.strip():
            name_candidates = [
                rf for rf, aliases in field_aliases.items()
                if any(
                    any(n in a.lower() for n in ("name", "customer", "vendor", "ledger", "party"))
                    for a in aliases
                )
            ]
            if name_candidates:
                target = name_candidates[0]
                if isinstance(filters, dict):
                    filters.setdefault(target, {})
                    if isinstance(filters[target], dict) and "contains" not in filters[target]:
                        filters[target]["contains"] = v
                        print(f"[SANITIZE] {name}: salvaged '{v}' -> filters.{target}.contains")
                        continue
        remaining[k] = v

    if not filters:
        del args["filters"]

    terms = []
    for k, v in remaining.items():
        if isinstance(v, str):
            terms.append(v)
        elif isinstance(v, (list, tuple)):
            terms.extend(str(t) for t in v)

    if terms:
        term_str = " ".join(terms)
        for search_key in ("search", "term"):
            if search_key in args:
                current = args.get(search_key) or ""
                if current:
                    current += " "
                args[search_key] = current + term_str
                break

    if remaining:
        print(f"[SANITIZE] {name}: stripped invalid filter keys {list(remaining.keys())}")
    if terms:
        print(f"[SANITIZE] {name}: moved values to search: {' '.join(terms)}")
    return args


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
        summary = state.get("summary", "")
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
            existing_memory_answer = state.get("memory_answer", "")
            if existing_memory_answer:
                print(f"[CHAT MODEL] Using pre-set memory_answer: {existing_memory_answer}")
                return {
                    "messages": [
                        HumanMessage(content=user_query),
                        AIMessage(content=existing_memory_answer),
                    ],
                    "memory_answer": existing_memory_answer,
                    "loop_count": loop_count + 1,
                }

            unsupported_reason = state.get("unsupported_reason")
            if unsupported_reason:
                reason = unsupported_reason
                print(f"[CHAT MODEL] Query unsupported, using fallback: {reason}")
                return {
                    "messages": [
                        HumanMessage(content=user_query),
                        AIMessage(content=reason),
                    ],
                    "memory_answer": reason,
                    "loop_count": loop_count + 1,
                }

            print("[CHAT MODEL] No available tools. Trying conversation memory...")
            previous_summary = state.get("summary", "") or ""
            conversation_context = state.get("conversation_context", {})
            has_messages = bool(state.get("messages"))
            if previous_summary or conversation_context or has_messages:
                mem_prompt = (
                    "You are an ERP assistant. Answer based ONLY on the conversation history below. "
                    "Do not make up information. If the answer is not in the history, say so plainly. "
                    "Reply in natural Hinglish (Hindi+English) like the user. NEVER mention tool names, API calls, or technical details.\n\n"
                )
                if previous_summary:
                    mem_prompt += f"Conversation History:\n{previous_summary}\n\n"
                elif has_messages:
                    narrative = _build_memory_context(state.get("messages", []), max_exchanges=5)
                    if narrative:
                        mem_prompt += narrative
                
                if conversation_context:
                    entities = conversation_context.get("entities", [])
                    if entities:
                        mem_prompt += f"Known Entities:\n{json.dumps(entities, indent=2, ensure_ascii=False)}\n"
                mem_prompt += "\n/no_think"
                try:
                    mem_resp = await summary_llm.ainvoke([
                        SystemMessage(content=mem_prompt),
                        HumanMessage(content=user_query),
                    ])
                    reason = (getattr(mem_resp, "content", "") or "").strip()
                except Exception as e:
                    print(f"[CHAT MODEL] Memory LLM error: {e}")
                    reason = state.get("unsupported_reason", "I can only answer ERP-related queries about customers, stock, GST, TDS, and TCS. Please ask a relevant business question.")
            else:
                reason = state.get("unsupported_reason", "I can only answer ERP-related queries about customers, stock, GST, TDS, and TCS. Please ask a relevant business question.")

            return {
                "messages": [
                    HumanMessage(content=user_query),
                    AIMessage(content=reason),
                ],
                "memory_answer": reason,
                "loop_count": loop_count + 1,
            }

        step = now()
        print("[3] Using bind_tools")

        prompt_start = time.perf_counter()

        system_prompt_text = build_system_prompt(
            user_query=user_query,
            selected_tools=selected_tools,
            query_parts=query_parts,
            summary=summary,
            messages=state.get("messages", []),
            last_tool_call=state.get("last_tool_call"),
            conversation_context=state.get("conversation_context"),
        )

        prompt_duration = time.perf_counter() - prompt_start

        print(f"[4] Built system prompt: {prompt_duration:.3f}s")
        print("system_prompt_chars:", len(system_prompt_text))

        chat_history = [
            msg for msg in state.get("messages", [])
            if not isinstance(msg, SystemMessage)
        ]

        system_prompt = SystemMessage(content=system_prompt_text + "\n\n/no_think")

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

        while remaining_names:
            if retry_count >= 3:
                break
            if retry_count >= 2 and called_names:
                break
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
                if name:
                    called_names.add(name)
                all_raw_calls.append(call)

            remaining_names = [
                n for n in selected_tools if n not in called_names
            ]

            if not remaining_names:
                break

            # If LLM chose not to call tools and query is conversational, respect that
            query_type = (state.get("query_type") or "").strip()
            is_meta = query_type == "conversational" or any(
                re.search(p, user_query, re.IGNORECASE) for p in META_QUESTION_PATTERNS_GLOBAL
            )
            if is_meta:
                print(f"[RETRY] Skipping retry — conversational query: {user_query}")
                break

            print(f"[RETRY] Missing tool calls for: {remaining_names}")
            loop_input = (
                [llm_input[0]]  # system prompt
                + llm_input[1:-1]  # chat history (exclude prev human/ai messages from this node)
                + [HumanMessage(content=user_query)]
                + [response]
                + [HumanMessage(content=f"You still need to call the following tool(s): {', '.join(remaining_names)}. Call them now.")]
            )

        # ── Fallback: if no tool calls after all retries, force the LLM to pick the closest tool ──
        if not all_raw_calls:
            print("[FALLBACK] No tool calls after retries — forcing LLM to pick closest tool")
            remaining_tools = available_tools
            fallback_msg = HumanMessage(
                content=f"The user asked: {user_query}\n\n"
                        f"Available tools: {', '.join(t.name for t in remaining_tools)}\n\n"
                        f"None of these tools perfectly match, but you MUST pick the MOST RELEVANT one and call it. "
                        f"Do NOT refuse. Choose the tool whose purpose best aligns with: {user_query}"
            )
            fallback_response = await llm.bind_tools(remaining_tools).ainvoke([
                llm_input[0], HumanMessage(content=user_query), fallback_msg
            ])
            fallback_calls = getattr(fallback_response, "tool_calls", None) or []
            for call in fallback_calls:
                name = call.get("name", "")
                if name:
                    called_names.add(name)
                all_raw_calls.append(call)
            if fallback_calls:
                print(f"[FALLBACK] LLM produced {len(fallback_calls)} tool call(s): {[c.get('name') for c in fallback_calls]}")
            else:
                print("[FALLBACK] LLM still refused — will return empty")
                # Last resort: reuse last_tool_call if available and relevant
                last_tc = state.get("last_tool_call", {})
                for tool_name, tool_args in last_tc.items():
                    if tool_name in {t.name for t in available_tools}:
                        all_raw_calls.append({
                            "name": tool_name,
                            "args": tool_args,
                            "id": f"call_lr_{tool_name}_{uuid.uuid4().hex[:8]}",
                            "type": "tool_call",
                        })
                        called_names.add(tool_name)
                        print(f"[FALLBACK] Last resort: reused last tool call: {tool_name}")
                        break

        # Force-inject missing tools whose domains match the query parts
        query_parts = state.get("query_parts", [original_query])
        all_part_domains = set()
        for qp in query_parts:
            hd, _ = classify_domains(qp)
            all_part_domains |= hd
        # Track domains already covered by tools the LLM called
        covered_domains = set()
        for tn in called_names:
            covered_domains |= set(TOOL_DOMAINS.get(tn, []))
        for tn in selected_tools:
            if tn not in called_names:
                td = set(TOOL_DOMAINS.get(tn, []))
                # Skip if no domain overlap with query parts
                if td and all_part_domains and not td & all_part_domains:
                    print(f"[FORCE-INJECT] Skipping {tn} (domain {td} no overlap with {all_part_domains})")
                    continue
                # Skip if tool's domains are already covered by an already-called tool
                if td and covered_domains and td & covered_domains:
                    print(f"[FORCE-INJECT] Skipping {tn} (domain {td} already covered by {covered_domains})")
                    continue
                all_raw_calls.append({
                    "name": tn,
                    "args": {},
                    "id": f"call_force_{tn}_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                })
                called_names.add(tn)
                covered_domains |= td
                print(f"[FORCE-INJECT] Adding missing tool: {tn}")

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

                # If LLM sent no fields, build from query triggers only (no force-added extras).
                # Use meta-level default_fields (not repair-level — repair is the sub-dict).
                # If LLM sent fields, merge with default + always_include to prevent data loss.
                llm_fields = args.get("fields")
                defaults = list(meta.get("default_fields", repair.get("default_fields", [])))
                always = list(meta.get("always_include_fields", []))
                if not llm_fields:
                    args["fields"] = defaults
                else:
                    merged = list(dict.fromkeys(defaults + always + llm_fields))
                    args["fields"] = merged

                # Clear term if it looks like a filter expression, not a product name
                term = args.get("term")
                if term and isinstance(term, str):
                    if re.search(r"\b(lt|gt|lte|gte|eq|ne|in|\$lt|\$gt)\b|(?:<=|>=|!=)", term):
                        args["term"] = ""

            worker_has = {}

            if args:
                for dk in ("from_date", "to_date"):
                    v = args.get(dk)

                    if v and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(v)):
                        worker_has[dk] = v

            # Discard hallucinated dates: if LLM provided dates but neither the query
            # nor the conversation summary has a date reference, they are invented.
            # Also check recent tool call context — follow-ups often reuse dates
            # from previous calls without mentioning them in the query text.
            summary_text = state.get("summary", "") or ""
            messages_list = state.get("messages", [])
            recent_tool_dates = ""
            for msg in reversed(messages_list[:-1]):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                        for dk in ("from_date", "to_date"):
                            dv = tc_args.get(dk, "") if isinstance(tc_args, dict) else ""
                            if dv:
                                recent_tool_dates += " " + str(dv)
                    if recent_tool_dates.strip():
                        break
            if worker_has and not re.search(r"\d{4}-\d{2}-\d{2}|\b\d{4}\b", combined_q + " " + summary_text + recent_tool_dates):
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
                cm = re.search(r"(?:customer|party|client)\s*(?:id|number|no|#)?\s*[:#-]?\s*(\d+)", combined_q)

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

            # Auto-extract customer ID number for get_customer tool (e.g. "customer 76" -> search="76")
            if name == "get_customer":
                cust_num = re.search(r"(?:customer|party|client)\s*(?:number|no|#)?\s*:?\s*(\d+)", combined_q)
                if cust_num and not new_args.get("search"):
                    new_args["search"] = cust_num.group(1)

            # Auto-inject invoice number filter for invoice tools
            INVOICE_TOOLS = {"get_overdue_invoices", "get_outstanding_sales_invoices", "get_outstanding_purchase_invoices"}
            if name in INVOICE_TOOLS:
                inv_patterns = [
                    r'[A-Z]{2,5}/\d{2,4}[-/]\d{2,4}/\d{2,5}',  # AI/22-23/050
                    r'\b[A-Z]{2,5}-\d{2,5}\b',                   # PR-52
                    r'\d{3,5}[-/]\d{7,10}[-/]\d{7,10}',         # 171-2645423-3220305
                ]
                all_matches = []
                for pat in inv_patterns:
                    all_matches.extend(re.findall(pat, combined_q, re.IGNORECASE))
                all_matches = list(dict.fromkeys(m.strip() for m in all_matches if m.strip()))
                if all_matches:
                    inv_filters = new_args.get("filters") or {}
                    if not isinstance(inv_filters, dict):
                        inv_filters = {}
                    if "invoiceNo" not in inv_filters:
                        if len(all_matches) == 1:
                            inv_filters["invoiceNo"] = all_matches[0]
                        else:
                            inv_filters["invoiceNo"] = all_matches
                    new_args["filters"] = inv_filters

            # Auto-inject ledger/customer name filter for invoice & ledger tools
            LEDGER_FIELDS = {"ledgerName", "ledger", "customerName", "vendor", "supplier", "party"}
            tool_fields = set(meta.get("fields", []))
            if tool_fields & LEDGER_FIELDS:
                existing_filters = new_args.get("filters") or {}
                already_has_ledger = any(
                    k in existing_filters for k in ("ledgerName", "ledger", "customerName", "vendor")
                )
                if not already_has_ledger:
                    raw_queries = [q for q in [state.get("canonical_query", ""), state.get("original_query", ""), user_query] if q]
                    name_candidates = set()
                    for raw_q in raw_queries:
                        for m in re.finditer(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', raw_q):
                            candidate = m.group(0)
                            if not re.search(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|From|To|Show|List|Get|Fetch|Find|Search|Total|All|Sales|Purchase|Invoice|Overdue|Current|Aging|Balance|Summary|Amount|Type|Date|Name|Number|Code|Value)\b', candidate):
                                name_candidates.add(candidate)
                        for m in re.finditer(r'[A-Z][a-z]+\s*&\s*[A-Z][a-z]+', raw_q):
                            name_candidates.add(m.group(0))
                    if name_candidates:
                        best_name = max(name_candidates, key=len)
                        inv_filters = new_args.get("filters") or {}
                        if not isinstance(inv_filters, dict):
                            inv_filters = {}
                        inv_filters.setdefault("ledgerName", {"contains": best_name})
                        new_args["filters"] = inv_filters
                        print(f"[AUTO-INJECT] {name}: injected ledgerName contains '{best_name}'")

            # Auto-inject TDS section filter
            if name == "get_tds_outstanding":
                section_match = re.search(r'\b(194[A-J])\b', combined_q)
                if section_match:
                    tds_filters = new_args.get("filters") or {}
                    if not isinstance(tds_filters, dict):
                        tds_filters = {}
                    tds_filters["section"] = section_match.group(1)
                    new_args["filters"] = tds_filters

            # Auto-inject TCS section filter
            if name == "get_tcs_outstanding":
                if re.search(r'\b206C\b', combined_q):
                    tcs_filters = new_args.get("filters") or {}
                    if not isinstance(tcs_filters, dict):
                        tcs_filters = {}
                    tcs_filters["section"] = "206C"
                    new_args["filters"] = tcs_filters

            # Generic value-comparison filter: detect "negative <field>",
            # "less than 0" / "0 se kam" / "< 0" with a field keyword → {field: {lt: 0}}.
            # And "positive" / "greater than 0" / "more than 0" / "0 se jyada" → {field: {gt: 0}}.
            # Applies to fields that look numeric (contain Qty/Value/Rate/Amount/Balance/Count/Gst/St).
            # Generic numeric threshold: "N se jada/kam", "N se upar/less"
            num_compare = None
            compare_op = None
            num_m = re.search(r"(\d+(?:\.\d+)?)\s+se\s+(jyada|jada|zada|upar|kam|less|km)\b", combined_q, re.IGNORECASE)
            if num_m:
                num_compare = float(num_m.group(1))
                compare_op = "gt" if num_m.group(2).lower() in ("jyada", "jada", "upar", "zada") else "lt"
            is_lt_zero = not num_m and bool(re.search(
                r"\bnegative\b|\bless\s+th[ae]n\s+0\b|<\s*0\b|\bbelow\s+0\b|0\s+se\s+kam\b",
                combined_q,
            ))
            is_gt_zero = not num_m and bool(re.search(
                r"\bpositive\b|\bgreater\s+th[ae]n\s+0\b|\bmore\s+th[ae]n\s+0\b|>\s*0\b|\babove\s+0\b|0\s+se\s+jyada\b|0\s+se\s+upar\b",
                combined_q,
            ))
            if is_lt_zero or is_gt_zero or (num_compare is not None and compare_op):
                operator = compare_op or ("gt" if is_gt_zero else "lt")
                threshold = num_compare if num_compare is not None else 0
                for field, aliases in TOOL_INTENT_REGISTRY.get(name, {}).get("field_aliases", {}).items():
                    if not re.search(r"(Qty|Value|Rate|Amount|Balance|Count|gst|igst|cgst|sgst|cess)", field, re.IGNORECASE):
                        continue
                    for alias in aliases:
                        if (" " in alias and alias in combined_q) or re.search(rf"\b{re.escape(alias)}\b", combined_q):
                            new_args.setdefault("filters", {})[field] = {operator: threshold}
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
            # When overwrite=True with base_args.fields, use base_args as source.
            # Then apply curated field_triggers as a safety net for fields the
            # user asked for but LLM missed. Skip triggers when query has
            # "sirf"/"only"/"just"/"bas" (user wants ONLY those fields).
            llm_sent_fields = "fields" in args
            if repair.get("overwrite") and "fields" in (repair.get("base_args") or {}):
                fields = list(repair["base_args"]["fields"])
            elif llm_sent_fields:
                fields = list(args.get("fields") or [])
            else:
                fields = list(repair.get("default_fields") or [])
            for kw, fld in repair.get("field_triggers", {}).items():
                match = kw in combined_q if " " in kw else bool(re.search(rf'\b{re.escape(kw)}\b', combined_q))
                if match and fld not in fields:
                    fields.append(fld)
            if fields:
                new_args["fields"] = fields

            # Ensure always_include_fields from tool meta are never lost
            always_meta = list(TOOL_INTENT_REGISTRY.get(name, {}).get("always_include_fields", []))
            for af in always_meta:
                if af not in new_args.get("fields", []):
                    new_args.setdefault("fields", []).append(af)

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

                # Remap LLM-invented field names to real API fields using field_aliases
                field_aliases = meta.get("field_aliases", {})
                alias_to_real = {}
                for real_field, aliases in field_aliases.items():
                    for alias in aliases:
                        alias_to_real[alias] = real_field
                for i, f in enumerate(flds):
                    if f not in field_aliases and f in alias_to_real:
                        flds[i] = alias_to_real[f]

            # ── Also remap filter keys in new_args via same alias lookup ──
            flds_dict = new_args.get("filters")
            if flds_dict and isinstance(flds_dict, dict):
                for k in list(flds_dict.keys()):
                    if k not in field_aliases and k in alias_to_real:
                        flds_dict[alias_to_real[k]] = flds_dict.pop(k)

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
                    if isinstance(cat_val, list):
                        new_args.setdefault("filters", {})["category"] = ",".join(cat_val)
                    else:
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

        def _generic_validate_tool_args(name: str, args: dict) -> dict:
            """Post-repair generic validation for ALL tools.
            1) Strips top-level params not in the function signature
            2) Resets hallucinated sort/order/metric/category params to defaults
            3) Salvages stripped values into name-like filters via field_aliases
            """
            from src.tools_api import tools_dict as _tools_dict
            tool_obj = _tools_dict.get(name)
            if not tool_obj:
                return args

            meta = TOOL_INTENT_REGISTRY.get(name, {})
            field_aliases = meta.get("field_aliases", {})
            combined_q = f"{original_query or ''} {state.get('canonical_query', '') or ''}".lower()

            alias_to_real = {}
            for real_field, aliases in field_aliases.items():
                for alias in aliases:
                    alias_to_real[alias.lower()] = real_field
                alias_to_real[real_field.lower()] = real_field

            valid_params = set(tool_obj.args.keys())

            stripped_extra = {}
            for k in list(args.keys()):
                if k not in valid_params:
                    stripped_extra[k] = args.pop(k)

            ENUM_HINTS = ("sort", "order", "metric", "category", "type")
            for k in list(args.keys()):
                if not any(hint in k.lower() for hint in ENUM_HINTS):
                    continue
                v = args.get(k)
                if not isinstance(v, str) or not v.strip():
                    continue
                v_lower = v.strip().lower()
                if v_lower in alias_to_real:
                    continue
                schema = tool_obj.args.get(k, {})
                if isinstance(schema, dict):
                    default_val = schema.get("default")
                    if default_val is not None:
                        old_val = args[k]
                        args[k] = default_val
                        print(f"[GENERIC VALIDATE] {name}: reset {k} from '{old_val}' to '{default_val}'")

            salvage_values = []
            for k, v in stripped_extra.items():
                if isinstance(v, str) and v.strip():
                    salvage_values.append(v)
                elif isinstance(v, (list, tuple)):
                    salvage_values.extend(str(x) for x in v if isinstance(x, str))

            if salvage_values:
                text = " ".join(salvage_values)
                name_candidates = [
                    rf for rf, aliases in field_aliases.items()
                    if any(
                        any(n in a.lower() for n in ("name", "customer", "vendor", "ledger", "party"))
                        for a in aliases
                    )
                ]
                if name_candidates:
                    filters = args.get("filters")
                    if not isinstance(filters, dict):
                        filters = {}
                        args["filters"] = filters
                    target = name_candidates[0]
                    if target not in filters or not isinstance(filters.get(target), dict):
                        filters[target] = {"contains": text}
                    elif isinstance(filters[target], dict) and "contains" not in filters[target]:
                        filters[target]["contains"] = text

            return args

        def _check_tool_alignment(name: str, combined_q: str, already_called: set | None = None) -> str | None:
            """Check if a different selected tool better matches the query's
            field_alias tokens. Returns better tool name or None.
            Skips tools already in the current batch (already_called)."""
            already_called = already_called or set()
            # Build excluded tools based on invoice patterns
            excluded = set(already_called)
            for domain, patterns in INVOICE_PATTERNS.items():
                if any(re.search(p, combined_q, re.IGNORECASE) for p in patterns):
                    for tn in selected_tools:
                        td = TOOL_DOMAINS.get(tn, [])
                        if td and domain not in td:
                            excluded.add(tn)
            def _get_tool_tokens(tool_name: str) -> set[str]:
                m = TOOL_INTENT_REGISTRY.get(tool_name, {})
                fa = m.get("field_aliases", {})
                tokens = set()
                for real_field, aliases in fa.items():
                    tokens.add(real_field.lower())
                    for a in aliases:
                        tokens.update(a.lower().split())
                return tokens

            qtokens = set(re.findall(r'\w+', combined_q))
            current_score = len(qtokens & _get_tool_tokens(name))

            best_tool = name
            best_score = current_score
            for tn in selected_tools:
                if tn == name or tn not in tools_dict:
                    continue
                if tn in excluded:
                    continue
                ts = len(qtokens & _get_tool_tokens(tn))
                if ts >= current_score + 1 and ts > best_score:
                    best_score = ts
                    best_tool = tn

            if best_tool != name:
                orig_domains = set(TOOL_DOMAINS.get(name, []))
                better_domains = set(TOOL_DOMAINS.get(best_tool, []))
                if orig_domains and better_domains and not orig_domains & better_domains:
                    print(f"[TOOL ALIGNMENT] blocked: {name} ({orig_domains}) -> {best_tool} ({better_domains}) - cross-domain")
                    return None
                print(f"[TOOL ALIGNMENT] {name} (score={current_score}) -> {best_tool} (score={best_score}) query: {combined_q}")
                return best_tool
            return None

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

            result = _apply_repair(name, args, original_query)
            if result:
                result["args"] = _generic_validate_tool_args(result["name"], result["args"])
                combined_q = f"{original_query or ''} {state.get('canonical_query', '') or ''}".lower()
                already_called = {tc.get("name") for tc in tool_calls} if tool_calls else set()
                # Don't redirect if the original tool already has query-specific args
                has_query_args = bool(result["args"].get("search") or result["args"].get("term") or result["args"].get("filters"))
                better = _check_tool_alignment(result["name"], combined_q, already_called) if not has_query_args else None
                if better:
                    preserved = {k: result["args"][k] for k in ("limit", "page") if k in result["args"]}
                    result = _apply_repair(better, preserved, original_query)
                    if result:
                        result["args"] = _generic_validate_tool_args(result["name"], result["args"])
            return result

        # Process bind_tools output — raw_tool_calls from bind_tools is already structured
        for call in raw_tool_calls:
            name = call.get("name", "")
            args = dict(call.get("args", {}))

            repaired = _repair_tool_call(name, args)

            if repaired:
                tool_calls.append({
                    "name": repaired["name"],
                    "args": repaired["args"],
                    "id": f"call_{repaired['name']}_{uuid.uuid4().hex[:12]}",
                    "type": "tool_call",
                })

        # Dedup same-named tool calls: keep the one with more complete args
        deduped = {}
        for call in tool_calls:
            n = call["name"]
            a = call["args"]
            if n not in deduped:
                deduped[n] = call
            else:
                existing = deduped[n]["args"]
                # Pick the call with more non-empty values
                existing_filled = sum(1 for v in existing.values() if v not in ("", None, [], {}))
                new_filled = sum(1 for v in a.values() if v not in ("", None, [], {}))
                if new_filled > existing_filled:
                    deduped[n] = call
        if len(deduped) < len(tool_calls):
            print(f"[DEDUP] tool_calls: {len(tool_calls)} -> {len(deduped)}")
            tool_calls = list(deduped.values())

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

        # Generic filter sanitization: strip invalid filter keys, move to search/term
        sanitized = [call for call in tool_calls if sanitize_tool_filters(call["name"], call["args"]) is not None]
        tool_calls = sanitized

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
            response = AIMessage(
                content=response.content or "",
                tool_calls=tool_calls,
                additional_kwargs=response.additional_kwargs,
                response_metadata=response.response_metadata,
                id=response.id,
            )

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
            "memory_answer": "",
            "loop_count": loop_count + 1,
        }

    except Exception as e:
        print(f"[CHAT MODEL ERROR]: {e}")
        print(f"[TOTAL chat_model_node before error]: {sec(node_start)}s")

        return {
            "messages": [
                HumanMessage(content=state.get("user_query", "")),
                AIMessage(content="Chat model error: The model encountered an issue while processing your request. Please try again."),
            ],
            "memory_answer": "",
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

        if state.get("memory_answer"):
            print("Memory answer detected, routing to response_generation...")
            return "response_generation"

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


def make_summary(data: dict, errors: list, unsupported_parts: list | None = None, total_rows: int = 0) -> str:
    parts = []
    KEY_FIELDS = {"name", "invoiceNo", "netAmount", "outstanding", "taxableAmount", "ledgerName", "category", "totalInvoices", "totalOutstanding"}

    for tool_name, records in data.items():
        count = len(records) if isinstance(records, list) else 0

        if count == 0:
            parts.append(f"{tool_name}: no records found")
        elif count == 1:
            summary = f"{tool_name}: found 1 record"
            record = records[0] if isinstance(records, list) else records
            if isinstance(record, dict):
                vals = {k: v for k, v in record.items() if k in KEY_FIELDS and v is not None}
                if vals:
                    summary += " | " + ", ".join(f"{k}={v}" for k, v in vals.items())
            parts.append(summary)
        else:
            summary = f"{tool_name}: found {count} records"
            record = records[0] if isinstance(records, list) else records
            if isinstance(record, dict):
                vals = {k: v for k, v in record.items() if k in KEY_FIELDS and v is not None}
                if vals:
                    summary += " (sample: " + ", ".join(f"{k}={v}" for k, v in vals.items()) + ")"
            parts.append(summary)

    if total_rows > 0:
        parts.append(f"total_rows: {total_rows}")

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
    Final deterministic cleanup — deduplicate records within each tool result.
    """
    if not isinstance(final_data, dict):
        return final_data

    import json as _json
    for tool_name, records in final_data.items():
        if isinstance(records, list):
            seen = set()
            deduped = []
            for r in records:
                key = _json.dumps(r, sort_keys=True) if isinstance(r, dict) else str(r)
                if key not in seen:
                    seen.add(key)
                    deduped.append(r)
            final_data[tool_name] = deduped

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

    has_category = any(isinstance(r,dict) and "category" in r for r in records)
    if not has_category:
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
    total_rows = 0
    current_tool_call_ids = set()
    # Accumulate tool calls across rounds — start from existing state
    last_tool_call = dict(state.get("last_tool_call", {}))
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            current_tool_call_ids = {tc.get("id") for tc in msg.tool_calls if tc.get("id")}
            for tc in msg.tool_calls:
                name = tc.get("name")
                args = tc.get("args")
                if name and args:
                    last_tool_call[name] = args  # overwrite with latest args for this tool
            break
    tool_messages = [
        msg for msg in messages
        if isinstance(msg, ToolMessage)
        and (not current_tool_call_ids or msg.tool_call_id in current_tool_call_ids)
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

            error_text = parsed.get("error", "Unknown tool error")
            errors.append({
                "tool": tool_name,
                "error": error_text,
            })

            # Track failure for conversation context
            entity_hint = None
            for src in [user_query, canonical_query]:
                m = re.search(r'(?:PR[-\s]?\d+|A/\d+/\w+|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b)', src or '')
                if m:
                    entity_hint = m.group(0)
                    break
            if entity_hint:
                tool_failures = list(ctx.get("tool_failures", []))
                tool_failures.append({
                    "tool": tool_name,
                    "entity": entity_hint,
                    "query": user_query,
                })
                ctx["tool_failures"] = tool_failures

            continue

        records = parsed.get("data", [])

        if records is None:
            records = []

        if not isinstance(records, list):
            records = [records]

        if isinstance(parsed, dict):
            total_rows += parsed.get("total_rows", 0) or 0
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
    # Build conversation_context: extract entity names/IDs for memory
    ctx = dict(state.get("conversation_context", {}))
    entities = list(ctx.get("entities", []))
    seen_names = {e["name"] for e in entities if "name" in e}
    for tool_msg in tool_messages:
        parsed = parse_tool_output(tool_msg.content)
        recs = parsed.get("data", []) if isinstance(parsed, dict) else []
        if not isinstance(recs, list):
            recs = [recs]
        for rec in recs:
            if isinstance(rec, dict):
                name = rec.get("name") or rec.get("ledgerName") or rec.get("customerName") or ""
                id_ = rec.get("id") or rec.get("ledgerId") or rec.get("customerId")
                if name and name not in seen_names:
                    seen_names.add(name)
                    entry = {"name": name}
                    if id_ is not None:
                        entry["id"] = id_
                    entities.append(entry)
    ctx["entities"] = entities
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
        "summary": make_summary(data, errors, unsupported_parts, total_rows),
        "errors": errors,
    }

    if total_rows > 0:
        final_response["total_rows"] = total_rows

    if unsupported_parts:
        final_response["unsupported_parts"] = unsupported_parts

    return {
        "final_response": final_response,
        "tools_utilized": tools_used,
        "last_tool_call": last_tool_call,
        "conversation_context": ctx,
    }
# ============================================
# TOOL NODE
# ============================================
tools_node = ToolNode(tools)

#==============================================
# SUMMARIZATION NODE
#==============================================

@traceable(name="summarization_node", run_type="chain")
async def summarization_node(state: MainState):
    """
    Summarizes the context of the conversation so far, including tool calls and their results after reaching a certain number of iterations 
    or when the user asks for a summary. This helps to keep the conversation focused and provides a recap for the user.
    """
    print("Summarization node activated............")
    messages = state.get("messages", [])
    current_summary = state.get("summary","")
    try:
        #We will get all the indices of the human messages in the conversation
        human_indices = [i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage)]

        #Threshold check will be done after 3 iterations of human messages
        if len(human_indices) < 6:
            print(f"Summary skipped............... Only {len(human_indices)} human messages so far.")
            return{}
        #Safe Cutoff: We keep the most recent query and everything after it.
        # Everything BEFORE it gets summarized and deleted.
        cutoff_index = human_indices[-1]

        #We wille exclude system messages from the summary payload to save tokens
        messages_to_summarize = [m for m in messages[:cutoff_index] if not isinstance(m, SystemMessage)]

        # Strip raw_response from tool messages to avoid blowing up summary LLM input
        stripped_messages = []
        for m in messages_to_summarize:
            if isinstance(m, ToolMessage):
                try:
                    parsed = json.loads(m.content)
                    if isinstance(parsed, dict) and "raw_response" in parsed:
                        del parsed["raw_response"]
                        stripped_messages.append(ToolMessage(content=json.dumps(parsed, ensure_ascii=False), tool_call_id=m.tool_call_id, name=m.name))
                        continue
                except Exception:
                    pass
            stripped_messages.append(m)
        messages_to_summarize = stripped_messages
        
        if not messages_to_summarize:
            return{}
        
        print(f"Summary triggered after and now removing  {len(human_indices)}old messages...............")


        summary_prompt = (
            f"You are an ERP assistant memory manager.\n"
            f"TASK: Write a concise summary of the conversation below. "
            f"Include key facts: which tools were called, what data was requested, "
            f"and any important results or conclusions. "
            f"Do NOT include raw data dumps — just the gist.\n\n"
            f"CONVERSATION:\n"
            f"/no_think\n"
        )
        summary_input = [SystemMessage(content=summary_prompt)] + messages_to_summarize
        response = await summary_llm.ainvoke(summary_input)
        new_summary = response.content
        if not new_summary:
            new_summary = ""
        MAX_SUMMARY_CHARS = 16000 #16000 characters will be roughly aroound 4000 tokens.
        if len(new_summary) > MAX_SUMMARY_CHARS:
            tail = new_summary[-MAX_SUMMARY_CHARS:]
            idx = tail.find("\n\n")
            if idx != -1:
                tail = tail[idx+2:]
            new_summary = "... (truncated) ...\n\n" + tail
        #we will issue deletion request to langgraph
        #We will need an id to delete messages which langchain generates automatically so we will use it

        delete_messages = [RemoveMessage(id=msg.id) for msg in messages_to_summarize if msg.id]

        print(f"Deleting {len(delete_messages)} old messages")
        return {
            "summary": new_summary,
            "messages": delete_messages,
        }
    except Exception as e:
        print(f"Error while removing messages and updating summary: {e}")
        return {
            "summary": current_summary or "",
        }
    
@traceable(name="response_generation_node", run_type="chain")
async def response_generation_node(state: MainState):
    """
    Generates a Natural-Language response that mirrors the user's language.
    Falls back to format_response_as_chat_text on any LLM failure. 
    """
    memory_answer = state.get("memory_answer", "")
    if memory_answer:
        return {"response_text": memory_answer,"memory_answer":""}

    final_response = state.get("final_response", {})
    messages = state.get("messages", [])

    original_query = (
                        state.get("original_query", "")
                        or state.get("user_query", "")
                        or ""
                    )
    if not original_query:
        for msg in reversed(messages):
            if isinstance(msg,HumanMessage):
                original_query = getattr(msg,"content","") or ""
                break
    detected_language = state.get("detected_language") or "auto"

    # Fallback: if detected_language is english/mixed but query has Hinglish words, override
    if detected_language not in ("hinglish", "hindi"):
        hinglish_words = {"batao", "chaia", "wale", "ka", "ki", "kya", "hai", "kitne", "konse", "konsa", "karli", "hua", "hue"}
        tokens = re.findall(r"\w+", original_query.lower())
        if any(w in tokens for w in hinglish_words):
            detected_language = "hinglish"

    previous_summary = state.get("summary", "") or ""
    conversation_context = state.get("conversation_context", {})
    system_prompt = (
        "You are an ERP assistant. Write a SHORT natural conversational reply using ONLY the tool results below.\n"
        "HARD RULES:\n"
    )
    if previous_summary:
        system_prompt += f"\nFor background, the conversation history is:\n{previous_summary}\n\n"
    if conversation_context:
        entities = conversation_context.get("entities", [])
        if entities:
            system_prompt += f"KNOWN ENTITIES:\n{json.dumps(entities, indent=2, ensure_ascii=False)}\n\n"
    if detected_language == "hinglish":
        system_prompt += (
            "1. LANGUAGE: The user wrote in Hinglish (Hindi words written with English letters).\n"
            "   Your ENTIRE reply MUST use ONLY a-z A-Z 0-9 and basic punctuation (. , ? !).\n"
            "   Do NOT use Devanagari (Hindi script), Chinese, or any other non-Latin characters.\n"
            "   Write Hindi words with English letters: 'aap', 'hai', 'nahi', 'se', 'ka', 'kaunsa'.\n"
            "   Use the exact same words the user used when possible.\n"
            "\n"
            "EXAMPLE:\n"
            "  User: muje customer details chaie\n"
            "  Tool result: Customer name: Rohan\n"
            "  Correct reply: aapke customer Rohan hai\n"
            "  WRONG: आपके ग्राहक रोहन हैं (NO Devanagari at all)\n"
        )
    elif detected_language == "hindi":
        system_prompt += (
            "1. LANGUAGE: The user wrote in Hindi (Devanagari script). Reply in Hindi Devanagari.\n"
        )
    else:
        system_prompt += (
            "1. LANGUAGE: The user wrote in English. Reply in English.\n"
        )
    system_prompt += (
        "2. Tool results are definitive answers. NEVER end your reply with '?' — use '.' even when stating a fact.\n"
        "   WRONG: aapke customer Rohan hai?\n"
        "   CORRECT: aapke customer Rohan hai.\n"
        "3. NEVER invent field names, values, IDs, or numbers. If a value is missing, say so plainly.\n"
        "   The TOOL RESULTS JSON below is the ONLY source of truth. Do NOT add IDs, names, or values not present in it.\n"
        "   WRONG: \"productId F12 ka quantity 5 se jada hai\" (when tool results have no productId or F12)\n"
        "   CORRECT: \"is baare mein data nahi mila\" or \"closingQty wala column nahi tha\"\n"
        "4. Do NOT use headers like '--- Customers ---' or '--- Results ---' or any section labels.\n"
        "5. Do NOT use bullet points or numbered lists unless the user explicitly asked for a list.\n"
         "6. Keep the reply to 1-4 short sentences (up to 8 if the user asked for multiple separate items).\n"
         "7. ONLY when a JSON record contains the literal key '__note' (showing 'X of Y records not shown'), "
         "tell the user: 'i can only show the records i have given you.Pleace give a specific filter or query to see the other records.'\n"
         "8. If TOOL RESULTS are empty [] for the relevant tool, that means no data was found. "
         "Say so plainly: 'X ka koi record nahi mila' in the user's language. Do NOT say records are hidden.\n"
         "9. If the tool results contain MULTIPLE records that match the user's query (e.g. same invoiceNo from different parties/dates), "
         "report ALL of them. Never omit any. List each with its distinguishing fields so the user can tell them apart.\n"
         "   WRONG: 'PR-269 ka net amount 3292.7 hai' (only one of two records)\n"
         "   CORRECT: 'PR-269 ke 2 records hain: Bigfoot se 3292.7 aur Amazon se 1215.95.'\n"
         "10. If the user's query has MULTIPLE distinct parts (e.g. 'TDS aur B2B', 'sales aur purchase'), "
         "your FIRST response MUST call a SEPARATE tool for EACH part — never skip a part. "
         "You have the full tool list; use every tool that matches a query part.\n"
           "11. Your reply text MUST mention the results of EVERY tool call. "
           "Do not skip any tool's output. Include the key numbers/facts from each tool in your sentences.\n"
           "    WRONG: 'NYKAA customers ka koi b2c detail nahi mila' (B2C data WAS returned: 250 vouchers, 30232.35 taxable)\n"
           "    CORRECT: 'April 2024 mein B2C Small ke 250 vouchers hain jinka taxable amount 30232.35 hai'\n"
           "    When the user asks for 'detail', 'sara detail', 'all details', or 'full info', "
           "include EVERY field value from the tool results in your reply (voucherCount, taxableAmount, "
           "igst, cgst, sgst, cess, tax, invoiceAmount, etc.). Do NOT cherry-pick only 1-2 fields.\n"
           "12. If the user asks for the ID of a category/item (e.g. 'X ka id', 'find id of X'), "
           "use the search-ledger tool. Set `groupType` based on the noun: "
           "expense for office expenses/salary/rent, party for customers/vendors, "
           "asset for fixed assets. Infer the noun from the query.\n"
           "13. B2B / B2C (Small/Large) / Exports / Nil-Rated / Exempt are GST categories, "
           "NOT sales categories. When the user asks for B2B, B2C, or any GST-category data, "
           "count, or split, ALWAYS use `get_gst_summary` with `filters.category` set to the "
           "appropriate value (b2b, b2cSmall, b2cLarge, exports, nilRated). "
           "Do NOT use `get_sales_summary` for B2B or B2C data — it returns overall sales totals, "
           "not the GST category breakdown.\n"
           "/no_think\n"
    )

    # Truncate large tool results to avoid overwhelming the LLM's context window
    MAX_SAMPLE_RECORDS = 50
    final_response_prompt = dict(final_response)
    data = final_response_prompt.get("data", {})
    if isinstance(data, dict):
        truncated_data = {}
        for tool_name, records in data.items():
            if isinstance(records, list) and len(records) > MAX_SAMPLE_RECORDS:
                truncated_data[tool_name] = records[:MAX_SAMPLE_RECORDS]
                if not _is_specific_lookup(original_query):
                    extra = len(records) - MAX_SAMPLE_RECORDS
                    truncated_data[tool_name].append({
                        "__note": f"Showing {MAX_SAMPLE_RECORDS} of {len(records)} records. {extra} more records not shown."
                    })
            else:
                truncated_data[tool_name] = records
        final_response_prompt["data"] = truncated_data

    human_prompt = (
        f"USER QUERY:\n{original_query}\n\n"
        f"TOOL RESULTS (JSON):\n{json.dumps(final_response_prompt, indent=2, ensure_ascii=False)}\n\n"
        f"Summary : {final_response.get('summary','')}\n\n"
    )

    try:
        full_content = ""
        async for chunk in summary_llm.with_config({"tags": ["response_stream"]}).astream([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]):
            if hasattr(chunk, "content") and chunk.content:
                full_content += chunk.content
        response_text = full_content.strip()
        if not response_text:
            raise ValueError("Empty response from LLM")
    except Exception as e:
        print(f"Error in response generation node: {e}")
        # Inline fallback instead of calling format_response_as_chat_text (not imported here)
        data = final_response.get("data", {}) if isinstance(final_response, dict) else {}
        summary = final_response.get("summary", "") if isinstance(final_response, dict) else str(final_response)
        lines = [summary] if summary else []
        for tool_name, records in data.items() if isinstance(data, dict) else []:
            if records:
                lines.append(f"\n{tool_name}:")
                for r in records[:10] if isinstance(records, list) else [records]:
                    if isinstance(r, dict):
                        parts = [f"{k}={v}" for k, v in r.items()]
                        lines.append("  " + ", ".join(parts))
        response_text = "\n".join(lines) if lines else str(final_response)
    return {"response_text": response_text}
