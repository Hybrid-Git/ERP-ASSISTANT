from src.schema import MainState
from src.tools_api import tools_dict, tools
from src.tool_doc import TOOL_INTENT_REGISTRY, TOOL_NAME_ALIASES, CITY_WORDS
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
    "customer": ["customer", "client", "party", "ledger", "customer name", "customer list", "customer code", "name and id", "name aur id"],
    "vendor": ["vendor", "supplier", "vendor list"],
    "gst": ["gst", "gst summary", "gst detail", "gstr", "taxable", "igst", "cgst", "sgst", "b2b", "b2c"],
    "tax": ["tds", "tcs", "tax deducted", "tax collected", "tax outstanding"],
    "stock": ["stock", "inventory", "quantity", "hsn", "product", "slow moving", "stock level"],
    "analytics": ["top", "popular", "trend", "summary", "analytics", "report"],
}

INVOICE_PATTERNS = {
    "sales": [r"\bA/\d{4}/C\d{4}\b", r"\bAI/\d{4}/\d{4}\b", r"\bSI-?\d+\b", r"\bOUT-?\d+\b"],
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
    "get_overdue_invoices": ["sales", "purchase"],
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


def _has_own_identifiers(query: str) -> bool:
    q = query or ""
    if re.search(r'\b[A-Z]+/\d{2,}', q):
        return True
    if re.search(r'\b[A-Z]{2,}\d{4,}\b', q):
        return True
    if re.search(r'\b\d{6,}\b', q):
        return True
    return False


def _resolve_pronouns(query: str, conv_ctx: dict | None, last_tool: dict | None) -> tuple[str, list[dict]]:
    q_lower = query.lower()
    found = [p for p in HINGLISH_PRONOUNS if re.search(rf'\b{re.escape(p)}\b', q_lower)]
    if not found:
        return query, []

    if _has_own_identifiers(query):
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
    if re.search(r'\b[a-z]+/\d{4}/\d{3,}\b', q, re.IGNORECASE):
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
            text = " ".join(str(p) for p in text_parts if p)
            _tool_embeddings[tool_name] = embedding_model.embed_query(text)
    except Exception as e:
        print(f"[WARN] Failed to build tool embeddings: {e}")
        _tool_embeddings.clear()

def now():
    return time.perf_counter()

TRANSLATOR_PROMPT_BASE = """Normalize Hinglish/Hindi/Gujarati → clean English JSON.

SCHEMA: {"canonical_query":"...","document_type":"sales_invoice|purchase_invoice|customer|product|general","language":"...","confidence":"high|medium|low","query_type":"erp_query|conversational|ood|mixed","query_parts":["..."],"resolved_entities":[{"original":"...","resolved":"...","type":"..."}]}

WORD MAP: bill=sales_invoice, bikri=sales, kharidi=purchase, grahak=customer, rakam=amount, baki=outstanding, kam=less, zyada=greater, dikhao/batao=show, aur=and, kitne/kitna=how_many/much, hai/ho=is_are, kya=what, konse/konsa/jiska=which, kyu=why, chaia/chahiye=need, nahi=not, hamare/mera/uska/uski=our/my/his, wala/wale=with, sari/saari=all

RULES:
- query_type: "ood" if asking about non-ERP topics (movies, sports, recipes, general knowledge, news, weather, etc.), "conversational" if asking about conversation history (what we discussed, what was asked, recap, etc.), "erp_query" if asking about ERP data (customers/stock/GST/invoices), "mixed" if asking about both history AND data.
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
Q: muje avengers ke bare mai janna hai
A: {"canonical_query":"Tell me about Avengers","document_type":"general","language":"hinglish","confidence":"high","query_type":"ood"}
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

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "has", "have", "had",
    "and", "or", "but", "if", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "on", "off", "out", "over", "under",
    "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very",
    "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "myself", "you", "your", "yourself",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "we", "us", "our", "ours", "ourselves", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom",
    "ka", "ke", "ki", "ko", "se", "mai", "mein", "hai", "ho",
    "hu", "hain", "tha", "the", "thi", "thay", "hoga",
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

    query_tokens = {t for t in query_part.lower().split() if t not in _STOP_WORDS}

    scores = []
    for tool_name in registry:
        if tool_name not in _tool_embeddings:
            continue
        tool_emb = _tool_embeddings[tool_name]
        emb_sim = _cosine_sim(query_emb, tool_emb)
        meta = registry.get(tool_name, {})
        tool_text = f"{meta.get('description', '')} {' '.join(meta.get('aliases', []))} {' '.join(meta.get('keywords', []))}"
        tool_tokens = {t for t in tool_text.lower().split() if t not in _STOP_WORDS}
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

        GREETING_PATTERNS = [
            r"^(hello|hi|hey|hii|hiii|heyy|holla|namaste|namaskar|vanakkam|howdy|greetings|salam)\s*[!?.]*$",
            r"^(good\s*morning|good\s*afternoon|good\s*evening|good\s*night|gm|gn)\s*[!?.]*$",
            r"^(hey\s+there|hi\s+there|hello\s+there)\s*[!?.]*$",
            r"^(how\s+are\s+(you|u)|how\s+are\s+you\s+doing|how's\s+it\s+going|what's\s+up|wassup|sup)\s*[!?.]*$",
            r"^(kaise\s+ho|kya\s+haal|kya\s+kar\s+rahe|kya\s+kar\s+raha|kya\s+kar\s+rahi)\s*[!?.]*$",
            r"^(aap|ap|tu|tum|tumlog)\s+kaise\s+ho\s*[!?.]*$",
            r"^(aap|ap)\s+kese\s+ho\s*[!?.]*$",
            r"^(hello|hi|hey|hii|hiii|heyy|holla)\s+how\s+(are|r)\s+(you|u)\s*[!?.]*$",
            r"^(hello|hi|hey|hii|hiii|heyy|holla)\s+(how's|how is)\s+(it|everyone|you|things|going)\s*[!?.]*$",
        ]
        query_type = (state.get("query_type") or "").strip()
        if query_type == "conversational":
            # Check if translator misclassified a greeting (e.g. "ap kaise ho?" → "how are you")
            full_query = re.sub(r"[,/;:.!?]+", " ", original_query.strip().lower())
            full_query = re.sub(r"\s+", " ", full_query).strip()
            if any(re.match(p, full_query) for p in GREETING_PATTERNS):
                print(f"Translator misclassified greeting as conversational: {user_query}")
                # Fall through to greeting handling below
            else:
                print(f"Translator flagged as conversational — no tool needed: {user_query}")
                return {
                    "retrieved_tools": [],
                    "selected_tools": [],
                    "query_parts": query_parts,
                    "skip_router": True,
                }
        if query_type == "ood":
            print(f"Translator flagged as out-of-domain — no tool needed: {user_query}")
            return {
                "retrieved_tools": [],
                "selected_tools": [],
                "query_parts": query_parts,
                "skip_router": True,
                "query_type": "ood",
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

        CAPABILITY_PATTERNS = [
            r"what (can|do) (you|u) do",
            r"what('s| is) your purpose",
            r"what ('s|is) (this |the )?(chatbot|assistant|bot|tool) (for|about)",
            r"(tell|show) me (about|what) (you|u) (can |)do",
            r"what are your capabilities",
            r"how (can|do) (you|u) (help|assist)",
            r"what kind of (questions|queries) (can|do) (you|u) (answer|handle)",
            r"what is the use of (this |the )?(chatbot|assistant|bot|tool)",
            r"(what|which) (all |)(things|work|tasks) (can|do) (you|u) (do|help|handle)",
            r"(kaam|use|upayog) kya hai",
            r"kya kar sakte ho",
            r"kya (kaam|sahayta) kar sakte ho",
            r"aap kya kar sakte hain",
            r"ye (kya|kaisa) (hai|tool|chatbot)",
            r"aap (kya|kaise) (help|madad|sahayta) kar (sakte|sakta)",
        ]
        full_query = re.sub(r"[,/;:.!?]+", " ", original_query.strip().lower())
        full_query = re.sub(r"\s+", " ", full_query).strip()
        is_greeting = any(re.match(p, full_query) for p in GREETING_PATTERNS)
        if is_greeting:
            has_erp_keywords = any(kw in full_query for kw in ROUTE_KEYWORDS)
            if not has_erp_keywords:
                print(f"Greeting detected — routing to LLM for natural response: {user_query}")
                return {
                    "retrieved_tools": [],
                    "selected_tools": [],
                    "query_parts": query_parts,
                    "skip_router": True,
                    "query_type": "greeting",
                }
        is_capability = any(re.search(p, full_query, re.IGNORECASE) for p in CAPABILITY_PATTERNS)
        if is_capability:
            has_erp_keywords = any(kw in full_query for kw in ROUTE_KEYWORDS)
            if not has_erp_keywords:
                print(f"Capability query detected — routing to LLM: {user_query}")
                return {
                    "retrieved_tools": [],
                    "selected_tools": [],
                    "query_parts": query_parts,
                    "skip_router": True,
                    "query_type": "capability",
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

        combined_hard_domains: set[str] = set()
        for qp in query_parts:
            hd, _ = classify_domains(qp)
            combined_hard_domains |= hd

        maybe_append = []
        if document_type in {"product", "inventory", "stock"}:
            maybe_append = ["get_stock_levels"]
        elif document_type in {"customer", "party"}:
            maybe_append = ["get_customer"]
        elif document_type in {"customer_ledger", "ledger"}:
            maybe_append = ["get_customer_ledger"]
        elif document_type in {"purchase_invoice", "purchase"}:
            maybe_append = ["get_outstanding_purchase_invoices", "get_purchase_summary"]
        elif document_type in {"sales_invoice", "sales"}:
            maybe_append = ["get_outstanding_sales_invoices", "get_overdue_invoices", "get_sales_summary"]
        elif document_type in {"gst", "gst_report", "gst_summary"}:
            maybe_append = ["get_gst_summary"]

        if maybe_append:
            if combined_hard_domains:
                filtered = [t for t in maybe_append if set(TOOL_DOMAINS.get(t, [])) & combined_hard_domains]
                if filtered:
                    maybe_append = filtered
                else:
                    maybe_append = maybe_append[:1]
            selected_tool_groups.append(maybe_append)

        selected_tools = merge_unique_tools(selected_tool_groups)
        selected_tools = [t for t in selected_tools if t in tools_dict]
        MAX_TOOLS_FOR_LLM = 5
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
            if has_erp_kw or (document_type and document_type not in {"routeable", "unknown", ""}):
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
        if re.search(r'[A-Z]+/\d{2}-\d{2}/\d{3}', combined) or re.search(r'AI/\d{4}/\d{4}', combined):
            selected_tools = [t for t in selected_tools if t not in ('get_customer',)]

        # Exclude get_customer_ledger when query contains an invoice pattern
        if combined and any(re.search(p, combined, re.IGNORECASE) for pats in INVOICE_PATTERNS.values() for p in pats):
            selected_tools = [t for t in selected_tools if t != 'get_customer_ledger']

        # If document_type is specific, remove tools with exclusively wrong-domain
        if document_type in {"purchase_invoice", "purchase"}:
            selected_tools = [t for t in selected_tools if set(TOOL_DOMAINS.get(t, [])) != {"sales"}]
        elif document_type in {"sales_invoice", "sales"}:
            selected_tools = [t for t in selected_tools if set(TOOL_DOMAINS.get(t, [])) != {"purchase"}]

        if selected_tools:
            print(f"Final selected tools: {selected_tools}")
            return {
                "retrieved_tools": selected_tools,
                "selected_tools": selected_tools,
                "query_parts": query_parts,
                "skip_router": True,
            }

        OOD_PATTERNS = [
            r"^(who|what|why|when|where|how)\s+(is|are|was|were|does|do|did|can|could|will|would|shall|should)\s+",
            r"(tell me about|explain|describe|define)\s",
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

        print("No confident tool match. Marking query out-of-domain.")
        return {
            "retrieved_tools": [],
            "selected_tools": [],
            "query_parts": query_parts,
            "skip_router": True,
            "query_type": "ood",
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
    return f"{tool_name}={meta.get('category', '')}: {meta.get('description', '')}"

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


_REFERENCE_PATTERN = re.compile(
    r"\b(uska|uski|uske|iska|iski|iske|unka|unki|unke|inka|inki|inke"
    r"|this|that|these|those|its|it\b|they|them|their|he\b|she|his|her"
    r"|previous|last|first|same|also|too|again|another|previous"
    r"|pehle|pichle|pichli|pahle|baad\b|bad\b|aage"
    r"|aur|bhi\b|or\b|waise|aise|vaise"
    r"|kitne|kitna|konse|konsa|kaun\b|kaunse|kis\b|kisi"
    r"|dono|donu|wahi|wahee|yahi|yee|wohi|wohee)\b",
    re.IGNORECASE,
)

def build_system_prompt(
    user_query: str,
    selected_tools: list[str],
    query_parts: list[str] | None = None,
    summary: str | None = None,
    messages: list | None = None,
    last_tool_call: dict | None = None,
    conversation_context: dict | None = None,
    original_query: str = "",
) -> str:
    lines = [
        "You are an ERP assistant. Use the available tools to answer the user.",
        'Preserve all query text literally. Do not reinterpret or assume intent. If user says "mars" use "mars", not March.',
        "Never invent IDs, names, dates, or amounts.",
        "You MUST call at least one tool. Never answer in prose without a tool call.",
        "Do NOT output any thinking or reasoning — call the tool directly.",
        "",
        "FOLLOW-UP RULES:",
        "- If the answer is already in previous tool results, use it directly without a new API call.",
        "- When a tool has sort_field/sort_order parameters and the user asks for extreme/comparative values (highest, most, least, top, bottom, etc.), ALWAYS set sort_field to the field being compared and sort_order accordingly: 'desc' for highest/most/top, 'asc' for lowest/least/bottom.",
        "- CRITICAL: When the current query requires a DIFFERENT tool than the previous one (e.g. switching from get_customer to get_stock_levels), you MUST clear ALL old search terms and parameters. Reuse parameters ONLY within the same tool.",
    ]

    check_query = f"{original_query} {user_query}" if original_query else user_query
    is_follow_up = bool(_REFERENCE_PATTERN.search(check_query)) or len(user_query.split()) <= 3
    if messages and is_follow_up:
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
    lines.append("PARAMETER RULE:")
    lines.append("  NEVER copy parameters between different tools. Each tool has its own unique set of valid parameters.")

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
    return args





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
            query_type = (state.get("query_type") or "").strip()

            # ── Greeting: LLM generates warm, varied greeting ──
            if query_type == "greeting":
                greeting_prompt = (
                    "You are an ERP assistant. The user just greeted you.\n"
                    "Respond warmly and naturally like a friendly human. "
                    "Vary your greeting each time — don't repeat the same words. "
                    "You can say hi hello namaste, welcome them, and briefly offer help. "
                    "Keep it to 1-2 short sentences. Be warm, not robotic.\n"
                    "Do NOT mention tools, APIs, or technical details. "
                    "Speak in the same language the user used (English or Hinglish).\n"
                    "/no_think"
                )
                try:
                    resp = await summary_llm.ainvoke([
                        SystemMessage(content=greeting_prompt),
                        HumanMessage(content=state.get("original_query", "")),
                    ])
                    reason = (getattr(resp, "content", "") or "").strip()
                except Exception:
                    reason = "Hello! How can I help you with your ERP data today?"
                print(f"[CHAT MODEL] Greeting response: {reason}")
                return {
                    "messages": [
                        HumanMessage(content=user_query),
                        AIMessage(content=reason),
                    ],
                    "memory_answer": reason,
                    "loop_count": loop_count + 1,
                }

            # ── Capability: LLM describes what the assistant can do ──
            if query_type == "capability":
                tool_descriptions = []
                for tname, tmeta in TOOL_INTENT_REGISTRY.items():
                    desc = tmeta.get("description", "")
                    aliases = tmeta.get("aliases", [])
                    alias_str = ", ".join(aliases[:3])
                    tool_descriptions.append(f"- {alias_str}: {desc}")
                tools_text = "\n".join(tool_descriptions)
                cap_prompt = (
                    "You are an ERP assistant. The user asked about what you can do.\n"
                    "Describe your capabilities conversationally, like a helpful human.\n"
                    "Here are the tools/features available to you:\n"
                    f"{tools_text}\n\n"
                    "Explain in a natural, friendly way — not as a list of technical tools. "
                    "Say something like 'I can help you look up customers, check stock levels, "
                    "view GST reports, find outstanding invoices, and more.' "
                    "Keep it to 2-4 sentences. Be inviting and conversational. "
                    "Speak in the same language as the user (English or Hinglish).\n"
                    "Do NOT mention tool names, APIs, or technical details.\n"
                    "/no_think"
                )
                try:
                    resp = await summary_llm.ainvoke([
                        SystemMessage(content=cap_prompt),
                        HumanMessage(content=state.get("original_query", "")),
                    ])
                    reason = (getattr(resp, "content", "") or "").strip()
                except Exception:
                    reason = "I can help you with customer details, stock levels, GST reports, TDS/TCS, sales summaries, invoices, and more. Just ask!"
                print(f"[CHAT MODEL] Capability response: {reason}")
                return {
                    "messages": [
                        HumanMessage(content=user_query),
                        AIMessage(content=reason),
                    ],
                    "memory_answer": reason,
                    "loop_count": loop_count + 1,
                }

            # ── OOD (out-of-domain): LLM generates polite refusal ──
            if query_type == "ood":
                ood_prompt = (
                    "You are an ERP assistant. The user asked something OUTSIDE your domain.\n"
                    "CRITICAL: Do NOT answer the user's question. You do NOT have this information.\n"
                    "Instead, politely refuse and say you can only help with ERP-related "
                    "business queries (customers, stock, GST, TDS, TCS, invoices, sales, etc.).\n"
                    "Example: 'Sorry, I can only assist with ERP-related queries like customers, stock, GST, and invoices.'\n"
                    "Be friendly — don't sound robotic or defensive. "
                    "Suggest what you CAN help with. "
                    "Keep it to 1-2 short sentences. "
                    "Speak in the same language as the user (English or Hinglish).\n"
                    "Do NOT answer the question. Do NOT provide any information about the topic.\n"
                    "/no_think"
                )
                try:
                    resp = await summary_llm.ainvoke([
                        SystemMessage(content=ood_prompt),
                        HumanMessage(content=state.get("original_query", "")),
                    ])
                    reason = (getattr(resp, "content", "") or "").strip()
                except Exception:
                    reason = "I'm an ERP assistant — I can help with customers, stock, GST, TDS, invoices, and sales data. Could you ask about any of these?"
                print(f"[CHAT MODEL] OOD response: {reason}")
                return {
                    "messages": [
                        HumanMessage(content=user_query),
                        AIMessage(content=reason),
                    ],
                    "memory_answer": reason,
                    "loop_count": loop_count + 1,
                }

            # ── Existing hardcoded memory_answer/unsupported fallback ──
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
            original_query=state.get("original_query", ""),
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
        # Also respect the translator's document_type override as a domain signal
        doc_override = state.get("document_type", "").replace("_invoice", "").replace("_", "")
        if doc_override:
            all_part_domains.add(doc_override)
        # Track domains already covered by tools the LLM called
        covered_domains = set()
        for tn in called_names:
            covered_domains |= set(TOOL_DOMAINS.get(tn, []))
        combined_q = (f"{original_query or ''} {state.get('canonical_query', '') or ''}").lower()
        for tn in selected_tools:
            if tn not in called_names:
                td = set(TOOL_DOMAINS.get(tn, []))
                # Skip if no domain overlap with query parts
                if td and all_part_domains and not td & all_part_domains:
                    print(f"[FORCE-INJECT] Skipping {tn} (domain {td} no overlap with {all_part_domains})")
                    continue
                # Skip only if ALL of tool's domains are already covered by an already-called tool
                if td and covered_domains and td.issubset(covered_domains):
                    print(f"[FORCE-INJECT] Skipping {tn} (domain {td} already covered by {covered_domains})")
                    continue
                # Skip when no hard domains detected AND the query has no keyword match for this tool
                if not all_part_domains:
                    meta = TOOL_INTENT_REGISTRY.get(tn, {})
                    all_kw = set(meta.get("keywords", [])) | set(meta.get("aliases", []))
                    if not any(kw in combined_q for kw in all_kw):
                        print(f"[FORCE-INJECT] Skipping {tn} (no domain match and no keyword overlap with query)")
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
                return {"name": name, "args": args}

            combined_q = f"{original_query or ''} {state.get('canonical_query', '') or ''}".lower()

            worker_has = {}
            if args:
                for dk in ("from_date", "to_date"):
                    v = args.get(dk)
                    if v and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(v)):
                        worker_has[dk] = v

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

            worker_extra = {}
            if repair.get("overwrite") and args:
                for k, v in args.items():
                    if k not in ("from_date", "to_date") and v is not None:
                        worker_extra[k] = v

            new_args = (
                dict(repair.get("base_args", {}))
                if repair.get("overwrite")
                else dict(args or {})
            )

            for dk, dv in worker_has.items():
                new_args[dk] = dv

            for kw, kwar in repair.get("keyword_args", {}).items():
                if kw.lower() in combined_q:
                    new_args.update(kwar)

            if repair.get("extract_customer_id"):
                cm = re.search(r"(?:customer|party|client)\s*(?:id|number|no|#)?\s*[:#-]?\s*(\d+)", combined_q)
                if cm:
                    new_args["customer_id"] = int(cm.group(1))

            date_kws = repair.get("date_keywords")
            if date_kws and (not new_args.get("from_date") or not new_args.get("to_date")):
                f, t = extract_date_range_for_tool(combined_q, date_kws)
                if f:
                    new_args["from_date"] = f
                    new_args["to_date"] = t

            low_stock_kws = repair.get("low_stock_only_keywords")
            if low_stock_kws and new_args.get("low_stock_only") is True:
                if not any(kw in combined_q for kw in low_stock_kws):
                    new_args["low_stock_only"] = False

            if name == "get_customer":
                cust_num = re.search(r"(?:customer|party|client)\s*(?:number|no|#)?\s*:?\s*(\d+)", combined_q)
                if cust_num and not new_args.get("search"):
                    new_args["search"] = cust_num.group(1)

            INVOICE_TOOLS = {"get_outstanding_sales_invoices", "get_outstanding_purchase_invoices", "get_overdue_invoices"}
            if name in INVOICE_TOOLS and new_args.get("invoice_no"):
                new_args["limit"] = 5000
                new_args["sort_by"] = "invoiceDate"
                new_args["sort_order"] = "desc"

            param_aliases = repair.get("param_aliases", {})
            for llm_arg, real_param in param_aliases.items():
                if llm_arg in new_args and real_param not in new_args:
                    new_args[real_param] = new_args.pop(llm_arg)

            for k, v in worker_extra.items():
                if k not in new_args or new_args.get(k) in (None, "", []):
                    new_args[k] = v

            if re.search(r'\b(sabse\s+kam|sabse\s+jya[dz]a|sabse\s+zyada|least|most|lowest|highest|minimum|maximum)\b', combined_q) and not re.search(r'\b(top\s+\d+|first\s+\d+|last\s+\d+)\b', combined_q):
                if "limit" not in new_args or new_args.get("limit", 10) > 5:
                    new_args["limit"] = 1

            if name == "get_gst_summary" or name in ("get_tds_outstanding", "get_tcs_outstanding"):
                print(f"[{name.upper()} FINAL ARGS] {json.dumps(new_args, default=str)}")

            return {"name": name, "args": new_args}

        def _strip_unknown_params(tool_name: str, tool_args: dict) -> dict:
            t = tools_dict[tool_name]
            schema = t.args_schema
            valid = set(schema.model_fields.keys()) if schema and hasattr(schema, 'model_fields') else set()
            cleaned = {}
            for k, v in tool_args.items():
                if k in valid:
                    if v is None and schema and k in schema.model_fields:
                        field = schema.model_fields[k]
                        if not field.is_required():
                            cleaned[k] = field.default
                        else:
                            cleaned[k] = v
                    else:
                        cleaned[k] = v
                else:
                    print(f"[STRIP] {tool_name}: removing unknown param '{k}'")
            return cleaned

        def _repair_tool_call(name: str, args: dict) -> dict | None:
            name = TOOL_NAME_ALIASES.get(name, name)
            if name not in tools_dict:
                return None

            for alias, canonical in [
                ("date_from", "from_date"), ("date_to", "to_date"),
                ("startDate", "from_date"), ("endDate", "to_date"),
                ("fromDate", "from_date"), ("toDate", "to_date"),
                ("start_date", "from_date"), ("end_date", "to_date"),
            ]:
                if alias in args and canonical not in args:
                    args[canonical] = args.pop(alias)

            result = _apply_repair(name, args, original_query)
            if result:
                result["args"] = _strip_unknown_params(name, result["args"])
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

        # Generic filter sanitization: strip invalid filter keys
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

GST_CATEGORY_KEYWORDS = {
    "b2b": ["b2b"],
    "b2cSmall": ["b2c small", "b2csmall"],
    "b2cLarge": ["b2c large", "b2clarge"],
    "nilRated": ["nil rated", "nilrated", "nill rated", "nillrated"],
    "exempt": ["exempt"],
    "exports": ["export", "exports"],
    "creditNotesRegistered": ["creditnotesregistered", "credit note registered", "creditnoteregistered"],
    "creditNotesUnregistered": ["creditnotesunregistered", "credit note unregistered", "creditnoteunregistered"],
    "grandTotal": ["grand total", "total gst", "gst total", "grandtotal"],
}

def requested_gst_categories(query: str) -> list[str]:
    q = normalize_text(query)
    categories: list[str] = []

    def add(category: str):
        if category not in categories:
            categories.append(category)

    for cat_val, kws in GST_CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in q:
                add(cat_val)
                break

    if "b2c" in q and "b2cSmall" not in categories and "b2cLarge" not in categories:
        if not any(c.lower() in ("b2csmall", "b2clarge") for c in categories):
            add("b2cSmall")
            add("b2cLarge")

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
    document_type = (state.get("document_type", "") or "").lower().strip()

    data = {}
    tools_used = []
    errors = []
    total_rows = 0
    invoice_tool_match = None
    invoice_matches = {}
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
        if isinstance(parsed, dict) and parsed.get("_invoice_found") and parsed.get("matched_record"):
            invoice_matches[tool_name] = {
                "tool_name": tool_name,
                "matched_record": parsed["matched_record"],
                "_invoice_target": parsed.get("_invoice_target", ""),
            }
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
    # Resolve invoice match conflicts: prefer tool whose domain aligns with document_type
    if invoice_matches:
        doc_domain = document_type.replace("_invoice", "").replace("_", "")
        preferred = None
        # First pass: look for a match whose TOOL_DOMAINS contains the doc_domain
        for tn, match in invoice_matches.items():
            td = set(TOOL_DOMAINS.get(tn, []))
            if doc_domain in td:
                preferred = match
                break
        # Second pass: if no domain-aligned match, pick the one with the latest match
        if preferred is None:
            for tn, match in invoice_matches.items():
                preferred = match  # last one wins (insertion order preserved)
        final_response["_invoice_match"] = preferred
        invoice_tool_match = preferred

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
        "\n"
        "PERSONALITY:\n"
        "- Be warm and conversational — like a helpful human, not a robot.\n"
        "- Vary your phrasing. Don't repeat the same sentence patterns.\n"
        "- If the user used Hinglish/Hindi, mirror their language and tone.\n"
        "- Never say 'As an AI' or 'As a language model' — just be helpful.\n"
        "- Only end with a question (like 'Kuch aur?' or 'Anything else?') if the user's query itself ended with a question mark ('?'). Otherwise end with '.'.\n"
        "- If the user asks about something you already covered earlier, "
        "refer back naturally: 'Jaisa aapne pehle pucha tha...'\n"
        "\n"
        "PRIORITY RULE: When multiple tools returned results, focus on the tool(s) most relevant to the user's actual question. "
        "Ignore tool outputs that don't relate to what the user asked about. "
        "For example, if the user asked about 'Bangalore customers' and `get_stock_levels` also returned data, just answer about the customer.\n"
        "\n"
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
         "4. NO headings of any kind. NO 'Summary:', 'Details:', 'Sample Record:', "
         "'Additional Information:', 'Key Points:', 'Sample Invoice Details:', 'Outstanding X Summary:' — "
         "zero headings. Just plain conversational sentences.\n"
         "   WRONG: 'Sales Invoices Summary: AI/0324/0010 ka netAmount 29315.7 hai'\n"
         "   CORRECT: 'AI/0324/0010 ka netAmount 29315.7 hai'\n"
         "5. Do NOT use bullet points or numbered lists unless the user explicitly asked for a list.\n"
          "6. Answer ONLY what was asked. No analysis, advice, recommendations, "
          "'Next Steps:', 'Additional Insights:', or 'Key Points'. "
          "If the user asked for netAmount, give netAmount and nothing else. "
          "1-3 sentences max.\n"
          "   WRONG: 'Additional Insights: The invoices are from Oct-Dec 2024. Next Steps: Review high amounts.'\n"
          "   CORRECT: 'AI/0324/0010 ka netAmount 29315.7 hai, ledger CGLAM LIFESTYLE hai.'\n"
          "7. ONLY when a JSON record contains the literal key '__note' (showing 'X of Y records not shown'), "
          "tell the user: 'i can only show the records i have given you.Pleace give a specific filter or query to see the other records.'\n"
         "8. If TOOL RESULTS are empty [] for the relevant tool, that means no data was found. "
"Say so plainly: 'data nahi mila' or 'kuch nahi mila' in the user's language. "
"Do NOT repeat back any entity name, company name, ID, or invoice number that the user mentioned — "
"the tool found nothing, so there is nothing to confirm. Do NOT say records are hidden.\n"
          "9. If the tool results contain MULTIPLE records that match the user's query (e.g. same invoiceNo from different parties/dates), "
          "report ALL of them. Never omit any. List each with its distinguishing fields so the user can tell them apart.\n"
          "   WRONG: 'PR-269 ka net amount 3292.7 hai' (only one of two records)\n"
          "   CORRECT: 'PR-269 ke 2 records hain: Bigfoot se 3292.7 aur Amazon se 1215.95.'\n"
          "10. If the user's query has MULTIPLE distinct parts (e.g. 'TDS aur B2B', 'sales aur purchase'), "
          "your FIRST response MUST call a SEPARATE tool for EACH part — never skip a part. "
          "You have the full tool list; use every tool that matches a query part.\n"
            "11. When one tool found the exact record the user asked for (e.g. specific invoiceNo), "
            "answer ONLY from that tool's data. Ignore all other tool outputs completely. "
            "Do NOT describe, summarize, or mention any other tool's results.\n"
            "   WRONG: 'Purchase invoices have 684 records but Sales has AI/0324/0010 with netAmount 29315.7'\n"
            "   CORRECT: 'AI/0324/0010 ka netAmount 29315.7 hai' (purchase data not mentioned at all)\n"
            "    When the user asks for 'detail', 'sara detail', 'all details', or 'full info', "
            "include EVERY field value from the tool results in your reply (voucherCount, taxableAmount, "
            "igst, cgst, sgst, cess, tax, invoiceAmount, etc.). Do NOT cherry-pick only 1-2 fields.\n"
            "12. If the user asks for the ID of a category/item (e.g. 'X ka id', 'find id of X'), "
            "use the search-ledger tool. Set `groupType` based on the noun: "
            "expense for office expenses/salary/rent, party for customers/vendors, "
            "asset for fixed assets. Infer the noun from the query.\n"
             "13. B2B / B2C (Small/Large) / Exports / Nil-Rated / Exempt are GST categories, "
             "NOT sales categories. When the user asks for B2B, B2C, or any GST-category data, "
             "count, or split, ALWAYS use `get_gst_summary`. It returns ALL categories together — "
             "find the relevant one (b2b, b2cSmall, b2cLarge, exports, nilRated) in the result data. "
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

    invoice_match = final_response_prompt.pop("_invoice_match", None)
    if invoice_match and isinstance(invoice_match, dict):
        matched_tool = invoice_match.get("tool_name", "")
        if matched_tool and matched_tool in final_response_prompt.get("data", {}):
            filtered_data = {matched_tool: final_response_prompt["data"][matched_tool]}
            final_response_prompt["data"] = filtered_data
            final_response_prompt["summary"] = make_summary(filtered_data, [])
        final_response_prompt["_invoice_match"] = invoice_match

    human_prompt = (
        f"USER QUERY:\n{original_query}\n\n"
        f"TOOL RESULTS (JSON):\n{json.dumps(final_response_prompt, indent=2, ensure_ascii=False)}\n\n"
        f"Summary : {final_response_prompt.get('summary','')}\n\n"
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
