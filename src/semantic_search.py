import re
from collections import Counter
from langsmith import traceable
from src.schema import MainState
from src.config import embedding_model
from src.tool_doc import TOOL_INTENT_REGISTRY
from src.utils import (
    _cosine_sim, _build_tool_embeddings, _tool_embeddings,
    normalize_text, add_unique, TH_EMBEDDING_RECALL_MIN, TH_RERANKER_TOP_K,
    CONNECTORS, ROUTE_KEYWORDS, DOMAIN_KEYWORDS, INVOICE_PATTERNS, TOOL_DOMAINS,
    ERP_AMBIGUOUS_THRESHOLD, max_erp_similarity,
)
from src.tools_api import tools_dict
from langchain_core.messages import AIMessage
from src.prompts import META_QUESTION_PATTERNS_GLOBAL, GREETING_PATTERNS, CAPABILITY_PATTERNS, OOD_TOPICS, _STOP_WORDS, VAGUE_ACTION_WORDS


def classify_domains(query: str, resolved_entities: list | None = None) -> tuple[set[str], set[str]]:
    query_lower = query.lower()
    hard = set()
    entity_tokens = set()
    for e in (resolved_entities or []):
        n = (e.get("name") or "").lower()
        for tok in re.findall(r"[a-z0-9]+", n):
            entity_tokens.add(tok)
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                if domain in {"gst", "tax"} and kw.lower() in entity_tokens:
                    continue
                hard.add(domain)
                break
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


def score_tools_via_reranker(query_part: str, registry: dict) -> list[str]:
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


def split_query_parts(query: str) -> list[str]:
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
    parts: list[str] = []
    for query in [original_query, canonical_query]:
        for part in split_query_parts(query):
            if part and part not in parts:
                parts.append(part)
    return parts or [original_query or canonical_query]


def _keyword_fallback(part: str, registry: dict | None = None) -> list[str]:
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


def _filter_for_list_intent(part: str, tools: list[str]) -> list[str]:
    q = part.lower()
    is_list_all = bool(re.search(r'\b(sab|sabhi|sare|saare|sari|saari|all|every|poora|saara|har)\b', q))
    if not is_list_all:
        return tools
    SINGLE_RECORD_TOOLS = {"get_customer_ledger"}
    filtered = [t for t in tools if t not in SINGLE_RECORD_TOOLS]
    if len(filtered) < len(tools):
        print(f"[LIST-INTENT] {part}: removed single-record tools ({set(tools)-set(filtered)})")
    return filtered


def _matches_ood_topic(query: str) -> bool:
    q = query.lower().strip()
    if not q:
        return False
    for topic, keywords in OOD_TOPICS.items():
        for kw in keywords:
            if re.search(rf"(?<!\w){re.escape(kw.lower())}(?!\w)", q):
                print(f"[OOD_TOPIC] Matched topic '{topic}' via keyword '{kw}' in: {q[:60]}")
                return True
    return False


_ALL_DOMAIN_WORDS: set[str] | None = None

def _get_all_domain_words() -> set[str]:
    global _ALL_DOMAIN_WORDS
    if _ALL_DOMAIN_WORDS is not None:
        return _ALL_DOMAIN_WORDS
    words = set()
    for kw in ROUTE_KEYWORDS:
        for w in kw.lower().split():
            words.add(w)
    for meta in TOOL_INTENT_REGISTRY.values():
        for kw in meta.get("keywords", []):
            for w in kw.lower().split():
                words.add(w)
        for alias in meta.get("aliases", []):
            for w in alias.lower().split():
                words.add(w)
    for domain_kws in DOMAIN_KEYWORDS.values():
        for kw in domain_kws:
            for w in kw.lower().split():
                words.add(w)
    _ALL_DOMAIN_WORDS = words
    return words


def _has_domain_content(parts: list[str]) -> bool:
    domain_words = _get_all_domain_words()
    for part in parts:
        part_lower = part.lower()
        tokens = set(re.findall(r'[a-z0-9]+', part_lower))
        content_tokens = tokens - VAGUE_ACTION_WORDS - _STOP_WORDS
        if content_tokens & domain_words:
            return True
        for pat_list in INVOICE_PATTERNS.values():
            for pat in pat_list:
                if re.search(pat, part, re.IGNORECASE):
                    return True
    return False


@traceable(name="semantic_search_node", run_type="retriever")
async def semantic_search(state: MainState) -> MainState:
    try:
        print("Semantic search node triggered")
        original_query = state.get("original_query") or state.get("user_query", "") or ""
        canonical_query = state.get("canonical_query", "") or ""
        document_type = (state.get("document_type", "") or "").lower().strip()
        user_query = canonical_query or original_query

        if not user_query:
            return {"retrieved_tools": [], "selected_tools": [], "query_parts": [], "skip_router": True}

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

        full_query = re.sub(r"[,/;:.!?]+", " ", original_query.strip().lower())
        full_query = re.sub(r"\s+", " ", full_query).strip()

        canonical_lower = (canonical_query or "").strip().lower()
        canonical_clean = re.sub(r"[,/;:.!?]+", " ", canonical_lower).strip() if canonical_lower else ""
        is_greeting = any(re.match(p, full_query) for p in GREETING_PATTERNS) or (
            canonical_clean and any(re.match(p, canonical_clean) for p in GREETING_PATTERNS)
        )
        is_capability = any(re.search(p, full_query, re.IGNORECASE) for p in CAPABILITY_PATTERNS) or (
            canonical_clean and any(re.search(p, canonical_clean, re.IGNORECASE) for p in CAPABILITY_PATTERNS)
        )
        is_meta = any(
            any(re.search(p, q, re.IGNORECASE) for p in META_QUESTION_PATTERNS_GLOBAL)
            for q in [original_query, canonical_query] if q
        )
        is_pure_meta = is_meta or (
            len(query_parts) > 0 and all(
                any(re.search(p, part, re.IGNORECASE) for p in META_QUESTION_PATTERNS_GLOBAL)
                for part in query_parts
            )
        )

        # Greeting/capability/meta regex patterns take priority over translator's query_type
        if is_greeting:
            has_erp_keywords = any(kw in full_query for kw in ROUTE_KEYWORDS)
            if not has_erp_keywords:
                print(f"Greeting detected — routing to LLM for natural response: {user_query}")
                return {"retrieved_tools": [], "selected_tools": [], "query_parts": query_parts, "skip_router": True, "query_type": "greeting"}

        if is_capability:
            has_erp_keywords = any(kw in full_query for kw in ROUTE_KEYWORDS)
            if not has_erp_keywords:
                print(f"Capability query detected — routing to LLM: {user_query}")
                return {"retrieved_tools": [], "selected_tools": [], "query_parts": query_parts, "skip_router": True, "query_type": "capability"}

        if is_pure_meta:
            print(f"Meta-question detected — no tool needed: {user_query}")
            return {"retrieved_tools": [], "selected_tools": [], "query_parts": query_parts, "skip_router": True}

        if query_type == "conversational":
            print(f"Translator flagged as conversational — no tool needed: {user_query}")
            return {"retrieved_tools": [], "selected_tools": [], "query_parts": query_parts, "skip_router": True}

        if query_type == "ood":
            print(f"Translator flagged as out-of-domain — no tool needed: {user_query}")
            return {"retrieved_tools": [], "selected_tools": [], "query_parts": query_parts, "skip_router": True, "query_type": "ood"}

        # Domain-content check — skip tool selection for vague queries
        # that consist only of generic action words (list, show, batao, etc.)
        # without any domain-specific noun (customer, stock, invoice, etc.)
        # Always route to ambiguous handler — do not reuse previous tools.
        if query_type in ("", "unknown", "general") and not _has_domain_content(query_parts):
            print(f"No domain content detected — routing to ambiguous handler: {user_query}")
            return {"retrieved_tools": [], "selected_tools": [], "query_parts": query_parts, "skip_router": True, "query_type": "ambiguous"}

        selected_tool_groups: list[list[str]] = []
        resolved_entities = state.get("resolved_entities", [])

        for part in query_parts:
            hard_domains, soft_domains = classify_domains(part, resolved_entities)
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
                tools_for_part = _filter_for_list_intent(part, tools_for_part)
                print(f"Reranker tools for part '{part}': {tools_for_part}")
                if tools_for_part:
                    selected_tool_groups.append(tools_for_part)
                continue

            kw_tools = _keyword_fallback(part, filtered_registry)
            if kw_tools:
                kw_tools = _filter_for_list_intent(part, kw_tools)
                print(f"Keyword fallback tools for part '{part}': {kw_tools}")
                if kw_tools:
                    selected_tool_groups.append(kw_tools)
            else:
                print(f"No tools found for part '{part}' via reranker or keyword fallback")

        combined_hard_domains: set[str] = set()
        for qp in query_parts:
            hd, _ = classify_domains(qp, resolved_entities)
            combined_hard_domains |= hd

        maybe_append = []
        if document_type in {"product", "inventory", "stock"}:
            maybe_append = ["get_stock_levels"]
        elif document_type in {"customer", "party"}:
            maybe_append = ["get_customer"]
        elif document_type in {"customer_ledger", "ledger"}:
            maybe_append = ["get_customer_ledger"]
        elif document_type in {"purchase_invoice", "purchase"}:
            maybe_append = ["get_outstanding_purchase_invoices", "get_overdue_invoices", "get_purchase_summary"]
        elif document_type in {"sales_invoice", "sales"}:
            maybe_append = ["get_outstanding_sales_invoices", "get_overdue_invoices", "get_sales_summary"]
        elif document_type in {"gst", "gst_report", "gst_summary"}:
            maybe_append = ["get_gst_summary"]

        if not maybe_append and (not document_type or document_type in {"unknown", "routeable", ""}):
            combined = f"{original_query or ''} {canonical_query or ''}".lower()
            if "invoice" in combined or any(re.search(p, combined, re.IGNORECASE) for pats in INVOICE_PATTERNS.values() for p in pats):
                maybe_append = ["get_outstanding_sales_invoices", "get_outstanding_purchase_invoices", "get_overdue_invoices"]

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
        intents_count = len(query_parts)
        if intents_count == 1:
            is_targeted = bool(re.search(
                r'\b(konsa|kaunsa|konse|kaunse|which|who|find|search|show me|list|all|kitne|kitna)\b',
                user_query.lower()))
            MAX_TOOLS_FOR_LLM = 2 if is_targeted else 4
        else:
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

        if combined and any(re.search(p, combined, re.IGNORECASE) for pats in INVOICE_PATTERNS.values() for p in pats):
            has_ledger_intent = any(kw in combined.lower() for kw in ['khata', 'ledger', 'statement', 'hisaab', 'account', 'balance'])
            if not has_ledger_intent:
                selected_tools = [t for t in selected_tools if t != 'get_customer_ledger']

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

        # Before marking OOD, check if query has ERP context from conversation history
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

        # Check OOD topics as negative filter (clearly non-ERP content)
        raw_queries = [q for q in [original_query, canonical_query] if q]
        is_ood = any(_matches_ood_topic(q) for q in raw_queries)

        if not is_ood:
            # Use semantic similarity against ERP domain embedding for ambiguous vs OOD
            erp_sim = max_erp_similarity(user_query)
            print(f"[ERP SIMILARITY] domain similarity: {erp_sim:.3f} (threshold: {ERP_AMBIGUOUS_THRESHOLD})")
            if erp_sim < ERP_AMBIGUOUS_THRESHOLD:
                is_ood = True

        if is_ood:
            print(f"Marking query as out-of-domain: {user_query}")
            return {
                "retrieved_tools": [],
                "selected_tools": [],
                "query_parts": query_parts,
                "skip_router": True,
                "query_type": "ood",
            }

        print(f"ERP-adjacent but underspecified — routing to ambiguous handler: {user_query}")
        return {
            "retrieved_tools": [],
            "selected_tools": [],
            "query_parts": query_parts,
            "skip_router": True,
            "query_type": "ambiguous",
        }

    except Exception as e:
        print(f"Error in semantic search node: {e}")
        return {"retrieved_tools": [], "selected_tools": [], "query_parts": [], "skip_router": True}
