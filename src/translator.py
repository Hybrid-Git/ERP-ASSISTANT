# ── Translator node ──
# Uses normalizer_llm to normalize Hinglish/Hindi/Gujarati queries to clean English.
# Runs before semantic_search.

import json
import re
from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage
from src.schema import MainState
from src.config import normalizer_llm
from src.utils import log_token_usage, extract_json_object
# --- COMMENTED OUT (zero-regex migration): no word-list based routing ---
# from src.utils import NON_ENGLISH_HINTS, MULTILINGUAL_WORDS, ROUTE_KEYWORDS, INVOICE_PATTERNS, INVOICE_NO_PATTERNS
# from src.prompts import TRANSLATOR_PROMPT_BASE, META_QUESTION_PATTERNS_GLOBAL, HINGLISH_PRONOUNS, DONO_PRONOUNS, INVOICE_DOC_MAP
# from src.semantic_search import classify_domains
from src.prompts import TRANSLATOR_PROMPT_BASE

_translation_cache: dict[str, dict] = {}
_TRANSLATION_CACHE_MAX = 128

def _looks_tokenized_query_parts(parts: list[str]) -> bool:
    if not parts or len(parts) <= 2:
        return False
    single_word_count = sum(1 for p in parts if len(str(p).split()) == 1)
    return single_word_count / len(parts) >= 0.7

# --- COMMENTED OUT (zero-regex migration): identifier detection ---
# def _has_own_identifiers(query: str) -> bool:
#     q = query or ""
#     if re.search(r'\b[A-Z]+/\d{2,}', q):
#         return True
#     if re.search(r'\b[A-Z]{2,}\d{4,}\b', q):
#         return True
#     if re.search(r'\b\d{6,}\b', q):
#         return True
#     if any(re.search(p, q, re.IGNORECASE) for p in INVOICE_NO_PATTERNS):
#         return True
#     if re.search(r'\b[A-Z][A-Z_&.\-]{3,}\b', q):
#         return True
#     return False

def _resolve_pronouns(query: str, conv_ctx: dict | None, last_tool: dict | None) -> tuple[str, list[dict]]:
    """Pronoun resolution is handled by the translator LLM via the CONVERSATION CONTEXT section in the prompt."""
    return query, []

# --- COMMENTED OUT (zero-regex migration): dead code ---
# def needs_translation(query: str) -> bool:
#     q = query.lower()
#     words = set(re.sub(r"[^\w/.-]+", " ", q).split())
#     return bool(words & set(MULTILINGUAL_WORDS))
# def is_routeable_without_translator(query: str) -> bool:
#     q = re.sub(r"\s+", " ", (query or "").lower()).strip()
#     return any(keyword in q for keyword in ROUTE_KEYWORDS)
# def _split_multi_intent(query: str) -> list[str]:
#     parts = re.split(r'\s*(?:,|\.|;|\baur\b)\s+', query)
#     results = []
#     for p in parts:
#         p = re.sub(r'^(?:\s*and\s*|\s*aur\s*)', '', p.strip()).strip().rstrip(".,;")
#         if p and len(p.split()) > 1:
#             results.append(p)
#     return results if len(results) > 1 else [query]
# def _classify_query_type(query: str) -> str:
#     if not query:
#         return "unknown"
#     if any(re.search(p, query, re.IGNORECASE) for p in META_QUESTION_PATTERNS_GLOBAL):
#         return "conversational"
#     return "unknown"
# def _override_document_type(original: str, canonical: str, doc_type: str) -> str:
#     combined = f"{original or ''} {canonical or ''}"
#     for domain, patterns in INVOICE_PATTERNS.items():
#         if any(re.search(p, combined, re.IGNORECASE) for p in patterns):
#             mapped = INVOICE_DOC_MAP.get(domain)
#             if mapped and mapped != doc_type:
#                 print(f"[OVERRIDE] document_type: {doc_type} -> {mapped} (matched {domain} pattern)")
#                 return mapped
#     return doc_type

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

@traceable(name="translator_node", run_type="chain")
async def translator_node(state: MainState) -> MainState:
    try:
        print("→ translator")
        user_query = state.get("user_query", "") or ""
        import string
        user_query = user_query.strip().rstrip(string.punctuation + "/\\")
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
        cache_key = f"{user_query}::{summary[:100] if summary else ''}"
        cached = _translation_cache.get(cache_key)
        if cached:
            data = cached
        else:
            prompt = _build_translator_prompt(
                conversation_context=ctx,
                last_tool_call=ltc,
                summary=summary,
            )
            response = await normalizer_llm.ainvoke([
                SystemMessage(content=prompt),
                HumanMessage(content=user_query),
            ])
            input_text = prompt + "\n" + user_query
            log_token_usage(response, "translator", input_text=input_text, output_text=response.content)
            data = extract_json_object(response.content)
            if len(_translation_cache) >= _TRANSLATION_CACHE_MAX:
                _translation_cache.pop(next(iter(_translation_cache)))
            _translation_cache[cache_key] = data
        canonical_query = data.get("canonical_query") or user_query
        language = data.get("language", "mixed")
        if canonical_query.lower().strip() == user_query.lower().strip() and language != "english":
            print(f"[TRANSLATOR] Same output as input — treating as untranslated: {canonical_query}")
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
        language = data.get("language", "mixed")
        if language == "hindi" and not re.search(r'[\u0900-\u097F]', user_query):
            language = "hinglish"
        confidence = data.get("confidence", "medium")
        query_type = data.get("query_type", "")
        query_parts = data.get("query_parts") or []
        llm_resolved = data.get("resolved_entities") or []
        resolved_entities = (pre_resolved_entities or []) + (llm_resolved or [])
        final_canonical = user_query if query_type == "conversational" else canonical_query
        if _looks_tokenized_query_parts(query_parts):
            print(f"[TRANSLATOR FIX] Tokenized query_parts detected: {query_parts} -> using canonical query")
            query_parts = [final_canonical]
        elif not query_parts:
            query_parts = [final_canonical]
        print("Original query:", user_query)
        print("Canonical query:", canonical_query)
        print("Detected language:", language)
        print("Translator confidence:", confidence)
        print("Query type:", query_type)
        if query_parts:
            print("Query parts:", query_parts)
        if resolved_entities:
            print("Resolved entities:", resolved_entities)
        doc_type = data.get("document_type", "unknown")
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
            "query_parts": [user_query],
        }
