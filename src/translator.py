import json
import re
from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage
from src.schema import MainState
from src.config import normalizer_llm
from src.utils import log_token_usage, extract_json_object, NON_ENGLISH_HINTS, MULTILINGUAL_WORDS, ROUTE_KEYWORDS, INVOICE_PATTERNS, INVOICE_NO_PATTERNS
from src.prompts import TRANSLATOR_PROMPT_BASE, META_QUESTION_PATTERNS_GLOBAL, HINGLISH_PRONOUNS, DONO_PRONOUNS, INVOICE_DOC_MAP
from src.semantic_search import classify_domains


def _has_own_identifiers(query: str) -> bool:
    q = query or ""
    if re.search(r'\b[A-Z]+/\d{2,}', q):
        return True
    if re.search(r'\b[A-Z]{2,}\d{4,}\b', q):
        return True
    if re.search(r'\b\d{6,}\b', q):
        return True
    if any(re.search(p, q, re.IGNORECASE) for p in INVOICE_NO_PATTERNS):
        return True
    if re.search(r'\b[A-Z][A-Z_&.\-]{3,}\b', q):
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
    q_lower_for_match = query.lower()
    for ent in entities:
        ent_name = ent.get("name", "")
        if ent_name and ent_name.lower() in q_lower_for_match:
            resolved = query
            for p in found:
                resolved = re.sub(rf'\b{re.escape(p)}\b', '', resolved, flags=re.IGNORECASE)
            resolved = re.sub(r'\s+', ' ', resolved).strip()
            return resolved, [{"original": p, "resolved": "(self-reference)", "type": "self_reference"} for p in found]

    is_plural = any(p in DONO_PRONOUNS for p in found)
    if not is_plural:
        focus = (conv_ctx or {}).get("focus_entity")
        if focus and focus.get("name"):
            name = focus["name"]
            resolved = query
            for p in found:
                resolved = re.sub(rf'\b{re.escape(p)}\b', name, resolved, flags=re.IGNORECASE)
            return resolved, [{"original": p, "resolved": name, "type": "any"} for p in found]

    entities = (conv_ctx or {}).get("entities", [])
    if entities:
        _skip_entity_resolve = False
        last_entity = entities[-1]
        entity_domain = last_entity.get("domain", "")
        if entity_domain:
            query_domains, _ = classify_domains(query, [])
            if query_domains and entity_domain not in query_domains:
                print(f"[PRONOUN] Domain mismatch: entity domain '{entity_domain}' vs query domains {query_domains} — skipping")
                _skip_entity_resolve = True
        if not _skip_entity_resolve:
            is_plural = any(re.search(rf'\b{re.escape(p)}\b', query.lower()) for p in DONO_PRONOUNS)
            if is_plural and len(entities) >= 2:
                names = [e.get("name", "") for e in entities[-2:] if e.get("name")]
                if len(names) >= 2:
                    resolved = query
                    replacement = f"{names[0]} aur {names[1]}"
                    for p in found:
                        resolved = re.sub(rf'\b{re.escape(p)}\b', replacement, resolved, flags=re.IGNORECASE)
                    return resolved, [{"original": p, "resolved": replacement, "type": "any"} for p in found]
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


def is_plain_english_query(query: str) -> bool:
    q = query.lower().strip()
    if not q:
        return True
    for char in q:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            return False
        if 0x0A80 <= code <= 0x0AFF:
            return False
    words = set(q.replace(",", " ").replace("?", " ").split())
    return not any(word in words for word in NON_ENGLISH_HINTS)


def needs_translation(query: str) -> bool:
    q = query.lower()
    words = set(re.sub(r"[^\w/.-]+", " ", q).split())
    return bool(words & set(MULTILINGUAL_WORDS))


def is_routeable_without_translator(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").lower()).strip()
    return any(keyword in q for keyword in ROUTE_KEYWORDS)


def _classify_query_type(query: str) -> str:
    if not query:
        return "unknown"
    if any(re.search(p, query, re.IGNORECASE) for p in META_QUESTION_PATTERNS_GLOBAL):
        return "conversational"
    return "unknown"


def _override_document_type(original: str, canonical: str, doc_type: str) -> str:
    combined = f"{original or ''} {canonical or ''}"
    for domain, patterns in INVOICE_PATTERNS.items():
        if any(re.search(p, combined, re.IGNORECASE) for p in patterns):
            mapped = INVOICE_DOC_MAP.get(domain)
            if mapped and mapped != doc_type:
                print(f"[OVERRIDE] document_type: {doc_type} -> {mapped} (matched {domain} pattern)")
                return mapped
    return doc_type


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
        print("Translator node triggered")
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
            if language == "hindi" and not re.search(r'[\u0900-\u097F]', user_query):
                language = "hinglish"
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
