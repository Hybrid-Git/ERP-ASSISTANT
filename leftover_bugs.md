# Leftover Bugs & Remaining Issues

These were identified during the initial codebase review. Entries marked **[FIXED]** were resolved during the refactoring pass; the rest remain open.

---

~~## 1. Critical: `route_to` always includes `"time"` prefix~~ **[FIXED — resolved by nodes.py extraction into `src/routing.py`]**

~~## 6. Medium: config.yaml truncation — `TH_EMBEDDING_RECALL_MIN` and `TH_EMBEDDING_LIMIT_PER_TOOL` ignored~~ **[FIXED — `utils.py` now reads thresholds via `get_cfg()` from config.yaml]**

---

## 1. ~~Medium: Error field map uses hardcoded key ordering~~ **[FIXED — `KEY_FIELDS` set replaced with ordered `FIELD_DISPLAY_ORDER` tuple in `make_summary()`]**

## 2. ~~Medium: `/chat/` endpoint keeps full `raw_response` in thread history~~ **[FIXED — `deterministic_final_node` now strips `raw_response` from ToolMessages before returning; redundant stripping removed from `summarization.py`]**

## 3. ~~Medium: `_clean_llm_response` only removes left-side prefixes~~ **[FIXED — now strips full ```json ...``` blocks, inline JSON remnants, returns empty fallback when nothing remains]**

## 4. ~~Medium: `classify_domains` mutates `selected_domains` default argument~~ **[FIXED — `DOMAIN_KEYWORDS` wrapped with `types.MappingProxyType`]**

## 5. ~~Low: Unused imports in new modules~~ **[FIXED — removed across 5 modules; also added missing `import time` to `chat_model.py`]**

## 6. ~~Low: `suggestions.txt` references stale file path~~ **[FIXED — updated to `src/response_gen.py`]**

## 7. ~~Low: README outdated `src/nodes.py` reference~~ **[FIXED — tree now lists all 8 node modules]**

## 8. ~~Low: config.yaml line 410 references `nodes.py`~~ **[FIXED — updated to `src/utils.py`]**

---

All 10 identified issues have been resolved.
