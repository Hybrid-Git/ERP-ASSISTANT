import json
import logging
import sys


class StructuredLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key in [
            "session_id",
            "request_id",
            "node",
            "duration_sec",
            "tool_name",
            "status",
            "error_code",
            "config_key",
            "next_node",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "model",
            "provider",
            "original_query",
            "canonical_query",
            "query_parts",
            "query_part",
            "selected_tools",
            "tool_endpoint",
            "cache_status",
        ]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("erp_assistant")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredLogFormatter())
        logger.addHandler(handler)

    return logger