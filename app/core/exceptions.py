class ERPAssistantError(Exception):
    def __init__(self, message: str, error_code: str = "internal_error", user_message: str | None = None):
        self.message = message
        self.error_code = error_code
        self.user_message = user_message or "An unexpected error occurred. Please try again later."
        super().__init__(message)


class ERPAPIError(ERPAssistantError):
    def __init__(self, status_code: int | None = None, message: str = "", user_message: str | None = None):
        self.status_code = status_code
        super().__init__(
            message=f"ERP API error ({status_code}): {message}" if status_code else f"ERP API error: {message}",
            error_code="erp_api_error",
            user_message=user_message or "Could not reach the ERP system. Please try again later.",
        )


class ToolExecutionError(ERPAssistantError):
    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(
            message=f"Tool '{tool_name}' failed: {message}",
            error_code="tool_execution_error",
            user_message="The requested ERP operation failed. Please try again.",
        )


class ConfigError(ERPAssistantError):
    def __init__(self, config_key: str, message: str):
        self.config_key = config_key
        super().__init__(
            message=f"Configuration error for {config_key}: {message}",
            error_code="config_error",
            user_message="Server configuration error. Please contact administrator.",
        )