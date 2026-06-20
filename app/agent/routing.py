from app.schemas.state import MainState
import logging

logger = logging.getLogger("erp_assistant.routing")

async def routing_node(state: MainState):
    try:
        logger.info("Routing node started", extra={"node": "routing"})
        messages = state.get("messages", [])
        if not messages:
            return "__end__"
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        if state.get("memory_answer"):
            logger.info(
                            "Memory answer detected; routing to response_generation",
                            extra={"node": "routing", "next_node": "response_generation"},
                        )
            return "response_generation"
        loop_count = state.get("loop_count", 0)
        if loop_count > 5:
            return "__end__"
        logger.info(
                        "No tool call detected; ending graph",
                        extra={"node": "routing", "next_node": "__end__"},
                    )
        return "__end__"
    except Exception as e:
        print(f"Error in routing node: {e}")
        return "__end__"
