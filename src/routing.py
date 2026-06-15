from src.schema import MainState


async def routing_node(state: MainState):
    try:
        print("→ routing")
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
