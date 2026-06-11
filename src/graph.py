from langgraph.graph import StateGraph, END, START
from src.schema import MainState, InputState, OutputState
from src.semantic_search import semantic_search
from src.chat_model import chat_model_node
from src.tools import tools_node
from src.routing import routing_node
from src.deterministic_final import deterministic_final_node
from src.translator import translator_node
from src.summarization import summarization_node
from src.response_gen import response_generation_node
import time
import inspect
from langgraph.checkpoint.memory import MemorySaver

def timed_node(node_name: str, node_func):
    async def wrapper(state):
        start = time.perf_counter()

        try:
            if inspect.iscoroutinefunction(node_func):
                result = await node_func(state)
            elif hasattr(node_func, "ainvoke"):
                result = await node_func.ainvoke(state)
            else:
                result = node_func(state)

            duration = time.perf_counter() - start
            print(f"[TIMING] {node_name} took {duration:.3f}s")

            if result is None:
                result = {}

            if isinstance(result, list):
                result = {"messages": result}

            if isinstance(result, dict):
                result["step_timings"] = [
                    {
                        "node": node_name,
                        "duration_sec": round(duration, 3),
                    }
                ]

            return result

        except Exception as e:
            duration = time.perf_counter() - start
            print(f"[NODE ERROR] {node_name} failed after {duration:.3f}s: {e}")

            result = dict(state)
            result["step_timings"] = state.get("step_timings", []) + [{
                "node": node_name,
                "duration_sec": round(duration, 3),
                "error": str(e),
            }]
            return result

    return wrapper


def graph_builder():
    try:
        print("Building graph...")

        builder = StateGraph(
            MainState,
            input_schema=InputState,
            output_schema=OutputState,
        )

        builder.add_node("translator", timed_node("translator", translator_node))
        builder.add_node("semantic_search", timed_node("semantic_search", semantic_search))
        builder.add_node("response_generation", timed_node("response_generation", response_generation_node))
        builder.add_node("chat_model", timed_node("chat_model", chat_model_node))
        builder.add_node("tools", timed_node("tools", tools_node))
        builder.add_node(
            "deterministic_final",
            timed_node("deterministic_final", deterministic_final_node),
        )
        builder.add_node("summarization", timed_node("summarization", summarization_node))
        builder.add_edge(START, "translator")
        builder.add_edge("translator", "semantic_search")

        # Router LLM removed.
        # semantic_search now directly provides selected_tools to chat_model.
        builder.add_edge("semantic_search", "chat_model")

        builder.add_conditional_edges(
            "chat_model",
            routing_node,
            {
                "tools": "tools",
                "response_generation": "response_generation",
                "__end__": END,
            },
        )

        builder.add_edge("tools", "deterministic_final")
        builder.add_edge("deterministic_final", "response_generation")
        builder.add_edge("response_generation", "summarization")
        builder.add_edge("summarization", END)

        memory = MemorySaver()

        graph = builder.compile(checkpointer=memory)

    except Exception as e:
        print(f"Error building graph: {e}")
        raise e

    return graph