"""LangGraph execution graph for the customer support agent.

Flow: START → route_query → execute_tool → format_response → END

The graph is built once at startup via ``init_agent(router)`` and then
each request calls the stateless ``run_agent(query)`` function.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.faq_search import faq_search
from tools.order_status import order_status
from tools.raise_ticket import raise_ticket
from tools.refund_policy import refund_policy

if TYPE_CHECKING:
    from router.tool_router import ToolRouter

# ------------------------------------------------------------------ #
# State schema                                                         #
# ------------------------------------------------------------------ #

class AgentState(TypedDict):
    query: str
    selected_tool: str
    tool_output: str
    confidence: float
    status: str
    response: str


# ------------------------------------------------------------------ #
# Module-level compiled graph (set by init_agent)                      #
# ------------------------------------------------------------------ #

_compiled_graph = None


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

_ORDER_ID_PATTERNS = [
    r"#([\w]+-?\d+|\d+)",                  # #1234 or #ORD-2201
    r"order\s+(?:#?)([\w-]*\d[\w-]*)",     # order 1234 / order ORD-2201
    r"(?:number|num|no\.?)\s+#?(\d+)",     # number 5678
]


def _extract_order_id(query: str) -> str:
    for pattern in _ORDER_ID_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)
    return "UNKNOWN"


_RESPONSE_PREFIX = {
    "faq_search":    "Here's what I found:\n\n",
    "order_status":  "",
    "refund_policy": "Here is our refund policy:\n\n",
    "raise_ticket":  "",
}


# ------------------------------------------------------------------ #
# Graph construction                                                   #
# ------------------------------------------------------------------ #

def _build_graph(router: "ToolRouter"):
    """Return a compiled StateGraph, capturing *router* via closure."""

    def route_query(state: AgentState) -> dict:
        result = router.predict_with_confidence(state["query"])
        return {
            "selected_tool": result["tool"],
            "confidence": result["confidence"],
            "status": "routed",
        }

    def execute_tool(state: AgentState) -> dict:
        tool = state["selected_tool"]
        query = state["query"]

        if tool == "faq_search":
            output = faq_search(query)
        elif tool == "order_status":
            order_id = _extract_order_id(query)
            output = order_status(order_id)
        elif tool == "refund_policy":
            output = refund_policy()
        elif tool == "raise_ticket":
            output = raise_ticket(query)
        else:
            output = f"Tool '{tool}' is not recognised. Please contact support."

        return {"tool_output": output, "status": "executed"}

    def format_response(state: AgentState) -> dict:
        prefix = _RESPONSE_PREFIX.get(state["selected_tool"], "")
        return {
            "response": prefix + state["tool_output"],
            "status": "complete",
        }

    graph = StateGraph(AgentState)
    graph.add_node("route_query", route_query)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("format_response", format_response)

    graph.add_edge(START, "route_query")
    graph.add_edge("route_query", "execute_tool")
    graph.add_edge("execute_tool", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def init_agent(router: "ToolRouter") -> None:
    """Build and cache the compiled graph. Call once at startup.

    Args:
        router: A fully loaded :class:`ToolRouter` instance.
    """
    global _compiled_graph
    _compiled_graph = _build_graph(router)
    print("[LangGraphAgent] Graph compiled and ready.")


def run_agent(query: str) -> dict:
    """Execute the full routing + tool-call pipeline for *query*.

    Args:
        query: Customer support question.

    Returns:
        dict with keys: ``query``, ``selected_tool``, ``confidence``, ``response``.

    Raises:
        RuntimeError: If :func:`init_agent` has not been called.
    """
    if _compiled_graph is None:
        raise RuntimeError(
            "[LangGraphAgent] Graph not initialised. Call init_agent(router) at startup."
        )

    initial_state: AgentState = {
        "query": query,
        "selected_tool": "",
        "tool_output": "",
        "confidence": 0.0,
        "status": "pending",
        "response": "",
    }

    final_state = _compiled_graph.invoke(initial_state)

    return {
        "query": final_state["query"],
        "selected_tool": final_state["selected_tool"],
        "confidence": final_state["confidence"],
        "response": final_state["response"],
    }
