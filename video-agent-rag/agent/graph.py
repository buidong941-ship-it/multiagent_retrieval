from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import VideoRetrievalState
from agent.nodes.node_0_intent_router import intent_router_node
from agent.nodes.node_A_query_rewriter import query_rewriter_node
from agent.nodes.node_1_query_analyzer import query_analyzer_node
from agent.nodes.node_2_retriever import retrieval_node
from agent.nodes.node_B_temporal_resolver import temporal_resolver_node
from agent.nodes.node_C_re_ranker import re_ranker_node
from agent.nodes.node_D_result_merger import result_merger_node
from agent.nodes.node_3_evaluator import evaluator_node
from agent.nodes.node_4_clarifier import clarifier_node
from agent.nodes.node_5_responder import responder_node


# ──────────────────────────────────────────────
# Conditional edge functions
# ──────────────────────────────────────────────

def route_intent(state: VideoRetrievalState) -> str:
    """After intent_router: chat → responder directly; search → query_rewriter."""
    if state.get("intent") == "search":
        return "query_rewriter"
    return "responder"


def route_after_retrieval(state: VideoRetrievalState) -> str:
    """After retriever: temporal queries go to temporal_resolver, others straight to re_ranker."""
    if state.get("search_mode") == "temporal":
        return "temporal_resolver"
    return "re_ranker"


def route_evaluation(state: VideoRetrievalState) -> str:
    """After evaluator: clear results or exhausted clarifications → responder; else → clarifier."""
    is_clear = state.get("is_clear", False)
    clarification_count = state.get("clarification_count", 0)
    if is_clear or clarification_count >= 2:
        return "responder"
    return "clarifier"


# ──────────────────────────────────────────────
# Graph assembly
# ──────────────────────────────────────────────

def build_graph():
    workflow = StateGraph(VideoRetrievalState)

    # ── Register nodes ──────────────────────────
    workflow.add_node("intent_router",     intent_router_node)
    workflow.add_node("query_rewriter",    query_rewriter_node)
    workflow.add_node("query_analyzer",    query_analyzer_node)
    workflow.add_node("retriever",         retrieval_node)
    workflow.add_node("temporal_resolver", temporal_resolver_node)
    workflow.add_node("re_ranker",         re_ranker_node)
    workflow.add_node("result_merger",     result_merger_node)
    workflow.add_node("evaluator",         evaluator_node)
    workflow.add_node("clarifier",         clarifier_node)
    workflow.add_node("responder",         responder_node)

    # ── Entry point ─────────────────────────────
    workflow.set_entry_point("intent_router")

    # ── Edge: intent_router → (chat) responder | (search) query_rewriter ──
    workflow.add_conditional_edges(
        "intent_router",
        route_intent,
        {"query_rewriter": "query_rewriter", "responder": "responder"},
    )

    # ── Linear: rewriter → analyzer → retriever ──
    workflow.add_edge("query_rewriter",  "query_analyzer")
    workflow.add_edge("query_analyzer",  "retriever")

    # ── Edge: retriever → (temporal) temporal_resolver | (standard) re_ranker ──
    workflow.add_conditional_edges(
        "retriever",
        route_after_retrieval,
        {"temporal_resolver": "temporal_resolver", "re_ranker": "re_ranker"},
    )

    # ── temporal_resolver always flows into re_ranker ──
    workflow.add_edge("temporal_resolver", "re_ranker")

    # ── re_ranker → result_merger → evaluator ──
    workflow.add_edge("re_ranker",     "result_merger")
    workflow.add_edge("result_merger", "evaluator")

    # ── Edge: evaluator → (clear/exhausted) responder | (unclear) clarifier ──
    workflow.add_conditional_edges(
        "evaluator",
        route_evaluation,
        {"responder": "responder", "clarifier": "clarifier"},
    )

    # ── Terminal edges ──────────────────────────
    workflow.add_edge("responder", END)
    workflow.add_edge("clarifier", END)

    # ── Compile with persistent memory ─────────
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    return graph
