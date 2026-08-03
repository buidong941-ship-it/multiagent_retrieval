from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from agent.state import VideoRetrievalState
from agent.prompts.rewriter_prompt import rewriter_prompt, RewriterOutput
from core.config import settings
from core.logger import get_logger

log = get_logger(__name__)

llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=0)
rewriter_chain = rewriter_prompt | llm.with_structured_output(RewriterOutput)


async def query_rewriter_node(state: VideoRetrievalState) -> dict:
    log.info("--- [NODE] A_query_rewriter ---")
    messages = state["messages"]

    # Pass accumulated search context explicitly — this is the agent's short-term memory
    search_context = state.get("search_context_summary", "") or "(no previous search context)"

    result = await rewriter_chain.ainvoke({
        "messages": messages,
        "search_context": search_context,
    })
    log.info(
        f"  Prior context: '{search_context}'\n"
        f"  Reasoning: '{result.reasoning}'\n"
        f"  Rewritten: '{result.rewritten_query}' | "
        f"Temporal: '{result.temporal_intent}' | Mode: '{result.search_mode}'"
    )

    # For temporal queries, use the top retrieved frames from PREVIOUS turn as anchors.
    # (These are still in state from the last search.)
    anchor_frame_ids: list = []
    if result.search_mode == "temporal":
        prior_results = state.get("retrieved_videos", [])
        if prior_results:
            anchor_frame_ids = [r["frame_id"] for r in prior_results[:5]]
            log.info(f"  Anchor frames set: {anchor_frame_ids}")
        else:
            # No prior results to anchor on → fall back to standard search
            log.warning("  Temporal intent but no prior results found — downgrading to standard search.")
            result = RewriterOutput(
                reasoning=result.reasoning,
                rewritten_query=result.rewritten_query,
                temporal_intent="none",
                search_mode="standard",
            )

    return {
        "rewritten_query": result.rewritten_query,
        "temporal_intent": result.temporal_intent,
        "anchor_frame_ids": anchor_frame_ids,
        "search_mode": result.search_mode,
        # Snapshot current results BEFORE resetting — merger node will use them for intersection
        "previous_retrieved_videos": state.get("retrieved_videos", []),
        # Reset retrieval state for the new search turn
        "retrieved_videos": [],
        "is_clear": False,
        "confidence_score": 0.0,
    }

