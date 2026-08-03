from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from agent.state import VideoRetrievalState
from agent.prompts.analyzer_prompt import analyzer_prompt, QueryParamsOutput
from core.config import settings
from core.logger import get_logger

log = get_logger(__name__)

llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=0)
analyzer_chain = analyzer_prompt | llm.with_structured_output(QueryParamsOutput)


async def query_analyzer_node(state: VideoRetrievalState) -> dict:
    log.info("--- [NODE] 1_query_analyzer ---")

    # Prefer the enriched rewritten_query from the Rewriter node.
    # Fall back to raw messages if rewriter was skipped (shouldn't happen in normal flow).
    rewritten_query = state.get("rewritten_query", "")
    if rewritten_query:
        # Inject the rewritten query as the effective user turn so Analyzer sees it clearly
        effective_messages = list(state["messages"]) + [
            HumanMessage(content=f"[REWRITTEN QUERY]: {rewritten_query}")
        ]
    else:
        effective_messages = list(state["messages"])

    result = await analyzer_chain.ainvoke({"messages": effective_messages})
    log.info(
        f"  Reasoning: '{result.reasoning}'\n"
        f"  CLIP: '{result.clip_query}' | OCR: '{result.ocr_query}' | "
        f"YOLO: {result.yolo_filters} | Complete: {result.is_query_complete}"
    )

    return {
        "clip_query": result.clip_query,
        "ocr_query": result.ocr_query,
        "yolo_filters": result.yolo_filters,
        "is_query_complete": result.is_query_complete,
        # Update the structured memory so rewriter can use it next turn
        "search_context_summary": (
            f"CLIP: '{result.clip_query}'"
            + (f" | OCR: '{result.ocr_query}'" if result.ocr_query else "")
            + (f" | YOLO: {result.yolo_filters}" if result.yolo_filters else "")
        ),
    }
