from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from agent.state import VideoRetrievalState
from agent.prompts.clarifier_prompt import clarifier_prompt
from core.config import settings
from core.logger import get_logger

log = get_logger(__name__)

llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=0.6)
clarifier_chain = clarifier_prompt | llm


async def clarifier_node(state: VideoRetrievalState) -> dict:
    log.info("--- [NODE] 4_clarifier ---")
    messages = state["messages"]
    clarification_count = state.get("clarification_count", 0)

    response = await clarifier_chain.ainvoke({"messages": messages})
    new_count = clarification_count + 1
    log.info(f"  Clarification #{new_count} asked: '{response.content}'")

    return {
        "messages": [response],
        "clarification_count": new_count,
    }
