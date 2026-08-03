from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class VideoRetrievalState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

    # --- Intent routing ---
    intent: str                          # "search" | "chat"
    chat_response: str                   # used when intent == "chat"

    # --- Query rewriting (Node A) ---
    rewritten_query: str                 # Full self-contained query after resolving coreferences
    temporal_intent: str                 # "before" | "after" | "around" | "none"
    anchor_frame_ids: List[str]          # Frame IDs used as time anchor for temporal queries
    search_mode: str                     # "standard" | "temporal"

    # --- Query analysis (Node 1) ---
    clip_query: str                      # Visual semantic query (English)
    ocr_query: str                       # On-screen text keyword
    yolo_filters: Dict[str, int]         # Object detection filters

    # --- Retrieval & evaluation ---
    retrieved_videos: List[Dict[str, Any]]
    previous_retrieved_videos: List[Dict[str, Any]]  # Results from the previous search turn (for intersection)
    confidence_score: float
    is_clear: bool
    is_query_complete: bool              # Set by Analyzer: query specific enough?
    clarification_count: int             # Times Clarifier has asked
    search_context_summary: str          # Structured summary of accumulated search params across turns
