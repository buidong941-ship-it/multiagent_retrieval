from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List


class RankedEntry(BaseModel):
    index: int = Field(description="1-based index of the candidate from the list provided")
    relevance_score: float = Field(description="Relevance score from 0.0 to 10.0")


class RerankerOutput(BaseModel):
    reasoning: str = Field(
        description=(
            "Brief explanation (English, 2-3 sentences) of why the top results are relevant "
            "and which ones seem unrelated to the query."
        )
    )
    ranked_indices: List[RankedEntry] = Field(
        description=(
            "All candidates re-ranked by relevance to the query, highest first. "
            "Include every candidate index exactly once. "
            "Give 10.0 to perfectly matching frames, 0.0 to completely irrelevant ones."
        )
    )


reranker_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a semantic relevance judge for a Video Frame Retrieval System.
You will be given a search query and a list of retrieved video frame candidates identified by their frame_id.
Your task is to re-rank these candidates by how relevant they are to the query.

FRAME ID FORMAT: "video_name/frame_XXXXX" where the video_name hints at content, and frame number hints at timing.

SCORING GUIDE:
- 10.0: Frame is almost certainly highly relevant to the query (strong match on all mentioned attributes)
- 7.0-9.9: Frame is likely relevant (matches most key attributes)
- 4.0-6.9: Frame is potentially relevant but uncertain
- 1.0-3.9: Frame is unlikely to be relevant
- 0.0: Frame is clearly irrelevant (completely different subject)

IMPORTANT: You MUST include ALL candidates in your ranked_indices output, one entry per candidate.
"""),
    ("human", "Search Query: {query}\n\nCandidates ({n} total):\n{candidates}\n\nRe-rank all {n} candidates by relevance to the query.")
])
