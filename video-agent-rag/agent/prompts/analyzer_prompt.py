from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Dict


class QueryParamsOutput(BaseModel):
    # Chain-of-thought scratchpad: helps small models extract correctly
    reasoning: str = Field(
        description=(
            "Brief analysis in English (max 3 sentences): "
            "(1) What is the visual scene/action to search for CLIP? "
            "(2) Is there any on-screen text, signs, license plates, or banners for OCR? "
            "(3) Are specific objects and their counts mentioned? "
            "(4) Is the query specific enough to yield good results?"
        )
    )
    clip_query: str = Field(
        description=(
            "Semantic visual search query for Milvus vector search (MUST be in English). "
            "Describe the scene, setting, actions, colors, and objects visually. "
            "Example: 'red car driving on highway at night', 'crowd of people in a market'. "
            "Leave empty only if there is absolutely no visual content to search for."
        ),
        default=""
    )
    ocr_query: str = Field(
        description=(
            "Keywords for BM25 text search — use ONLY if the user mentions specific "
            "text visible in the video: signs, banners, license plates, subtitles, etc. "
            "Keep as the exact words/phrase mentioned. Leave empty if no text is mentioned."
        ),
        default=""
    )
    yolo_filters: Dict[str, int] = Field(
        description=(
            "Object detection filters: map each specific object to its required minimum count. "
            "Use lowercase English object names (e.g., {'car': 2, 'person': 1, 'dog': 1}). "
            "If a count is not specified but the object is mentioned, default to 1. "
            "Leave empty if no specific objects with counts are required."
        ),
        default_factory=dict
    )
    is_query_complete: bool = Field(
        description=(
            "True if the query is specific enough to search (has at least a visual description, "
            "an object, or OCR text). False if the query is too vague, ambiguous, or only "
            "contains a general category with no distinguishing features (e.g., just 'a car' "
            "with no color, context, or other detail)."
        ),
        default=True
    )


analyzer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert query analyzer for a Video Frame Retrieval System.
Your task: parse the user's request (potentially in Vietnamese) and extract structured search parameters.

EXTRACTION RULES:
- `clip_query`: ALWAYS translate to English. Capture visual appearance: scene, environment, lighting, colors, actions, emotions, clothing. Be descriptive. This is the MOST important field.
- `ocr_query`: ONLY for explicit on-screen text. Do NOT paraphrase — use the exact words/letters the user mentions. Vietnamese text is OK here.
- `yolo_filters`: ONLY for countable physical objects explicitly mentioned. Common labels: 'person', 'car', 'truck', 'motorcycle', 'bicycle', 'bus', 'dog', 'cat'.
- `is_query_complete`: Set False when the request is a single generic noun with no other details (e.g., "xe", "người", "cái gì đó").

FEW-SHOT EXAMPLES:
User: "tìm video có xe ô tô màu đỏ chạy trên đường cao tốc"
→ clip_query="red car driving on a highway", ocr_query="", yolo_filters={{"car":1}}, is_query_complete=True

User: "xe tải có bảng hiệu chữ TAXI màu vàng"
→ clip_query="yellow taxi truck with signage", ocr_query="TAXI", yolo_filters={{"truck":1}}, is_query_complete=True

User: "hai người đang đánh nhau trong một con hẻm tối"
→ clip_query="two people fighting in a dark alley", ocr_query="", yolo_filters={{"person":2}}, is_query_complete=True

User: "tìm video có người"
→ clip_query="person", ocr_query="", yolo_filters={{"person":1}}, is_query_complete=False (too vague)

User: "màu xanh" (answering a prior clarifying question about car color)
→ clip_query="blue car" (infer from prior context), ocr_query="", yolo_filters={{}}, is_query_complete=True
"""),
    ("placeholder", "{messages}")
])
