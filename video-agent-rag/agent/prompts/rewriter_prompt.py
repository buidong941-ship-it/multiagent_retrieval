from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal, List


class RewriterOutput(BaseModel):
    reasoning: str = Field(
        description=(
            "Brief analysis (English, max 3 sentences): "
            "(1) Does the current message reference something from earlier? If yes, what? "
            "(2) Is there a temporal relationship expressed (before/after/around)? "
            "(3) What is the fully self-contained query after resolving all references?"
        )
    )
    rewritten_query: str = Field(
        description=(
            "A fully self-contained search query in Vietnamese (or mixed Vietnamese-English). "
            "Resolve ALL coreferences from conversation history: replace 'cái đó', 'xe kia', "
            "'nó', 'cảnh đó', etc. with the actual description. "
            "Include all previously mentioned details (color, object, text, scene). "
            "Example: 'màu xanh' → 'xe ô tô màu xanh đang đậu trên đường phố' "
            "(inferred from prior context about a car)."
        )
    )
    temporal_intent: Literal["before", "after", "around", "none"] = Field(
        description=(
            "'before': user wants frames occurring BEFORE a reference event/frame. "
            "  Keywords: 'trước đó', 'trước cảnh này', 'trước khi', 'xảy ra trước'. "
            "'after': user wants frames occurring AFTER a reference event/frame. "
            "  Keywords: 'sau đó', 'sau cảnh này', 'sau khi', 'tiếp theo là gì'. "
            "'around': user wants frames near a reference event. "
            "  Keywords: 'xung quanh', 'gần đó', 'lúc đó', 'cùng thời điểm'. "
            "'none': standard search with no temporal relationship."
        ),
        default="none"
    )
    search_mode: Literal["standard", "temporal", "intersect"] = Field(
        description=(
            "'temporal': temporal_intent is before/after/around AND prior results exist as anchor. "
            "'intersect': user is REFINING/ADDING details to a previous search (e.g. 'màu đỏ', "
            "'thêm điều kiện', 'có chữ X') AND there is accumulated search context from before. "
            "The system will intersect new results with previous results to narrow down. "
            "'standard': fresh query with no prior context, or user explicitly starts a new topic."
        ),
        default="standard"
    )


rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Query Rewriter for a Video Frame Retrieval System.
Your job is to analyze the FULL conversation history and rewrite the user's LATEST message into a complete, self-contained search query that can be understood without any conversation context.

CORE TASKS:
1. COREFERENCE RESOLUTION: Replace vague pronouns and references with their full descriptions.
   - "cái đó" / "xe kia" / "nó" → full description of the referenced object
   - "tìm lại nhưng màu khác" → "tìm [original object] màu [new color]"
   - "không, tôi muốn cái màu xanh" → "tìm [original subject] màu xanh"

2. CONTEXT ACCUMULATION: ALWAYS merge new details with previously established ones.
   The ACCUMULATED SEARCH CONTEXT block below is your most reliable source of truth.
   When the user adds a detail ("màu đỏ", "ban đêm", "có chữ TAXI"), COMBINE it with
   everything in the accumulated context — do NOT discard previous details.
   Example: accumulated="xe tải có chữ TAXI" + user says "màu xanh" → rewrite="xe tải màu xanh có chữ TAXI"

3. SEARCH MODE SELECTION — this is critical:
   - 'intersect': Use this when the user is ADDING or NARROWING details on the SAME topic.
     The system will take the new search results and find frames that ALSO appeared in the
     previous search — keeping only the overlap (intersection). This is the most powerful
     way to refine results without losing precision.
     Triggers: "thêm điều kiện", "lọc thêm", "cụ thể hơn", "chỉ lấy những cái", "có thêm",
     "màu [color]" (when prior search exists), "có chữ [text]" (when prior search exists),
     short one/two-word refinements like "màu đỏ", "ban đêm", "2 người".
   - 'temporal': Use when user asks for frames BEFORE/AFTER/AROUND a reference point.
   - 'standard': Use for completely fresh queries or when user explicitly changes the main subject.
     Triggers: "thôi tìm cái khác", "đổi chủ đề", first query in session.

4. TEMPORAL DETECTION:
   - "Trước cảnh đó có gì?" → temporal_intent="before"
   - "Ngay sau khi xe đỏ xuất hiện thì cảnh nào?" → temporal_intent="after"
   - "Xung quanh cảnh đó ra sao?" → temporal_intent="around"

FEW-SHOT EXAMPLES:
Example 1 — Intersect (adding detail):
  Accumulated context: "CLIP: 'truck on road' | OCR: TAXI"
  User: "màu vàng"
  → rewritten_query="xe tải màu vàng có chữ TAXI", search_mode="intersect"
  (Intersect: keep only frames from previous TAXI search that ALSO match 'màu vàng')

Example 2 — Intersect (adding object count):
  Accumulated context: "CLIP: 'xe tải màu vàng'"
  User: "thêm điều kiện có 2 người ngồi trong xe"
  → rewritten_query="xe tải màu vàng có 2 người ngồi bên trong", search_mode="intersect"

Example 3 — Standard (fresh query):
  Accumulated context: (none)
  User: "tìm video có người đi xe máy ban đêm"
  → rewritten_query="người đi xe máy ban đêm", search_mode="standard"

Example 4 — Standard (topic change):
  Accumulated context: "CLIP: 'yellow truck'"
  User: "thôi, giờ tìm cảnh đám cháy đi"
  → rewritten_query="đám cháy", search_mode="standard"

Example 5 — Temporal:
  Accumulated context: "CLIP: 'xe đỏ chạy trên đường'"
  User: "Ngay sau cảnh đó có gì vậy?"
  → rewritten_query="xe đỏ chạy trên đường", temporal_intent="after", search_mode="temporal"
"""),
    ("human", "ACCUMULATED SEARCH CONTEXT FROM PREVIOUS TURNS:\n{search_context}"),
    ("placeholder", "{messages}")
])
