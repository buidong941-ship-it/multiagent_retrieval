from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class IntentOutput(BaseModel):
    # NOTE: field order matters for with_structured_output — the model fills
    # fields in declaration order, so putting `reasoning` first gives it a
    # short scratchpad before committing to `intent`. This measurably helps
    # small local models (7B) on ambiguous, short replies.
    reasoning: str = Field(
        description=(
            "Brief step-by-step check, in English, max 2 sentences: "
            "(1) does the message contain any visual/search clue (color, object, "
            "count, on-screen text, scene, time-of-day)? "
            "(2) was the assistant's immediately preceding message a clarifying "
            "question — if so, does this message answer it, even briefly? "
            "(3) is there also a greeting/thanks mixed in?"
        )
    )
    intent: Literal["search", "chat"] = Field(
        description=(
            "'search' if the message contains ANY retrieval clue OR answers a "
            "clarifying question the assistant just asked, even briefly/vaguely. "
            "'chat' ONLY if there is zero retrieval-relevant content."
        )
    )
    chat_response: str = Field(
        description=(
            "Vietnamese response. Required (non-empty) if intent is 'chat'. "
            "If intent is 'search' but the message ALSO contains a greeting/thanks, "
            "put a short natural acknowledgement here (e.g. 'Chào bạn, để mình tìm ngay nhé!') "
            "instead of discarding it. Otherwise leave empty."
        ),
        default=""
    )

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert intent router for a Video Retrieval chat system. Decide the user's intent using the FULL conversation history provided, not just the last message in isolation.

Pay special attention to whether the assistant's immediately preceding message was a clarifying question (asking about color / on-screen text / object count / etc). If it was, treat the user's reply as answering that question — even a one or two word reply like "màu đỏ", "cả hai", "không nhớ rõ" is a search refinement, NOT small talk.

DECISION RULES (in priority order):
1. ANY visual clue (color, object, count, on-screen text, action, scene, time-of-day) or an explicit request to find/search a video -> intent = 'search'.
2. A short or vague reply that directly answers a clarifying question the assistant just asked -> intent = 'search', even if the reply doesn't look like a typical standalone query.
3. Greeting/thanks/small talk combined WITH a search clue in the SAME message -> intent = 'search', but write a short natural acknowledgement into chat_response instead of discarding the pleasantry.
4. ONLY pure greeting / thanks / meta questions about what the assistant can do, with ZERO retrieval-relevant content and NOT answering a prior clarifying question -> intent = 'chat', with a natural Vietnamese chat_response.
5. When genuinely uncertain, bias towards 'search' — a missed search request is worse than one unnecessary search.

FEW-SHOT EXAMPLES:
- Previous AI message: "Bạn có thể cho mình biết thêm màu sắc hoặc chữ trên xe không?" | User: "màu đỏ" -> intent=search (short answer to a clarifying question)
- Previous AI message: "Bạn có thể cho mình biết thêm màu sắc hoặc chữ trên xe không?" | User: "không rõ nữa, thử tìm cái khác xem" -> intent=search (still answering, just negatively)
- User (no prior clarifying question): "chào bạn, tìm giúp mình video có con mèo màu trắng nhé" -> intent=search, chat_response="Chào bạn! Mình tìm ngay đây nhé."
- User (no prior clarifying question, no clue): "cảm ơn nhé" -> intent=chat, chat_response="Không có gì! Bạn cần tìm video gì cứ nói mình nhé."
- User: "bạn làm được những gì vậy?" -> intent=chat
"""),
    ("placeholder", "{messages}")
])
