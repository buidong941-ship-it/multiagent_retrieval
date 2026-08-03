from typing import Callable, List
from smolagents import CodeAgent, tool
from utils.logging_utils import get_logger
import os
import re
import ast
from dotenv import load_dotenv

import asyncio
import nest_asyncio
nest_asyncio.apply()

from pathlib import Path

# Tải biến môi trường từ file .env với đường dẫn tuyệt đối
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

log = get_logger(__name__)


class ReActAgent:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        gemini_model_id: str = "gemini-2.5-pro",
        max_steps: int = 12,
    ):
        """
        model_name: the Ollama model id used ONLY for the fallback path (when
            no GEMINI_API_KEY is set).
        gemini_model_id: the Gemini model id used for the primary path. Kept
            as its own parameter (instead of overloading model_name) since
            the two paths use unrelated model catalogs and previously this
            was silently hardcoded to "gemini-2.5-pro" regardless of what
            the caller passed in.
        """
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("RETRIEVAL_GEMINI_API_KEY")
        if api_key:
            from smolagents import OpenAIServerModel
            self.model = OpenAIServerModel(
                model_id=gemini_model_id,
                api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=api_key,
                client_kwargs={"timeout": 60.0, "max_retries": 2},
            )
            log.info(f"ReAct Agent initialized with Gemini ({gemini_model_id})")
        else:
            # Fallback về Ollama nếu không có key
            from smolagents import OpenAIServerModel
            self.model = OpenAIServerModel(
                model_id=model_name,
                api_base=base_url,
                api_key="ollama",  # dummy key
                client_kwargs={"timeout": 60.0, "max_retries": 1},
            )
            log.warning(f"ReAct Agent fallback to Ollama ({model_name}) — No GEMINI_API_KEY found")
        self.tools = []
        self.agent = None
        self.max_steps = max_steps

    def register_tool(self, tool_obj):
        """Register a smolagents-compatible tool: either a Tool subclass
        instance (as produced by agent_tools.build_tools()) or an @tool
        decorated function. Both are valid entries for CodeAgent(tools=...)."""
        self.tools.append(tool_obj)

    def _init_agent(self):
        if not self.agent:
            # CodeAgent writes python code and executes it.
            # Since we fixed the getsource indent bug, this will work smoothly and use less context tokens.
            agent_kwargs = dict(
                tools=self.tools,
                model=self.model,
                add_base_tools=False,
                max_steps=self.max_steps,
                additional_authorized_imports=["json", "ast"],
            )
            try:
                # planning_interval lets the agent re-plan every N steps,
                # which helps keep a multi-tool retrieval flow (parse ->
                # search -> fuse -> rerank -> answer) on track. Guarded with
                # try/except in case the installed smolagents version
                # doesn't support this kwarg.
                self.agent = CodeAgent(planning_interval=4, **agent_kwargs)
            except TypeError:
                log.warning("Installed smolagents version has no planning_interval support; continuing without it")
                self.agent = CodeAgent(**agent_kwargs)

    def _build_prompt(self, query: str) -> str:
        return (
            "You are a video retrieval agent. Your job is to find the video frame(s) that "
            "best match the user's query, using ONLY the provided tools.\n\n"
            f"User query (Vietnamese): {query}\n\n"
            "=== HOW THE TOOLS ACTUALLY WORK (read carefully) ===\n"
            "- query_parser_tool(query=...) returns a DICT (not a JSON string) with keys: "
            "objects, actions, attributes, ocr_text, temporal_relation, scene, translated_query. "
            "translated_query is ALREADY a short natural-language caption optimized for the "
            "SigLIP2 visual search model — reuse it directly as the `query` argument to "
            "search_visual / search_attribute / search_scene instead of writing your own "
            "translation. Access it as parsed['translated_query'], NOT parsed.translated_query.\n"
            "- search_visual, search_attribute, search_scene, search_object, search_text_in_video, "
            "find_frames_after, find_frames_before, search_sequence, fusion_tool, and rerank_tool "
            "ALL return a LIST OF DICTS shaped like "
            "{'frame_id': 'L21_V001_frame_000390', 'score': 0.8123}. "
            "Always access fields with ['frame_id'] / ['score'] — e.g. results[0]['frame_id']. "
            "This is plain Python data already, NOT a JSON string, so never call json.loads on it.\n"
            "- To use a previous result as an anchor for find_frames_after / find_frames_before, "
            "pass the frame_id STRING, not the whole dict:\n"
            "  find_frames_after(anchor_frame_id=results[0]['frame_id'], query='...')\n\n"
            "=== RECOMMENDED WORKFLOW ===\n"
            "1. parsed = query_parser_tool(query='...')  # always call this first\n"
            "2. Run 1-3 search tools using parsed['translated_query'] (and parsed['scene'] / "
            "parsed['objects'] where relevant). If parsed['temporal_relation'] is "
            "'after'/'before'/'then'/'second', first find an anchor frame for the primary event, "
            "then use find_frames_after / find_frames_before / search_sequence with the SECOND "
            "event's description (see parsed['actions']).\n"
            "3. If you ran more than one search tool, call fusion_tool(branch_names=[...]) to "
            "combine them — you may omit branch_names (or pass an empty list) to fuse everything "
            "searched so far.\n"
            "4. Optionally call rerank_tool(frame_ids=..., query=...) as a final quality pass.\n"
            "5. Call final_answer with a PLAIN LIST OF FRAME_ID STRINGS — no scores, no dicts. "
            "Example:\n"
            "```python\n"
            "final_ids = []\n"
            "for item in reranked:\n"
            "    final_ids.append(item['frame_id'])\n"
            "final_answer(final_ids)\n"
            "```\n\n"
            "=== RULES ===\n"
            "1. Write real Python code in ```python``` blocks to call tools. Simple for-loops and "
            "list comprehensions ARE allowed and often necessary (e.g. to pull frame_id out of a "
            "list of dicts). Avoid nested function definitions, file I/O, network calls, and any "
            "imports beyond json/ast.\n"
            "2. You may ONLY call the tools provided — never invent a tool or argument name. Use "
            "the EXACT keyword argument names shown in each tool's signature (e.g. branch_names=... "
            "for fusion_tool, not fusion_branch_names=...).\n"
            "3. After every tool call, print() the result so you can see it before deciding the "
            "next step, e.g. `print(results)`.\n"
            "4. Use only ASCII/English characters inside your Python code and string literals — no "
            "Vietnamese diacritics in code (Vietnamese is fine only in the original query text above).\n"
            "5. Never index into a result without checking it's non-empty first: `if results:`.\n"
            "6. Variable names must not contain spaces (use `bird_frames`, not `bird frames`).\n"
            "7. If you are stuck, getting empty results after a couple of reformulations, or are on "
            "your last available step, call final_answer immediately with whatever frame_ids you "
            "have (even an empty list) rather than looping further.\n"
        )

    def _normalize_result(self, result) -> List:
        """Best-effort coercion of whatever the agent's final_answer produced
        into a plain list. The agent is instructed to always pass a list of
        strings, but LLMs occasionally slip (e.g. return a dict, a
        stringified list, or a comma-separated string) — this avoids
        silently discarding a correct answer over a formatting slip."""
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for key in ("frame_ids", "results", "frames"):
                if key in result and isinstance(result[key], list):
                    return result[key]
            return list(result.values())
        if isinstance(result, str):
            try:
                parsed = ast.literal_eval(result)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            parts = [p.strip() for p in re.split(r"[,\n]", result) if p.strip()]
            return parts
        return []

    async def run(self, query: str) -> List:
        """
        Run the smolagents ReAct loop.
        Note: smolagents is mostly synchronous. Tools that need to await async
        code handle that internally (see agent_tools._run_async), and
        nest_asyncio lets that coexist with this coroutine's own event loop.
        """
        self._init_agent()
        log.info(f"smolagents starts reasoning for query: '{query}'")

        try:
            prompt = self._build_prompt(query)
            result = self.agent.run(prompt)
            log.info(f"smolagents finished. Result: {result}")
            return self._normalize_result(result)
        except Exception as e:
            log.error(f"smolagents Error: {e}")
            return []