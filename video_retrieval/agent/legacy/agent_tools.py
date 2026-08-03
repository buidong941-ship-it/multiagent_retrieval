"""
Agent Tools Module — Class-based Tool API for smolagents.

Avoids the `inspect.getsource()` indent bug by using Tool subclasses
instead of @tool decorators on nested functions.

v2 changes (see module-level NOTE comments for rationale of each):
  - Every search tool now returns "frame_id score" strings instead of bare
    frame_id, so the agent (and reflection_tool) can actually reason about
    confidence instead of guessing.
  - branch_results accumulation is deduped by frame_id (keeps best score)
    instead of growing unbounded across repeated calls.
  - asyncio.run() replaced with _run_async(), which is safe even if the
    host application already has an event loop running in this thread.
  - query_parser_tool uses Ollama's JSON mode + temperature=0 and a
    balanced-brace JSON extractor instead of a greedy regex.
  - search_visual / search_attribute / search_scene / search_action now
    share one implementation (_ClipSearchTool) to remove ~80 lines of
    copy-pasted logic and keep future fixes in one place.
  - search_action now uses the dedicated `action_branch` passed into
    build_tools() instead of silently reusing clip_branch (this was a
    wiring bug in the original file: action_branch was accepted as a
    parameter but never actually used).
  - rerank_tool is now honest about what it does: score-sort plus an
    optional lexical boost (OCR/object-label overlap with the query) when
    meta_db is available, instead of accepting an unused `query` argument.
  - get_neighbor_frames now converts window_seconds to frames using the
    video's actual FPS (falling back to a configurable default) instead of
    treating "seconds" and "frames" as the same unit.
  - find_frames_after/before/search_sequence share a small per-session
    cache (_TemporalCache) so chasing several anchors in the same video
    doesn't refetch the full frame list every time.
  - reflection_tool reads real scores from branch_results when available
    instead of relying entirely on the agent self-reporting a score.
  - spatial_reasoning_tool normalizes bounding boxes to [0,1] using frame
    width/height when coordinates look like raw pixels, and supports a
    minimum-confidence filter.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import requests
from smolagents import Tool

from interfaces.base_interfaces import ParsedQuery, RetrievalResult
from utils.logging_utils import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_frame_id(frame_id: str) -> tuple[str, int]:
    """Parse video_id and frame_idx from a frame_id string."""
    frame_id = frame_id.split(" ")[0]
    match = re.search(r"_frame_(\d+)$", frame_id)
    if not match:
        return frame_id, 0
    frame_idx = int(match.group(1))
    video_id = frame_id[: match.start()]
    return video_id, frame_idx


def _run_async(coro):
    """Run an async coroutine from sync Tool.forward() code.

    smolagents Tool.forward() is synchronous, but the host app (API server,
    notebook, etc.) may already be running an asyncio event loop in this
    thread. Plain asyncio.run() raises RuntimeError in that case. This
    detects that situation and runs the coroutine on a fresh loop in a
    worker thread instead of crashing the tool call.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result()


def _call_ollama(
    prompt: str,
    base_url: str,
    model: str,
    *,
    json_mode: bool = False,
    temperature: float = 0.0,
    timeout: int = 60,
) -> str:
    """Call LLM synchronously (supports Gemini via LiteLLM if key is present, fallback to Ollama)."""
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    load_dotenv(BASE_DIR / ".env")
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("RETRIEVAL_GEMINI_API_KEY")
    if api_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature}
            }
            if json_mode:
                payload["generationConfig"]["responseMimeType"] = "application/json"
            
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            log.error(f"Gemini API call failed: {e}")
            return ""

    url = base_url.replace("/v1", "") + "/api/generate"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        log.error(f"Ollama call failed: {e}")
        return ""


def _extract_json_object(text: str) -> dict | None:
    """Extract the first balanced {...} JSON object from text.

    Unlike a greedy regex, this stops at the matching closing brace instead
    of the last closing brace in the whole response, so it survives models
    that add explanation text after the JSON.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def _fmt(r: RetrievalResult) -> dict:
    """Render a RetrievalResult as a dict with frame_id and score so the agent can 
    index it easily (e.g. result['frame_id'])."""
    score = getattr(r, "score", None)
    if score is None:
        return {"frame_id": r.frame_id, "score": 0.0}
    return {"frame_id": r.frame_id, "score": round(score, 4)}


def _top_n(results: list[RetrievalResult], n: int) -> list[dict]:
    return [_fmt(r) for r in results[:n]]


def _merge_dedup(existing: list[RetrievalResult], new: list[RetrievalResult]) -> list[RetrievalResult]:
    """Merge new retrieval results into an existing branch list, deduping by
    frame_id and keeping the higher-scoring copy. Without this, calling the
    same search tool twice (very common when an agent refines a query)
    silently double-counts frames and skews RRF fusion toward whatever was
    searched most often rather than what actually matched best."""
    by_id: dict[str, RetrievalResult] = {r.frame_id: r for r in existing}
    for r in new:
        prev = by_id.get(r.frame_id)
        if prev is None or getattr(r, "score", 0) > getattr(prev, "score", 0):
            by_id[r.frame_id] = r
    return list(by_id.values())


class _TemporalCache:
    """Per-session cache shared by the temporal/sequence tools so that
    chasing several anchors in the same video doesn't refetch the full
    frame list or FPS lookup every single call."""

    def __init__(self, default_fps: float = 25.0):
        self.default_fps = default_fps
        self._frames_by_video: dict[str, list[dict]] = {}
        self._fps_by_video: dict[str, float] = {}

    def get_frames(self, meta_db, video_id: str) -> list[dict]:
        if video_id not in self._frames_by_video:
            self._frames_by_video[video_id] = _run_async(meta_db.get_frames_by_video_async(video_id))
        return self._frames_by_video[video_id]

    def get_fps(self, meta_db, video_id: str) -> float:
        if video_id not in self._fps_by_video:
            getter = getattr(meta_db, "get_video_fps_async", None)
            if getter is None:
                self._fps_by_video[video_id] = self.default_fps
            else:
                try:
                    fps = _run_async(getter(video_id))
                    self._fps_by_video[video_id] = float(fps) if fps else self.default_fps
                except Exception as e:
                    log.warning(f"[_TemporalCache] fps lookup failed for {video_id}: {e}")
                    self._fps_by_video[video_id] = self.default_fps
        return self._fps_by_video[video_id]

    def clear(self) -> None:
        self._frames_by_video.clear()
        self._fps_by_video.clear()


def reset_session(branch_results: dict, agent_memory: dict, temporal_cache: "_TemporalCache | None" = None) -> None:
    """Call this from the host application before starting a NEW user query
    that reuses the same branch_results/agent_memory dicts. Without this,
    results and memory from a previous question leak into fusion/rerank for
    the next one, since these dicts are shared mutable state across tool
    calls by design."""
    branch_results.clear()
    agent_memory.clear()
    if temporal_cache is not None:
        temporal_cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Core Tools
# ─────────────────────────────────────────────────────────────────────────────

_QUERY_SCHEMA_DEFAULTS: dict[str, Any] = {
    "objects": [],
    "ocr_text": [],
    "actions": [],
    "attributes": [],
    "temporal_relation": "none",
    "scene": None,
    "translated_query": "",
}


def _normalize_parsed_query(result: dict, original_query: str) -> dict:
    """Fill in any keys the LLM dropped so downstream code can always rely
    on the full schema being present, instead of KeyError-ing later."""
    normalized = dict(_QUERY_SCHEMA_DEFAULTS)
    normalized.update({k: v for k, v in result.items() if k in _QUERY_SCHEMA_DEFAULTS})
    if not normalized.get("translated_query"):
        normalized["translated_query"] = original_query
    return normalized


class QueryParserTool(Tool):
    name = "query_parser_tool"
    description = (
        "Parse a Vietnamese search query into structured components. "
        "ALWAYS call this first. Returns JSON with: objects, actions, attributes, "
        "ocr_text, temporal_relation, scene, translated_query. The translated_query "
        "field is the best English text to pass into search_visual and friends — "
        "prefer reusing it instead of re-translating the query yourself."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The raw Vietnamese search query from the user.",
        }
    }
    output_type = "any"

    def __init__(self, agent_memory: dict, ollama_url: str, ollama_model: str):
        super().__init__()
        self.agent_memory = agent_memory
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def forward(self, query: str) -> dict:
        prompt = f"""You are a Vietnamese Video Retrieval Query Analyzer.
Parse the query into JSON. Respond ONLY with valid JSON.

Structure:
{{
  "objects": ["noun1"],
  "ocr_text": ["visible text in video"],
  "actions": ["verb1"],
  "attributes": ["color/size"],
  "temporal_relation": "before|after|then|second|first|none",
  "scene": "scene type or null",
  "translated_query": "English visual caption for image-text search"
}}

Rules:
- temporal_relation: use "second" for 2nd item, "first" for 1st, "after"/"before" for relative time.
- IMPORTANT: Translate ALL items in "objects", "actions", and "attributes" into ENGLISH.
- Keep "ocr_text" in its original language (e.g., Vietnamese).
- ABSOLUTELY NO CHINESE CHARACTERS.

translated_query is fed DIRECTLY into a SigLIP2 image-text retrieval model, so it
must read like a real photo caption, not a keyword list or a search query:
- Write ONE short, natural, present-tense sentence describing what is visually
  IN THE FRAME. Max ~15 words. SigLIP2 truncates long text, so keep it tight.
- Do NOT start with "a photo of" / "an image of" / "a picture of" — SigLIP2 was
  trained on natural captions, not that classification-style template, so the
  prefix only wastes tokens and adds no signal for retrieval.
- Do NOT include any temporal/sequence connector words ("then", "after",
  "before", "first", "second", "next", "while"). SigLIP2 scores ONE static
  frame at a time and has no notion of time order — those words are noise to
  the visual embedding. If the query describes two sequential events
  (temporal_relation is not "none"), translated_query must describe ONLY the
  first/primary visual scene; the second event stays in "actions" for the
  agent to search separately (e.g. with find_frames_after / search_sequence).
- Do NOT include any on-screen/OCR text inside translated_query — that only
  belongs in "ocr_text". Mixing literal signage text into a visual caption
  confuses the image-text embedding.
- Prefer concrete, visual words (people, objects, setting, pose/action) over
  abstract or emotional words a camera can't literally see.

Examples:
Query: "cảnh biển báo sạt lở nguy hiểm"
{{"objects": ["warning sign"], "ocr_text": ["sạt lở nguy hiểm"], "actions": [], "attributes": ["dangerous"], "temporal_relation": "none", "scene": "street", "translated_query": "a warning road sign on a street"}}

Query: "người đàn ông xuống xe rồi bỏ chạy"
{{"objects": ["man", "car"], "ocr_text": [], "actions": ["get out of car", "run away"], "attributes": [], "temporal_relation": "after", "scene": null, "translated_query": "a man getting out of a car"}}

Query: "một chiếc xe ô tô màu đỏ đang cháy"
{{"objects": ["car"], "ocr_text": [], "actions": ["burning"], "attributes": ["red"], "temporal_relation": "none", "scene": null, "translated_query": "a red car on fire"}}

Query: "người phụ nữ mặc áo vàng đang nói chuyện trên tivi, sau đó xuất hiện dòng chữ 'khẩn cấp'"
{{"objects": ["woman", "television"], "ocr_text": ["khẩn cấp"], "actions": ["talking"], "attributes": ["yellow shirt"], "temporal_relation": "after", "scene": "news studio", "translated_query": "a woman in a yellow shirt talking on a television broadcast"}}

Query: "{query}"
JSON:"""
        raw = _call_ollama(prompt, self.ollama_url, self.ollama_model, json_mode=True, temperature=0.0)
        parsed = _extract_json_object(raw)
        if parsed is not None:
            result = _normalize_parsed_query(parsed, query)
            self.agent_memory["parsed_query"] = result
            log.info(f"[QueryParser] {result}")
            return result

        log.warning("[QueryParser] failed to parse LLM output as JSON, using fallback")
        fallback = _normalize_parsed_query({"ocr_text": [query]}, query)
        self.agent_memory["parsed_query"] = fallback
        return fallback


class _ClipSearchTool(Tool):
    """Shared base for tools that all query the same CLIP/SigLIP branch but
    differ only in which ParsedQuery field they populate and which branch
    key they log results under (search_visual / search_attribute /
    search_scene / search_action in the original file were ~95% identical
    copy-pasted code — this centralizes the one real implementation so a
    fix here doesn't need to be repeated four times).
    """

    inputs = {
        "query": {
            "type": "string",
            "description": "English visual description.",
        }
    }
    output_type = "array"

    _branch_key: str = "clip"
    _parsed_field: str | None = None

    def __init__(self, branch, cfg, branch_results: dict):
        super().__init__()
        self.branch = branch
        self.cfg = cfg
        self.branch_results = branch_results

    def forward(self, query: str) -> list[str]:
        log.info(f"[{self.name}] '{query}'")
        kwargs: dict[str, Any] = {"original_query": query, "translated_query": query}
        if self._parsed_field:
            kwargs[self._parsed_field] = [query]
        p = ParsedQuery(**kwargs)
        top_k = getattr(self.cfg, "clip_top_k", 20)
        try:
            results = _run_async(self.branch.retrieve(p, top_k))
        except Exception as e:
            log.error(f"[{self.name}] retrieve failed: {e}")
            return []
        self.branch_results[self._branch_key] = _merge_dedup(
            self.branch_results.get(self._branch_key, []), results
        )
        n = getattr(self.cfg, "tool_output_top_n", 10)
        return _top_n(results, n)


class SearchVisualTool(_ClipSearchTool):
    name = "search_visual"
    description = (
        "Search video frames by semantic image-text similarity (SigLIP2). "
        "Best for: general visual descriptions, scenes, moods. "
        "Use a short, natural English caption for best results (avoid 'a photo of' "
        "prefixes and temporal words like 'then'/'after' — see query_parser_tool's "
        "translated_query, which is already formatted this way). "
        "Returns a list of dicts, e.g. [{'frame_id': 'L21_V001_frame_000390', 'score': 0.8123}, ...]."
    )
    _branch_key = "clip"
    _parsed_field = None


class SearchTextTool(Tool):
    name = "search_text_in_video"
    description = (
        "Search frames that contain specific text visible in the video (OCR/BM25). "
        "Best for: signs, banners, on-screen text, headlines, captions. "
        "Returns frame IDs with their score."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Text expected to be visible/printed in the video (e.g. 'STOP', 'Bản tin').",
        }
    }
    output_type = "array"

    def __init__(self, bm25_branch, cfg, branch_results: dict):
        super().__init__()
        self.bm25_branch = bm25_branch
        self.cfg = cfg
        self.branch_results = branch_results

    def forward(self, query: str) -> list[str]:
        log.info(f"[search_text_in_video] '{query}'")
        p = ParsedQuery(original_query=query, ocr_text=[query])
        top_k = getattr(self.cfg, "ocr_bm25_top_k", 20)
        try:
            results = _run_async(self.bm25_branch.retrieve(p, top_k))
        except Exception as e:
            log.error(f"[search_text_in_video] retrieve failed: {e}")
            return []
        self.branch_results["ocr_bm25"] = _merge_dedup(self.branch_results.get("ocr_bm25", []), results)
        n = getattr(self.cfg, "tool_output_top_n", 10)
        return _top_n(results, n)


class SearchObjectTool(Tool):
    name = "search_object"
    description = (
        "Search frames containing a specific object detected by YOLO. "
        "WARNING: ONLY use this for the 80 standard COCO classes (e.g., person, car, dog, chair, backpack). "
        "For non-COCO objects (e.g., 'staircase', 'building', 'mountain', 'sky', 'weapon'), DO NOT USE THIS TOOL! Use search_visual instead."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "COCO object name in English or Vietnamese (e.g. 'car', 'person', 'xe máy').",
        }
    }
    output_type = "array"

    def __init__(self, obj_branch, cfg, branch_results: dict):
        super().__init__()
        self.obj_branch = obj_branch
        self.cfg = cfg
        self.branch_results = branch_results

    def forward(self, query: str) -> list[str]:
        log.info(f"[search_object] '{query}'")
        p = ParsedQuery(original_query=query, objects=[query])
        top_k = getattr(self.cfg, "object_top_k", 20)
        try:
            results = _run_async(self.obj_branch.retrieve(p, top_k))
        except Exception as e:
            log.error(f"[search_object] retrieve failed: {e}")
            return []
        self.branch_results["object"] = _merge_dedup(self.branch_results.get("object", []), results)
        n = getattr(self.cfg, "tool_output_top_n", 10)
        return _top_n(results, n)


class FusionTool(Tool):
    name = "fusion_tool"
    description = (
        "Fuse results from multiple already-executed search branches using RRF ranking. "
        "Call after running multiple search tools to combine their results. "
        "If branch_names is empty or none of the names match branches that were "
        "actually searched, ALL available branches are fused instead. "
        "WARNING: DO NOT use this tool to re-rank frames. This tool ONLY takes 'branch_names'. "
        "If you want to re-rank frame_ids using a query, you MUST use 'rerank_tool' instead."
    )
    inputs = {
        "branch_names": {
            "type": "array",
            "description": "Branch names to fuse, e.g. ['clip', 'ocr_bm25', 'object']. "
                            "Valid names are the branch keys actually produced by search "
                            "tools you've already called (e.g. clip, ocr_bm25, object, "
                            "clip_after, clip_before, clip_attr, clip_scene, sequence).",
            "nullable": True,
        }
    }
    output_type = "array"

    _DEFAULT_WEIGHT = 1.0

    _WEIGHT_ATTR_MAP = {
        "clip": "weight_clip",
        "ocr_bm25": "weight_ocr_bm25",
        "object": "weight_object",
        "clip_after": "weight_clip_aux",
        "clip_before": "weight_clip_aux",
        "clip_attr": "weight_clip_aux",
        "clip_scene": "weight_clip_aux",
        "sequence": "weight_clip",
    }

    def __init__(self, fusion, cfg, branch_results: dict, agent_memory: dict, fuse_top_k: int = 50):
        super().__init__()
        self.fusion = fusion
        self.cfg = cfg
        self.branch_results = branch_results
        self.agent_memory = agent_memory
        self.fuse_top_k = getattr(cfg, "fusion_top_k", fuse_top_k)

    def _weight_for(self, branch: str) -> float:
        attr = self._WEIGHT_ATTR_MAP.get(branch)
        if attr is None:
            log.warning(f"[fusion_tool] no configured weight for branch '{branch}', using default {self._DEFAULT_WEIGHT}")
            return self._DEFAULT_WEIGHT
        return getattr(self.cfg, attr, self._DEFAULT_WEIGHT)

    def forward(self, branch_names: list[str] = None) -> list[str]:
        if branch_names is None:
            branch_names = []
        elif not isinstance(branch_names, list):
            branch_names = [branch_names]
            
        log.info(f"[fusion_tool] requested branches: {branch_names}")

        requested = set(branch_names or [])
        matched = requested & self.branch_results.keys()
        unmatched = requested - self.branch_results.keys()
        if unmatched:
            log.warning(f"[fusion_tool] unknown/not-yet-searched branch names ignored: {sorted(unmatched)}")

        if matched:
            selected = {k: v for k, v in self.branch_results.items() if k in matched}
        else:
            if requested:
                log.warning("[fusion_tool] none of the requested branch_names matched; fusing ALL available branches instead")
            selected = dict(self.branch_results)

        if not selected:
            log.warning("[fusion_tool] no branch results available yet — did you run a search tool first?")
            return []

        weights = {branch: self._weight_for(branch) for branch in selected}

        try:
            fused = self.fusion.fuse(selected, weights, top_k=self.fuse_top_k)
        except Exception as e:
            log.error(f"[fusion_tool] fusion failed for branches {list(selected)}: {e}")
            return []

        self.agent_memory["fused_results"] = fused
        n = getattr(self.cfg, "tool_output_top_n", 10)
        return _top_n(fused, n)


class RerankTool(Tool):
    name = "rerank_tool"
    description = (
        "Combine and re-rank candidate frames gathered so far. Sorts by "
        "retrieval score; if OCR/object metadata is available, frames whose "
        "visible text or detected objects lexically match `query` get a "
        "small boost. Use this as a final pass before answering."
    )
    inputs = {
        "frame_ids": {
            "type": "array",
            "description": "Candidate frame IDs to re-rank (with or without a trailing score).",
            "nullable": True,
        },
        "query": {
            "type": "string",
            "description": "English description used to compute the lexical match boost.",
            "nullable": True,
        },
    }
    output_type = "array"

    def __init__(self, branch_results: dict, agent_memory: dict, cfg=None, meta_db=None, lexical_boost_weight: float = 0.15):
        super().__init__()
        self.branch_results = branch_results
        self.agent_memory = agent_memory
        self.cfg = cfg
        self.meta_db = meta_db
        self.lexical_boost_weight = lexical_boost_weight
        if meta_db is None:
            log.info("[rerank_tool] no meta_db provided — falling back to pure score sort (query is unused)")

    def _lexical_boost(self, frame_id: str, query_tokens: set[str]) -> float:
        if not self.meta_db or not query_tokens:
            return 0.0
        try:
            frame_meta = _run_async(self.meta_db.get_frame_async(frame_id))
        except Exception as e:
            log.warning(f"[rerank_tool] lexical boost lookup failed for {frame_id}: {e}")
            return 0.0
        if not frame_meta:
            return 0.0
        text_bag = " ".join(frame_meta.get("ocr_text", []) or [])
        text_bag += " " + " ".join(d.get("class_name", "") for d in (frame_meta.get("detections") or []))
        text_tokens = set(re.findall(r"\w+", text_bag.lower()))
        overlap = len(query_tokens & text_tokens)
        return min(overlap, 3) / 3.0

    def forward(self, frame_ids: list[str] = None, query: str = "") -> list[str]:
        if frame_ids is None:
            frame_ids = []
        log.info(f"[rerank_tool] {len(frame_ids)} frames for '{query}'")
        clean_ids = [fid.get("frame_id", "") if isinstance(fid, dict) else fid.split(" ")[0] for fid in frame_ids]

        candidate_scores: dict[str, list[float]] = {}
        candidate_obj: dict[str, RetrievalResult] = {}
        
        # Gather scores from all branches
        for results in self.branch_results.values():
            for r in results:
                if r.frame_id not in candidate_scores:
                    candidate_scores[r.frame_id] = []
                    candidate_obj[r.frame_id] = r
                candidate_scores[r.frame_id].append(r.score)
                if r.score > candidate_obj[r.frame_id].score:
                    candidate_obj[r.frame_id] = r
                    
        # Gather scores from fused results
        for r in self.agent_memory.get("fused_results", []):
            if r.frame_id not in candidate_scores:
                candidate_scores[r.frame_id] = []
                candidate_obj[r.frame_id] = r
            candidate_scores[r.frame_id].append(r.score)
            if r.score > candidate_obj[r.frame_id].score:
                candidate_obj[r.frame_id] = r

        # Calculate average score for each frame
        all_candidates: dict[str, RetrievalResult] = {}
        for fid, scores in candidate_scores.items():
            avg_score = sum(scores) / len(scores)
            obj = candidate_obj[fid]
            all_candidates[fid] = RetrievalResult(
                frame_id=obj.frame_id,
                video_id=obj.video_id,
                frame_idx=obj.frame_idx,
                timestamp=obj.timestamp,
                frame_path=obj.frame_path,
                score=avg_score,
                source=obj.source
            )

        candidates = [all_candidates[fid] for fid in clean_ids if fid in all_candidates]
        missing = [fid for fid in clean_ids if fid not in all_candidates]
        if missing:
            log.warning(f"[rerank_tool] {len(missing)} frame_ids had no known score and were dropped: {missing[:5]}")

        stopwords = {"a", "an", "the", "in", "on", "at", "of", "and", "or", "to", "for", "with", "is", "are", "was", "were", "this", "that", "it", "from", "by", "as"}
        query_tokens = set(re.findall(r"\w+", query.lower())) if query else set()
        query_tokens = query_tokens - stopwords

        def combined_score(r: RetrievalResult) -> float:
            boost = self._lexical_boost(r.frame_id, query_tokens)
            return r.score + self.lexical_boost_weight * boost

        candidates.sort(key=combined_score, reverse=True)
        self.agent_memory["reranked_results"] = [r.frame_id for r in candidates]
        n = getattr(self.cfg, "tool_output_top_n", 20) if self.cfg else 20
        return _top_n(candidates, n)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Temporal & Sequence Tools
# ─────────────────────────────────────────────────────────────────────────────

class FindFramesAfterTool(Tool):
    name = "find_frames_after"
    description = (
        "Find frames appearing AFTER an anchor frame matching a query. "
        "Use for: 'thứ hai', 'tiếp theo', 'sau đó', 'kế tiếp', 'after'. "
        "Automatically extracts video and timestamp from anchor_frame_id."
    )
    inputs = {
        "anchor_frame_id": {
            "type": "string",
            "description": "Frame ID from a previous search (e.g. 'L21_V001_frame_000390').",
        },
        "query": {
            "type": "string",
            "description": "English description of what to find after the anchor.",
        },
    }
    output_type = "array"

    def __init__(self, clip_branch, meta_db, cfg, branch_results: dict, temporal_cache: _TemporalCache):
        super().__init__()
        self.clip_branch = clip_branch
        self.meta_db = meta_db
        self.cfg = cfg
        self.branch_results = branch_results
        self.cache = temporal_cache

    def forward(self, anchor_frame_id: str, query: str) -> list[str]:
        if not anchor_frame_id or not query:
            return []
        log.info(f"[find_frames_after] anchor={anchor_frame_id}, query='{query}'")
        video_id, frame_idx = _parse_frame_id(anchor_frame_id)
        try:
            all_frames = self.cache.get_frames(self.meta_db, video_id)
            subsequent_ids = {f["frame_id"] for f in all_frames if f.get("frame_idx", 0) > frame_idx}
            if not subsequent_ids:
                return []
            p = ParsedQuery(original_query=query, translated_query=query)
            top_k = getattr(self.cfg, "clip_top_k", 20)
            multiplier = getattr(self.cfg, "temporal_search_multiplier", 2)
            results = _run_async(self.clip_branch.retrieve(p, top_k * multiplier))
            filtered = [r for r in results if r.frame_id in subsequent_ids]
            self.branch_results["clip_after"] = _merge_dedup(self.branch_results.get("clip_after", []), filtered)
            log.info(f"find_frames_after: {len(filtered)} frames found after {anchor_frame_id}")
            n = getattr(self.cfg, "tool_output_top_n", 10)
            return _top_n(filtered, n)
        except Exception as e:
            log.error(f"find_frames_after error: {e}")
            return []


class FindFramesBeforeTool(Tool):
    name = "find_frames_before"
    description = (
        "Find frames appearing BEFORE an anchor frame matching a query. "
        "Use for: 'trước đó', 'trước khi', 'before'."
    )
    inputs = {
        "anchor_frame_id": {
            "type": "string",
            "description": "Frame ID from a previous search.",
        },
        "query": {
            "type": "string",
            "description": "English description of what to find before the anchor.",
        },
    }
    output_type = "array"

    def __init__(self, clip_branch, meta_db, cfg, branch_results: dict, temporal_cache: _TemporalCache):
        super().__init__()
        self.clip_branch = clip_branch
        self.meta_db = meta_db
        self.cfg = cfg
        self.branch_results = branch_results
        self.cache = temporal_cache

    def forward(self, anchor_frame_id: str, query: str) -> list[str]:
        if not anchor_frame_id or not query:
            return []
        log.info(f"[find_frames_before] anchor={anchor_frame_id}, query='{query}'")
        video_id, frame_idx = _parse_frame_id(anchor_frame_id)
        try:
            all_frames = self.cache.get_frames(self.meta_db, video_id)
            preceding_ids = {f["frame_id"] for f in all_frames if f.get("frame_idx", 0) < frame_idx}
            if not preceding_ids:
                return []
            p = ParsedQuery(original_query=query, translated_query=query)
            top_k = getattr(self.cfg, "clip_top_k", 20)
            multiplier = getattr(self.cfg, "temporal_search_multiplier", 2)
            results = _run_async(self.clip_branch.retrieve(p, top_k * multiplier))
            filtered = [r for r in results if r.frame_id in preceding_ids]
            self.branch_results["clip_before"] = _merge_dedup(self.branch_results.get("clip_before", []), filtered)
            n = getattr(self.cfg, "tool_output_top_n", 10)
            return _top_n(filtered, n)
        except Exception as e:
            log.error(f"find_frames_before error: {e}")
            return []


class GetNeighborFramesTool(Tool):
    name = "get_neighbor_frames"
    description = (
        "Get frames surrounding a given frame within a time window. "
        "Useful to understand scene context or find the sharpest frame in a scene."
    )
    inputs = {
        "frame_id": {
            "type": "string",
            "description": "Target frame ID.",
        },
    }
    output_type = "array"

    def __init__(self, meta_db, temporal_cache: _TemporalCache, cfg):
        super().__init__()
        self.meta_db = meta_db
        self.cache = temporal_cache
        self.cfg = cfg

    def forward(self, frame_id: str) -> list[str]:
        window_frames = getattr(self.cfg, "temporal_window", 20)
        log.info(f"[get_neighbor_frames] frame={frame_id}, window_frames={window_frames}")
        video_id, frame_idx = _parse_frame_id(frame_id)
        fps = self.cache.get_fps(self.meta_db, video_id)
        try:
            neighbors = _run_async(
                self.meta_db.get_neighbouring_frames_async(video_id, frame_idx, window_frames)
            )
            result_ids = [{"frame_id": n["frame_id"], "score": 0.0} for n in neighbors if "frame_id" in n]
            log.info(f"get_neighbor_frames: {len(result_ids)} neighbors (fps={fps}, window_frames={window_frames})")
            return result_ids[:20]
        except Exception as e:
            log.error(f"get_neighbor_frames error: {e}")
            return []


class SearchSequenceTool(Tool):
    name = "search_sequence"
    description = (
        "Search for a sequence of events: first_query THEN then_query. "
        "Use for: 'người ngã rồi đứng dậy', 'xe dừng rồi quay đầu'."
    )
    inputs = {
        "first_query": {
            "type": "string",
            "description": "English description of the first event.",
        },
        "then_query": {
            "type": "string",
            "description": "English description of the subsequent event.",
        },
    }
    output_type = "array"

    def __init__(self, clip_branch, meta_db, cfg, branch_results: dict, temporal_cache: _TemporalCache):
        super().__init__()
        self.clip_branch = clip_branch
        self.meta_db = meta_db
        self.cfg = cfg
        self.branch_results = branch_results
        self.cache = temporal_cache

    def forward(self, first_query: str, then_query: str) -> list[str]:
        log.info(f"[search_sequence] '{first_query}' → '{then_query}'")
        p1 = ParsedQuery(original_query=first_query, translated_query=first_query)
        top_k = getattr(self.cfg, "clip_top_k", 20)
        try:
            first_results = _run_async(self.clip_branch.retrieve(p1, top_k))
        except Exception as e:
            log.error(f"[search_sequence] first_query retrieve failed: {e}")
            return []
        if not first_results:
            return []

        max_gap = getattr(self.cfg, "sequence_max_gap_frames", None)

        sequence_hits: list[RetrievalResult] = []
        seen_videos: set[str] = set()
        for anchor in first_results[:5]:
            if anchor.video_id in seen_videos:
                continue
            seen_videos.add(anchor.video_id)
            video_id, frame_idx = _parse_frame_id(anchor.frame_id)
            try:
                all_frames = self.cache.get_frames(self.meta_db, video_id)
                subsequent_ids = {
                    f["frame_id"]
                    for f in all_frames
                    if f.get("frame_idx", 0) > frame_idx
                    and (max_gap is None or f.get("frame_idx", 0) - frame_idx <= max_gap)
                }
                p2 = ParsedQuery(original_query=then_query, translated_query=then_query)
                results2 = _run_async(self.clip_branch.retrieve(p2, top_k))
                filtered2 = [r for r in results2 if r.frame_id in subsequent_ids]
                sequence_hits.extend(filtered2[:3])
            except Exception as e:
                log.error(f"search_sequence error for {anchor.frame_id}: {e}")
        sequence_hits.sort(key=lambda r: r.score, reverse=True)
        self.branch_results["sequence"] = _merge_dedup(self.branch_results.get("sequence", []), sequence_hits)
        n = getattr(self.cfg, "tool_output_top_n", 10)
        return _top_n(sequence_hits, n)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Semantic Variant Tools (share _ClipSearchTool, see Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

class SearchAttributeTool(_ClipSearchTool):
    name = "search_attribute"
    description = (
        "Search frames by visual attributes: colors, clothing, size, weather "
        "conditions. Returns frame IDs with their score."
    )
    _branch_key = "clip_attr"
    _parsed_field = "attributes"
    inputs = {
        "query": {
            "type": "string",
            "description": "Attribute in English (e.g. 'red shirt', 'rainy weather', 'large crowd').",
        }
    }


class SearchSceneTool(_ClipSearchTool):
    name = "search_scene"
    description = (
        "Search frames by scene type or location (beach, airport, office, "
        "market, etc.). Returns frame IDs with their score."
    )
    _branch_key = "clip_scene"
    _parsed_field = None
    inputs = {
        "query": {
            "type": "string",
            "description": "Scene type in English (e.g. 'beach', 'busy intersection', 'hospital room').",
        }
    }




# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Meta Tools
# ─────────────────────────────────────────────────────────────────────────────

class SpatialReasoningTool(Tool):
    name = "spatial_reasoning_tool"
    description = (
        "Filter frames by spatial position of an object (left, right, above, below, center). "
        "Uses YOLO bounding boxes from metadata."
    )
    inputs = {
        "frame_ids": {
            "type": "array",
            "description": "Candidate frame IDs to filter.",
        },
        "spatial_relation": {
            "type": "string",
            "description": "One of: 'left', 'right', 'above', 'below', 'center'.",
        },
        "target_object": {
            "type": "string",
            "description": "Object that should be in the spatial position (English COCO name).",
        },
    }
    output_type = "array"

    def __init__(self, meta_db, min_confidence: float = 0.3):
        super().__init__()
        self.meta_db = meta_db
        self.min_confidence = min_confidence

    def forward(self, frame_ids: list[str], spatial_relation: str, target_object: str) -> list[str]:
        log.info(f"[spatial_reasoning_tool] {spatial_relation} {target_object} in {len(frame_ids)} frames")
        clean_ids = [fid.get("frame_id", "") if isinstance(fid, dict) else fid.split(" ")[0] for fid in frame_ids]
        passing = []
        for fid in clean_ids:
            try:
                frame_meta = _run_async(self.meta_db.get_frame_async(fid))
                if not frame_meta:
                    continue
                fw = frame_meta.get("width") or frame_meta.get("frame_width")
                fh = frame_meta.get("height") or frame_meta.get("frame_height")
                for det in frame_meta.get("detections", []):
                    if target_object.lower() not in det.get("class_name", "").lower():
                        continue
                    conf = det.get("confidence", det.get("score", 1.0))
                    if conf < self.min_confidence:
                        continue
                    x1, y1 = det.get("x1", 0), det.get("y1", 0)
                    x2, y2 = det.get("x2", 1), det.get("y2", 1)
                    if fw and fh and (x2 > 1.0 or y2 > 1.0):
                        x1, x2 = x1 / fw, x2 / fw
                        y1, y2 = y1 / fh, y2 / fh
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    match = (
                        (spatial_relation == "left" and cx < 0.4) or
                        (spatial_relation == "right" and cx > 0.6) or
                        (spatial_relation == "above" and cy < 0.4) or
                        (spatial_relation == "below" and cy > 0.6) or
                        (spatial_relation == "center" and 0.3 < cx < 0.7 and 0.3 < cy < 0.7)
                    )
                    if match:
                        passing.append(fid)
                        break
            except Exception as e:
                log.warning(f"spatial_reasoning_tool error for {fid}: {e}")
        return passing[:20]


class ReflectionTool(Tool):
    name = "reflection_tool"
    description = (
        "Reflect on search strategy and suggest concrete next steps. When the "
        "session's branch results are available, this reads the ACTUAL best "
        "score achieved so far instead of relying on your self-reported estimate."
    )
    inputs = {
        "search_attempts": {
            "type": "string",
            "description": "Comma-separated list of tools already tried.",
        },
        "current_best_score": {
            "type": "number",
            "description": "Your estimate of the best score seen so far — only used as a "
                            "fallback if no branch data is available in this session.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, branch_results: dict | None = None, cfg=None):
        super().__init__()
        self.branch_results = branch_results
        self.cfg = cfg

    def _actual_best_score(self) -> float | None:
        if not self.branch_results:
            return None
        best, found = 0.0, False
        for results in self.branch_results.values():
            for r in results:
                found = True
                best = max(best, getattr(r, "score", 0.0))
        return best if found else None

    def forward(self, search_attempts, current_best_score: float = 0.0) -> str:
        if isinstance(search_attempts, list):
            tried = {str(t).strip() for t in search_attempts}
        else:
            tried = {t.strip() for t in str(search_attempts).split(",") if t.strip()}

        actual_score = self._actual_best_score()
        best_score = actual_score if actual_score is not None else current_best_score
        score_source = "measured" if actual_score is not None else "self-reported"

        low = getattr(self.cfg, "reflection_low_score", 0.2) if self.cfg else 0.2
        mid = getattr(self.cfg, "reflection_mid_score", 0.3) if self.cfg else 0.3

        suggestions = []
        if self.branch_results is not None and not self.branch_results:
            suggestions.append("No search has been run yet — start with search_visual or query_parser_tool")
        if "search_visual" not in tried:
            suggestions.append("Try search_visual with a more specific English description")
        if "search_text_in_video" not in tried:
            suggestions.append("Try search_text_in_video for any visible text")
        if "search_object" not in tried:
            suggestions.append("Try search_object with specific COCO object names")
        if best_score < low:
            suggestions.append("Reformulate with simpler, more visual terms")
            suggestions.append("Try search_scene to find location context first")
        elif best_score < mid and "fusion_tool" not in tried:
            suggestions.append("Use fusion_tool to combine all branch results")
        if not suggestions:
            suggestions.append("Results look reasonable — consider rerank_tool for a final pass")

        advice = f"(best_score={best_score:.3f}, {score_source}) " + "; ".join(suggestions)
        log.info(f"[reflection_tool] {advice}")
        return advice


class MemorySaveTool(Tool):
    name = "memory_save"
    description = "Save an intermediate result to agent memory for later use."
    inputs = {
        "key": {"type": "string", "description": "Short identifier (e.g. 'first_news_frame')."},
        "value": {"type": "string", "description": "Value to save (e.g. a frame_id)."},
    }
    output_type = "string"

    def __init__(self, agent_memory: dict):
        super().__init__()
        self.agent_memory = agent_memory

    def forward(self, key: str, value: str) -> str:
        self.agent_memory[key] = value
        log.info(f"[memory_save] {key} = {value}")
        return f"Saved: {key} = {value}"


class MemoryGetTool(Tool):
    name = "memory_get"
    description = "Retrieve a previously saved value from agent memory."
    inputs = {
        "key": {"type": "string", "description": "The key to retrieve."},
    }
    output_type = "string"

    def __init__(self, agent_memory: dict):
        super().__init__()
        self.agent_memory = agent_memory

    def forward(self, key: str) -> str:
        value = self.agent_memory.get(key, "Not found")
        log.info(f"[memory_get] {key} → {value}")
        return str(value)


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def build_tools(
    clip_branch,
    bm25_branch,
    obj_branch,
    fusion,
    meta_db,
    vector_db,
    embedding_svc,
    cfg,
    branch_results: dict,
    ollama_url: str,
    ollama_model: str,
) -> list:
    """
    Instantiate all Tool classes and return a list for smolagents.
    Uses class-based Tools to avoid the inspect.getsource() indent bug.

    IMPORTANT: branch_results (passed in) and the internal agent_memory dict
    are shared, mutable state across every tool call in a session. Call
    reset_session(branch_results, agent_memory) from the host app before
    starting a new, unrelated user query if you reuse these dicts — otherwise
    fusion/rerank/reflection will mix results from the previous question in.

    NOTE: vector_db and embedding_svc are currently unused by any tool here
    — kept as constructor parameters for forward-compatibility (e.g. a future
    dense-memory or vector-cache tool) rather than removed, so this function
    signature doesn't have to change again when that lands.
    """
    agent_memory: dict[str, Any] = {}
    temporal_cache = _TemporalCache(default_fps=getattr(cfg, "default_fps", 25.0))

    return [
        # Phase 1: Core
        QueryParserTool(agent_memory, ollama_url, ollama_model),
        SearchVisualTool(clip_branch, cfg, branch_results),
        SearchTextTool(bm25_branch, cfg, branch_results),
        SearchObjectTool(obj_branch, cfg, branch_results),
        FusionTool(fusion, cfg, branch_results, agent_memory),
        RerankTool(branch_results, agent_memory, cfg=cfg, meta_db=meta_db),
        # Phase 2: Temporal & Sequence
        FindFramesAfterTool(clip_branch, meta_db, cfg, branch_results, temporal_cache),
        FindFramesBeforeTool(clip_branch, meta_db, cfg, branch_results, temporal_cache),
        GetNeighborFramesTool(meta_db, temporal_cache, cfg),
        SearchSequenceTool(clip_branch, meta_db, cfg, branch_results, temporal_cache),
        # Phase 3: Semantic Variants
        SearchAttributeTool(clip_branch, cfg, branch_results),
        SearchSceneTool(clip_branch, cfg, branch_results),
        # Phase 4: Meta
        SpatialReasoningTool(meta_db, min_confidence=getattr(cfg, "spatial_min_confidence", 0.3)),
        ReflectionTool(branch_results=branch_results, cfg=cfg),
        MemorySaveTool(agent_memory),
        MemoryGetTool(agent_memory),
    ]