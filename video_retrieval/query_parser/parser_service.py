import json
import requests
from typing import Any, Optional
from config.retrieval_config import RetrievalConfig
from interfaces.base_interfaces import ParsedQuery
from utils.logging_utils import get_logger

log = get_logger(__name__)

class LLMQueryParser:
    def __init__(self, config: RetrievalConfig):
        self.config = config

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a Vietnamese video retrieval query using Qwen2.5:7B via Ollama.
        """
        log.info(f"Parsing query using Ollama ({self.config.ollama_model}): '{query}'")
        
        prompt = f"""You are a Video Retrieval Query Parser. You ONLY respond in ENGLISH. NEVER use Chinese, Japanese, Korean or any non-Latin characters in ANY field.

CRITICAL LANGUAGE RULE: All output values MUST be written in plain English only (except ocr_text which can be Vietnamese). If you write Chinese characters anywhere, your output is WRONG.

Parse the following Vietnamese query into this exact JSON object:
{{
  "expanded_queries": [
    "<English-only detailed visual caption, max 30 words, present tense, no 'a photo of'>",
    "<English-only caption rephrased focusing on different synonyms>",
    "<English-only caption rephrased focusing on actions or attributes>"
  ],
  "objects": ["<COCO class in English only>"],
  "ocr_text": ["<visible on-screen text kept in original language>"],
  "temporal_relation": "none|then|before|after",
  "second_event_query": "<English-only caption of second event, or empty string>"
}}

COCO classes (ONLY pick from this list for "objects"):
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

Translation rules:
1. expanded_queries: Generate up to 3 diverse English captions for the main visual scene. No temporal words.
2. Clothing: "áo" → shirt/top/t-shirt/jacket. NEVER "dress" (only for "váy"/"đầm").
3. Objects: map to exact COCO names in English. If no match, leave objects as empty list [].
4. ocr_text: keep original language as-is (Vietnamese/numbers/etc).
5. Sequences: if two events described, set temporal_relation and fill second_event_query in English.
6. IMPORTANT: Do NOT output Chinese characters. Do NOT add explanations. Output ONLY the raw JSON.

Vietnamese query: "{query}"
JSON:"""

        try:
            raw_text = self._call_llm(prompt).strip()
            
            # Extract JSON block if LLM added formatting
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1:
                raw_text = raw_text[start:end+1]
                
            data = json.loads(raw_text)
            
            # Filter generic objects to prevent dilution
            generic_classes = {"person", "car", "motorcycle", "bicycle", "bus", "truck", "train", "airplane", "boat", "chair", "table", "dining table", "bench"}
            raw_objects = data.get("objects", [])
            filtered_objects = [obj for obj in raw_objects if obj not in generic_classes]

            expanded_queries = data.get("expanded_queries", [])
            if not expanded_queries and "translated_query" in data:
                expanded_queries = [data["translated_query"]]
            
            # Extract main translated query for backward compatibility
            translated = expanded_queries[0] if expanded_queries else query
            second_event = data.get("second_event_query", "")

            # Sanitize: if Qwen still outputs CJK in translated fields, fall back to raw query
            if self._contains_cjk(translated):
                log.warning(f"CJK detected in translated_query: '{translated}'. Using raw query as fallback.")
                translated = query
                expanded_queries = [query]
            if self._contains_cjk(second_event):
                log.warning(f"CJK detected in second_event_query: '{second_event}'. Clearing it.")
                second_event = ""

            raw_ocr = data.get("ocr_text", data.get("ocr", []))
            cleaned_ocr = [text for text in raw_ocr if not self._contains_cjk(text)]
            
            # Thêm luôn nguyên văn câu truy vấn gốc tiếng Việt vào mảng tìm kiếm để làm màng lưới an toàn
            if query not in expanded_queries:
                expanded_queries.append(query)

            parsed = ParsedQuery(
                original_query=query,
                objects=filtered_objects,
                ocr_text=cleaned_ocr,
                translated_query=translated,
                expanded_queries=expanded_queries,
            )
            # Store second_event_query and temporal_relation
            parsed.relations = [data.get("temporal_relation", "none")]
            if second_event:
                parsed.actions = [second_event]
                
            log.info(f"Successfully parsed query: {parsed.__dict__}")
            return parsed
            
        except Exception as e:
            log.error(f"Ollama parsing failed: {e}. Falling back to default parser.")
            return ParsedQuery(
                original_query=query,
                ocr_text=[query],
                translated_query=query,
            )

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        """Return True if text contains any Chinese/Japanese/Korean characters."""
        for char in text:
            cp = ord(char)
            if (0x4E00 <= cp <= 0x9FFF   # CJK Unified Ideographs
                    or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
                    or 0x20000 <= cp <= 0x2A6DF  # CJK Extension B
                    or 0x3000 <= cp <= 0x303F   # CJK Symbols & Punctuation
                    or 0xFF00 <= cp <= 0xFFEF):  # Fullwidth forms
                return True
        return False

    def _call_llm(self, prompt: str) -> str:
        """Call LLM based on configured provider. Re-reads .env at call time."""
        import os
        from pathlib import Path
        # Re-read .env to pick up live changes without restart
        env_path = Path(__file__).resolve().parent.parent / ".env"
        live_env: dict = {}
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    live_env[k.strip()] = v.strip()

        provider_str = live_env.get("RETRIEVAL_LLM_PROVIDER", "ollama").lower()
        log.info(f"LLM provider (from .env): '{provider_str}'")
        if provider_str == "gemini":
            return self._call_gemini(prompt)
        elif provider_str == "openai":
            return self._call_openai(prompt)
        return self._call_ollama(prompt)

    def _call_gemini(self, prompt: str) -> str:
        api_key = getattr(self.config, "gemini_api_key", None) or __import__("os").environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment or .env file (uncomment RETRIEVAL_GEMINI_API_KEY)")
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.llm_model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": getattr(self.config, "llm_temperature", 0.0),
                "responseMimeType": "application/json"
            }
        }
        import requests
        resp = requests.post(url, json=payload, timeout=getattr(self.config, "llm_timeout", 60.0))
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise ValueError(f"Unexpected response from Gemini: {data}")

    def _call_openai(self, prompt: str) -> str:
        api_key = getattr(self.config, "openai_api_key", None) or __import__("os").environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment")
            
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": self.config.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": getattr(self.config, "llm_temperature", 0.0),
            "response_format": {"type": "json_object"}
        }
        import requests
        resp = requests.post(url, headers=headers, json=payload, timeout=getattr(self.config, "llm_timeout", 60.0))
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama generation endpoint."""
        url = self.config.ollama_base_url.replace("/v1", "") + "/api/generate"
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0}
        }
        resp = requests.post(url, json=payload, timeout=getattr(self.config, "llm_timeout", 60.0))
        resp.raise_for_status()
        return resp.json().get("response", "")
