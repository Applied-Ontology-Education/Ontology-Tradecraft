from __future__ import annotations

import json as _json
import logging
from typing import Any

import requests

from util.logger_config import config

logger = logging.getLogger(__name__)

config(logger)


def call_llm_over_http(url: str, query: str) -> Any:
    data = {
        "model": "gemma3n",
        "prompt": query,
        "stream": False
    }

    ollama_raw_response = requests.post(url, json=data)
    ollama_json_response = extract_json(ollama_raw_response.text)
    if "error" in ollama_json_response:
        logger.error(ollama_json_response["error"])
        raise Exception(ollama_json_response["error"])
    prompt_json_response = extract_json(ollama_json_response.get("response", ""))
    return prompt_json_response


def strip_wrapping_quotes(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    # remove matching leading/trailing quotes/backticks
    if (s.startswith("\"") and s.endswith("\"")) or (s.startswith("'") and s.endswith("'")) or (
        s.startswith("```") and s.endswith("```")
    ):
        return s[1:-1] if len(s) >= 2 and s[0] != "`" else s.strip("`")
    return s


def extract_json(text: str | Any) -> Any:
    try:
        obj = _json.loads(text)
    except Exception:
        # Robustly handle Markdown code fences and stray text around JSON
        cleaned = str(text).strip()
        # If response is fenced as ```json ... ``` (or ``` ... ```), remove fences
        if cleaned.startswith("```"):
            # Drop the opening fence line (which may include a language tag like 'json')
            nl = cleaned.find("\n")
            if nl != -1:
                cleaned = cleaned[nl + 1:]
            # Remove trailing closing fence if present
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        # Try direct parse again after fence removal
        try:
            obj = _json.loads(cleaned)
        except Exception:
            # Fallback: extract the JSON object substring between the first '{' and last '}'
            l = cleaned.find("{")
            r = cleaned.rfind("}")
            if l != -1 and r != -1 and r > l:
                snippet = cleaned[l: r + 1]
                obj = _json.loads(snippet)
            else:
                # Re-raise to be handled by outer exception block
                raise
    return obj
