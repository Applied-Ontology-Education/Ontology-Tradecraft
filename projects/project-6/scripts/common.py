"""Shared paths, namespaces, cached LLM client and ROBOT helper for the Project 6 pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # projects/project-6
RAW, PATTERNS, ONTO, GEN = ROOT / "data" / "raw", ROOT / "patterns", ROOT / "ontology", ROOT / "generated"
SHACL_DIR, SPARQL_DIR, REPORTS, DATA = ROOT / "shacl", ROOT / "sparql", ROOT / "reports", ROOT / "data"
CATALOG, IMPORTS = ONTO / "catalog-v001.xml", ONTO / "imports"
SPEC_FILE = PATTERNS / "gtfs_patterns.json"
BASE_EXT = ONTO / "p6_extension_base.ttl"          # approved local classes/properties (from the spec)
EXT = ONTO / "p6_extension.ttl"                    # base + reasoner-accepted LLM axioms
INSTANCES = DATA / "instances.ttl"
SLIM = ONTO / "cco_slim.ttl"

P6 = "https://stucker7-buffalo-edu.github.io/ontology-tradecraft/p6/"
CCO = "https://www.commoncoreontologies.org/"
OBO = "http://purl.obolibrary.org/obo/"
OFFLINE = os.environ.get("P6_OFFLINE") == "1"       # CI: never call the API, cache only

# ---------------------------------------------------------------- LLM (Claude API) with on-disk cache
LLM_MODEL = os.environ.get("P5_LLM_MODEL", "claude-sonnet-5-5")
CACHE_FILE = DATA / "llm_cache.json"
_lock = threading.Lock()
_cache: dict | None = None
_client = None


def _load_cache() -> dict:
    global _cache
    with _lock:
        if _cache is None:
            _cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
        return _cache


def save_cache() -> None:
    cache = _load_cache()
    with _lock:
        CACHE_FILE.write_text(json.dumps(cache, indent=1, sort_keys=True, ensure_ascii=False))


def _parse_json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"LLM reply had no JSON object: {text[:200]!r}")
    return json.loads(m.group(0))


def _call(kwargs):
    """Large budgets are streamed (required by the API for long outputs; leaves room for model thinking)."""
    if kwargs["max_tokens"] > 8000:
        with _client.messages.stream(**kwargs) as st:
            return st.get_final_message()
    return _client.messages.create(**kwargs)


def ask_text(system: str, prompt: str, max_tokens: int = 1500) -> str:
    """Like ask_json but returns raw text (for replies that carry code, where JSON escaping is fragile)."""
    return ask_json(system, prompt, max_tokens, raw=True)["text"]


def ask_json(system: str, prompt: str, max_tokens: int = 1500, raw: bool = False):
    """Deterministic (temperature 0) Claude call returning parsed JSON. Cached by prompt hash,
    so re-runs are reproducible and need no API key once data/llm_cache.json exists."""
    global _client
    key = hashlib.sha256(f"{LLM_MODEL}\n{system}\n{prompt}{chr(10) + 'raw' if raw else ''}".encode()).hexdigest()
    cache = _load_cache()
    if key in cache:
        return cache[key]["json"]
    if OFFLINE or not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("LLM cache miss and no API access (P6_OFFLINE=1 or ANTHROPIC_API_KEY unset). "
                         "Run once locally with the key to populate data/llm_cache.json.")
    import anthropic
    if _client is None:
        _client = anthropic.Anthropic()
    kwargs = dict(model=LLM_MODEL, max_tokens=max_tokens, system=system,
                  messages=[{"role": "user", "content": prompt}], temperature=0)
    for attempt in range(5):
        try:
            try:
                msg = _call(kwargs)
            except (anthropic.BadRequestError, TypeError) as e:  # some SDKs/models reject temperature
                if "temperature" not in str(e):
                    raise
                kwargs.pop("temperature")
                msg = _call(kwargs)
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            stop = getattr(msg, "stop_reason", None)
            if stop not in (None, "end_turn"):                # truncated / refused / paused: never cache it
                last_stop = f"stop_reason={stop}, {len(text)} chars"
                raise ValueError(f"incomplete LLM reply ({last_stop})")
            out = {"text": text} if raw else _parse_json(text)
            break
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError):
            time.sleep(2 ** attempt * 3)
        except ValueError as e:
            print(f"ask_json: retry {attempt + 1}/5 after: {str(e)[:120]}", flush=True)
            if attempt == 4:
                raise
    else:
        raise RuntimeError("LLM call failed after retries")
    with _lock:
        cache[key] = {"model": LLM_MODEL, "json": out}
    return out


# ---------------------------------------------------------------- ROBOT
def robot_cmd() -> str:
    for c in (os.environ.get("ROBOT"), shutil.which("robot"), str(Path.home() / "bin" / "robot")):
        if c and Path(c).exists():
            return c
    raise SystemExit("ROBOT not found: put it on PATH, in ~/bin/robot, or set ROBOT=/path/to/robot")


def robot(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [robot_cmd(), "--catalog", str(CATALOG), *args]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"ROBOT failed: {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return p


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def load_spec() -> dict:
    return json.loads(SPEC_FILE.read_text())


def expand(curie: str) -> str:
    pfx, local = curie.split(":", 1)
    return {"p6": P6, "cco": CCO, "obo": OBO, "xsd": "http://www.w3.org/2001/XMLSchema#"}[pfx] + local
