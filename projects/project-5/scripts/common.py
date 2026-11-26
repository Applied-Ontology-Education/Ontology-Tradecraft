"""Shared paths, constants, cached LLM client and ROBOT helper for the Project 5 pipeline."""
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

ROOT = Path(__file__).resolve().parents[1]          # projects/project-5
SRC, DATA, GEN, REPORTS = ROOT / "src", ROOT / "data", ROOT / "generated", ROOT / "reports"
MODULE = SRC / "cco-module.ttl"                     # original selected CCO module (FacilityOntology)
MODULE_ALIAS = SRC / "module.ttl"                   # README step 1 names this path; identical copy
CATALOG = SRC / "catalog-v001.xml"
IMPORTS = SRC / "imports"

CCO = "https://www.commoncoreontologies.org/"
OBO = "http://purl.obolibrary.org/obo/"
BEARER_OF = OBO + "BFO_0000196"                      # bearer of
ARTIFACT_FUNCTION = CCO + "ont00000323"              # Artifact Function
SKOS_DEF = "http://www.w3.org/2004/02/skos/core#definition"

SEEDS = [1, 2, 3]
TAU_GRID = [round(0.60 + 0.01 * i, 2) for i in range(21)]   # 0.60 .. 0.80
W_COS, W_LLM = 0.7, 0.3

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


def ask_json(system: str, prompt: str, max_tokens: int = 800):
    """Deterministic (temperature 0) Claude call returning parsed JSON. Cached by prompt hash,
    so re-runs are reproducible and need no API key once data/llm_cache.json exists."""
    global _client
    key = hashlib.sha256(f"{LLM_MODEL}\n{system}\n{prompt}".encode()).hexdigest()
    cache = _load_cache()
    if key in cache:
        return cache[key]["json"]
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set and the LLM cache has no entry for this prompt.")
    import anthropic
    if _client is None:
        _client = anthropic.Anthropic()
    kwargs = dict(model=LLM_MODEL, max_tokens=max_tokens, system=system,
                  messages=[{"role": "user", "content": prompt}], temperature=0)
    for attempt in range(5):
        try:
            try:
                msg = _client.messages.create(**kwargs)
            except (anthropic.BadRequestError, TypeError) as e:          # some models reject temperature
                if "temperature" not in str(e):
                    raise
                kwargs.pop("temperature")
                msg = _client.messages.create(**kwargs)
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            out = _parse_json(text)
            break
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError):
            time.sleep(2 ** attempt * 3)
        except ValueError:
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
