import logging
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from util.logger_config import config

logger = logging.getLogger(__name__)
from pathlib import Path

config(logger)


def _parse_markdown_prompts(md_text: str):
    """Parse a simple Markdown schema into {type: {system, user}}.

    Expected structure (case-insensitive headings):
    ## Class
    ### System
    ...text...
    ### User
    ...text...

    ## Property
    ### System
    ...
    ### User
    ...

    Any text under a subsection continues until the next heading starting
    with '### ' or '## '. Leading/trailing whitespace is trimmed.
    """
    sections = {
        "class": {"system": None, "user": None},
        "property": {"system": None, "user": None},
    }
    cur_top = None  # "class" | "property" | None
    cur_sub = None  # "system" | "user" | None
    buf = []

    def commit():
        nonlocal buf, cur_top, cur_sub
        if cur_top in sections and cur_sub in {"system", "user"}:
            text = "\n".join(buf).strip()
            if text:
                sections[cur_top][cur_sub] = text
        buf = []

    for raw in md_text.splitlines():
        line = raw.rstrip("\n")
        h = line.strip()
        h_lower = h.lower()
        if h_lower.startswith("## ") and not h_lower.startswith("### "):
            # Top-level section
            commit()
            title = h[3:].strip().lower()
            if title.startswith("class"):
                cur_top = "class"
            elif title.startswith("property"):
                cur_top = "property"
            else:
                cur_top = None
            cur_sub = None
            continue
        if h_lower.startswith("### "):
            commit()
            subtitle = h[4:].strip().lower()
            if subtitle.startswith("system"):
                cur_sub = "system"
            elif subtitle.startswith("user"):
                cur_sub = "user"
            else:
                cur_sub = None
            continue
        # Regular content
        if cur_top and cur_sub:
            buf.append(line)
    # Final commit
    commit()

    # Return only filled items (None -> empty string to allow fallbacks outside)
    return {k: {kk: (vv if vv is not None else "") for kk, vv in v.items()} for k, v in sections.items()}


def load_markdown_prompt_templates(cfg_path: Path):
    templates = {}
    try:
        p = Path(cfg_path)
        md_text = p.read_text(encoding="utf-8")
        parsed = _parse_markdown_prompts(md_text)

        for section in ("class", "property"):
            sys_txt = (parsed.get(section, {}).get("system"))
            usr_txt = (parsed.get(section, {}).get("user"))
            templates[section] = {"system": sys_txt, "user": usr_txt}

        logger.info("Loaded Markdown prompt templates from %s.", p)
    except Exception as e:
        logger.warning("Failed to read or parse prompt templates: %s.", e)

    return templates


def build_prompts(llm: ChatOllama, prompt_texts: dict[Any, Any]):
    prompts = {}
    chains = {}
    for t in ("class", "property"):
        t_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_texts[t]["system"]),
            ("user", prompt_texts[t]["user"]),
        ])
        prompts[t] = t_prompt
        chains[t] = t_prompt | llm

    return prompts, chains


def _inject_iris_into_context(ref_ctx: str, label_to_iri: Dict[str, str], target_label: str) -> str:
    out_lines = []
    for line in ref_ctx.splitlines():
        if line.startswith("- ") and ":" in line:
            # Pattern: "- Label: definition"
            try:
                label = line[2:].split(":", 1)[0].strip("'")
                if label != target_label:
                    iri = label_to_iri.get(label.strip("'"), "")
                    if iri:
                        line = f"- {label} <{iri}>:" + line.split(":", 1)[1]
                else:
                    line = None
            except Exception:
                pass
        if line:
            out_lines.append(line)
    return "\n".join(out_lines)
