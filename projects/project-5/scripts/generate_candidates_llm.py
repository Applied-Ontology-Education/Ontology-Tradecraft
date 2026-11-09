import argparse
import csv
import hashlib
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import anthropic
import rdflib
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from rdflib import OWL, RDF, RDFS, BNode, URIRef

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config.yaml"


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_graph(path):
    g = rdflib.Graph()
    g.parse(path, format="turtle")
    return g


def norm(label):
    """Lowercase and collapse whitespace so 'Social  Act' matches 'social act'."""
    return " ".join(str(label).lower().split())


def build_class_index(*graphs):
    """normalized label -> IRI, plus IRI -> original label (for the prompt)."""
    index, label_of = {}, {}
    for g in graphs:
        for c in g.subjects(RDF.type, OWL.Class):
            if not isinstance(c, URIRef):
                continue
            for lab in g.objects(c, RDFS.label):
                key = norm(lab)
                if key in index and index[key] != str(c):
                    raise ValueError(f"label '{lab}' is used by both {index[key]} and {c}")
                index[key] = str(c)
                label_of[str(c)] = str(lab)
    return index, label_of


def build_property_index(module, bfo, cco_labels):
    """Allowed properties = the ones the module already uses in its restrictions."""
    used = {p for r in module.subjects(RDF.type, OWL.Restriction)
              for p in module.objects(r, OWL.onProperty)}
    index, label_of = {}, {}
    for p in used:
        lab = bfo.value(p, RDFS.label) or cco_labels.get(str(p))
        if lab is None:
            raise ValueError(f"no label found for property {p}")
        index[norm(lab)] = str(p)
        label_of[str(p)] = str(lab)
    return index, label_of


# --- The JSON shape the LLM must return ---------------------------------
class SubclassCandidate(BaseModel):
    superclass: str
    rationale: str

class SomeCandidate(BaseModel):
    property: str
    filler: str
    rationale: str

class Candidates(BaseModel):
    subclass_of: List[SubclassCandidate]
    some_values_from: List[SomeCandidate]


# --- The prompt -----------------------------------------------------------
SYSTEM_TEMPLATE = """You turn textual definitions from the Common Core Ontologies (CCO) Event module into candidate OWL 2 EL axioms.

For the class you are given, propose at most {max_n} axioms in total, of two kinds:
- subclass_of: "<class> is a subclass of <superclass>"
- some_values_from: "<class> is a subclass of (<property> some <filler>)"

Rules:
1. Propose only what the definition itself states. Do not add background knowledge. Zero axioms is a fine answer.
2. Every axiom must hold for every instance of the class, not just typical ones.
3. Do not repeat the class's current parents.
4. If the definition says "designed to X", do not turn X into a "realizes", "bearer of" or participation axiom: a broken or unused thing still counts.
5. Use labels exactly as written in the lists below. If a needed class or property is not in the lists, leave that axiom out.
6. Give a one-sentence rationale quoting the part of the definition it comes from.

Allowed properties:
{properties}

Allowed classes:
{classes}"""

USER_TEMPLATE = """Class: {label}
Current parents: {parents}
Definition: {definition}"""


def make_system_prompt(class_label, prop_label, max_n):
    return SYSTEM_TEMPLATE.format(
        max_n=max_n,
        properties="\n".join(sorted(prop_label.values(), key=str.lower)),
        classes="\n".join(sorted(class_label.values(), key=str.lower)),
    )


def named_parents(module, iri, class_label):
    return sorted(class_label.get(str(o), str(o))
                  for o in module.objects(URIRef(iri), RDFS.subClassOf)
                  if isinstance(o, URIRef))


# --- The call ------------------------------------------------------------
def ask_llm(client, ccfg, system_prompt, user_msg):
    response = client.beta.messages.parse(
        model=ccfg["model"],
        max_tokens=16000,
        output_config={"effort": ccfg["effort"]},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                       # retry on another model if a safety filter declines
        system=[{"type": "text", "text": system_prompt,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_msg}],
        output_format=Candidates,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError(f"no parsed output (stop_reason={response.stop_reason})")
    return response.parsed_output.model_dump(), response.usage


def generate_all(rows, module, class_label, prop_label, ccfg, cache_path):
    load_dotenv()                                  # ANTHROPIC_API_KEY from .env
    client = anthropic.Anthropic(max_retries=5)
    model = ccfg["model"]
    system_prompt = make_system_prompt(class_label, prop_label, ccfg["max_candidates_per_class"])

    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    errors = []
    usage = {"calls": 0, "input": 0, "cache_write": 0, "cache_read": 0, "output": 0}
    lock = threading.Lock()

    def work(row):
        parents = named_parents(module, row["IRI"], class_label)
        user_msg = USER_TEMPLATE.format(label=row["label"],
                                        parents=", ".join(parents) or "(none)",
                                        definition=row["definition_enriched"])
        key = hashlib.sha256((model + system_prompt + user_msg).encode("utf-8")).hexdigest()
        if key not in cache:
            try:
                data, u = ask_llm(client, ccfg, system_prompt, user_msg)
            except RuntimeError as e:
                errors.append((row["IRI"], str(e)))
                return row["IRI"], None
            cache[key] = {"iri": row["IRI"], **data}
            with lock:
                usage["calls"] += 1
                usage["input"] += u.input_tokens
                usage["cache_write"] += u.cache_creation_input_tokens or 0
                usage["cache_read"] += u.cache_read_input_tokens or 0
                usage["output"] += u.output_tokens
        return row["IRI"], cache[key]

    try:
        with ThreadPoolExecutor(max_workers=ccfg["workers"]) as pool:
            results = dict(pool.map(work, rows))
    finally:                                        # save even if the run crashes
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
    return results, errors, usage


def print_usage(usage, prices, n_total):
    cost = sum(usage[k] * prices[k] for k in ("input", "cache_write", "cache_read", "output")) / 1e6
    print(f"New LLM calls: {usage['calls']}  tokens in={usage['input']} "
          f"cache_write={usage['cache_write']} cache_read={usage['cache_read']} out={usage['output']}")
    print(f"Estimated cost of these calls: ${cost:.2f}")
    if usage["calls"]:
        print(f"Projected cost for all {n_total} classes: ${cost / usage['calls'] * n_total:.2f}")


# --- Turning answers into axioms -----------------------------------------
def asserted_axioms(module):
    """Named superclasses and (property, filler) existentials already in the module."""
    subs, somes = set(), set()
    for a, b in module.subject_objects(RDFS.subClassOf):
        if isinstance(b, URIRef):
            subs.add((str(a), str(b)))
        elif (b, RDF.type, OWL.Restriction) in module:
            p, f = module.value(b, OWL.onProperty), module.value(b, OWL.someValuesFrom)
            if isinstance(f, URIRef):
                somes.add((str(a), str(p), str(f)))
    return subs, somes


def build_candidates(results, class_index, prop_index, module, max_n):
    """Resolve labels, apply the checks, and return (kept axioms, log rows)."""
    subs, somes = asserted_axioms(module)
    kept, log, seen = [], [], set()
    for iri, ans in sorted(results.items()):
        if ans is None:
            continue
        proposals = ([("subclass", None, c["superclass"], c["rationale"]) for c in ans["subclass_of"]] +
                     [("some", c["property"], c["filler"], c["rationale"]) for c in ans["some_values_from"]])
        for i, (kind, prop_lab, target_lab, why) in enumerate(proposals):
            target = class_index.get(norm(target_lab))
            prop = prop_index.get(norm(prop_lab)) if prop_lab else None
            key = (iri, prop, target)
            if i >= max_n:
                status = "over_cap"
            elif target is None or (kind == "some" and prop is None):
                status = "not_in_vocabulary"
            elif target == iri:
                status = "self_reference"
            elif (kind == "subclass" and (iri, target) in subs) or \
                 (kind == "some" and (iri, prop, target) in somes):
                status = "already_asserted"
            elif key in seen:
                status = "duplicate"
            else:
                status = "kept"
                seen.add(key)
                kept.append(key)
            log.append({"IRI": iri, "kind": kind, "property": prop_lab or "",
                        "target": target_lab, "status": status, "rationale": why})
    return kept, log


def write_candidates_ttl(kept, path):
    g = rdflib.Graph()
    g.bind("owl", OWL)
    for a, p, b in kept:
        g.add((URIRef(a), RDF.type, OWL.Class))
        g.add((URIRef(b), RDF.type, OWL.Class))
        if p is None:
            g.add((URIRef(a), RDFS.subClassOf, URIRef(b)))
        else:
            r = BNode()
            g.add((URIRef(p), RDF.type, OWL.ObjectProperty))
            g.add((r, RDF.type, OWL.Restriction))
            g.add((r, OWL.onProperty, URIRef(p)))
            g.add((r, OWL.someValuesFrom, URIRef(b)))
            g.add((URIRef(a), RDFS.subClassOf, r))
    path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(path, format="turtle")


def write_log(log, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["IRI", "kind", "property", "target", "status", "rationale"])
        w.writeheader()
        w.writerows(log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="only process the first N classes (for testing)")
    args = parser.parse_args()

    cfg = load_config()
    paths, ccfg = cfg["paths"], cfg["candidates"]

    module = load_graph(PROJECT_DIR / paths["module"])
    bfo = load_graph(PROJECT_DIR / paths["bfo"])

    class_index, class_label = build_class_index(module, bfo)
    prop_index, prop_label = build_property_index(module, bfo, ccfg["cco_property_labels"])
    print(f"Vocabulary: {len(class_index)} classes, {len(prop_index)} properties")

    module_classes = {str(c) for c in module.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef)}
    with open(PROJECT_DIR / paths["definitions_enriched"], newline="", encoding="utf-8") as f:
        all_rows = [r for r in csv.DictReader(f)
                    if r["IRI"] in module_classes and r["definition_enriched"].strip()]
    rows = all_rows[:args.limit] if args.limit else all_rows

    results, errors, usage = generate_all(rows, module, class_label, prop_label, ccfg,
                                          PROJECT_DIR / paths["llm_cache_candidates"])
    n = sum(len(r["subclass_of"]) + len(r["some_values_from"]) for r in results.values() if r)
    print(f"Classes processed: {len(results)}  raw candidates: {n}  errors: {len(errors)}")
    for iri, msg in errors:
        print(f"  error: {iri}: {msg}")
    print_usage(usage, ccfg["price_per_mtok"], len(all_rows))

    kept, log = build_candidates(results, class_index, prop_index, module,
                                 ccfg["max_candidates_per_class"])
    write_candidates_ttl(kept, PROJECT_DIR / paths["candidates"])
    write_log(log, PROJECT_DIR / paths["candidates_log"])
    counts = Counter(r["status"] for r in log)
    print(f"Candidates kept: {len(kept)}  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
