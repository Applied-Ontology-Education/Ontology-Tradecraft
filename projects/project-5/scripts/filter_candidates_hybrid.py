import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import numpy as np
import rdflib
from dotenv import load_dotenv
from pydantic import BaseModel
from rdflib import OWL, RDF, RDFS, URIRef
from rdflib.collection import Collection

from generate_candidates_llm import (PROJECT_DIR, asserted_axioms, build_class_index,
                                     build_property_index, load_config, load_graph, print_usage,
                                     write_candidates_ttl)


def merged_graph(module, bfo):
    """Module + BFO in one graph, without ontology headers or owl:imports."""
    g = rdflib.Graph()
    for src in (module, bfo):
        headers = set(src.subjects(RDF.type, OWL.Ontology))
        for s, p, o in src:
            if s not in headers and p != OWL.imports:
                g.add((s, p, o))
    return g


def inferred_ancestors(merged):
    """Run ELK through ROBOT; return {class IRI: set of all inferred named superclasses}."""
    robot = shutil.which("robot")
    if robot is None:
        sys.exit("robot not found on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "module_bfo.ttl", Path(tmp) / "inferred.ttl"
        merged.serialize(src, format="turtle")
        subprocess.run([robot, "reason", "--reasoner", "ELK", "--input", str(src),
                        "--axiom-generators", "SubClass", "--include-indirect", "true",
                        # ROBOT's default (true) deletes the indirect links again
                        "--remove-redundant-subclass-axioms", "false",
                        "--output", str(out)], check=True)
        inferred = load_graph(out)
    anc = {}
    for a, b in inferred.subject_objects(RDFS.subClassOf):
        if isinstance(a, URIRef) and isinstance(b, URIRef):
            anc.setdefault(str(a), set()).add(str(b))
    return anc


def existentials(merged):
    """{class IRI: set of (property, filler)} for asserted A ⊑ ∃R.B in module + BFO."""
    out = {}
    for a, b in merged.subject_objects(RDFS.subClassOf):
        if isinstance(a, URIRef) and (b, RDF.type, OWL.Restriction) in merged:
            p, f = merged.value(b, OWL.onProperty), merged.value(b, OWL.someValuesFrom)
            if isinstance(f, URIRef):
                out.setdefault(str(a), set()).add((str(p), str(f)))
    return out


def named_bounds(g, node):
    """Named classes a domain/range expression allows. Unions -> all members;
    intersections -> their named parts; complements -> nothing."""
    if isinstance(node, URIRef):
        return {str(node)}
    for pred in (OWL.unionOf, OWL.intersectionOf):
        lst = g.value(node, pred) if node is not None else None
        if lst is not None:
            return set().union(*(named_bounds(g, m) for m in Collection(g, lst)))
    return set()


def domain_range(bfo, prop_iris, cco_dr):
    dr = {}
    for p in prop_iris:
        if p in cco_dr:
            dr[p] = (set(cco_dr[p]["domain"]), set(cco_dr[p]["range"]))
        else:
            dr[p] = (named_bounds(bfo, bfo.value(URIRef(p), RDFS.domain)),
                     named_bounds(bfo, bfo.value(URIRef(p), RDFS.range)))
    return dr


def falls_under(cls, allowed, anc):
    """True if cls is one of the allowed classes or below one (no constraint = True)."""
    return not allowed or cls in allowed or bool(allowed & anc.get(cls, set()))


def load_embeddings(path):
    e = np.load(path)
    return (dict(zip(e["class_iris"], e["class_vecs"])),
            dict(zip(e["prop_iris"], e["prop_vecs"])))


def hard_check(cand, anc, ex, dr, class_vecs, prop_vecs):
    """Return a rejection reason, or None if the candidate passes."""
    a, p, b = cand
    if a not in class_vecs or b not in class_vecs or (p and p not in prop_vecs):
        return "no_embedding"
    if p is None:                                   # A ⊑ B
        if b in anc.get(a, set()):
            return "already_entailed"
        if a in anc.get(b, set()):
            return "would_create_cycle"
        return None
    for x in anc.get(a, set()) | {a}:               # A ⊑ ∃R.B
        for q, f in ex.get(x, ()):
            if q == p and (f == b or b in anc.get(f, set())):
                return "already_entailed"
    dom, rng = dr[p]
    if not falls_under(a, dom, anc):
        return "domain_violation"
    if not falls_under(b, rng, anc):
        return "range_violation"
    return None


# --- Cosine score ------------------------------------------------------------
def cosine(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def cos_score(cand, class_vecs, prop_vecs):
    """A ⊑ B: cos(A, B).  A ⊑ ∃R.B: cos(A + R, B), matching ELEmbeddings' training."""
    a, p, b = cand
    u = class_vecs[a] + prop_vecs[p] if p else class_vecs[a]
    return cosine(u, class_vecs[b])


# --- LLM plausibility -------------------------------------------------------
class Plausibility(BaseModel):
    score: float
    reason: str


PLAUS_SYSTEM = """You review candidate OWL axioms proposed for the Common Core Ontologies (CCO) Event module, which is built on Basic Formal Ontology (BFO).

Judge how plausible the candidate axiom is, using the class's original CCO definition as the main evidence. Consider:
1. Does the definition support the axiom?
2. Does the axiom hold for every instance of the class, not just typical ones?
3. Does it fit BFO's categories, including the property's domain and range?

Return a score from 0 to 1 (1 = clearly true and supported by the definition, 0 = false or unsupported) and a one-sentence reason."""

PLAUS_USER = """Class: {label}
Original definition: {definition}
Candidate axiom: {sentence}
{prop_note}"""


def describe(cand, class_label, prop_label, dr):
    a, p, b = cand
    if p is None:
        return f"Every {class_label[a]} is a {class_label[b]}.", ""
    dom, rng = dr[p]
    names = lambda s: " or ".join(sorted(class_label.get(x, x) for x in s)) or "unrestricted"
    return (f"Every {class_label[a]} {prop_label[p]} some {class_label[b]}.",
            f"Property '{prop_label[p]}': domain {names(dom)}; range {names(rng)}.")


def score_plausibility(cands, definitions, class_label, prop_label, dr, hcfg, cache_path):
    load_dotenv()
    client = anthropic.Anthropic(max_retries=5)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    scores, errors = {}, []
    usage = {"calls": 0, "input": 0, "cache_write": 0, "cache_read": 0, "output": 0}
    lock = threading.Lock()

    def work(cand):
        sentence, prop_note = describe(cand, class_label, prop_label, dr)
        user_msg = PLAUS_USER.format(label=class_label[cand[0]], definition=definitions[cand[0]],
                                     sentence=sentence, prop_note=prop_note)
        key = hashlib.sha256((hcfg["model"] + PLAUS_SYSTEM + user_msg).encode("utf-8")).hexdigest()
        if key not in cache:
            r = client.beta.messages.parse(
                model=hcfg["model"], max_tokens=16000,
                output_config={"effort": hcfg["effort"]},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                system=PLAUS_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                output_format=Plausibility,
            )
            if r.stop_reason == "refusal" or r.parsed_output is None:
                errors.append((cand, f"no parsed output (stop_reason={r.stop_reason})"))
                return
            cache[key] = {"candidate": list(cand), **r.parsed_output.model_dump()}
            u = r.usage
            with lock:
                usage["calls"] += 1
                usage["input"] += u.input_tokens
                usage["cache_write"] += u.cache_creation_input_tokens or 0
                usage["cache_read"] += u.cache_read_input_tokens or 0
                usage["output"] += u.output_tokens
        hit = cache[key]
        scores[cand] = (min(1.0, max(0.0, hit["score"])), hit["reason"])

    try:
        with ThreadPoolExecutor(max_workers=hcfg["workers"]) as pool:
            list(pool.map(work, cands))
    finally:                                        # save even if the run crashes
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
    return scores, errors, usage


# --- Decision ------------------------------------------------------------------
DECISION_TYPES = ["accepted", "rejected_below_tau", "rejected_llm_floor", "no_llm_score",
                  "already_entailed", "would_create_cycle", "domain_violation",
                  "range_violation", "no_embedding"]


def decide(candidates, hard, cos, llm, tau, hcfg):
    """Accept if combined = w_cos*cos + w_llm*llm >= tau and llm >= floor.
    Returns (accepted candidates, count per decision type)."""
    accepted, decisions = [], Counter({k: 0 for k in DECISION_TYPES})
    for c in candidates:
        if hard[c]:
            decisions[hard[c]] += 1
            continue
        if c not in llm:
            decisions["no_llm_score"] += 1
            continue
        ls = llm[c][0]
        combined = hcfg["w_cos"] * cos[c] + hcfg["w_llm"] * ls
        if ls < hcfg["llm_floor"]:
            decisions["rejected_llm_floor"] += 1
        elif combined < tau:
            decisions["rejected_below_tau"] += 1
        else:
            decisions["accepted"] += 1
            accepted.append(c)
    return accepted, decisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="only LLM-score the first N candidates (for testing)")
    args = parser.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    module = load_graph(PROJECT_DIR / paths["module"])
    bfo = load_graph(PROJECT_DIR / paths["bfo"])

    subs, somes = asserted_axioms(load_graph(PROJECT_DIR / paths["candidates"]))
    candidates = sorted([(a, None, b) for a, b in subs] + list(somes), key=str)

    merged = merged_graph(module, bfo)
    anc = inferred_ancestors(merged)
    ex = existentials(merged)
    dr = domain_range(bfo, {p for _, p, _ in candidates if p}, cfg["candidates"]["cco_property_domain_range"])
    class_vecs, prop_vecs = load_embeddings(PROJECT_DIR / paths["embeddings"])

    hard = {c: hard_check(c, anc, ex, dr, class_vecs, prop_vecs) for c in candidates}
    counts = Counter(r or "passed" for r in hard.values())
    print(f"Candidates: {len(candidates)}  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    hcfg = cfg["hybrid"]
    _, class_label = build_class_index(module, bfo)
    _, prop_label = build_property_index(module, bfo, cfg["candidates"]["cco_property_labels"])
    with open(PROJECT_DIR / paths["definitions"], newline="", encoding="utf-8") as f:
        definitions = {r["IRI"]: r["definition"] for r in csv.DictReader(f)}

    passed = [c for c in candidates if hard[c] is None]
    cos = {c: cos_score(c, class_vecs, prop_vecs) for c in passed}
    to_score = passed[:args.limit] if args.limit else passed
    llm, errors, usage = score_plausibility(to_score, definitions, class_label, prop_label, dr,
                                            hcfg, PROJECT_DIR / paths["llm_cache_plausibility"])

    vals = np.array(list(cos.values()))
    print(f"cos: mean {vals.mean():.3f}  ≥0.43 (can pass with a perfect LLM score): "
          f"{(vals >= 0.43).sum()}/{len(vals)}")
    print(f"LLM scored: {len(llm)}  mean {np.mean([s for s, _ in llm.values()]):.2f}  errors: {len(errors)}")
    print_usage(usage, hcfg["price_per_mtok"], len(passed))

    metrics_path = PROJECT_DIR / paths["mowl_metrics"]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    tau = metrics["chosen_tau"]                    # chosen by train_mowl.py (step 5)

    accepted, decisions = decide(candidates, hard, cos, llm, tau, hcfg)
    write_candidates_ttl(accepted, PROJECT_DIR / paths["accepted"])

    carried = sum(cos[c] < tau for c in accepted)  # accepted only thanks to the LLM score
    metrics["hybrid"] = {
        "w_cos": hcfg["w_cos"],
        "w_llm": hcfg["w_llm"],
        "llm_floor": hcfg["llm_floor"],
        "n_candidates": len(candidates),
        "n_accepted": len(accepted),
        "llm_contribution_rate": round(carried / len(accepted), 4) if accepted else 0.0,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nDecisions: " + "  ".join(f"{k}={decisions[k]}" for k in DECISION_TYPES))
    print(f"Accepted: {len(accepted)}  LLM contribution rate: "
          f"{metrics['hybrid']['llm_contribution_rate']:.1%}  (tau={tau})")
    print(f"Wrote {paths['accepted']} and updated {paths['mowl_metrics']}")


if __name__ == "__main__":
    main()
