"""Step 6: hybrid filter -> generated/accepted_el.ttl.

score = 0.7 * MOWL cosine + 0.3 * LLM plausibility (0-1); keep a candidate when score >= tau (from step 5).
Cosine: pattern A (Sub, Super) from M1F; pattern B (Sub, Function) from M2F (functions exist only there);
both are the step-5 models refit on the full module, averaged over seeds.
Accepted axioms are then checked together with the module and its imports by ELK (via ROBOT); any axiom
that makes the ontology inconsistent or makes a class unsatisfiable is dropped.
LLM contribution rate = share of accepted axioms whose cosine alone would not have reached tau."""
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from rdflib import Graph, Namespace
from rdflib import URIRef
from rdflib.namespace import OWL, SKOS

sys.path.insert(0, str(Path(__file__).parent))
from common import GEN, IMPORTS, MODULE, REPORTS, W_COS, W_LLM, ask_json, robot, save_cache, write_json  # noqa: E402
from generate_candidates_llm import write_axioms  # noqa: E402

SYSTEM = "You are a careful reviewer of OWL axioms for the BFO-based Common Core Ontologies. Reply with JSON only."
PROMPT = """Rate how plausible this candidate axiom is, i.e. whether it is true of EVERY instance of the subject class.

Subject class: {sub} -- {sub_def}
Candidate axiom: {axiom}
Target class: {tgt} -- {tgt_def}

Score 0.0 (false or unsupported) to 1.0 (certainly true for every instance).
Reply with exactly: {{"plausibility": <number 0-1>, "reason": "<one sentence, under 30 words>"}}"""


def cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n else 0.0


def ensemble_cos(models, s, o):
    vals = [cos(m["classes"][s], m["classes"][o]) for m in models if s in m["classes"] and o in m["classes"]]
    return float(np.mean(vals)) if vals else None


def elk_ok(recs) -> tuple[bool, str]:
    g = Graph(); write_axioms(g, recs)
    tmp = GEN / "_elk_check.ttl"; g.serialize(tmp, format="turtle")
    p = robot("merge", "--input", str(MODULE), "--input", str(tmp), "reason", "--reasoner", "ELK", check=False)
    tmp.unlink(missing_ok=True)
    return p.returncode == 0, (p.stdout + p.stderr)[-400:]


def main():
    metrics = json.loads((REPORTS / "mowl_metrics.json").read_text())
    tau = metrics["tau"]
    if tau is None:
        raise SystemExit("No tau: mean_cos < 0.70 in step 5, so the hybrid filter has no valid threshold.")
    E = json.loads((GEN / "embeddings.json").read_text())
    cands = list(csv.DictReader(open(GEN / "candidates.csv")))
    g = Graph()
    for f in [MODULE, *IMPORTS.glob("*.ttl")]:
        g.parse(f)
    d = lambda i: " ".join(str(g.value(URIRef(i), SKOS.definition) or "(no definition)").split())

    def work(c):
        if c["pattern"] == "A":
            axiom = f"{c['sub_label']} SubClassOf {c['target_label']}"
            cs = ensemble_cos(E["M1F"], c["sub_iri"], c["target_iri"]); model = "M1F"
        else:
            axiom = f"{c['sub_label']} SubClassOf (bearer of) some {c['target_label']}"
            cs = ensemble_cos(E["M2F"], c["sub_iri"], c["target_iri"]); model = "M2F"
        out = ask_json(SYSTEM, PROMPT.format(sub=c["sub_label"], sub_def=d(c["sub_iri"]), axiom=axiom,
                                             tgt=c["target_label"], tgt_def=d(c["target_iri"])), max_tokens=800)
        llm = min(1.0, max(0.0, float(out.get("plausibility", 0))))
        cs0 = cs if cs is not None else 0.0
        score = W_COS * cs0 + W_LLM * llm
        return {**c, "axiom": axiom, "cos_model": model, "cosine": round(cs0, 4), "llm_plausibility": round(llm, 3),
                "llm_reason": out.get("reason", ""), "hybrid_score": round(score, 4),
                "passes_tau": score >= tau, "cosine_alone_passes": cs0 >= tau}

    with ThreadPoolExecutor(8) as ex:
        scored = list(ex.map(work, cands))
    save_cache()

    passing = [r for r in scored if r["passes_tau"]]
    ok, log = elk_ok(passing)
    dropped = []
    if not ok:                                   # isolate offenders one by one
        keep = []
        for r in sorted(passing, key=lambda r: -r["hybrid_score"]):
            if elk_ok(keep + [r])[0]:
                keep.append(r)
            else:
                dropped.append(r)
        passing = keep
    acc_ids = {r["id"] for r in passing}
    for r in scored:
        r["decision"] = ("accepted" if r["id"] in acc_ids else
                         "rejected: ELK inconsistency/unsatisfiable" if r in dropped else
                         "rejected: below tau")
        r["accepted_because_of_llm"] = r["id"] in acc_ids and not r["cosine_alone_passes"]

    out = Graph(); out.bind("owl", OWL); out.bind("obo", Namespace("http://purl.obolibrary.org/obo/"))
    out.bind("cco", Namespace("https://www.commoncoreontologies.org/"))
    write_axioms(out, passing)
    out.serialize(GEN / "accepted_el.ttl", format="turtle")
    cols = ["id", "pattern", "axiom", "cos_model", "cosine", "llm_plausibility", "hybrid_score", "decision",
            "accepted_because_of_llm", "evidence", "llm_reason", "sub_iri", "target_iri"]
    with open(REPORTS / "hybrid_review.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        w.writerows(sorted(scored, key=lambda r: -r["hybrid_score"]))

    n_acc = len(passing)
    by_llm = sum(r["accepted_because_of_llm"] for r in scored)
    pat = lambda p, rs: sum(r["pattern"] == p for r in rs)
    metrics["hybrid"] = {
        "weights": {"cosine": W_COS, "llm": W_LLM}, "tau": tau, "llm_model": __import__("common").LLM_MODEL,
        "candidates": len(scored), "accepted": n_acc,
        "accepted_by_pattern": {"A_subclass": pat("A", passing), "B_function": pat("B", passing)},
        "candidates_by_pattern": {"A_subclass": pat("A", scored), "B_function": pat("B", scored)},
        "dropped_by_elk": len(dropped),
        "llm_contribution_rate": round(by_llm / n_acc, 3) if n_acc else 0.0,
        "llm_contribution_definition": "share of accepted axioms whose cosine alone was below tau",
        "mean_llm_share_of_score_accepted": round(float(np.mean([W_LLM * r["llm_plausibility"] / r["hybrid_score"] for r in passing])), 3) if n_acc else 0.0,
        "accepted_before_elk_at_tau_instructor_60th_percentile": sum(r["hybrid_score"] >= metrics["tau_instructor_60th_percentile"] for r in scored),
        "diagnostic_weight_sensitivity_not_applied": {f"{w:.1f}cos/{1 - w:.1f}llm": sum(w * r["cosine"] + (1 - w) * r["llm_plausibility"] >= tau for r in scored)
                                                      for w in (0.7, 0.6, 0.5, 0.4, 0.3, 0.0)},
        "mean_cosine": {p: round(float(np.mean([r["cosine"] for r in scored if r["pattern"] == p])), 3)
                        for p in "AB" if any(r["pattern"] == p for r in scored)},
    }
    write_json(REPORTS / "mowl_metrics.json", metrics)
    print(f"filter_candidates_hybrid: {n_acc}/{len(scored)} accepted at tau={tau} "
          f"(A={pat('A', passing)}, B={pat('B', passing)}, ELK-dropped={len(dropped)}) -> generated/accepted_el.ttl")


if __name__ == "__main__":
    main()
