"""Step 3: LLM proposes OWL 2 EL candidate axioms from enriched definitions -> generated/candidate_el.ttl.

Two EL patterns, both using only IRIs that already exist (no invented terms):
  A  Sub SubClassOf Super                                   (Super = another class in this module)
  B  Sub SubClassOf (bearer of) some <Artifact Function>     (BFO_0000196 + a function from ArtifactOntology)
Two LLM passes per class: pass 1 (both patterns), pass 2 (pattern A only, multiple inheritance within the module).
Every candidate must cite the phrase of the definition that supports it. Candidates that are not new
(already asserted or already entailed), cyclic, or that use unknown labels are discarded and logged."""
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rdflib import BNode, Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

sys.path.insert(0, str(Path(__file__).parent))
from common import ARTIFACT_FUNCTION, BEARER_OF, DATA, GEN, IMPORTS, MODULE, REPORTS, ask_json, save_cache  # noqa: E402

SYSTEM = ("You are an ontology engineer extending the Common Core Ontologies (BFO-based) with OWL 2 EL axioms. "
          "You only use class labels from the lists you are given, and reply with JSON only.")
PROMPT = """Class: {label}
Definition: {definition}
Currently asserted superclass(es): {parents}

Propose up to 3 NEW axioms for this class that its definition states or directly implies, using only these patterns:
  "subclass": {label} is a subclass of another class from MODULE CLASSES (not one of its current superclasses or their ancestors).
  "function": every {label} is the bearer of some function from ARTIFACT FUNCTIONS (use for "designed to ..." / purpose clauses).
Propose an axiom only if it holds for EVERY instance of the class. Returning an empty list is fine.

MODULE CLASSES:
{module_classes}

ARTIFACT FUNCTIONS:
{functions}

Reply with exactly this JSON object (labels copied exactly from the lists):
{{"axioms": [{{"pattern": "subclass" or "function", "target": "<label>", "evidence": "<phrase from the definition>"}}]}}"""


PROMPT_A = """Subclass review for the CCO class "{label}".
Definition: {definition}
Currently asserted superclass(es): {parents}

Is "{label}" ALSO a kind of any other class in this module (multiple inheritance)? Answer only from the
definitions; propose a class only if EVERY {label} is necessarily also one of them. Do not repeat a current
superclass or any of its ancestors. Returning an empty list is the normal answer.

MODULE CLASSES (label :: definition):
{module_defs}

Reply with exactly this JSON object (labels copied exactly):
{{"axioms": [{{"pattern": "subclass", "target": "<label>", "evidence": "<phrase from the definitions>"}}]}}"""


def closure_graph():
    g = Graph()
    for f in [MODULE, *IMPORTS.glob("*.ttl")]:
        g.parse(f)
    return g


def main():
    rows = list(csv.DictReader(open(DATA / "definitions_enriched.csv")))
    g, mod = closure_graph(), Graph().parse(MODULE)
    lab = lambda c: str(g.value(c, RDFS.label) or c)
    module_classes = sorted({c for c in mod.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef)}, key=lab)
    sup = {}
    for s, o in g.subject_objects(RDFS.subClassOf):
        if isinstance(o, URIRef):
            sup.setdefault(s, set()).add(o)

    def ancestors(c):
        seen, st = set(), list(sup.get(c, ()))
        while st:
            x = st.pop()
            if x not in seen:
                seen.add(x)
                st += sup.get(x, ())
        return seen

    functions = sorted({c for c in g.subjects(RDF.type, OWL.Class) if URIRef(ARTIFACT_FUNCTION) in ancestors(c)}, key=lab)
    by_label_mod = {lab(c).lower(): c for c in module_classes}
    by_label_fn = {lab(c).lower(): c for c in functions}
    # existing "bearer of some F" restrictions per class
    has_fn = {}
    for s, r in g.subject_objects(RDFS.subClassOf):
        if isinstance(r, BNode) and g.value(r, OWL.onProperty) == URIRef(BEARER_OF):
            has_fn.setdefault(s, set()).add(g.value(r, OWL.someValuesFrom))

    mc_txt = "\n".join(f"- {lab(c)}" for c in module_classes)
    norm = {r["IRI"]: r["normalized_definition"] for r in rows}
    md_txt = "\n".join(f"- {lab(c)} :: {norm.get(str(c), '')}" for c in module_classes)
    fn_txt = "\n".join(f"- {lab(c)}" for c in functions)

    def work(r):
        c = URIRef(r["IRI"])
        parents = "; ".join(sorted(lab(p) for p in sup.get(c, ()) if isinstance(p, URIRef))) or "(none)"
        out = ask_json(SYSTEM, PROMPT.format(label=r["label"], definition=r["normalized_definition"], parents=parents,
                                             module_classes=mc_txt, functions=fn_txt), max_tokens=1000)
        # second pass (subclass only): pass 1 favours function axioms, so ask for in-module parents explicitly
        out_a = ask_json(SYSTEM, PROMPT_A.format(label=r["label"], definition=r["normalized_definition"], parents=parents,
                                                 module_defs=md_txt), max_tokens=800)
        return r, c, (out.get("axioms") or [])[:3] + (out_a.get("axioms") or [])[:2]

    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(work, rows))
    save_cache()

    cands, rejected, seen = [], [], set()
    for r, c, axioms in results:
        anc = ancestors(c)
        for a in axioms:
            pat, tgt, ev = str(a.get("pattern", "")).lower(), str(a.get("target", "")).strip(), str(a.get("evidence", ""))
            why = None
            if pat == "subclass":
                t = by_label_mod.get(tgt.lower())
                if t is None: why = "unknown module class label"
                elif t == c: why = "self"
                elif t in anc: why = "already asserted or entailed"
                elif c in ancestors(t): why = "would create a cycle (target is a subclass)"
            elif pat == "function":
                t = by_label_fn.get(tgt.lower())
                if t is None: why = "unknown function label"
                elif any(t == f or t in ancestors(f) for f in has_fn.get(c, ())): why = "already asserted or entailed"
            else:
                t, why = None, f"unknown pattern {pat!r}"
            key = (pat, str(c), str(t))
            if why is None and key in seen: why = "duplicate"
            rec = {"id": "", "pattern": "A" if pat == "subclass" else "B", "sub_iri": str(c), "sub_label": r["label"],
                   "target_iri": str(t) if t else "", "target_label": tgt, "evidence": ev}
            if why:
                rejected.append({**rec, "reason": why})
            else:
                seen.add(key)
                cands.append(rec)
    for i, x in enumerate(cands, 1):
        x["id"] = f"C{i:03d}"

    out = Graph()
    out.bind("owl", OWL); out.bind("obo", Namespace("http://purl.obolibrary.org/obo/")); out.bind("cco", Namespace("https://www.commoncoreontologies.org/"))
    write_axioms(out, cands)
    GEN.mkdir(exist_ok=True)
    out.serialize(GEN / "candidate_el.ttl", format="turtle")
    cols = ["id", "pattern", "sub_iri", "sub_label", "target_iri", "target_label", "evidence"]
    with open(GEN / "candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(cands)
    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / "candidates_discarded.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols[1:] + ["reason"]); w.writeheader()
        w.writerows({k: v for k, v in x.items() if k != "id"} for x in rejected)
    nA = sum(x["pattern"] == "A" for x in cands)
    print(f"generate_candidates_llm: {len(cands)} candidates (A subclass={nA}, B function={len(cands) - nA}); "
          f"{len(rejected)} discarded -> generated/candidate_el.ttl")


def write_axioms(g: Graph, recs) -> None:
    """Serialize candidate records as OWL 2 EL axioms, with the declarations needed to parse standalone."""
    g.add((URIRef(BEARER_OF), RDF.type, OWL.ObjectProperty))
    for x in recs:
        s, t = URIRef(x["sub_iri"]), URIRef(x["target_iri"])
        g.add((s, RDF.type, OWL.Class)); g.add((t, RDF.type, OWL.Class))
        if x["pattern"] == "A":
            g.add((s, RDFS.subClassOf, t))
        else:
            b = BNode()
            g.add((b, RDF.type, OWL.Restriction)); g.add((b, OWL.onProperty, URIRef(BEARER_OF))); g.add((b, OWL.someValuesFrom, t))
            g.add((s, RDFS.subClassOf, b))


if __name__ == "__main__":
    main()
