"""Stage 5: LLM-proposed OWL axioms for the new domain classes -> ontology/p6_extension.ttl.

Claude proposes EL axioms (Sub SubClassOf Named, or Sub SubClassOf P some Filler) for the local classes, using only
listed IRIs. A candidate is accepted only if it passes every gate:
  1. vocabulary: all IRIs come from the allowed lists; not already asserted;
  2. ELK (ROBOT): base extension + CCO slim + candidate stays coherent (no unsatisfiable classes);
  3. HermiT (ROBOT): consistent together with a sample of the instance data (catches domain/disjointness errors
     ELK ignores, e.g. data properties whose domain is Information Bearing Entity);
  4. data support: for "P some F", every instance of Sub in the data has a P-link to an instance of F.
Accepted axioms are added to the base extension; every decision is logged in reports/axiom_review.csv."""
import csv
import json
import sys
from pathlib import Path

from rdflib import BNode, Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS

sys.path.insert(0, str(Path(__file__).parent))
from common import BASE_EXT, EXT, GEN, INSTANCES, P6, REPORTS, SLIM, ask_json, expand, load_spec, robot, save_cache  # noqa: E402

SYSTEM = "You are an ontology engineer extending the BFO-based Common Core Ontologies with OWL 2 EL axioms. Reply with JSON only."
PROMPT = """Propose OWL 2 EL axioms that strengthen these new domain classes (already subclasses of the parents shown):
{classes}

Patterns in use:
{patterns}

Allowed classes for parents or fillers:
{allowed_classes}

Allowed object properties:
{allowed_props}

Use only two forms: "subclass" (subject SubClassOf filler, filler a named class) and "some"
(subject SubClassOf property some filler). Propose an axiom only if it holds for EVERY instance of the subject.
Reply with exactly:
{{"axioms": [{{"subject": "p6:...", "pattern": "subclass" | "some", "property": "<curie, for some>", "filler": "<curie>", "rationale": "..."}}]}}"""

ALLOWED_PROPS = {"obo:BFO_0000171": "located in", "obo:BFO_0000101": "is carrier of", "obo:BFO_0000176": "continuant part of",
                 "obo:BFO_0000178": "has continuant part", "cco:ont00001808": "is about", "cco:ont00001916": "designates",
                 "cco:ont00001942": "prescribes", "cco:ont00001908": "uses time zone identifier"}
ALLOWED_CCO = {"cco:ont00000487": "Geospatial Location", "cco:ont00000226": "Transportation Facility", "cco:ont00000192": "Facility",
               "cco:ont00000253": "Information Bearing Entity", "cco:ont00000958": "Information Content Entity",
               "cco:ont00000853": "Descriptive Information Content Entity", "cco:ont00000965": "Prescriptive Information Content Entity",
               "cco:ont00000686": "Designative Information Content Entity", "cco:ont00000003": "Designative Name",
               "cco:ont00000077": "Code Identifier", "cco:ont00000923": "Arbitrary Identifier", "cco:ont00000827": "Time of Day Identifier",
               "cco:ont00000829": "Time Zone Identifier", "cco:ont00001180": "Organization", "cco:ont00000890": "Act of Travel",
               "cco:ont00000228": "Planned Act"}


def sample_instances(g: Graph) -> Graph:
    """Stations, agency, feed, route and the first two trips with their stops (HermiT gate)."""
    trips = sorted(str(t) for t in g.subjects(RDF.type, URIRef(P6 + "TrainTripSpecification")))[:2]
    keep = lambda s: "/trip/" not in s or any(s.startswith(t) for t in trips)
    out = Graph()
    for t in g:
        if keep(str(t[0])):
            out.add(t)
    return out


def axiom_graph(recs) -> Graph:
    g = Graph()
    for r in recs:
        s, f = URIRef(expand(r["subject"])), URIRef(expand(r["filler"]))
        g.add((s, RDF.type, OWL.Class)); g.add((f, RDF.type, OWL.Class))
        if r["pattern"] == "subclass":
            g.add((s, RDFS.subClassOf, f))
        else:
            b, p = BNode(), URIRef(expand(r["property"]))
            g.add((p, RDF.type, OWL.ObjectProperty))
            g.add((b, RDF.type, OWL.Restriction)); g.add((b, OWL.onProperty, p)); g.add((b, OWL.someValuesFrom, f))
            g.add((s, RDFS.subClassOf, b))
    return g


def reasoner_ok(reasoner: str, extra: Graph, with_data: Graph | None = None) -> tuple[bool, str]:
    g = Graph().parse(BASE_EXT) + Graph().parse(SLIM) + extra
    g.remove((None, OWL.imports, None))
    if with_data is not None:
        g += with_data
    tmp = GEN / f"_{reasoner}_check.ttl"; g.serialize(tmp, format="turtle")
    p = robot("reason", "--input", str(tmp), "--reasoner", reasoner, check=False)
    tmp.unlink(missing_ok=True)
    lines = [l for l in (p.stdout + p.stderr).splitlines() if l.strip() and "JAVA_OPTIONS" not in l and "JAVA_TOOL_OPTIONS" not in l]
    key = [l for l in lines if any(k in l.lower() for k in ("unsatisfiable", "inconsistent", "error"))]
    return p.returncode == 0, ((key or lines or [""])[0]).strip()[:200]


def superclasses(onto: Graph, c: URIRef) -> set:
    out, st = set(), [o for o in onto.objects(c, RDFS.subClassOf) if isinstance(o, URIRef)]
    while st:
        x = st.pop()
        if x not in out:
            out.add(x); st += [o for o in onto.objects(x, RDFS.subClassOf) if isinstance(o, URIRef)]
    return out


def counterexamples(data: Graph, onto: Graph, r) -> int:
    """Instances of subject (incl. subclasses) lacking a property link to an instance of the filler."""
    sup = {}
    for s, o in onto.subject_objects(RDFS.subClassOf):
        if isinstance(o, URIRef):
            sup.setdefault(s, set()).add(o)

    def types(x):
        out, st = set(), list(data.objects(x, RDF.type))
        while st:
            c = st.pop()
            if c not in out:
                out.add(c); st += sup.get(c, ())
        return out
    S, P, F = URIRef(expand(r["subject"])), URIRef(expand(r["property"])), URIRef(expand(r["filler"]))
    inv = {a: b for a, b in onto.subject_objects(OWL.inverseOf)} | {b: a for a, b in onto.subject_objects(OWL.inverseOf)}
    subs = [x for x in set(data.subjects(RDF.type, None)) if S in types(x)]

    def linked(x):                     # x P o, or o P^-1 x (owl:inverseOf from BFO/CCO)
        return set(data.objects(x, P)) | (set(data.subjects(inv[P], x)) if P in inv else set())
    return sum(not any(F in types(o) for o in linked(x)) for x in subs)


def main():
    spec = load_spec()
    base = Graph().parse(BASE_EXT)
    onto = base + Graph().parse(SLIM)
    data = Graph().parse(INSTANCES)
    sample = sample_instances(data)
    classes = "\n".join(f"- {c} ({d['label']}) SubClassOf {d['parent']} ({d['parent_label']}): {d['definition']}"
                        for c, d in spec["local_classes"].items())
    allowed_cls = {**ALLOWED_CCO, **{c: d["label"] for c, d in spec["local_classes"].items()}}
    out = ask_json(SYSTEM, PROMPT.format(classes=classes, patterns=json.dumps(spec["patterns"], indent=1),
                                         allowed_classes="\n".join(f"- {k} {v}" for k, v in allowed_cls.items()),
                                         allowed_props="\n".join(f"- {k} {v}" for k, v in ALLOWED_PROPS.items())), max_tokens=4000)
    save_cache()

    ok_ref, _ = reasoner_ok("HermiT", Graph(), sample)
    if not ok_ref:
        raise SystemExit("baseline (approved extension + CCO slim + data sample) is not HermiT-consistent; fix the patterns first")
    review, accepted = [], []
    for r in out.get("axioms", []):
        r = {k: str(v) for k, v in r.items()}
        row = {**r, "decision": "", "reason": ""}
        try:
            if r.get("subject") not in spec["local_classes"]:
                raise ValueError("subject is not a new domain class")
            if r.get("filler") not in allowed_cls or (r.get("pattern") == "some" and r.get("property") not in ALLOWED_PROPS):
                raise ValueError("IRI outside the allowed vocabulary")
            if r.get("pattern") not in ("subclass", "some"):
                raise ValueError("pattern must be subclass or some")
            if r["pattern"] == "subclass" and URIRef(expand(r["filler"])) in superclasses(onto, URIRef(expand(r["subject"]))):
                raise ValueError("already asserted or entailed")
            ok, msg = reasoner_ok("ELK", axiom_graph(accepted + [r]))
            if not ok:
                raise ValueError(f"ELK: incoherent ({msg})")
            ok, msg = reasoner_ok("HermiT", axiom_graph(accepted + [r]), sample)
            if not ok:
                raise ValueError(f"HermiT: inconsistent with the data sample ({msg})")
            if r["pattern"] == "some":
                n = counterexamples(data, onto, r)
                if n:
                    raise ValueError(f"not supported by the data: {n} instances of the subject lack this link")
            row.update(decision="accepted"); accepted.append(r)
        except ValueError as e:
            row.update(decision="rejected", reason=str(e))
        review.append(row)

    ext = base + axiom_graph(accepted)
    for p, ns in base.namespaces():
        ext.bind(p, ns)
    ext.serialize(EXT, format="turtle")
    cols = ["subject", "pattern", "property", "filler", "rationale", "decision", "reason"]
    with open(REPORTS / "axiom_review.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(review)
    print(f"llm_propose_axioms: {len(accepted)}/{len(review)} candidate axioms accepted (ELK + HermiT + data support) "
          f"-> ontology/p6_extension.ttl")


if __name__ == "__main__":
    main()
