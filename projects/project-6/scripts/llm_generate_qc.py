"""Stage 4: LLM-generated SHACL shapes and SPARQL QC queries, kept only if they survive a test harness.

Claude sees the approved patterns, the exact IRIs and a sample of real instance triples, and writes shapes and
queries. It does NOT see the fault list below. Each artifact is then:
  1. parsed (Turtle / SPARQL) and executed on the clean graph;
  2. rejected if it flags clean data, except the acknowledged real defect (reports/known_issues.json);
  3. run against fault-injected copies of the graph (mutation testing) to record which faults it catches.
Kept artifacts -> shacl/llm_shapes.ttl and sparql/qc_*.rq; evidence -> reports/qc_generation.json."""
import copy
import json
import re
import sys
from pathlib import Path

from pyshacl import validate
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, SH, XSD

sys.path.insert(0, str(Path(__file__).parent))
from common import BASE_EXT, CCO, INSTANCES, OBO, P6, REPORTS, SHACL_DIR, SLIM, SPARQL_DIR, ask_text, load_spec, save_cache, write_json  # noqa: E402

C, B, X, ID = Namespace(CCO), Namespace(OBO), Namespace(P6), Namespace(P6 + "id/")
PREFIXES = f"""PREFIX p6: <{P6}>
PREFIX id: <{P6}id/>
PREFIX cco: <{CCO}>
PREFIX obo: <{OBO}>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX sh: <http://www.w3.org/ns/shacl#>"""

SYSTEM = "You are a semantic-web QA engineer. You write correct SHACL (Turtle) and SPARQL 1.1."
PROMPT = """We load GTFS rail schedule data into CCO/BFO with these approved patterns:
{patterns}

Term IRIs:
{terms}

Prefixes available to you (declare them yourself in SHACL Turtle; they are prepended to every SPARQL query):
{prefixes}

Sample of the instance data (Turtle):
{sample}

Write data-quality checks for this graph:
{task}
Reply with only these blocks, code inside them verbatim (no Markdown fences, no JSON, no commentary):
{form}"""
TASK_SHAPES = ("- 8 SHACL shapes (each a self-contained Turtle document with @prefix lines) covering structure,\n"
               "  cardinality, datatypes and value ranges of the patterns. Keep each shape compact.")
FORM_SHAPES = '<shape name="ShortName" description="one sentence">\n...Turtle...\n</shape>'
TASK_QUERIES = ("- 8 SPARQL SELECT queries for checks SHACL Core expresses poorly (ordering, cross-record and temporal logic,\n"
                "  references to missing entities, code-list and format checks). Each query must return one row per violation\n"
                "  with variables ?focus and ?message, and must not include PREFIX lines.")
FORM_QUERIES = '<query name="short_name" severity="error|warning" description="one sentence">\nSELECT ?focus ?message WHERE { ... }\n</query>'



def parse_blocks(text: str) -> dict:
    """Parse <shape>/<query> blocks from the reply."""
    attrs = lambda a: dict(re.findall(r'(\w+)="([^"]*)"', a))
    strip = lambda c: re.sub(r"^```\w*\n|\n```$", "", c.strip()).strip()
    shapes = [{**attrs(a), "turtle": strip(c)} for a, c in re.findall(r"<shape\b([^>]*)>(.*?)</shape>", text, re.S)]
    queries = [{**attrs(a), "sparql": strip(c)} for a, c in re.findall(r"<query\b([^>]*)>(.*?)</query>", text, re.S)]
    return {"shapes": shapes, "queries": queries}

TERMS = """p6:RailStation, p6:GTFSRecord (an Information Bearing Entity), p6:StationCoordinateRecord, p6:TransitRouteSpecification,
p6:TrainTripSpecification, p6:ScheduledStopSpecification; p6:stopSequence, p6:directionCode, p6:routeTypeCode,
p6:wheelchairBoardingCode (xsd:integer), p6:isPeakService (xsd:boolean), p6:headsignText (xsd:string)
cco:ont00000487 Geospatial Location; cco:ont00000003 Designative Name; cco:ont00000077 Code Identifier;
cco:ont00000923 Arbitrary Identifier; cco:ont00000829 Time Zone Identifier; cco:ont00000827 Time of Day Identifier;
cco:ont00001180 Organization; cco:ont00001916 designates; cco:ont00001808 is about; cco:ont00001908 uses time zone identifier;
cco:ont00001765 has text value; cco:ont00001766 has latitude value; cco:ont00001764 has longitude value (xsd:decimal);
cco:ont00001767 has datetime value (xsd:dateTime); obo:BFO_0000171 located in; obo:BFO_0000101 is carrier of;
obo:BFO_0000176 continuant part of. Arrival/departure Time of Day Identifiers carry rdfs:label "arrival time" / "departure time"."""


# ---------------------------------------------------------------- fault injection (hidden from the LLM)
def _one(g, s=None, p=None, o=None):
    return next(iter(sorted(g.triples((s, p, o)))), None)


def _stops(g):
    return sorted(g.subjects(RDF.type, X.ScheduledStopSpecification))


def _time_rec(g, stop, which):
    tod = URIRef(str(stop) + "/" + which)
    return URIRef(str(tod) + "/record")


def _seq(g, stop):
    return int(g.value(URIRef(str(stop) + "/record"), X.stopSequence))


def f_missing_station_name(g):
    st = sorted(g.subjects(RDF.type, X.RailStation))[0]
    for d in list(g.subjects(C.ont00001916, st)):
        if (d, RDF.type, C.ont00000003) in g:
            g.remove((d, None, None)); g.remove((None, None, d))
            g.remove((URIRef(str(d) + "/record"), None, None))


def f_departure_before_arrival(g):
    sp = [s for s in _stops(g) if g.value(_time_rec(g, s, "arrival"), C.ont00001767)][5]
    a = g.value(_time_rec(g, sp, "arrival"), C.ont00001767).toPython()
    from datetime import timedelta
    g.set((_time_rec(g, sp, "departure"), C.ont00001767, Literal((a - timedelta(minutes=5)).isoformat(), datatype=XSD.dateTime)))


def f_coordinate_out_of_range(g):
    rec = sorted(g.subjects(C.ont00001766, None))[0]
    g.set((rec, C.ont00001766, Literal("95.0", datatype=XSD.decimal)))


def f_duplicate_stop_sequence(g):
    sp = _stops(g)[3]
    trip = g.value(sp, B.BFO_0000176)
    sibs = sorted((s for s in g.subjects(B.BFO_0000176, trip) if (s, RDF.type, X.ScheduledStopSpecification) in g), key=lambda s: _seq(g, s))
    g.set((URIRef(str(sibs[2]) + "/record"), X.stopSequence, Literal(_seq(g, sibs[1]))))


def f_dangling_station_reference(g):
    sp = _stops(g)[7]
    g.set((sp, C.ont00001808, ID["station/DOES-NOT-EXIST"]))


def f_malformed_time_text(g):
    rec = _time_rec(g, _stops(g)[9], "arrival")
    g.set((rec, C.ont00001765, Literal("25:61:00"))); g.remove((rec, C.ont00001767, None))


def f_non_monotonic_times(g):
    sp = _stops(g)[12]
    trip = g.value(sp, B.BFO_0000176)
    sibs = sorted((s for s in g.subjects(B.BFO_0000176, trip) if (s, RDF.type, X.ScheduledStopSpecification) in g), key=lambda s: _seq(g, s))
    from datetime import timedelta
    prev_dep = g.value(_time_rec(g, sibs[1], "departure"), C.ont00001767).toPython()
    for w in ("arrival", "departure"):
        g.set((_time_rec(g, sibs[2], w), C.ont00001767, Literal((prev_dep - timedelta(minutes=10)).isoformat(), datatype=XSD.dateTime)))


def f_invalid_agency_timezone(g):
    rec = sorted(s for s in g.subjects(C.ont00001765, None) if "/agency/" in str(s) and str(s).endswith("/timezone/record"))[0]
    g.set((rec, C.ont00001765, Literal("Eastern")))


def f_trip_without_stops(g):
    trip = sorted(g.subjects(RDF.type, X.TrainTripSpecification))[4]
    for s in list(g.subjects(B.BFO_0000176, trip)):
        if (s, RDF.type, X.ScheduledStopSpecification) in g:
            g.remove((s, B.BFO_0000176, trip))


def f_wrong_datatype(g):
    rec = URIRef(str(_stops(g)[15]) + "/record")
    g.set((rec, X.stopSequence, Literal("third")))


FAULTS = {f.__name__[2:]: f for f in [f_missing_station_name, f_departure_before_arrival, f_coordinate_out_of_range,
                                       f_duplicate_stop_sequence, f_dangling_station_reference, f_malformed_time_text,
                                       f_non_monotonic_times, f_invalid_agency_timezone, f_trip_without_stops, f_wrong_datatype]}


# ---------------------------------------------------------------- execution helpers
def known_issue(focus: str) -> bool:
    return any(k in focus for k in json.loads((REPORTS / "known_issues.json").read_text())["focus_contains"])


def shacl_foci(data: Graph, shapes: Graph) -> dict:
    """source shape IRI -> set of focus nodes with violations/warnings."""
    _, rg, _ = validate(data, shacl_graph=shapes, inference="none", allow_warnings=True, abort_on_first=False)
    out = {}
    for r in rg.subjects(RDF.type, SH.ValidationResult):
        out.setdefault(str(rg.value(r, SH.sourceShape)), set()).add(str(rg.value(r, SH.focusNode)))
    return out


def _eval_worker(blobs: dict, q: str) -> dict:
    import pyoxigraph as ox
    out = {}
    for name, nt in blobs.items():
        st = ox.Store(); st.load(nt, format=ox.RdfFormat.N_TRIPLES)
        out[name] = {getattr(r["focus"], "value", str(r["focus"])) for r in st.query(PREFIXES + "\n" + q) if r["focus"] is not None}
    return out


def sparql_eval(blobs: dict, q: str, timeout: int = 60) -> dict:
    """Run one query on every graph with Oxigraph in a child process; a query slower than `timeout` s is rejected."""
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply_async(_eval_worker, (blobs, q)).get(timeout=timeout)


def main():
    spec = load_spec()
    clean = Graph().parse(INSTANCES)
    onto = Graph().parse(BASE_EXT) + Graph().parse(SLIM)
    st = str(sorted(clean.subjects(RDF.type, X.RailStation))[0])
    tr = str(sorted(clean.subjects(RDF.type, X.TrainTripSpecification))[0])
    feed = next(str(s) for s in clean.subjects(RDF.type, X.GTFSRecord) if "/id/feed/" in str(s))
    agency = next(str(s) for s in clean.subjects(RDF.type, C.ont00001180))
    def wanted(subj: str) -> bool:
        if subj.startswith(st + "/") or subj in (st, tr) or subj.startswith(feed) or subj.startswith(agency):
            return True
        return subj.startswith(tr + "/") and ("/stop/" not in subj or subj.startswith(tr + "/stop/1/") or subj == tr + "/stop/1")
    sample = Graph()
    for t in clean:
        if wanted(str(t[0])):
            sample.add(t)
    for p, ns in (("p6", X), ("id", ID), ("cco", C), ("obo", B)):
        sample.bind(p, ns)
    common = dict(patterns=json.dumps(spec["patterns"], indent=1), terms=TERMS, prefixes=PREFIXES,
                  sample=sample.serialize(format="turtle"))
    shapes = parse_blocks(ask_text(SYSTEM, PROMPT.format(task=TASK_SHAPES, form=FORM_SHAPES, **common), max_tokens=48000))
    queries = parse_blocks(ask_text(SYSTEM, PROMPT.format(task=TASK_QUERIES, form=FORM_QUERIES, **common), max_tokens=48000))
    out = {"shapes": shapes["shapes"], "queries": queries["queries"]}
    save_cache()

    data = clean + onto                         # class hierarchy available to sh:targetClass
    mutated = {}
    for name, fn in FAULTS.items():
        g = copy.deepcopy(clean); fn(g); mutated[name] = g + onto

    results, kept_shapes, kept_queries = [], Graph(), []
    # ---- SHACL: parse each doc; validate all parsed docs together per graph, attribute by sh:sourceShape
    docs = []
    for i, s in enumerate(out.get("shapes", []), 1):
        rec = {"kind": "shacl", "name": s.get("name", f"shape{i}"), "description": s.get("description", "")}
        try:
            sg = Graph().parse(data=s["turtle"], format="turtle")
            ids = {str(x) for x in sg.subjects(RDF.type, SH.NodeShape)} | {str(x) for x in sg.subjects(RDF.type, SH.PropertyShape)}
            ids |= {str(x) for x in sg.objects(None, SH.property)}
            if not ids:
                raise ValueError("no sh:NodeShape / sh:PropertyShape")
            docs.append((rec, sg, ids))
        except Exception as e:
            rec.update(status="rejected", reason=f"parse error: {str(e)[:160]}"); results.append(rec)
    if docs:
        allsg = Graph()
        for _, sg, _ in docs:
            allsg += sg
        try:
            base = shacl_foci(data, allsg)
            per_fault = {n: shacl_foci(g, allsg) for n, g in mutated.items()}
        except Exception:                        # one bad doc can break joint validation: fall back to per-doc
            base, per_fault = None, None
        for rec, sg, ids in docs:
            try:
                bm = base if base is not None else shacl_foci(data, sg)
                b = set().union(set(), *[v for k, v in bm.items() if k in ids])
                fp = sorted(f for f in b if not known_issue(f))
                caught = []
                for n, g in mutated.items():
                    m = per_fault[n] if per_fault else shacl_foci(g, sg)
                    mf = set().union(set(), *[v for k, v in m.items() if k in ids])
                    if mf - b:
                        caught.append(n)
                rec.update(clean_findings=len(b), known_issue_findings=len(b) - len(fp), false_positives=fp[:5], catches=caught)
                if fp:
                    rec.update(status="rejected", reason=f"flags {len(fp)} clean-data nodes (false positives)")
                else:
                    rec.update(status="kept"); kept_shapes += sg
            except Exception as e:
                rec.update(status="rejected", reason=f"execution error: {str(e)[:160]}")
            results.append(rec)
    # ---- SPARQL (Oxigraph; graphs serialized once)
    blobs = {"__clean__": data.serialize(format="nt", encoding="utf-8")}
    blobs.update({n: g.serialize(format="nt", encoding="utf-8") for n, g in mutated.items()})
    for i, q in enumerate(out.get("queries", []), 1):
        rec = {"kind": "sparql", "name": q.get("name", f"query{i}"), "description": q.get("description", ""), "severity": q.get("severity", "error")}
        text = re.sub(r"(?im)^\s*PREFIX\s+\S+\s*<[^>]*>\s*$", "", q.get("sparql", "")).strip()
        try:
            if not re.search(r"(?is)^\s*SELECT\b[^{]*\?focus\b", text):
                raise ValueError("query is not a SELECT projecting ?focus")
            res = sparql_eval(blobs, text)
            b = res.pop("__clean__")
            fp = sorted(f for f in b if not known_issue(f))
            caught = [n for n in mutated if res[n] - b]
            rec.update(clean_findings=len(b), known_issue_findings=len(b) - len(fp), false_positives=fp[:5], catches=caught)
            if fp:
                rec.update(status="rejected", reason=f"flags {len(fp)} clean-data nodes (false positives)")
            else:
                rec.update(status="kept", sparql=text); kept_queries.append(rec)
        except Exception as e:
            msg = "timed out (> 60 s)" if type(e).__name__ == "TimeoutError" else f"parse/execution error: {str(e)[:160]}"
            rec.update(status="rejected", reason=msg)
        results.append(rec)

    SHACL_DIR.mkdir(exist_ok=True); SPARQL_DIR.mkdir(exist_ok=True)
    for old in SPARQL_DIR.glob("qc_*.rq"):
        old.unlink()
    kept_shapes.bind("sh", SH); kept_shapes.bind("p6", X); kept_shapes.bind("cco", C); kept_shapes.bind("obo", B)
    kept_shapes.serialize(SHACL_DIR / "llm_shapes.ttl", format="turtle")
    for i, q in enumerate(kept_queries, 1):
        slug = re.sub(r"[^a-z0-9]+", "_", q["name"].lower()).strip("_")[:40]
        (SPARQL_DIR / f"qc_{i:02d}_{slug}.rq").write_text(f"# {q['description']}\n# severity: {q['severity']}\n{PREFIXES}\n{q['sparql']}\n")
    kept = [r for r in results if r["status"] == "kept"]
    coverage = {n: sorted(r["name"] for r in kept if n in r.get("catches", [])) for n in FAULTS}
    summary = {"generated": {"shacl": len(out.get("shapes", [])), "sparql": len(out.get("queries", []))},
               "kept": {"shacl": sum(r["kind"] == "shacl" for r in kept), "sparql": len(kept_queries)},
               "faults_caught": sum(bool(v) for v in coverage.values()), "faults_total": len(FAULTS),
               "coverage": coverage, "artifacts": results}
    write_json(REPORTS / "qc_generation.json", summary)
    print(f"llm_generate_qc: kept {summary['kept']['shacl']}/{summary['generated']['shacl']} shapes, "
          f"{summary['kept']['sparql']}/{summary['generated']['sparql']} queries; injected faults caught "
          f"{summary['faults_caught']}/{len(FAULTS)} -> shacl/llm_shapes.ttl, sparql/qc_*.rq")
    missed = [n for n, v in coverage.items() if not v]
    if missed:
        print(f"llm_generate_qc: faults not caught by any kept check: {', '.join(missed)}")


if __name__ == "__main__":
    main()
