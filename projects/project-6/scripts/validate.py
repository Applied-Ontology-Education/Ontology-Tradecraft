"""Stage 6: verify the final graph -> reports/validation.json + reports/validation_report.md (non-zero exit on failure).

  - ELK (ROBOT): extension + CCO slim + all instances: consistent, no unsatisfiable classes.
  - HermiT (ROBOT): the same, full DL semantics (domains, disjointness, data properties).
  - SHACL (pyshacl): kept LLM-generated shapes.
  - SPARQL QC (Oxigraph): kept LLM-generated queries.
Findings on acknowledged real defects (reports/known_issues.json) are reported but do not fail the build;
any other finding, or any reasoner failure, fails it."""
import json
import sys
import time
from pathlib import Path

from pyshacl import validate
from rdflib import Graph
from rdflib.namespace import OWL, RDF, SH

sys.path.insert(0, str(Path(__file__).parent))
from common import EXT, GEN, INSTANCES, REPORTS, SHACL_DIR, SLIM, SPARQL_DIR, robot, write_json  # noqa: E402
from llm_generate_qc import known_issue  # noqa: E402


def reason(reasoner, path):
    t = time.time()
    p = robot("reason", "--input", str(path), "--reasoner", reasoner, check=False)
    lines = [l for l in (p.stdout + p.stderr).splitlines() if "ERROR" in l or "unsatisfiable" in l.lower()]
    return {"ok": p.returncode == 0, "seconds": round(time.time() - t, 1), "message": (lines[0] if lines else "")[:200]}


def main():
    onto = Graph().parse(EXT) + Graph().parse(SLIM)
    onto.remove((None, OWL.imports, None))
    data = Graph().parse(INSTANCES)
    full = onto + data
    merged = GEN / "merged_for_reasoning.ttl"
    full.serialize(merged, format="turtle")
    res = {"triples": {"ontology": len(onto), "instances": len(data)},
           "elk": reason("ELK", merged), "hermit": reason("HermiT", merged)}
    merged.unlink(missing_ok=True)

    shapes = Graph().parse(SHACL_DIR / "llm_shapes.ttl")
    conforms, rg, _ = validate(full, shacl_graph=shapes, inference="none", allow_warnings=True)
    sh = []
    for r in rg.subjects(RDF.type, SH.ValidationResult):
        f = str(rg.value(r, SH.focusNode))
        sh.append({"focus": f, "shape": str(rg.value(r, SH.sourceShape)), "message": str(rg.value(r, SH.resultMessage) or ""),
                   "known_issue": known_issue(f)})
    res["shacl"] = {"shapes": len(set(shapes.subjects(RDF.type, SH.NodeShape))), "findings": len(sh),
                    "unexpected": sum(not x["known_issue"] for x in sh), "results": sh[:50]}

    import pyoxigraph as ox
    st = ox.Store(); st.load(full.serialize(format="nt", encoding="utf-8"), format=ox.RdfFormat.N_TRIPLES)
    qs = []
    for q in sorted(SPARQL_DIR.glob("*.rq")):                 # qc_* = LLM-generated (kept), human_* = human-authored
        rows = [(getattr(r["focus"], "value", str(r["focus"])), getattr(r["message"], "value", "") if r["message"] is not None else "")
                for r in st.query(q.read_text())]
        qs.append({"query": q.name, "origin": "human" if q.name.startswith("human_") else "llm", "findings": len(rows), "unexpected": sum(not known_issue(f) for f, _ in rows),
                   "examples": [{"focus": f, "message": m} for f, m in rows[:5]]})
    res["sparql"] = {"queries": len(qs), "findings": sum(q["findings"] for q in qs),
                     "unexpected": sum(q["unexpected"] for q in qs), "results": qs}
    res["known_issues"] = json.loads((REPORTS / "known_issues.json").read_text())["issues"]
    res["checks_present"] = res["shacl"]["shapes"] > 0 and res["sparql"]["queries"] > 0
    res["passed"] = res["checks_present"] and res["elk"]["ok"] and res["hermit"]["ok"] and res["shacl"]["unexpected"] == 0 and res["sparql"]["unexpected"] == 0
    write_json(REPORTS / "validation.json", res)

    md = ["# Project 6 validation report", "",
          f"Ontology {res['triples']['ontology']} triples (extension + CCO slim), instances {res['triples']['instances']} triples.", "",
          "| Check | Result |", "|---|---|",
          f"| ELK (ROBOT) | {'consistent, no unsatisfiable classes' if res['elk']['ok'] else 'FAILED: ' + res['elk']['message']} ({res['elk']['seconds']} s) |",
          f"| HermiT (ROBOT) | {'consistent' if res['hermit']['ok'] else 'FAILED: ' + res['hermit']['message']} ({res['hermit']['seconds']} s) |",
          f"| SHACL ({res['shacl']['shapes']} node shapes) | {res['shacl']['findings']} findings, {res['shacl']['unexpected']} unexpected |",
          f"| SPARQL QC ({res['sparql']['queries']} queries) | {res['sparql']['findings']} findings, {res['sparql']['unexpected']} unexpected |",
          f"| QC checks present | {'yes' if res['checks_present'] else 'NO: no LLM-generated checks survived; nothing was verified'} |",
          f"| **Overall** | **{'PASS' if res['passed'] else 'FAIL'}** |", "", "## Known (acknowledged) source-data issues"]
    md += [f"- `{i['file']}` `{i['column']}` = `{i['value']}`: expected {i['expected']}. {i['note']}" for i in res["known_issues"]]
    md += ["", "## SPARQL QC detail", ""] + [f"- `{q['query']}` ({q['origin']}): {q['findings']} findings ({q['unexpected']} unexpected)" for q in qs]
    (REPORTS / "validation_report.md").write_text("\n".join(md) + "\n")
    print(f"validate: ELK {'OK' if res['elk']['ok'] else 'FAIL'}, HermiT {'OK' if res['hermit']['ok'] else 'FAIL'} "
          f"({res['hermit']['seconds']} s); SHACL {res['shacl']['findings']} findings ({res['shacl']['unexpected']} unexpected); "
          f"SPARQL {res['sparql']['findings']} findings ({res['sparql']['unexpected']} unexpected) -> {'PASS' if res['passed'] else 'FAIL'}")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
