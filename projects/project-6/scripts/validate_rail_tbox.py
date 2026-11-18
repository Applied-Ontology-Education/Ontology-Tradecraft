#!/usr/bin/env python3
"""Project 6 Step 4: deterministic, auto-generated SPARQL QC + SHACL for rail TBox.

Run from project-6 root:
    python scripts/validate_rail_tbox.py

Requires Steps 1-3A and pyshacl in the project's Python environment.
For generating/checking rules without pyshacl (does NOT pass Step 4):
    python scripts/validate_rail_tbox.py --generate-only

All rules are generated from the Step 3A refined TBox, which ultimately derives
from the LLM's Step 2 rail proposal; no new API call in this stage. Neither the
source ontology nor any prior-stage input is edited or overwritten.

This stage checks *explicit TBox structure*; it is not OWL-DL consistency or
approval for merging. A later ROBOT/OWL reasoner stage is still required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SH, SKOS

VERSION = "project6-rail-qc-shacl-v1"
RAIL_NS = "https://example.org/ontologies/intel-ontology#"
QC_NS = "https://example.org/ontologies/project6-qc#"
TERMS = (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def parse_turtle(path: Path) -> Graph:
    return Graph().parse(location=str(path), format="turtle")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def sorted_turtle(g: Graph, header: str) -> str:
    # N-Triples statements are valid Turtle. All nodes here have named IRIs.
    return "# " + header + "\n\n" + "\n".join(
        sorted(f"{s.n3()} {p.n3()} {o.n3()} ." for s, p, o in g)
    ) + "\n"


def require_inputs(root: Path) -> tuple[Graph, Graph, Graph, dict, dict, dict, dict]:
    paths = {
        "source": root / "src/module.ttl",
        "candidate": root / "generated/rail_axioms_candidates.ttl",
        "refined": root / "generated/rail_axioms_refined.ttl",
        "proposal": root / "generated/rail_pattern_proposal.json",
        "audit": root / "reports/step3a_refinement_audit.json",
        "summary": root / "reports/step3a_summary.json",
    }
    missing = [str(p.relative_to(root)) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing upstream input(s): " + ", ".join(missing))
    hashes = {name: file_hash(p) for name, p in paths.items()}
    source = parse_turtle(paths["source"])
    candidate = parse_turtle(paths["candidate"])
    refined = parse_turtle(paths["refined"])
    proposal = load_json(paths["proposal"])
    audit = load_json(paths["audit"])
    summary = load_json(paths["summary"])
    if (summary.get("status") != "PASS_REFINED_UNAPPROVED"
        or audit.get("status") != "REFINED_UNAPPROVED"
        or summary.get("merged_into_source") is not False
        or audit.get("no_merge") is not True):
        raise ValueError("Step 3A has not passed its required unmerged status")
    if not (hashes["source"] == summary.get("source_sha256")
            == audit.get("source_sha256")
            == proposal.get("provenance", {}).get("source_sha256")):
        raise ValueError("Source hash differs from Step 2/3A provenance")
    if hashes["proposal"] != audit.get("proposal_sha256"):
        raise ValueError("Step 3A audit references a different proposal")
    if hashes["candidate"] != audit.get("step3_candidate_sha256"):
        raise ValueError("Step 3A audit references a different original candidate")
    if audit.get("refined_triples") != len(refined) or summary.get("refined_triples") != len(refined):
        raise ValueError("Refined TBox count differs from Step 3A report")
    if audit.get("baseline_triples") != len(candidate):
        raise ValueError("Original candidate count differs from Step 3A audit")
    if summary.get("refinement_operations") != len(audit.get("edits", [])):
        raise ValueError("Refinement operation count differs from audit")
    # Reconstruct Step 3A's explicit delta and confirm *exact* graph equality.
    expected = Graph()
    for t in candidate:
        expected.add(t)
    edits = audit.get("edits", [])
    if len(edits) != 5:
        raise ValueError("Expected exactly five recorded Step 3A refinements")
    for edit in edits:
        s = URIRef(edit["subject"])
        p = URIRef(edit["predicate"])
        def make_term(value):
            return Literal(value, lang="en") if p == SKOS.definition else URIRef(value)
        new = make_term(edit["after"])
        if edit["action"] == "REPLACE_TRIPLE_OBJECT":
            old = make_term(edit["before"])
            if (s, p, old) not in expected:
                raise ValueError("Step 3A audit before-value not found in candidate")
            expected.remove((s, p, old))
        elif edit["action"] != "ADD_TRIPLE":
            raise ValueError("Unexpected Step 3A audit edit action")
        expected.add((s, p, new))
    if set(expected) != set(refined):
        raise ValueError("Refined OWL file does not equal audited Step 3A changes")
    if len(refined) != 63:
        raise ValueError("Unexpected refined TBox size: Step 3A 63-triple policy changed")
    if any(isinstance(x, BNode) for t in refined for x in t):
        raise ValueError("Refined TBox unexpectedly contains blank nodes")
    if any(True for _ in refined.subjects(RDF.type, OWL.NamedIndividual)):
        raise ValueError("Refined TBox unexpectedly declares named individuals")
    imports_hashes = proposal.get("provenance", {}).get("local_import_hashes", {})
    imports_dir = root / "imports"
    if not isinstance(imports_hashes, dict) or not imports_hashes:
        raise ValueError("Missing Step 2 imported-ontology provenance")
    from generate_rail_axioms import SUPPORTED_EXTENSIONS  # already required by Step 3A
    found = {
        "imports/" + p.relative_to(imports_dir).as_posix(): p
        for p in imports_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    } if imports_dir.is_dir() else {}
    if set(imports_hashes) != set(found) or any(file_hash(found[k]) != v for k,v in imports_hashes.items()):
        raise ValueError("Local imports changed since Step 2; rerun upstream pipeline")
    # Exact class/property set for current eight-class / six-property design.
    expected_count = {OWL.Class: 8, OWL.ObjectProperty: 4, OWL.DatatypeProperty: 2}
    for typ, count in expected_count.items():
        found_terms = {s for s in refined.subjects(RDF.type, typ) if isinstance(s, URIRef)}
        if len(found_terms) != count or any(not str(s).startswith(RAIL_NS) for s in found_terms):
            raise ValueError(f"Unexpected rail TBox {typ} declarations")
    return source, candidate, refined, proposal, audit, summary, hashes


def declared_terms(g: Graph) -> dict[URIRef, list[URIRef]]:
    return {typ: sorted(set(g.subjects(RDF.type, typ)), key=str) for typ in TERMS}


def exact_assertions(g: Graph, term: URIRef, predicate: URIRef) -> list:
    return sorted(set(g.objects(term, predicate)), key=str)


def create_queries(g: Graph) -> dict[str, str]:
    sets = declared_terms(g)
    classes, objects, datas = (sets[t] for t in TERMS)
    all_terms = sorted(classes + objects + datas, key=str)
    props = sorted(objects + datas, key=str)
    def values(terms):
        return " ".join(x.n3() for x in terms)
    def query(body):
        return ("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
                "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
                "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n"
                "# Zero rows means this quality check passes.\n" + body.strip() + "\n")
    out = {}
    for slug, term_list, class_iri in (
        ("01_missing_class_declarations", classes, OWL.Class),
        ("02_missing_object_property_declarations", objects, OWL.ObjectProperty),
        ("03_missing_data_property_declarations", datas, OWL.DatatypeProperty),
    ):
        out[slug] = query(f"SELECT ?entity WHERE {{ VALUES ?entity {{ {values(term_list)} }} FILTER NOT EXISTS {{ ?entity rdf:type {class_iri.n3()} }} }}")
    for slug, predicate in (("04_missing_english_label", RDFS.label),
                            ("05_missing_english_definition", SKOS.definition)):
        out[slug] = query(f'''SELECT ?entity WHERE {{
          VALUES ?entity {{ {values(all_terms)} }}
          FILTER NOT EXISTS {{ ?entity {predicate.n3()} ?annotation .
            FILTER (LANGMATCHES(LANG(?annotation), "en") && STRLEN(STR(?annotation)) > 0) }}
        }}''')
    parent_pairs = sorted((s, o) for s in classes for o in g.objects(s, RDFS.subClassOf))
    if not parent_pairs:
        raise ValueError("Expected candidate classes to have superclass axioms")
    pairs = " ".join(f"({s.n3()} {p.n3()})" for s,p in parent_pairs)
    out["06_missing_required_superclass"] = query(f'''SELECT ?child ?parent WHERE {{
      VALUES (?child ?parent) {{ {pairs} }}
      FILTER NOT EXISTS {{ ?child rdfs:subClassOf ?parent }}
    }}''')
    triplets = []
    for prop in props:
        domains = exact_assertions(g, prop, RDFS.domain)
        ranges = exact_assertions(g, prop, RDFS.range)
        if len(domains) != 1 or len(ranges) != 1:
            raise ValueError(f"Expected exactly one domain and range for {prop}")
        triplets.append((prop, domains[0], ranges[0]))
    expected_prop_values = " ".join(f"({p.n3()} {d.n3()} {r.n3()})" for p,d,r in triplets)
    out["07_incorrect_property_domain_or_range"] = query(f'''SELECT ?property ?domain ?range WHERE {{
      VALUES (?property ?domain ?range) {{ {expected_prop_values} }}
      FILTER (NOT EXISTS {{ ?property rdfs:domain ?domain }} ||
              NOT EXISTS {{ ?property rdfs:range ?range }})
    }}''')
    out["08_subclass_cycle"] = query(f'''SELECT ?child WHERE {{
      VALUES ?child {{ {values(classes)} }}
      ?child rdfs:subClassOf+ ?child .
    }}''')
    # Air analogues are disallowed as equivalents, not as RDF references.
    air_analogues = sorted({URIRef(i) for i in (
        "https://www.commoncoreontologies.org/AgentOntology/Airplane",
        "https://www.commoncoreontologies.org/AgentOntology/airport",
        "https://www.commoncoreontologies.org/AgentOntology/Flight_plan",
        "https://www.commoncoreontologies.org/AgentOntology/boarding_pass",
        "https://www.commoncoreontologies.org/AgentOntology/passenger",
    )}, key=str)
    out["09_no_rail_air_equivalence"] = query(f'''SELECT ?rail ?air WHERE {{
      VALUES ?rail {{ {values(classes)} }}
      VALUES ?air {{ {values(air_analogues)} }}
      {{ ?rail owl:equivalentClass ?air }} UNION {{ ?air owl:equivalentClass ?rail }}
    }}''')
    out["10_no_named_individuals_in_candidate"] = query('''SELECT ?person WHERE {
       ?person a owl:NamedIndividual .
    }''')
    bad = " ".join(f"({p.n3()} {OWL.DatatypeProperty.n3()})" for p in objects) + " " + \
          " ".join(f"({p.n3()} {OWL.ObjectProperty.n3()})" for p in datas)
    out["11_no_property_kind_conflicts"] = query(f'''SELECT ?property ?wrongType WHERE {{
      VALUES (?property ?wrongType) {{ {bad} }}
      ?property rdf:type ?wrongType .
    }}''')
    return out


def create_shapes(g: Graph) -> Graph:
    terms = declared_terms(g)
    output = Graph()
    for typ in TERMS:
        for node in terms[typ]:
            local = str(node).removeprefix(RAIL_NS)
            if not re.fullmatch(r"[a-z][a-z0-9_]*", local):
                raise ValueError("Unexpected local rail entity IRI")
            shape = URIRef(QC_NS + local + "_Shape")
            output.add((shape, RDF.type, SH.NodeShape))
            output.add((shape, SH.targetNode, node))
            output.add((shape, SH.closed, Literal(False)))
            # SHACL targets are *the named TBox vocabulary nodes*, not instances.
            assertions = [(RDF.type, typ)]
            if typ == OWL.Class:
                assertions += [(RDFS.subClassOf, parent) for parent in exact_assertions(g, node, RDFS.subClassOf)]
            else:
                assertions += [(RDFS.domain, domain) for domain in exact_assertions(g, node, RDFS.domain)]
                assertions += [(RDFS.range, value) for value in exact_assertions(g, node, RDFS.range)]
            annotations = [(RDFS.label, exact_assertions(g, node, RDFS.label)),
                           (SKOS.definition, exact_assertions(g, node, SKOS.definition))]
            for predicate, annotation_values in annotations:
                if len(annotation_values) != 1 or not isinstance(annotation_values[0], Literal) or annotation_values[0].language != "en":
                    raise ValueError(f"Expected exactly one English annotation: {node}")
                assertions.append((predicate, annotation_values[0]))
            for i, (predicate, value) in enumerate(assertions):
                prop_shape = URIRef(QC_NS + local + "_p" + str(i).zfill(2))
                output.add((shape, SH.property, prop_shape))
                output.add((prop_shape, RDF.type, SH.PropertyShape))
                output.add((prop_shape, SH.path, predicate))
                output.add((prop_shape, SH.hasValue, value))
                output.add((prop_shape, SH.minCount, Literal(1)))
                if predicate in (RDFS.label, SKOS.definition, RDFS.domain, RDFS.range):
                    output.add((prop_shape, SH.maxCount, Literal(1)))
    return output


def validate(root: Path, *, generate_only: bool = False) -> dict:
    root = root.resolve()
    source, candidate, refined, proposal, audit, summary, hashes = require_inputs(root)
    input_paths = {
        "source": root / "src/module.ttl",
        "candidate": root / "generated/rail_axioms_candidates.ttl",
        "refined": root / "generated/rail_axioms_refined.ttl",
        "proposal": root / "generated/rail_pattern_proposal.json",
        "audit": root / "reports/step3a_refinement_audit.json",
        "summary": root / "reports/step3a_summary.json",
    }
    generated_queries = create_queries(refined)
    shapes = create_shapes(refined)
    # Validate generated SPARQL syntax and run queries on merged (in-memory) graph.
    combined = Graph()
    for graph in (source, refined):
        for triple in graph:
            combined.add(triple)
    results = []
    for slug, q in sorted(generated_queries.items()):
        target = refined if slug == "10_no_named_individuals_in_candidate" else combined
        rows = list(target.query(q))
        results.append({
            "check": slug,
            "status": "PASS" if not rows else "FAIL",
            "row_count": len(rows),
            "examples": [[str(value) for value in row] for row in rows[:10]],
        })
    failed_checks = [r["check"] for r in results if r["status"] != "PASS"]

    # Render the deterministic rule artifacts, irrespective of validation status.
    query_dir = root / "generated/sparql_qc"
    query_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {slug + ".rq" for slug in generated_queries}
    for stale in query_dir.glob("*.rq"):
        if stale.name not in expected_names:
            stale.unlink()   # remove only stale outputs owned by this stage
    for slug, query in generated_queries.items():
        atomic_text(query_dir / (slug + ".rq"), query)
    shape_path = root / "generated/rail_tbox_shapes.ttl"
    atomic_text(shape_path, sorted_turtle(shapes, "Generated Step 4 rail TBox SHACL constraints"))
    # Round-trip read the generated shapes from disk; no hidden blank-node state.
    if set(parse_turtle(shape_path)) != set(shapes):
        raise RuntimeError("SHACL shape round-trip mismatch")

    shacl_result = {"executed": False, "conforms": None, "detail": "Not executed"}
    if not generate_only:
        try:
            from pyshacl import validate as pyshacl_validate
        except ImportError as exc:
            raise RuntimeError(
                "pyshacl is not installed in this Python environment. "
                "Run: python -m pip install pyshacl; then rerun Step 4. "
                "The shapes and SPARQL queries were generated, but SHACL was NOT validated."
            ) from exc
        conforms, report_graph, report_text = pyshacl_validate(
            data_graph=combined,
            shacl_graph=shapes,
            inference="none",  # Don't hide missing explicit TBox assertions
            advanced=False,
            allow_infos=False,
            allow_warnings=False,
            abort_on_first=False,
        )
        shacl_result = {
            "executed": True,
            "conforms": bool(conforms),
            "report_text": str(report_text)[:12000],
        }
        atomic_text(root / "reports/step4_shacl_report.txt", str(report_text).strip() + "\n")
    # Fail if any original file was changed during processing.
    for key, path in input_paths.items():
        if file_hash(path) != hashes[key]:
            raise RuntimeError(f"Upstream file modified during Step 4: {key}")
    passed = not failed_checks and shacl_result["conforms"] is True
    status = ("PASS_QC_AND_SHACL_UNAPPROVED" if passed else
              "GENERATED_ONLY_NOT_SHACL_VALIDATED" if generate_only and not failed_checks else
              "FAIL_QC_OR_SHACL")
    payload = {
        "version": VERSION,
        "status": status,
        "source_unchanged": file_hash(input_paths["source"]) == hashes["source"],
        "source_sha256": hashes["source"],
        "refined_sha256": hashes["refined"],
        "source_triples": len(source),
        "imports_verified_against_step2_provenance": len(proposal["provenance"]["local_import_hashes"]),
        "pending_step3a_design_decisions": audit.get("open_design_decisions", []),
        "refined_triples": len(refined),
        "combined_triples": len(combined),
        "generated_query_count": len(generated_queries),
        "generated_shacl_node_shapes": len(set(shapes.subjects(RDF.type, SH.NodeShape))),
        "sparql_passed": len(failed_checks) == 0,
        "failed_sparql_checks": failed_checks,
        "shacl": shacl_result,
        "merged_into_source": False,
        "named_individuals_created": False,
        "review_required_before_merge": True,
        "scope_note": "Explicit candidate TBox QC only, not OWL consistency or ontological approval",
    }
    atomic_json(root / "reports/step4_qc_results.json", {"checks": results, "status": status})
    atomic_json(root / "reports/step4_summary.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--generate-only", action="store_true",
                        help="Generate queries/shapes and run SPARQL; do NOT claim SHACL passed")
    args = parser.parse_args()
    try:
        result = validate(args.project_root, generate_only=args.generate_only)
    except Exception as exc:
        print("STEP 4 — FAIL: " + str(exc), file=sys.stderr)
        return 1
    print("STEP 4 — " + result["status"])
    print(json.dumps({k:v for k,v in result.items() if k != "shacl"}, indent=2))
    return 0 if result["status"] in {"PASS_QC_AND_SHACL_UNAPPROVED", "GENERATED_ONLY_NOT_SHACL_VALIDATED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
