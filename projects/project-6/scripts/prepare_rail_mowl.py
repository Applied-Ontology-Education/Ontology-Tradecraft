"""Project 6 Step 5A: reproducible 80/20, no-leakage taxonomy data for mOWL.

Projects *named subclass axioms* from the source, six imported ontologies,
and the refined rail candidate. Blank-node GCAs, ABox and non-EL features are
not silently presented as trained OWL logic. Never edits src/module.ttl.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL

from validate_rail_tbox import require_inputs, file_hash, atomic_json, atomic_text, sorted_turtle
from generate_rail_axioms import SUPPORTED_EXTENSIONS, parse_graph

VERSION = "project6-mowl-80-20-v2"
BASE_NS = "https://example.org/ontologies/intel-ontology#"
AGENT_NS = "https://www.commoncoreontologies.org/AgentOntology/"
RAIL_ROLE_PAIR = (BASE_NS + "rail_passenger_role", AGENT_NS + "passenger")
EXTENSIONS = SUPPORTED_EXTENSIONS


def load_ontology(path: Path) -> Graph:
    # Match the parser and extension rules already verified in Steps 3/4.
    return parse_graph(path)


def pairs(g: Graph) -> set[tuple[str, str]]:
    return {(str(a), str(b)) for a,b in g.subject_objects(RDFS.subClassOf)
            if isinstance(a, URIRef) and isinstance(b, URIRef) and a != b}


def reachable(edges: set[tuple[str,str]], start: str, target: str) -> bool:
    parents = defaultdict(set)
    for a,b in edges:
        parents[a].add(b)
    q = [start]
    seen = {start}
    while q:
        x=q.pop()
        if x==target:
            return True
        for y in parents[x]:
            if y not in seen:
                seen.add(y); q.append(y)
    return False


def degrees(edges: set[tuple[str,str]]) -> dict[str,int]:
    deg=defaultdict(int)
    for a,b in edges:
        deg[a]+=1; deg[b]+=1
    return deg


def select_holdout(base: set[tuple[str,str]], candidate_train: set[tuple[str,str]],
                   seed: int = 42, fraction: float = .20) -> tuple[set, set]:
    """Reserve source/import edges only if removing them maintains endpoint support
    and they are not reachable through remaining training and rail edges."""
    rng = random.Random(seed)
    choices = sorted(base)
    rng.shuffle(choices)
    # Do not fabricate a validation split when no eligible edges exist.
    target = max(1, round(len(base)*fraction))
    train = set(base)
    hold = set()
    for edge in choices:
        if len(hold) >= target:
            break
        trial = train - {edge}
        support = degrees(trial | candidate_train)
        if support[edge[0]] == 0 or support[edge[1]] == 0:
            continue
        if reachable(trial | candidate_train, *edge):
            continue
        train.remove(edge)
        hold.add(edge)
    if not hold:
        raise ValueError("No source/import edges eligible for nonleaking held-out evaluation")
    if (train | candidate_train) & hold:
        raise AssertionError("Direct leakage")
    if any(reachable(train | candidate_train, *e) for e in hold):
        raise AssertionError("Transitive subclass leakage")
    return train, hold


def select_rail_holdout(base: set[tuple[str,str]],
                        candidate: set[tuple[str,str]], *,
                        seed: int = 42, fraction: float = .20) -> tuple[set, set]:
    """Create a seeded 80/20 candidate split without cold-start or path leakage.

    Each held-out edge must retain *both* endpoint nodes in at least one
    training taxonomy edge; OWL.Class declarations alone do NOT count as
    learned embedding support. A candidate is not eligible when its child has
    no other subclass connections, or its parent is only linked by that edge.

    If the requested number of held-out edges cannot be obtained, fail rather
    than silently reducing the rail validation sample or making up evidence.
    """
    if not candidate:
        raise ValueError("No rail candidate subclass axioms available")
    target = max(1, round(len(candidate) * fraction))
    rng = random.Random(seed)
    choices = sorted(candidate)
    rng.shuffle(choices)
    train = set(candidate)
    hold = set()
    for edge in choices:
        if len(hold) >= target:
            break
        trial = train - {edge}
        remaining = base | trial
        support = degrees(remaining)
        if support[edge[0]] == 0 or support[edge[1]] == 0:
            continue
        if reachable(remaining, *edge):
            continue
        train.remove(edge)
        hold.add(edge)
    if len(hold) != target:
        raise ValueError(
            f"Rail 80/20 split requires {target} held-out candidate subclass pairs "
            f"out of {len(candidate)}; only {len(hold)} retain training support "
            "without direct/transitive leakage. More independently supported "
            "rail axioms are needed; do not count cold-start edges as cosine validation."
        )
    if train & hold or any(reachable(base | train, *e) for e in hold):
        raise AssertionError("Candidate train/validation leakage")
    return train, hold


def assemble_data(training_edges: set[tuple[str,str]],
                  validation_edges: set[tuple[str,str]],
                  classes: set[str]) -> tuple[Graph,Graph]:
    train, valid = Graph(), Graph()
    # Declarations allow mOWL to index otherwise isolated rail classes, but
    # no random embedding is accepted as independently grounded evidence.
    for iri in sorted(classes):
        train.add((URIRef(iri), RDF.type, OWL.Class))
    for child,parent in sorted(training_edges):
        train.add((URIRef(child),RDFS.subClassOf,URIRef(parent)))
    for child,parent in sorted(validation_edges):
        valid.add((URIRef(child),RDFS.subClassOf,URIRef(parent)))
    return train,valid


def confirm_step4(root: Path):
    source, _candidate, refined, proposal, audit, _step3a, hashes = require_inputs(root)
    report = root/"reports/step4_summary.json"
    qc = root/"reports/step4_qc_results.json"
    shacl = root/"reports/step4_shacl_report.txt"
    for p in (report,qc,shacl):
        if not p.is_file():
            raise FileNotFoundError(f"Missing verified Step 4 result: {p}")
    summary = json.loads(report.read_text(encoding="utf-8"))
    qc_data = json.loads(qc.read_text(encoding="utf-8"))
    if not (summary.get("status") == "PASS_QC_AND_SHACL_UNAPPROVED"
            and summary.get("sparql_passed") is True
            and summary.get("shacl",{}).get("executed") is True
            and summary.get("shacl",{}).get("conforms") is True
            and summary.get("source_sha256") == hashes["source"]
            and summary.get("refined_sha256") == hashes["refined"]
            and summary.get("merged_into_source") is False
            and qc_data.get("status") == summary["status"]
            and len(qc_data.get("checks",[])) == summary.get("generated_query_count")
            and all(c.get("status") == "PASS" for c in qc_data["checks"])
            and "Conforms: True" in shacl.read_text(encoding="utf-8")):
        raise ValueError("Step 4 QC/SHACL status or provenance is not verified")
    return source, refined, proposal, audit, summary, hashes


def collect_imports(root: Path, proposal: dict) -> tuple[Graph, dict]:
    found = sorted(p for p in (root / "imports").rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    expected = proposal["provenance"]["local_import_hashes"]
    actual = {"imports/"+p.relative_to(root/"imports").as_posix():file_hash(p) for p in found}
    if actual != expected:
        raise ValueError("Local imports do not match Step 2 provenance")
    g = Graph()
    for p in found:
        for t in load_ontology(p):
            g.add(t)
    return g, actual


def plan(root: Path, *, seed: int = 42, validation_fraction: float = .20) -> dict:
    root=root.resolve()
    if not (0.05 <= validation_fraction <= .40):
        raise ValueError("validation_fraction must be between 0.05 and 0.40")
    source, refined, proposal, audit, step4, hashes = confirm_step4(root)
    imports, import_hashes = collect_imports(root,proposal)
    base_graph=Graph()
    for graph in (source, imports):
        for t in graph: base_graph.add(t)
    base_pairs=pairs(base_graph)
    all_candidate=pairs(refined)
    # The source/import axioms and rail candidate axioms each receive an 80/20
    # split. This prevents a large source taxonomy from masking a 1-of-9 rail
    # holdout, as occurred in Step 5 v1.
    training_candidate, heldout_candidate = select_rail_holdout(
        base_pairs, all_candidate, seed=seed, fraction=validation_fraction)
    training_base, heldout_base = select_holdout(
        base_pairs, training_candidate, seed, validation_fraction)
    training_edges = training_base | training_candidate
    validation_edges = heldout_base | heldout_candidate
    if training_edges & validation_edges:
        raise ValueError("Direct train/validation leakage")
    if any(reachable(training_edges,*e) for e in validation_edges):
        raise ValueError("Transitive train/validation leakage")
    all_classes = {str(s) for graph in (base_graph,refined) for s in graph.subjects(RDF.type,OWL.Class) if isinstance(s,URIRef)}
    for a,b in training_edges | validation_edges: all_classes.update((a,b))
    train_graph,valid_graph=assemble_data(training_edges,validation_edges,all_classes)
    before_hashes={"source":hashes["source"],"refined":hashes["refined"],"step4_summary":file_hash(root/"reports/step4_summary.json")}
    train_path=root/"generated/mowl_rail_train.ttl"
    valid_path=root/"generated/mowl_rail_valid.ttl"
    atomic_text(train_path,sorted_turtle(train_graph,"Project 6 mOWL taxonomy-only train data (80pct source and rail; no validation edges)"))
    atomic_text(valid_path,sorted_turtle(valid_graph,"Project 6 mOWL taxonomy-only validation data, never used to fit embeddings"))
    # Safety: emitted files must be graph-equivalent to our in-memory projection.
    if set(load_ontology(train_path)) != set(train_graph) or set(load_ontology(valid_path)) != set(valid_graph):
        raise AssertionError("Deterministic ontology serialization failed round-trip")
    if (file_hash(root/"src/module.ttl") != before_hashes["source"] or
            file_hash(root/"generated/rail_axioms_refined.ttl") != before_hashes["refined"] or
            file_hash(root/"reports/step4_summary.json") != before_hashes["step4_summary"]):
        raise RuntimeError("Upstream input modified while building mOWL data")
    out={
        "version":VERSION,"status":"PASS_SPLIT_READY_FOR_MOWL",
        "source_unchanged":True,"source_sha256":hashes["source"],"refined_sha256":hashes["refined"],
        "step4_summary_sha256":before_hashes["step4_summary"],
        "import_hashes":import_hashes,
        "seed":seed,"validation_fraction_requested":validation_fraction,
        "source_import_explicit_named_subclass_pairs":len(base_pairs),
        "source_import_train_pairs":len(training_base),"source_import_holdout_pairs":len(heldout_base),
        "rail_candidate_pairs_total":len(all_candidate),
        "rail_candidate_training_pairs":len(training_candidate),
        "rail_candidate_holdout_pairs":len(heldout_candidate),
        "rail_candidate_holdouts":[{"child":a,"parent":b} for a,b in sorted(heldout_candidate)],
        "rail_candidate_holdout":({"child":RAIL_ROLE_PAIR[0], "parent":RAIL_ROLE_PAIR[1]}
                                  if RAIL_ROLE_PAIR in heldout_candidate else None),
        "overall_train_fraction":round(len(training_edges)/(len(training_edges)+len(validation_edges)),6),
        "overall_validation_fraction":round(len(validation_edges)/(len(training_edges)+len(validation_edges)),6),
        "train_axiom_pairs":len(training_edges),"valid_axiom_pairs":len(validation_edges),
        "indexed_class_declarations":len(all_classes),
        "train_sha256":file_hash(train_path),"valid_sha256":file_hash(valid_path),
        "holdout_pairs":[{"child":a,"parent":b} for a,b in sorted(validation_edges)],
        "candidate_pairs":[{"child":a,"parent":b,
                           "evaluation":"HELD_OUT" if (a,b) in heldout_candidate
                           else "IN_TRAINING_UNSCORABLE_INDEPENDENTLY"}
                           for a,b in sorted(all_candidate)],
        "projection_scope":"Named rdfs:subClassOf + owl:Class declarations only. Imported/source GCAs, restrictions, ABox and property domain/range not used in mOWL loss.",
        "limitation":"Some rail classes have a single asserted parent; these edges remain in training because holding them out would leave an unsupported class-center embedding. New rail held-outs must have other training support, and a 0.70 cosine mean is not guaranteed by the 80/20 split.",
        "source_not_merged":True,"individuals_generated":False,
    }
    atomic_json(root/"reports/step5_split_report.json",out)
    return out


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--validation-fraction",type=float,default=.20)
    a=ap.parse_args()
    try:
        r=plan(a.project_root,seed=a.seed,validation_fraction=a.validation_fraction)
    except Exception as e:
        print("STEP 5A — FAIL: " +str(e),file=sys.stderr);return 1
    print("STEP 5A — "+r["status"])
    print(json.dumps({k:r[k] for k in ("source_import_train_pairs","source_import_holdout_pairs","rail_candidate_holdout_pairs","train_axiom_pairs","valid_axiom_pairs")},indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
