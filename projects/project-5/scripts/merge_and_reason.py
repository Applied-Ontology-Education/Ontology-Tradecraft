"""Project 5 — Step 7: ROBOT merge + ELK reasoning, offline import snapshot.

Run from project-5 (or elsewhere): python scripts/merge_and_reason.py

Requires ROBOT executable on PATH OR ROBOT_JAR pointing to robot.jar OR
project-5/tools/robot.jar. Needs Java and rdflib. Uses the verified import
snapshot produced automatically by Step 5; no ontology HTTP access here.

Input:
    src/module.ttl
    generated/accepted_el.ttl
    reports/candidate_filter_summary.json
    reports/mowl_import_cache/{imported_axioms.owl,manifest.json}
Output:
    src/module_augmented.ttl
    reports/augmentation_reasoner_report.json
    reports/augmentation_elk_baseline_reasoned.ttl
    reports/augmentation_elk_reasoned.ttl

Only publishes final output after ROBOT+ELK and preservation audits pass.
The augmented ontology contains the local asserted triples and retains original
owl:imports statements. The separately saved reasoned artifact contains imports
materialized from Step 5's verified cache plus ELK inferences.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "module.ttl"
ACCEPTED = ROOT / "generated" / "accepted_el.ttl"
STEP6_SUMMARY = ROOT / "reports" / "candidate_filter_summary.json"
OUTPUT = ROOT / "src" / "module_augmented.ttl"
REPORT = ROOT / "reports" / "augmentation_reasoner_report.json"
IMPORT_SNAPSHOT = ROOT / "reports" / "mowl_import_cache" / "imported_axioms.owl"
IMPORT_MANIFEST = ROOT / "reports" / "mowl_import_cache" / "manifest.json"
BASELINE_REASONED = ROOT / "reports" / "augmentation_elk_baseline_reasoned.ttl"
AUGMENTED_REASONED = ROOT / "reports" / "augmentation_elk_reasoned.ttl"
HEAP = os.environ.get("ROBOT_JAVA_HEAP", "6g")


def load_ttl(path: Path, fmt="turtle") -> Graph:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    graph = Graph()
    graph.parse(str(path), format=fmt)
    return graph


def store_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def save_ttl(path: Path, graph: Graph) -> None:
    graph.serialize(destination=str(path), format="turtle")
    if set(load_ttl(path)) != set(graph):
        raise RuntimeError(f"Temporary TTL failed round-trip verification: {path}")


def clone_graph(graph: Graph) -> Graph:
    copied = Graph()
    for prefix, iri in graph.namespaces():
        copied.bind(prefix, iri)
    for triple in graph:
        copied.add(triple)
    return copied


def named_subclasses(graph: Graph) -> set[tuple[str, str]]:
    return {
        (str(a), str(b))
        for a, b in graph.subject_objects(RDFS.subClassOf)
        if isinstance(a, URIRef) and isinstance(b, URIRef) and a != b
    }


def get_robot_command() -> tuple[list[str], str]:
    """Prefer explicit ROBOT_JAR, then project-local jar, then executable."""
    jar_candidates = []
    if os.environ.get("ROBOT_JAR"):
        jar_candidates.append(Path(os.environ["ROBOT_JAR"]).expanduser())
    jar_candidates.extend([
        ROOT / "tools" / "robot.jar",
        ROOT / "robot.jar",
        ROOT.parents[1] / "tools" / "robot.jar",
    ])
    for path in jar_candidates:
        if path.is_file():
            java = shutil.which("java")
            if not java:
                raise RuntimeError("Java executable is not available on PATH")
            return [java, f"-Xmx{HEAP}", "-jar", str(path.resolve())], str(path.resolve())
    for name in ("robot", "robot.exe", "robot.bat", "robot.cmd"):
        executable = shutil.which(name)
        if executable:
            return [executable], executable
    raise RuntimeError(
        "ROBOT not found. Place robot.jar in project-5/tools/robot.jar, "
        "set ROBOT_JAR, or add the ROBOT launcher to PATH. "
        "Download from https://github.com/ontodev/robot/releases"
    )


def run_robot(command: list[str], args: list[str], label: str) -> str:
    actual = [*command, *args]
    print(f"ROBOT: {label}", flush=True)
    # stdout and stderr are captured for auditing. No shell=True, even on Windows.
    completed = subprocess.run(
        actual, cwd=str(ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, errors="replace",
        timeout=600, check=False,
    )
    output = completed.stdout or ""
    if output.strip():
        print(output[-3000:].strip(), flush=True)
    if completed.returncode:
        raise RuntimeError(
            f"ROBOT {label} exited {completed.returncode}.\n"
            f"Command: {' '.join(map(str, actual))}\n"
            f"Output (tail):\n{output[-7000:]}"
        )
    return output


def validate_snapshot(source: Graph) -> dict:
    if not IMPORT_MANIFEST.is_file() or not IMPORT_SNAPSHOT.is_file():
        raise RuntimeError(
            "Step 5 import cache is missing. Run Step 5 successfully first; "
            "expected reports/mowl_import_cache/{manifest.json,imported_axioms.owl}."
        )
    manifest = json.loads(IMPORT_MANIFEST.read_text(encoding="utf-8"))
    direct_imports = sorted({str(iri) for iri in source.objects(None, OWL.imports)})
    sha = hashlib.sha256(IMPORT_SNAPSHOT.read_bytes()).hexdigest()
    if not (
        manifest.get("cache_schema") == 1
        and manifest.get("direct_import_iris") == direct_imports
        and manifest.get("snapshot_sha256") == sha
    ):
        raise RuntimeError(
            "Step 5 import snapshot is stale or corrupt: import IRIs/checksum mismatch. "
            "Regenerate it with Step 5; refusing an incomplete ELK reasoning check."
        )
    imported = load_ttl(IMPORT_SNAPSHOT, fmt="xml")
    if any(imported.triples((None, OWL.imports, None))):
        raise RuntimeError("Import snapshot unexpectedly contains owl:imports")
    if direct_imports and len(imported) == 0:
        raise RuntimeError("Import snapshot is empty despite declared imports")
    return {
        "direct_imports": direct_imports,
        "snapshot_sha256": sha,
        "imported_ontologies": manifest.get("imported_ontologies"),
        "imported_axioms": manifest.get("imported_axioms"),
        "snapshot_path": str(IMPORT_SNAPSHOT),
    }


def is_robot_serialization_extra(triple, merged: Graph, accepted_pairs: list[tuple[str, str]]) -> bool:
    """Allow only harmless OWLAPI serialization-generated declarations/metadata.

    ROBOT may explicitly emit OWL class declarations for the endpoint IRIs of
    newly added axioms, even if those were only declared by imported ontologies.
    It may also declare four standard SKOS/Dublin Core annotation properties
    when these are used by the source ontology.
    It may also emit an ontology header when serializing an anonymous ontology.
    Never allow new logical relations, restrictions, or owl:imports as metadata.
    """
    subject, predicate, obj = triple
    endpoints = {URIRef(iri) for edge in accepted_pairs for iri in edge}
    if predicate == RDF.type and obj == OWL.Class and subject in endpoints:
        return True
    # OWLAPI/ROBOT explicitly declares standard annotation properties when
    # reserializing annotations found in CCO. These four declarations are
    # harmless vocabulary declarations, NOT new subclass relationships.
    # This is an exact allowlist, not a blanket exception for rdf:type.
    standard_annotation_properties = {
        URIRef("http://www.w3.org/2004/02/skos/core#example"),
        URIRef("http://www.w3.org/2004/02/skos/core#definition"),
        URIRef("http://www.w3.org/2004/02/skos/core#altLabel"),
        URIRef("http://purl.org/dc/terms/license"),
    }
    if (predicate == RDF.type and obj == OWL.AnnotationProperty
            and subject in standard_annotation_properties):
        return True
    if predicate == RDF.type and obj == OWL.Ontology:
        return True
    # ROBOT can serialize an ontology IRI declaration/header annotation.
    # Allow only recognized annotation properties, on a declared ontology IRI.
    ontology_subjects = set(merged.subjects(RDF.type, OWL.Ontology))
    if subject in ontology_subjects and predicate in {
        RDFS.label, RDFS.comment, OWL.versionIRI, OWL.versionInfo,
        OWL.priorVersion, OWL.backwardCompatibleWith,
        OWL.incompatibleWith,
    }:
        return True
    return False


def normalize_robot_local_merge(
    source_without_imports: Graph,
    accepted: Graph,
    robot_result: Graph,
    accepted_pairs: list[tuple[str, str]],
) -> tuple[Graph, list[tuple]]:
    """Validate ROBOT logical merge, removing only benign auto-added headers.

    Publish exactly the RDF assertions provided by source and Step 6; ROBOT is
    the merge tool, while RDFLib handles lossless output normalization.
    """
    expected = set(source_without_imports) | set(accepted)
    actual = set(robot_result)
    missing = expected - actual
    surprises = actual - expected
    disallowed = [t for t in surprises
                  if not is_robot_serialization_extra(t, robot_result, accepted_pairs)]
    if missing or disallowed:
        preview = "\n".join(repr(t) for t in disallowed[:10])
        missing_preview = "\n".join(repr(t) for t in list(missing)[:5])
        raise RuntimeError(
            "ROBOT merge changed asserted ontology semantics: "
            f"missing={len(missing)}, unexpected_logical={len(disallowed)}, "
            f"serialization_extras={len(surprises)-len(disallowed)}.\n"
            f"Missing examples:\n{missing_preview}\n"
            f"Unexpected examples:\n{preview}"
        )
    cleaned = clone_graph(source_without_imports)
    for triple in accepted:
        cleaned.add(triple)
    return cleaned, sorted(surprises, key=str)


def assert_no_missing_or_surprises(
    original: Graph, accepted: Graph, merged: Graph,
) -> tuple[set, set]:
    before = set(original)
    after = set(merged)
    missing = before - after
    if missing:
        raise RuntimeError(
            f"ROBOT output lost {len(missing)} source RDF triples. "
            f"First lost triple: {next(iter(missing))}"
        )
    expected_added = set(accepted) - before
    new = after - before
    if new != expected_added:
        unexpected = new - expected_added
        absent = expected_added - new
        raise RuntimeError(
            "ROBOT merge differs from source + accepted triples: "
            f"unexpected={len(unexpected)}, absent={len(absent)}. "
            "Inspect serialization differences rather than silently changing data."
        )
    return expected_added, new


def main() -> None:
    print("=" * 70)
    print("PROJECT 5 - STEP 7: ROBOT MERGE + ELK REASONING")
    print("=" * 70)
    original = load_ttl(SOURCE)
    accepted = load_ttl(ACCEPTED)
    if not STEP6_SUMMARY.is_file():
        raise FileNotFoundError(f"Missing Step 6 report: {STEP6_SUMMARY}")
    step6 = json.loads(STEP6_SUMMARY.read_text(encoding="utf-8"))
    accepted_pairs = sorted(named_subclasses(accepted))
    if not accepted_pairs or len(accepted_pairs) != len(accepted):
        raise RuntimeError(
            "accepted_el.ttl must contain at least one named subclass axiom "
            "and no other triples"
        )
    if step6.get("accepted_provisional_axioms") != len(accepted_pairs):
        raise RuntimeError(
            "Step 6 accepted-axiom count disagrees with accepted_el.ttl"
        )
    expected_local = set(accepted) - set(original)
    if not expected_local:
        raise RuntimeError("No new asserted subclass triples to add")
    print(f"Original triples: {len(original)}")
    print(f"Step 6 accepted subclass axioms: {len(accepted_pairs)}")
    print(f"New asserted subclass triples: {len(expected_local)}")

    snapshot = validate_snapshot(original)
    print("Using verified Step 5 imported-axiom snapshot (offline)")
    robot, robot_location = get_robot_command()
    print(f"ROBOT executable: {robot_location}")
    print("Reasoner: ELK (OWL 2 EL; not a full OWL 2 DL completeness guarantee)")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    # A temporary workspace avoids stale intermediate data and prevents accidentally
    # publishing a failed run. Files are removed when the block exits.
    with tempfile.TemporaryDirectory(prefix="project5_robot_", dir=str(REPORT.parent)) as tmp:
        workspace = Path(tmp)
        source_no_imports = workspace / "source_local.ttl"
        accepted_local = workspace / "accepted_local.ttl"
        merged_no_imports = workspace / "merged_local.ttl"
        bundle_before = workspace / "reason_input_original.ttl"
        bundle_after = workspace / "reason_input_augmented.ttl"
        baseline_reasoned = workspace / "baseline_reasoned.ttl"
        augmented_reasoned = workspace / "augmented_reasoned.ttl"
        final_pending = workspace / "module_augmented.ttl"

        # OWLAPI tries to resolve owl:imports even for a merge that doesn't collapse
        # imports. Strip import statements from temporary ROBOT inputs only; use
        # the verified complete imported logical axioms snapshot for reasoning.
        stripped = clone_graph(original)
        stripped.remove((None, OWL.imports, None))
        save_ttl(source_no_imports, stripped)
        save_ttl(accepted_local, accepted)

        # ROBOT (not RDFLib) performs the merge of asserted ontology axioms.
        run_robot(robot, [
            "merge", "--input", str(source_no_imports),
            "--input", str(accepted_local),
            "--collapse-import-closure", "false",
            "--output", str(merged_no_imports),
        ], "merge source and accepted axioms")
        raw_robot_merge = load_ttl(merged_no_imports)
        merged_local, robot_extras = normalize_robot_local_merge(
            stripped, accepted, raw_robot_merge, accepted_pairs
        )
        print(
            f"ROBOT returned {len(robot_extras)} additional serialization "
            "triples (declarations / ontology metadata); "
            "excluded from the published local ontology."
        )
        for triple in robot_extras:
            print(f"  ROBOT serialization extra: {triple}")
        # Re-serialize the normalized ROBOT result for the next ROBOT merge.
        # This avoids accidental extra ontology declarations in the final output.
        save_ttl(merged_no_imports, merged_local)

        # A second ROBOT merge constructs each reasoner's import-complete input.
        # This is solely for ELK analysis; it is not the published local ontology.
        for name, asserted, destination in (
            ("baseline", source_no_imports, bundle_before),
            ("augmented", merged_no_imports, bundle_after),
        ):
            run_robot(robot, [
                "merge", "--input", str(asserted),
                "--input", str(IMPORT_SNAPSHOT),
                "--collapse-import-closure", "false",
                "--output", str(destination),
            ], f"build offline {name} ELK input")

        # ROBOT reason performs validation BEFORE classification and fails with a
        # non-zero exit status for inconsistency or unsatisfiable named classes.
        for name, inp, result in (
            ("original", bundle_before, baseline_reasoned),
            ("augmented", bundle_after, augmented_reasoned),
        ):
            run_robot(robot, [
                "reason", "--reasoner", "ELK",
                "--remove-redundant-subclass-axioms", "false",
                "--include-indirect", "true",
                "--exclude-tautologies", "structural",
                "--input", str(inp),
                "--output", str(result),
            ], f"ELK check/classification: {name}")
            if not result.is_file():
                raise RuntimeError(f"ROBOT reason returned without writing {result}")

        # The published artifact retains the exact original import statements.
        augmented_local = clone_graph(merged_local)
        for triple in original.triples((None, OWL.imports, None)):
            augmented_local.add(triple)
        added, _ = assert_no_missing_or_surprises(original, accepted, augmented_local)
        save_ttl(final_pending, augmented_local)
        disk = load_ttl(final_pending)
        assert_no_missing_or_surprises(original, accepted, disk)

        baseline_reasoned_graph = load_ttl(baseline_reasoned)
        augmented_reasoned_graph = load_ttl(augmented_reasoned)
        # With --include-indirect true, ROBOT's ELK classification includes
        # indirect named superclass relationships. Refuse to count an accepted
        # axiom that was already entailed in the baseline EL classification.
        baseline_entailments = named_subclasses(baseline_reasoned_graph)
        redundant = sorted(set(accepted_pairs) & baseline_entailments)
        if redundant:
            raise RuntimeError(
                f"{len(redundant)} Step 6 candidate axioms were already ELK-entailed "
                f"before augmentation: {redundant}. Refusing to count these as new."
            )
        new_reasoned_triples = set(augmented_reasoned_graph) - set(baseline_reasoned_graph)
        # Report inferred, separately from the two explicitly accepted additions.
        inferred_extra = new_reasoned_triples - added

        # Publish the validated final files, then the report.
        shutil.copyfile(baseline_reasoned, BASELINE_REASONED)
        shutil.copyfile(augmented_reasoned, AUGMENTED_REASONED)
        final_pending.replace(OUTPUT)

    result = {
        "stage": 7,
        "status": "PASS",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "reasoner": "ELK (via ROBOT command-line)",
        "merge_tool": "ROBOT merge",
        "robot_location": robot_location,
        "source_file": str(SOURCE),
        "accepted_file": str(ACCEPTED),
        "augmented_file": str(OUTPUT),
        "original_triples": len(original),
        "augmented_triples": len(disk),
        "new_local_triples": len(added),
        "robot_serialization_extra_triples_excluded": len(robot_extras),
        "robot_serialization_extras": [list(map(str, t)) for t in robot_extras],
        "accepted_axioms_from_step_6": len(accepted_pairs),
        "axioms_added": [
            {"child_iri": a, "parent_iri": b} for a, b in accepted_pairs
            if (URIRef(a), RDFS.subClassOf, URIRef(b)) in added
        ],
        "already_asserted_before_merge": len(accepted_pairs) - len(added),
        "already_entailed_before_merge_by_elk": 0,
        "baseline": {"consistent": True, "unsatisfiable_named_classes": []},
        "augmented": {"consistent": True, "unsatisfiable_named_classes": []},
        "baseline_elk_classification": str(BASELINE_REASONED),
        "augmented_elk_classification": str(AUGMENTED_REASONED),
        "reasoned_difference_triples": len(new_reasoned_triples),
        "inferred_difference_triples_not_explicit_additions": len(inferred_extra),
        "import_snapshot": snapshot,
        "limitations": (
            "ELK checks consistency and coherence for OWL 2 EL constructs it supports; "
            "it does not replace HermiT for arbitrary OWL 2 DL. "
            "Structural novelty is not proof that a proposed axiom was not "
            "already entailed by the baseline. ROBOT materializes inferred axioms "
            "into separately saved classification artifacts, not the published "
            "local ontology. Reasoned differences need not be independent axioms."
        ),
    }
    store_json(REPORT, result)
    print("=" * 70)
    print("STEP 7 RESULTS")
    print("=" * 70)
    print(f"ROBOT merged asserted axioms: {len(added)}")
    print(f"Augmented ontology triples: {len(disk)}")
    print("Baseline ELK coherence/consistency: PASS")
    print("Augmented ELK coherence/consistency: PASS")
    print(f"ELK reasoned differences (not necessarily novel axioms): {len(new_reasoned_triples)}")
    print(f"Augmented ontology: {OUTPUT}")
    print(f"Reasoner report: {REPORT}")
    print("STEP 7 STATUS: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STEP 7 STATUS: FAIL - {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
