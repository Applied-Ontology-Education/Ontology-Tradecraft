#!/usr/bin/env python3
"""Project 6: rebuild the rail TBox and ingest/validate an onboard passenger manifest.

Run from the Project 6 repository's project directory:

    python -u scripts/run_all.py --manifest templates/passenger_manifest_template.xlsx

Only project DATA inputs required before execution:
    src/module.ttl
    imports/{AgentOntology.ttl,EventOntology.ttl,InformationEntityOntology.ttl,
             bfo-core.ttl,core.ttl,iao-edit.owl}
    a structured in-transit .xlsx manifest, under templates/ or data/manifests/

All earlier stage scripts, Python packages, Java/ROBOT, and XAI_API_KEY are
runtime/software prerequisites; intermediate TTL/JSON/reports/vectors are NOT.

The pipeline archives existing generated/ and reports/ folders, regenerates all
artifacts (including fresh embeddings and live Grok requests), then writes
'generated/module_augmented.ttl' and the final populated ontology
'generated/module_augmented_with_manifest_enriched.ttl'. It never overwrites
src/module.ttl, the original six imports, or the selected Excel workbook.

Passing means the prescribed structural checks, source-scoped HermiT checks,
and approved design policy passed for this run. This is NOT evidence that all
unloaded external imports are coherent, or that Grok is deterministic.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

VERSION = "project6-run-all-tbox-manifest-scoped-abox-cosine-v3"
FINAL_OUTPUT = "generated/module_augmented_with_manifest_enriched.ttl"
V3_SCRIPTS = {
    "ingest_passenger_manifest.py": "project6-manifest-abox-v3-in-transit",
    "enrich_manifest_relationships.py": "project6-manifest-relationships-v3-in-transit",
    "validate_manifest_abox.py": "project6-manifest-semantic-validation-v3-in-transit",
}


IMPORT_NAMES = frozenset({
    "AgentOntology.ttl", "EventOntology.ttl", "InformationEntityOntology.ttl",
    "bfo-core.ttl", "core.ttl", "iao-edit.owl",
})
IMPORT_SUFFIXES = frozenset({".ttl", ".owl", ".rdf", ".xml", ".nt", ".n3", ".jsonld", ".trig"})

@dataclass(frozen=True)
class Stage:
    key: str
    script: str
    args: tuple[str, ...]
    report: str
    status: str
    outputs: tuple[str, ...]
    timeout: int
    accepts_project_root: bool = True


def stage_plan(*, epochs: int = 200, model: str = "grok-4.7",
               reasoner_timeout: int = 300, reasoner_hard_timeout: int = 900,
               robot_jar: str | None = None,
               manifest: str = "templates/passenger_manifest_template.xlsx",
               flight_documents: tuple[str, ...] = (),
               manifest_hard_timeout: int = 1200) -> tuple[Stage, ...]:
    """Explicit dependency ordering; absolutely no --reuse-vectors or --check-only."""
    if epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if not model.strip():
        raise ValueError("xAI model must be non-empty")
    if reasoner_timeout < 30 or reasoner_hard_timeout < reasoner_timeout * 2:
        raise ValueError("Reasoner timeout >=30 and hard timeout >=2x the per-run timeout are required")
    if manifest_hard_timeout < 30:
        raise ValueError("--manifest-hard-timeout must be at least 30 seconds")
    for item in flight_documents:
        if "=" not in item or not all(part.strip() for part in item.split("=", 1)):
            raise ValueError("--flight-document requires TRIP_ID=DOCUMENT_ID")
    robot_arg = ("--robot-jar", robot_jar) if robot_jar else ()
    workbook_args = ("--input", manifest)
    return (
        Stage("01_inventory", "inventory_ontology.py", (),
              "reports/step1_summary.json", "PASS",
              ("reports/ontology_inventory.json", "reports/rail_gap_analysis.json"), 240),
        Stage("02_grok_proposal", "propose_rail_patterns_llm.py", ("--model", model),
              "reports/step2_summary.json", "PASS_PROPOSAL_ONLY",
              ("generated/rail_pattern_proposal.json", "generated/rail_design_request.json"), 1200),
        Stage("03_candidate_axioms", "generate_rail_axioms.py", (),
              "reports/step3_summary.json", "PASS_CANDIDATES_REVIEW_REQUIRED",
              ("generated/rail_axioms_candidates.ttl", "reports/step3_axiom_review.json"), 300),
        Stage("03a_refine_axioms", "refine_rail_axioms.py", (),
              "reports/step3a_summary.json", "PASS_REFINED_UNAPPROVED",
              ("generated/rail_axioms_refined.ttl", "reports/step3a_refinement_audit.json"), 300),
        Stage("04_qc_shacl", "validate_rail_tbox.py", (),
              "reports/step4_summary.json", "PASS_QC_AND_SHACL_UNAPPROVED",
              ("reports/step4_qc_results.json", "reports/step4_shacl_report.txt"), 420),
        Stage("05a_train_valid_split", "prepare_rail_mowl.py", ("--seed", "42"),
              "reports/step5_split_report.json", "PASS_SPLIT_READY_FOR_MOWL",
              ("generated/mowl_rail_train.ttl", "generated/mowl_rail_valid.ttl"), 480),
        Stage("05b_mowl_and_grok", "evaluate_rail_mowl.py",
              ("--epochs", str(epochs), "--dim", "50", "--seed", "42", "--model", model, "--with-llm"),
              "reports/step5_summary.json", "PASS_HYBRID_REVIEW_ONLY",
              ("reports/step5_cosine_scores.json", "reports/step5_hybrid_review.json",
               "reports/step5_model_manifest.json", "generated/mowl_rail_class_vectors.npz"), 3600),
        Stage("05c_grounded_grok", "refine_rail_hybrid.py", ("--model", model, "--with-llm"),
              "reports/step5_v2_summary.json", "PASS_GROUNDED_HYBRID_REVIEW_ONLY",
              ("reports/step5_v2_grounded_review.json",), 1200),
        Stage("06_scoped_hermit", "validate_rail_reasoner.py",
              ("--scope", "module", "--reasoner", "hermit", "--timeout", str(reasoner_timeout),
               "--hard-timeout", str(reasoner_hard_timeout)) + robot_arg,
              "reports/step6_reasoner_summary.json", "PASS_REASONER_COHERENT_UNAPPROVED",
              ("reports/step6_reasoner_detail.json", "generated/step6_baseline_flattened.ttl",
               "generated/step6_candidate_flattened.ttl", "reports/step6_baseline_robot.log",
               "reports/step6_candidate_robot.log"), reasoner_hard_timeout + 120),
        Stage("07_controlled_merge", "merge_rail_ontology.py", (),
              "reports/step7_summary.json", "PASS_CONTROLLED_MERGE_VALIDATED_NOT_PROMOTED",
              ("reports/step7_merge_audit.json", "generated/module_augmented.ttl"), 480),
        Stage("08_manifest_ingestion", "ingest_passenger_manifest.py", workbook_args,
              "reports/manifest_ingestion_report.json", "PASS_MANIFEST_ABOX_GENERATED",
              ("generated/manifest_instances.ttl", "generated/module_augmented_with_manifest.ttl"),
              600, accepts_project_root=False),
        Stage("09_manifest_enrichment", "enrich_manifest_relationships.py",
              workbook_args + tuple(arg for item in flight_documents for arg in ("--flight-document", item)),
              "reports/manifest_relationship_report.json", "PASS_RELATIONSHIPS_ENRICHED_NOT_MERGED",
              ("generated/manifest_tbox_extension.ttl", "generated/manifest_instances_enriched.ttl",
               FINAL_OUTPUT), 600, accepts_project_root=False),
        Stage("09a_manifest_cosine", "evaluate_manifest_delta_cosine.py",
              ("--threshold", "0.70"),
              "reports/manifest_delta_cosine.json", "PASS_SCOPED_DELTA_COSINE_EVALUATED",
              ("reports/manifest_delta_cosine.json",), 900),
        Stage("10_manifest_validation", "validate_manifest_abox.py",
              ("--workbook", manifest, "--strict-occurrence", "--timeout", str(reasoner_timeout),
               "--hard-timeout", str(manifest_hard_timeout)) + robot_arg,
              "reports/manifest_semantic_validation.json", "PASS_SCOPED_HERMIT_AND_SHACL",
              ("generated/manifest_validation/reasoned_baseline.ttl",
               "generated/manifest_validation/reasoned_candidate.ttl",
               "generated/manifest_validation/manifest_validation_shapes.ttl",
               "reports/manifest_validation/baseline_hermit.log",
               "reports/manifest_validation/candidate_hermit.log"),
              manifest_hard_timeout + 180),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def required_script_paths(root: Path, plan: tuple[Stage, ...]) -> list[Path]:
    return sorted({root / "scripts" / stage.script for stage in plan})


def fingerprint_inputs(root: Path, plan: tuple[Stage, ...], manifest: Path | None = None) -> dict[str, str]:
    """Hash immutable source, local imports, and selected manifest workbook."""
    source = root / "src" / "module.ttl"
    if not source.is_file():
        raise FileNotFoundError("Required input missing: src/module.ttl")
    import_dir = root / "imports"
    if not import_dir.is_dir():
        raise FileNotFoundError("Required input directory missing: imports/")
    imports = {
        p.relative_to(import_dir).as_posix(): p
        for p in import_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMPORT_SUFFIXES
    }
    if set(imports) != IMPORT_NAMES:
        missing = sorted(IMPORT_NAMES - set(imports))
        unexpected = sorted(set(imports) - IMPORT_NAMES)
        raise ValueError(f"Expected exactly six local ontology imports; missing={missing}, unexpected={unexpected}")
    missing_scripts = [p.name for p in required_script_paths(root, plan) if not p.is_file()]
    if missing_scripts:
        raise FileNotFoundError("Missing stage scripts in scripts/: " + ", ".join(missing_scripts))
    result = {"src/module.ttl": sha256_file(source),
              **{"imports/" + k: sha256_file(path) for k, path in sorted(imports.items())}}
    if manifest is not None:
        workbook = manifest if manifest.is_absolute() else root / manifest
        workbook = workbook.resolve()
        for unsafe in (root / "generated", root / "reports", root / ".pipeline-archives"):
            if workbook.is_relative_to(unsafe.resolve()):
                raise ValueError("Manifest workbook must not be inside generated/, reports/, or .pipeline-archives/")
        if not workbook.is_file() or workbook.suffix.lower() != ".xlsx":
            raise FileNotFoundError("Required Excel manifest missing or not an .xlsx file: " + str(workbook))
        if workbook.stat().st_size == 0:
            raise ValueError("Manifest workbook is empty")
        result[str(workbook)] = sha256_file(workbook)
    return result


def preflight(root: Path, plan: tuple[Stage, ...], *, robot_jar: str | None = None,
              manifest: Path | None = None, check_runtime: bool = True) -> dict[str, str]:
    inputs = fingerprint_inputs(root, plan, manifest=manifest)
    if not check_runtime:
        return inputs
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("Use the Project 6 .venv-mowl with Python 3.10–3.12 (3.12 recommended), not Python 3.14")
    for module_name in ("rdflib", "pyshacl", "numpy", "torch", "scipy", "mowl", "jpype", "openai", "sentence_transformers"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise RuntimeError(f"Missing/broken Python package {module_name} in {sys.executable}: {exc}") from exc
    # Reject stale v1/v2 manifest scripts before spending Grok tokens.
    import ast
    for script_name, expected_version in V3_SCRIPTS.items():
        script = root / "scripts" / script_name
        syntax = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        versions = [stmt.value.value for stmt in syntax.body
                    if isinstance(stmt, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "VERSION"
                            for target in stmt.targets)
                    and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)]
        if versions != [expected_version]:
            raise RuntimeError(f"Incorrect manifest stage version: {script_name}; reinstall v3 package")
    if manifest is not None:
        # Validate row-level contract before archiving previous outputs / paying for Grok.
        import ingest_passenger_manifest
        workbook = manifest if manifest.is_absolute() else root / manifest
        rows = ingest_passenger_manifest.parse_manifest(workbook)
        ingest_passenger_manifest.verify_records(rows)
        if not rows:
            raise RuntimeError("Excel manifest has no passenger records")
    if not os.environ.get("XAI_API_KEY", "").strip():
        raise RuntimeError("XAI_API_KEY is not set in this PowerShell session; add it securely before running")
    if not shutil.which("java"):
        raise RuntimeError("Java not on PATH; set JAVA_HOME and ensure JAVA_HOME\\bin is on PATH")
    if robot_jar:
        if not Path(robot_jar).is_file():
            raise RuntimeError("ROBOT JAR not found: " + robot_jar)
    elif not shutil.which("robot"):
        raise RuntimeError("ROBOT not on PATH; install its CLI or pass --robot-jar PATH")
    if os.name == "nt":
        import jpype
        try:
            jvm_path = Path(jpype.getDefaultJVMPath())
        except Exception as exc:
            raise RuntimeError("JPype cannot locate jvm.dll; reset JAVA_HOME to the actual JDK folder") from exc
        if not jvm_path.is_file():
            raise RuntimeError("JPype JVM library missing: " + str(jvm_path))
    return inputs


def check_input_integrity(root: Path, original: dict[str, str]) -> None:
    changed = []
    for relative, expected in original.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            changed.append(relative)
    if changed:
        raise RuntimeError("Protected original input file(s) changed: " + ", ".join(changed))


def archive_previous_outputs(root: Path) -> str | None:
    """Archive prior runs rather than silently deleting or reusing intermediates."""
    existing = [root / name for name in ("generated", "reports") if (root / name).exists()]
    if not existing:
        (root / "generated").mkdir(parents=True, exist_ok=True)
        (root / "reports").mkdir(parents=True, exist_ok=True)
        return None
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    dest = root / ".pipeline-archives" / suffix
    dest.mkdir(parents=True, exist_ok=False)
    for path in existing:
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"Refusing to archive non-directory or symlink: {path}")
        shutil.move(str(path), str(dest / path.name))
    (root / "generated").mkdir(exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    return dest.relative_to(root).as_posix()


def terminate_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            import signal
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def execute_stage(root: Path, stage: Stage) -> None:
    """Stream a live log without pipe inheritance, with an outer process-tree limit."""
    log_path = root / "reports" / "pipeline_logs" / f"{stage.key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(root / "scripts" / stage.script)]
    if stage.accepts_project_root:
        cmd += ["--project-root", str(root)]
    cmd += list(stage.args)
    print(f"\n>>> {stage.key}  ({stage.script})  — log: {log_path.relative_to(root)}", flush=True)
    if stage.key == "05b_mowl_and_grok":
        print("    Training fresh ELEmbeddings; live Grok call follows (no cached vectors).", flush=True)
    pos = 0
    with log_path.open("w", encoding="utf-8", errors="replace") as stream:
        stream.write("STEP8 COMMAND: " + " ".join(cmd) + "\n")
        stream.flush()
        kwargs = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                  else {"start_new_session": True})
        process = subprocess.Popen(cmd, cwd=str(root), stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, **kwargs)
        started = time.monotonic()
        # Independent reader handle: do NOT seek on the subprocess's inherited
        # stdout descriptor (the parent and child may share its file offset).
        with log_path.open("r", encoding="utf-8", errors="replace") as reader:
            try:
                while True:
                    try:
                        exit_code = process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        exit_code = None
                    reader.seek(pos)
                    appended = reader.read()
                    pos = reader.tell()
                    if appended:
                        for line in appended.rstrip("\n").splitlines():
                            print("    " + line, flush=True)
                    if exit_code is not None:
                        if exit_code != 0:
                            raise RuntimeError(f"{stage.key} subprocess exit code {exit_code}; see {log_path.relative_to(root)}")
                        return
                    if time.monotonic() - started > stage.timeout:
                        terminate_tree(process)
                        raise RuntimeError(f"{stage.key} timed out at {stage.timeout}s; see {log_path.relative_to(root)}")
            except KeyboardInterrupt:
                terminate_tree(process)
                raise


def read_and_validate_stage(root: Path, stage: Stage, inputs: dict[str, str]) -> dict:
    report_file = root / stage.report
    if not report_file.is_file():
        raise RuntimeError(f"Stage {stage.key} did not create required report {stage.report}")
    try:
        record = json.loads(report_file.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise RuntimeError(f"Unreadable report {stage.report}: {exc}") from exc
    if record.get("status") != stage.status:
        raise RuntimeError(f"Stage {stage.key} status {record.get('status')!r}, expected {stage.status!r}")
    for relative in stage.outputs:
        file = root / relative
        if not file.is_file() or file.stat().st_size == 0:
            raise RuntimeError(f"Stage {stage.key} missing/empty output: {relative}")
    source_hash = record.get("source_sha256")
    expected = (sha256_file(root / "generated/module_augmented.ttl")
                if stage.key == "09_manifest_enrichment" else inputs["src/module.ttl"])
    if source_hash and source_hash != expected:
        raise RuntimeError(f"Stage {stage.key} report source/augmented hash mismatch")
    if record.get("source_unchanged") is False:
        raise RuntimeError(f"Stage {stage.key} reported source_unchanged=false")
    if stage.key == "05b_mowl_and_grok":
        if record.get("llm_executed") is not True or record.get("mowl_training_completed") is not True:
            raise RuntimeError("Step 5B did not execute both the embedding training and Grok")
    if stage.key == "05c_grounded_grok":
        if record.get("llm_executed") is not True or record.get("mowl_training_reexecuted") is not False:
            raise RuntimeError("Step 5C must execute grounded Grok and reuse previous mOWL model")
    if stage.key == "06_scoped_hermit":
        if (record.get("reasoning_scope") != "SOURCE_PLUS_STAR_LOCALITY_MODULE_OF_IMPORTS"
                or not record.get("baseline_reasoner_passed") or not record.get("augmented_reasoner_passed")):
            raise RuntimeError("Step 6 did not pass both source-scoped HermiT runs")
    if stage.key == "07_controlled_merge":
        if (record.get("output") != "generated/module_augmented.ttl"
                or record.get("source_overwritten") is not False
                or record.get("merged_triples") != 484
                or record.get("added_rail_triples") != 63
                or record.get("source_triples") != 421):
            raise RuntimeError("Step 7 ontology merge violates the approved source-preserving policy")
        merged = root / "generated" / "module_augmented.ttl"
        if sha256_file(merged) != record.get("merged_sha256"):
            raise RuntimeError("Augmented TTL hash does not match Step 7 report")
        with (root / "src" / "module.ttl").open("rb") as src, merged.open("rb") as aug:
            if aug.read((root / "src" / "module.ttl").stat().st_size) != src.read():
                raise RuntimeError("Augmented TTL does not preserve source bytes as prefix")
    if stage.key == "08_manifest_ingestion":
        if record.get("version") != V3_SCRIPTS[stage.script]:
            raise RuntimeError("Ingestion report is not the in-transit v3 contract")
        if (record.get("input_sha256") != inputs[next(k for k in inputs if k.lower().endswith('.xlsx'))]
                or record.get("ontology_sha256") != sha256_file(root / "generated/module_augmented.ttl")
                or record.get("source_and_augmented_ontology_unchanged") is not True
                or record.get("no_threat_or_person_of_interest_classification") is not True
                or record.get("generated_classes_or_properties") != 0
                or not str(record.get("travel_occurrence_policy", "")).startswith("MANIFEST_ONBOARD_IN_TRANSIT")):
            raise RuntimeError("Manifest ingestion violated protected ontology / onboard contract")
        counts = record.get("counts", {})
        if not isinstance(counts.get("rows"), int) or counts["rows"] < 1 or counts.get("onboard_passenger_rows") != counts["rows"]:
            raise RuntimeError("Ingestion did not represent every listed onboard passenger")
    if stage.key == "09_manifest_enrichment":
        if record.get("version") != V3_SCRIPTS[stage.script]:
            raise RuntimeError("Enrichment report is not the in-transit v3 contract")
        if (record.get("workbook_sha256") != inputs[next(k for k in inputs if k.lower().endswith('.xlsx'))]
                or record.get("protected_ontology_and_manifest_files_unchanged") is not True
                or record.get("flight_and_rail_occurrence_assumed_from_manifest") is not True
                or record.get("original_abox_sha256") != sha256_file(root / "generated/manifest_instances.ttl")):
            raise RuntimeError("Manifest enrichment integrity / onboard assertion failure")
        if record.get("combined_triples") != (record.get("augmented_triples", -1)
                + record.get("new_tbox_triples", -2) + record.get("enriched_abox_triples", -3)):
            raise RuntimeError("Enrichment output triple accounting is inconsistent")
    if stage.key == "09a_manifest_cosine":
        expected_baseline = sha256_file(root / "generated/module_augmented.ttl")
        expected_final = sha256_file(root / FINAL_OUTPUT)
        results = record.get("results") or {}
        cats = record.get("selected_assertions") or {}
        if (record.get("scope") != "NEW_EXCEL_ABOX_ASSERTIONS_ONLY"
                or record.get("original_classes_scored") != 0
                or record.get("original_subclass_axioms_scored") != 0
                or record.get("rail_tbox_subclass_axioms_scored") != 0
                or record.get("baseline_sha256") != expected_baseline
                or record.get("final_sha256") != expected_final
                or record.get("source_sha256") != inputs["src/module.ttl"]
                or not isinstance(results.get("mean_cosine"), (int, float))
                or results.get("total_scored_assertions", 0) < 1
                or sum(cats.values()) != results["total_scored_assertions"]):
            raise RuntimeError("Cosine report did not evaluate only new ABox assertions")
        # A sub-threshold score is a measured failure of the target, not a
        # software failure: preserve it, run the reasoner, and report it honestly.
    if stage.key == "10_manifest_validation":
        if record.get("version") != V3_SCRIPTS[stage.script]:
            raise RuntimeError("Manifest validation report is not the in-transit v3 contract")
        assertion = record.get("manifest_assertions") or {}
        shacl = record.get("shacl") or {}
        reasoner = record.get("reasoning") or {}
        if (record.get("source_unchanged") is not True
                or record.get("source_or_augmented_overwritten") is not False
                or assertion.get("status") != "PASS_ONBOARD_IN_TRANSIT_ASSERTIONS"
                or assertion.get("warnings") != []
                or shacl.get("conforms") is not True or shacl.get("violations") != 0
                or reasoner.get("owl_reasoner_executed") is not True
                or reasoner.get("scope") != "SOURCE_PLUS_STAR_LOCALITY_MODULE_OF_SIX_LOCAL_IMPORTS"):
            raise RuntimeError("Manifest SHACL, onboard assertions, or scoped reasoning did not pass")
        for run in ("extract", "baseline_reasoner", "candidate_reasoner"):
            entry = reasoner.get(run) or {}
            if entry.get("return_code") != 0 or entry.get("timeout") is not False:
                raise RuntimeError(f"Manifest validation {run} failed or timed out")
        checks = record.get("protected_hashes") or {}
        if checks.get("workbook") != inputs[next(k for k in inputs if k.lower().endswith('.xlsx'))]:
            raise RuntimeError("Manifest validation workbook hash mismatch")
        for name, path in {
            "source": root / "src/module.ttl",
            "augmented": root / "generated/module_augmented.ttl",
            "initial_abox": root / "generated/manifest_instances.ttl",
            "schema": root / "generated/manifest_tbox_extension.ttl",
            "enriched_abox": root / "generated/manifest_instances_enriched.ttl",
            "combined": root / FINAL_OUTPUT,
        }.items():
            if checks.get(name) != sha256_file(path):
                raise RuntimeError(f"Manifest validation protected hash mismatch: {name}")
        if (record.get("counts", {}).get("combined_triples")
                != json.loads((root / "reports/manifest_relationship_report.json").read_text(encoding="utf8"))["combined_triples"]):
            raise RuntimeError("Validated combined triples differ from enrichment")
        augmented = (root / "generated/module_augmented.ttl").read_bytes()
        if not (root / FINAL_OUTPUT).read_bytes().startswith(augmented):
            raise RuntimeError("Final enriched ontology does not preserve augmented bytes")
    return record


def run_pipeline(root: Path, *, epochs: int = 200, model: str = "grok-4.7",
                 reasoner_timeout: int = 300, reasoner_hard_timeout: int = 900,
                 robot_jar: str | None = None, preflight_only: bool = False,
                 manifest: str | Path = "templates/passenger_manifest_template.xlsx",
                 flight_documents: tuple[str, ...] = (), manifest_hard_timeout: int = 1200,
                 check_runtime: bool = True, stage_executor=None) -> dict:
    root = root.resolve()
    workbook = Path(manifest)
    workbook = (workbook if workbook.is_absolute() else root / workbook).resolve()
    plan = stage_plan(epochs=epochs, model=model, reasoner_timeout=reasoner_timeout,
                      reasoner_hard_timeout=reasoner_hard_timeout, robot_jar=robot_jar,
                      manifest=str(workbook), flight_documents=flight_documents,
                      manifest_hard_timeout=manifest_hard_timeout)
    input_hashes = preflight(root, plan, robot_jar=robot_jar, manifest=workbook,
                             check_runtime=check_runtime)
    print(f"STEP 8 preflight passed: {len(IMPORT_NAMES)} local imports, "
          f"{len(plan)} required stages, source={input_hashes['src/module.ttl'][:12]}...", flush=True)
    if preflight_only:
        return {"version": VERSION, "status": "PASS_PREFLIGHT_ONLY", "steps": len(plan),
                "inputs": input_hashes, "source_unchanged": True}
    archived = archive_previous_outputs(root)
    if archived:
        print("Archived previous intermediate files under " + archived, flush=True)
    print("Generating all intermediates from source + local imports + Excel manifest; no cached reports/vectors used.", flush=True)
    manifest = {"version": VERSION, "status": "RUNNING", "started_at_utc": now(),
                "python_executable": sys.executable, "python_version": sys.version.split()[0],
                "source_sha256": input_hashes["src/module.ttl"],
                "manifest_path": str(workbook), "manifest_sha256": input_hashes[str(workbook)],
                "import_sha256": {k:v for k,v in input_hashes.items() if k.startswith("imports/")},
                "previous_outputs_archive": archived,
                "config": {"epochs": epochs, "embedding_dimension": 50, "seed": 42,
                           "model": model, "reasoner": "hermit", "reasoner_scope": "module",
                           "reasoner_timeout": reasoner_timeout,
                           "reasoner_hard_timeout": reasoner_hard_timeout,
                           "manifest_hard_timeout": manifest_hard_timeout,
                           "flight_document_count": len(flight_documents)},
                "stages": [], "source_unchanged": True,
                "llm_repeatability_note": "Live Grok requests occur during proposal, hybrid evaluation, and grounded review; outputs can vary.",
                "mowl_repeatability_note": "CPU+fixed seeds help reproducibility but may not guarantee identical embeddings across versions/hardware."}
    summary_path = root / "reports" / "step8_run_summary.json"
    runner = stage_executor or execute_stage
    try:
        for stage in plan:
            check_input_integrity(root, input_hashes)
            started = time.monotonic()
            entry = {"stage": stage.key, "script": stage.script, "started_at_utc": now()}
            try:
                runner(root, stage)
                check_input_integrity(root, input_hashes)
                report = read_and_validate_stage(root, stage, input_hashes)
                entry.update({"status": "PASS", "reported_status": report["status"],
                              "report": stage.report, "report_sha256": sha256_file(root / stage.report)})
            except Exception as exc:
                entry.update({"status": "FAIL", "error": str(exc)})
                manifest["stages"].append(entry)
                raise
            finally:
                entry["duration_seconds"] = round(time.monotonic() - started, 3)
                entry["finished_at_utc"] = now()
            manifest["stages"].append(entry)
        check_input_integrity(root, input_hashes)
        tbox = json.loads((root / "reports/step7_summary.json").read_text(encoding="utf-8"))
        ingestion = json.loads((root / "reports/manifest_ingestion_report.json").read_text(encoding="utf-8"))
        enrichment = json.loads((root / "reports/manifest_relationship_report.json").read_text(encoding="utf-8"))
        semantic = json.loads((root / "reports/manifest_semantic_validation.json").read_text(encoding="utf-8"))
        cosine = json.loads((root / "reports/manifest_delta_cosine.json").read_text(encoding="utf-8"))
        cosine_met = cosine["results"]["metric_threshold_met"] is True
        manifest.update({
            "status": ("PASS_FULL_TBOX_ABOX_VALIDATED_COSINE_MET" if cosine_met
                       else "PASS_FULL_TBOX_ABOX_VALIDATED_COSINE_BELOW_TARGET"),
            "finished_at_utc": now(),
            "cosine_scope": "NEW_EXCEL_ABOX_ASSERTIONS_ONLY",
            "cosine_mean": cosine["results"]["mean_cosine"],
            "cosine_threshold": cosine["results"]["threshold"],
            "cosine_threshold_met": cosine_met,
            "cosine_scored_assertions": cosine["results"]["total_scored_assertions"],
            "output": FINAL_OUTPUT, "output_sha256": sha256_file(root / FINAL_OUTPUT),
            "augmented_ontology": "generated/module_augmented.ttl",
            "augmented_sha256": sha256_file(root / "generated/module_augmented.ttl"),
            "source_unchanged": True, "source_overwritten": False,
            "tbox_triples": tbox["merged_triples"],
            "manifest_rows": ingestion["counts"]["rows"],
            "manifest_people": ingestion["counts"]["people"],
            "manifest_initial_abox_triples": ingestion["abox_triples"],
            "manifest_extension_triples": enrichment["new_tbox_triples"],
            "enriched_abox_triples": enrichment["enriched_abox_triples"],
            "final_combined_triples": semantic["counts"]["combined_triples"],
            "shacl_conforms": True, "scoped_hermit_passed": True,
            "validation_scope_note": "STAR-locality source-scoped HermiT; not full import closure.",
            "manifest_truth_assumption": "All listed passengers and vehicles are in transit as stated by the workbook; not independently verified."
        })
    except BaseException as exc:
        manifest.update({"status": "FAIL_FAST", "finished_at_utc": now(),
                         "error": str(exc), "output": None})
        try:
            check_input_integrity(root, input_hashes)
        except Exception as source_exc:
            manifest["source_unchanged"] = False
            manifest["source_integrity_failure"] = str(source_exc)
        atomic_json(summary_path, manifest)
        raise
    atomic_json(summary_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--preflight-only", action="store_true", help="Check inputs and dependencies; do not run stages or replace outputs")
    parser.add_argument("--epochs", type=int, default=200, help="Fresh mOWL training epochs (default 200)")
    parser.add_argument("--model", default=os.getenv("XAI_MODEL", "grok-4.7"))
    parser.add_argument("--reasoner-timeout", type=int, default=300, help="Seconds per ROBOT/HermiT invocation")
    parser.add_argument("--reasoner-hard-timeout", type=int, default=900, help="Hard limit for complete Step 6 subprocess")
    parser.add_argument("--robot-jar", help="Path to ROBOT executable JAR if robot is not on PATH")
    parser.add_argument("--manifest", "--input", dest="manifest", default="templates/passenger_manifest_template.xlsx",
                        help=".xlsx passenger manifest (default: templates/passenger_manifest_template.xlsx)")
    parser.add_argument("--flight-document", action="append", default=[], metavar="TRIP_ID=DOCUMENT_ID",
                        help="Optional VERIFIED physical flight-plan document; repeat as needed")
    parser.add_argument("--manifest-hard-timeout", type=int, default=1200,
                        help="Overall HermiT watchdog for manifest validation")
    args = parser.parse_args(argv)
    try:
        result = run_pipeline(args.project_root, epochs=args.epochs, model=args.model,
                              reasoner_timeout=args.reasoner_timeout,
                              reasoner_hard_timeout=args.reasoner_hard_timeout,
                              robot_jar=args.robot_jar, preflight_only=args.preflight_only,
                              manifest=args.manifest, flight_documents=tuple(args.flight_document),
                              manifest_hard_timeout=args.manifest_hard_timeout)
    except KeyboardInterrupt:
        print("PROJECT 6 — INTERRUPTED: previous artifacts were archived; inspect current reports and logs", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print("PROJECT 6 — FAIL: " + str(exc), file=sys.stderr, flush=True)
        print("Inspect reports/step8_run_summary.json and reports/pipeline_logs/", file=sys.stderr, flush=True)
        return 1
    print("PROJECT 6 — " + result["status"], flush=True)
    if result.get("output"):
        print("OUTPUT: " + str(args.project_root.resolve() / result["output"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
