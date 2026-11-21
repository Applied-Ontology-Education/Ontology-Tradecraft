#!/usr/bin/env python3
"""Project 6 Step 8: reproduce the rail TBox and final v9 mOWL evaluation.

Run from the Project 6 repository's project directory:

    python -u scripts/run_all.py

Required source and version-controlled configuration before execution:
    src/module.ttl
    imports/{AgentOntology.ttl,EventOntology.ttl,InformationEntityOntology.ttl,
             bfo-core.ttl,core.ttl,iao-edit.owl}
    config/rail_mowl_v9_approved_anchors.json  (author-reviewed policy, NOT a generated intermediary)

All earlier stage scripts, Python packages, Java/ROBOT, and XAI_API_KEY are
runtime/software prerequisites; intermediate TTL/JSON/reports/vectors are NOT.

The pipeline archives existing generated/ and reports/ folders, regenerates all
artifacts (including new embeddings and two live Grok requests), and only writes
'generated/module_augmented.ttl' plus isolated v9 fold reports. It never overwrites
src/module.ttl, and requires the reviewed v9 anchors in versioned config/.

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

VERSION = "project6-run-all-v9-integration-v1"
V9_ANCHORS_CONFIG = "config/rail_mowl_v9_approved_anchors.json"
V9_REPORT_DIR = "reports/experiments/rail_mowl_v9_reviewed"
V9_SUMMARY_PATH = V9_REPORT_DIR + "/eight_class_summary.json"
V9_EXPECTED_VERSION = "project6-v9-nested-dev-geometry-protected-v1"
V9_EXPECTED_STATUSES = frozenset({
    "TARGET_AND_CONTROLS_MET_DIAGNOSTIC_ONLY",
    "TARGET_OR_CONTROLS_NOT_MET",
})

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


def stage_plan(*, epochs: int = 200, model: str = "grok-4.7",
               reasoner_timeout: int = 300, reasoner_hard_timeout: int = 900,
               robot_jar: str | None = None) -> tuple[Stage, ...]:
    """Explicit dependency ordering; absolutely no --reuse-vectors or --check-only."""
    if epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if not model.strip():
        raise ValueError("xAI model must be non-empty")
    if reasoner_timeout < 30 or reasoner_hard_timeout < reasoner_timeout * 2:
        raise ValueError("Reasoner timeout >=30 and hard timeout >=2x the per-run timeout are required")
    robot_arg = ("--robot-jar", robot_jar) if robot_jar else ()
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
        # Final selected experiment. It needs the augmented ontology from Step 7,
        # and the Step 5A split graphs. It never changes the source ontology.
        Stage("08_v9_rail_mowl", "experiment_rail_mowl_v9.py",
              ("--reviewed-anchors", V9_ANCHORS_CONFIG,
               "--native-epochs", "1000", "--projection-epochs", "100",
               "--configs", "10", "--dev-pairs", "40", "--seed", "101"),
              V9_SUMMARY_PATH, "V9_DIAGNOSTIC_FINISHED",
              (V9_REPORT_DIR + "/preparation.json",
               V9_REPORT_DIR + "/all_pair_scores.csv",
               *(V9_REPORT_DIR + f"/fold_{i}_development.json" for i in range(1, 5)),
               *(V9_REPORT_DIR + f"/fold_{i}_metrics.json" for i in range(1, 5))),
              5400),
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


def fingerprint_inputs(root: Path, plan: tuple[Stage, ...]) -> dict[str, str]:
    """Hash only immutable user-owned source and imports, not generated outputs."""
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
    for dependency in ("experiment_rail_mowl_v8.py", "audit_rail_superclass_support.py"):
        if not (root / "scripts" / dependency).is_file():
            missing_scripts.append(dependency)
    if missing_scripts:
        raise FileNotFoundError("Missing stage scripts in scripts/: " + ", ".join(sorted(set(missing_scripts))))
    review_path = root / V9_ANCHORS_CONFIG
    if not review_path.is_file():
        raise FileNotFoundError(
            f"Missing reviewed v9 anchor config: {V9_ANCHORS_CONFIG}. "
            "Copy the actual approved_anchors.json into config/ before running; "
            "do not fabricate an approval record."
        )
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unparseable reviewed v9 anchor config: {exc}") from exc
    anchors = review.get("approved_anchors", [])
    if (review.get("human_review_status") != "SEVEN_APPROVED_FOR_EXPERIMENT_2026_11_19"
            or review.get("target_holdouts_must_remain_excluded") is not True
            or not isinstance(anchors, list) or len(anchors) != 7
            or any(not isinstance(row, dict) or not isinstance(row.get("child"), str)
                   or not isinstance(row.get("parent"), str) for row in anchors)
            or len({(row["child"], row["parent"]) for row in anchors}) != 7):
        raise ValueError("Invalid v9 reviewed anchor record: exactly seven user-approved, "
                         "unique anchors with excluded holdouts are required")
    # v9 itself independently validates the seven exact approved IRIs and no leakage.
    return {"src/module.ttl": sha256_file(source),
            **{"imports/" + k: sha256_file(path) for k, path in sorted(imports.items())},
            V9_ANCHORS_CONFIG: sha256_file(review_path)}


def preflight(root: Path, plan: tuple[Stage, ...], *, robot_jar: str | None = None,
              check_runtime: bool = True) -> dict[str, str]:
    inputs = fingerprint_inputs(root, plan)
    if not check_runtime:
        return inputs
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("Use the Project 6 .venv-mowl with Python 3.10–3.12 (3.12 recommended), not Python 3.14")
    for module_name in ("rdflib", "pyshacl", "numpy", "torch", "scipy", "mowl", "jpype", "openai"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise RuntimeError(f"Missing/broken Python package {module_name} in {sys.executable}: {exc}") from exc
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
    cmd = [sys.executable, "-u", str(root / "scripts" / stage.script),
           "--project-root", str(root), *stage.args]
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
    if stage.key == "08_v9_rail_mowl":
        # Scientific cosine target is not a condition of successful execution.
        if record.get("status") not in V9_EXPECTED_STATUSES:
            raise RuntimeError(f"v9 experiment incomplete/unrecognized: {record.get('status')!r}")
    elif record.get("status") != stage.status:
        raise RuntimeError(f"Stage {stage.key} status {record.get('status')!r}, expected {stage.status!r}")
    for relative in stage.outputs:
        file = root / relative
        if not file.is_file() or file.stat().st_size == 0:
            raise RuntimeError(f"Stage {stage.key} missing/empty output: {relative}")
    source_hash = record.get("source_sha256")
    if source_hash and source_hash != inputs["src/module.ttl"]:
        raise RuntimeError(f"Stage {stage.key} report source hash mismatch")
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
    if stage.key == "08_v9_rail_mowl":
        if (record.get("version") != V9_EXPECTED_VERSION
                or record.get("folds_completed") != 4
                or record.get("outer_test_sample_size") != 8
                or record.get("dev_pairs_per_fold") != [40, 40, 40, 40]
                or record.get("configs_per_fold") != 10
                or record.get("all_8_rail_targets_unseen_during_model_selection") is not True
                or record.get("all_inner_development_targets_unseen_by_inner_mowl_and_projection") is not True
                or record.get("outer_named_subclass_leakage_detected") is not False
                or record.get("development_named_subclass_leakage_detected") is not False):
            raise RuntimeError("v9 completeness, fold or held-out leakage checks failed")
        # Check provenance of the regenerated upstream inputs and approved config.
        for rel, expected in record.get("input_sha256", {}).items():
            if not (root / rel).is_file() or sha256_file(root / rel) != expected:
                raise RuntimeError("v9 input provenance mismatch: " + rel)
        if record.get("review_file_sha256") != inputs[V9_ANCHORS_CONFIG]:
            raise RuntimeError("v9 approved-anchor provenance mismatch")
        from math import isfinite
        for key in ("native_8_class_mean", "native_auc", "native_gap",
                    "v9_8_class_mean", "v9_auc", "v9_gap"):
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise RuntimeError("v9 missing/invalid numerical metric: " + key)
        for key in ("native_top1_of_four", "v9_top1_of_four"):
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 8:
                raise RuntimeError("v9 missing/invalid superclass rank: " + key)
        if record.get("target_0_70_met") != (record["v9_8_class_mean"] >= 0.70):
            raise RuntimeError("v9 cosine target flag disagrees with measured mean")
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
    return record


def run_pipeline(root: Path, *, epochs: int = 200, model: str = "grok-4.7",
                 reasoner_timeout: int = 300, reasoner_hard_timeout: int = 900,
                 robot_jar: str | None = None, preflight_only: bool = False,
                 check_runtime: bool = True, stage_executor=None) -> dict:
    root = root.resolve()
    plan = stage_plan(epochs=epochs, model=model, reasoner_timeout=reasoner_timeout,
                      reasoner_hard_timeout=reasoner_hard_timeout, robot_jar=robot_jar)
    input_hashes = preflight(root, plan, robot_jar=robot_jar, check_runtime=check_runtime)
    print(f"STEP 8 preflight passed: {len(IMPORT_NAMES)} local imports, "
          f"{len(plan)} required stages, source={input_hashes['src/module.ttl'][:12]}...", flush=True)
    if preflight_only:
        return {"version": VERSION, "status": "PASS_PREFLIGHT_ONLY", "steps": len(plan),
                "inputs": input_hashes, "source_unchanged": True}
    archived = archive_previous_outputs(root)
    if archived:
        print("Archived previous intermediate files under " + archived, flush=True)
    print("Generating all intermediates from source + local imports; no prior vectors/reports reused.", flush=True)
    manifest = {"version": VERSION, "status": "RUNNING", "started_at_utc": now(),
                "python_executable": sys.executable, "python_version": sys.version.split()[0],
                "source_sha256": input_hashes["src/module.ttl"],
                "import_sha256": {k:v for k,v in input_hashes.items() if k.startswith("imports/")},
                "previous_outputs_archive": archived,
                "config": {"epochs": epochs, "embedding_dimension": 50, "seed": 42,
                           "model": model, "reasoner": "hermit", "reasoner_scope": "module",
                           "reasoner_timeout": reasoner_timeout,
                           "reasoner_hard_timeout": reasoner_hard_timeout},
                "stages": [], "source_unchanged": True,
                "llm_repeatability_note": "Two live Grok calls occur; model responses and numerical scores can change between runs.",
                 "mowl_repeatability_note": "CPU+fixed seeds help reproducibility but may not guarantee identical embeddings across versions/hardware.",
                 "v9_interpretation_note": "The scientific target is not a pipeline pass criterion; full external OWL import closure is not verified."}
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
        final = json.loads((root / "reports" / "step7_summary.json").read_text(encoding="utf-8"))
        v9 = json.loads((root / V9_SUMMARY_PATH).read_text(encoding="utf-8"))
        manifest.update({"status": "PASS_ALL_STAGES_SOURCE_UNCHANGED", "finished_at_utc": now(),
                         "v9_final_model": "native ELEmbeddings (comparison baseline; not residual projection)",
                         "v9_experiment_status": v9["status"],
                         "v9_native_mean_cosine": v9["native_8_class_mean"],
                         "v9_native_auc": v9["native_auc"],
                         "v9_native_gap": v9["native_gap"],
                         "v9_native_top1_of_eight": v9["native_top1_of_four"],
                         "v9_dev_selected_projection_folds": v9["projection_selected_on_development_folds"],
                         "v9_target_0_70_met": v9["target_0_70_met"],
                         "v9_report": V9_SUMMARY_PATH,
                         "output": "generated/module_augmented.ttl",
                         "output_sha256": sha256_file(root / "generated" / "module_augmented.ttl"),
                         "source_unchanged": True,
                         "source_overwritten": False, "new_named_individuals": final["new_named_individuals"],
                         "merged_triples": final["merged_triples"],
                         "hindsight_note": "Source-scoped HermiT passed; this does not validate unrelated unimported axioms."})
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
    args = parser.parse_args(argv)
    try:
        result = run_pipeline(args.project_root, epochs=args.epochs, model=args.model,
                              reasoner_timeout=args.reasoner_timeout,
                              reasoner_hard_timeout=args.reasoner_hard_timeout,
                              robot_jar=args.robot_jar, preflight_only=args.preflight_only)
    except KeyboardInterrupt:
        print("STEP 8 — INTERRUPTED: previous artifacts were archived; inspect current reports and logs", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print("STEP 8 — FAIL: " + str(exc), file=sys.stderr, flush=True)
        print("Inspect reports/step8_run_summary.json and reports/pipeline_logs/", file=sys.stderr, flush=True)
        return 1
    print("STEP 8 — " + result["status"], flush=True)
    if result.get("output"):
        print("OUTPUT: " + str(args.project_root.resolve() / result["output"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
