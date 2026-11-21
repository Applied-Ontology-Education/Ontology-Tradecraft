"""Offline regression tests for Project 6 Step 8 orchestration.

No Grok API calls, JVM startups, or actual mOWL training are performed here.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import run_all as m


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "module.ttl").write_text("# immutable source\n", encoding="utf8")
        (self.root / "imports").mkdir()
        for name in m.IMPORT_NAMES:
            (self.root / "imports" / name).write_text("# local import " + name + "\n", encoding="utf8")
        (self.root / "scripts").mkdir()
        for path in m.required_script_paths(self.root, m.stage_plan()):
            path.write_text("# stub, fake runner used\n", encoding="utf8")
        self.before = m.sha256_file(self.root / "src" / "module.ttl")

    @staticmethod
    def fake_stage(root, stage):
        for relative in stage.outputs:
            dest = root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("generated artifact\n", encoding="utf8")
        body = {"status": stage.status, "source_unchanged": True,
                "source_sha256": m.sha256_file(root / "src" / "module.ttl")}
        if stage.key == "05b_mowl_and_grok":
            body.update({"llm_executed": True, "mowl_training_completed": True})
        if stage.key == "05c_grounded_grok":
            body.update({"llm_executed": True, "mowl_training_reexecuted": False})
        if stage.key == "06_scoped_hermit":
            body.update({"reasoning_scope": "SOURCE_PLUS_STAR_LOCALITY_MODULE_OF_IMPORTS",
                         "baseline_reasoner_passed": True, "augmented_reasoner_passed": True})
        if stage.key == "07_controlled_merge":
            source_bytes = (root / "src/module.ttl").read_bytes()
            merged_path = root / "generated/module_augmented.ttl"
            merged_path.write_bytes(source_bytes + b"# new rail triples\n")
            body.update({"output": "generated/module_augmented.ttl",
                         "merged_sha256": m.sha256_file(merged_path),
                         "source_overwritten": False,
                         "merged_triples": 484, "added_rail_triples": 63,
                         "source_triples": 421, "new_named_individuals": 0})
        report = root / stage.report
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(body), encoding="utf8")

    def run_fake(self, **kwargs):
        return m.run_pipeline(self.root, check_runtime=False,
                              stage_executor=self.fake_stage, **kwargs)

    def test_stage_order_and_fresh_embedding_run(self):
        stages = m.stage_plan()
        self.assertEqual(len(stages), 10)
        self.assertEqual(stages[0].key, "01_inventory")
        self.assertEqual(stages[-1].key, "07_controlled_merge")
        embedding = next(s for s in stages if s.key == "05b_mowl_and_grok")
        self.assertIn("--with-llm", embedding.args)
        self.assertNotIn("--reuse-vectors", embedding.args)
        self.assertEqual(embedding.args[embedding.args.index("--epochs")+1], "200")
        reasoner = next(s for s in stages if s.key == "06_scoped_hermit")
        self.assertEqual(reasoner.args[reasoner.args.index("--scope")+1], "module")
        self.assertEqual(reasoner.args[reasoner.args.index("--reasoner")+1], "hermit")

    def test_expected_inputs_without_any_generated_intermediates(self):
        inputs = m.preflight(self.root, m.stage_plan(), check_runtime=False)
        self.assertEqual(len(inputs), 7)
        self.assertNotIn("generated/module_augmented.ttl", inputs)
        self.assertEqual(inputs["src/module.ttl"], self.before)

    def test_missing_import_blocks_before_any_output_changes(self):
        (self.root / "imports/iao-edit.owl").unlink()
        (self.root / "generated").mkdir()
        (self.root / "generated" / "previous.txt").write_text("preserve me")
        with self.assertRaisesRegex(ValueError, "missing"):
            self.run_fake()
        self.assertTrue((self.root / "generated" / "previous.txt").is_file())

    def test_preflight_does_not_archive_or_run(self):
        (self.root / "reports").mkdir()
        (self.root / "reports" / "old.json").write_text("preserve me")
        result = self.run_fake(preflight_only=True)
        self.assertEqual(result["status"], "PASS_PREFLIGHT_ONLY")
        self.assertTrue((self.root / "reports" / "old.json").exists())
        self.assertFalse((self.root / ".pipeline-archives").exists())

    def test_clean_end_to_end_mock_and_archive_previous(self):
        (self.root / "reports").mkdir()
        (self.root / "reports" / "stale.json").write_text("old")
        (self.root / "generated").mkdir()
        (self.root / "generated" / "module_augmented.ttl").write_text("STALE")
        result = self.run_fake()
        self.assertEqual(result["status"], "PASS_ALL_STAGES_SOURCE_UNCHANGED")
        self.assertEqual(len(result["stages"]), 10)
        self.assertEqual(self.before, m.sha256_file(self.root / "src/module.ttl"))
        self.assertTrue((self.root / "generated/module_augmented.ttl").read_bytes().startswith(
            (self.root / "src/module.ttl").read_bytes()))
        self.assertTrue((self.root / result["previous_outputs_archive"] / "reports/stale.json").exists())
        self.assertEqual(json.loads((self.root / "reports/step8_run_summary.json").read_text())["status"],
                         "PASS_ALL_STAGES_SOURCE_UNCHANGED")

    def test_fail_fast_on_missing_stage_output(self):
        invoked = []
        def broken(root, stage):
            invoked.append(stage.key)
            if stage.key == "03_candidate_axioms":
                return
            self.fake_stage(root, stage)
        with self.assertRaisesRegex(RuntimeError, "did not create required report"):
            m.run_pipeline(self.root, check_runtime=False, stage_executor=broken)
        self.assertEqual(invoked, ["01_inventory", "02_grok_proposal", "03_candidate_axioms"])
        report = json.loads((self.root / "reports/step8_run_summary.json").read_text())
        self.assertEqual(report["status"], "FAIL_FAST")
        self.assertFalse((self.root / "generated/module_augmented.ttl").exists())

    def test_fail_fast_if_source_tampered_with(self):
        def corrupt(root, stage):
            self.fake_stage(root, stage)
            if stage.key == "02_grok_proposal":
                (root / "src/module.ttl").write_text("tampered", encoding="utf8")
        with self.assertRaisesRegex(RuntimeError, "Protected original"):
            m.run_pipeline(self.root, check_runtime=False, stage_executor=corrupt)
        report = json.loads((self.root / "reports/step8_run_summary.json").read_text())
        self.assertFalse(report["source_unchanged"])
        self.assertIn("source_integrity_failure", report)

    def test_stage_report_with_wrong_success_status_is_rejected(self):
        def incorrect(root, stage):
            self.fake_stage(root, stage)
            if stage.key == "01_inventory":
                path = root / stage.report
                data = json.loads(path.read_text())
                data["status"] = "PASS_STALE_REPORT"
                path.write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, "expected 'PASS'"):
            m.run_pipeline(self.root, check_runtime=False, stage_executor=incorrect)

    def test_repeated_runs_do_not_reuse_outputs(self):
        one = self.run_fake()
        self.assertIsNone(one["previous_outputs_archive"])
        (self.root / "generated" / "injected.tmp").write_text("artifact")
        two = self.run_fake()
        self.assertIsNotNone(two["previous_outputs_archive"])
        self.assertFalse((self.root / "generated/injected.tmp").exists())
        self.assertTrue((self.root / two["previous_outputs_archive"] / "generated/injected.tmp").exists())

    def test_subprocess_execution_writes_live_log(self):
        stub = self.root / "scripts/subprocess_stub.py"
        stub.write_text(
            "import sys\n"
            "print('STEP8 stub progress', flush=True)\n"
            "print('STEP8 stub complete', flush=True)\n",
            encoding="utf8",
        )
        stage = m.Stage("00_stub", "subprocess_stub.py", (),
                        "reports/unused.json", "PASS", (), 30)
        (self.root / "reports").mkdir()
        m.execute_stage(self.root, stage)
        content = (self.root / "reports/pipeline_logs/00_stub.log").read_text()
        self.assertIn("STEP8 stub progress", content)
        self.assertIn("STEP8 stub complete", content)

    def test_reject_reasoner_timeout_configuration(self):
        with self.assertRaisesRegex(ValueError, "hard timeout"):
            m.stage_plan(reasoner_timeout=300, reasoner_hard_timeout=400)


if __name__ == "__main__":
    unittest.main()
