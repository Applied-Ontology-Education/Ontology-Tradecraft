"""Step 7: merge generated/accepted_el.ttl into src/cco-module.ttl with ROBOT, reason with ELK -> src/module_augmented.ttl.

- owl:imports are kept as import statements (resolved offline through src/catalog-v001.xml), not copied in.
- ELK must find the merged ontology consistent with no unsatisfiable classes, or ROBOT exits non-zero.
- Asserted axioms are never removed (--remove-redundant-subclass-axioms false); inferred axioms about module
  classes are added and annotated as inferred.
- QC: the accepted axioms are checked against the OWL 2 EL profile; a ROBOT report is written for review."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import GEN, MODULE, REPORTS, SRC, robot, write_json  # noqa: E402

OUT = SRC / "module_augmented.ttl"


def main():
    acc = GEN / "accepted_el.ttl"
    el_cand = robot("validate-profile", "--input", str(GEN / "candidate_el.ttl"), "--profile", "EL", check=False).returncode == 0
    el_ok = robot("validate-profile", "--input", str(acc), "--profile", "EL", check=False).returncode == 0
    robot("merge", "--input", str(MODULE), "--input", str(acc), "--collapse-import-closure", "false",
          "reason", "--reasoner", "ELK", "--remove-redundant-subclass-axioms", "false",
          "--exclude-external-entities", "true", "--annotate-inferred-axioms", "true",
          "--output", str(OUT))
    # independent re-check of the saved file (consistency + no unsatisfiable classes)
    chk = robot("reason", "--input", str(OUT), "--reasoner", "ELK", check=False)
    consistent = chk.returncode == 0
    rep = robot("report", "--input", str(OUT), "--fail-on", "none", "--output", str(REPORTS / "robot_report.tsv"), check=False)
    status = {"el_profile_candidate_axioms": el_cand, "el_profile_accepted_axioms": el_ok, "consistent": consistent,
              "reasoner": "ELK (ROBOT)", "output": "src/module_augmented.ttl",
              "robot_report": "reports/robot_report.tsv" if rep.returncode == 0 else f"report failed: {rep.stderr[-200:]}"}
    m = json.loads((REPORTS / "mowl_metrics.json").read_text())
    m["merge"] = status
    write_json(REPORTS / "mowl_metrics.json", m)
    print(f"merge_and_reason: EL profile candidates {'OK' if el_cand else 'VIOLATION'}, accepted {'OK' if el_ok else 'VIOLATION'}; ELK {'consistent' if consistent else 'INCONSISTENT'} "
          f"-> src/module_augmented.ttl")
    if not (consistent and el_ok and el_cand):
        sys.exit(1)


if __name__ == "__main__":
    main()
