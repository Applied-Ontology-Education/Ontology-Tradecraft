from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

from rdflib import Graph, URIRef
from rdflib.namespace import RDFS


# ============================================================
# VERSION
# ============================================================

SCRIPT_VERSION = "robot-merge-elk-v2-preserve-accepted-subclass-axioms"


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
GENERATED_DIR = PROJECT_ROOT / "generated"
REPORTS_DIR = PROJECT_ROOT / "reports"

BASE_ONTOLOGY = (
    SRC_DIR
    / "module.ttl"
)

ACCEPTED_AXIOMS = (
    GENERATED_DIR
    / "accepted_el.ttl"
)

OUTPUT_ONTOLOGY = (
    SRC_DIR
    / "module_augmented.ttl"
)

DEBUG_ONTOLOGY = (
    REPORTS_DIR
    / "elk_unsatisfiable.ttl"
)

ROBOT_LOG = (
    REPORTS_DIR
    / "robot_merge_reason.log"
)

REPORT_FILE = (
    REPORTS_DIR
    / "robot_merge_reason_report.json"
)


# ============================================================
# HELPERS
# ============================================================

def require_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n  {path}"
        )


def load_turtle(path):

    graph = Graph()

    graph.parse(
        str(path),
        format="turtle",
    )

    return graph


def named_subclass_pairs(graph):

    pairs = set()

    for child, parent in graph.subject_objects(
        RDFS.subClassOf
    ):

        if not isinstance(
            child,
            URIRef,
        ):
            continue

        if not isinstance(
            parent,
            URIRef,
        ):
            continue

        pairs.add(
            (
                str(child),
                str(parent),
            )
        )

    return pairs


# ============================================================
# JAVA
# ============================================================

def find_java():

    java = shutil.which(
        "java"
    )

    if java:

        return java

    java_home = os.getenv(
        "JAVA_HOME"
    )

    if java_home:

        candidate = (
            Path(java_home)
            / "bin"
            / "java.exe"
        )

        if candidate.exists():

            return str(
                candidate
            )

    raise RuntimeError(
        "Java could not be found.\n\n"
        "Make sure JAVA_HOME is set and Java is "
        "available in PATH."
    )


# ============================================================
# ROBOT LOCATION
#
# Priority:
#
# 1. ROBOT_JAR environment variable
# 2. robot executable in PATH
# 3. common repository locations
# ============================================================

def find_robot():

    # --------------------------------------------------------
    # 1. Explicit ROBOT_JAR
    # --------------------------------------------------------

    robot_jar_env = os.getenv(
        "ROBOT_JAR"
    )

    if robot_jar_env:

        robot_jar = Path(
            robot_jar_env
        ).expanduser()

        if not robot_jar.exists():

            raise FileNotFoundError(
                "ROBOT_JAR is set, but the file "
                f"does not exist:\n  {robot_jar}"
            )

        java = find_java()

        return (
            [
                java,
                "-Xmx4G",
                "-jar",
                str(
                    robot_jar.resolve()
                ),
            ],
            f"JAR: {robot_jar.resolve()}",
        )

    # --------------------------------------------------------
    # 2. ROBOT executable in PATH
    # --------------------------------------------------------

    robot_executable = shutil.which(
        "robot"
    )

    if robot_executable:

        return (
            [
                robot_executable
            ],
            f"Executable: {robot_executable}",
        )

    # --------------------------------------------------------
    # 3. Common local JAR locations
    # --------------------------------------------------------

    repository_root = (
        PROJECT_ROOT
        .parent
        .parent
    )

    candidates = [
        PROJECT_ROOT
        / "robot.jar",

        PROJECT_ROOT
        / "tools"
        / "robot.jar",

        PROJECT_ROOT.parent
        / "robot.jar",

        repository_root
        / "robot.jar",

        repository_root
        / "tools"
        / "robot.jar",

        Path("C:/Windows/robot.jar"),
    ]

    for candidate in candidates:

        if candidate.exists():

            java = find_java()

            return (
                [
                    java,
                    "-Xmx4G",
                    "-jar",
                    str(
                        candidate.resolve()
                    ),
                ],
                f"JAR: {candidate.resolve()}",
            )

    raise RuntimeError(
        "ROBOT could not be found.\n\n"
        "Either:\n"
        "  1. Put robot in PATH, or\n"
        "  2. Set ROBOT_JAR to the full path "
        "of robot.jar.\n\n"
        "PowerShell example:\n"
        '$env:ROBOT_JAR = '
        '"C:\\Windows\\robot.jar"'
    )


# ============================================================
# MAIN
# ============================================================

def main():

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"ROBOT_MERGE_VERSION    : "
        f"{SCRIPT_VERSION}"
    )

    print()

    print("=" * 78)

    print(
        "PROJECT 5 - ROBOT MERGE + ELK REASONING"
    )

    print("=" * 78)

    print()

    print(
        f"Base ontology           : "
        f"{BASE_ONTOLOGY}"
    )

    print(
        f"Accepted EL axioms      : "
        f"{ACCEPTED_AXIOMS}"
    )

    print(
        f"Final ontology          : "
        f"{OUTPUT_ONTOLOGY}"
    )

    print()

    # ========================================================
    # PRE-FLIGHT FILE CHECKS
    # ========================================================

    require_file(
        BASE_ONTOLOGY
    )

    require_file(
        ACCEPTED_AXIOMS
    )

    print(
        "Validating Turtle inputs..."
    )

    try:

        base_graph = load_turtle(
            BASE_ONTOLOGY
        )

        accepted_graph = load_turtle(
            ACCEPTED_AXIOMS
        )

    except Exception as exc:

        raise RuntimeError(
            "One of the input ontology files could "
            f"not be parsed as Turtle:\n{exc}"
        ) from exc

    accepted_pairs = named_subclass_pairs(
        accepted_graph
    )

    if not accepted_pairs:

        raise RuntimeError(
            "accepted_el.ttl contains no named "
            "rdfs:subClassOf axioms."
        )

    print(
        f"Base triples            : "
        f"{len(base_graph)}"
    )

    print(
        f"Accepted triples        : "
        f"{len(accepted_graph)}"
    )

    print(
        f"Accepted subclass axioms: "
        f"{len(accepted_pairs)}"
    )

    print(
        "Input validation        : PASS"
    )

    print()

    # ========================================================
    # FIND ROBOT
    # ========================================================

    (
        robot_prefix,
        robot_description,
    ) = find_robot()

    print(
        f"ROBOT                   : "
        f"{robot_description}"
    )

    print()

    # ========================================================
    # REMOVE STALE OUTPUTS
    # ========================================================

    if OUTPUT_ONTOLOGY.exists():

        OUTPUT_ONTOLOGY.unlink()

    if DEBUG_ONTOLOGY.exists():

        DEBUG_ONTOLOGY.unlink()

    # ========================================================
    # ROBOT COMMAND
    #
    # Important options:
    #
    # --collapse-import-closure false
    #
    # preserves the import structure.
    #
    # --remove-redundant-subclass-axioms false
    #
    # preserves accepted subclass assertions even when ELK
    # determines that the same relationship can be inferred
    # transitively or by another path.
    # ========================================================

    command = (
        robot_prefix
        +
        [
            "merge",

            "--collapse-import-closure",
            "false",

            "--input",
            str(
                BASE_ONTOLOGY.resolve()
            ),

            "--input",
            str(
                ACCEPTED_AXIOMS.resolve()
            ),

            "reason",

            "--reasoner",
            "ELK",

            "--remove-redundant-subclass-axioms",
            "false",

            "-D",
            str(
                DEBUG_ONTOLOGY.resolve()
            ),

            "--output",
            str(
                OUTPUT_ONTOLOGY.resolve()
            ),
        ]
    )

    print(
        "Running ROBOT merge + ELK..."
    )

    print()

    print(
        "Command:"
    )

    print(
        " ".join(
            f'"{part}"'
            if " " in part
            else part

            for part in command
        )
    )

    print()

    # ========================================================
    # EXECUTE ROBOT
    # ========================================================

    process = subprocess.run(
        command,
        cwd=str(
            PROJECT_ROOT
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # ========================================================
    # WRITE ROBOT LOG
    # ========================================================

    with open(
        ROBOT_LOG,
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "COMMAND\n"
        )

        handle.write(
            "=" * 78
        )

        handle.write(
            "\n"
        )

        handle.write(
            " ".join(
                command
            )
        )

        handle.write(
            "\n\nSTDOUT\n"
        )

        handle.write(
            "=" * 78
        )

        handle.write(
            "\n"
        )

        handle.write(
            process.stdout
            or ""
        )

        handle.write(
            "\n\nSTDERR\n"
        )

        handle.write(
            "=" * 78
        )

        handle.write(
            "\n"
        )

        handle.write(
            process.stderr
            or ""
        )

        handle.write(
            "\n"
        )

    # ========================================================
    # DISPLAY ROBOT OUTPUT
    # ========================================================

    if process.stdout:

        print(
            process.stdout
        )

    if process.stderr:

        print(
            process.stderr
        )

    # ========================================================
    # CHECK ROBOT / ELK EXIT STATUS
    # ========================================================

    if process.returncode != 0:

        print()

        print("=" * 78)

        print(
            "ROBOT / ELK RESULT: FAIL"
        )

        print("=" * 78)

        print()

        print(
            f"Exit code               : "
            f"{process.returncode}"
        )

        print(
            f"Full ROBOT log          : "
            f"{ROBOT_LOG}"
        )

        if DEBUG_ONTOLOGY.exists():

            print(
                f"ELK debug ontology      : "
                f"{DEBUG_ONTOLOGY}"
            )

        print()

        sys.exit(
            process.returncode
        )

    # ========================================================
    # VERIFY OUTPUT EXISTS
    # ========================================================

    if not OUTPUT_ONTOLOGY.exists():

        raise RuntimeError(
            "ROBOT returned success but "
            "module_augmented.ttl was not created."
        )

    # ========================================================
    # VERIFY OUTPUT IS VALID TURTLE
    # ========================================================

    try:

        augmented_graph = load_turtle(
            OUTPUT_ONTOLOGY
        )

    except Exception as exc:

        raise RuntimeError(
            "module_augmented.ttl was created but "
            f"could not be parsed as Turtle:\n{exc}"
        ) from exc

    # ========================================================
    # VERIFY ALL ACCEPTED AXIOMS SURVIVED
    # ========================================================

    augmented_pairs = named_subclass_pairs(
        augmented_graph
    )

    missing_accepted = (
        accepted_pairs
        -
        augmented_pairs
    )

    retained_accepted = (
        accepted_pairs
        &
        augmented_pairs
    )

    print()

    print(
        "Verifying accepted subclass axioms..."
    )

    print(
        f"Accepted expected       : "
        f"{len(accepted_pairs)}"
    )

    print(
        f"Accepted retained       : "
        f"{len(retained_accepted)}"
    )

    print(
        f"Accepted missing        : "
        f"{len(missing_accepted)}"
    )

    if missing_accepted:

        print()

        print(
            "ERROR: Some accepted axioms are still "
            "missing from the augmented ontology."
        )

        for child, parent in sorted(
            missing_accepted
        ):

            print(
                f"  {child} -> {parent}"
            )

        raise RuntimeError(
            f"{len(missing_accepted)} accepted "
            "axiom(s) were missing after merge/reason."
        )

    print(
        "Accepted-axiom check   : PASS"
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "script_version":
            SCRIPT_VERSION,

        "status":
            "PASS",

        "reasoner":
            "ELK",

        "base_ontology":
            str(
                BASE_ONTOLOGY
            ),

        "accepted_axioms_file":
            str(
                ACCEPTED_AXIOMS
            ),

        "output_ontology":
            str(
                OUTPUT_ONTOLOGY
            ),

        "accepted_named_subclass_axioms":
            len(
                accepted_pairs
            ),

        "accepted_axioms_retained":
            len(
                retained_accepted
            ),

        "accepted_axioms_missing":
            len(
                missing_accepted
            ),

        "accepted_axioms_preserved":
            (
                len(
                    missing_accepted
                )
                ==
                0
            ),

        "output_triples":
            len(
                augmented_graph
            ),

        "output_named_subclass_axioms":
            len(
                augmented_pairs
            ),

        "collapse_import_closure":
            False,

        "remove_redundant_subclass_axioms":
            False,

        "robot_exit_code":
            process.returncode,

        "debug_ontology_created":
            DEBUG_ONTOLOGY.exists(),

        "robot_log":
            str(
                ROBOT_LOG
            ),
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            report,
            handle,
            indent=2,
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()

    print("=" * 78)

    print(
        "ROBOT + ELK RESULTS"
    )

    print("=" * 78)

    print()

    print(
        "ROBOT exit code         : 0"
    )

    print(
        "ELK logical validation  : PASS"
    )

    print(
        "ELK reasoning           : PASS"
    )

    print(
        f"Accepted axioms retained: "
        f"{len(retained_accepted)}/"
        f"{len(accepted_pairs)}"
    )

    print(
        f"Final triples           : "
        f"{len(augmented_graph)}"
    )

    print(
        f"Named subclass axioms   : "
        f"{len(augmented_pairs)}"
    )

    print()

    print(
        "FINAL ONTOLOGY:"
    )

    print(
        f"  {OUTPUT_ONTOLOGY}"
    )

    print()

    print(
        "REPORT:"
    )

    print(
        f"  {REPORT_FILE}"
    )

    print()

    print(
        "ROBOT LOG:"
    )

    print(
        f"  {ROBOT_LOG}"
    )

    if DEBUG_ONTOLOGY.exists():

        print()

        print(
            "DEBUG ONTOLOGY:"
        )

        print(
            f"  {DEBUG_ONTOLOGY}"
        )


if __name__ == "__main__":

    main()