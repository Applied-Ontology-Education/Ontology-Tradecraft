from pathlib import Path
from datetime import datetime
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

from rdflib import Graph, URIRef
from rdflib.namespace import RDFS


SCRIPT_VERSION = (
    "project-5-sequential-clean-room-driver-v6-java-preflight"
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
GENERATED_DIR = PROJECT_ROOT / "generated"
REPORTS_DIR = PROJECT_ROOT / "reports"

LOG_DIR = REPORTS_DIR / "run_all_logs"
BACKUP_ROOT = PROJECT_ROOT / ".run_all_backups"


# ============================================================
# SOURCE INPUT
# ============================================================

MODULE_FILE = SRC_DIR / "module.ttl"


# ============================================================
# GENERATED PIPELINE ARTIFACTS
# ============================================================

DEFINITIONS_FILE = DATA_DIR / "definitions.csv"
DEFINITIONS_ENRICHED_FILE = DATA_DIR / "definitions_enriched.csv"

CANDIDATE_FILE = GENERATED_DIR / "candidate_el.ttl"

TRAIN_FILE = SRC_DIR / "train.ttl"
VALID_FILE = SRC_DIR / "valid.ttl"
SPLIT_REPORT_FILE = REPORTS_DIR / "split_train_valid_report.json"

MOWL_METRICS_FILE = REPORTS_DIR / "mowl_metrics.json"
NATIVE_MODEL_FILE = REPORTS_DIR / "elembeddings_v8_native.pt"
PROJECTION_FILE = REPORTS_DIR / "mowl_v8_residual_projection_best.pt"
BEST_CONFIG_FILE = REPORTS_DIR / "mowl_v8_best_config.json"
V8_VALIDATION_FILE = REPORTS_DIR / "mowl_v8_best_validation_pairs.csv"

HYBRID_SCORES_FILE = GENERATED_DIR / "hybrid_candidate_scores.csv"
HYBRID_CACHE_FILE = GENERATED_DIR / "hybrid_llm_cache.jsonl"
RECIPROCAL_REVIEW_FILE = GENERATED_DIR / "hybrid_reciprocal_review.csv"
ACCEPTED_FILE = GENERATED_DIR / "accepted_el.ttl"

ROBOT_REPORT_FILE = REPORTS_DIR / "robot_merge_reason_report.json"
ROBOT_LOG_FILE = REPORTS_DIR / "robot_merge_reason.log"
ELK_DEBUG_FILE = REPORTS_DIR / "elk_unsatisfiable.ttl"
AUGMENTED_FILE = SRC_DIR / "module_augmented.ttl"

SUMMARY_FILE = REPORTS_DIR / "run_all_summary.json"


# ============================================================
# PIPELINE SCRIPTS
# ============================================================

EXTRACT_SCRIPT = SCRIPTS_DIR / "extract_definitions.py"
PREPROCESS_SCRIPT = SCRIPTS_DIR / "preprocess_definitions_llm.py"
GENERATE_SCRIPT = SCRIPTS_DIR / "generate_candidates_llm.py"
SPLIT_SCRIPT = SCRIPTS_DIR / "split_train_valid.py"
TRAIN_MOWL_SCRIPT = SCRIPTS_DIR / "train_mowl.py"
FILTER_SCRIPT = SCRIPTS_DIR / "filter_candidates_hybrid.py"
MERGE_REASON_SCRIPT = SCRIPTS_DIR / "merge_reason_robot.py"


# ============================================================
# PROJECT REQUIREMENTS
# ============================================================

MIN_MEAN_COSINE = 0.70
MIN_ACCEPTED_AXIOMS = 3
CONFIGURED_COSINE_WEIGHT = 0.70
CONFIGURED_LLM_WEIGHT = 0.30


# ============================================================
# BACKUP / CLEAN-ROOM CONFIGURATION
# ============================================================

PROTECTED_DIRECTORIES = [
    SRC_DIR,
    DATA_DIR,
    GENERATED_DIR,
    REPORTS_DIR,
]

# These are generated artifacts.
# module.ttl is deliberately absent so it remains
# the immutable source ontology.
PIPELINE_OUTPUTS = [
    DEFINITIONS_FILE,
    DEFINITIONS_ENRICHED_FILE,
    CANDIDATE_FILE,
    TRAIN_FILE,
    VALID_FILE,
    SPLIT_REPORT_FILE,
    MOWL_METRICS_FILE,
    NATIVE_MODEL_FILE,
    PROJECTION_FILE,
    BEST_CONFIG_FILE,
    V8_VALIDATION_FILE,
    HYBRID_SCORES_FILE,
    HYBRID_CACHE_FILE,
    RECIPROCAL_REVIEW_FILE,
    ACCEPTED_FILE,
    AUGMENTED_FILE,
    ROBOT_REPORT_FILE,
    ROBOT_LOG_FILE,
    ELK_DEBUG_FILE,
    SUMMARY_FILE,
]


# ============================================================
# BASIC HELPERS
# ============================================================

def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n  {path}"
        )


def remove_path(path: Path):
    if not path.exists():
        return

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def load_json(path: Path):
    require_file(path)

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_graph(path: Path):
    require_file(path)

    graph = Graph()

    graph.parse(
        str(path),
        format="turtle",
    )

    return graph


def named_subclass_pairs(graph: Graph):
    pairs = set()

    for child, parent in graph.subject_objects(
        RDFS.subClassOf
    ):
        if (
            isinstance(child, URIRef)
            and isinstance(parent, URIRef)
            and child != parent
        ):
            pairs.add(
                (
                    str(child),
                    str(parent),
                )
            )

    return pairs


# ============================================================
# JAVA / MOWL PREFLIGHT
# ============================================================

def configure_java_environment():
    """
    Locate a usable Windows JDK, configure JAVA_HOME and PATH
    for this process, and return (java_home, jvm_dll).

    All pipeline child processes inherit this environment
    because run_stage() passes os.environ.copy().
    """

    candidates = []

    # --------------------------------------------------------
    # 1. Prefer an already configured JAVA_HOME.
    # --------------------------------------------------------

    current_java_home = os.getenv(
        "JAVA_HOME"
    )

    if current_java_home:
        candidates.append(
            Path(current_java_home)
        )

    # --------------------------------------------------------
    # 2. Known JDK used successfully for this project.
    # --------------------------------------------------------

    candidates.append(
        Path(
            r"C:\Program Files\Java\jdk-26.0.2.1"
        )
    )

    # --------------------------------------------------------
    # 3. Discover other installed JDKs if necessary.
    # --------------------------------------------------------

    java_root = Path(
        r"C:\Program Files\Java"
    )

    if java_root.exists():
        discovered = [
            path
            for path in java_root.iterdir()
            if (
                path.is_dir()
                and path.name.lower().startswith(
                    "jdk"
                )
            )
        ]

        discovered.sort(
            key=lambda path: path.name,
            reverse=True,
        )

        candidates.extend(
            discovered
        )

    # --------------------------------------------------------
    # De-duplicate candidates while preserving order.
    # --------------------------------------------------------

    unique_candidates = []
    seen = set()

    for candidate in candidates:
        key = str(
            candidate
        ).lower()

        if key not in seen:
            seen.add(
                key
            )

            unique_candidates.append(
                candidate
            )

    selected_home = None
    selected_jvm = None

    # --------------------------------------------------------
    # Locate jvm.dll.
    # --------------------------------------------------------

    for java_home in unique_candidates:
        if not java_home.exists():
            continue

        expected_jvm = (
            java_home
            / "bin"
            / "server"
            / "jvm.dll"
        )

        if expected_jvm.exists():
            selected_home = (
                java_home
            )

            selected_jvm = (
                expected_jvm
            )

            break

        try:
            matches = list(
                java_home.rglob(
                    "jvm.dll"
                )
            )

        except Exception:
            matches = []

        if matches:
            selected_home = (
                java_home
            )

            selected_jvm = (
                matches[0]
            )

            break

    if (
        selected_home is None
        or selected_jvm is None
    ):
        searched = "\n".join(
            f"  {path}"
            for path in unique_candidates
        )

        raise RuntimeError(
            "Could not locate a usable Java JVM.\n\n"
            "The pipeline requires a JDK containing "
            "jvm.dll.\n\n"
            "Locations checked:\n"
            f"{searched}"
        )

    # --------------------------------------------------------
    # Verify java.exe exists.
    # --------------------------------------------------------

    java_exe = (
        selected_home
        / "bin"
        / "java.exe"
    )

    if not java_exe.exists():
        raise RuntimeError(
            "A JVM library was found, but java.exe "
            "was not found where expected.\n\n"
            f"JAVA_HOME:\n"
            f"  {selected_home}\n\n"
            f"Expected java.exe:\n"
            f"  {java_exe}"
        )

    # --------------------------------------------------------
    # Configure environment for this driver and all child
    # processes launched afterward.
    # --------------------------------------------------------

    os.environ[
        "JAVA_HOME"
    ] = str(
        selected_home
    )

    java_bin = str(
        selected_home
        / "bin"
    )

    current_path = os.environ.get(
        "Path",
        os.environ.get(
            "PATH",
            "",
        ),
    )

    path_entries = [
        entry.strip().lower()
        for entry in current_path.split(
            os.pathsep
        )
        if entry.strip()
    ]

    if (
        java_bin.lower()
        not in path_entries
    ):
        updated_path = (
            java_bin
            + os.pathsep
            + current_path
        )

    else:
        updated_path = (
            current_path
        )

    # Windows environment names are case-insensitive,
    # but setting both makes inheritance predictable.
    os.environ[
        "Path"
    ] = updated_path

    os.environ[
        "PATH"
    ] = updated_path

    return (
        selected_home,
        selected_jvm,
    )


# ============================================================
# PREFLIGHT
# ============================================================

def preflight():
    print(
        "Preflight checks..."
    )

    # ========================================================
    # SOURCE ONTOLOGY
    # ========================================================

    require_file(
        MODULE_FILE
    )

    # ========================================================
    # REQUIRED PIPELINE SCRIPTS
    # ========================================================

    required_scripts = [
        EXTRACT_SCRIPT,
        PREPROCESS_SCRIPT,
        GENERATE_SCRIPT,
        SPLIT_SCRIPT,
        TRAIN_MOWL_SCRIPT,
        FILTER_SCRIPT,
        MERGE_REASON_SCRIPT,
    ]

    for script in required_scripts:
        require_file(
            script
        )

    # ========================================================
    # PYTHON SYNTAX CHECK
    #
    # Catch SyntaxError / IndentationError before any
    # expensive LLM calls.
    # ========================================================

    print(
        "  Checking Python syntax..."
    )

    for script in required_scripts:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "py_compile",
                str(script),
            ],
            cwd=str(
                PROJECT_ROOT
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=os.environ.copy(),
        )

        if (
            result.returncode
            !=
            0
        ):
            raise RuntimeError(
                "Python syntax check failed.\n\n"
                f"Script:\n"
                f"  {script}\n\n"
                "Compiler output:\n"
                f"{result.stdout}"
            )

    print(
        "  Python syntax         : PASS"
    )

    # ========================================================
    # MOWL PYTHON PACKAGE CHECK
    # ========================================================

    print(
        "  Checking mOWL package..."
    )

    mowl_import_check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import mowl; "
                "print(mowl.__file__)"
            ),
        ],
        cwd=str(
            PROJECT_ROOT
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=os.environ.copy(),
    )

    if (
        mowl_import_check.returncode
        !=
        0
    ):
        raise RuntimeError(
            "The Python interpreter running "
            "run_all.py cannot import mOWL.\n\n"
            f"Current interpreter:\n"
            f"  {sys.executable}\n\n"
            "Expected project environment:\n"
            r"  C:\Users\cappa\Ontology-Tradecraft\.venv-mowl"
            "\n\n"
            "Import output:\n"
            f"{mowl_import_check.stdout}"
        )

    print(
        "  mOWL Python package   : PASS"
    )

    # ========================================================
    # JAVA AUTO-CONFIGURATION
    # ========================================================

    print(
        "  Configuring Java..."
    )

    (
        java_home,
        jvm_dll,
    ) = configure_java_environment()

    print(
        f"  JAVA_HOME             : "
        f"{java_home}"
    )

    print(
        f"  JVM library           : "
        f"{jvm_dll}"
    )

    # ========================================================
    # JAVA EXECUTABLE CHECK
    # ========================================================

    java_check = subprocess.run(
        [
            str(
                java_home
                / "bin"
                / "java.exe"
            ),
            "-version",
        ],
        cwd=str(
            PROJECT_ROOT
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=os.environ.copy(),
    )

    if (
        java_check.returncode
        !=
        0
    ):
        raise RuntimeError(
            "Java executable check failed.\n\n"
            f"JAVA_HOME:\n"
            f"  {java_home}\n\n"
            "Output:\n"
            f"{java_check.stdout}"
        )

    print(
        "  Java executable       : PASS"
    )

    # ========================================================
    # JPYPE JVM DISCOVERY
    #
    # This checks the exact condition that caused Stage 5
    # to fail previously.
    # ========================================================

    print(
        "  Checking JPype JVM discovery..."
    )

    jpype_check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import jpype; "
                "print(jpype.getDefaultJVMPath())"
            ),
        ],
        cwd=str(
            PROJECT_ROOT
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=os.environ.copy(),
    )

    if (
        jpype_check.returncode
        !=
        0
    ):
        raise RuntimeError(
            "JPype could not locate jvm.dll.\n\n"
            f"JAVA_HOME:\n"
            f"  {os.getenv('JAVA_HOME')}\n\n"
            "JPype output:\n"
            f"{jpype_check.stdout}"
        )

    discovered_jvm = (
        jpype_check.stdout.strip()
    )

    print(
        "  JPype JVM discovery   : PASS"
    )

    print(
        f"  JPype JVM path        : "
        f"{discovered_jvm}"
    )

    # ========================================================
    # ACTUAL MOWL JVM INITIALIZATION
    #
    # Importing mOWL alone is not enough. This starts the JVM
    # in a separate subprocess using the same operation used
    # later in Stage 5.
    # ========================================================

    print(
        "  Testing mOWL JVM startup..."
    )

    mowl_jvm_check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import mowl; "
                "mowl.init_jvm('2g'); "
                "print('MOWL_JVM_STARTUP_PASS')"
            ),
        ],
        cwd=str(
            PROJECT_ROOT
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=os.environ.copy(),
    )

    if (
        mowl_jvm_check.returncode
        !=
        0
    ):
        raise RuntimeError(
            "mOWL JVM initialization failed "
            "during preflight.\n\n"
            f"Python:\n"
            f"  {sys.executable}\n\n"
            f"JAVA_HOME:\n"
            f"  {os.getenv('JAVA_HOME')}\n\n"
            f"jvm.dll:\n"
            f"  {jvm_dll}\n\n"
            "mOWL output:\n"
            f"{mowl_jvm_check.stdout}"
        )

    if (
        "MOWL_JVM_STARTUP_PASS"
        not in
        mowl_jvm_check.stdout
    ):
        raise RuntimeError(
            "mOWL JVM subprocess returned "
            "success but did not emit the "
            "expected confirmation marker.\n\n"
            "Output:\n"
            f"{mowl_jvm_check.stdout}"
        )

    print(
        "  mOWL JVM startup      : PASS"
    )

    # ========================================================
    # XAI API KEY
    # ========================================================

    if not os.getenv(
        "XAI_API_KEY"
    ):
        raise RuntimeError(
            "XAI_API_KEY is not set.\n\n"
            "Set it in PowerShell with:\n\n"
            '$env:XAI_API_KEY = '
            '"your-new-api-key-here"'
        )

    print(
        "  xAI API key           : SET"
    )

    print()

    print(
        "Preflight checks        : PASS"
    )

    print()


# ============================================================
# BACKUP
# ============================================================

def create_backup():
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    backup_dir = (
        BACKUP_ROOT
        / timestamp
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    manifest = {
        "script_version":
            SCRIPT_VERSION,

        "created":
            datetime.now().isoformat(),

        "protected_directories":
            {},
    }

    print(
        "Creating pre-run safety backup..."
    )

    print(
        f"Backup location         : "
        f"{backup_dir}"
    )

    for source in PROTECTED_DIRECTORIES:
        destination = (
            backup_dir
            / source.name
        )

        existed = (
            source.exists()
        )

        manifest[
            "protected_directories"
        ][
            source.name
        ] = {
            "source":
                str(source),

            "backup":
                str(destination),

            "existed_before_run":
                existed,
        }

        if existed:
            print(
                f"  Backing up "
                f"{source.name}/ ..."
            )

            shutil.copytree(
                source,
                destination,
                copy_function=shutil.copy2,
            )

        else:
            print(
                f"  {source.name}/ "
                "did not exist; "
                "recording absent state."
            )

    with open(
        backup_dir
        / "backup_manifest.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
        )

    print(
        "Safety backup           : PASS"
    )

    print()

    return (
        backup_dir,
        manifest,
    )


# ============================================================
# FAILURE DIAGNOSTICS
# ============================================================

def preserve_failure_diagnostics(
    backup_dir: Path,
    exception,
):
    diagnostics_dir = (
        backup_dir
        / "failure_diagnostics"
    )

    diagnostics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        diagnostics_dir
        / "failure.txt",
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            f"RUN_ALL_VERSION: "
            f"{SCRIPT_VERSION}\n"
        )

        handle.write(
            f"TIME: "
            f"{datetime.now().isoformat()}\n\n"
        )

        handle.write(
            f"EXCEPTION:\n"
            f"{repr(exception)}\n\n"
        )

        handle.write(
            "TRACEBACK:\n"
        )

        handle.write(
            traceback.format_exc()
        )

    if LOG_DIR.exists():
        shutil.copytree(
            LOG_DIR,
            diagnostics_dir
            / "run_all_logs",
            dirs_exist_ok=True,
            copy_function=shutil.copy2,
        )

    useful_files = [
        SPLIT_REPORT_FILE,
        MOWL_METRICS_FILE,
        BEST_CONFIG_FILE,
        V8_VALIDATION_FILE,
        HYBRID_SCORES_FILE,
        RECIPROCAL_REVIEW_FILE,
        ROBOT_REPORT_FILE,
        ROBOT_LOG_FILE,
        ELK_DEBUG_FILE,
    ]

    for source in useful_files:
        if source.exists():
            try:
                shutil.copy2(
                    source,
                    diagnostics_dir
                    / source.name,
                )

            except Exception:
                pass

    return (
        diagnostics_dir
    )


# ============================================================
# ROLLBACK
# ============================================================

def rollback_from_backup(
    backup_dir: Path,
    manifest,
):
    print()

    print(
        "=" * 78
    )

    print(
        "ROLLBACK - RESTORING PRE-RUN STATE"
    )

    print(
        "=" * 78
    )

    print()

    errors = []

    for (
        name,
        info,
    ) in manifest[
        "protected_directories"
    ].items():

        source = Path(
            info[
                "source"
            ]
        )

        backup_copy = Path(
            info[
                "backup"
            ]
        )

        existed_before = bool(
            info[
                "existed_before_run"
            ]
        )

        print(
            f"Restoring {name}/ ..."
        )

        try:
            remove_path(
                source
            )

            if existed_before:
                if not backup_copy.exists():
                    raise RuntimeError(
                        "Backup copy is missing: "
                        f"{backup_copy}"
                    )

                shutil.copytree(
                    backup_copy,
                    source,
                    copy_function=shutil.copy2,
                )

        except Exception as exc:
            errors.append(
                (
                    name,
                    str(exc),
                )
            )

    if errors:
        print()

        print(
            "ROLLBACK STATUS         : "
            "PARTIAL FAILURE"
        )

        for (
            name,
            error,
        ) in errors:
            print(
                f"  {name}: "
                f"{error}"
            )

        raise RuntimeError(
            "Automatic rollback encountered "
            "errors.\n"
            f"Untouched backup: "
            f"{backup_dir}"
        )

    print()

    print(
        "ROLLBACK STATUS         : PASS"
    )

    print(
        "Original files restored: YES"
    )

    print(
        f"Safety backup retained : "
        f"{backup_dir}"
    )


# ============================================================
# CLEAN GENERATED OUTPUTS
# ============================================================

def clean_pipeline_outputs():
    print(
        "Preparing clean generated-output state..."
    )

    for path in PIPELINE_OUTPUTS:
        remove_path(
            path
        )

    remove_path(
        LOG_DIR
    )

    print(
        "Generated outputs reset : PASS"
    )

    print(
        "Source module preserved : PASS"
    )

    print()


# ============================================================
# STAGE LOGGING
# ============================================================

def read_log_tail(
    path: Path,
    lines=60,
):
    if not path.exists():
        return ""

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        content = (
            handle.readlines()
        )

    return "".join(
        content[
            -lines:
        ]
    )


# ============================================================
# RUN ONE PIPELINE STAGE
# ============================================================

def run_stage(
    number,
    total,
    name,
    script,
    expected_outputs,
):
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        LOG_DIR
        /
        f"{number:02d}_{script.stem}.log"
    )

    command = [
        sys.executable,
        str(script),
    ]

    print(
        f"[{number}/{total}] "
        f"{name} ..."
    )

    start = (
        time.time()
    )

    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as log_handle:
        log_handle.write(
            "COMMAND\n"
        )

        log_handle.write(
            "=" * 78
            +
            "\n"
        )

        log_handle.write(
            " ".join(
                command
            )
            +
            "\n\n"
        )

        log_handle.write(
            "OUTPUT\n"
        )

        log_handle.write(
            "=" * 78
            +
            "\n"
        )

        process = subprocess.Popen(
            command,
            cwd=str(
                PROJECT_ROOT
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=os.environ.copy(),
        )

        if (
            process.stdout
            is not None
        ):
            for line in process.stdout:
                log_handle.write(
                    line
                )

                log_handle.flush()

        return_code = (
            process.wait()
        )

    elapsed = (
        time.time()
        -
        start
    )

    if (
        return_code
        !=
        0
    ):
        print(
            f"      FAIL "
            f"(exit code "
            f"{return_code})"
        )

        print()

        print(
            "Last lines from stage log:"
        )

        print(
            "-" * 78
        )

        print(
            read_log_tail(
                log_path
            )
        )

        print(
            "-" * 78
        )

        raise RuntimeError(
            f"Pipeline stage failed: "
            f"{name}\n"
            f"Log: "
            f"{log_path}"
        )

    missing = [
        path
        for path in expected_outputs
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            f"Stage '{name}' returned "
            "success but did not create:\n"
            +
            "\n".join(
                f"  {path}"
                for path in missing
            )
        )

    print(
        f"      PASS "
        f"({elapsed:.1f}s)"
    )

    return {
        "stage":
            number,

        "name":
            name,

        "script":
            str(script),

        "status":
            "PASS",

        "elapsed_seconds":
            elapsed,

        "log":
            str(log_path),

        "outputs":
            [
                str(path)
                for path in expected_outputs
            ],
    }


# ============================================================
# RECURSIVE JSON LOOKUP
# ============================================================

def find_json_value(
    obj,
    candidate_keys,
):
    if isinstance(
        obj,
        dict,
    ):
        for key in candidate_keys:
            if (
                key in obj
                and
                obj[
                    key
                ] is not None
            ):
                return (
                    obj[
                        key
                    ]
                )

        for value in obj.values():
            found = find_json_value(
                value,
                candidate_keys,
            )

            if (
                found
                is not None
            ):
                return found

    elif isinstance(
        obj,
        list,
    ):
        for item in obj:
            found = find_json_value(
                item,
                candidate_keys,
            )

            if (
                found
                is not None
            ):
                return found

    return None


# ============================================================
# MOWL METRICS
# ============================================================

def read_mowl_summary():
    metrics = load_json(
        MOWL_METRICS_FILE
    )

    mean_cos = find_json_value(
        metrics,
        [
            "selected_mean",
            "mean_cos",
            "mean_cosine",
            "validation_mean",
        ],
    )

    tau = find_json_value(
        metrics,
        [
            "selected_tau",
            "chosen_tau",
            "tau",
        ],
    )

    if (
        mean_cos
        is None
    ):
        raise RuntimeError(
            "Could not locate mean cosine in "
            "reports/mowl_metrics.json."
        )

    if (
        tau
        is None
    ):
        raise RuntimeError(
            "Could not locate selected tau in "
            "reports/mowl_metrics.json."
        )

    mean_cos = float(
        mean_cos
    )

    tau = float(
        tau
    )

    if (
        mean_cos
        <
        MIN_MEAN_COSINE
    ):
        raise RuntimeError(
            "mOWL mean cosine did not "
            "meet the project requirement.\n"
            f"Mean cosine: "
            f"{mean_cos:.4f}\n"
            f"Required:    "
            f"{MIN_MEAN_COSINE:.4f}"
        )

    return {
        "mean_cos":
            mean_cos,

        "tau":
            tau,
    }


# ============================================================
# ACCEPTED AXIOMS
# ============================================================

def count_accepted_axioms():
    graph = load_graph(
        ACCEPTED_FILE
    )

    count = len(
        named_subclass_pairs(
            graph
        )
    )

    if (
        count
        <
        MIN_ACCEPTED_AXIOMS
    ):
        raise RuntimeError(
            f"Too few accepted axioms: "
            f"{count}. "
            f"Minimum expected: "
            f"{MIN_ACCEPTED_AXIOMS}."
        )

    return count


# ============================================================
# LLM DECISION IMPACT
# ============================================================

def read_llm_decision_impact():
    if not HYBRID_SCORES_FILE.exists():
        return None

    scored = 0
    changed = 0
    rescued = 0
    vetoed = 0

    with open(
        HYBRID_SCORES_FILE,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for row in reader:
            cosine_text = (
                row.get(
                    "mowl_cosine",
                    "",
                )
                or ""
            ).strip()

            if not cosine_text:
                continue

            cosine = float(
                cosine_text
            )

            threshold_text = (
                row.get(
                    "threshold",
                    "",
                )
                or ""
            ).strip()

            threshold = (
                float(
                    threshold_text
                )
                if threshold_text
                else 0.60
            )

            cosine_only = (
                cosine
                >=
                threshold
            )

            preliminary = (
                row.get(
                    "preliminary_pass",
                    "",
                )
                or ""
            ).strip().lower()

            if preliminary:
                hybrid_accept = (
                    preliminary
                    in {
                        "true",
                        "1",
                        "yes",
                        "y",
                    }
                )

            else:
                decision = (
                    row.get(
                        "decision",
                        "",
                    )
                    or ""
                ).strip().upper()

                hybrid_accept = (
                    decision
                    ==
                    "ACCEPT"
                )

            scored += 1

            if (
                hybrid_accept
                !=
                cosine_only
            ):
                changed += 1

                if hybrid_accept:
                    rescued += 1

                else:
                    vetoed += 1

    if scored == 0:
        return None

    return {
        "scored_candidates":
            scored,

        "decision_changes":
            changed,

        "decision_impact_rate":
            changed
            /
            scored,

        "rescued":
            rescued,

        "vetoed":
            vetoed,
    }


# ============================================================
# ROBOT / ELK REPORT
# ============================================================

def read_consistency_status():
    report = load_json(
        ROBOT_REPORT_FILE
    )

    status = find_json_value(
        report,
        [
            "status",
            "consistency_status",
            "reasoning_status",
        ],
    )

    exit_code = find_json_value(
        report,
        [
            "robot_exit_code",
            "exit_code",
        ],
    )

    preserved = find_json_value(
        report,
        [
            "accepted_axioms_preserved",
            "accepted_preserved",
        ],
    )

    normalized_status = (
        str(
            status
        ).upper()
        if (
            status
            is not None
        )
        else ""
    )

    try:
        normalized_exit_code = int(
            exit_code
        )

    except (
        TypeError,
        ValueError,
    ):
        normalized_exit_code = (
            None
        )

    passed = (
        normalized_status
        ==
        "PASS"
        and
        normalized_exit_code
        ==
        0
        and
        preserved
        is True
    )

    return {
        "status":
            (
                "PASS"
                if passed
                else "FAIL"
            ),

        "robot_exit_code":
            exit_code,

        "accepted_axioms_preserved":
            preserved,
    }


# ============================================================
# EXECUTE FULL SEQUENTIAL PIPELINE
# ============================================================

def execute_pipeline(
    backup_dir,
):
    total = 7
    stages = []

    start = (
        time.time()
    )

    # ========================================================
    # 1. EXTRACT DEFINITIONS
    # ========================================================

    stages.append(
        run_stage(
            1,
            total,
            "Extract definitions from module.ttl",
            EXTRACT_SCRIPT,
            [
                DEFINITIONS_FILE,
            ],
        )
    )

    # ========================================================
    # 2. PREPROCESS DEFINITIONS WITH LLM
    # ========================================================

    stages.append(
        run_stage(
            2,
            total,
            "Preprocess extracted definitions with LLM",
            PREPROCESS_SCRIPT,
            [
                DEFINITIONS_ENRICHED_FILE,
            ],
        )
    )

    # ========================================================
    # 3. GENERATE CANDIDATE EL AXIOMS
    # ========================================================

    stages.append(
        run_stage(
            3,
            total,
            "Generate OWL 2 EL candidate axioms",
            GENERATE_SCRIPT,
            [
                CANDIDATE_FILE,
            ],
        )
    )

    # ========================================================
    # 4. SPLIT MODULE INTO TRAIN / VALID
    # ========================================================

    stages.append(
        run_stage(
            4,
            total,
            "Split module.ttl into train.ttl and valid.ttl",
            SPLIT_SCRIPT,
            [
                TRAIN_FILE,
                VALID_FILE,
                SPLIT_REPORT_FILE,
            ],
        )
    )

    split_report = load_json(
        SPLIT_REPORT_FILE
    )

    if (
        str(
            split_report.get(
                "status",
                "",
            )
        ).upper()
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Stage 4 split report "
            "did not report PASS."
        )

    # ========================================================
    # 5. TRAIN + EVALUATE MOWL
    # ========================================================

    stages.append(
        run_stage(
            5,
            total,
            "Train and evaluate mOWL",
            TRAIN_MOWL_SCRIPT,
            [
                MOWL_METRICS_FILE,
                NATIVE_MODEL_FILE,
                PROJECTION_FILE,
                BEST_CONFIG_FILE,
                V8_VALIDATION_FILE,
            ],
        )
    )

    mowl = (
        read_mowl_summary()
    )

    # ========================================================
    # 6. HYBRID MOWL + LLM FILTER
    # ========================================================

    stages.append(
        run_stage(
            6,
            total,
            "Filter candidates with mOWL + LLM",
            FILTER_SCRIPT,
            [
                ACCEPTED_FILE,
                HYBRID_SCORES_FILE,
            ],
        )
    )

    accepted_count = (
        count_accepted_axioms()
    )

    llm_impact = (
        read_llm_decision_impact()
    )

    # ========================================================
    # 7. ROBOT MERGE + ELK
    # ========================================================

    stages.append(
        run_stage(
            7,
            total,
            "Merge accepted axioms and reason with ELK",
            MERGE_REASON_SCRIPT,
            [
                AUGMENTED_FILE,
                ROBOT_REPORT_FILE,
            ],
        )
    )

    consistency = (
        read_consistency_status()
    )

    if (
        consistency[
            "status"
        ]
        !=
        "PASS"
    ):
        raise RuntimeError(
            "ROBOT/ELK consistency stage "
            "did not report PASS."
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    final_graph = load_graph(
        AUGMENTED_FILE
    )

    elapsed = (
        time.time()
        -
        start
    )

    summary = {
        "script_version":
            SCRIPT_VERSION,

        "status":
            "PASS",

        "source_module":
            str(
                MODULE_FILE
            ),

        "backup":
            str(
                backup_dir
            ),

        "mean_cos":
            mowl[
                "mean_cos"
            ],

        "chosen_tau":
            mowl[
                "tau"
            ],

        "accepted_axioms":
            accepted_count,

        "cosine_contribution_rate":
            CONFIGURED_COSINE_WEIGHT,

        "llm_contribution_rate":
            CONFIGURED_LLM_WEIGHT,

        "llm_decision_impact":
            llm_impact,

        "consistency_status":
            consistency[
                "status"
            ],

        "robot_exit_code":
            consistency[
                "robot_exit_code"
            ],

        "accepted_axioms_preserved":
            consistency[
                "accepted_axioms_preserved"
            ],

        "split":
            split_report,

        "final_ontology":
            str(
                AUGMENTED_FILE
            ),

        "final_triples":
            len(
                final_graph
            ),

        "elapsed_seconds":
            elapsed,

        "stages":
            stages,
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print()

    print(
        "=" * 78
    )

    print(
        "PROJECT 5 - FINAL PIPELINE REPORT"
    )

    print(
        "=" * 78
    )

    print()

    print(
        f"mean_cos               : "
        f"{mowl['mean_cos']:.4f}"
    )

    print(
        f"chosen tau             : "
        f"{mowl['tau']:.2f}"
    )

    print(
        f"accepted axioms        : "
        f"{accepted_count}"
    )

    print(
        f"LLM contribution rate  : "
        f"{CONFIGURED_LLM_WEIGHT * 100:.1f}%"
    )

    if (
        llm_impact
        is not None
    ):
        print(
            f"LLM decision impact    : "
            f"{llm_impact['decision_impact_rate'] * 100:.1f}% "
            f"("
            f"{llm_impact['decision_changes']}/"
            f"{llm_impact['scored_candidates']}"
            f")"
        )

    print(
        f"consistency status     : "
        f"{consistency['status']}"
    )

    train_count = (
        split_report.get(
            "training_named_subclass_axioms",
            "?",
        )
    )

    valid_count = (
        split_report.get(
            "validation_named_subclass_axioms",
            "?",
        )
    )

    print(
        f"train / valid axioms   : "
        f"{train_count} / "
        f"{valid_count}"
    )

    retention = (
        split_report.get(
            "closure_retention"
        )
    )

    if (
        retention
        is not None
    ):
        print(
            f"closure retention      : "
            f"{float(retention) * 100:.1f}%"
        )

    print(
        f"final ontology triples : "
        f"{len(final_graph)}"
    )

    print()

    print(
        f"FINAL ONTOLOGY         : "
        f"{AUGMENTED_FILE}"
    )

    print(
        f"SUMMARY REPORT         : "
        f"{SUMMARY_FILE}"
    )

    print(
        f"STAGE LOGS             : "
        f"{LOG_DIR}"
    )

    print(
        f"SAFETY BACKUP          : "
        f"{backup_dir}"
    )

    print()

    print(
        "=" * 78
    )

    print(
        "PROJECT 5 PIPELINE: PASS"
    )

    print(
        "=" * 78
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        f"RUN_ALL_VERSION        : "
        f"{SCRIPT_VERSION}"
    )

    print()

    print(
        "=" * 78
    )

    print(
        "PROJECT 5 - FULL SEQUENTIAL PIPELINE"
    )

    print(
        "=" * 78
    )

    print()

    print(
        f"Project root            : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Python                  : "
        f"{sys.executable}"
    )

    print(
        f"Source ontology         : "
        f"{MODULE_FILE}"
    )

    print()

    # ========================================================
    # VERIFY ALL LOCAL REQUIREMENTS FIRST
    #
    # This happens before the backup and before any expensive
    # xAI calls.
    # ========================================================

    preflight()

    # ========================================================
    # BACKUP CURRENT PROJECT OUTPUTS
    # ========================================================

    (
        backup_dir,
        manifest,
    ) = create_backup()

    try:
        # ----------------------------------------------------
        # Clean generated artifacts while preserving
        # src/module.ttl.
        # ----------------------------------------------------

        clean_pipeline_outputs()

        # ----------------------------------------------------
        # Run all seven scripts in assignment order.
        # ----------------------------------------------------

        execute_pipeline(
            backup_dir
        )

    except BaseException as exc:
        diagnostics_dir = (
            None
        )

        try:
            diagnostics_dir = (
                preserve_failure_diagnostics(
                    backup_dir,
                    exc,
                )
            )

        except Exception as diagnostic_error:
            print()

            print(
                "WARNING: could not preserve "
                "all diagnostics: "
                f"{diagnostic_error}"
            )

        try:
            rollback_from_backup(
                backup_dir,
                manifest,
            )

        except Exception as rollback_error:
            print()

            print(
                "=" * 78
            )

            print(
                "CRITICAL: AUTOMATIC "
                "ROLLBACK ERROR"
            )

            print(
                "=" * 78
            )

            print()

            print(
                str(
                    rollback_error
                )
            )

            print()

            print(
                "Your untouched pre-run "
                "backup is at:"
            )

            print(
                f"  {backup_dir}"
            )

            raise

        print()

        print(
            "=" * 78
        )

        print(
            "PROJECT 5 PIPELINE: FAILED"
        )

        print(
            "=" * 78
        )

        print()

        print(
            "The original pre-run files "
            "were restored."
        )

        print(
            f"Pre-run backup          : "
            f"{backup_dir}"
        )

        if (
            diagnostics_dir
            is not None
        ):
            print(
                f"Failure diagnostics     : "
                f"{diagnostics_dir}"
            )

        print(
            f"Original error          : "
            f"{exc}"
        )

        raise


if __name__ == "__main__":
    main()