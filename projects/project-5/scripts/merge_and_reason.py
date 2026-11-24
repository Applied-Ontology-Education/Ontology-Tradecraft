import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from rdflib import Graph, OWL, RDF, RDFS, URIRef

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

# Single source of truth for the module file name.
from split_ontology import MODULE_PATH  # noqa: E402

ACCEPTED_PATH = PROJECT_DIR / "generated" / "accepted_el.ttl"
OUTPUT_PATH = PROJECT_DIR / "src" / "module_augmented.ttl"
REPORT_PATH = PROJECT_DIR / "reports" / "merge_report.json"

TOOLS_DIR = PROJECT_DIR / "tools"
ROBOT_JAR = TOOLS_DIR / "robot.jar"
BFO_PATH = TOOLS_DIR / "bfo-core.ttl"

ROBOT_URL = "https://github.com/ontodev/robot/releases/download/v1.9.7/robot.jar"
BFO_URL = (
    "https://raw.githubusercontent.com/BFO-ontology/BFO-2020/"
    "master/21838-2/owl/bfo-core.ttl"
)


# ============================================================
# SETUP
# ============================================================

def ensure_file(path, url, description):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {description} (one-time) ...")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(path, "wb") as out:
            shutil.copyfileobj(response, out)
    except Exception as error:
        if path.exists():
            path.unlink()
        raise RuntimeError(
            f"Could not download {description}: {error}\n"
            f"Download it manually from {url}\n"
            f"and save it as {path}"
        ) from error


def ensure_java():
    if shutil.which("java") is None:
        raise RuntimeError("Java is not on PATH; ROBOT needs Java 11 or newer.")


def robot(*args):
    command = ["java", "-jar", str(ROBOT_JAR), *args]
    completed = subprocess.run(command, capture_output=True, text=True)
    output = "\n".join(
        line
        for line in (completed.stdout + completed.stderr).splitlines()
        if "JAVA_TOOL_OPTIONS" not in line
    ).strip()
    return completed.returncode, output


# ============================================================
# HELPERS
# ============================================================

def named_subclass_pairs(graph):
    return {
        (s, o)
        for s, o in graph.subject_objects(RDFS.subClassOf)
        if isinstance(s, URIRef) and isinstance(o, URIRef) and o != OWL.Thing
    }


def write_import_free_copy(source, destination):
    graph = Graph()
    graph.parse(str(source), format="turtle")
    imports = list(graph.triples((None, OWL.imports, None)))
    graph.remove((None, OWL.imports, None))
    graph.serialize(str(destination), format="turtle")
    return imports


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nMERGE + ELK REASONING")
    print("=============================================")

    for path in (MODULE_PATH, ACCEPTED_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    ensure_java()
    ensure_file(ROBOT_JAR, ROBOT_URL, "ROBOT")
    ensure_file(BFO_PATH, BFO_URL, "BFO 2020 core")

    accepted = Graph()
    accepted.parse(str(ACCEPTED_PATH), format="turtle")
    accepted_pairs = named_subclass_pairs(accepted)

    module = Graph()
    module.parse(str(MODULE_PATH), format="turtle")
    original_pairs = named_subclass_pairs(module)

    new_pairs = accepted_pairs - original_pairs
    print(f"Accepted axioms in file:          {len(accepted_pairs)}")
    print(f"Of which new to the module:       {len(new_pairs)}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        module_noimports = tmp / "module_noimports.ttl"
        original_imports = write_import_free_copy(MODULE_PATH, module_noimports)

        # --- 1. Consistency check (with BFO) -------------------------
        code, output = robot(
            "merge",
            "--input", str(module_noimports),
            "--input", str(ACCEPTED_PATH),
            "--input", str(BFO_PATH),
            "reason", "--reasoner", "ELK",
            "--equivalent-classes-allowed", "all",
            "--output", str(tmp / "check.ttl"),
        )
        consistent = code == 0
        print(f"\nELK consistency check (module + accepted + BFO): "
              f"{'CONSISTENT' if consistent else 'FAILED'}")
        if not consistent:
            print(output)

        # --- 2. Augmented module ------------------------------------
        reasoned_path = tmp / "augmented_noimports.ttl"
        code, output = robot(
            "merge",
            "--input", str(module_noimports),
            "--input", str(ACCEPTED_PATH),
            "reason", "--reasoner", "ELK",
            "--equivalent-classes-allowed", "all",
            "--remove-redundant-subclass-axioms", "false",
            "--annotate-inferred-axioms", "true",
            "--output", str(reasoned_path),
        )
        if code != 0:
            print(output)
            raise RuntimeError("ROBOT merge/reason failed; module_augmented.ttl not written.")

        augmented = Graph()
        augmented.parse(str(reasoned_path), format="turtle")
        # ROBOT adds trivial "X SubClassOf owl:Thing" for root classes; drop them.
        augmented.remove((None, RDFS.subClassOf, OWL.Thing))
        for triple in original_imports:
            augmented.add(triple)
        for prefix, namespace in module.namespaces():
            augmented.bind(prefix, namespace, override=False)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        augmented.serialize(str(OUTPUT_PATH), format="turtle")

    augmented_pairs = named_subclass_pairs(augmented)
    added_by_accepted = len(new_pairs & augmented_pairs)
    inferred = augmented_pairs - original_pairs - new_pairs
    lost = original_pairs - augmented_pairs

    report = {
        "module": str(MODULE_PATH.name),
        "accepted_axioms": len(accepted_pairs),
        "new_axioms_added": added_by_accepted,
        "inferred_subclass_axioms_added_by_elk": len(inferred),
        "original_axioms_removed": len(lost),
        "consistent": consistent,
        "reasoner": "ELK (via ROBOT)",
        "consistency_check_included_bfo": True,
        "output": str(OUTPUT_PATH),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nMERGE COMPLETE")
    print("---------------------------------------------")
    print(f"New axioms added:                 {added_by_accepted}")
    print(f"Inferred subclass axioms (ELK):   {len(inferred)}")
    print(f"Original axioms removed:          {len(lost)}")
    print(f"Consistency:                      {'CONSISTENT' if consistent else 'INCONSISTENT'}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Report: {REPORT_PATH}")

    if not consistent:
        sys.exit(1)


if __name__ == "__main__":
    main()