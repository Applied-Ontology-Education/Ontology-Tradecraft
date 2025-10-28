import os
import pathlib
import subprocess

# Path settings
ASSIGNMENT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = pathlib.Path(__file__).resolve().parents[1]
ROBOT = ASSIGNMENT_ROOT / 'robot/robot'

def infer_file(input_file: str, output_file: str, reasoner: str) -> None:
    """
    Executes reasoning on an input ontology file using a specified reasoner and
    writes the inferred ontology to an output file. The reasoning process
    relies on the ROBOT command-line tool, which must be accessible via the
    `ROBOT` environment variable.

    :param input_file: The file containing the input ontology.
    :type input_file: str
    :param output_file: The file where the inferred ontology will be written.
    :type output_file: str
    :param reasoner: The reasoning engine to be used (e.g., ELK, Hermit).
    :type reasoner: str
    :return: None
    :raises Exception: If the ROBOT environment variable is not set.
    """

    # If you want the results, use res = subprocess.run(
    res = subprocess.run(
        [
            ROBOT,
            "reason",
            "--input",
            input_file,
            "--output",
            output_file,
            "--create-new-ontology-with-annotations",
            "true",
            "--equivalent-classes-allowed",
            "all",
            "--include-indirect",
            "true",
            "--axiom-generators",
            '"SubClass EquivalentClass ClassAssertion PropertyAssertion"',
            "--reasoner",
            reasoner,
        ],
        capture_output=True,
        text=True,
    )
    print(res.stdout)
    print(res.stderr)
    if res.returncode != 0:
        raise Exception(f"Error running ROBOT: {res.stderr}")

def main():
    input_file = SRC_ROOT / "measure_cco.ttl"
    output_file = SRC_ROOT / "measure_cco_inferred.ttl"
    # input_file = ASSIGNMENT_ROOT / "test.ttl"
    # output_file = ASSIGNMENT_ROOT / "test_reasoned.ttl"
    print(output_file)
    infer_file(input_file, output_file, "hermit")


if __name__ == "__main__":
    main()