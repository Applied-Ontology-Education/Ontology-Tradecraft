from pathlib import Path
import pathlib
from typing import List

from rdflib import Graph, Namespace

# Path settings
SRC_ROOT = pathlib.Path(__file__).resolve().parents[1]

def load_graph(ttl_paths: List[Path]) -> Graph:
    """

    """
    # Check file paths
    for path in ttl_paths:
        assert path.exists(), f"Missing file path: {path}"

    # Make repeatable
    ttl_paths = sorted(ttl_paths)

    # Create merged graph
    g = Graph()
    errors = []
    for path in ttl_paths:
        try:
            g.parse(str(path), format="turtle")
        except Exception as e:
            try:
                txt = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                txt = "<unable to read file>"
            tail = "\n".join(txt.splitlines()[-40:])
            errors.append(f"\n--- Parse error in {path} ---\n{e}\nLast 40 lines:\n{tail}\n")
    assert not errors, "".join(errors)
    return g

def main():
    # load the ttl's
    ttl_paths = [SRC_ROOT / "measure_cco_inferred.ttl", SRC_ROOT / "cco_merged.ttl"]
    graph = load_graph(ttl_paths)

    # set the default namespace
    default_ns = Namespace("http://www.newfoundland.nl/otc/project-4")
    graph.bind("", default_ns)

    # output the graph
    output_path = SRC_ROOT / "measure_cco_inferred_merged.ttl"
    graph.serialize(str(output_path), format="turtle")
    print(f"Merged graph saved to {output_path}")

if __name__ == "__main__":
    main()
