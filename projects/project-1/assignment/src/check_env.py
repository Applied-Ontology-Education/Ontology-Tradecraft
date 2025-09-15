# src/check_env.py
import json, sys
from pathlib import Path
import rdflib

def main():
    # assignment dir = .../projects/project-X/assignment OR repo root, but tests run from assignment dir
    base = Path(__file__).resolve().parents[1]
    ttl = base / "notebooks" / "sample.ttl"

    # Fallbacks if notebook/test runner executes from a different CWD
    if not ttl.exists():
        alt = Path.cwd() / "notebooks" / "sample.ttl"
        ttl = alt if alt.exists() else ttl

    g = rdflib.Graph() 
    g.parse(str(ttl), format="turtle")

    print(json.dumps({
        "python": sys.version.split()[0],
        "rdflib": rdflib.__version__,
        "triple_count": len(g)
    }))

if __name__ == "__main__":
    main()
