#!/usr/bin/env python3
"""
Validate the RDF graph (measure_cco.ttl) against SHACL shapes (cco_shapes.ttl).

This script resolves file paths relative to itself, so you can run it from anywhere.
"""

import sys
from pathlib import Path
from pyshacl import validate

def main():
    # Determine base directory: .../assignment/src/scripts -> go up twice to /assignment/src
    SRC_DIR = Path(__file__).resolve().parents[1]
    DATA = SRC_DIR / "measure_cco.ttl"
    SHAPES = SRC_DIR / "shacl" / "cco_shapes.ttl"

    print("[shacl] DATA   :", DATA)
    print("[shacl] exists?:", DATA.exists())
    print("[shacl] SHAPES :", SHAPES)
    print("[shacl] exists?:", SHAPES.exists())

    if not DATA.exists() or not SHAPES.exists():
        print("❌ Could not locate required TTL files.", file=sys.stderr)
        sys.exit(2)

    conforms, report_graph, report_text = validate(
        data_graph=str(DATA),
        shacl_graph=str(SHAPES),
        data_graph_format="turtle",
        shacl_graph_format="turtle",
        inference="rdfs",
        allow_infos=True,
        allow_warnings=True,
        advanced=True,
    )

    print(report_text)
    if not conforms:
        print("❌ SHACL validation failed.")
        sys.exit(1)
    print("✅ SHACL validation passed.")

if __name__ == "__main__":
    main()
