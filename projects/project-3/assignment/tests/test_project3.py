import re
from pathlib import Path
import pandas as pd
import pytest
from rdflib import Graph, RDF, RDFS, OWL
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = SRC / "data"
NB = ROOT / "notebooks"

ONTO_ABBRS = ["bfo", "ies", "ccom", "qudt", "ccot", "to"]
PAIRS = [("bfo-core", "ies"), ("ccom", "qudt"), ("ccot", "time")]

# ---------- Helpers ----------

def _any_exists(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def _find_cols(df: pd.DataFrame, token: str, min_count: int = 1):
    # return column names containing token (case-insensitive)
    cols = [c for c in df.columns if token in c.lower()]
    return cols if len(cols) >= min_count else []

def _has_two_iri_cols(df: pd.DataFrame):
    iri_cols = _find_cols(df, "iri", min_count=2)
    if iri_cols:
        return iri_cols[:2]
    # Fall back to known pairs
    for pair in (("left_iri", "right_iri"), ("iri_left", "iri_right")):
        if all(c in df.columns for c in pair):
            return list(pair)
    return None

def _parse_ttl(path: Path) -> Graph:
    g = Graph()
    g.parse(str(path), format="turtle")
    return g

# ---------- Tests ----------

def test_notebooks_present():
    # Presence-only (do not execute in CI)
    required = [
        NB / "week5_endpoint_client.ipynb",
        NB / "reasoner_run.ipynb",
        NB / "augment_w_definitions.ipynb",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    assert not missing, "Missing required notebooks:\n- " + "\n- ".join(missing)

def test_core_ttls_parse():
    core_ttls = [
        SRC / "bfo-core.ttl",
        SRC / "ies.ttl",
        SRC / "ccom.ttl",
        SRC / "qudt.ttl",
        SRC / "ccot.ttl",
        SRC / "time.ttl",
    ]
    missing = [str(p.relative_to(ROOT)) for p in core_ttls if not p.exists()]
    assert not missing, "Missing core TTLs:\n- " + "\n- ".join(missing)

    g = Graph()
    for p in core_ttls:
        try:
            g.parse(str(p), format="turtle")
        except Exception as e:
            raise AssertionError(f"Failed to parse {p}:\n{e}")

    assert len(g) >= 10, "Parsed TTLs but graph is suspiciously small (<10 triples)."

def test_class_exports_exist_and_have_iri_label_cols():
    problems = []
    for abbr in ONTO_ABBRS:
        f = DATA / f"{abbr}-class.xlsx"
        if not f.exists():
            problems.append(f"Missing class export: {f.relative_to(ROOT)}")
            continue
        try:
            df = pd.read_excel(f)
        except Exception as e:
            problems.append(f"Cannot open {f.name}: {e}")
            continue

        iri_cols = _find_cols(df, "iri")
        label_cols = _find_cols(df, "label")
        if not iri_cols or not label_cols:
            problems.append(
                f"{f.name}: need columns for IRIs and labels (case-insensitive match on 'iri' and 'label'); "
                f"found: {list(df.columns)}"
            )
    assert not problems, "Class export issues:\n- " + "\n- ".join(problems)

def test_axiom_exports_exist_and_have_iri_axiom_cols():
    problems = []
    for abbr in ONTO_ABBRS:
        f = DATA / f"{abbr}-axioms.xlsx"
        if not f.exists():
            problems.append(f"Missing axioms export: {f.relative_to(ROOT)}")
            continue
        try:
            df = pd.read_excel(f)
        except Exception as e:
            problems.append(f"Cannot open {f.name}: {e}")
            continue

        iri_cols = _find_cols(df, "iri")
        ax_cols = _find_cols(df, "axiom")
        if not iri_cols or not ax_cols:
            problems.append(
                f"{f.name}: need IRI and Axiom(s) columns (match on 'iri' and 'axiom'); "
                f"found: {list(df.columns)}"
            )

        # light check for deprecation traces in labels/axioms
        if _find_cols(df, "label"):
            lbl_col = _find_cols(df, "label")[0]
            if df[lbl_col].astype(str).str.contains("deprecated", case=False, na=False).any():
                problems.append(f"{f.name}: appears to include deprecated entries in labels.")
        for ac in ax_cols:
            if df[ac].astype(str).str.contains(r"owl:deprecated\s+\"?true\"?", case=False, na=False).any():
                problems.append(f"{f.name}: appears to include owl:deprecated axioms.")
    assert not problems, "Axiom export issues:\n- " + "\n- ".join(problems)

def test_structural_matches_exist_with_two_iri_columns():
    problems = []
    for a, b in PAIRS:
        fwd = DATA / f"{a}-{b}-structural-matches.xlsx"
        rev = DATA / f"{b}-{a}-structural-matches.xlsx"
        path = _any_exists(fwd, rev)
        if not path:
            problems.append(f"Missing structural matches for pair {a}-{b} (accepts forward or reverse filename).")
            continue

        try:
            df = pd.read_excel(path)
        except Exception as e:
            problems.append(f"Cannot open {path.name}: {e}")
            continue

        cols = _has_two_iri_cols(df)
        if not cols:
            problems.append(f"{path.name}: need two IRI columns; found {list(df.columns)}")
        if len(df) == 0:
            problems.append(f"{path.name}: no rows present.")
    assert not problems, "Structural matches issues:\n- " + "\n- ".join(problems)

def test_definitions_augmented_exist_and_have_two_definition_columns():
    problems = []
    for a, b in PAIRS:
        # Accept either direction and the "-with-definitions.xlsx" suffix
        base_fwd = DATA / f"{a}-{b}-structural-matches-with-defs.xlsx"
        base_rev = DATA / f"{b}-{a}-structural-matches-with-defs.xlsx"
        path = _any_exists(base_fwd, base_rev)
        if not path:
            problems.append(
                f"Missing definitions-augmented matches for pair {a}-{b} "
                f"(expected '*-structural-matches-with-defs.xlsx')."
            )
            continue

        try:
            df = pd.read_excel(path)
        except Exception as e:
            problems.append(f"Cannot open {path.name}: {e}")
            continue

        # Look for two definition columns (allow flexible naming)
        def_cols = [c for c in df.columns if "definition" in c.lower()]
        if len(def_cols) < 2:
            problems.append(f"{path.name}: need two definition columns (left/right); found {list(df.columns)}")
        else:
            # At least some rows have content in each definition column
            for c in def_cols[:2]:
                if not df[c].astype(str).str.strip().replace({"nan": ""}).any():
                    problems.append(f"{path.name}: column '{c}' is entirely empty.")
    assert not problems, "Definition augmentation issues:\n- " + "\n- ".join(problems)

def test_mapping_ttls_if_present_parse_and_contain_axioms():
    # Optional enrichment files: if present, they should parse and contain some mapping axiom
    mapping_files = [
        DATA / "bfo-mapping.ttl",
        DATA / "ies-mapping.ttl",
        DATA / "ccom-mapping.ttl",
        DATA / "qudt-mapping.ttl",
        DATA / "ccot-mapping.ttl",
        DATA / "to-mapping.ttl",
    ]
    present = [p for p in mapping_files if p.exists()]
    if not present:
        pytest.skip("No mapping TTLs present — skipping enrichment checks.")

    g = Graph()
    for p in present:
        try:
            g.parse(str(p), format="turtle")
        except Exception as e:
            raise AssertionError(f"Failed to parse mapping file {p}:\n{e}")

    # Require that at least one mapping axiom appears (either equivalentClass or subClassOf)
    has_equiv = any(True for _ in g.triples((None, OWL.equivalentClass, None)))
    has_sub = any(True for _ in g.triples((None, RDFS.subClassOf, None)))
    assert has_equiv or has_sub, (
        "Mapping TTLs present but no owl:equivalentClass or rdfs:subClassOf triples found."
    )

def test_core_files_still_parse_after_enrichment():
    # Ensure core files in src/ still parse (no syntax breakage after edits)
    ttls = list(SRC.glob("*.ttl"))
    assert ttls, "No TTL files found in src/."
    g = Graph()
    for t in ttls:
        try:
            g.parse(str(t), format="turtle")
        except Exception as e:
            raise AssertionError(f"Failed to parse {t} after enrichment:\n{e}")
