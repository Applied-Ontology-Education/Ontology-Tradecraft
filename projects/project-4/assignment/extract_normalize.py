#!/usr/bin/env python3
"""
Normalize heterogeneous measurement feeds to a common CSV:
observation_id, observed_entity_id, quantity_kind, value, unit_code, timestamp, source

Run:
  python scripts/normalize_readings.py
Outputs:
  data/interim/readings_normalized.csv
"""

import csv
import json
import logging
import math
import pathlib
from datetime import timezone

import pandas as pd

# Updated path to your GitHub repository
ROOT = pathlib.Path(r"C:\Users\crist\Documents\GitHub\Ontology-Tradecraft\Ontology-Tradecraft\projects\project-4\assignment\src")
SRC = ROOT / "data" 
OUT_DIR = ROOT / "data" / "interim"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "readings_normalized.csv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ------------------------
# Controlled vocab & maps
# ------------------------

# TODO: Adjust to your course's vocabulary.
KIND_MAP = {
    # sensor_A.csv -> normalized
    "temperature": "temperature",
    "pressure": "pressure",
    # sensor_B.json "kind" values
    "temp": "temperature",
}

# TODO: Normalize unit spellings to a compact code (no IRIs yet—this is pre-ontology).
UNIT_MAP = {
    "F": "F",
    "°F": "F",
    "degF": "F",
    "C": "C",
    "°C": "C",
    "PSI": "psi",
    "psi": "psi",
    "kpa": "kPa",
    "KPA": "kPa",
    "kPa": "kPa",
}

# Basic sanity bounds (pre-ETL triage). Adjust as needed.
BOUNDS = {
    "temperature": {"min": -100.0, "max": 200.0},  # in reported units (mixed!)
    "pressure": {"min": 0.0, "max": 1_000_000.0},
}

# ------------------------
# Utility functions
# ------------------------

def _canon_entity_id(s: str) -> str:
    """Trim, collapse spaces, standardize hyphen/space pattern."""
    if s is None:
        return None
    s = str(s).strip()
    # collapse multiple spaces
    s = " ".join(s.split())
    # normalize "Chiller 3" -> "Chiller-3"
    s = s.replace(" ", "-")
    return s

def _canon_kind(s: str) -> str:
    if s is None:
        return None
    key = str(s).strip().lower()
    return KIND_MAP.get(key, key)  # leave unknowns for later review

def _canon_unit(s: str) -> str:
    if s is None:
        return None
    key = str(s).strip()
    # uppercase letters only for lookup robustness
    mapped = UNIT_MAP.get(key, UNIT_MAP.get(key.upper(), UNIT_MAP.get(key.lower())))
    return mapped if mapped else key

def _parse_timestamp_any(s: str) -> pd.Timestamp:
    """Try multiple formats; force UTC. Leave NaT if unparseable."""
    if s is None or str(s).strip() == "":
        return pd.NaT
    try:
        ts = pd.to_datetime(s, utc=True, infer_datetime_format=True)
        return ts
    except Exception:
        return pd.NaT

def _coerce_float(x):
    """Return float or NaN; strings like 'not_a_number' become NaN."""
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return math.nan
        return float(x)
    except Exception:
        return math.nan

# ------------------------
# Normalizers per source
# ------------------------

def normalize_csv_sensor_a(path: pathlib.Path) -> pd.DataFrame:
    """
    sensor_A.csv columns:
      Device Name, Reading Type, Reading Value, Units, Time (Local)
    Encoding:
      Windows-1252 (cp1252)
    """
    logging.info(f"Reading {path.name} (cp1252)")
    df = pd.read_csv(path, encoding="cp1252")

    df = df.rename(
        columns={
            "Device Name": "observed_entity_id",
            "Reading Type": "quantity_kind",
            "Reading Value": "value",
            "Units": "unit_code",
            "Time (Local)": "timestamp",
        }
    )

    # Canonicalize
    df["observed_entity_id"] = df["observed_entity_id"].map(_canon_entity_id)
    df["quantity_kind"] = df["quantity_kind"].map(_canon_kind)
    df["unit_code"] = df["unit_code"].map(_canon_unit)
    df["value"] = df["value"].map(_coerce_float)
    df["timestamp"] = df["timestamp"].map(_parse_timestamp_any)

    # Add source
    df["source"] = "sensor_A"

    # Order & select
    df = df[["observed_entity_id", "quantity_kind", "value", "unit_code", "timestamp", "source"]]
    return df


def normalize_json_sensor_b(path: pathlib.Path) -> pd.DataFrame:
    """
    sensor_B.json structure:
    {
      "site": "...",
      "stream_id": "...",
      "readings": [
        {"entity_id": "...", "data": [{"kind": "...", "value": ..., "unit": "...", "time": "..."}]}
      ]
    }
    """
    logging.info(f"Reading {path.name} (utf-8)")
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)

    rows = []
    for entry in j.get("readings", []):
        entity = _canon_entity_id(entry.get("entity_id"))
        for d in entry.get("data", []):
            kind = _canon_kind(d.get("kind"))
            unit = _canon_unit(d.get("unit"))
            val = _coerce_float(d.get("value"))
            ts = _parse_timestamp_any(d.get("time"))
            rows.append(
                {
                    "observed_entity_id": entity,
                    "quantity_kind": kind,
                    "value": val,
                    "unit_code": unit,
                    "timestamp": ts,
                    "source": "sensor_B",
                }
            )
    df = pd.DataFrame.from_records(rows)
    return df

# ------------------------
# Validation / triage helpers
# ------------------------

def quick_triage(df: pd.DataFrame) -> None:
    """Lightweight checks BEFORE ontology. Print warnings, not hard failures."""
    # Types
    if df["value"].dtype == "object":
        logging.warning("Column 'value' is object dtype (mixed). Ensure numeric coercion happened.")

    # Missingness
    miss = df.isna().sum()
    if miss.get("value", 0) > 0:
        logging.warning(f"Missing numeric values: {miss['value']}")
    if miss.get("timestamp", 0) > 0:
        logging.warning(f"Unparseable timestamps (NaT): {miss['timestamp']}")

    # Unknown kinds/units
    unk_kinds = set(df["quantity_kind"].dropna()) - set(KIND_MAP.values())
    if unk_kinds:
        logging.warning(f"Unknown quantity_kind values present: {sorted(unk_kinds)}")

    # Sanity bounds (rough, pre-conversion)
    for kind, bounds in BOUNDS.items():
        sub = df[df["quantity_kind"] == kind]
        if not sub.empty:
            too_low = sub["value"] < bounds["min"]
            too_high = sub["value"] > bounds["max"]
            if too_low.any() or too_high.any():
                logging.warning(
                    f"{kind}: found out-of-range values "
                    f"(<{bounds['min']} or >{bounds['max']}) — check units/parsing."
                )

def add_observation_ids(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Concatenate and add stable observation_id like A-000001 / B-000001."""
    df_a = df_a.copy()
    df_b = df_b.copy()
    df_a.insert(0, "observation_id", ["A-" + str(i + 1).zfill(6) for i in range(len(df_a))])
    df_b.insert(0, "observation_id", ["B-" + str(i + 1).zfill(6) for i in range(len(df_b))])
    return pd.concat([df_a, df_b], ignore_index=True)

# ------------------------
# Main
# ------------------------

def main():
    a_path = SRC / "sensor_A.csv"
    b_path = SRC / "sensor_B.json"

    if not a_path.exists() or not b_path.exists():
        raise SystemExit(f"Missing input files in {SRC}: sensor_A.csv and/or sensor_B.json")

    df_a = normalize_csv_sensor_a(a_path)
    df_b = normalize_json_sensor_b(b_path)

    df = add_observation_ids(df_a, df_b)

    # Quick triage (prints warnings)
    quick_triage(df)

    # Enforce column order
    cols = ["observation_id", "observed_entity_id", "quantity_kind", "value", "unit_code", "timestamp", "source"]
    df = df[cols]

    # Final: write CSV with explicit encoding & ISO dates
    # NOTE: timestamps will be written as ISO strings by pandas.
    df.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"\n✓ Successfully wrote normalized data to: {OUT_CSV}")
    print(f"  Total records: {len(df)}")

    # Minimal summary students can expand:
    print("\n=== Normalized Summary ===")
    print(df.dtypes)
    print("\nCounts by quantity_kind:")
    print(df["quantity_kind"].value_counts(dropna=False))
    print("\nCounts by unit_code:")
    print(df["unit_code"].value_counts(dropna=False))
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nBasic stats by quantity_kind:")
    print(df.groupby("quantity_kind")["value"].agg(["count", "min", "max", "mean"]).to_string())

if __name__ == "__main__":
    main()