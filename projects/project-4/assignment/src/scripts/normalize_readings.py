#!/usr/bin/env python3

import json
import logging
import math
import pathlib
from datetime import datetime

import pandas as pd
import pytz

pd.options.display.max_columns = None
pd.options.display.max_rows = None

SRC_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_SOURCE = SRC_ROOT / "data"
OUT_CSV = DATA_SOURCE / "readings_normalized.csv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ------------------------
# Controlled vocab & maps
# ------------------------


KIND_MAP = {
    # sensor_A.csv -> normalized
    "temperature": "temperature",
    "pressure": "pressure",
    # sensor_B.json "kind" values
    "temp": "temperature",
    "resistance": "resistance",
    "voltage": "voltage"
}

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
    "pa": "Pa",
    "PA": "Pa",
    "Volt": "volt",
    "VOLT": "volt",
    "Ohm": "ohm",
    "OHM": "ohm",

}


# ------------------------
# Utility functions
# ------------------------

def standardize_artifact_id(artifact_id: str) -> str:
    """Trim, collapse spaces, standardize hyphen/space pattern."""
    if artifact_id is None:
        return None
    artifact_id = str(artifact_id).strip()
    # collapse multiple spaces
    artifact_id = " ".join(artifact_id.split())
    # normalize "Chiller 3" -> "Chiller-3"
    artifact_id = artifact_id.replace(" ", "-")
    return artifact_id


def standardize_kind(s: str) -> str:
    if s is None:
        return None
    key = str(s).strip().lower()
    return KIND_MAP.get(key, key)  # leave unknowns for later review


def standardize_unit(s: str) -> str:
    if s is None:
        return None
    key = str(s).strip()
    # uppercase letters only for lookup robustness
    mapped = UNIT_MAP.get(key, UNIT_MAP.get(key.upper(), UNIT_MAP.get(key.lower())))
    return mapped if mapped else key


def parse_timestamp_string(s: str) -> pd.Timestamp:
    if s is None or str(s).strip() == "":
        return pd.NaT
    try:
        ts = pd.to_datetime(s, utc=True).strftime('%Y-%m-%dT%H:%M:%SZ')
        return ts
    except Exception:
        return pd.NaT


def cast_to_float(x):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return math.nan
        return float(x)
    except Exception:
        return math.nan


# ------------------------
# Normalizers per source
# ------------------------
def local_time_to_utc(local_time_str):
    # Define the timezone for Buffalo, New York
    buffalo_tz = pytz.timezone('America/New_York')

    # Parse the local time string
    naive_dt = datetime.strptime(local_time_str, "%m/%d/%y %H:%M")

    # Localize to Buffalo timezone (this will handle DST automatically)
    local_dt = buffalo_tz.localize(naive_dt)

    # Convert to UTC
    utc_dt = local_dt.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')

    return utc_dt


def normalize_csv_sensor_a(path: pathlib.Path) -> pd.DataFrame:
    logging.info(f"Reading {path.name}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=["", "NA", "NaN"])

    df = df.rename(
        columns={
            "Device Name": "artifact_id",
            "Reading Type": "sdc_kind",
            "Reading Value": "value",
            "Units": "unit_label",
            "Time (Local)": "timestamp",
        }
    )

    # Keep only canonical columns that exist
    df = df[[c for c in ["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"] if c in df.columns]]

    # Canonicalize
    df["artifact_id"] = df["artifact_id"].map(standardize_artifact_id)
    df["sdc_kind"] = df["sdc_kind"].map(standardize_kind)
    df["value"] = df["value"].map(cast_to_float)
    df["timestamp"] = df["timestamp"].map(local_time_to_utc)

    return df


def normalize_json_sensor_b(path: pathlib.Path) -> pd.DataFrame:
    logging.info(f"Reading {path.name} (utf-8)")
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)

    rows = []
    for entry in j.get("readings", []):
        entity = standardize_artifact_id(entry.get("entity_id"))
        for d in entry.get("data", []):
            unit = standardize_unit(d.get("unit"))
            val = cast_to_float(d.get("value"))
            kind = standardize_kind(d.get("kind"))
            ts = parse_timestamp_string(d.get("time"))
            rows.append(
                {
                    "artifact_id": entity,
                    "sdc_kind": kind,
                    "unit_label": unit,
                    "value": val,
                    "timestamp": ts
                }
            )
    df = pd.DataFrame.from_records(rows)

    return df


def standardize_to_si(df):
    """Convert units to SI"""
    mask = df.unit_label == 'F'
    df.loc[mask, 'value'] = (df.loc[mask, 'value'] - 32) * 5 / 9
    df.loc[mask, 'unit_label'] = 'C'

    # Convert PSI values to kPa
    mask_psi = df.unit_label == 'psi'
    df.loc[mask_psi, 'value'] = df.loc[mask_psi, 'value'] * 6.89476 * 1000
    df.loc[mask_psi, 'unit_label'] = 'Pa'

    # Convert kPa values to Pa
    mask_psi = df.unit_label == 'kPa'
    df.loc[mask_psi, 'value'] = df.loc[mask_psi, 'value'] * 1000
    df.loc[mask_psi, 'unit_label'] = 'Pa'
    return df


def main():
    a_path = DATA_SOURCE / "sensor_A.csv"
    b_path = DATA_SOURCE / "sensor_B.json"
    c_path = DATA_SOURCE / "sensor_C.csv"

    print("[paths] A:", a_path)
    print("[paths] B:", b_path)
    print("[paths] C:", c_path)

    if not a_path.exists() or not b_path.exists() or not c_path.exists():
        logging.error(f"Missing one or more input files: {a_path}, {b_path}, {c_path}")

    df_a = normalize_csv_sensor_a(a_path)
    df_b = normalize_json_sensor_b(b_path)
    df_c = normalize_csv_sensor_a(c_path)

    # drop NaN
    df_a = df_a.dropna(subset=["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"])
    df_b = df_b.dropna(subset=["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"])
    df_c = df_b.dropna(subset=["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"])

    df = pd.concat([df_a, df_b, df_c], ignore_index=True)

    for col in ["artifact_id", "sdc_kind", "unit_label"]:
        df[col] = df[col].astype(str).str.strip()

    # numeric
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Timestamps are parsed per source

    # unit label standardization
    df["unit_label"] = df["unit_label"].map(standardize_unit)

    # sort values
    df = df.sort_values(["artifact_id", "timestamp"]).reset_index(drop=True)

    # Standardize units to SI further (psi to Pa, kPa to Pa)
    df = standardize_to_si(df)

    # Output
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} with {len(df)} rows.")


if __name__ == "__main__":
    main()
