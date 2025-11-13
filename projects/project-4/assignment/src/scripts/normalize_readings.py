import pandas as pd
import json
from dateutil import parser as dateparser
from pathlib import Path
import datetime
import os
import argparse

# Determine repo root (adjust depth if needed)
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../assignment

# Defaults relative to repo
DEFAULT_IN_A = REPO_ROOT / "src" / "data" / "sensor_A.csv"
DEFAULT_IN_B = REPO_ROOT / "src" / "data" / "sensor_B.json"
DEFAULT_OUT = REPO_ROOT / "src" / "data" / "readings_normalized.csv"

# Environment overrides
env_in_a = os.getenv("IN_A", DEFAULT_IN_A)
env_in_b = os.getenv("IN_B", DEFAULT_IN_B)
env_out = os.getenv("OUT", DEFAULT_OUT)

# CLI overrides
parser = argparse.ArgumentParser()
parser.add_argument("--in_a", type=Path, default=env_in_a)
parser.add_argument("--in_b", type=Path, default=env_in_b)
parser.add_argument("--out", type=Path, default=env_out)
args = parser.parse_args()

IN_A, IN_B, OUT = args.in_a, args.in_b, args.out

# The data folder to watch for additional CSVs
DATA_DIR = IN_A.parent

# Files to exclude from auto-loading
EXCLUDE_CSV_NAMES = {IN_A.name, OUT.name}

# Canonical schema
RENAME_MAP_SENSOR_A = {
    "device name": "artifact_id",
    "reading type": "sdc_kind",
    "units": "unit_label",
    "reading value": "value",
    "time (local)": "timestamp",
}
CANONICAL_COLS = ["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"]

# Helper: load CSV like Sensor A
def load_csv_like_sensor_a(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=["", "NA", "NaN"])
    lower_to_original = {c.lower().strip(): c for c in df.columns}
    rename_dict = {}
    for src_lower, target in RENAME_MAP_SENSOR_A.items():
        if src_lower in lower_to_original:
            rename_dict[lower_to_original[src_lower]] = target
    df = df.rename(columns=rename_dict)
    keep_cols = [c for c in CANONICAL_COLS if c in df.columns]
    df = df[keep_cols]
    return df

# Load Sensor A
df_a = load_csv_like_sensor_a(IN_A)

# Load Sensor B
with open(IN_B, "r", encoding="utf-8") as f:
    data = json.load(f)

records = []
for reading in data.get("readings", []):
    entity_id = reading.get("entity_id")
    for entry in reading.get("data", []):
        records.append({
            "artifact_id": entity_id,
            "sdc_kind": entry.get("kind"),
            "unit_label": entry.get("unit"),
            "value": entry.get("value"),
            "timestamp": entry.get("time")
        })

df_b = pd.DataFrame(records)

# Load extra CSVs
extra_frames = []
for p in sorted(DATA_DIR.glob("*.csv")):
    if p.name not in EXCLUDE_CSV_NAMES:
        try:
            df_extra = load_csv_like_sensor_a(p)
            if not df_extra.empty:
                extra_frames.append(df_extra)
        except Exception as exc:
            print(f"[ERROR] Failed to load {p.name}: {exc}")

# Combine all
df = pd.concat([df_a] + extra_frames + [df_b], ignore_index=True)

# Strip whitespace from all canonical columns
for col in CANONICAL_COLS:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Convert value to numeric
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# Timestamp normalization (ISO 8601 UTC with Z)
def to_iso8601(x):
    try:
        dt = dateparser.parse(str(x))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    except Exception:
        return None

df["timestamp"] = df["timestamp"].apply(to_iso8601)

# Unit normalization
UNIT_MAP = {
    "celsius": "C", "°c": "C", "c": "C",
    "fahrenheit": "F", "°f": "F", "f": "F",
    "kilogram": "kg", "kg": "kg",
    "meter": "m", "metre": "m", "m": "m",
    "psi": "pound per square inch", "psi ": "pound per square inch",
    "kpa": "kilopascal", "kpa ": "kilopascal"
}
orig_unit = df["unit_label"]
normalized = orig_unit.str.lower().str.strip().map(UNIT_MAP)
df["unit_label"] = normalized.fillna(orig_unit)

# Drop rows with missing critical values
df = df.dropna(subset=CANONICAL_COLS)

# Remove rows with non-numeric values in 'value' (recommended for strict requirement)
df = df[pd.to_numeric(df['value'], errors='coerce').notna()]
# Or, raise an explicit error with information for debugging
non_numeric = df[~pd.to_numeric(df['value'], errors='coerce').notna()]
if not non_numeric.empty:
    raise ValueError(f"Non-numeric values in 'value': {non_numeric}")

# Remove duplicates
df = df.drop_duplicates(subset=CANONICAL_COLS)

# Sort for readability
df = df.sort_values(["artifact_id", "timestamp"]).reset_index(drop=True)

# Validate non-empty
if df.empty:
    raise ValueError("No valid rows after normalization. Cannot write empty CSV.")

# Ensure exact column order
df = df[CANONICAL_COLS]

# Write output
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Wrote {OUT} with {len(df)} rows.")