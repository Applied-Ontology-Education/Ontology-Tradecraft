# 1 Imports & paths
import pandas as pd
import json
import datetime
from dateutil import parser as dateparser
from pathlib import Path

# 2 Define input/output locations relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent  # .../src/scripts
SRC_DIR = SCRIPT_DIR.parent  # .../src

IN_A = SRC_DIR / "data" / "sensor_A.csv"
IN_B = SRC_DIR / "data" / "sensor_B.json"
IN_C = SRC_DIR / "data" / "sensor_C.csv"

OUT = SRC_DIR / "data" / "readings_normalized.csv"


# 3 Define helper functions
def load_sensor_a(filepath):
    """Load Sensor A CSV data and normalize column names"""
    df = pd.read_csv(filepath, dtype=str, keep_default_na=False, na_values=["", "NA", "NaN"])
    
    # Map columns to canonical names
    df = df.rename(columns={
        "asset_id": "artifact_id",
        "measure_type": "sdc_kind",
        "unit": "unit_label",
        "reading": "value",
        "time": "timestamp",
    })
    
    # Keep only canonical columns that exist
    df = df[[c for c in ["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"] if c in df.columns]]
    
    return df


def load_sensor_b(filepath):
    """Load Sensor B JSON data and normalize to DataFrame"""
    raw_txt = filepath.read_text(encoding="utf-8").strip()
    
    try:
        obj = json.loads(raw_txt)
        
        # Handle nested structure with readings array
        if "readings" in obj and isinstance(obj["readings"], list):
            records = []
            for reading_entry in obj["readings"]:
                entity_id = reading_entry.get("entity_id")
                for data_point in reading_entry.get("data", []):
                    records.append({
                        "artifact": entity_id,
                        "kind": data_point.get("kind"),
                        "uom": data_point.get("unit"),
                        "val": data_point.get("value"),
                        "ts": data_point.get("time"),
                    })
        # Fallback: flat structure
        elif "records" in obj and isinstance(obj, dict):
            records = obj["records"]
        elif isinstance(obj, list):
            records = obj
        else:
            records = [obj]
    
    except json.JSONDecodeError:
        # NDJSON fallback
        records = [json.loads(line) for line in raw_txt.splitlines() if line.strip()]
    
    df = pd.DataFrame([{
        "artifact_id": r.get("artifact") or r.get("asset") or r.get("entity_id"),
        "sdc_kind": r.get("kind") or r.get("measure_type") or r.get("sdc_kind"),
        "unit_label": r.get("uom") or r.get("unit") or r.get("unit_label"),
        "value": r.get("val") or r.get("reading") or r.get("value"),
        "timestamp": r.get("ts") or r.get("time") or r.get("timestamp"),
    } for r in records])
    
    return df


def to_iso8601(x):
    """Convert timestamp to ISO8601 format with UTC timezone"""
    if pd.isna(x) or str(x).strip() == "":
        return None
    try:
        dt = dateparser.parse(str(x))
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        iso_str = dt.astimezone(datetime.timezone.utc).isoformat()
        # Ensure it ends with Z
        iso_str = iso_str.replace("+00:00", "Z")
        if not iso_str.endswith("Z"):
            iso_str += "Z"
        return iso_str
    except Exception:
        return None


def normalize_and_clean(df):
    """Normalize units, kinds, and clean the dataframe"""
    # Trim whitespace + basic normalization
    for col in ["artifact_id", "sdc_kind", "unit_label"]:
        df[col] = df[col].astype(str).str.strip()
    
    # Convert to numeric
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    
    # Timestamp conversion
    df["timestamp"] = df["timestamp"].apply(to_iso8601)
    
    # Normalize units to short, simple labels (per ETL guide)
    UNIT_MAP = {
        "celsius": "C", "°c": "C", "c": "C", "degc": "C", "°C": "C",
        "fahrenheit": "F", "f": "F", "degf": "F", "°f": "F", "°F": "F",
        "pounds per square inch": "psi", "psi": "psi", "PSI": "psi",
        "kilopascal": "Pa", "kpa": "Pa", "KPA": "Pa", "kPa": "Pa",
        "volt": "V", "volts": "V", "v": "V", "V": "V",
        "ohm": "Ω", "ohms": "Ω", "Ω": "Ω",
    }
    
    df["unit_label"] = df["unit_label"].str.lower().map(UNIT_MAP).fillna(df["unit_label"])
    
    # Normalize quantity kinds
    KIND_MAP = {
        "temp": "temperature",
        "temperature": "temperature",
        "pressure": "pressure",
        "voltage": "voltage",
        "resistance": "resistance",
    }
    
    df["sdc_kind"] = df["sdc_kind"].str.lower().map(KIND_MAP).fillna(df["sdc_kind"])
    
    # Standardize artifact IDs (spaces to hyphens)
    df["artifact_id"] = df["artifact_id"].str.replace(" ", "-")
    
    # Drop incomplete rows
    df = df.dropna(subset=["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"])
    
    # Sort for readability
    df = df.sort_values(["artifact_id", "timestamp"]).reset_index(drop=True)
    
    return df


# 4 Load all sensor data
print("[normalize_readings] Loading sensor data...")
df_a = load_sensor_a(IN_A)
df_b = load_sensor_b(IN_B)
df_c = load_sensor_a(IN_C)  # Sensor C uses same format as A

print(f"[normalize_readings] Input A rows: {len(df_a)}")
print(f"[normalize_readings] Input B rows: {len(df_b)}")
print(f"[normalize_readings] Input C rows: {len(df_c)}")

# 5 Concatenate all dataframes
combined = pd.concat([df_a, df_b, df_c], ignore_index=True)
print(f"[normalize_readings] Combined rows: {len(combined)}")

# 6 Normalize and clean
cleaned = normalize_and_clean(combined)
print(f"[normalize_readings] Cleaned rows: {len(cleaned)}")

# 7 Write output
OUT.parent.mkdir(parents=True, exist_ok=True)
cleaned.to_csv(OUT, index=False)
print(f"[normalize_readings] ✓ Output written to: {OUT}")