import pandas as pd
import json
from dateutil import parser as dateparser
from pathlib import Path
import datetime

# Paths
IN_A = Path("src/data/sensor_A.csv")
IN_B = Path("src/data/sensor_B.json")
IN_C = Path("src/data/sensor_C.csv")
OUT = Path("src/data/readings_normalized.csv")

# Read CSV A
df_a = pd.read_csv(IN_A, dtype=str, keep_default_na=False, na_values=["", "NA", "NaN"])
df_a = df_a.rename(columns={
    "Device Name": "artifact_id",
    "Reading Type": "sdc_kind",
    "Units": "unit_label",
    "Reading Value": "value",
    "Time (Local)": "timestamp",
})
df_a = df_a[[c for c in ["artifact_id","sdc_kind","unit_label","value","timestamp"] if c in df_a.columns]]

# Read CSV C
df_c = pd.read_csv(IN_C, dtype=str, keep_default_na=False, na_values=["", "NA", "NaN"])
df_c = df_c.rename(columns={
    "Device Name": "artifact_id",
    "Reading Type": "sdc_kind",
    "Units": "unit_label",
    "Reading Value": "value",
    "Time (Local)": "timestamp",
})
df_c = df_c[[c for c in ["artifact_id","sdc_kind","unit_label","value","timestamp"] if c in df_c.columns]]

# Read JSON B
raw_txt = Path(IN_B).read_text(encoding="utf-8").strip()
try:
    obj = json.loads(raw_txt)

    if "readings" in obj and isinstance(obj["readings"], list):
        # Nested structure with readings array
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

df_b = pd.DataFrame([{
    "artifact_id": r.get("artifact") or r.get("asset") or r.get("entity_id"),
    "sdc_kind":    r.get("kind") or r.get("measure_type") or r.get("sdc_kind"),
    "unit_label":  r.get("uom") or r.get("unit") or r.get("unit_label"),
    "value":       r.get("val") or r.get("reading") or r.get("value"),
    "timestamp":   r.get("ts") or r.get("time") or r.get("timestamp"),
} for r in records])

# PRINT STATEMENTS
print(f"[normalize_readings] Input A rows: {len(df_a)}")
print(f"[normalize_readings] Input B rows: {len(df_b)}")
print(f"[normalize_readings] Input C rows: {len(df_c)}")

# Combine
df = pd.concat([df_a, df_b, df_c], ignore_index=True)

# Clean strings
for col in ["artifact_id","sdc_kind","unit_label"]:
    df[col] = df[col].astype(str).str.strip()

# Convert to numeric
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# Timestamp conversion
def to_iso8601(x):
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

df["timestamp"] = df["timestamp"].apply(to_iso8601)

# Normalize units
UNIT_MAP = {
    # Temperature
    "celsius": "degC", "°c": "degC", "c": "degC", "degc": "degC", "°C": "degC",
    "fahrenheit": "degF", "f": "degF", "degf": "degF", "°f": "degF", "°F": "degF",
    # Pressure
    "pounds per square inch": "psi", "psi": "psi", "PSI": "psi",
    "kilopascal": "kPa", "kpa": "kPa", "KPA": "kPa", "kPa": "kPa",
    # Electrical
    "volt": "V", "volts": "V", "v": "V", "V": "V",
    "ohm": "Ω", "ohms": "Ω", "Ω": "Ω",
}

# FIXED: Apply unit normalization properly
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
df = df.dropna(subset=["artifact_id","sdc_kind","unit_label","value","timestamp"])

# Sort
df = df.sort_values(["artifact_id", "timestamp"]).reset_index(drop=True)

# Save
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"[normalize_readings] Wrote {OUT} with {len(df)} rows.")

