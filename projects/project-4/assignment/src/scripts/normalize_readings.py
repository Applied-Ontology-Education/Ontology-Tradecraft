import pandas as pd
import json
from dateutil import parser as dateparser
from pathlib import Path
import datetime

# Input files and output path
BASE = Path(__file__).resolve().parents[1]   # points to .../src
IN_A = BASE / "data" / "sensor_A.csv"
IN_B = BASE / "data" / "sensor_B.json"
OUT  = BASE / "data" / "readings_normalized.csv"

# 3. **Load Sensor A (CSV)**
df_a = pd.read_csv(IN_A, dtype=str, keep_default_na=False, na_values=["", "NA", "NaN"])

# Map columns to canonical names
df_a = df_a.rename(columns={
    "Device Name": "artifact_id",
    "Reading Type": "sdc_kind",
    "Units": "unit_label",
    "Reading Value": "value",
    "Time (Local)": "timestamp",
})

# Keep only canonical columns that exist
df_a = df_a[[c for c in ["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"] if c in df_a.columns]]

print(f"Loaded Sensor A: {len(df_a)} rows")

# 4. **Load Sensor B (JSON)** - FIXED to handle nested structure
raw_txt = Path(IN_B).read_text(encoding="utf-8").strip()
obj = json.loads(raw_txt)

# Flatten the nested structure: readings -> entity_id + data array
flattened_records = []
if "readings" in obj:
    for reading in obj["readings"]:
        entity_id = reading.get("entity_id")
        for data_point in reading.get("data", []):
            flattened_records.append({
                "artifact_id": entity_id,
                "sdc_kind": data_point.get("kind"),
                "unit_label": data_point.get("unit"),
                "value": data_point.get("value"),
                "timestamp": data_point.get("time")
            })

df_b = pd.DataFrame(flattened_records)

print(f"Loaded Sensor B: {len(df_b)} rows")

# 5. **Concatenate A + B**
df = pd.concat([df_a, df_b], ignore_index=True)

print(f"Combined dataset: {len(df)} rows")

# 6. **Trim whitespace + basic normalization**
for col in ["artifact_id", "sdc_kind", "unit_label"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Convert to numeric (will convert "not_a_number" and empty strings to NaN)
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# 7. **Timestamp parsing to ISO 8601**
def to_iso8601(x):
    try:
        # Auto-detect; if timezone missing, assume UTC
        dt = dateparser.parse(str(x))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return None

df["timestamp"] = df["timestamp"].apply(to_iso8601)

# 8. **Unit normalization**
UNIT_MAP = {
    "celsius": "C", "°c": "C", "c": "C",
    "fahrenheit": "F", "°f": "F", "f": "F",
    "kilogram": "kg", "KG": "kg", "kg": "kg",
    "meter": "m", "M": "m", "m": "m",
    "kilopascal": "kPa", "kpa": "kPa",
    "psi": "psi",
    "volt": "volt",
    "ohm": "ohm",
}
df["unit_label"] = df["unit_label"].str.lower().map(UNIT_MAP).fillna(df["unit_label"])

# 9. **Drop rows with missing critical values**
df = df.dropna(subset=["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"])

# 10. **Sort for readability**
df = df.sort_values(["artifact_id", "timestamp"]).reset_index(drop=True)

# 11. **Write output**
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Wrote {OUT} with {len(df)} rows.")
print("\nSample of output:")
print(df.head(10))