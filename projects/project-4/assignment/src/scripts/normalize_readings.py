#1 Imports & paths
import pandas as pd
import json
from dateutil import parser as dateparser
from pathlib import Path

#2 Define input/output locations
<<<<<<< HEAD
IN_A = Path("projects/project-4/assignment/src/data/sensor_A.csv")
IN_B = Path("projects/project-4/assignment/src/data/sensor_B.json")

OUT  = Path("projects/project-4/assignment/src/data/readings_normalized.csv")
=======
IN_A = Path("src/data/sensor_A.csv")
IN_B = Path("src/data/sensor_B.json")

OUT  = Path("src/data/readings_normalized.csv")
>>>>>>> 10621c69cd26268eea7dddb9746f6f76015b3119

#3 Load Sensor A (CSV):
df_a = pd.read_csv(IN_A, dtype=str, keep_default_na=False, na_values=["", "NA", "NaN"])
   # Map columns to canonical names (EDIT to match the actual headers)
df_a = df_a.rename(columns={
       "asset_id": "artifact_id",
       "measure_type": "sdc_kind",
       "unit": "unit_label",
       "reading": "value",
       "time": "timestamp",
   })

# Keep only canonical columns that exist
df_a = df_a[[c for c in ["artifact_id","sdc_kind","unit_label","value","timestamp"] if c in df_a.columns]]

#4Handle either a list of objects or newline‑delimited JSON:
raw_txt = Path(IN_B).read_text(encoding="utf-8").strip()
try:
<<<<<<< HEAD
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
=======
       obj = json.loads(raw_txt)
       records = obj["records"] if isinstance(obj, dict) and "records" in obj else (obj if isinstance(obj, list) else [obj])
except json.JSONDecodeError:
       # NDJSON fallback
       records = [json.loads(line) for line in raw_txt.splitlines() if line.strip()]

df_b = pd.DataFrame([{
       "artifact_id": r.get("artifact") or r.get("asset") or r.get("artifact_id"),
       "sdc_kind":    r.get("sdc") or r.get("measure_type") or r.get("sdc_kind"),
       "unit_label":  r.get("uom") or r.get("unit") or r.get("unit_label"),
       "value":       r.get("val") or r.get("reading") or r.get("value"),
       "timestamp":   r.get("ts") or r.get("time") or r.get("timestamp"),
   } for r in records])
>>>>>>> 10621c69cd26268eea7dddb9746f6f76015b3119

#5**Concatenate A + B**
df = pd.concat([df_a, df_b], ignore_index=True)

#6 Trim whitespace + basic normalization
for col in ["artifact_id","sdc_kind","unit_label"]:
       df[col] = df[col].astype(str).str.strip()

<<<<<<< HEAD
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



# Normalize units
UNIT_MAP = {
    "celsius": "degC", "°c": "degC", "c": "degC", "degc": "degC", "°C": "degC",
    "fahrenheit": "degF", "f": "degF", "degf": "degF", "°f": "degF", "°F": "degF",
    "pounds per square inch": "PSI_gauge", "psi": "PSI_gauge",
    "kilopascal": "kPa_gauge", "kpa": "kPa_gauge", "KPA": "kPa_gauge", "kPa": "kPa_gauge",
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

=======
       # numeric
       df["value"] = pd.to_numeric(df["value"], errors="coerce")

       #7 Timestamp parsing to ISO 8601
       def to_iso8601(x):
        try:
           # auto-detect; if timezone missing, assume UTC
           dt = dateparser.parse(str(x))
           if dt.tzinfo is None:
               # You can choose a policy; here we treat naive as UTC
               import datetime, pytz
               dt = dt.replace(tzinfo=datetime.timezone.utc)
           return dt.astimezone(datetime.timezone.utc).isoformat().replace("+00:00","Z")
        except Exception:
           return None

        df["timestamp"] = df["timestamp"].apply(to_iso8601)

#8 Unit nomalizations
UNIT_MAP = {
       "celsius": "C", "°c": "C", "C": "C",
       "kilogram": "kg", "KG": "kg", "kg": "kg",
       "meter": "m", "M": "m", "m": "m",
   }
df["unit_label"] = df["unit_label"].str.lower().map(UNIT_MAP).fillna(df["unit_label"])

>>>>>>> 10621c69cd26268eea7dddb9746f6f76015b3119
#10 Sort for readability (optional)
df = df.sort_values(["artifact_id", "timestamp"]).reset_index(drop=True)

#11 Write output
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Wrote {OUT} with {len(df)} rows.")
