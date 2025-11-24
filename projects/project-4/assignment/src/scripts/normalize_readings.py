
import pandas as pd
import json
from dateutil import parser as dateparser
from pathlib import Path

#input files and output path
# IN_A = Path("//src/data/sensor_A.csv")
# IN_B = Path("src/data/sensor_B.json")
# OUT  = Path("src/data/readings_normalized.csv")


BASE = Path(__file__).resolve().parents[1]   # points to .../src
IN_A = BASE / "data" / "sensor_A.csv"
IN_B = BASE / "data" / "sensor_B.json"
OUT  = BASE / "data" / "readings_normalized.csv"



# 3. **Load Sensor A (CSV)**   Device Name,Reading Type,Reading Value,Units,Time (Local)

df_a = pd.read_csv(IN_A, dtype=str, keep_default_na=False, na_values=["", "NA", "NaN"])
# Map columns to canonical names (EDIT to match the actual headers)
df_a = df_a.rename(columns={
    "Device Name": "artifact_id",
    "Reading Type": "sdc_kind",
    "Units": "unit_label",
    "Reading Value": "value",
    "Time (Local)": "timestamp",
})
   # Keep only canonical columns that exist
df_a = df_a[[c for c in ["artifact_id","sdc_kind","unit_label","value","timestamp"] if c in df_a.columns]]

#print(df_a)


# 4. **Load Sensor B (JSON)**  
#   Handle either a list of objects or newline‑delimited JSON:

raw_txt = Path(IN_B).read_text(encoding="utf-8").strip()
try:
    obj = json.loads(raw_txt)
    records = obj["records"] if isinstance(obj, dict) and "records" in obj else (obj if isinstance(obj, list) else [obj])
except json.JSONDecodeError:
    # NDJSON fallback
    records = [json.loads(line) for line in raw_txt.splitlines() if line.strip()]

# WRONG HEADERS TO FIX IN THE DATASET!
df_b = pd.DataFrame([{
    "artifact_id": r.get("artifact") or r.get("asset") or r.get("artifact_id"),
    "sdc_kind":    r.get("sdc") or r.get("measure_type") or r.get("sdc_kind"),
    "unit_label":  r.get("uom") or r.get("unit") or r.get("unit_label"),
    "value":       r.get("val") or r.get("reading") or r.get("value"),
    "timestamp":   r.get("ts") or r.get("time") or r.get("timestamp"),
} for r in records])


# df_b = pd.DataFrame([{
#     "artifact_id": r.get("entity_id"),
#     "sdc_kind": r.get("kind"),
#     "unit_label": r.get("unit"),
#     "value": r.get("value"),
#     "timestamp": r.get("time")
# } for r in records])

# df_b = pd.DataFrame([{
#     "artifact_id": r.get("artifact") or r.get("asset") or r.get("artifact_id") or r.get("entity_id"),
#     "sdc_kind":    r.get("sdc") or r.get("measure_type") or r.get("sdc_kind") or r.get("kind"),
#     "unit_label":  r.get("uom") or r.get("unit") or r.get("unit_label") or r.get("unit"),
#     "value":       r.get("val") or r.get("reading") or r.get("value") or r.get("value"),
#     "timestamp":   r.get("ts") or r.get("time") or r.get("timestamp") or r.get("time"),
# } for r in records])


print(df_b)

#5. **Concatenate A + B**

df = pd.concat([df_a, df_b], ignore_index=True)

#print(df.tail) 

# # 6. **Trim whitespace + basic normalization**

# for col in ["artifact_id","sdc_kind","unit_label"]:
#     df[col] = df[col].astype(str).str.strip()

#    # numeric
# df["value"] = pd.to_numeric(df["value"], errors="coerce")


# #7. **Timestamp parsing to ISO 8601**

# def to_iso8601(x):
#     try:
#         # auto-detect; if timezone missing, assume UTC
#         dt = dateparser.parse(str(x))
#         if dt.tzinfo is None:
#             # You can choose a policy; here we treat naive as UTC
#             import datetime, pytz
#             dt = dt.replace(tzinfo=datetime.timezone.utc)
#         return dt.astimezone(datetime.timezone.utc).isoformat().replace("+00:00","Z")
#     except Exception:
#         return None

# df["timestamp"] = df["timestamp"].apply(to_iso8601)


# # 8. **Unit normalization** (example mapping — edit to your real rules)

# UNIT_MAP = {
#        "celsius": "C", "°c": "C", "C": "C",
#        "kilogram": "kg", "KG": "kg", "kg": "kg",
#        "meter": "m", "M": "m", "m": "m",
#    }
# df["unit_label"] = df["unit_label"].str.lower().map(UNIT_MAP).fillna(df["unit_label"])


# # 9. **Drop rows with missing critical values**

# #df = df.dropna(subset=["artifact_id","sdc_kind","unit_label","value","timestamp"])


# # 10. **Sort for readability (optional)**

# df = df.sort_values(["artifact_id", "timestamp"]).reset_index(drop=True)


# # 11. **Write output**

# OUT.parent.mkdir(parents=True, exist_ok=True)
# df.to_csv(OUT, index=False)
# print(f"Wrote {OUT} with {len(df)} rows.")
