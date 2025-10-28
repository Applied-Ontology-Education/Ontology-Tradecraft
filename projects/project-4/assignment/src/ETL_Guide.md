
# ETL Guide: Normalizing Sensor Measurements with Pandas

**Goal:** Combine raw sources, such as `src/data/sensor_A.csv` and `src/data/sensor_B.json`, into a single clean CSV: `src/data/readings_normalized.csv`.

**“Clean”**
- **Consistent columns**: `artifact_id, sdc_kind, unit_label, value, timestamp`
- **Consistent datatypes**: `artifact_id|sdc_kind|unit_label` = strings; `value` = numeric; `timestamp` = ISO 8601 string (UTC preferred)
- **Consistent units**: normalize unit labels to a single spelling/abbreviation (e.g., `C` not `celsius`, `kg` not `Kilogram`)

You will write a **small Python script** (e.g., `src/scripts/normalize_readings.py`) that uses **Pandas** to load inputs, standardize, and output the normalized CSV. Use the following canonical scheme as a guide: 

| column       | type   | examples                          |
|--------------|--------|-----------------------------------|
| artifact_id  | str    | `A-001`, `pump-42`                |
| sdc_kind     | str    | `temperature`, `mass`, `length`   |
| unit_label   | str    | `C`, `kg`, `m`                    |
| value        | float  | `23.5`, `72.0`                    |
| timestamp    | str    | `2025-10-16T12:00:00Z`            |

## Outline for your Script 

1. **Imports & paths**
   ```python
   import pandas as pd
   import json
   from dateutil import parser as dateparser
   from pathlib import Path
   ```

2. **Define input/output locations**
   ```python
   IN_A = Path("src/data/sensor_A.csv")
   IN_B = Path("src/data/sensor_B.json")
   OUT  = Path("src/data/readings_normalized.csv")
   ```

3. **Load Sensor A (CSV)**
   ```python
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
   ```

4. **Load Sensor B (JSON)**  
   Handle either a list of objects or newline‑delimited JSON:
   ```python
   raw_txt = Path(IN_B).read_text(encoding="utf-8").strip()
   try:
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
   ```

5. **Concatenate A + B**
   ```python
   df = pd.concat([df_a, df_b], ignore_index=True)
   ```

6. **Trim whitespace + basic normalization**
   ```python
   for col in ["artifact_id","sdc_kind","unit_label"]:
       df[col] = df[col].astype(str).str.strip()

   # numeric
   df["value"] = pd.to_numeric(df["value"], errors="coerce")
   ```

7. **Timestamp parsing to ISO 8601**
   ```python
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
   ```

8. **Unit normalization** (example mapping — edit to your real rules)
   ```python
   UNIT_MAP = {
       "celsius": "C", "°c": "C", "C": "C",
       "kilogram": "kg", "KG": "kg", "kg": "kg",
       "meter": "m", "M": "m", "m": "m",
   }
   df["unit_label"] = df["unit_label"].str.lower().map(UNIT_MAP).fillna(df["unit_label"])
   ```

9. **Drop rows with missing critical values**
   ```python
   df = df.dropna(subset=["artifact_id","sdc_kind","unit_label","value","timestamp"])
   ```

10. **Sort for readability (optional)**
    ```python
    df = df.sort_values(["artifact_id", "timestamp"]).reset_index(drop=True)
    ```

11. **Write output**
    ```python
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(df)} rows.")
    ```