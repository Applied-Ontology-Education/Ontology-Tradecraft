# src/scripts/normalize_readings.py

import json
import datetime
from pathlib import Path

import pandas as pd
from dateutil import parser as dateparser


# --- Paths ---------------------------------------------------------

# This makes the script robust no matter where it is run from
SRC_DIR = Path(__file__).resolve().parents[1]      # .../assignment/src
DATA_DIR = SRC_DIR / "data"

IN_A = DATA_DIR / "sensor_A.csv"
IN_B = DATA_DIR / "sensor_B.json"
OUT = DATA_DIR / "readings_normalized.csv"


# --- Helpers -------------------------------------------------------

def to_iso8601(x):
    """
    Convert whatever timestamp we get into an ISO 8601 UTC string.
    If no timezone is present, treat it as UTC.
    """
    try:
        dt = dateparser.parse(str(x))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def load_sensor_a() -> pd.DataFrame:
    """
    Load src/data/sensor_A.csv and map it to the canonical schema:
    artifact_id, sdc_kind, unit_label, value, timestamp
    """
    df_a = pd.read_csv(
        IN_A,
        dtype=str,
        keep_default_na=False,
        na_values=["", "NA", "NaN"],
    )

    # Map actual headers to canonical names
    df_a = df_a.rename(
        columns={
            "Device Name": "artifact_id",
            "Reading Type": "sdc_kind",
            "Reading Value": "value",
            "Units": "unit_label",
            "Time (Local)": "timestamp",
        }
    )

    canonical_cols = ["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"]
    df_a = df_a[[c for c in canonical_cols if c in df_a.columns]]

    return df_a


def load_sensor_b() -> pd.DataFrame:
    """
    Load src/data/sensor_B.json and flatten it to the canonical schema.

    JSON structure:
    {
      "site": "...",
      "stream_id": "...",
      "readings": [
        {
          "entity_id": "...",
          "data": [
            {"kind": "...", "value": ..., "unit": "...", "time": "..."},
            ...
          ]
        },
        ...
      ]
    }
    """
    raw = json.loads(IN_B.read_text(encoding="utf-8"))

    rows = []
    for reading_block in raw.get("readings", []):
        entity_id = reading_block.get("entity_id")
        for d in reading_block.get("data", []):
            rows.append(
                {
                    "artifact_id": entity_id,
                    "sdc_kind": d.get("kind"),
                    "unit_label": d.get("unit"),
                    "value": d.get("value"),
                    "timestamp": d.get("time"),
                }
            )

    df_b = pd.DataFrame(rows)

    return df_b


def normalize_and_write() -> None:
    # 3 + 4: load both sources
    df_a = load_sensor_a()
    df_b = load_sensor_b()

    # 5: Concatenate A + B
    df = pd.concat([df_a, df_b], ignore_index=True)

    # 6: Trim whitespace + basic normalization
    for col in ["artifact_id", "sdc_kind", "unit_label"]:
        df[col] = df[col].astype(str).str.strip()

    # numeric value
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # ------------------------------------------------------------------
    # Convert pressure units psi and kPa to Pascals (Pa)
    # ------------------------------------------------------------------
    # Any row where unit_label is 'psi'
    mask_psi = df["unit_label"].astype(str).str.lower() == "psi"
    df.loc[mask_psi, "value"] = df.loc[mask_psi, "value"] * 6894.76  # psi → Pa
    df.loc[mask_psi, "unit_label"] = "Pa"

    # Any row where unit_label is 'kPa'
    mask_kpa = df["unit_label"].astype(str).str.lower() == "kpa"
    df.loc[mask_kpa, "value"] = df.loc[mask_kpa, "value"] * 1000.0   # kPa → Pa
    df.loc[mask_kpa, "unit_label"] = "Pa"


    # 7: Timestamp parsing to ISO 8601
    df["timestamp"] = df["timestamp"].apply(to_iso8601)

    # 8: Unit normalization (example mapping from guide)
    UNIT_MAP = {
        "celsius": "C",
        "°c": "C",
        "c": "C",
        "kilogram": "kg",
        "kg": "kg",
        "kilograms": "kg",
        "meter": "m",
        "m": "m",
        "meters": "m",
    }

    df["unit_label"] = (
        df["unit_label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(UNIT_MAP)
        .fillna(df["unit_label"])
    )

    # 9: Drop rows with missing critical values
    df = df.dropna(
        subset=["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"]
    )

    # Ensure canonical column order
    df = df[["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"]]

    # 10: Sort for readability (artifact_id then timestamp)
    df = df.sort_values(["artifact_id", "timestamp"]).reset_index(drop=True)

    # 11: Write output
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(df)} rows.")


def main():
    normalize_and_write()


if __name__ == "__main__":
    main()
