#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import json
import pandas as pd
from dateutil import parser as dateparser
import datetime as _dt

# ---------- Paths resolved relative to this script ----------
SCRIPT_DIR = Path(__file__).resolve().parent          # .../src/scripts
SRC_DIR    = SCRIPT_DIR.parent                         # .../src
DATA_DIR   = SRC_DIR / "data"                          # .../src/data
IN_A       = DATA_DIR / "sensor_A.csv"
IN_B       = DATA_DIR / "sensor_B.json"
IN_C       = DATA_DIR / "sensor_C.csv"
OUT        = DATA_DIR / "readings_normalized.csv"

CANON = ["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"]

# ---------- Helpers ----------
def _to_iso_utc(x: Any) -> str | None:
    """Parse any timestamp to ISO-8601 in UTC with 'Z'. If naive, assume UTC."""
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    try:
        dt = dateparser.parse(s)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return dt.astimezone(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _norm_kind(k: Any) -> str | None:
    """Normalize reading type/kind labels (string only, no semantic conversion)."""
    if k is None:
        return None
    s = str(k).strip()
    if not s:
        return None
    low = s.lower()
    if low in {"temp", "temperature"}:
        return "temperature"
    if low in {"pressure"}:
        return "pressure"
    if low in {"voltage"}:
        return "voltage"
    if low in {"resistance"}:
        return "resistance"
    # fallback: return trimmed as-is
    return s


def _norm_unit(u: Any) -> str | None:
    """Normalize unit spelling/abbrev only (no numeric conversions)."""
    if u is None:
        return None
    s = str(u).strip()
    if not s:
        return None
    low = s.lower()
    # temperature
    if low in {"celsius", "°c", "c"}:
        return "C"
    if low in {"fahrenheit", "°f", "f"}:
        return "F"
    # pressure
    if low == "psi":
        return "psi"
    if low in {"kpa", "kilopascal", "kilopascals"}:
        return "kPa"
    # electrical
    if low in {"v", "volt", "volts"}:
        return "V"
    if low in {"ohm", "ohms", "Ω", "ω"}:
        return "ohm"
    # pass through otherwise
    return s


# ---------- Loaders for each source ----------
def load_sensor_a(path: Path) -> pd.DataFrame:
    """
    CSV columns (exact): Device Name, Reading Type, Reading Value, Units, Time (Local)
    Maps to canonical columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Rename using exact headers shown
    rename_map = {
        "Device Name": "artifact_id",
        "Reading Type": "sdc_kind",
        "Reading Value": "value",
        "Units": "unit_label",
        "Time (Local)": "timestamp",
    }
    # Be tolerant to stray whitespace/casing in headers
    fixed_cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=fixed_cols)
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure canonical columns exist
    for c in CANON:
        if c not in df.columns:
            df[c] = None

    df = df[CANON].copy()
    return df


def load_sensor_b(path: Path) -> pd.DataFrame:
    """
    JSON structure per the example in the prompt.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    readings = obj.get("readings", []) if isinstance(obj, dict) else []

    rows: List[Dict[str, Any]] = []
    for entry in readings:
        entity = entry.get("entity_id")
        for d in entry.get("data", []) or []:
            rows.append({
                "artifact_id": entity,
                "sdc_kind": d.get("kind"),
                "unit_label": d.get("unit"),
                "value": d.get("value"),
                "timestamp": d.get("time"),
            })

    df = pd.DataFrame(rows, columns=CANON)
    return df


# ---------- Normalize + Diagnostics ----------
def normalize_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Trim strings
    for col in ["artifact_id", "sdc_kind", "unit_label", "timestamp"]:
        df[col] = df[col].astype(str).str.strip()

    # Normalize labels
    df["sdc_kind"] = df["sdc_kind"].apply(_norm_kind)
    df["unit_label"] = df["unit_label"].apply(_norm_unit)

    # Coerce types
    df["value"] = df["value"].apply(_to_float)
    df["timestamp"] = df["timestamp"].apply(_to_iso_utc)

    # Diagnostics
    total = len(df)
    missing_counts = {
        "artifact_id": int(df["artifact_id"].isna().sum() + (df["artifact_id"] == "").sum()),
        "sdc_kind":    int(df["sdc_kind"].isna().sum() + (df["sdc_kind"] == "").sum()),
        "unit_label":  int(df["unit_label"].isna().sum() + (df["unit_label"] == "").sum()),
        "value":       int(df["value"].isna().sum()),
        "timestamp":   int(df["timestamp"].isna().sum()),
    }
    print("[diagnostics] total rows:", total, "| missing:", missing_counts)

    # Drop rows with any missing criticals
    df = df.replace({"": pd.NA})
    df = df.dropna(subset=["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"])

    # Sort deterministically
    df = df.sort_values(["artifact_id", "timestamp"]).reset_index(drop=True)

    # Reorder columns
    df = df[CANON]
    return df


def main():
    print("[paths] A:", IN_A)
    print("[paths] B:", IN_B)
    print("[paths] C:", IN_C)
    df_a = load_sensor_a(IN_A)
    df_b = load_sensor_b(IN_B)
    df_c = load_sensor_a(IN_C)

    print(f"[normalize_readings] Input A rows: {len(df_a)}")
    print(f"[normalize_readings] Input B rows: {len(df_b)}")
    print(f"[normalize_readings] Input C rows: {len(df_c)}")

    combined = pd.concat([df_a, df_b, df_c], ignore_index=True)
    cleaned = normalize_and_clean(combined)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUT, index=False)

    print(f"[normalize_readings] Output rows : {len(cleaned)}")
    print(f"[normalize_readings] Wrote       : {OUT}")


if __name__ == "__main__":
    main()
