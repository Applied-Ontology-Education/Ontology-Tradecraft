#!/usr/bin/env python3
"""ETL: combine every raw sensor file in src/data/ into src/data/readings_normalized.csv.

Canonical columns (ETL_Guide.md): artifact_id, sdc_kind, unit_label, value, timestamp.

Normalisation decisions (evidence in DECISIONS.md):
- Sources: every *.csv (Sensor-A layout) and *.json (Sensor-B layout) in src/data/, except
  our own output. New datasets such as sensor_C.csv are picked up without code changes, and a
  file that is not there yet is simply skipped.
- Time: the CSV column is "Time (Local)". Readings that sensors A and B share line up at exactly
  UTC-4 (Boiler-07 212 F at 19:03 local = 100 C at 23:03Z), and the site is Buffalo, so local
  time is America/New_York (DST-aware). JSON times already carry 'Z'. All output is UTC 'Z'.
- Artifact ids: trim and replace internal whitespace with '-', so "Chiller 3" and "Chiller-3"
  (the same asset in two sources) become one id.
- Units: one spelling per unit. Pressure is converted to pascal (Pa): 'psi' and 'kPa' are
  rejected by the canonical-unit check, and CCO provides a Pascal Measurement Unit individual
  but no kilopascal. The conversions are exact, so nothing is lost. Temperature keeps its
  source unit (C or F, both CCO individuals).
- Missing or non-numeric values ('not_a_number', blank, null) are dropped, as are exact duplicate
  rows after normalisation.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dateutil import parser as dateparser
from dateutil import tz

SCRIPT_DIR = Path(__file__).resolve().parent          # .../src/scripts
DATA_DIR = SCRIPT_DIR.parent / "data"                  # .../src/data
OUT = DATA_DIR / "readings_normalized.csv"
CANON = ["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"]

CSV_SOURCE_TZ = tz.gettz("America/New_York")

KIND = {"temp": "temperature", "temperature": "temperature", "pressure": "pressure",
        "voltage": "voltage", "resistance": "resistance"}
# spelling -> (canonical unit label, factor to canonical unit)
UNIT = {
    "c": ("C", 1.0), "°c": ("C", 1.0), "celsius": ("C", 1.0),
    "f": ("F", 1.0), "°f": ("F", 1.0), "fahrenheit": ("F", 1.0),
    "pa": ("Pa", 1.0), "kpa": ("Pa", 1000.0), "kilopascal": ("Pa", 1000.0),
    "psi": ("Pa", 6894.757293168), "lbf/in2": ("Pa", 6894.757293168),
    "v": ("V", 1.0), "volt": ("V", 1.0), "volts": ("V", 1.0),
    "ohm": ("ohm", 1.0), "ohms": ("ohm", 1.0), "ω": ("ohm", 1.0),
}


def load_sensor_a(path: Path) -> pd.DataFrame:
    """Sensor-A CSV layout: Device Name, Reading Type, Reading Value, Units, Time (Local)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.rename(columns={c: c.strip() for c in df.columns}).rename(columns={
        "Device Name": "artifact_id", "Reading Type": "sdc_kind", "Reading Value": "value",
        "Units": "unit_label", "Time (Local)": "timestamp"})
    df["source_tz"] = "local"
    return df.reindex(columns=CANON + ["source_tz"])


def load_sensor_b(path: Path) -> pd.DataFrame:
    """Sensor-B JSON layout: {readings: [{entity_id, data: [{kind, value, unit, time}]}]}."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = [{"artifact_id": e.get("entity_id"), "sdc_kind": d.get("kind"), "unit_label": d.get("unit"),
             "value": d.get("value"), "timestamp": d.get("time"), "source_tz": "utc"}
            for e in obj.get("readings", []) for d in (e.get("data") or [])]
    return pd.DataFrame(rows, columns=CANON + ["source_tz"])


def to_iso_utc(value: Any, source_tz: str) -> str | None:
    try:
        dt = dateparser.parse(str(value).strip())
    except (ValueError, OverflowError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=CSV_SOURCE_TZ if source_tz == "local" else _dt.timezone.utc)
    return dt.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_float(value: Any) -> float | None:
    try:
        f = float(str(value).strip())
    except ValueError:
        return None
    return None if f != f else f          # NaN -> None


def normalize_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["artifact_id"] = df["artifact_id"].astype(str).str.strip().map(lambda s: re.sub(r"\s+", "-", s))
    df["sdc_kind"] = df["sdc_kind"].astype(str).str.strip().str.lower().map(KIND)
    unit = df["unit_label"].astype(str).str.strip().str.lower().map(UNIT)
    df["unit_label"] = unit.map(lambda u: u[0] if isinstance(u, tuple) else None)
    factor = unit.map(lambda u: u[1] if isinstance(u, tuple) else None)
    df["value"] = [None if v is None or f is None else round(v * f, 6)
                   for v, f in zip(df["value"].map(to_float), factor)]
    df["timestamp"] = [to_iso_utc(t, s) for t, s in zip(df["timestamp"], df["source_tz"])]

    before = len(df)
    df = df.replace({"": None, "None": None, "nan": None}).dropna(subset=CANON)
    df = df[CANON].drop_duplicates()
    print(f"[normalize_readings] dropped {before - len(df)} incomplete or duplicate rows")
    return df.sort_values(["artifact_id", "timestamp", "sdc_kind"]).reset_index(drop=True)


def main() -> None:
    frames = []
    for path in sorted(DATA_DIR.iterdir()):
        if path == OUT or path.suffix.lower() not in {".csv", ".json"}:
            continue
        df = load_sensor_a(path) if path.suffix.lower() == ".csv" else load_sensor_b(path)
        print(f"[normalize_readings] {path.name}: {len(df)} rows")
        frames.append(df)
    cleaned = normalize_and_clean(pd.concat(frames, ignore_index=True))
    cleaned.to_csv(OUT, index=False)
    print(f"[normalize_readings] wrote {OUT} ({len(cleaned)} rows)")


if __name__ == "__main__":
    main()
