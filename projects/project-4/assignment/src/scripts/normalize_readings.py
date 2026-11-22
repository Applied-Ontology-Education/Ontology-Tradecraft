#!/usr/bin/env python3
"""ETL: combine every raw sensor file in src/data/ into src/data/readings_normalized.csv.

Canonical columns: artifact_id, sdc_kind, unit_label, value, timestamp (ISO-8601 UTC, 'Z').

Fixes over the starter script:
  * Sensor CSVs record "Time (Local)": parsed as America/New_York (site: Buffalo) and
    converted to UTC, not mislabeled as UTC. Sensor A 19:03 local == Sensor B 23:03Z.
  * Artifact IDs normalized ("Chiller 3", "Chiller-3 " -> "Chiller-3") so one device is one artifact.
  * Every *.csv / *.json in src/data/ is read, so a new file (e.g. sensor_C.csv) is picked up automatically.
  * Suspect readings are reported (not silently changed), e.g. chiller temperatures below 32 F.
"""
from __future__ import annotations
import json
import re
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dateutil import parser as dateparser

SRC_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = SRC_DIR / "data"
OUT = DATA_DIR / "readings_normalized.csv"
CANON = ["artifact_id", "sdc_kind", "unit_label", "value", "timestamp"]
SITE_TZ = ZoneInfo("America/New_York")

KIND = {"temp": "temperature", "temperature": "temperature", "pressure": "pressure",
        "voltage": "voltage", "resistance": "resistance"}
UNIT = {"f": "F", "°f": "F", "fahrenheit": "F", "c": "C", "°c": "C", "celsius": "C",
        "psi": "psi", "kpa": "kPa", "kilopascal": "kPa", "kilopascals": "kPa",
        "v": "V", "volt": "V", "volts": "V", "ohm": "ohm", "ohms": "ohm", "ω": "ohm"}


def norm_artifact(x) -> str | None:
    s = str(x).strip() if x is not None else ""
    return re.sub(r"[\s_]+", "-", s) or None


def to_float(x) -> float | None:
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return None


def to_utc(x, local: bool) -> str | None:
    """Parse a timestamp; naive local times are read in the site's time zone."""
    s = str(x).strip() if x is not None else ""
    if not s or s.lower() in {"nan", "none"}:
        return None
    try:
        t = dateparser.parse(s)
    except (ValueError, OverflowError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=SITE_TZ if local else dt.timezone.utc)
    return t.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    local = "Time (Local)" in df.columns
    df = df.rename(columns={"Device Name": "artifact_id", "Reading Type": "sdc_kind",
                            "Reading Value": "value", "Units": "unit_label", "Time (Local)": "timestamp"})
    df["timestamp"] = df["timestamp"].apply(lambda v: to_utc(v, local))
    return df[CANON]


def load_json(path: Path) -> pd.DataFrame:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = [{"artifact_id": e.get("entity_id"), "sdc_kind": d.get("kind"), "unit_label": d.get("unit"),
             "value": d.get("value"), "timestamp": to_utc(d.get("time"), local=False)}
            for e in obj.get("readings", []) for d in (e.get("data") or [])]
    return pd.DataFrame(rows, columns=CANON)


def main() -> None:
    frames = []
    for path in sorted(DATA_DIR.iterdir()):
        if path == OUT or path.suffix.lower() not in {".csv", ".json"}:
            continue
        df = load_csv(path) if path.suffix.lower() == ".csv" else load_json(path)
        print(f"[etl] {path.name}: {len(df)} rows")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    df["artifact_id"] = df["artifact_id"].apply(norm_artifact)
    df["sdc_kind"] = df["sdc_kind"].astype(str).str.strip().str.lower().map(KIND)
    df["unit_label"] = df["unit_label"].astype(str).str.strip().str.lower().map(UNIT)
    df["value"] = df["value"].apply(to_float)

    dropped = df[df[CANON].isna().any(axis=1)]
    for _, r in dropped.iterrows():
        print(f"[etl] dropped incomplete row: {r.to_dict()}")
    df = df.dropna(subset=CANON)

    suspect = df[(df.sdc_kind == "temperature") & (df.unit_label == "F") & (df.value < 32)
                 & df.artifact_id.str.startswith("Chiller")]
    for _, r in suspect.iterrows():
        print(f"[etl] SUSPECT (chiller below freezing in F; Celsius mislabeled?): {r.to_dict()}")

    df = df.sort_values(["artifact_id", "timestamp", "sdc_kind", "unit_label"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"[etl] wrote {OUT.name}: {len(df)} rows")


if __name__ == "__main__":
    main()
