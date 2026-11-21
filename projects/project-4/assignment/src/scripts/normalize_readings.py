#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import datetime as _dt
import json

import pandas as pd
from dateutil import parser as dateparser


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
DATA_DIR = SRC_DIR / "data"

IN_A = DATA_DIR / "sensor_A.csv"
IN_B = DATA_DIR / "sensor_B.json"
IN_C = DATA_DIR / "sensor_C.csv"

OUT = DATA_DIR / "readings_normalized.csv"


# ============================================================
# CANONICAL OUTPUT COLUMNS
# ============================================================

CANON = [
    "artifact_id",
    "sdc_kind",
    "unit_label",
    "value",
    "timestamp",
]


# ============================================================
# HELPERS
# ============================================================

def _to_iso_utc(x: Any) -> str | None:
    """
    Parse a timestamp to ISO-8601 UTC format.

    If the timestamp has no timezone information,
    assume UTC.
    """
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

        return (
            dt.astimezone(_dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    except Exception:
        return None


def _to_float(x: Any) -> float | None:
    """
    Convert a reading to a float.

    Invalid or blank values return None.
    """
    if x is None:
        return None

    s = str(x).strip()

    if s == "" or s.lower() in {"nan", "none"}:
        return None

    try:
        return float(s)

    except Exception:
        return None


def _norm_artifact_id(x: Any) -> str | None:
    """
    Normalize artifact identifiers.

    Sensor A uses 'Chiller 3' while Sensor B uses
    'Chiller-3'. Normalize both to 'Chiller-3'.
    """
    if x is None:
        return None

    s = str(x).strip()

    if not s:
        return None

    aliases = {
        "chiller 3": "Chiller-3",
        "chiller-3": "Chiller-3",
    }

    return aliases.get(s.lower(), s)


def _norm_kind(k: Any) -> str | None:
    """
    Normalize measurement type labels.
    """
    if k is None:
        return None

    s = str(k).strip()

    if not s:
        return None

    low = s.lower()

    if low in {"temp", "temperature"}:
        return "temperature"

    if low == "pressure":
        return "pressure"

    if low == "voltage":
        return "voltage"

    if low == "resistance":
        return "resistance"

    return s


def _norm_unit(u: Any) -> str | None:
    """
    Normalize unit spelling and abbreviations.

    Numeric unit conversions are performed later.
    """
    if u is None:
        return None

    s = str(u).strip()

    if not s:
        return None

    low = s.lower()

    # Temperature
    if low in {"celsius", "°c", "c"}:
        return "C"

    if low in {"fahrenheit", "°f", "f"}:
        return "F"

    # Pressure
    if low == "psi":
        return "psi"

    if low in {
        "kpa",
        "kilopascal",
        "kilopascals",
    }:
        return "kPa"

    if low in {
        "pa",
        "pascal",
        "pascals",
    }:
        return "Pa"

    # Electrical
    if low in {
        "v",
        "volt",
        "volts",
    }:
        return "V"

    if low in {
        "ohm",
        "ohms",
        "Ω",
        "ω",
    }:
        return "ohm"

    return s


# ============================================================
# SENSOR A / SENSOR C CSV LOADER
# ============================================================

def load_sensor_a(path: Path) -> pd.DataFrame:
    """
    Load CSV files using the Sensor A-style schema:

    Device Name
    Reading Type
    Reading Value
    Units
    Time (Local)
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    # Remove whitespace from headers.
    df.columns = [
        column.strip()
        for column in df.columns
    ]

    rename_map = {
        "Device Name": "artifact_id",
        "Reading Type": "sdc_kind",
        "Reading Value": "value",
        "Units": "unit_label",
        "Time (Local)": "timestamp",
    }

    df = df.rename(columns=rename_map)

    # Make sure every canonical column exists.
    for column in CANON:
        if column not in df.columns:
            df[column] = None

    return df[CANON].copy()


# ============================================================
# SENSOR B JSON LOADER
# ============================================================

def load_sensor_b(path: Path) -> pd.DataFrame:
    """
    Flatten the nested Sensor B JSON structure into
    canonical Project 4 columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    obj = json.loads(
        path.read_text(encoding="utf-8")
    )

    if isinstance(obj, dict):
        readings = obj.get("readings", [])
    else:
        readings = []

    rows: List[Dict[str, Any]] = []

    for entry in readings:
        entity = entry.get("entity_id")

        for reading in entry.get("data", []) or []:
            rows.append(
                {
                    "artifact_id": entity,
                    "sdc_kind": reading.get("kind"),
                    "unit_label": reading.get("unit"),
                    "value": reading.get("value"),
                    "timestamp": reading.get("time"),
                }
            )

    return pd.DataFrame(
        rows,
        columns=CANON,
    )


# ============================================================
# NORMALIZATION AND CLEANING
# ============================================================

def normalize_and_clean(
    df: pd.DataFrame
) -> pd.DataFrame:

    # --------------------------------------------------------
    # 1. Trim whitespace.
    # --------------------------------------------------------

    for column in [
        "artifact_id",
        "sdc_kind",
        "unit_label",
        "timestamp",
    ]:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # 2. Normalize artifact identifiers.
    # --------------------------------------------------------

    df["artifact_id"] = (
        df["artifact_id"]
        .apply(_norm_artifact_id)
    )

    # --------------------------------------------------------
    # 3. Normalize measurement-kind labels.
    # --------------------------------------------------------

    df["sdc_kind"] = (
        df["sdc_kind"]
        .apply(_norm_kind)
    )

    # --------------------------------------------------------
    # 4. Normalize unit labels.
    # --------------------------------------------------------

    df["unit_label"] = (
        df["unit_label"]
        .apply(_norm_unit)
    )

    # --------------------------------------------------------
    # 5. Convert values to numbers.
    # --------------------------------------------------------

    df["value"] = (
        df["value"]
        .apply(_to_float)
    )

    # --------------------------------------------------------
    # 6. Normalize timestamps.
    # --------------------------------------------------------

    df["timestamp"] = (
        df["timestamp"]
        .apply(_to_iso_utc)
    )

    # ========================================================
    # 7. CONVERT TEMPERATURE TO CELSIUS
    # ========================================================

    fahrenheit_mask = (
        (df["sdc_kind"] == "temperature")
        & (df["unit_label"] == "F")
        & df["value"].notna()
    )

    df.loc[
        fahrenheit_mask,
        "value",
    ] = (
        (
            df.loc[
                fahrenheit_mask,
                "value",
            ]
            - 32
        )
        * 5
        / 9
    )

    df.loc[
        fahrenheit_mask,
        "unit_label",
    ] = "C"

    # ========================================================
    # 8. CONVERT PRESSURE TO PASCALS
    # ========================================================
    #
    # Canonical pressure unit:
    #
    #     Pa
    #
    # 1 psi = 6894.757293168 Pa
    # 1 kPa = 1000 Pa
    # ========================================================

    # Convert psi to Pa.
    psi_mask = (
        (df["sdc_kind"] == "pressure")
        & (df["unit_label"] == "psi")
        & df["value"].notna()
    )

    df.loc[
        psi_mask,
        "value",
    ] = (
        df.loc[
            psi_mask,
            "value",
        ]
        * 6894.757293168
    )

    df.loc[
        psi_mask,
        "unit_label",
    ] = "Pa"

    # Convert kPa to Pa.
    kpa_mask = (
        (df["sdc_kind"] == "pressure")
        & (df["unit_label"] == "kPa")
        & df["value"].notna()
    )

    df.loc[
        kpa_mask,
        "value",
    ] = (
        df.loc[
            kpa_mask,
            "value",
        ]
        * 1000
    )

    df.loc[
        kpa_mask,
        "unit_label",
    ] = "Pa"

    # --------------------------------------------------------
    # 9. Round converted values.
    # --------------------------------------------------------

    df["value"] = (
        df["value"]
        .round(6)
    )

    # --------------------------------------------------------
    # 10. Diagnostics.
    # --------------------------------------------------------

    total = len(df)

    missing_counts = {
        "artifact_id": int(
            df["artifact_id"].isna().sum()
            + (df["artifact_id"] == "").sum()
        ),
        "sdc_kind": int(
            df["sdc_kind"].isna().sum()
            + (df["sdc_kind"] == "").sum()
        ),
        "unit_label": int(
            df["unit_label"].isna().sum()
            + (df["unit_label"] == "").sum()
        ),
        "value": int(
            df["value"].isna().sum()
        ),
        "timestamp": int(
            df["timestamp"].isna().sum()
        ),
    }

    print(
        "[diagnostics] total rows:",
        total,
        "| missing:",
        missing_counts,
    )

    # --------------------------------------------------------
    # 11. Replace blank strings with NA.
    # --------------------------------------------------------

    df = df.replace(
        {"": pd.NA}
    )

    # --------------------------------------------------------
    # 12. Drop invalid rows.
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "artifact_id",
            "sdc_kind",
            "unit_label",
            "value",
            "timestamp",
        ]
    )

    # ========================================================
    # 13. VERIFY UNIT CONSISTENCY
    # ========================================================

    expected_units = {
        "temperature": "C",
        "pressure": "Pa",
        "voltage": "V",
        "resistance": "ohm",
    }

    for kind, expected_unit in expected_units.items():
        observed_units = set(
            df.loc[
                df["sdc_kind"] == kind,
                "unit_label",
            ]
            .dropna()
            .unique()
        )

        if observed_units and observed_units != {expected_unit}:
            raise ValueError(
                f"Inconsistent units for {kind}. "
                f"Expected only '{expected_unit}', "
                f"but found {sorted(observed_units)}"
            )

    # --------------------------------------------------------
    # 14. Sort deterministically.
    # --------------------------------------------------------

    df = (
        df.sort_values(
            [
                "artifact_id",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 15. Put columns in canonical order.
    # --------------------------------------------------------

    df = df[CANON]

    return df


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("[paths] A:", IN_A)
    print("[paths] B:", IN_B)
    print("[paths] C:", IN_C)

    # --------------------------------------------------------
    # Load all three sources.
    # --------------------------------------------------------

    df_a = load_sensor_a(IN_A)
    df_b = load_sensor_b(IN_B)
    df_c = load_sensor_a(IN_C)

    print(
        f"[normalize_readings] "
        f"Input A rows: {len(df_a)}"
    )

    print(
        f"[normalize_readings] "
        f"Input B rows: {len(df_b)}"
    )

    print(
        f"[normalize_readings] "
        f"Input C rows: {len(df_c)}"
    )

    # --------------------------------------------------------
    # Combine all sources.
    # --------------------------------------------------------

    combined = pd.concat(
        [
            df_a,
            df_b,
            df_c,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Normalize and clean.
    # --------------------------------------------------------

    cleaned = normalize_and_clean(
        combined
    )

    # --------------------------------------------------------
    # Save output.
    # --------------------------------------------------------

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned.to_csv(
        OUT,
        index=False,
    )

    print(
        f"[normalize_readings] "
        f"Output rows : {len(cleaned)}"
    )

    print(
        f"[normalize_readings] "
        f"Wrote       : {OUT}"
    )

    # --------------------------------------------------------
    # Show final canonical units.
    # --------------------------------------------------------

    print(
        "\n[normalize_readings] "
        "Canonical units:"
    )

    for kind in sorted(
        cleaned["sdc_kind"].unique()
    ):
        units = sorted(
            cleaned.loc[
                cleaned["sdc_kind"] == kind,
                "unit_label",
            ].unique()
        )

        print(
            f"  {kind}: {', '.join(units)}"
        )


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()