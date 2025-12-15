import logging
from typing import Any

import pandas as pd

from util.logger_config import config

logger = logging.getLogger(__name__)
from pathlib import Path

config(logger)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=0, skipinitialspace=True)


def write_df_to_csv(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False)


def read_reference_entries(settings: dict) -> list[Any]:
    """Load reference entries using paths and limits from settings.

    Expected settings keys:
    - bfo_cco_terms: Path to the CSV with reference terms
    - ref_max_chars: int, max characters for full reference context (used for logging)
    - reference_mode: str, mode description for logging
    """
    reference_entries_file = settings["bfo_cco_terms"]

    ref_entries = []
    if reference_entries_file.exists():
        try:
            ref_df = read_csv(reference_entries_file)
            # Build a list of dicts and a precomputed full context string
            ref_entries = []
            for _, row in ref_df.iterrows():
                if pd.notna(row.get("label")) and pd.notna(row.get("definition")):
                    entry = {
                        "label": str(row["label"]).strip(),
                        "definition": str(row["definition"]).strip(),
                        "type": str(row.get("type", "unknown")).strip().lower() if row.get("type",
                                                                                           None) is not None else "unknown",
                    }
                    ref_entries.append(entry)
            # Build quick lookup map for types by label (labels are expected unique enough for our purposes)
            ref_type_by_label = {e["label"]: e.get("type", "unknown") for e in ref_entries}
            glossary_lines = [
                f"- {e['label']}: {e['definition']}" for e in ref_entries
            ]
            full_reference_context = "\n".join(glossary_lines)
            logger.info(
                "Loaded %d reference terms from %s",
                len(ref_entries),
                reference_entries_file
            )
        except Exception as e:
            logger.warning("Failed to load reference terms from %s: %s", reference_entries_file, e)
    return ref_entries


def load_entries(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize expected columns
    for col in ("iri", "label", "definition", "type"):
        if col not in df.columns:
            df[col] = ""

    # Clean
    df["label"] = df["label"].astype(str).str.strip()
    df["definition"] = df["definition"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip().str.lower().replace({"": "unknown"})
    df["iri"] = df["iri"].astype(str).str.strip()

    # Filter out empty labels/definitions
    df = df[(df["label"] != "") & (df["definition"] != "")]
    return df
