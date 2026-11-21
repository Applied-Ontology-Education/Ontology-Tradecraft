from pathlib import Path
import argparse
import os
import time

import pandas as pd
from openai import OpenAI


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "definitions.csv"
OUTPUT_FILE = DATA_DIR / "definitions_enriched.csv"


# ============================================================
# XAI SETTINGS
# ============================================================

XAI_BASE_URL = "https://api.x.ai/v1"

MODEL = os.getenv(
    "XAI_MODEL",
    "grok-4.7",
)

MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 5


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are assisting with ontology engineering using the
Common Core Ontologies (CCO).

Your task is to normalize and enrich textual ontology definitions.

Follow these rules:

1. Preserve the semantic meaning of the original definition while also improving upon it.
2. Do not invent new facts, relations, restrictions, or axioms.
3. Do not generate OWL axioms.
4. Rewrite the definition into a clear canonical form.
5. Prefer a genus-differentia form such as:
   "X is a Y that ..."
   when appropriate.
6. Remove unnecessary ambiguity without changing meaning.
7. Expand abbreviations only when their meaning is clear.
8. Preserve technical distinctions.
9. Use terminology appropriate for CCO and formal ontology.
10. Keep the result concise.
11. Return exactly one normalized definition.
12. Return no commentary, markdown, notes, or explanations.

Return ONLY the normalized definition.
""".strip()


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N definitions.",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start over instead of resuming existing output.",
    )

    return parser.parse_args()


# ============================================================
# LOAD INPUT
# ============================================================

def load_definitions():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False,
    )

    required = [
        "IRI",
        "label",
        "definition",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    for column in required:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    df = df[
        df["definition"] != ""
    ].copy()

    df = df.drop_duplicates(
        subset=[
            "IRI",
            "definition",
        ]
    )

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# XAI CLIENT
# ============================================================

def create_client():

    api_key = os.getenv(
        "XAI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "XAI_API_KEY is not set.\n\n"
            "Run in PowerShell:\n"
            '$env:XAI_API_KEY="your-xai-key"'
        )

    return OpenAI(
        api_key=api_key,
        base_url=XAI_BASE_URL,
    )


# ============================================================
# NORMALIZE ONE DEFINITION
# ============================================================

def normalize_definition(
    client,
    iri,
    label,
    definition,
):

    user_prompt = f"""
IRI:
{iri}

Class label:
{label}

Original definition:
{definition}

Rewrite this as a standardized CCO-style textual definition.

Preserve the original semantic meaning.

Do not introduce new ontological claims.

Return only the normalized definition.
""".strip()

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = client.responses.create(
                model=MODEL,
                input=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                store=False,
            )

            result = (
                response.output_text
                .strip()
            )

            if not result:
                raise ValueError(
                    "xAI returned an empty response."
                )

            return result

        except Exception as error:

            last_error = error

            status = getattr(
                error,
                "status_code",
                None,
            )

            print()
            print(
                f"    xAI API attempt "
                f"{attempt}/{MAX_RETRIES} failed:"
            )
            print(
                f"    {error}"
            )

            # Authentication and permission errors
            # should not be retried.
            if status in (
                401,
                402,
                403,
            ):
                raise

            if attempt < MAX_RETRIES:

                print(
                    f"    Retrying in "
                    f"{RETRY_DELAY_SECONDS} seconds..."
                )

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

    raise RuntimeError(
        f"Unable to process definition: {label}"
    ) from last_error


# ============================================================
# EXISTING OUTPUT
# ============================================================

OUTPUT_COLUMNS = [
    "IRI",
    "label",
    "definition",
    "definition_enriched",
]


def load_existing_output(
    fresh,
):

    if fresh or not OUTPUT_FILE.exists():

        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    df = pd.read_csv(
        OUTPUT_FILE,
        dtype=str,
        keep_default_na=False,
    )

    return df[
        OUTPUT_COLUMNS
    ].copy()


def save_output(
    rows,
):

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    print("=" * 70)
    print(
        "PROJECT 5 - XAI GROK DEFINITION PREPROCESSING"
    )
    print("=" * 70)

    print()
    print(
        f"Project root : {PROJECT_ROOT}"
    )
    print(
        f"Input file   : {INPUT_FILE}"
    )
    print(
        f"Output file  : {OUTPUT_FILE}"
    )
    print(
        f"xAI endpoint : {XAI_BASE_URL}"
    )
    print(
        f"xAI model    : {MODEL}"
    )
    print()

    source_df = load_definitions()

    print(
        f"Definitions available: "
        f"{len(source_df)}"
    )

    if args.limit is not None:

        if args.limit <= 0:
            raise ValueError(
                "--limit must be greater than zero."
            )

        source_df = (
            source_df
            .head(args.limit)
            .copy()
        )

        print(
            f"TEST MODE: processing only "
            f"{len(source_df)} definition(s)."
        )

    print()

    existing_df = load_existing_output(
        args.fresh
    )

    completed = set(
        zip(
            existing_df["IRI"],
            existing_df["definition"],
        )
    )

    output_rows = (
        existing_df
        .to_dict(
            orient="records"
        )
    )

    client = create_client()

    total = len(source_df)

    for index, row in source_df.iterrows():

        iri = row["IRI"]
        label = row["label"]
        definition = row["definition"]

        key = (
            iri,
            definition,
        )

        if key in completed:

            print(
                f"[{index + 1}/{total}] "
                f"Already complete: {label}"
            )

            continue

        print(
            f"[{index + 1}/{total}] "
            f"Processing: {label}"
        )

        enriched = normalize_definition(
            client=client,
            iri=iri,
            label=label,
            definition=definition,
        )

        output_rows.append(
            {
                "IRI": iri,
                "label": label,
                "definition": definition,
                "definition_enriched": enriched,
            }
        )

        completed.add(
            key
        )

        # Save after each successful definition.
        save_output(
            output_rows
        )

    save_output(
        output_rows
    )

    print()
    print("=" * 70)
    print(
        "PREPROCESSING COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        f"Definitions in output: "
        f"{len(output_rows)}"
    )
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()