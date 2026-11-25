"""
preprocess_definitions_llm.py

Normalize and enrich data/definitions.csv with Gemini and write
data/definitions_enriched.csv.

Canonical form:  "<Label> is a/an <Parent Label> that ..."
where <Parent Label> is the class's asserted rdfs:subClassOf parent.

Gemini is asked to: put each definition in canonical form, remove ambiguity,
expand abbreviations, fix typos, align wording with CCO style, and name
existing ontology classes where the definition refers to them.

Every Gemini output is validated. If a definition does not start with the
required "<Label> is a/an <Parent Label>" genus, the class is retried once;
if it still fails, a deterministic canonical rewrite of the original
definition is used and the fallback is recorded in normalization_note.

Responses are cached in generated/gemini_definition_cache.json, so reruns
make no API calls unless the inputs or the prompt change.
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from rdflib import Graph, OWL, RDF, RDFS, URIRef

PROJECT_DIR = Path(__file__).resolve().parents[1]

SOURCE_PATH = PROJECT_DIR / "src" / "cco-module.ttl"
INPUT_PATH = PROJECT_DIR / "data" / "definitions.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "definitions_enriched.csv"
CACHE_PATH = PROJECT_DIR / "generated" / "gemini_definition_cache.json"

GEMINI_MODEL = "gemini-3.5-flash-lite"
PROMPT_VERSION = "normalize-v1"   # change this to invalidate the cache
BATCH_SIZE = 20
SECONDS_BETWEEN_CALLS = 7          # stays under ~10 requests/minute
MAX_RETRIES = 4

OUTPUT_COLUMNS = [
    "IRI",
    "label",
    "parent_IRI",
    "parent_label",
    "original_definition",
    "definition",
    "normalization_note",
]

# Labels for BFO parents (the module imports BFO, so they are not in the file).
BFO_LABELS = {
    "http://purl.obolibrary.org/obo/BFO_0000019": "Quality",
    "http://purl.obolibrary.org/obo/BFO_0000145": "Relational Quality",
    "http://purl.obolibrary.org/obo/BFO_0000016": "Disposition",
    "http://purl.obolibrary.org/obo/BFO_0000017": "Realizable Entity",
}


# ============================================================
# ENVIRONMENT
# ============================================================

def load_env():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (PROJECT_DIR / ".env", PROJECT_DIR.parents[1] / ".env"):
        if candidate.exists():
            load_dotenv(candidate)


# ============================================================
# INPUTS
# ============================================================

def load_records():
    ontology = Graph()
    ontology.parse(str(SOURCE_PATH), format="turtle")

    def label_of(iri):
        return BFO_LABELS.get(str(iri)) or str(
            ontology.value(URIRef(str(iri)), RDFS.label) or ""
        )

    records = []
    with open(INPUT_PATH, "r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            iri = row["IRI"].strip()
            parents = [
                p for p in ontology.objects(URIRef(iri), RDFS.subClassOf)
                if isinstance(p, URIRef)
            ]
            if len(parents) != 1:
                raise ValueError(f"Expected exactly one named parent for {iri}")
            parent = str(parents[0])
            records.append({
                "IRI": iri,
                "label": row["label"].strip(),
                "parent_IRI": parent,
                "parent_label": label_of(parent),
                "original_definition": row["definition"].strip(),
            })

    class_labels = sorted(
        {str(ontology.value(c, RDFS.label)) for c in ontology.subjects(RDF.type, OWL.Class)
         if ontology.value(c, RDFS.label)} | set(BFO_LABELS.values())
    )
    return records, class_labels


# ============================================================
# PROMPT
# ============================================================

EXAMPLES = """
Illustrative examples (invented classes, not part of this ontology):

Input:  label "Hexagonal", parent "Shape Quality",
        definition "A Shape quality inhering in a bearer in virtue of it having 6 sides."
Output: "Hexagonal is a Shape Quality that inheres in a bearer in virtue of the bearer having exactly six sides and exactly six angles."
Note:   "Changed 'inhering in' to 'that inheres in'; 'it' -> 'the bearer'; '6' -> 'six'."

Input:  label "Teal", parent "Color",
        definition "A Color that is between grn and blue, approx 480-500nm."
Output: "Teal is a Color that inheres in a bearer in virtue of the bearer reflecting or emitting light with a dominant wavelength of approximately 480 to 500 nanometers, between Green and Blue in the visible light spectrum."
Note:   "Expanded abbreviations 'grn', 'approx', 'nm'; capitalized existing class labels Green and Blue; added bearer clause."
"""


def article(word):
    lowered = word.lower()
    # Vowel letters with a consonant sound take "a" ("a One Dimensional ...").
    if lowered.startswith(("one", "uni", "use", "eu")):
        return "a"
    return "an" if lowered[:1] in "aeiou" else "a"


def build_prompt(batch, class_labels):
    items = [
        {
            "iri": r["IRI"],
            "label": r["label"],
            "required_start": f'{r["label"]} is {article(r["parent_label"])} {r["parent_label"]}',
            "original_definition": r["original_definition"],
        }
        for r in batch
    ]
    return f"""
You are normalizing textual definitions in the Common Core Ontologies (CCO)
Quality Ontology, which is built on BFO.

For each item, rewrite the original definition so that:
1. It begins EXACTLY with the item's "required_start" text, followed by
   " that ..." (for example "Square is a Rectangular that ...").
2. It keeps the original meaning. Do not add facts that the original does
   not state or directly imply. Do not drop any condition.
3. Ambiguity is removed: replace vague pronouns ("it", "its") with
   "the bearer"; make comparisons explicit (e.g. "lower than Blue" ->
   "with a shorter wavelength than Blue").
4. Abbreviations are expanded and typos are corrected.
5. Wording follows CCO style: "that inheres in a bearer in virtue of ...",
   and names of existing ontology classes are written in Title Case.
6. It is a single sentence.

Also give a short normalization_note saying what you changed. If the
original definition looks wrong or inconsistent with its parent class,
keep its meaning but begin the note with "FLAG:" and explain.

Existing ontology class labels (use these names when the definition
refers to them):
{json.dumps(class_labels)}
{EXAMPLES}
Return ONLY valid JSON in exactly this structure, with one entry per item:
{{"definitions": [{{"iri": "...", "definition": "...", "normalization_note": "..."}}]}}

ITEMS:
{json.dumps(items, indent=2)}
"""


# ============================================================
# GEMINI
# ============================================================

def extract_json(text):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("Gemini response did not contain a JSON object.")
    result, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    return result


def call_gemini(prompt):
    from google import genai

    if not os.environ.get("GEMINI_API_KEY"):
        raise EnvironmentError("GEMINI_API_KEY is not configured (.env or terminal).")

    delay = SECONDS_BETWEEN_CALLS
    for attempt in range(1, MAX_RETRIES + 1):
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        try:
            response = client.interactions.create(model=GEMINI_MODEL, input=prompt)
            return response.output_text
        except Exception as error:
            if attempt == MAX_RETRIES:
                raise
            print(f"  Gemini call failed ({error.__class__.__name__}); retrying in {delay}s")
            time.sleep(delay)
            delay *= 2
        finally:
            client.close()


def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def normalize_batch(batch, class_labels, cache):
    """Return {iri: {"definition", "normalization_note"}} from Gemini (cached)."""
    key = json.dumps(
        {
            "version": PROMPT_VERSION,
            "model": GEMINI_MODEL,
            "items": [[r["IRI"], r["parent_label"], r["original_definition"]] for r in batch],
        },
        sort_keys=True,
    )
    if key in cache:
        return cache[key], False

    parsed = extract_json(call_gemini(build_prompt(batch, class_labels)))
    results = {}
    for item in parsed.get("definitions", []):
        if isinstance(item, dict) and isinstance(item.get("iri"), str):
            results[item["iri"]] = {
                "definition": str(item.get("definition", "")).strip(),
                "normalization_note": str(item.get("normalization_note", "")).strip(),
            }
    cache[key] = results
    save_cache(cache)
    return results, True


# ============================================================
# VALIDATION
# ============================================================

def is_valid(record, definition):
    required = re.compile(
        rf'^{re.escape(record["label"])} is an? {re.escape(record["parent_label"])}\b'
    )
    return (
        bool(definition)
        and required.match(definition) is not None
        and len(definition) <= 600
        and "\n" not in definition
    )


def canonical_fallback(record):
    original = record["original_definition"].strip()
    body = original[0].lower() + original[1:] if original else ""
    return f'{record["label"]} is {body}'


# ============================================================
# MAIN
# ============================================================

def main():
    print("QUALITY ONTOLOGY - LLM DEFINITION NORMALIZATION")
    print("-" * 50)
    load_env()

    records, class_labels = load_records()
    print(f"Definitions to normalize: {len(records)}")

    cache = load_cache()
    batches = [records[i:i + BATCH_SIZE] for i in range(0, len(records), BATCH_SIZE)]
    results = {}
    api_calls = 0

    for number, batch in enumerate(batches, start=1):
        if api_calls:
            time.sleep(SECONDS_BETWEEN_CALLS)
        batch_results, called = normalize_batch(batch, class_labels, cache)
        api_calls += called
        results.update(batch_results)
        print(f"Batch {number}/{len(batches)}: "
              f"{'Gemini' if called else 'cached'}, {len(batch_results)} definitions")

    # Retry invalid or missing items once, together.
    invalid = [r for r in records if not is_valid(r, results.get(r["IRI"], {}).get("definition", ""))]
    if invalid:
        print(f"Retrying {len(invalid)} definitions that failed validation ...")
        if api_calls:
            time.sleep(SECONDS_BETWEEN_CALLS)
        retry_results, called = normalize_batch(invalid, class_labels, cache)
        api_calls += called
        for iri, value in retry_results.items():
            record = next(r for r in invalid if r["IRI"] == iri)
            if is_valid(record, value["definition"]):
                results[iri] = value

    rows, fallbacks, flags = [], 0, 0
    for record in records:
        value = results.get(record["IRI"], {})
        definition = value.get("definition", "")
        note = value.get("normalization_note", "")
        if not is_valid(record, definition):
            definition = canonical_fallback(record)
            note = "FALLBACK: LLM output failed validation; deterministic canonical rewrite of the original."
            fallbacks += 1
        if note.startswith("FLAG"):
            flags += 1
        rows.append({**record, "definition": definition, "normalization_note": note})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("\nNORMALIZATION SUMMARY")
    print("-" * 50)
    print(f"Definitions written:        {len(rows)}")
    print(f"Normalized by Gemini:       {len(rows) - fallbacks}")
    print(f"Deterministic fallbacks:    {fallbacks}")
    print(f"Flagged for review by LLM:  {flags}")
    print(f"Gemini API calls this run:  {api_calls}")
    print(f"Output: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)