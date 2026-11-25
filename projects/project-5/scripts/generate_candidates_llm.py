"""
generate_candidates_llm.py

Generate OWL 2 EL candidate axioms (named SubClassOf) from the enriched
definitions with Gemini, and write generated/candidate_el.ttl.

Method: LLM-as-verifier over enumerated pairs.
  1. Classes are grouped by hierarchy branch (Shape Quality, Size Quality,
     Disposition to Interact with Electromagnetic Radiation, ...).
  2. For each class X, the script lists every class Y in the same branch
     that is NOT already related to X (not X itself, not an ancestor of X,
     not a descendant of X).
  3. Gemini receives X's definition and the definitions of those Y's and
     returns the Y's whose definitions are entailed by X's definition.
  4. Each returned pair is validated again in Python and written as
     "X rdfs:subClassOf Y".

Asking the model to verify a fixed list of pairs (instead of searching the
whole ontology) prevents it from simply re-proposing existing parents.

Responses are cached in generated/gemini_generated_candidates.json, so
reruns make no API calls unless the inputs, prompt or model change.
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from google import genai
from rdflib import Graph, OWL, RDF, RDFS, URIRef, Literal


# ============================================================
# PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

SOURCE_PATH = PROJECT_DIR / "src" / "cco-module.ttl"
DEFINITIONS_PATH = PROJECT_DIR / "data" / "definitions_enriched.csv"

OUTPUT_PATH = PROJECT_DIR / "generated" / "candidate_el.ttl"
CACHE_PATH = PROJECT_DIR / "generated" / "gemini_generated_candidates.json"

GEMINI_MODEL = "gemini-3.5-flash-lite"
PROMPT_VERSION = "pairwise-verify-v5"   # change this to invalidate the cache

MIN_BRANCH_SIZE = 3            # smaller branches are pooled into "Other qualities"
MAX_CANDIDATES_PER_CLASS = 5   # at most this many superclasses kept per class

SECONDS_BETWEEN_CALLS = 7      # stays under ~10 requests/minute on the free tier
MAX_RETRIES = 4


# ============================================================
# ENVIRONMENT AND GEMINI
# ============================================================

def load_env():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (PROJECT_DIR / ".env", PROJECT_DIR.parents[1] / ".env"):
        if candidate.exists():
            load_dotenv(candidate)


def call_gemini(prompt):
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


def extract_json(text):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("Gemini response did not contain a JSON object.")
    result, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    return result


def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


# ============================================================
# INPUTS
# ============================================================

def load_source_ontology():
    ontology = Graph()
    ontology.parse(str(SOURCE_PATH), format="turtle")

    valid_classes = {
        str(c) for c in ontology.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef)
    }
    asserted = {
        (str(child), str(parent))
        for child, parent in ontology.subject_objects(RDFS.subClassOf)
        if isinstance(child, URIRef) and isinstance(parent, URIRef)
    }
    print(f"Source classes: {len(valid_classes)}")
    print(f"Asserted named subclass relationships: {len(asserted)}")
    return valid_classes, asserted


def load_definitions():
    definitions = {}
    with open(DEFINITIONS_PATH, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"IRI", "label", "definition"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Definitions CSV is missing columns: {sorted(missing)}")
        for row in reader:
            iri = row["IRI"].strip()
            if iri:
                definitions[iri] = {
                    "label": row["label"].strip(),
                    "definition": row["definition"].strip(),
                }
    print(f"Definition records: {len(definitions)}")
    return definitions


def transitive_closure(asserted):
    parents = {}
    for child, parent in asserted:
        parents.setdefault(child, set()).add(parent)

    closure = set()
    for child in parents:
        seen, pending = set(), list(parents[child])
        while pending:
            parent = pending.pop()
            if parent in seen:
                continue
            seen.add(parent)
            closure.add((child, parent))
            pending.extend(parents.get(parent, ()))
    return closure


# ============================================================
# BRANCHES AND CANDIDATE PAIRS
# ============================================================

def build_branches(definitions, asserted):
    """Group classes by their highest ancestor inside this module."""
    parent = {}
    for child, par in asserted:
        parent.setdefault(child, par)

    def branch_root(iri):
        seen = set()
        while parent.get(iri) in definitions and iri not in seen:
            seen.add(iri)
            iri = parent[iri]
        return iri

    groups = {}
    for iri in sorted(definitions):
        groups.setdefault(branch_root(iri), []).append(iri)

    branches, other = [], []
    for root, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) < MIN_BRANCH_SIZE:
            other.extend(members)
        else:
            branches.append((definitions[root]["label"], members))
    if other:
        branches.append(("Other qualities", other))
    return branches


def unrelated_options(focus, members, closure):
    """Branch members that are not focus, an ancestor of focus, or a descendant of focus."""
    return [
        other for other in members
        if other != focus
        and (focus, other) not in closure
        and (other, focus) not in closure
    ]


# ============================================================
# PROMPT
# ============================================================

def build_prompt(focus, options, definitions):
    focus_record = {
        "iri": focus,
        "label": definitions[focus]["label"],
        "definition": definitions[focus]["definition"],
    }
    option_records = [
        {"iri": o, "label": definitions[o]["label"], "definition": definitions[o]["definition"]}
        for o in options
    ]
    return f"""
You are checking candidate subclass relationships in an OWL ontology of
qualities (the CCO Quality Ontology, built on BFO).

FOCUS CLASS:
{json.dumps(focus_record, indent=2)}

For each CANDIDATE SUPERCLASS below, decide whether the focus class's
definition entails the candidate's definition, so that every instance of
the focus class is necessarily also an instance of the candidate.

- Use only the two definitions. Do not rely on typical or common cases.
- If the focus definition does not establish the candidate's condition,
  the answer is no.

CANDIDATE SUPERCLASSES:
{json.dumps(option_records, indent=2)}

Return ONLY valid JSON, no Markdown fences, listing only the candidates
whose definitions ARE entailed (an empty list if none):
{{
  "entailed": [
    {{
      "iri": "exact candidate IRI",
      "rationale": "One sentence: which part of the focus definition entails the candidate definition."
    }}
  ]
}}
"""


# ============================================================
# REQUEST AND VALIDATE
# ============================================================

def request_for_focus(focus, options, definitions, cache):
    key = json.dumps(
        {
            "version": PROMPT_VERSION,
            "model": GEMINI_MODEL,
            "focus": [focus, definitions[focus]["definition"]],
            "options": [[o, definitions[o]["definition"]] for o in options],
        },
        sort_keys=True,
    )
    if key in cache:
        return cache[key], False

    parsed = extract_json(call_gemini(build_prompt(focus, options, definitions)))
    entailed = parsed.get("entailed", []) if isinstance(parsed, dict) else []
    if not isinstance(entailed, list):
        entailed = []
    cache[key] = entailed
    save_cache(cache)
    return entailed, True


def validate(focus, raw, options, closure):
    """Keep only well-formed answers that name one of the offered options."""
    allowed = set(options)
    kept, seen = [], set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not isinstance(item.get("iri"), str):
            print(f"  Skipping answer {index}: malformed.")
            continue
        target = item["iri"].strip()
        if target not in allowed:
            print(f"  Skipping answer {index}: {target} was not an offered candidate.")
            continue
        if target in seen:
            continue
        if (focus, target) in closure or (target, focus) in closure:
            continue  # defensive; options already exclude these
        seen.add(target)
        rationale = item.get("rationale")
        kept.append({
            "subclass_iri": focus,
            "superclass_iri": target,
            "rationale": rationale.strip() if isinstance(rationale, str) and rationale.strip()
                         else "(no rationale given)",
        })
    if len(kept) > MAX_CANDIDATES_PER_CLASS:
        print(f"  Keeping first {MAX_CANDIDATES_PER_CLASS} of {len(kept)} answers.")
        kept = kept[:MAX_CANDIDATES_PER_CLASS]
    return kept


# ============================================================
# OUTPUT
# ============================================================

def write_candidates(candidates):
    graph = Graph()
    graph.bind("owl", OWL)
    graph.bind("rdfs", RDFS)

    ontology_iri = URIRef("https://www.commoncoreontologies.org/QualityOntology/llm-candidates")
    graph.add((ontology_iri, RDF.type, OWL.Ontology))
    graph.add((ontology_iri, RDFS.comment, Literal(
        "LLM-generated named subclass candidates (pairwise verification). "
        "To be scored by filter_candidates_hybrid.py before inclusion."
    )))

    for c in candidates:
        graph.add((URIRef(c["subclass_iri"]), RDFS.subClassOf, URIRef(c["superclass_iri"])))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(OUTPUT_PATH), format="turtle")

    rationale_path = OUTPUT_PATH.with_name("candidate_rationales.json")
    rationale_path.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    return rationale_path


# ============================================================
# MAIN
# ============================================================

def generate_candidates():
    valid_classes, asserted = load_source_ontology()
    definitions = load_definitions()

    if set(definitions) - valid_classes:
        raise ValueError("Definitions CSV contains IRIs that are not classes in the ontology.")

    closure = transitive_closure(asserted)
    print(f"Existing direct and indirect subclass relationships: {len(closure)}")

    branches = build_branches(definitions, asserted)
    total_focus = sum(len(m) for _, m in branches)
    print(f"\nBranches: {len(branches)} | classes to check: {total_focus}")

    cache = load_cache()
    candidates, api_calls, pairs_checked = [], 0, 0

    for name, members in branches:
        print(f"\nBranch: {name} ({len(members)} classes)")
        branch_found = 0
        for focus in members:
            options = unrelated_options(focus, members, closure)
            if not options:
                continue
            pairs_checked += len(options)
            if api_calls:
                time.sleep(SECONDS_BETWEEN_CALLS)
            raw, called = request_for_focus(focus, options, definitions, cache)
            api_calls += called
            kept = validate(focus, raw, options, closure)
            for c in kept:
                print(f"  + {definitions[focus]['label']} -> "
                      f"{definitions[c['superclass_iri']]['label']}")
            candidates.extend(kept)
            branch_found += len(kept)
        print(f"Candidates from branch: {branch_found}")

    rationale_path = write_candidates(candidates)

    print("\nCANDIDATE GENERATION SUMMARY")
    print("=" * 45)
    print(f"Pairs checked by Gemini:    {pairs_checked}")
    print(f"Gemini API calls this run:  {api_calls}")
    print(f"Unique new candidates:      {len(candidates)}")
    print(f"Output:     {OUTPUT_PATH}")
    print(f"Rationales: {rationale_path}")

    if not candidates:
        print("ERROR: no candidates were generated; the filter would have "
              "nothing to score.", file=sys.stderr)
        return 1
    return 0


def main():
    load_env()
    try:
        return generate_candidates()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())