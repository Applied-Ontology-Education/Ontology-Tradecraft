"""Stage 2: LLM schema normalization and pattern mapping -> generated/llm_schema_mapping.json + reports/schema_mapping_review.csv.

Claude receives the schema profile (no raw rows beyond examples) and the CCO/BFO pattern vocabulary, and returns,
for each column: meaning, whether it is standard GTFS, normalized datatype, normalization rule, and the pattern
element it maps to, plus data-quality issues it detects. The result is compared with the approved spec
(patterns/gtfs_patterns.json): agreement is reported per column; disagreements are for human review, never auto-applied."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import GEN, REPORTS, ask_json, load_spec, save_cache, write_json  # noqa: E402

SYSTEM = ("You are a data and ontology engineer integrating public transit (GTFS) data into the Common Core Ontologies "
          "(CCO, BFO-based). You reply with JSON only.")
PROMPT = """Here is a profile of a GTFS Schedule feed for a commuter rail branch (one service day).

PROFILE (per file: rows; per column: inferred type, null share, distinct count, examples, gtfs_standard flag):
{profile}

PATTERN VOCABULARY you may map columns to (use these names exactly):
{vocab}

Tasks:
1. For EVERY column of EVERY file, give: meaning; standard (true if part of the GTFS Schedule reference);
   datatype (normalized XSD type); normalization (rule needed before loading, or "none");
   maps_to (pattern element from the vocabulary, or "not modeled" if it is presentation-only or not needed).
2. List data-quality issues visible in the profile (invalid codes, non-standard columns, values needing normalization).
Reply with exactly:
{{"files": {{"<file>": {{"<column>": {{"meaning": "...", "standard": true, "datatype": "xsd:...", "normalization": "...", "maps_to": "..."}}}}}},
 "issues": [{{"file": "...", "column": "...", "issue": "...", "severity": "error|warning|info"}}]}}"""

VOCAB = ["Rail Station", "GTFS Record (Information Bearing Entity)", "Station Coordinate Record", "Geospatial Location",
         "Transit Route Specification", "Train Trip Specification", "Scheduled Stop Specification", "Organization",
         "Designative Name", "Code Identifier", "Arbitrary Identifier", "Time Zone Identifier", "Time of Day Identifier",
         "has text value", "has latitude value", "has longitude value", "has datetime value", "stopSequence",
         "directionCode", "routeTypeCode", "wheelchairBoardingCode", "isPeakService", "headsignText",
         "continuant part of", "is about", "designates", "uses time zone identifier", "service day selection", "not modeled"]
KEY = [v.split(" (")[0].lower() for v in VOCAB if v != "not modeled"] + ["route_type", "service day"]


def terms(text: str) -> set:
    t = text.lower()
    if "not modeled" in t:
        return {"not modeled"}
    return {k for k in KEY if k in t}


def main():
    prof = json.loads((REPORTS / "schema_profile.json").read_text())
    spec = load_spec()
    out = ask_json(SYSTEM, PROMPT.format(profile=json.dumps(prof, indent=1), vocab="\n".join(f"- {v}" for v in VOCAB)),
                   max_tokens=12000)
    save_cache()
    write_json(GEN / "llm_schema_mapping.json", out)
    rows, agree, n = [], 0, 0
    for f, cols in spec["columns"].items():
        for c, approved in cols.items():
            llm = (out.get("files", {}).get(f, {}) or {}).get(c, {})
            a, b = terms(approved), terms(str(llm.get("maps_to", "")))
            ok = bool(a & b) or (a == b == set())
            n += 1; agree += ok
            rows.append({"file": f, "column": c, "approved": approved, "llm_maps_to": llm.get("maps_to", "(missing)"),
                         "llm_standard": llm.get("standard", ""), "llm_normalization": llm.get("normalization", ""),
                         "agrees": ok})
    with open(REPORTS / "schema_mapping_review.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    issues = out.get("issues", [])
    write_json(REPORTS / "schema_issues_llm.json", issues)
    flagged = sorted({f"{i.get('file')}:{i.get('column')}" for i in issues})
    print(f"llm_normalize_schema: {agree}/{n} columns agree with the approved spec ({agree / n:.0%}); "
          f"{len(issues)} issues flagged ({', '.join(flagged[:6])}{' ...' if len(flagged) > 6 else ''})")
    write_json(REPORTS / "schema_mapping_summary.json", {"columns": n, "agree": agree, "agreement": round(agree / n, 3),
                                                        "issues": len(issues)})


if __name__ == "__main__":
    main()
