import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import anthropic
from dotenv import load_dotenv

load_dotenv()                             # puts ANTHROPIC_API_KEY into the environment
client = anthropic.Anthropic(max_retries=5)  # SDK backs off on 429s/overloads

PROMPT = """Rewrite this CCO textual definition in the form
"A/An <label> is a/an <genus> that <differentia>."
Do not add or remove meaning. Expand abbreviations. Keep the word "designed" if present.
Label: {label}
Definition: {definition}
Return only the rewritten definition."""

def normalize(label, definition):
    msg = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": PROMPT.format(label=label, definition=definition)}],
    )
    for b in msg.content:
        if b.type == "text":
            return b.text.strip()
    raise RuntimeError(f"no text block (stop_reason={msg.stop_reason}) for {label}")

def enrich(row):
    d = row["definition"].strip()
    if not d:
        row["definition_enriched"] = ""
        return row
    try:
        row["definition_enriched"] = normalize(row["label"], d)
    except StopIteration:
        raise RuntimeError(f"no text block in response for {row['IRI']} ({row['label']})")
    return row

def load_previous(path):
    """Earlier rewrites, keyed by (IRI, original definition), so unchanged rows are reused."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, newline="", encoding="utf-8") as f:
        return {(r["IRI"], r["definition"]): r["definition_enriched"] for r in csv.DictReader(f)}

def main(in_path="data/definitions.csv", out_path="data/definitions_enriched.csv", workers=16):
    with open(in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    previous = load_previous(out_path)

    def enrich_or_reuse(row):
        old = previous.get((row["IRI"], row["definition"]))
        if old:
            row["definition_enriched"] = old
            return row
        return enrich(row)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(enrich_or_reuse, rows))   # preserves row order
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["IRI", "label", "definition", "definition_enriched"])
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
