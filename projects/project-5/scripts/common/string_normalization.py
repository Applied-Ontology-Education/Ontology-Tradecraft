import re
from typing import Iterable, Set, List

from nltk.stem.snowball import SnowballStemmer


def remove_snake_case(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if not text:
        return ""
    # Matches snake_case words (e.g. 'is_subject_of') and converts to "'is subject of'"
    result = re.sub(
        r'\b[a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)+\b',
        lambda m: f"'{m.group(0).replace('_', ' ')}'",
        text
    )

    return result


def apply_single_quotes(text: str, labels: Iterable[str]) -> str:
    """Wrap occurrences of given labels in single quotes.

    Requirements:
    - Do not double-quote labels that are already surrounded by single quotes.
    - Prefer the longest label when labels overlap/contain each other.

    Matching is case-insensitive, but the original matched text's casing is
    preserved in the output. Only full-word matches are considered (via \b
    word boundaries).
    """
    # Prepare labels: deduplicate, drop empties, sort by length desc so that
    # longer alternatives are preferred by the regex alternation.
    alts = [l for l in {lbl for lbl in labels if lbl}]
    if not alts:
        return text
    alts.sort(key=len, reverse=True)

    try:
        # Build a single-pass pattern with alternation of all labels.
        # Use negative lookbehind/ahead to avoid matches already wrapped in '
        # Example it avoids: 'Black Dog' -> we won't wrap again.
        alternation = "|".join(re.escape(l) for l in alts)
        pattern = re.compile(r"(?<!')\b(?:" + alternation + r")\b(?!')", re.IGNORECASE)
    except re.error:
        # If something goes wrong compiling the big pattern (very unlikely
        # since each part is escaped), fall back to original text.
        return text

    def _wrap(m: re.Match) -> str:
        # Preserve the original matched casing/content, just add quotes.
        return "'" + m.group(0) + "'"

    return pattern.sub(_wrap, text)


def contains(text: str, value: str) -> str:
    return re.search(rf"\b{re.escape(value.lower())}\b", text.lower()) is not None


def replace_x_and_y(line: str):
    line = re.sub(
        r"^x ",
        "individual x ",
        line
    )

    line = re.sub(
        r" x,",
        " individual x,",
        line
    )

    line = re.sub(
        r" x\.",
        " individual x.",
        line
    )

    line = re.sub(
        r" x ",
        " individual x ",
        line
    )

    line = re.sub(
        r"^y ",
        "individual y ",
        line
    )

    line = re.sub(
        r" y,",
        " individual y,",
        line
    )

    line = re.sub(
        r" y\.",
        " individual y.",
        line
    )

    line = re.sub(
        r" y ",
        " individual y ",
        line
    )

    return line


def collect_labels(text: str, labels: Iterable[str]) -> Set[str]:
    collection = set()
    for lbl in labels:
        if lbl in text:
            collection.add(lbl)
    return collection


def filter_labels(text: str, labels: Iterable[str]) -> str:
    for lbl in labels:
        text = text.replace(lbl, "")
    text.replace("''", "")
    return text


def stem_words(words: Iterable[str]) -> List[str]:
    stemmed_words = []

    # Initialize stemmer for English
    stemmer = SnowballStemmer('english')

    # Stem the words
    for word in words:
        stem = stemmer.stem(word)
        stemmed_words.append(stem)


def filter_stopwords(words: Iterable[str], stopwords: Iterable[str]) -> List[str]:
    return list(set(words) - set(stopwords))


def collect_words(text: str) -> Set[str]:
    collection = set()

    # Remove all non-alphanumeric characters
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)

    for entry in text.split(' '):
        if entry.strip() != '':
            collection.add(entry)

    return collection


def apply_label_casing(text: str, labels: Iterable[str]) -> str:
    """Apply canonical casing of provided labels to the text.

    Requirements:
    - Match labels case-insensitively in the text, but replace with the label's
      canonical casing as provided in `labels`.
    - Prefer the longest label when labels overlap/contain each other, so the
      casing of the longest matching label is applied.
    - Only match full words using word boundaries.
    """
    if not labels:
        return text

    # Build a canonical map case-insensitively: for each lowercased form, keep a
    # single canonical spelling (casing) to apply. If multiple inputs differ only
    # by case, keep the first longest; if lengths equal, keep the first seen.
    canonical = {}
    order = []  # track insertion order of chosen canonicals
    for lbl in labels:
        if not lbl:
            continue
        key = lbl.lower()
        if key not in canonical or len(lbl) > len(canonical[key]):
            # If this is a new lowercased key, remember order for alternation build.
            if key not in canonical:
                order.append(key)
            canonical[key] = lbl

    if not canonical:
        return text

    # Build alternation of canonical spellings, sorted by length descending to
    # ensure longest match wins when overlaps occur.
    alts = [canonical[k] for k in order]
    alts.sort(key=len, reverse=True)

    try:
        alternation = "|".join(re.escape(a) for a in alts)
        pattern = re.compile(r"\b(?:" + alternation + r")\b", re.IGNORECASE)
    except re.error:
        return text

    def _to_canonical(m: re.Match) -> str:
        return canonical.get(m.group(0).lower(), m.group(0))

    return pattern.sub(_to_canonical, text)


def apply_property_snakecase(text: str, snake_labels: Iterable[str]) -> str:
    """Normalize occurrences of property labels to snake_case and lowercase.

    For each snake_case label like "is_about", we will also normalize variants
    appearing with spaces or hyphens (e.g., "is about", "is-about") and case
    differences to the canonical snake_case form.
    """
    out = text
    for lbl in sorted({l for l in snake_labels if l}, key=len, reverse=True):
        try:
            parts = [re.escape(p) for p in lbl.split("_") if p]
            if not parts:
                continue
            # Pattern that matches the label if written with spaces, hyphens, or underscores between parts
            sep = r"[ _-]+"
            pattern = r"\b" + sep.join(parts) + r"\b"
            out = re.sub(pattern, lbl, out, flags=re.IGNORECASE)
            # Also normalize already-underscore form but wrong casing to lowercase canonical
            u_pattern = r"\b" + re.escape(lbl) + r"\b"
            out = re.sub(u_pattern, lbl, out, flags=re.IGNORECASE)
        except re.error:
            continue
    return out
