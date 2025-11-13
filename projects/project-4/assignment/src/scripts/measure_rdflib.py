
import pandas as pd
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD, RDFS, OWL
import re
from collections import defaultdict
from pathlib import Path
import os
import argparse
#
# --- Resolve defaults relative to THIS script's directory ---
# If this file lives at .../assignment/src/scripts/measure_rdflib.py,
# then we want BASE_DIR = .../assignment/src (because data/ is a sibling of scripts/)
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

# Defaults relative to .../assignment/src/
DEFAULT_CSV = BASE_DIR / "data" / "readings_normalized.csv"
DEFAULT_TTL = BASE_DIR / "measure_cco.ttl"

# --- Allow env vars to override defaults (handy in CI/local) ---
CSV_ENV = os.environ.get("CSV_PATH")
TTL_ENV = os.environ.get("TTL_OUT")

# --- CLI args override everything ---
parser = argparse.ArgumentParser(description="Generate TTL from measurements CSV.")
parser.add_argument("--csv", default=CSV_ENV or str(DEFAULT_CSV), help="Path to input CSV")
parser.add_argument("--ttl-out", default=TTL_ENV or str(DEFAULT_TTL), help="Path to output TTL")
args, unknown = parser.parse_known_args()

CSV_PATH = str(Path(args.csv).expanduser().resolve())
TTL_OUT  = str(Path(args.ttl_out).expanduser().resolve())

# (Optional) quick sanity prints to help if paths drift:
print("Resolved CSV_PATH:", CSV_PATH)
print("Resolved TTL_OUT :", TTL_OUT)

# === Create RDF graph ===
g = Graph()

# === Namespaces ===
rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
xsd = Namespace("http://www.w3.org/2001/XMLSchema#")
cco = Namespace("https://www.commoncoreontologies.org/")
obo = Namespace("http://purl.obolibrary.org/obo/")
ex = Namespace("http://example.org/instances#")
exprop= Namespace("http://example.org/props#")
# dedicated namespace for classes we generate
exc = Namespace("http://example.org/classes#")

# === Bind prefixes ===
g.bind("rdf", rdf)
g.bind("xsd", xsd)
g.bind("cco", cco)
g.bind("obo", obo)
g.bind("ex", ex)
g.bind("exc", exc)
g.bind("exprop", exprop)
g.bind("rdfs", RDFS)
g.bind("owl", OWL)

# === Ontology header ===
ontology_uri = URIRef("http://example.org/ontology")
g.add((ontology_uri, RDF.type, OWL.Ontology))
g.add((ontology_uri, RDFS.label, Literal("cco conformant measurements", lang="en")))

# ----------------------------------------------------------------------------------------------------
# EXPLICIT labels for BFO terms (so definitions show term names, not IRIs)
# ----------------------------------------------------------------------------------------------------
g.add((obo.BFO_0000020, RDFS.label, Literal("Specifically Dependent Continuant", lang="en")))
g.add((obo.BFO_0000031, RDFS.label, Literal("Generically Dependent Continuant", lang="en")))
g.add((obo.BFO_0000040, RDFS.label, Literal("Material Entity", lang="en")))

# ----------------------------------------------------------------------------------------------------
# Requested labels (ensure definitions use these labels, not IRIs)
# ----------------------------------------------------------------------------------------------------
# === Labels for CCO Measurement Units ===
g.add((cco.ont00001450, RDFS.label, Literal("Volt Measurement Unit", lang="en")))
g.add((cco.ont00001559, RDFS.label, Literal("Pascal Measurement Unit", lang="en")))  # Note: Pascal uses ont00001559 in your code
g.add((cco.ont00001606, RDFS.label, Literal("Degree Celsius Measurement Unit", lang="en")))
g.add((cco.ont00001694, RDFS.label, Literal("Pounds Per Square Inch Measurement Unit", lang="en")))
g.add((cco.ont00001724, RDFS.label, Literal("Degree Fahrenheit Measurement Unit", lang="en")))
g.add((cco.ont00000120, RDFS.label, Literal("Measurement Unit", lang="en")))
g.add((cco.ont00001163, RDFS.label, Literal("Measurement Information Content Entity", lang="en")))
g.add((cco.ont00000995, RDFS.label, Literal("Artifact", lang="en")))
# Switched from BFO_0000196 to BFO_0000197 and labeled it "inheres in"
g.add((obo.BFO_0000197, RDFS.label, Literal("inheres in", lang="en")))
g.add((cco.ont00001863, RDFS.label, Literal("uses measurement unit", lang="en")))
# Keep CCO measurement role labels
g.add((cco.ont00001966, RDFS.label, Literal("is a measurement of", lang="en")))
g.add((cco.ont00001904, RDFS.label, Literal("is measured by", lang="en")))
g.add((cco.ont00001769, RDFS.label, Literal("has decimal value", lang="en")))

# === Labels for inverse properties brought in from CCO/BFO ===
g.add((obo.BFO_0000196, RDFS.label, Literal("bearer of", lang="en")))  # inverse of inheres in
g.add((cco.ont00001961, RDFS.label, Literal("is measurement unit of", lang="en")))  # inverse of cco:ont00001863

# ----------------------------------------------------------------------------------------------------
# EXPLICIT DEFINITIONS (exact text preserved as-is where provided)
# ----------------------------------------------------------------------------------------------------
CLASS_DEFS = {
    obo.BFO_0000020: "A specifically dependent continuant is a continuant & there is some independent continuant c which is not a spatial region and which is such that b s-depends_on c at every time t during the course of b’s existence.",
    obo.BFO_0000031: "A generically dependent continuant is a continuant that g-depends_on one or more other entities.",
    obo.BFO_0000040: "A material entity is an independent continuant that has some portion of matter as proper or improper continuant part.",
}
# Object properties with verbatim definitions (ensures our new inverses get real definitions)
OBJECT_PROPERTY_DEFS = {
    obo.BFO_0000196: "b bearer of c =Def c inheres in b",  # from BFO (inverse of inheres in)
    cco.ont00001904: "y is_measured_by x iff x is an instance of Information Content Entity and y is an instance of Entity, such that x describes some attribute of y relative to some scale or classification scheme.",
    # Corrected explicit definition for 'is a measurement of'
    cco.ont00001966: "x is_a_measurement_of y iff x is an instance of Measurement Information Content Entity and y is an instance of Specifically Dependent Continuant (a reading), such that x specifies a value describing some attribute of y relative to some scale or classification scheme.",
    # ⬇️ BROADENED to allow y to be an SDC (not solely an IBE / ICE)
    cco.ont00001961: "x is_measurement_unit_of y iff x is an instance of Measurement Unit and y is an instance of Measurement Information Content Entity or Specifically Dependent Continuant, such that x describes or qualifies the magnitude of the measured physical quantity referenced in y.",
}
DATATYPE_PROPERTY_DEFS = {
}

# ----------------------------------------------------------------------------------------------------
# DIFFERENTIA mapping
# ----------------------------------------------------------------------------------------------------
DIFFERENTIA = {
    # Classes
    cco.ont00000995: "is intentionally produced to realize some function or purpose",  # Artifact
    cco.ont00001163: "specifies a numeric measurement value together with its associated unit",  # Measurement ICE
    cco.ont00000120: "serves to standardize quantities for measurement information content entities and specifically dependent continuants",  # ⬅ broadened text
    # Object properties
    obo.BFO_0000197: "relates a specifically dependent continuant to the independent continuant it inheres in",  # 'inheres in'
    cco.ont00001966: "links a measurement information content entity to the reading (specifically dependent continuant) that it specifies",
    # ⬇️ BROADENED: can link either a Measurement ICE or an SDC to a unit
    cco.ont00001863: "links a measurement information content entity or a specifically dependent continuant to the unit that qualifies its value",
    # Datatype properties
    cco.ont00001769: "associates a measurement information content entity with a numeric value literal",  # 'has decimal value'
}

# ----------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------
_defined_terms = set()

def article_for(s: str) -> str:
    return "An" if s[:1].lower() in ("a", "e", "i", "o", "u") else "A"

def label_or_localname(iri: URIRef) -> str:
    for _, _, lab in g.triples((iri, RDFS.label, None)):
        if isinstance(lab, Literal) and (lab.language or "").lower().startswith("en") and str(lab).strip():
            return str(lab).strip()
    s = str(iri)
    local = s.rsplit("#", 1)[-1] if "#" in s else s.rsplit("/", 1)[-1]
    return local.replace("_", " ").replace("-", " ").strip()

def parent_of(term: URIRef):
    for _, _, parent in g.triples((term, RDFS.subClassOf, None)):
        if isinstance(parent, URIRef):
            return parent
    return None

def ensure_definition(term_iri: URIRef, definition_text: str):
    if not definition_text or not str(definition_text).strip():
        definition_text = "Definition not provided."
    if term_iri not in _defined_terms and not has_english_definition(term_iri):
        g.add((term_iri, RDFS.comment, Literal(definition_text.strip(), lang="en")))
        _defined_terms.add(term_iri)

def has_english_definition(term_iri: URIRef) -> bool:
    for _, _, c in g.triples((term_iri, RDFS.comment, None)):
        if isinstance(c, Literal) and c.language and c.language.lower().startswith("en") and str(c).strip():
            return True
    return False

def make_non_self_referential(def_text: str, class_iri: URIRef) -> str:
    """
    Ensures the portion AFTER the first ' is a ' / ' is an ' does not contain the
    class label or the local IRI token. If it does, replaces them with neutral phrases.
    Works for class definitions of the form: '<Head> is a ... <Body>'.
    """
    if not isinstance(def_text, str) or not def_text.strip():
        return def_text
    # Class label (preferred) and fallback to local token
    class_label = label_or_localname(class_iri).strip()
    iri_str = str(class_iri)
    local_token = iri_str.rsplit("#", 1)[-1] if "#" in iri_str else iri_str.rsplit("/", 1)[-1]
    local_token = local_token.strip()

    # Find the split point (" is a " / " is an "), case-insensitive on search,
    # but keep the original casing in the returned string.
    lower_def = def_text.lower()
    split_needles = (" is a ", " is an ")
    split_idx = -1
    needle_used = None
    for needle in split_needles:
        i = lower_def.find(needle)
        if i != -1:
            split_idx = i
            needle_used = def_text[i:i+len(needle)]  # preserve original casing in that slice
            break
    if split_idx == -1:
        # No " is a / is an " head/body split; return as-is (we won't rewrite free-text defs)
        return def_text

    # Keep head + delimiter, scrub the body
    head = def_text[:split_idx + len(needle_used)]
    body = def_text[split_idx + len(needle_used):]

    def _whole_word_replace(text: str, token: str, replacement: str) -> str:
        if not token:
            return text
        pattern = r'(?i)(^|[^A-Za-z0-9_])(' + re.escape(token) + r')([^A-Za-z0-9_]|$)'
        return re.sub(pattern, r'\1' + replacement + r'\3', text)

    # Replace label first, then the local token (if different)
    new_body = _whole_word_replace(body, class_label, "this quality")
    if local_token and local_token.lower() != class_label.lower():
        new_body = _whole_word_replace(new_body, local_token, "this class")
    return head + new_body

def ensure_clean_definition(term_iri: URIRef, definition_text: str):
    """
    Wrapper that makes a class definition non-self-referential before adding it.
    For non-classes, it passes through unchanged.
    """
    # Determine if term_iri is asserted as a class (we only scrub class definitions)
    is_class = any(g.triples((term_iri, RDF.type, OWL.Class))) or any(g.triples((term_iri, RDF.type, RDFS.Class)))
    clean_text = make_non_self_referential(definition_text, term_iri) if is_class else definition_text
    ensure_definition(term_iri, clean_text)

def _slug(text: str) -> str:
    # safe slug for URIs: keep letters, digits, and underscores; convert others to '-'
    t = re.sub(r"[^A-Za-z0-9_]+", "-", (text or "")).strip("-")
    t = re.sub(r"-+", "-", t)
    return t or "na"

# --- Canonicalization of sdc_kind ---
KIND_ALIASES = {
    # canonical label -> set of acceptable surface forms (lowercased/stripped)
    "Temperature": {"temperature", "temp", "tmp", "t", "degc", "degf", "c", "f"},
    "Pressure": {"pressure", "press", "psi", "bar", "pa", "kpa"},
    "Humidity": {"humidity", "rh", "rel humidity", "relative humidity"},
    # NEW qualities
    "Resistance": {"resistance", "res", "ohm", "ohms", "ω", "r"},
    "Voltage": {"voltage", "volt", "volts", "v"},
}

# Only these qualities get the extra classing as top-level quality classes
QUALITIES_FOR_CLASSING = {"Pressure", "Temperature", "Resistance", "Voltage"}

def canonicalize_kind(raw: str) -> tuple[str, str]:
    """
    Returns (canonical_label, canonical_slug).
    If no alias match is found, uses title-cased cleaned form.
    """
    base = (raw or "").strip().lower()
    base = re.sub(r"[^a-z0-9]+", " ", base).strip()
    for canonical, forms in KIND_ALIASES.items():
        if base in forms:
            return canonical, _slug(canonical.lower())
    # default: title case the cleaned form
    canon = base.title() if base else "Unknown"
    return canon, _slug(canon.lower())

# --- Unit normalization and mapping to CCO URIs ---
# Canonical aliases for unit labels as they may appear in the CSV
UNIT_ALIASES = {
    "c": {"c", "degc", "celsius", "°c"},
    "f": {"f", "degf", "fahrenheit", "°f"},
    "pa": {"pa", "pascal"},
    "kpa": {"kpa", "kilopascal", "kilopascals"},
    "psi": {"psi", "pound per square inch"},
    "volt": {"v", "volt", "volts"},
    "ohm": {"ohm", "ohms", "ω", "omega"},
}

def _normalize_unit_token(u_raw: str) -> str:
    base = (u_raw or "").strip().lower()
    if base:
        for canon, forms in UNIT_ALIASES.items():
            if base in forms:
                return canon
    base2 = re.sub(r"[^a-z0-9]+", "", base)
    for canon, forms in UNIT_ALIASES.items():
        if base2 in {re.sub(r"[^a-z0-9]+", "", f) for f in forms}:
            return canon
    return base2 or "na"

def resolve_unit_and_value(unit_raw: str, value_in: float):
    """
    Returns (unit_uri, adjusted_value, is_external_unit) where:
      - unit_uri is the chosen URIRef for the unit
      - adjusted_value may be scaled (e.g., kPa -> Pa * 1000)
      - is_external_unit is True when using a CCO IRI (False for local)
    """
    canon = _normalize_unit_token(unit_raw)

    # Map to CCO IRIs where available (per user requirement)
    if canon == "c":
        return cco.ont00001606, value_in, True  # Celsius
    if canon == "f":
        return cco.ont00001724, value_in, True  # Fahrenheit
    if canon == "kpa":
        return cco.ont00001559, value_in * 1000.0, True  # kPa -> Pa (Pascal)
    if canon == "pa":
        return cco.ont00001559, value_in, True  # Pascal
    if canon == "psi":
        return cco.ont00001694, value_in, True  # PSI
    if canon == "volt":
        return cco.ont00001450, value_in, True  # Volt

    # No CCO IRI for ohm per requirement; keep it local
    if canon == "ohm":
        return URIRef(ex + "ohm"), value_in, False

    # Fallback: keep any other units local (typed as cco:Measurement Unit)
    return URIRef(ex + _slug(unit_raw)), value_in, False

# ----------------------------------------------------------------------------------------------------
# Up-front declarations
# ----------------------------------------------------------------------------------------------------
EXPLICIT_CLASSES = [
    cco.ont00000441,  # Temperature
    cco.ont00000995,  # Artifact
    obo.BFO_0000020,  # Specifically Dependent Continuant
    cco.ont00001163,  # Measurement Information Content Entity
    cco.ont00000120,  # Measurement Unit
    obo.BFO_0000040,  # Material Entity
    obo.BFO_0000031,  # Generically Dependent Continuant
]
EXPLICIT_OBJECT_PROPS = [
    obo.BFO_0000197,  # inheres in
    obo.BFO_0000196,  # bearer of (inverse)
    cco.ont00001966,  # is a measurement of
    cco.ont00001904,  # is measured by (inverse)
    cco.ont00001863,  # uses measurement unit
    cco.ont00001961,  # is measurement unit of (inverse)
]
EXPLICIT_DATATYPE_PROPS = [
    cco.ont00001769,
    exprop.hasTimestamp,
]
for k in EXPLICIT_CLASSES:
    g.add((k, RDF.type, OWL.Class))
for p in EXPLICIT_OBJECT_PROPS:
    g.add((p, RDF.type, OWL.ObjectProperty))
for p in EXPLICIT_DATATYPE_PROPS:
    g.add((p, RDF.type, OWL.DatatypeProperty))

# Provide human-friendly label for timestamp property
g.add((exprop.hasTimestamp, RDFS.label, Literal("has timestamp", lang="en")))

for klass, desc in CLASS_DEFS.items():
    g.add((klass, RDF.type, OWL.Class))
    ensure_definition(klass, desc)
for prop, desc in OBJECT_PROPERTY_DEFS.items():
    g.add((prop, RDF.type, OWL.ObjectProperty))
    ensure_definition(prop, desc)
for prop, desc in DATATYPE_PROPERTY_DEFS.items():
    g.add((prop, RDF.type, OWL.DatatypeProperty))
    ensure_definition(prop, desc)

# === Inverse property axioms ===
g.add((obo.BFO_0000196, OWL.inverseOf, obo.BFO_0000197))
g.add((obo.BFO_0000197, OWL.inverseOf, obo.BFO_0000196))
g.add((cco.ont00001904, OWL.inverseOf, cco.ont00001966))
g.add((cco.ont00001966, OWL.inverseOf, cco.ont00001904))
g.add((cco.ont00001961, OWL.inverseOf, cco.ont00001863))
g.add((cco.ont00001863, OWL.inverseOf, cco.ont00001961))

# ----------------------------------------------------------------------------------------------------
# Load CSV and build instance data
# CHANGE: ONE SDC per (artifact, canonical kind, timestamp) to enforce 1–1 MICE↔SDC
# ----------------------------------------------------------------------------------------------------
# key: (artifact_slug, canon_kind_slug, ts_slug) -> reading_uri
reading_cache = {}
# cache classes to avoid re-adding (top-level only; NO artifact-specific subclasses)
quality_class_cache = {}  # canon_kind_slug -> class URI (exc or CCO for Temperature)

df = pd.read_csv(CSV_PATH)
for _, row in df.iterrows():
    artifact_id_raw = str(row['artifact_id']).strip()
    sdc_kind_raw = str(row['sdc_kind']).strip()
    unit_raw = str(row['unit_label']).strip()
    timestamp_raw = str(row['timestamp']).strip()

    # skip if key fields missing
    if not artifact_id_raw or not sdc_kind_raw or not unit_raw or not timestamp_raw:
        continue
    try:
        value = float(row['value'])
    except Exception:
        continue

    # Canonicalize
    artifact_label = artifact_id_raw.replace(" ", "-")
    artifact_slug = _slug(artifact_label)
    canon_kind_label, canon_kind_slug = canonicalize_kind(sdc_kind_raw)
    ts_slug = _slug(timestamp_raw)  # ⬅ participates in SDC key

    # URIs
    artifact_uri = URIRef(ex + artifact_slug)
    # Ensure artifact typed & labeled (idempotent)
    g.add((artifact_uri, RDF.type, cco.ont00000995))  # Artifact
    g.add((artifact_uri, RDFS.label, Literal(artifact_label, lang="en")))

    # Ensure the QUALITY CLASS (top-level only; NO artifact-specific subclass)
    if canon_kind_label in QUALITIES_FOR_CLASSING:
        if canon_kind_slug not in quality_class_cache:
            if canon_kind_label == "Temperature":
                # Use CCO Temperature as the top quality class
                qual_class_uri = cco.ont00000441
                quality_class_cache[canon_kind_slug] = qual_class_uri
                # Assert type/label locally (harmless if already present)
                g.add((qual_class_uri, RDF.type, OWL.Class))
                g.add((qual_class_uri, RDFS.label, Literal("Temperature", lang="en")))
            else:
                # Default behavior for other qualities (Pressure, Resistance, Voltage)
                qual_class_uri = URIRef(exc + canon_kind_slug)
                quality_class_cache[canon_kind_slug] = qual_class_uri
                g.add((qual_class_uri, RDF.type, OWL.Class))
                g.add((qual_class_uri, RDFS.subClassOf, obo.BFO_0000020))
                g.add((qual_class_uri, RDFS.label, Literal(canon_kind_label, lang="en")))
                _qdef = (
                    f"{article_for(canon_kind_label)} {canon_kind_label} is a Specifically Dependent Continuant quality "
                    f"that can inhere in a material entity and is typically subject to measurement."
                )
                ensure_definition(qual_class_uri, _qdef)
        else:
            qual_class_uri = quality_class_cache[canon_kind_slug]
    else:
        qual_class_uri = None

    # === ONE SDC per (artifact, canonical kind, timestamp) ===
    reading_key = (artifact_slug, canon_kind_slug, ts_slug)
    if reading_key in reading_cache:
        reading_uri = reading_cache[reading_key]
    else:
        reading_id = f"{artifact_slug}_{canon_kind_slug}_{ts_slug}"
        reading_uri = URIRef(ex + reading_id)
        reading_cache[reading_key] = reading_uri

    # Create the SDC node (direct typing; no artifact-specific class)
    g.add((reading_uri, RDF.type, obo.BFO_0000020))  # SDC
    # Also type the reading as an instance of the top-level quality class
    if qual_class_uri is not None:
        g.add((reading_uri, RDF.type, qual_class_uri))
    g.add((reading_uri, RDFS.label, Literal(f"{artifact_label}_{canon_kind_label} @ {timestamp_raw}", lang="en")))
    g.add((reading_uri, obo.BFO_0000197, artifact_uri))  # inheres in
    g.add((artifact_uri, obo.BFO_0000196, reading_uri))

    # --- Resolve unit to CCO IRIs and adjust numeric value if needed ---
    unit_uri, value, is_external_unit = resolve_unit_and_value(unit_raw, value)

    # Declare unit as a Measurement Unit (safe even for external IRIs per current modeling)
    g.add((unit_uri, RDF.type, cco.ont00000120))
    # Only add a label for local (non-CCO) units to avoid mislabeling external IRIs
    if not is_external_unit:
        g.add((unit_uri, RDFS.label, Literal(unit_raw, lang="en")))

    # For each row, create a MICE (value + unit + timestamp)
    mice_id = f"MICE_{artifact_slug}_{canon_kind_slug}_{ts_slug}"
    mice_uri = URIRef(ex + mice_id)
    g.add((mice_uri, RDF.type, cco.ont00001163))
    g.add((mice_uri, RDFS.label, Literal(f"MICE for {artifact_label}_{canon_kind_label} @ {timestamp_raw}", lang="en")))
    g.add((mice_uri, cco.ont00001769, Literal(value, datatype=XSD.decimal)))  # has decimal value

    # -- Units --
    # Attach unit to MICE and to SDC
    g.add((mice_uri, cco.ont00001863, unit_uri))    # uses measurement unit (MICE -> Unit)
    g.add((reading_uri, cco.ont00001863, unit_uri)) # uses measurement unit (SDC -> Unit)

    # --- MICE <-> SDC linkage (1–to–1) ---
    g.add((mice_uri, cco.ont00001966, reading_uri))  # MICE is a measurement of this SDC
    g.add((reading_uri, cco.ont00001904, mice_uri))  # SDC is measured by this MICE

    # Timestamp on the MICE (xsd:dateTime lexical form as given)
    g.add((mice_uri, exprop.hasTimestamp, Literal(timestamp_raw, datatype=XSD.dateTime)))

# --- Explicit definition for CCO Temperature (idempotent) ---
ensure_clean_definition(
    cco.ont00000441,
    """Temperature is a specifically dependent continuant quality of a material entity that
    quantifies the thermal state of that entity (often characterized as the average kinetic energy
    of its constituent particles) and is assessed relative to a thermometric scale (e.g., Celsius,
    Fahrenheit, Kelvin)."""
)

# Parent classes
g.add((cco.ont00000441, RDFS.subClassOf, obo.BFO_0000020))
g.add((cco.ont00000995, RDFS.subClassOf, obo.BFO_0000040))
g.add((cco.ont00001163, RDFS.subClassOf, obo.BFO_0000031))
g.add((cco.ont00000120, RDFS.subClassOf, obo.BFO_0000031))

# ----------------------------------------------------------------------------------------------------
# Final pass: add genus–differentia definitions
# ----------------------------------------------------------------------------------------------------
# --- Final pass: add genus–differentia definitions (safe string assembly) ---
seen_classes = set()
for c in g.subjects(RDF.type, OWL.Class):
    if isinstance(c, URIRef):
        seen_classes.add(c)
for _, _, class_iri in g.triples((None, RDF.type, None)):
    if isinstance(class_iri, URIRef):
        seen_classes.add(class_iri)

for term in seen_classes:
    if not has_english_definition(term):

        # Pieces
        term_label = label_or_localname(term)
        parent = parent_of(term)
        parent_label = label_or_localname(parent) if parent else "parent class (unspecified)"

        # Default differentia if not in map
        default_diff = f"has not yet had its differentiating factor specified relative to {parent_label}"
        diff_text = DIFFERENTIA.get(term, default_diff)

        # Build the sentence
        article = article_for(term_label)
        # Example: "A Temperature is a Specifically Dependent Continuant that <diff>."
        def_text = f"{article} {term_label} is a {parent_label} that {diff_text}."

        ensure_clean_definition(term, def_text)
# ----------------------------------------------------------------------------------------------------
# Serialize
# ----------------------------------------------------------------------------------------------------
g.serialize(destination=TTL_OUT, format="turtle")
