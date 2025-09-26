import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple, List, Optional
import re
import pandas as pd
from rdflib import Graph, URIRef, BNode, Literal
from rdflib.namespace import RDF, RDFS, OWL, Namespace

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

# ---------- RDF list helper ----------
def iter_rdf_list(g: Graph, head) -> Iterable:
    """
    Iterates through an RDF collection provided as a linked list structure in the given RDF graph.
    The method processes the collection starting from the specified head node and yields each
    element in the list in sequential order. The iterations stop when the end of the list
    (`RDF.nil`) is reached or the head is `None`.

    :param g: The RDF graph in which the linked list is defined.
    :type g: Graph
    :param head: The head node of the RDF-linked list structure.
    :return: An iterable of elements contained in the RDF list, in sequential order.
    :rtype: Iterable
    """
    while head and head != RDF.nil:
        first = g.value(head, RDF.first)
        if first is not None:
            yield first
        head = g.value(head, RDF.rest)

# ---------- Labels ----------
def best_label(g: Graph, n: URIRef) -> str:
    """
    Determines the best available label for a given node in a graph. The function
    first attempts to retrieve the label associated with the node using preferred
    properties such as RDFS.label and SKOS.prefLabel. If no such label exists, it
    constructs a fallback label based on the URI of the node by extracting the
    fragment identifier or last path segment present.

    :param g:
        The graph containing the RDF triples. It is expected to be an instance
        of a graph structure, such as rdflib.Graph, that allows querying.
    :param n:
        The node (URI reference) for which the label is being fetched. The node
        is represented as a URIRef from the rdflib library.
    :return:
        Returns the best label as a string for the given node in the graph.
        If no explicit label exists, the method returns a fallback string
        derived from the URI of the node.
    """
    for p in (RDFS.label, SKOS.prefLabel):
        o = g.value(n, p)
        if isinstance(o, Literal):
            return str(o)
    s = str(n)
    return s.rsplit("#", 1)[-1].rsplit("/", 1)[-1]

def name_qname(g: Graph, n: URIRef) -> str:
    """
    Converts a given URIRef into its QName representation using the namespace
    manager from the graph. If an exception occurs during the QName conversion,
    it returns the string representation of the URIRef.

    :param g: A graph object from the RDFLib library that includes a namespace
        manager responsible for the conversion of URIs to QNames.
    :type g: Graph
    :param n: The URIRef that needs to be converted into a QName representation
        or returned as a string if the conversion fails.
    :type n: URIRef
    :return: The QName representation of the URIRef if successfully converted.
        Otherwise, returns the string representation of the URIRef.
    :rtype: str
    """
    try:
        return g.namespace_manager.qname(n)
    except Exception:
        return str(n)

# ---------- Shape extraction ----------
def _as_int(n) -> Optional[int]:
    """
    Converts the provided input into an integer if possible, by evaluating its type
    or content. This function handles special cases where the input is an instance
    of `Literal`, converting it to its integer representation directly, or attempts
    a conversion based on its string form. If conversion is unsuccessful, it returns
    `None`.

    :param n: The input value to be converted. May be of type Literal or any other
              type that can be cast to a string or an integer.
    :return: The integer representation of the input if convertible, otherwise
             `None`.
    :rtype: Optional[int]
    """
    try:
        if isinstance(n, Literal):
            return int(n.toPython())
        return int(str(n))
    except Exception:
        return None

def restriction_shape_token(g: Graph, rnode, granularity: str) -> Optional[str]:
    """
    Determines a restriction shape token for a given RDF restriction node within a graph.
    The function analyzes the `rnode` (typically a node representing an OWL Restriction) 
    within the provided RDF `graph`, checking for various predicates and attributes that 
    define qualified or regular cardinality, and generates a token representing the 
    restriction shape.

    The `granularity` parameter modifies the token representation by determining whether 
    to output a coarse or exact restriction format. This allows for fine-tuning the 
    specificity of the restriction token generated. If no valid restriction representation 
    is found, the function returns `None`.

    :param g: 
        An RDF graph representing the data structure containing the OWL ontology.
    :param rnode: 
        An individual RDF node, expected to represent an OWL Restriction within the graph.
    :param granularity: 
        A string indicating the granularity level, such as "coarse" or "exact", to control 
        how restriction tokens should be formatted.
    :return: 
        A string token representing the restriction shape, or None if no such restriction 
        is found or valid.
    """
    if (rnode, RDF.type, OWL.Restriction) not in g:
        return None

    def emit(key: str, n: Optional[int]) -> str:
        """
        Generates a restriction token string based on the provided key and optional number.

        :param key: A string representing the restriction type (e.g. 'qmin', 'qmax', 'qcard')
        :type key: str
        :param n: An optional integer value associated with the restriction
        :type n: Optional[int]
        :return: A formatted restriction token string in the form "R:key" or "R:key=n"
        :rtype: str

        If granularity is "coarse", qualified keys are converted to unqualified:
        - 'qmin' becomes 'min'
        - 'qmax' becomes 'max'
        - 'qcard' becomes 'card'

        If granularity is "exact" and n is not None, the number is appended:
        "R:key=n"

        Otherwise returns just "R:key"
        """

        if granularity == "coarse":
            key = key.replace("qmin","min").replace("qmax","max").replace("qcard","card")
        if granularity == "exact" and n is not None:
            return f"R:{key}={n}"
        return f"R:{key}"

    for pred, key in ((OWL.qualifiedCardinality, "qcard"),
                      (OWL.minQualifiedCardinality, "qmin"),
                      (OWL.maxQualifiedCardinality, "qmax")):
        n = g.value(rnode, pred); on_cls = g.value(rnode, OWL.onClass)
        if n is not None and on_cls is not None:
            return emit(key, _as_int(n))

    for pred, key in ((OWL.cardinality, "card"),
                      (OWL.minCardinality, "min"),
                      (OWL.maxCardinality, "max")):
        n = g.value(rnode, pred)
        if n is not None:
            return emit(key, _as_int(n))

    for pred, key in ((OWL.someValuesFrom, "some"),
                      (OWL.allValuesFrom,  "only"),
                      (OWL.hasValue,       "has")):
        if g.value(rnode, pred) is not None:
            return f"R:{key}"
    return None

def collect_shape_tokens(g: Graph, expr, granularity: str) -> Counter:
    """
    Collect shape tokens from the provided RDF graph expression.

    This function traverses an RDF graph and collects shape tokens based on the 
    given expression and granularity level. It identifies and processes specific 
    OWL constructs such as restrictions, intersections, and unions.

    :param g: The RDF graph to be analyzed.
    :type g: Graph
    :param expr: The expression node within the graph to process.
    :type expr: Any
    :param granularity: The level of granularity to be used while generating 
        shape tokens.
    :type granularity: str
    :return: A Counter object containing the collected shape tokens and their 
        frequencies.
    :rtype: Counter
    """
    C = Counter()
    if isinstance(expr, URIRef):
        return C
    if (expr, RDF.type, OWL.Restriction) in g:
        t = restriction_shape_token(g, expr, granularity)
        if t:
            C[t] += 1
        return C
    
    head = g.value(expr, OWL.intersectionOf)
    if head is not None:
        for m in iter_rdf_list(g, head):
            C += collect_shape_tokens(g, m, granularity)
        return C
    
    head = g.value(expr, OWL.unionOf)
    if head is not None:
        for m in iter_rdf_list(g, head):
            C += collect_shape_tokens(g, m, granularity)
        return C
    return C

def class_shape_signature(g: Graph, cls: URIRef, granularity: str) -> Counter:
    """
    Generate a signature for a given class in a graph based on shape tokens.

    This function computes a signature Counter by aggregating tokens collected
    from the class's superclasses and equivalent classes in the given graph
    based on the specified granularity.

    :param g: The RDF graph containing the ontology.
    :type g: Graph
    :param cls: The URI reference of the class for which the signature is
        generated.
    :type cls: URIRef
    :param granularity: The granularity level used for token collection
        (e.g., coarse or fine-grained identifiers).
    :type granularity: str
    :return: A Counter object representing the aggregated shape tokens
        for the given class.
    :rtype: Counter
    """
    sig = Counter()
    for sup in g.objects(cls, RDFS.subClassOf):
        sig += collect_shape_tokens(g, sup, granularity)
    for eq in g.objects(cls, OWL.equivalentClass):
        sig += collect_shape_tokens(g, eq, granularity)
    return sig

def classes_with_shape(g: Graph, granularity: str) -> Dict[URIRef, Counter]:
    """
    Identifies classes within the provided RDF graph and computes their shape
    signatures based on the specified granularity. Classes include those explicitly
    defined as RDF/OWL/RDFS classes or those involved in subclass or equivalent
    class relationships. The method computes a dictionary mapping each identified
    class URI to its shape signature represented as a counter.

    :param g: RDF Graph from which classes are identified
    :type g: Graph
    :param granularity: Level of detail for the class shape signature computation
    :type granularity: str
    :return: Dictionary mapping class URIs to their corresponding shape signatures
    :rtype: Dict[URIRef, Counter]
    """
    classes = set()
    classes.update(s for s in g.subjects(RDF.type, OWL.Class))
    classes.update(s for s in g.subjects(RDF.type, RDFS.Class))
    classes.update(s for s, _, _ in g.triples((None, RDFS.subClassOf, None)))
    classes.update(s for s, _, _ in g.triples((None, OWL.equivalentClass, None)))

    out: Dict[URIRef, Counter] = {}
    for c in classes:
        if not isinstance(c, URIRef):
            continue
        sig = class_shape_signature(g, c, granularity)
        if sig:
            out[c] = sig
    return out

# ---------- Normalization ----------
NUM_RE = re.compile(r"^R:(?P<k>[a-z]+)(?:=(?P<n>\d+))?$")

def parse_tok(tok: str) -> Tuple[str, Optional[int]]:
    """
    Parses a token to extract a string component and an optional integer component.

    The function matches the given token against a predefined regular expression
    (NUM_RE). If the token matches, it extracts the string part (key) and an optional
    numeric part. If the token does not match, it returns the token itself along with
    a `None` value.

    :param tok: A string token to be parsed.
    :type tok: str
    :return: A tuple where the first element is the string part of the token and the
        second element is the optional integer part, or `None` if not present.
    :rtype: Tuple[str, Optional[int]]
    """
    m = NUM_RE.match(tok)
    if not m:
        return tok, None
    k = m.group("k"); n = m.group("n")
    return k, (int(n) if n is not None else None)

def closure_entailment(sig: Counter) -> Counter:
    """
    Compute the closure entailment of a given signature counter.

    A closure entailment calculates the extended set of derived tokens and their
    counts based on specific rules applied to the tokens already defined in the
    input signature counter. Each token type in the input signature determines 
    further tokens that should be added or adjusted in the resulting output counter.

    :param sig: Counter object representing the signature tokens and their counts.
                The keys represent specific tokens (e.g., "has", "card", etc.), and
                the values represent their respective integer counts.
    :return: A new Counter object containing the extended set of tokens and their
             counts after applying the closure entailment rules.
    """
    out = sig.copy()
    def add(t,k=1): out[t] += k
    for tok, cnt in list(sig.items()):
        k, n = parse_tok(tok)
        if k == "has":
            add("R:some", cnt)
        if k == "card" and n is not None:
            if n >= 1: add("R:some", cnt)
            add(f"R:min={n}", cnt); add(f"R:max={n}", cnt)
        if k == "qcard" and n is not None:
            if n >= 1: add("R:some", cnt)
            add(f"R:qmin={n}", cnt); add(f"R:qmax={n}", cnt)
        if k in ("min","qmin") and n is not None and n >= 1:
            add("R:some", cnt)
    return out

def normalize_families(sig: Counter) -> Counter:
    """
    Normalize the families in a given signature represented as a Counter. This
    function processes the tokens and their corresponding counts in the input
    Counter object to classify and tally them into different family categories,
    returning an updated Counter with the categorized families.

    :param sig: Counter object representing tokens and their counts, where the key
        is a string token and the value is an integer count.
    :return: Counter object with the count of tokens classified into specific
        family categories.
    """
    fam = Counter()
    for tok, cnt in sig.items():
        k, n = parse_tok(tok)
        if k == "only":
            fam["F:U"] += 1
        elif k in ("some","has") or (k in ("min","qmin") and (n is None or n >= 1)) or (k in ("card","qcard") and (n is None or n >= 1)):
            fam["F:E"] += 1
        elif k in ("min","qmin"):
            fam["F:MIN"] += 1
        elif k in ("max","qmax"):
            fam["F:MAX"] += 1
        elif k in ("card","qcard"):
            fam["F:EXACT"] += 1
        else:
            fam["F:OTHER"] += 1
    return fam

def apply_normalization(sig: Counter, mode: str) -> Counter:
    """
    Apply normalization to the given signature based on the specified mode.

    This function modifies the input signature according to the provided mode.
    It supports different normalization techniques such as entailment closure
    and family-based normalization. If the mode is set to "off", the input
    signature is returned as-is without modification.

    :param sig: A Counter object representing the input signature to be 
        normalized.
    :param mode: A string specifying the normalization mode. Supported modes 
        include "off" (no normalization), "entailment" (closure entailment), 
        and "families" (family-based normalization).
    :return: A Counter object representing the normalized signature.
    """
    if mode == "off":
        return sig
    if mode == "entailment":
        return closure_entailment(sig)
    if mode == "families":
        return normalize_families(sig)
    return sig

# ---------- Rendering (for XLSX) ----------
def render_expr(g: Graph, expr) -> str:
    if isinstance(expr, URIRef):
        return name_qname(g, expr)
    if (expr, RDF.type, OWL.Restriction) in g:
        p = g.value(expr, OWL.onProperty)
        prop = name_qname(g, p) if isinstance(p, URIRef) else "?:prop"
        for pred, word in ((OWL.qualifiedCardinality, "exactly"),
                           (OWL.minQualifiedCardinality, "min"),
                           (OWL.maxQualifiedCardinality, "max")):
            n = g.value(expr, pred); on_cls = g.value(expr, OWL.onClass)
            if n is not None and on_cls is not None:
                n_int = _as_int(n); fill = render_expr(g, on_cls)
                return f"{prop} {word} {n_int} {fill}"
        for pred, word in ((OWL.cardinality, "exactly"),
                           (OWL.minCardinality, "min"),
                           (OWL.maxCardinality, "max")):
            n = g.value(expr, pred)
            if n is not None:
                n_int = _as_int(n)
                return f"{prop} {word} {n_int}"
        for pred, word in ((OWL.someValuesFrom, "some"),
                           (OWL.allValuesFrom,  "only"),
                           (OWL.hasValue,       "value")):
            v = g.value(expr, pred)
            if v is not None:
                if isinstance(v, (URIRef, BNode)):
                    return f"{prop} {word} {render_expr(g, v)}"
                return f"{prop} {word} {str(v)}"
        return f"{prop} [unknown restriction]"
    head = g.value(expr, OWL.intersectionOf)
    if head is not None:
        parts = [render_expr(g, m) for m in iter_rdf_list(g, head)]
        return "(" + " AND ".join(parts) + ")"
    head = g.value(expr, OWL.unionOf)
    if head is not None:
        parts = [render_expr(g, m) for m in iter_rdf_list(g, head)]
        return "(" + " OR ".join(parts) + ")"
    if isinstance(expr, BNode):
        return "(anonymous)"
    return str(expr)

def axioms_for_class(g: Graph, cls: URIRef) -> List[str]:
    """
    Generate a list of axioms related to a given class in an RDF graph.

    This function processes an RDF graph to extract axioms for a specific class. It retrieves
    all superclasses and equivalent classes of the provided class resource and formats them
    into strings that represent these axioms. Each axiom is created based on the relationships
    using RDFS and OWL vocabulary.

    :param g: An RDFLib graph instance representing the RDF data where the axioms are to
        be processed.
    :type g: Graph
    :param cls: A URI reference representing the class in the RDF graph for which the
        axioms are generated.
    :type cls: URIRef
    :return: A list of strings, where each string represents either a subclass or equivalence
        axiom for the provided class in the graph.
    :rtype: List[str]
    """
    out: List[str] = []
    for sup in g.objects(cls, RDFS.subClassOf):
        out.append("SubClassOf: {}".format(render_expr(g, sup)))
    for eq in g.objects(cls, OWL.equivalentClass):
        out.append("EquivalentTo: {}".format(render_expr(g, eq)))
    return out

# ---------- Parsing ----------
def parse_graph(path: Path, follow_imports: bool, depth: int) -> Graph:
    """
    Parses an RDF graph from the specified file path and optionally follows
    and parses imported RDF graphs up to a specified depth. The function
    produces a combined graph containing the original RDF content and any
    imported content up to the specified depth.

    :param path: Path to the RDF file to be parsed
    :type path: Path
    :param follow_imports: Indicates whether to follow imports and parse
        imported RDF documents
    :type follow_imports: bool
    :param depth: Depth up to which imported documents should be parsed.
        A depth of 0 indicates that only the given RDF file will be parsed,
        without following any imports
    :type depth: int
    :return: Parsed RDF graph with content from the specified file and
        optionally its imports up to the specified depth
    :rtype: Graph
    """
    g = Graph(); g.parse(path.as_posix())
    if follow_imports and depth > 0:
        from rdflib import URIRef
        frontier = [o for o in g.objects(None, OWL.imports) if isinstance(o, URIRef)]
        seen = set(); d = 0
        while frontier and d < depth:
            nxt = []
            for iri in frontier:
                if iri in seen: continue
                seen.add(iri)
                try:
                    g.parse(str(iri))
                    nxt += [o for o in g.objects(None, OWL.imports) if isinstance(o, URIRef)]
                except Exception:
                    pass
            frontier = nxt; d += 1
    return g

# ---------- Keys & Strings ----------
def key_for(sig: Counter, presence_only: bool) -> Tuple[Tuple[str,int], ...]:
    """
    Generate a sorted and processed tuple key from a Counter object.

    This function takes a Counter object and converts it into a tuple of sorted
    key-value pairs. Each value in the Counter is transformed based on the
    `presence_only` flag. When `presence_only` is True, all values are replaced
    with 1 to indicate only the presence of keys. Otherwise, the original
    count values are kept. The generated tuple provides a consistent and
    ordered representation of the Counter.

    :param sig: A Counter object containing key-value pairs, where the keys are
        strings and the values are integers.
    :param presence_only: A boolean indicating whether to retain only the presence
        of keys (True) or preserve their count values (False).
    :return: A tuple consisting of sorted key-value pairs, where each pair is a
        tuple of a string and an integer. Sorting ensures the order is consistent.
    :rtype: Tuple[Tuple[str, int], ...]
    """
    result = tuple(sorted((k, (1 if presence_only else v)) for k, v in sig.items()))
    return result

def shape_str(sig: Counter) -> str:
    """
    Generate a formatted string representation of a `Counter` object.

    The function takes a `Counter` object and returns a string where each key-value
    pair is formatted as "key×value" if the value is greater than 1 or simply as
    "key" if the value is 1. The pairs are sorted by the keys and separated by a
    semicolon.

    :param sig: A `Counter` object where keys are elements and values are their
        respective counts.
    :type sig: Counter
    :return: A formatted string representation of the `Counter`, with counts and
        keys displayed appropriately based on their values.
    :rtype: str
    """
    result =  "; ".join(f"{k}×{v}" if v != 1 else k for k, v in sorted(sig.items()))
    return result

# ---------- Main ----------
def run_pair(left: Path, right: Path, outdir: Path,
             follow_imports: bool, depth: int,
             granularity: str, presence_only: bool, normalize_mode: str):
    """
    Executes pairwise comparison between two ontology graphs and prepares structured output with matches.

    This function processes two ontology files, extracts relevant information about their classes,
    applies normalization, and identifies matching classes based on specified criteria. The results
    are stored in an Excel file within the given output directory. Comparison considers class shapes,
    axioms, and labels with options to control normalization and import resolution depth.

    :param left: The path to the first ontology file.
    :type left: Path
    :param right: The path to the second ontology file.
    :type right: Path
    :param outdir: The directory where the output Excel file will be saved.
    :type outdir: Path
    :param follow_imports: Whether to follow and include imported ontologies during graph parsing.
    :type follow_imports: bool
    :param depth: The depth limit for traversing the ontology graph.
    :type depth: int
    :param granularity: The level of detail for class inclusion (e.g., fine-grained or coarse-grained).
    :type granularity: str
    :param presence_only: If True, matches are determined based solely on the presence of classes.
    :type presence_only: bool
    :param normalize_mode: The mode used for applying normalization to class signatures.
    :type normalize_mode: str
    :return: None
    """
    gL = parse_graph(left,  follow_imports, depth)
    gR = parse_graph(right, follow_imports, depth)

    rawL = classes_with_shape(gL, granularity)
    rawR = classes_with_shape(gR, granularity)

    normL = {c: apply_normalization(sig, normalize_mode) for c, sig in rawL.items()}
    normR = {c: apply_normalization(sig, normalize_mode) for c, sig in rawR.items()}

    L_keys = {}
    for c, s in normL.items():
        L_keys.setdefault(key_for(s, presence_only), []).append(c)

    invR = {}
    for c, s in normR.items():
        invR.setdefault(key_for(s, presence_only), []).append(c)

    rows: List[Dict[str, str]] = []
    for k, left_classes in L_keys.items():
        right_classes = invR.get(k, [])
        if not right_classes:
            continue
        for cL in left_classes:
            for cR in right_classes:
                rows.append({
                    "left_iri": str(cL),
                    "left_label": best_label(gL, cL),
                    "left_shape": shape_str(rawL[cL]),
                    "left_axioms": " | ".join(axioms_for_class(gL, cL)),
                    "right_iri": str(cR),
                    "right_label": best_label(gR, cR),
                    "right_shape": shape_str(rawR[cR]),
                    "right_axioms": " | ".join(axioms_for_class(gR, cR)),
                })

    out = outdir / (left.stem + "-" + right.stem + "-structural-matches.xlsx")
    pd.DataFrame(rows, columns=[
        "left_iri","left_label","left_shape","left_axioms",
        "right_iri","right_label","right_shape","right_axioms"
    ]).to_excel(out.as_posix(), index=False)
    print(f"[OK] {out.name}: {len(rows)} pairs")

def main():
    """
    Parse command-line arguments and execute the ontology matching process.

    This function sets up an argument parser with several options for configuring the
    ontology matching, parses the provided arguments, and then invokes the `run_pair`
    function to process the specified ontologies and output the results to the given
    directory.

    :arg --left: Path to the left ontology file to be compared.
    :arg --right: Path to the right ontology file to be compared.
    :arg --outdir: Path to the output directory for saving results.
    :arg --shape: Matching granularity to use. Can be one of "exact", "kind", or "coarse".
    :arg --normalize: Normalization mode for the match results. Can be one of "off",
        "families", or "entailment".
    :arg --presence-only: Match based only on presence, without structural details.
    :arg --follow-imports: Whether to resolve and follow imports in the ontologies.
    :arg --imports-depth: Number of levels of imports to follow if imports are enabled.

    :param --left: Path to the left ontology file
    :type --left: Path
    :param --right: Path to the right ontology file
    :type --right: Path
    :param --outdir: Path to the output directory
    :type --outdir: Path
    :param --shape: Matching granularity option
    :type --shape: str
    :param --normalize: Normalization mode option
    :type --normalize: str
    :param --presence-only: Flag to enable matching based on presence only
    :type --presence-only: bool
    :param --follow-imports: Flag to enable following imports in the ontologies
    :type --follow-imports: bool
    :param --imports-depth: Depth of imports to follow
    :type --imports-depth: int

    :return: None
    """
    ap = argparse.ArgumentParser(description="Save ONLY the structural matches across two ontologies.")
    ap.add_argument("--left",  type=Path, required=True)
    ap.add_argument("--right", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("."))

    ap.add_argument("--shape", choices=["exact","kind","coarse"], default="exact")
    ap.add_argument("--normalize", choices=["off","families","entailment"], default="off")
    ap.add_argument("--presence-only", action="store_true")
    ap.add_argument("--follow-imports", action="store_true")
    ap.add_argument("--imports-depth", type=int, default=2)

    args = ap.parse_args()
    run_pair(args.left, args.right, args.outdir,
             args.follow_imports, args.imports_depth,
             args.shape, args.presence_only, args.normalize)

if __name__ == "__main__":
    main()
