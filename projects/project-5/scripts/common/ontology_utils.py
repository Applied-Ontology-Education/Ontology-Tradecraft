"""
Ontology utility helpers.

Provides functions to retrieve the immediate parent (genus) of a class or
property in a Turtle ontology, by the element's rdfs:label.

Usage example (programmatic):

    from scripts.common.ontology_utils import (
        get_parent_class_by_label,
        get_parent_property_by_label,
    )

    result1 = get_parent_class_by_label(
        label="Information Content Entity",
        ttl_path="src/CommonCoreOntologiesMerged.ttl",
    )
    result2 = get_parent_property_by_label(
        label="is about",
        ttl_path="src/CommonCoreOntologiesMerged.ttl",
    )
    print(result1, result2)

CLI usage:

    python -m scripts.common.ontology_utils --label "Information Content Entity" \
        --type class --ttl src/CommonCoreOntologiesMerged.ttl

Notes:
- The function prefers exact label matches. If no exact match is found, it
  falls back to a case-insensitive match.
- If multiple matches exist, the first is returned deterministically by IRI
  sort order.

Compatibility:
- The original get_parent_by_label(label, element_type, ttl_path) remains
  available and now delegates to the two specialized functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Dict, Any

from rdflib import Graph, URIRef, BNode

ElementType = Literal["class", "property"]


@dataclass
class ParentResult:
    child_iri: str
    child_label: Optional[str]
    parent_iri: Optional[str]
    parent_label: Optional[str]
    # Optional domain/range details (primarily for properties)
    child_domain: Optional[str] = None
    child_range: Optional[str] = None
    parent_domain: Optional[str] = None
    parent_range: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "child_iri": self.child_iri,
            "child_label": self.child_label,
            "parent_iri": self.parent_iri,
            "parent_label": self.parent_label,
            "child_domain": self.child_domain,
            "child_range": self.child_range,
            "parent_domain": self.parent_domain,
            "parent_range": self.parent_range,
        }


# Simple in-process cache to avoid reparsing the TTL on every call
_GRAPH_CACHE: dict[str, Graph] = {}


def _load_graph(ttl_path: Path) -> Graph:
    key = str(ttl_path.resolve())
    g = _GRAPH_CACHE.get(key)
    if g is None:
        g = Graph()
        g.parse(str(ttl_path), format="turtle")
        _GRAPH_CACHE[key] = g
    return g


def _resolve_child_by_label(g: Graph, label: str) -> tuple[Optional[str], Optional[str]]:
    """Resolve a resource by its rdfs:label. Returns (iri, label) or (None, None)."""
    prefixes = (
        "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
    )

    def run(lbl: str, case_insensitive: bool = False):
        label_filter = (
            "FILTER(LCASE(STR(?lbl)) = LCASE(STR(?q)))" if case_insensitive else "FILTER(STR(?lbl) = STR(?q))"
        )
        query = f"""
        {prefixes}
        SELECT ?s ?lbl WHERE {{
            BIND({_sparql_escape(lbl)} AS ?q)
            ?s rdfs:label ?lbl .
            {label_filter}
        }} ORDER BY STR(?s) LIMIT 1
        """
        return list(g.query(query))

    rows = run(label, case_insensitive=False) or run(label, case_insensitive=True)
    if not rows:
        return None, None
    s = str(rows[0][0]) if rows[0][0] is not None else None
    lbl = str(rows[0][1]) if rows[0][1] is not None else None
    return s, lbl


def get_parent_class_by_label(label: str, ttl_path: str | Path) -> Dict[str, Any]:
    """
    Retrieve the immediate parent class (via rdfs:subClassOf) of the class
    identified by the given rdfs:label.

    Args:
        label: rdfs:label of the class to look up.
        ttl_path: Path to the Turtle ontology file.

    Returns:
        Dict with keys child_iri, child_label, parent_iri, parent_label.
    """
    ttl_path = Path(ttl_path)
    if not ttl_path.exists():
        raise FileNotFoundError(f"TTL file not found: {ttl_path}")

    g = _load_graph(ttl_path)

    child_iri, child_label = _resolve_child_by_label(g, label)
    parent_iri = None
    parent_label = None

    if child_iri is not None:
        prefixes = (
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
            "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
        )
        query = f"""
        {prefixes}
        SELECT ?parent ?plbl WHERE {{
            BIND(IRI({_sparql_escape(child_iri)}) AS ?s)
            ?s rdfs:subClassOf ?parent .
            FILTER(isIRI(?parent))
            ?parent rdfs:label ?plbl .
        }} ORDER BY STR(?parent) LIMIT 1
        """
        rows = list(g.query(query))
        if rows:
            parent_iri = str(rows[0][0]) if rows[0][0] is not None else None
            parent_label = str(rows[0][1]) if rows[0][1] is not None else None

    return ParentResult(
        child_iri=child_iri,
        child_label=child_label,
        parent_iri=parent_iri,
        parent_label=parent_label,
    ).to_dict()


def get_parent_property_by_label(label: str, ttl_path: str | Path) -> Dict[str, Any]:
    """
    Retrieve the immediate parent property (via rdfs:subPropertyOf) of the
    property identified by the given rdfs:label.

    Args:
        label: rdfs:label of the property to look up.
        ttl_path: Path to the Turtle ontology file.

    Returns:
        Dict with keys:
          - child_iri, child_label, parent_iri, parent_label
          - child_domain, child_range (labels if present, otherwise None)
          - parent_domain, parent_range (labels if present and a parent exists, otherwise None)
    """
    ttl_path = Path(ttl_path)
    if not ttl_path.exists():
        raise FileNotFoundError(f"TTL file not found: {ttl_path}")

    g = _load_graph(ttl_path)

    child_iri, child_label = _resolve_child_by_label(g, label)
    parent_iri = None
    parent_label = None
    child_domain = None
    child_range = None
    parent_domain = None
    parent_range = None

    if child_iri is not None:
        prefixes = (
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
            "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
        )
        query = f"""
        {prefixes}
        SELECT ?parent ?plbl WHERE {{
            BIND(IRI({_sparql_escape(child_iri)}) AS ?s)
            ?s rdfs:subPropertyOf ?parent .
            FILTER(isIRI(?parent))
            ?parent rdfs:label ?plbl .
        }} ORDER BY STR(?parent) LIMIT 1
        """
        rows = list(g.query(query))
        if rows:
            parent_iri = str(rows[0][0]) if rows[0][0] is not None else None
            parent_label = str(rows[0][1]) if rows[0][1] is not None else None

        # Retrieve child domain/range labels (only for IRI parents with labels)
        q_child_dr = f"""
        {prefixes}
        SELECT ?dl ?rl WHERE {{
            BIND(IRI({_sparql_escape(child_iri)}) AS ?s)
            OPTIONAL {{
                ?s rdfs:domain ?d . FILTER(isIRI(?d))
                ?d rdfs:label ?dl .
            }}
            OPTIONAL {{
                ?s rdfs:range ?r . FILTER(isIRI(?r))
                ?r rdfs:label ?rl .
            }}
        }} LIMIT 1
        """
        dr_rows = list(g.query(q_child_dr))
        if dr_rows:
            dl = dr_rows[0][0]
            rl = dr_rows[0][1]
            child_domain = str(dl) if dl is not None else None
            child_range = str(rl) if rl is not None else None

        # If we have a parent property, also retrieve its domain/range labels
        if parent_iri:
            q_parent_dr = f"""
            {prefixes}
            SELECT ?dl ?rl WHERE {{
                BIND(IRI({_sparql_escape(parent_iri)}) AS ?p)
                OPTIONAL {{
                    ?p rdfs:domain ?d . FILTER(isIRI(?d))
                    ?d rdfs:label ?dl .
                }}
                OPTIONAL {{
                    ?p rdfs:range ?r . FILTER(isIRI(?r))
                    ?r rdfs:label ?rl .
                }}
            }} LIMIT 1
            """
            pdr_rows = list(g.query(q_parent_dr))
            if pdr_rows:
                dl = pdr_rows[0][0]
                rl = pdr_rows[0][1]
                parent_domain = str(dl) if dl is not None else None
                parent_range = str(rl) if rl is not None else None

    return ParentResult(
        child_iri=child_iri,
        child_label=child_label,
        parent_iri=parent_iri,
        parent_label=parent_label,
        child_domain=child_domain,
        child_range=child_range,
        parent_domain=parent_domain,
        parent_range=parent_range,
    ).to_dict()


def get_parent_by_label(label: str, element_type: ElementType, ttl_path: str | Path) -> Dict[str, Any]:
    """
    Backwards-compatible entry point that dispatches to the specialized
    functions for classes and properties.
    """
    etype = element_type.strip().lower()
    if etype == "class":
        return get_parent_class_by_label(label=label, ttl_path=ttl_path)
    if etype == "property":
        return get_parent_property_by_label(label=label, ttl_path=ttl_path)
    raise ValueError("element_type must be either 'class' or 'property'")


def _sparql_escape(s: str) -> str:
    """Escape a Python string for safe embedding as a SPARQL string literal."""
    # Escape backslash and quotes; represent as plain "..."
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def get_turtle_snippet_by_label(
    label: str,
    element_type: ElementType,
    ttl_path: str | Path,
) -> str:
    """
    Return a Turtle snippet (as a string) describing the resource with the given
    rdfs:label from the provided ontology. The element_type must be either
    "class" or "property" (validated for API consistency; it does not change
    the extraction logic).

    The snippet includes all triples where the resolved resource is the subject.
    If any object is a blank node, the blank node subgraph is included
    recursively to keep the snippet self-contained.

    Args:
        label: The rdfs:label to resolve.
        element_type: "class" or "property".
        ttl_path: Path to the Turtle ontology file (e.g., src/ConsolidatedCCO.ttl).

    Returns:
        A Turtle-formatted string containing the snippet.

    Raises:
        FileNotFoundError: If the ttl_path does not exist.
        ValueError: If element_type is invalid or the label cannot be resolved.
    """
    etype = (element_type or "").strip().lower()
    if etype not in {"class", "property"}:
        raise ValueError("element_type must be either 'class' or 'property'")

    ttl_path = Path(ttl_path)
    if not ttl_path.exists():
        raise FileNotFoundError(f"TTL file not found: {ttl_path}")

    g = _load_graph(ttl_path)

    subj_iri, _ = _resolve_child_by_label(g, label)
    if not subj_iri:
        raise ValueError(f"No resource found with rdfs:label '{label}' in {ttl_path}")

    subj = URIRef(subj_iri)

    return get_turtle_snippet_by_uri_ref(g, subj)


def get_turtle_snippet_by_uri_ref(
    uri_ref: URIRef,
    element_type: ElementType,
    ttl_path: str | Path,
) -> str:
    etype = (element_type or "").strip().lower()
    if etype not in {"class", "property"}:
        raise ValueError("element_type must be either 'class' or 'property'")

    ttl_path = Path(ttl_path)
    if not ttl_path.exists():
        raise FileNotFoundError(f"TTL file not found: {ttl_path}")

    g = _load_graph(ttl_path)

    # Build a minimal subgraph while preserving prefixes for nice Turtle output
    sg = Graph()
    for prefix, ns in g.namespaces():
        # Avoid rebinding default None prefixes that rdflib may not accept
        try:
            if prefix is not None:
                sg.bind(prefix, ns)
        except Exception:
            # Best-effort; skip binding issues
            pass

    # Collect subject triples and recursively include blank node closures
    def add_bnode_closure(node: BNode):
        for p, o in g.predicate_objects(node):
            sg.add((node, p, o))
            if isinstance(o, BNode):
                add_bnode_closure(o)

    for p, o in g.predicate_objects(uri_ref):
        sg.add((uri_ref, p, o))
        if isinstance(o, BNode):
            add_bnode_closure(o)

    # Fix bindings
    namespaces = {
        "cco": "https://www.commoncoreontologies.org/",
        "bfo": "http://purl.obolibrary.org/obo/bfo#"
    }
    for k, v in namespaces.items():
        sg.bind(k, v)

    data = sg.serialize(format="turtle")
    if isinstance(data, bytes):
        return data.decode("utf-8")

    # Expand CURIEs in the Turtle snippet using string manipulation only
    data = expand_curies(data, namespaces)

    return str(data)


def expand_curies(ttl_snippet: str, namespaces: dict[str, str]) -> str:
    """Expand CURIEs in a Turtle snippet to full IRIs using string manipulation only.

    Parameters:
    - ttl_snippet: A Turtle snippet that may use CURIEs (e.g., `ex:Foo a rdf:Class .`).
    - namespaces: Mapping of prefix -> namespace IRI (e.g., `{ "ex": "http://example.com/", "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#" }`).

    Returns:
    - A Turtle string where CURIEs using the provided prefixes are replaced with
      full IRIs in angle brackets, without using rdflib.

    Notes and limitations:
    - This is a conservative text-based replacement that avoids changing content
      inside string literals, existing angle-bracket IRIs, comments, and @prefix lines.
    - It handles typical QName local parts comprised of ASCII letters, digits,
      underscore, dash, and dot. More advanced Turtle escapes in local names are
      not supported by this simple expander.
    - @prefix declarations are preserved (not removed), even if no CURIEs remain.
    """
    if not isinstance(ttl_snippet, str):
        raise TypeError("ttl_snippet must be a string of Turtle content")
    if not isinstance(namespaces, dict):
        raise TypeError("namespaces must be a dict mapping prefix -> namespace IRI")

    # Normalize and validate namespaces (simple checks)
    ns: dict[str, str] = {}
    for k, v in namespaces.items():
        if not isinstance(k, str) or not k:
            raise ValueError("Invalid namespace prefix provided")
        if not isinstance(v, str) or not v:
            raise ValueError(f"Invalid namespace IRI for prefix '{k}'")
        ns[k] = v

    s = ttl_snippet
    n = len(s)
    i = 0
    out: list[str] = []

    in_angle = False  # between < and >
    in_comment = False  # after # to end of line (when not in string)
    in_string = False
    string_quote = ''  # ' or "
    string_is_triple = False

    # Track line starts to skip replacements on lines starting with @prefix
    line_start = 0
    skip_line_for_prefix = False

    def is_delim(ch: str) -> bool:
        return ch.isspace() or ch in '()[]{};,.=\t\r\n'

    while i < n:
        ch = s[i]

        # Handle end of line bookkeeping
        if ch == '\n':
            out.append(ch)
            in_comment = False
            line_start = i + 1
            skip_line_for_prefix = False
            i += 1
            continue

        # Comments (only when not in string or angle IRI)
        if not in_string and not in_angle and ch == '#':
            in_comment = True
        if in_comment:
            out.append(ch)
            i += 1
            continue

        # Angle-bracket IRI blocks
        if not in_string and ch == '<':
            in_angle = True
        if in_angle:
            out.append(ch)
            if ch == '>':
                in_angle = False
            i += 1
            continue

        # String literals (support '...', "...", '''...''', """...""")
        if not in_string and ch in ('\'', '"'):
            # Check triple quote
            q = ch
            if i + 2 < n and s[i + 1] == q and s[i + 2] == q:
                in_string = True
                string_quote = q
                string_is_triple = True
                out.append(q + q + q)
                i += 3
                continue
            else:
                in_string = True
                string_quote = q
                string_is_triple = False
                out.append(ch)
                i += 1
                continue
        if in_string:
            out.append(ch)
            if ch == '\\':  # escape next char
                if i + 1 < n:
                    out.append(s[i + 1])
                    i += 2
                    continue
            if string_is_triple:
                if ch == string_quote and i + 2 < n and s[i + 1] == string_quote and s[i + 2] == string_quote:
                    out.append(string_quote)
                    out.append(string_quote)
                    i += 3
                    in_string = False
                    string_is_triple = False
                    string_quote = ''
                else:
                    i += 1
                continue
            else:
                if ch == string_quote:
                    in_string = False
                    string_quote = ''
                i += 1
                continue

        # At this point: not in string, not in angle, not in comment

        # Detect if the current line starts with @prefix to skip replacements on this line
        if not skip_line_for_prefix:
            # read from current line_start to first non-space char
            j = line_start
            while j < n and s[j] in ' \t':
                j += 1
            if j < n and s.startswith('@prefix', j):
                skip_line_for_prefix = True

        if skip_line_for_prefix:
            out.append(ch)
            i += 1
            continue

        # Try to match a CURIE: prefix:local
        # Conditions: prefix known, boundary before prefix, and a local part
        if ch.isalpha():
            # parse prefix
            j = i + 1
            while j < n and (s[j].isalnum() or s[j] in ['_', '-']):
                j += 1
            if j < n and s[j] == ':':
                prefix = s[i:j]
                if prefix in ns:
                    # ensure left boundary
                    prev = out[-1] if out else ''
                    if i == 0 or is_delim(prev):
                        # parse local part
                        k = j + 1
                        if k < n:
                            # local: letters/digits/_ - .
                            m = k
                            while m < n and (s[m].isalnum() or s[m] in ['_', '-', '.']):
                                m += 1
                            if m > k:
                                local = s[k:m]
                                full = f"<{ns[prefix]}{local}>"
                                out.append(full)
                                i = m
                                continue
            # no match; fall through to append single char and advance
        # default: just copy
        out.append(ch)
        i += 1

    return ''.join(out)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Get parent by label from a TTL ontology")
    parser.add_argument("--label", required=True, help="Class or property rdfs:label")
    parser.add_argument("--type", choices=["class", "property"], required=True, help="Element type")
    parser.add_argument("--ttl", required=True, help="Path to TTL file")
    args = parser.parse_args()

    result = get_parent_by_label(args.label, args.type, args.ttl)
    print(json.dumps(result, indent=2))
