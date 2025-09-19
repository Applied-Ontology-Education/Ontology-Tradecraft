#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pair classes across two ontologies that share the SAME structural type of axioms.
Writes ONLY the matches file: <left>-<right>-structural-matches.xlsx

Structure = multiset of restriction tokens from SubClassOf / EquivalentClass:
  R:some, R:only, R:has, R:(q)min(=n), R:(q)max(=n), R:(q)card(=n)
Ignores properties, fillers, and boolean grouping (AND/OR flattened).

--shape exact|kind|coarse
  exact  : keep numbers & qualifiedness (qmin vs min)
  kind   : drop numbers, keep qualifiedness
  coarse : drop numbers & collapse qualifiedness (qmin→min, qcard→card)
--normalize off|families|entailment
  families   : collapse to {E,U,MIN,MAX,EXACT} families
  entailment : tiny closure (card n ⇒ min n + max n; qcard n ⇒ qmin n + qmax n; min≥1/qmin≥1/has ⇒ some)
--presence-only : ignore multiplicities when comparing structures
--follow-imports / --imports-depth : pull in owl:imports
"""

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
    while head and head != RDF.nil:
        first = g.value(head, RDF.first)
        if first is not None:
            yield first
        head = g.value(head, RDF.rest)

# ---------- Labels ----------
def best_label(g: Graph, n: URIRef) -> str:
    for p in (RDFS.label, SKOS.prefLabel):
        o = g.value(n, p)
        if isinstance(o, Literal):
            return str(o)
    s = str(n)
    return s.rsplit("#", 1)[-1].rsplit("/", 1)[-1]

def name_qname(g: Graph, n: URIRef) -> str:
    try:
        return g.namespace_manager.qname(n)
    except Exception:
        return str(n)

# ---------- Shape extraction ----------
def _as_int(n) -> Optional[int]:
    try:
        if isinstance(n, Literal):
            return int(n.toPython())
        return int(str(n))
    except Exception:
        return None

def restriction_shape_token(g: Graph, rnode, granularity: str) -> Optional[str]:
    if (rnode, RDF.type, OWL.Restriction) not in g:
        return None

    def emit(key: str, n: Optional[int]) -> str:
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
    sig = Counter()
    for sup in g.objects(cls, RDFS.subClassOf):
        sig += collect_shape_tokens(g, sup, granularity)
    for eq in g.objects(cls, OWL.equivalentClass):
        sig += collect_shape_tokens(g, eq, granularity)
    return sig

def classes_with_shape(g: Graph, granularity: str) -> Dict[URIRef, Counter]:
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
    m = NUM_RE.match(tok)
    if not m:
        return tok, None
    k = m.group("k"); n = m.group("n")
    return k, (int(n) if n is not None else None)

def closure_entailment(sig: Counter) -> Counter:
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
    out: List[str] = []
    for sup in g.objects(cls, RDFS.subClassOf):
        out.append("SubClassOf: {}".format(render_expr(g, sup)))
    for eq in g.objects(cls, OWL.equivalentClass):
        out.append("EquivalentTo: {}".format(render_expr(g, eq)))
    return out

# ---------- Parsing ----------
def parse_graph(path: Path, follow_imports: bool, depth: int) -> Graph:
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
    return tuple(sorted((k, (1 if presence_only else v)) for k, v in sig.items()))

def shape_str(sig: Counter) -> str:
    return "; ".join(f"{k}×{v}" if v != 1 else k for k, v in sorted(sig.items()))

# ---------- Main ----------
def run_pair(left: Path, right: Path, outdir: Path,
             follow_imports: bool, depth: int,
             granularity: str, presence_only: bool, normalize_mode: str):

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
