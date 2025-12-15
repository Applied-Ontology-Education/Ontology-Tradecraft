import logging
import argparse

from owlready2 import get_ontology, sync_reasoner, ThingClass, Ontology
from owlready2.prop import ObjectPropertyClass, DataPropertyClass
from rdflib import Graph

from common.settings import build_settings
from util.logger_config import config

logger = logging.getLogger(__name__)
from pathlib import Path

config(logger)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


def get_entity_by_iri(onto, iri: str):
    ent = onto.world[iri]
    if ent is None:
        logger.error(f"Entity not found for IRI: {iri}")
        return None
    return ent


def class_in_domain_of_property(
    onto,
    class_iri: str,
    prop_iri: str,
):
    """Return (bool, matched_expressions) indicating whether the class
    (by IRI) is in the (inferred) domain of the property (by IRI),
    considering EL-compatible expressions (named classes, intersections, someValuesFrom).
    """
    cls = get_entity_by_iri(onto, class_iri)
    prop = get_entity_by_iri(onto, prop_iri)

    if not cls or not prop:
        return False, []

    if not isinstance(cls, ThingClass):
        raise TypeError(f"{class_iri} is not an OWL class")
    if not isinstance(prop, (ObjectPropertyClass, DataPropertyClass)):
        raise TypeError(f"{prop_iri} is not an OWL property")

    matched = []
    for expr in prop.domain:
        # Let the reasoner handle complex EL expressions:
        # cls ⊑ expr ?
        try:
            if issubclass(cls, expr):
                matched.append(expr)
        except Exception:
            # Some domain entries may not be class expressions Owlready2 can compare against
            continue

    return bool(matched), matched


def class_in_range_of_property(
    onto,
    class_iri: str,
    prop_iri: str,
):
    """Return (bool, matched_expressions) for range membership.

    Note: Only meaningful for ObjectProperty ranges (class expressions). For DataProperty,
    ranges are datatypes; this function will raise a TypeError for data properties.
    """
    cls = get_entity_by_iri(onto, class_iri)
    prop = get_entity_by_iri(onto, prop_iri)

    if not cls or not prop:
        return False, []

    if not isinstance(cls, ThingClass):
        raise TypeError(f"{class_iri} is not an OWL class")

    if isinstance(prop, ObjectPropertyClass):
        matched = []
        for expr in prop.range:
            try:
                if issubclass(cls, expr):
                    matched.append(expr)
            except Exception:
                continue
        return bool(matched), matched
    elif isinstance(prop, DataPropertyClass):
        raise TypeError(
            "Range membership for DataProperty compares datatypes, not classes; provide an ObjectProperty."
        )
    else:
        raise TypeError(f"{prop_iri} is not an OWL property")


def _load_ontology_any(ontology_path: Path):
    """Try to load ontology directly; if TTL is not supported by Owlready2 env,
    convert TTL->RDF/XML via rdflib and load that as a fallback.
    """
    uri = ontology_path.resolve().as_uri()
    try:
        return get_ontology(uri).load()
    except Exception as e:
        logger.info("Direct load failed (%s). Falling back to TTL->RDF/XML conversion…", e)
        # Fallback convert
        rdfxml_path = ontology_path.with_suffix(".owl")
        g = Graph()
        # Try to infer format by suffix; default to turtle if .ttl
        fmt = "turtle" if ontology_path.suffix.lower() in {".ttl", ".turtle"} else None
        g.parse(uri, format=fmt)
        g.serialize(rdfxml_path.as_posix(), format="xml")
        return get_ontology(rdfxml_path.resolve().as_uri()).load()


def check_domain_range(onto: Ontology, class_iri: str, prop_iri: str, which: str) -> bool:
    in_dom = False
    in_rng = False

    ci = get_entity_by_iri(onto, class_iri)
    pi = get_entity_by_iri(onto, prop_iri)

    if not ci or not isinstance(ci, ThingClass):
        return False
    if not pi or not isinstance(pi, (ObjectPropertyClass, DataPropertyClass)):
        return False

    if which in ("domain", "both"):
        in_dom, dom_exprs = class_in_domain_of_property(onto, class_iri, prop_iri)
        logger.info("In domain? %s", in_dom)
        for e in dom_exprs:
            logger.info("  matched domain expression: %s", e)

    if which in ("range", "both"):
        try:
            in_rng, rng_exprs = class_in_range_of_property(onto, class_iri, prop_iri)
            logger.info("In range? %s", in_rng)
            for e in rng_exprs:
                logger.info("  matched range expression: %s", e)
        except TypeError as te:
            logger.info("Range check skipped: %s", te)

    if which == "both":
        return in_dom and in_rng
    elif which == "domain":
        return in_dom
    else:
        return in_rng


def parse_args(settings: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check if a class is in the domain/range of a property")
    p.add_argument(
        "--ontology",
        default=str(settings.get("reference_ontology", "")),
        help="Path to ontology (TTL/OWL). Defaults to settings.reference_ontology",
    )
    p.add_argument("--class-iri", required=True, help="Class IRI to test")
    p.add_argument("--prop-iri", required=True, help="Property IRI to test")
    p.add_argument(
        "--which",
        choices=["domain", "range", "both"],
        default="both",
        help="Which check to perform",
    )
    return p.parse_args()


def main(
    ontology: str,
    class_iri: str,
    prop_iri: str,
    which: str = "both",
) -> bool:
    onto_path = Path(ontology) if ontology else None
    if onto_path is None or not onto_path.exists():
        raise FileNotFoundError(
            f"Ontology not found: {onto_path}. Provide --ontology or set reference_ontology in settings."
        )

    onto = _load_ontology_any(onto_path)

    # Classify once
    with onto:
        # Owlready2 typically uses HermiT-like reasoner through Java bridge
        sync_reasoner()

    result = check_domain_range(onto, class_iri, prop_iri, which)

    print(f"Class {class_iri} is {'in' if result else 'not in'} [domain|range] {which} of {prop_iri}.")
    return result


if __name__ == "__main__":
    settings = build_settings(PROJECT_ROOT, DATA_ROOT)
    args = parse_args(settings)
    main(args.ontology, args.class_iri, args.prop_iri, args.which)
