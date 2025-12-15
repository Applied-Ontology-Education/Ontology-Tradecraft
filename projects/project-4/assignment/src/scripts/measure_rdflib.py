import base64
import logging
import pathlib
import re
from functools import lru_cache
from hashlib import sha1
from typing import Iterable

import pandas as pd
import rdflib
from pandas import DataFrame
from rdflib import Namespace, URIRef, Literal, Graph
from rdflib import OWL, RDF, RDFS, SKOS, XSD

from bfo import BFO
from cco import CCO
from merge_ontologies import merge_graphs

logger = logging.getLogger(__name__)

# Path settings
SRC_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_SOURCE = SRC_ROOT / "data"
INPUT_PATH = DATA_SOURCE / "readings_normalized.csv"
OUTPUT_PATH = SRC_ROOT / "measure_cco.ttl"

# Namespace settings
DEFAULT_NS = Namespace("http://www.newfoundland.nl/otc/project-4/")
MATERIAL_ARTIFACT_CLASS_NAME = "MaterialArtifact"


def main():
    # Settings
    input_file = INPUT_PATH
    output_file = OUTPUT_PATH
    namespace = DEFAULT_NS
    ns = Namespace(namespace)

    # Read CSV
    df = pd.read_csv(input_file, header=0)

    # Create ontology
    g = Graph()
    create_ontology(g, namespace)

    # Translate data to RDF
    translate_to_rdf(df, g, ns)

    # Write graph to file
    unmerged_file = SRC_ROOT / "measure_cco_unmerged.ttl"
    print(f"Writing plain KG to {unmerged_file}.")
    write_ttl_kg(g, unmerged_file, ns)

    # Merge with CCO
    print(f"Merging plain KG with CCO to {output_file}.")
    ttl_paths = [unmerged_file, SRC_ROOT / "cco_merged.ttl"]
    merged_g = merge_graphs(ttl_paths)

    # Write merged graph to file
    write_ttl_kg(merged_g, output_file, ns)
    print(f"Merged graph saved to {output_file}")


def translate_to_rdf(df: DataFrame, g: Graph, ns: Namespace):
    create_material_artifacts(df, g, ns)
    create_sdc_instances(df, g, ns)
    create_mice_sensor_observations(df, g, ns)


def create_ontology(g: Graph, namespace: str):
    ontology_uri = URIRef(namespace)
    g.add((ontology_uri, RDF.type, OWL.Ontology))
    g.add((ontology_uri, RDFS.label, Literal("Sensor Data Ontology", lang='en')))
    g.add((ontology_uri, RDFS.comment, Literal("An ontology for sensor readings and measurements", lang='en')))
    logger.info("Created ontology term.")


def create_material_artifacts(df: DataFrame, g: Graph, ns: Namespace):
    n = 0
    artifact_ids = df['artifact_id'].unique()
    for artifact_id in artifact_ids:
        n += 1
        material_artifact_q_name = unique_qname(MATERIAL_ARTIFACT_CLASS_NAME, [artifact_id])
        label = f"Material artifact {artifact_id}"
        g.add((ns[material_artifact_q_name], RDF.type, CCO.materialArtifact))
        g.add((ns[material_artifact_q_name], RDFS.label, Literal(label, lang='en')))
    logger.info("Created {n} material artifact instances.")


def create_sdc_instances(df: DataFrame, g: Graph, ns: Namespace):
    # Create SDC class mappings
    sdc_class_mapping = get_sdc_class_mapping(g, ns)

    # Create SDC instances
    sdc_instances = df[['sdc_kind', 'artifact_id']].drop_duplicates()
    for _, row in sdc_instances.iterrows():
        sdc_subclass_uri = sdc_class_mapping.get(row.sdc_kind, None)
        if sdc_subclass_uri is None:
            print(f"Unknown SDC subclass for unit: {row.sdc_kind}")
            exit(1)

        # Create sdc instance
        sdc_instance_q_name = unique_qname(row.sdc_kind, [row.sdc_kind, row.artifact_id])
        label = f"Specifically dependent continuant instance of kind {row.sdc_kind} for {row.artifact_id}."
        g.add((ns[sdc_instance_q_name], RDF.type, BFO.specificallyDependentContinuant))
        g.add((ns[sdc_instance_q_name], RDF.type, sdc_subclass_uri))
        g.add((ns[sdc_instance_q_name], RDFS.label, Literal(label, lang='en')))

        # Add these instances to the material artifacts via bearer of
        material_artifact_q_name = unique_qname(MATERIAL_ARTIFACT_CLASS_NAME, [row.artifact_id])
        g.add((ns[material_artifact_q_name], BFO.bearerOf, ns[sdc_instance_q_name]))

    logger.info(f"Created {len(sdc_instances)} sdc instances and added these to the material artifacts.")


def get_sdc_class_mapping(g: Graph, ns: Namespace) -> dict[str, URIRef]:
    sdc_class_mapping = {
        "temperature": CCO.temperature,
        "pressure": create_subclass(ns,
                                    "PressureSdc",
                                    "A pressure is an amount of force excerted on a surface.",
                                    BFO.specificallyDependentContinuant, g),
        "voltage": create_subclass(ns,
                                   "Voltage",
                                   "A voltage is a difference in potential between two points in a circuit.",
                                   BFO.specificallyDependentContinuant, g),
        "resistance": create_subclass(ns,
                                      "Resistance",
                                      "A resistance is a measure of opposition to the flow of electric current.",
                                      BFO.specificallyDependentContinuant, g)
    }
    return sdc_class_mapping


def create_subclass(ns: Namespace, class_q_name: str, definition: str, subsuming_class: URIRef, g: Graph) -> URIRef:
    uri_ref = ns[class_q_name]
    g.add((uri_ref, RDFS.subClassOf, subsuming_class))
    g.add((uri_ref, RDFS.label, Literal(label_from_class(class_q_name), lang='en')))
    g.add((uri_ref, SKOS.definition, Literal(definition, lang='en')))
    return uri_ref


def create_mice_sensor_observations(df: DataFrame, g: Graph, ns: Namespace):
    unit_mapping = get_measurement_unit_mapping(g, ns)

    # Create sensor observation instances
    observation_count = 0
    for index, row in df.iterrows():
        observation_count += 1

        # Create instance name, label and URI
        mice_instance_name = unique_qname("mice-sensor-observation",
                                          [row.artifact_id, row.unit_label, row.timestamp,
                                           str(row.value), row.sdc_kind])
        mice_instance_label = (f"Sensor observation on {row.artifact_id} of type {row.sdc_kind} "
                               f"at {row.timestamp} of {row.value}")
        mice_instance_uri = ns[mice_instance_name]

        # Resolve measurement unit URI
        unit_uri = unit_mapping.get(row.unit_label, None)
        if unit_uri is None:
            print(f"Unknown unit label: {row.unit_label}")
            exit(1)

        # Create observation instance
        g.add((mice_instance_uri, RDF.type, CCO.measurementInformationContentEntity))
        g.add((mice_instance_uri, RDFS.label, Literal(mice_instance_label, lang='en')))

        # Add measurement value
        g.add((mice_instance_uri, CCO.hasDecimalValue, Literal(row.value, datatype=XSD.decimal, normalize=False)))

        # Add measurement unit
        g.add((mice_instance_uri, CCO.usesMeasurementUnit, unit_uri))

        # Link to SDC instance
        sdc_instance_q_name = unique_qname(row.sdc_kind, [row.sdc_kind, row.artifact_id])
        g.add((mice_instance_uri, CCO.isAMeasurementOf, URIRef(ns[sdc_instance_q_name])))

    logger.info(f"Created {observation_count} MICE sensor observation instances.")


def get_measurement_unit_mapping(g: Graph, ns: Namespace) -> dict[str, URIRef]:
    # Create measurement unit instances and build mapping
    unit_mapping = {
        "Pa": create_instance("Pa", "Pascal measurement unit instance", CCO.measurementUnit, g, ns),
        "C": create_instance("C", "Celsius measurement unit instance", CCO.measurementUnit, g, ns),
        "volt": create_instance("Volt", "Volt measurement unit instance", CCO.measurementUnit, g, ns),
        "ohm": create_instance("ohm", "Ohm measurement unit instance", CCO.measurementUnit, g, ns)
    }
    return unit_mapping


def create_instance(instance_q_name: str, instance_label: str, instance_type: URIRef, g: Graph,
                    ns: Namespace) -> URIRef:
    instance_uri = ns[instance_q_name]
    g.add((instance_uri, RDF.type, instance_type))
    g.add((instance_uri, RDFS.label, Literal(instance_label, lang='en')))
    return instance_uri


def write_ttl_kg(
        graph: Graph, filename: pathlib.Path, default_ns: Namespace = None, base: str = None
) -> None:
    """
    Writes an RDF graph to a Turtle (.ttl) file with optional default namespace
    and base URI configuration.

    This function allows serializing an RDF graph object to the Turtle format
    and writing it to a file. A default namespace can optionally be bound to
    the graph, and a base URI can also be specified for serialization.

    :param graph: The RDF graph to be serialized and written to the file.
    :type graph: Graph
    :param filename: The path to the file where the Turtle data will be written.
    :type filename: str
    :param default_ns: Optional default namespace to bind to the graph.
    :type default_ns: Namespace, optional
    :param base: Optional base URI to use during serialization.
    :type base: str, optional
    :return: This method does not return any value.
    :rtype: None
    """
    logger.info(f"Writing graph to {filename}.")
    logger.info(f"Using default namespace {default_ns}.")
    logger.info(f"And base URI {base}.")

    if default_ns:
        graph.bind("", default_ns)

    try:
        rdflib.NORMALIZE_LITERALS = False
        with open(filename, "wb") as f:
            graph.serialize(f, format="turtle", base=base)
    except Exception as ex:
        logger.info(ex)
        raise Exception(f"Could not write graph to {filename}: {ex}.") from ex


def unique_qname(class_name: str, elements: Iterable[str]) -> URIRef:
    """
    Generate a qualified name (QName) for an instance based on its namespace, class name,
    and a collection of elements. The QName is constructed by converting the class name
    to lowercase with dashes and appending a unique identifier derived from the elements.
    This function is helpful in namespaces and linked data contexts for creating unique
    identifiers.

    :param class_name: The name of the class for which the instance QName is being generated.
    :type class_name: str
    :param elements: An iterable collection of elements used to derive a unique identifier.
        This ensures the uniqueness of the generated QName.
    :type elements: Iterable[str]
    :return: A generated QName that uniquely identifies an instance within the namespace.
    :rtype: URIRef
    """
    qname_prefix = qname_from_class(class_name)
    qname_suffix = qname_id_suffix(elements)
    qname = f"{qname_prefix}-{qname_suffix}"
    return qname


def qname_from_class(class_name: str) -> str:
    """
    Generates a "qualified" name for a class by converting its class name to a string that is
    in lowercase, hyphen-separated format. The class name is first converted from camel case
    to a human-readable string with spaces, and then further processed to match the desired format.

    :param class_name: The name of the class to convert.
    :type class_name: str
    :return: A string representation of the class name in lowercase, hyphen-separated format.
    :rtype: str
    """
    return camel_case_to_words(class_name).lower().replace(' ', '-')


def qname_id_suffix(elements: Iterable[str]) -> str:
    """Generate a SHA-
    1 hash-based identifier from a collection of string elements.

    Args:
        elements: An iterable of strings to be hashed together

    Returns:
        Base64-encoded string representation of the SHA-1 hash, safe for use as a QName
    """
    joined_elements = ':'.join(elements)
    encoded_elements = joined_elements.encode('utf-8')
    hash_object = sha1(encoded_elements)
    # Use urlsafe_b64encode for QName compatibility (- and _ instead of + and /)
    # Remove padding (=) and decode to string
    qname_safe_id = base64.urlsafe_b64encode(hash_object.digest()).rstrip(b'=').decode('ascii')
    return qname_safe_id


def label_from_class(class_name: str, lang: str = 'en') -> Literal:
    """
    Converts a camel case class name into a readable label and wraps it
    as an RDF `Literal` with a specified language code. The function
    modifies the class name to make the first letter uppercase and
    the rest lowercase for better readability.

    :param class_name: The camel case class name to be converted
        into a readable label.
    :type class_name: str
    :param lang: The language code for the RDF `Literal`. Defaults to 'en'.
    :type lang: str, optional

    :return: An RDF `Literal` object containing the converted readable
        label and language code.
    :rtype: Literal
    """
    label = camel_case_to_words(class_name)
    label = label[0].upper() + label[1:].lower()
    literal = Literal(label, lang=lang)
    return literal


@lru_cache(maxsize=1024 * 1024)
def camel_case_to_words(text) -> str:
    """
    Convert camel case words to separate words with spaces.
    Results are cached for improved performance on repeated calls.

    Args:
        text: String containing camel case words

    Returns:
        String with camel case words separated by spaces

    Examples:
        >>> camel_case_to_words("thisIsATest")
        'this Is A Test'
        >>> camel_case_to_words("readingValue")
        'reading Value'
        >>> camel_case_to_words("deviceName")
        'device Name'
    """
    if not text:
        return text

    # Insert space before uppercase letters that follow lowercase letters
    # or before uppercase letters that are followed by lowercase letters (for acronyms)
    result = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    result = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', result)

    return result


if __name__ == "__main__":
    main()
