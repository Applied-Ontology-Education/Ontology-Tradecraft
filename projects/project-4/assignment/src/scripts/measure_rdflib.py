import argparse
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

OBSERVED_ENTITY_CLASS_NAME = "ObservedEntity"

logger = logging.getLogger(__name__)

# Path settings
SRC_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_SOURCE = SRC_ROOT / "data"
INPUT_PATH = DATA_SOURCE / "readings_normalized.csv"
OUTPUT_PATH = SRC_ROOT / "measure_cco.ttl"

# Namespace settings
DEFAULT_NS = Namespace("http://www.newfoundland.nl/otc/project-4/")
NS_CCO = Namespace("https://www.commoncoreontologies.org/")


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


def label_from_class_id(class_name: str, id: str) -> str:
    """
    Generates a descriptive label by combining a class name with its identifier.

    This function facilitates the creation of a formatted label string that combines
    a given class name and its identifier. It is intended to enhance readability
    and identification of objects or entities based on their class and unique ID.

    :param class_name: The name of the class to generate the label for.
    :type class_name: str
    :param id: The unique identifier associated with the class.
    :type id: str
    :return: The formatted label combining the class name and identifier.
    :rtype: str
    """
    label = f"{label_from_class(class_name)} '{id}'"
    return label


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


def get_args() -> argparse.Namespace:
    """
    Fetches command-line arguments required for processing SPARQL query files against Turtle files
    and outputs the results into a CSV file. Parses the provided arguments and validates their
    presence and expected types.

    :return: Namespace object containing parsed command-line arguments.
    :rtype: argparse.Namespace
    """
    # Initialize
    parser = argparse.ArgumentParser(
        description="The program import a normalized readings CSV input file and outputs an ontology in Turtle (TTL) file.",
        epilog="Happy ontology hacking!",
        prog="create-rdf",
    )

    # Adding optional parameters
    parser.add_argument(
        "-i", "--input-file", help="CSV input file.", default=INPUT_PATH, type=str
    )

    parser.add_argument(
        "-o", "--output-file", help="Turtle output file.", default=OUTPUT_PATH, type=str
    )

    parser.add_argument(
        "-ns", "--namespace", help="Namespace.", default=DEFAULT_NS, type=str
    )

    return parser.parse_args()


def main():
    # Resolve arguments
    args = get_args()
    input_file = args.input_file
    output_file = args.output_file
    namespace = args.namespace
    ns = Namespace(namespace)

    # Read CSV
    df = pd.read_csv(input_file, header=0)

    # Create ontology
    g = Graph()
    create_ontology(g, namespace)

    # Translate data to RDF
    translate_to_rdf(df, g, ns)

    # Write graph to file
    write_ttl_kg(g, output_file, Namespace(ns))


def translate_to_rdf(df: DataFrame, g: Graph, ns: Namespace):
    create_sensors(df, g, ns)
    create_observed_entities(df, g, ns)
    create_qualities(df, g, ns)
    create_sensor_observations(df, g, ns)


def create_ontology(g: Graph, namespace):
    ontology_uri = URIRef(namespace)
    g.add((ontology_uri, RDF.type, OWL.Ontology))
    g.add((ontology_uri, RDFS.label, Literal("Sensor Data Ontology", lang='en')))
    g.add((ontology_uri, RDFS.comment, Literal("An ontology for sensor readings and measurements", lang='en')))
    logger.info("Created ontology.")


def create_sensors(df: DataFrame, g: Graph, ns: Namespace):
    """
    Creates sensor classes and instances in the given graph based on the provided DataFrame.

    The function defines a new OWL class for sensors and adds sensor instances
    for each unique combination of `artifact_id` and `source` in the input
    DataFrame. Each sensor instance will have a unique identifier and a label
    to describe its relation with the observed entity and the source.

    :param df: A pandas DataFrame containing the data for observed entities and
        sources. It must have the columns `artifact_id` and `source`.
    :param g: The RDF graph where sensor classes and instances will be added.
    :param ns: The RDF namespace for creating classes and instances.
    :return: None
    """
    # Sensor https://www.commoncoreontologies.org/ont00000569

    ## Create separate sensor classes
    temperature_sensor_class_name = "TemperatureSensor"
    g.add((ns[temperature_sensor_class_name], RDFS.subClassOf,
           URIRef("https://www.commoncoreontologies.org/ont00000569")))
    g.add((ns[temperature_sensor_class_name], RDFS.label, label_from_class(temperature_sensor_class_name)))
    g.add((ns[temperature_sensor_class_name], SKOS.definition,
           Literal("A temperature sensor is a sensor that can generate a temperature measurement.", lang='en')))
    logger.info("Created TemperatureSensor class.")

    pressure_sensor_class_name = "PressureSensor"
    g.add((ns[pressure_sensor_class_name], RDFS.subClassOf, URIRef("https://www.commoncoreontologies.org/ont00000569")))
    g.add((ns[pressure_sensor_class_name], RDFS.label, label_from_class(pressure_sensor_class_name)))
    g.add((ns[pressure_sensor_class_name], SKOS.definition,
           Literal("A pressure sensor is a sensor that can generate a pressure measurement.", lang='en')))
    logger.info("Created PressureSensor class.")

    voltage_sensor_class_name = "VoltageSensor"
    g.add((ns[voltage_sensor_class_name], RDFS.subClassOf, URIRef("https://www.commoncoreontologies.org/ont00000569")))
    g.add((ns[voltage_sensor_class_name], RDFS.label, label_from_class(voltage_sensor_class_name)))
    g.add((ns[voltage_sensor_class_name], SKOS.definition,
           Literal("A pressure sensor is a sensor that can generate a pressure measurement.", lang='en')))
    logger.info("Created PressureSensor class.")

    resistance_sensor_class_name = "ResistanceSensor"
    g.add(
        (ns[resistance_sensor_class_name], RDFS.subClassOf, URIRef("https://www.commoncoreontologies.org/ont00000569")))
    g.add((ns[resistance_sensor_class_name], RDFS.label, label_from_class(resistance_sensor_class_name)))
    g.add((ns[resistance_sensor_class_name], SKOS.definition,
           Literal("A pressure sensor is a sensor that can generate a pressure measurement.", lang='en')))
    logger.info("Created PressureSensor class.")

    ## Create sensor instances
    source_ids = df[['artifact_id', 'sdc_kind']].drop_duplicates()
    for _, row in source_ids.iterrows():
        if row.sdc_kind == "temperature":
            instance_type = temperature_sensor_class_name
        elif row.sdc_kind == "pressure":
            instance_type = pressure_sensor_class_name
        elif row.sdc_kind == "voltage":
            instance_type = voltage_sensor_class_name
        elif row.sdc_kind == "resistance":
            instance_type = resistance_sensor_class_name
        else:
            print(f"Unknown sensor kind: {row.sdc_kind}")
            exit(1)

        instance_q_name = unique_qname(instance_type, [row.artifact_id, row.sdc_kind])
        label = f"Sensor instance measuring {row.sdc_kind} on {row.artifact_id}"
        g.add((ns[instance_q_name], RDF.type, ns[instance_type]))
        g.add((ns[instance_q_name], RDFS.label, Literal(label, lang='en')))


def create_observed_entities(df: DataFrame, g: Graph, ns: Namespace):
    # Material Artifact https://www.commoncoreontologies.org/ont00000995
    observed_entity_class_name = OBSERVED_ENTITY_CLASS_NAME
    # g.add((ns[observed_entity_class_name], RDFS.subClassOf, URIRef("https://www.commoncoreontologies.org/ont00000995")))
    # g.add((ns[observed_entity_class_name], RDFS.label, label_from_class(observed_entity_class_name)))
    # g.add((ns[observed_entity_class_name], SKOS.definition,
    #        Literal("An observed entity is an entity that is measured by a sensor.", lang='en')))
    # logger.info("Created ObservedEntity class.")

    n = 0
    artifact_ids = df['artifact_id'].unique()
    for artifact_id in artifact_ids:
        n += 1
        observed_entity_q_name = unique_qname(observed_entity_class_name, [artifact_id])
        label = f"Observed entity {artifact_id}"
        g.add((ns[observed_entity_q_name], RDF.type, URIRef("https://www.commoncoreontologies.org/ont00000995")))
        g.add((ns[observed_entity_q_name], RDFS.label, Literal(label, lang='en')))

    logger.info("Created {n} Observed Entity instances.")


def create_qualities(df: DataFrame, g: Graph, ns: Namespace):
    # Temperature https://www.commoncoreontologies.org/ont00000441
    temperature_uri = URIRef("https://www.commoncoreontologies.org/ont00000441")

    # Pressure https://www.commoncoreontologies.org/ont00000380
    pressure_uri = URIRef("https://www.commoncoreontologies.org/ont00000380")

    # Quality http://purl.obolibrary.org/obo/BFO_0000019
    # Voltage
    voltage_class_name = "Voltage"
    g.add((ns[voltage_class_name], RDFS.subClassOf, URIRef("http://purl.obolibrary.org/obo/BFO_0000019")))
    g.add((ns[voltage_class_name], RDFS.label, label_from_class(voltage_class_name)))
    g.add((ns[voltage_class_name], SKOS.definition,
           Literal("A voltage is a measure of an electrical potential difference.", lang='en')))
    voltage_uri = ns[voltage_class_name]
    logger.info("Created Voltage class.")

    # Electrical resistance
    resistance_class_name = "ElectricalResistance"
    g.add((ns[resistance_class_name], RDFS.subClassOf, URIRef("http://purl.obolibrary.org/obo/BFO_0000019")))
    g.add((ns[resistance_class_name], RDFS.label, label_from_class(resistance_class_name)))
    g.add((ns[resistance_class_name], SKOS.definition,
           Literal("A electrical resistance is a measure of its opposition to the flow of electric current.",
                   lang='en')))
    resitance_uri = ns[resistance_class_name]
    logger.info("Created electrical resistance class.")

    # Create instances
    n = 0
    qualities = df[['artifact_id', 'sdc_kind']].drop_duplicates()
    for _, row in qualities.iterrows():
        if (row.sdc_kind == "temperature"):
            instance_type = temperature_uri
        elif (row.sdc_kind == "pressure"):
            instance_type = pressure_uri
        elif (row.sdc_kind == "voltage"):
            instance_type = voltage_uri
        elif (row.sdc_kind == "resistance"):
            instance_type = resitance_uri
        else:
            print(f"Unknown quality kind: {row.sdc_kind}")
            exit(1)

        # Add qualities
        # quality http://purl.obolibrary.org/obo/BFO_0000019
        quality_instance_q_name = unique_qname("quality", [row.artifact_id, row.sdc_kind])
        label = f"Quality instance for {row.artifact_id} of kind {row.sdc_kind}"
        g.add((ns[quality_instance_q_name], RDF.type, URIRef("http://purl.obolibrary.org/obo/BFO_0000020")))
        g.add((ns[quality_instance_q_name], RDFS.label, Literal(label, lang='en')))

        # Add bearer of
        # bearer of http://purl.obolibrary.org/obo/BFO_0000196
        observed_entity_q_name = unique_qname(OBSERVED_ENTITY_CLASS_NAME, [row.artifact_id])
        g.add((ns[observed_entity_q_name], URIRef("http://purl.obolibrary.org/obo/BFO_0000196"),
               ns[quality_instance_q_name]))
    logger.info("Created {n} quality instances.")


def create_sensor_observations(df: DataFrame, g: Graph, ns: Namespace):
    # Pa instance https://www.commoncoreontologies.org/ont00001559"
    # pa_instance_uri = URIRef("https://www.commoncoreontologies.org/ont00001559")

    # Create ohm measurement unit instance
    pa_instance_q_name = "Pa"
    g.add((ns[pa_instance_q_name], RDF.type, URIRef("https://www.commoncoreontologies.org/ont00000120")))
    g.add((ns[pa_instance_q_name], RDFS.label, Literal("Pa measurement unit instance", lang='en')))
    logger.info("Created Pa measurement unit instance.")
    pa_instance_uri = ns[pa_instance_q_name]

    # Celsius instance https://www.commoncoreontologies.org/ont00001606
    # celsius_instance_uri = URIRef("https://www.commoncoreontologies.org/ont00001606")
    celsius_instance_q_name = "C"
    g.add((ns[celsius_instance_q_name], RDF.type, URIRef("https://www.commoncoreontologies.org/ont00000120")))
    g.add((ns[celsius_instance_q_name], RDFS.label, Literal("Celsius measurement unit instance", lang='en')))
    logger.info("Created Celsius measurement unit instance.")
    celsius_instance_uri = ns[celsius_instance_q_name]

    # Volt instance https://www.commoncoreontologies.org/ont00001450
    #volt_instance_uri = URIRef("https://www.commoncoreontologies.org/ont00001450")
    volt_instance_q_name = "C"
    g.add((ns[volt_instance_q_name], RDF.type, URIRef("https://www.commoncoreontologies.org/ont00000120")))
    g.add((ns[volt_instance_q_name], RDFS.label, Literal("Volt measurement unit instance", lang='en')))
    logger.info("Created volt measurement unit instance.")
    volt_instance_uri = ns[volt_instance_q_name]

    # Create ohm measurement unit class
    # Measurement unit https://www.commoncoreontologies.org/ont00000120
    ohm_class_name = "OhmMeasurementUnit"
    g.add((ns[ohm_class_name], RDFS.subClassOf, URIRef("https://www.commoncoreontologies.org/ont00000120")))
    g.add((ns[ohm_class_name], RDFS.label, label_from_class(ohm_class_name)))
    g.add((ns[ohm_class_name], SKOS.definition,
           Literal("An ohm is a measurement unit of electromagnetic resistance.", lang='en')))
    logger.info("Created ohm measurement unit class.")

    # Create ohm measurement unit instance
    ohm_instance_q_name = "ohm"
    g.add((ns[ohm_instance_q_name], RDF.type, URIRef("https://www.commoncoreontologies.org/ont00000120")))
    g.add((ns[ohm_instance_q_name], RDFS.label, Literal("Ohm measurement unit instance", lang='en')))
    logger.info("Created ohm measurement unit instance.")
    ohm_instance_uri = ns[ohm_instance_q_name]

    # Create sensor readings class
    class_name = "SensorObservation"
    g.add((ns[class_name], RDFS.subClassOf, URIRef("https://www.commoncoreontologies.org/ont00001163")))
    g.add((ns[class_name], RDFS.label, label_from_class(class_name)))
    g.add((ns[class_name], SKOS.definition,
           Literal("A sensor observation is a measurement observation generated by a sensor.", lang='en')))
    logger.info("Created SensorObservation class.")

    # Create sensor readings instances
    n = 0
    for index, row in df.iterrows():
        n += 1

        # Create observation instance
        # MICE https://www.commoncoreontologies.org/ont00001163
        instance_name = unique_qname("sensor-observation",
                                     [row.artifact_id, row.unit_label, row.timestamp, str(row.value), row.sdc_kind])
        label = f"Sensor observation on {row.artifact_id} of type {row.sdc_kind} at {row.timestamp} of {row.value}"
        g.add((ns[instance_name], RDF.type, URIRef("https://www.commoncoreontologies.org/ont00001163")))
        g.add((ns[instance_name], RDFS.label, Literal(label, lang='en')))

        # Add value
        # has double value https://www.commoncoreontologies.org/ont00001770
        # has decimal value https://www.commoncoreontologies.org/ont00001769
        g.add((ns[instance_name], URIRef("https://www.commoncoreontologies.org/ont00001769"),
               Literal(row.value, datatype=XSD.decimal, normalize=False)))

        # Add uses measurement unit
        if (row.unit_label == "Pa"):
            type_uri = pa_instance_uri
        elif (row.unit_label == "C"):
            type_uri = celsius_instance_uri
        elif (row.unit_label == "ohm"):
            type_uri = ohm_instance_uri
        elif (row.unit_label == "volt"):
            type_uri = volt_instance_uri
        else:
            print(f"Unknown unit label: {row.unit_label}")
            exit(1)

        # uses measurement unit https://www.commoncoreontologies.org/ont00001863
        g.add((ns[instance_name], URIRef("https://www.commoncoreontologies.org/ont00001863"),
               type_uri))

        # Is measurement of quality
        # is a measurement of https://www.commoncoreontologies.org/ont00001966
        quality_instance_q_name = unique_qname("quality", [row.artifact_id, row.sdc_kind])
        g.add((ns[instance_name], URIRef("https://www.commoncoreontologies.org/ont00001966"),
               URIRef(ns[quality_instance_q_name])))

    logger.info("Created {n} SensorObservation instances.")


if __name__ == "__main__":
    main()

# Temperature https://www.commoncoreontologies.org/ont00000441
# Nominal MICE https://www.commoncoreontologies.org/ont00000293
# Ratio measurement https://www.commoncoreontologies.org/ont00001022
# is a ratio measurement of https://www.commoncoreontologies.org/ont00001983
# is a ordinal meseasurement of https://www.commoncoreontologies.org/ont00001811
# Measurment unit https://www.commoncoreontologies.org/ont00000120


# Sensor https://www.commoncoreontologies.org/ont00000569
# Material Artifact https://www.commoncoreontologies.org/ont00000995
