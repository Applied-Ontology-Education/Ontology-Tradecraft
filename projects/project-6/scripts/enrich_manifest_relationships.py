#!/usr/bin/env python3
"""Project 6 manifest stage 2: explicit relationship and metadata enrichment.

This stage DOES NOT change src/module.ttl, generated/module_augmented.ttl,
manifest_instances.ttl, or any existing passenger records. It adds a small,
reviewable OWL TBox extension plus additive ABox triples as new generated files.

Run after ingest_passenger_manifest.py and with identical input workbook:

    python scripts/enrich_manifest_relationships.py \
      --input templates/passenger_manifest_template.xlsx \
      --flight-document FL-20261119-AA101=FPDOC-AA101

--flight-document is optional and asserts an ACTUAL physical flight-plan document.
Do not use it if only electronic flight-plan information is known.

Per assignment contract, ALL passenger rows are authoritative in-transit/onboard
manifest assertions. Actual air/rail processes and their participating people and
vehicles are created by the initial ingestion script. This enrichment adds a
separate air-flight class, explicit air-flight associations, and planning metadata.
No role or threat designation is inferred.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

from ingest_passenger_manifest import (
    INTEL, AGENT, BFO, _sid, _sha_file, _turtle, _atomic_write,
    _atomic_write_bytes, build_abox, parse_manifest, verify_records,
    verify_ontology, ManifestError
)

VERSION = 'project6-manifest-relationships-v3-in-transit'
SKOS_DEF = URIRef('http://www.w3.org/2004/02/skos/core#definition')
IAO_MATERIAL_BEARER = BFO.IAO_0000178
IS_ABOUT = BFO.IAO_0000136
PARTICIPATES_IN = BFO.BFO_0000056
IS_CARRIER_OF = BFO.BFO_0000101

# All NEW predicates are explicit and auditable: no invented predicate is
# silently added to a user ontology, and no existing domain/range is changed.
NEW_DATA_PROPS = {
    INTEL.has_physical_rail_ticket_evidence_reference: (
        INTEL.rail_ticket, XSD.string, 'has physical rail ticket evidence reference',
        'An optional record reference supporting a physical ticket bearer.'),
    INTEL.has_flight_service_code: (
        AGENT.Flight_plan, XSD.string, 'has flight service code',
        'Relates flight plan information to a service identifier.'),
    INTEL.has_planned_air_departure_time: (
        AGENT.Flight_plan, XSD.dateTime, 'has planned air departure time',
        'Relates flight plan information to its scheduled departure instant.'),
    INTEL.has_train_departure_station_name: (
        INTEL.passenger_train, XSD.string, 'has train departure station name',
        'Records the manifest departure station name for this in-transit train.'),
    INTEL.has_train_arrival_station_name: (
        INTEL.passenger_train, XSD.string, 'has train arrival station name',
        'Records the manifest arrival station name for this in-transit train.'),
}
NEW_OBJECT_PROPS = {
    INTEL.plans_to_use_train: (
        INTEL.rail_journey_plan, INTEL.passenger_train, 'plans to use train',
        'Connects a rail journey plan to its train.'),
    INTEL.has_planned_departure_station: (
        INTEL.rail_journey_plan, INTEL.railway_station, 'has planned departure station',
        'Relates a journey plan to its departure facility.'),
    INTEL.has_planned_arrival_station: (
        INTEL.rail_journey_plan, INTEL.railway_station, 'has planned arrival station',
        'Relates a journey plan to its arrival facility.'),
    INTEL.occurs_aboard_airplane: (
        INTEL.air_flight, AGENT.Commcercial_airplane, 'occurs aboard airplane',
        'Relates an actual flight process to the airplane conveying its participants.'),
    INTEL.has_flight_plan_information: (
        INTEL.air_flight, AGENT.Flight_plan, 'has flight plan information',
        'Relates an actual flight process to its flight-plan information content entity.'),
}

def parse_kv_arguments(items: list[str]) -> dict[str, str]:
    result = {}
    for item in items:
        if '=' not in item:
            raise ManifestError(f'Expected TRIP_ID=DOCUMENT_ID, received {item!r}')
        trip, document = item.split('=', 1)
        trip, document = trip.strip(), document.strip()
        if not trip or not document or trip in result:
            raise ManifestError('Each --flight-document must have unique nonempty TRIP_ID=DOCUMENT_ID')
        result[trip] = document
    return result


def _schema_graph(base: Graph) -> Graph:
    g = Graph()
    for prop, (domain, range_, label, definition) in NEW_DATA_PROPS.items():
        if list(base.predicate_objects(prop)):
            raise ManifestError(f'New data property already appears in augmented ontology: {prop}')
        g.add((prop, RDF.type, OWL.DatatypeProperty))
        g.add((prop, RDFS.domain, domain))
        g.add((prop, RDFS.range, range_))
        g.add((prop, RDFS.label, Literal(label, lang='en')))
        g.add((prop, SKOS_DEF, Literal(definition, lang='en')))
    for prop, (domain, range_, label, definition) in NEW_OBJECT_PROPS.items():
        if list(base.predicate_objects(prop)):
            raise ManifestError(f'New object property already appears in augmented ontology: {prop}')
        g.add((prop, RDF.type, OWL.ObjectProperty))
        g.add((prop, RDFS.domain, domain))
        g.add((prop, RDFS.range, range_))
        g.add((prop, RDFS.label, Literal(label, lang='en')))
        g.add((prop, SKOS_DEF, Literal(definition, lang='en')))
    # Distinct actual-flight process class, not the planning Act or ICE.
    cls = INTEL.air_flight
    if list(base.predicate_objects(cls)):
        raise ManifestError('New air_flight class already appears in augmented ontology')
    g.add((cls, RDF.type, OWL.Class))
    g.add((cls, RDFS.subClassOf, BFO.BFO_0000015))
    g.add((cls, RDFS.label, Literal('air flight', lang='en')))
    g.add((cls, SKOS_DEF, Literal('An actual process of an airplane flying while conveying passengers.', lang='en')))
    return g


def _expected_core_records(records, base: Graph, existing_abox: Graph) -> None:
    verify_records(records)
    verify_ontology(base, records)
    generated, _ = build_abox(records)
    if set(generated) != set(existing_abox):
        raise ManifestError('Existing manifest_instances.ttl does not exactly match the supplied Excel file; rerun initial ingestion first')


def add_relationships(records, base: Graph, existing_abox: Graph,
                      flight_documents: dict[str, str] | None = None) -> tuple[Graph, Graph, dict]:
    """Add fully declared flight-process, planning and rail-itinerary relations."""
    flight_documents = flight_documents or {}
    _expected_core_records(records, base, existing_abox)
    air_trips, rail_trips = {}, {}
    for r in records:
        (air_trips if r.mode == 'air' else rail_trips)[r.values['trip_id']] = r.values
    if set(flight_documents) - set(air_trips):
        raise ManifestError('A --flight-document trip ID was not found in Air_Manifest')
    if len(set(flight_documents.values())) != len(flight_documents):
        raise ManifestError('A physical flight-plan document ID cannot refer to multiple trips')

    schema = _schema_graph(base)
    delta = Graph()
    audit = {'air_planning_processes': 0, 'actual_air_flights': 0,
             'aircraft_planning_participations': 0, 'aircraft_flight_participations': 0,
             'flight_plan_document_bearers': 0, 'air_information_about_process_links': 0,
             'rail_plan_train_links': 0, 'rail_plan_station_links': 0,
             'rail_train_participation_links': 0,
             'station_direction_checks': [], 'source_unchanged': True,
             'flight_and_rail_occurrence_assumed_from_manifest': True,
             'physical_document_evidence_required': True}

    for trip_id, d in sorted(air_trips.items()):
        info = _sid('flight_plan', trip_id)
        flight = _sid('trip', 'air', trip_id)
        plane = _sid('airframe', d['aircraft_tail_number'])
        if (info, RDF.type, AGENT.Flight_plan) not in existing_abox:
            raise ManifestError(f'Flight plan {trip_id} missing from initial ABox')
        if (flight, RDF.type, BFO.BFO_0000015) not in existing_abox:
            raise ManifestError(f'Actual flight {trip_id} missing from initial ABox')
        if (plane, PARTICIPATES_IN, flight) not in existing_abox:
            raise ManifestError(f'Airplane participation {trip_id} missing from initial ABox')
        delta.add((flight, RDF.type, INTEL.air_flight))
        delta.add((flight, INTEL.occurs_aboard_airplane, plane))
        delta.add((flight, INTEL.has_flight_plan_information, info))
        delta.add((info, INTEL.has_flight_service_code,
                   Literal(d['service_code'], datatype=XSD.string)))
        delta.add((info, INTEL.has_planned_air_departure_time,
                   Literal(d['departure_time'], datatype=XSD.dateTime)))
        act = _sid('act_of_flight_plan', trip_id)
        delta.add((act, RDF.type, OWL.NamedIndividual))
        delta.add((act, RDF.type, AGENT.Act_of_Flight_Plan))
        delta.add((act, RDFS.label, Literal('Act of Flight Plan '+d['service_code'])))
        delta.add((info, IS_ABOUT, act))
        delta.add((plane, PARTICIPATES_IN, act))
        audit['actual_air_flights'] += 1
        audit['air_planning_processes'] += 1
        audit['aircraft_planning_participations'] += 1
        audit['aircraft_flight_participations'] += 1
        audit['air_information_about_process_links'] += 1
        if trip_id in flight_documents:
            doc_id = flight_documents[trip_id]
            document = _sid('physical_flight_plan_document', doc_id)
            delta.add((document, RDF.type, OWL.NamedIndividual))
            delta.add((document, RDF.type, IAO_MATERIAL_BEARER))
            delta.add((document, RDFS.label, Literal('Physical flight plan document '+doc_id)))
            delta.add((document, IS_CARRIER_OF, info))
            audit['flight_plan_document_bearers'] += 1

    for trip_id, d in sorted(rail_trips.items()):
        train = _sid('passenger_train', trip_id)
        process = _sid('trip', 'rail', trip_id)
        if (train, RDF.type, INTEL.passenger_train) not in existing_abox:
            raise ManifestError(f'Rail train {trip_id} missing from initial ABox')
        if (process, RDF.type, INTEL.rail_journey) not in existing_abox:
            raise ManifestError(f'Actual rail journey {trip_id} missing from initial ABox')
        if (train, PARTICIPATES_IN, process) not in existing_abox:
            raise ManifestError(f'Train participation {trip_id} missing from initial ABox')
        delta.add((train, INTEL.has_train_departure_station_name,
                   Literal(d['departure_name'], datatype=XSD.string)))
        delta.add((train, INTEL.has_train_arrival_station_name,
                   Literal(d['arrival_name'], datatype=XSD.string)))
        dep = _sid('railway_station', d['departure_id'])
        arr = _sid('railway_station', d['arrival_id'])
        if ((dep, RDF.type, INTEL.railway_station) not in existing_abox or
                (arr, RDF.type, INTEL.railway_station) not in existing_abox):
            raise ManifestError('Expected railway station individual missing from initial ABox')
        for r in [x for x in records if x.mode=='rail' and x.values['trip_id']==trip_id]:
            plan = _sid('rail_journey_plan', r.values['entry_id'])
            delta.add((plan, INTEL.plans_to_use_train, train))
            delta.add((plan, INTEL.has_planned_departure_station, dep))
            delta.add((plan, INTEL.has_planned_arrival_station, arr))
            if r.values['physical_ticket_evidence_reference']:
                ticket = _sid('rail_ticket', r.values['ticket_id'])
                if (ticket, RDF.type, INTEL.rail_ticket) not in existing_abox:
                    raise ManifestError('Physical ticket evidence exists without ticket individual')
                delta.add((ticket, INTEL.has_physical_rail_ticket_evidence_reference,
                           Literal(r.values['physical_ticket_evidence_reference'], datatype=XSD.string)))
            audit['rail_plan_train_links'] += 1
            audit['rail_plan_station_links'] += 2
        audit['rail_train_participation_links'] += 1
        audit['station_direction_checks'].append({
            'trip_id': trip_id, 'service_code': d['service_code'],
            'departure_station_name': d['departure_name'],
            'arrival_station_name': d['arrival_name'],
            'in_transit_assumed_from_manifest': True})

    if set(delta)&set(existing_abox) or set(delta)&set(base) or set(schema)&set(base):
        raise ManifestError('Enrichment overlaps protected ABox or augmented TBox')
    if list(delta.subjects(RDF.type, OWL.Class)) or list(delta.subjects(RDF.type, OWL.ObjectProperty)):
        raise ManifestError('Schema class or property incorrectly declared in ABox')
    audit['new_abox_triples'] = len(delta)
    audit['new_tbox_triples'] = len(schema)
    audit['new_named_individuals'] = len(set(delta.subjects(RDF.type, OWL.NamedIndividual)))
    return schema, delta, audit

def enrich(input_file: Path, augmented: Path, abox_file: Path,
           schema_out: Path, abox_out: Path, combined_out: Path, report_out: Path,
           *, flight_documents: dict[str, str] | None = None,
           validate_only: bool = False) -> dict:
    protected = {str(x): _sha_file(x) for x in [input_file, augmented, abox_file]}
    records = parse_manifest(input_file)
    base = Graph().parse(augmented, format='turtle')
    existing_abox = Graph().parse(abox_file, format='turtle')
    schema, delta, audit = add_relationships(records, base, existing_abox,
                                            flight_documents)
    enriched_abox = existing_abox + delta
    full = base + schema + enriched_abox
    expected = len(base) + len(schema) + len(existing_abox) + len(delta)
    if len(full) != expected:
        raise ManifestError('Expected disjoint TBox/ABox overlay; overlapping or lost triples')
    report = {'version': VERSION, 'status': 'PASS_RELATIONSHIPS_VALIDATED_ONLY' if validate_only
              else 'PASS_RELATIONSHIPS_ENRICHED_NOT_MERGED',
              'augmented_triples': len(base), 'original_abox_triples': len(existing_abox),
              'new_tbox_triples': len(schema), 'new_abox_triples': len(delta),
              'enriched_abox_triples': len(enriched_abox), 'combined_triples': len(full),
              'source_sha256': protected[str(augmented)], 'workbook_sha256': protected[str(input_file)],
              'original_abox_sha256': protected[str(abox_file)],
              'protected_ontology_and_manifest_files_unchanged': True,
              'owl_reasoner_executed': False,
              'disclaimer': 'This is asserted-data enrichment, not OWL reasoner approval.',
              'output_schema': str(schema_out), 'output_abox': str(abox_out),
              'output_combined': str(combined_out), **audit}
    for path, h in protected.items():
        if _sha_file(Path(path)) != h:
            raise ManifestError('Protected input changed while enriching: '+path)
    if validate_only:
        return report
    _atomic_write(schema_out, _turtle(schema))
    _atomic_write(abox_out, _turtle(enriched_abox))
    # Preserve the exact augmented ontology bytes as the prefix.
    original = augmented.read_bytes()
    payload = (original + (b'' if original.endswith(b'\n') else b'\n') + b'\n'
               + _turtle(schema).encode('utf-8') + b'\n' + _turtle(enriched_abox).encode('utf-8'))
    _atomic_write_bytes(combined_out, payload)
    reread = Graph().parse(combined_out, format='turtle')
    if len(reread) != expected:
        raise ManifestError('Combined ontology output lost triples on roundtrip')
    for path, h in protected.items():
        if _sha_file(Path(path)) != h:
            raise ManifestError('Protected input changed after enrichment: '+path)
    _atomic_write(report_out, json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Existing .xlsx passenger manifest')
    parser.add_argument('--ontology', type=Path, default=Path('generated/module_augmented.ttl'))
    parser.add_argument('--existing-abox', type=Path, default=Path('generated/manifest_instances.ttl'))
    parser.add_argument('--schema-out', type=Path, default=Path('generated/manifest_tbox_extension.ttl'))
    parser.add_argument('--output', type=Path, default=Path('generated/manifest_instances_enriched.ttl'))
    parser.add_argument('--combined-output', type=Path, default=Path('generated/module_augmented_with_manifest_enriched.ttl'))
    parser.add_argument('--report', type=Path, default=Path('reports/manifest_relationship_report.json'))
    parser.add_argument('--flight-document', action='append', default=[], metavar='TRIP_ID=DOCUMENT_ID',
                        help='Explicit evidence that a physical flight-plan document exists for this trip')
    parser.add_argument('--validate-only', action='store_true')
    a = parser.parse_args(argv)
    try:
        documents = parse_kv_arguments(a.flight_document)
        report = enrich(a.input, a.ontology, a.existing_abox, a.schema_out, a.output,
                        a.combined_output, a.report, flight_documents=documents,
                        validate_only=a.validate_only)
        print('MANIFEST ENRICH — '+report['status'])
        print(json.dumps({k:report[k] for k in (
            'augmented_triples','original_abox_triples','new_tbox_triples',
            'new_abox_triples','enriched_abox_triples','combined_triples',
            'station_direction_checks','flight_plan_document_bearers',
            'rail_train_participation_links','actual_air_flights')}, indent=2))
        return 0
    except (ManifestError, OSError, ValueError) as exc:
        print('MANIFEST ENRICH — FAIL: '+str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
