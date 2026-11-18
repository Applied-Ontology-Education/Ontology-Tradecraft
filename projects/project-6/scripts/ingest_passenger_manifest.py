#!/usr/bin/env python3
"""Ingest a strictly structured .xlsx passenger manifest into an ABox overlay.

Manifest contract: a row means the passenger is actually physically aboard an
in-transit aircraft or train. This is accepted as input truth for the assignment;
no separate occurrence/boarding evidence is checked.

No changes to the source/augmented ontology. No speculative threat labels.
No new ontology classes or properties in this first-stage output. No LLM calls.

Quick start from project-6 root:
  python scripts/ingest_passenger_manifest.py --input templates/passenger_manifest_template.xlsx
  python scripts/ingest_passenger_manifest.py --input data/passenger_manifest.xlsx --validate-only

Requires rdflib; the .xlsx reader uses Python's standard library.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import posixpath
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile, BadZipFile

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

VERSION = "project6-manifest-abox-v3-in-transit"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "pr": "http://schemas.openxmlformats.org/package/2006/relationships"}
INTEL = Namespace("https://example.org/ontologies/intel-ontology#")
AGENT = Namespace("https://www.commoncoreontologies.org/AgentOntology/")
BFO = Namespace("http://purl.obolibrary.org/obo/")
M = Namespace("https://example.org/ontologies/intel-ontology/manifest#")

COMMON = ["entry_id", "person_id", "first_name", "last_name", "trip_id",
          "service_code", "departure_id", "departure_name", "arrival_id",
          "arrival_name", "departure_time", "reservation_id"]
AIR = COMMON + ["boarding_pass_id", "boarding_gate", "boarding_group", "boarding_time", "aircraft_tail_number"]
RAIL = COMMON + ["ticket_id", "physical_ticket_evidence_reference"]
# Assignment contract: each manifest row is an authoritative assertion that
# its person is physically aboard the named vehicle on an in-transit journey.
# No occurrence-evidence or booking-status fields are required or inferred.
# A physical *paper* rail ticket remains a different claim; its optional evidence
# reference is retained so a ticket ID alone does not manufacture a material bearer.

SHEETS = {"Air_Manifest": AIR, "Rail_Manifest": RAIL}
ID_FIELDS = {"entry_id", "person_id", "trip_id", "service_code", "departure_id",
             "arrival_id", "reservation_id", "boarding_pass_id", "ticket_id", "aircraft_tail_number",
             "physical_ticket_evidence_reference"}
REQUIRED = set(COMMON) - {"reservation_id"}  # ticket/boarding-pass receipt still created without a reservation number

class ManifestError(ValueError):
    """Invalid input that must be corrected rather than silently coerced."""

@dataclass(frozen=True)
class Record:
    mode: str
    sheet: str
    row: int
    values: dict


def _sha_file(path: Path) -> str:
    d = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def _sid(kind: str, *parts: str) -> URIRef:
    # Opaque IDs prevent raw manifest identifiers from appearing in resource IRIs.
    # This is stable pseudonymization, NOT anonymization of exported personal data.
    raw = json.dumps([kind, *parts], ensure_ascii=False, separators=(",", ":"))
    return M[f"{kind}_{sha256(raw.encode('utf-8')).hexdigest()[:24]}"]


def _sheet_mapping(z: ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    by_id = {}
    for rel in rels.findall("pr:Relationship", NS):
        target = rel.get("Target", "")
        if rel.get("TargetMode") == "External":
            continue
        if target.startswith("/"):
            path = target.lstrip("/")
        elif target.startswith("xl/"):
            path = target
        else:
            path = posixpath.normpath(posixpath.join("xl", target))
        by_id[rel.get("Id")] = path
    out = {}
    for sh in workbook.findall("m:sheets/m:sheet", NS):
        name, rid = sh.get("name"), sh.get(f"{{{NS['r']}}}id")
        if rid not in by_id:
            raise ManifestError(f"Unresolvable worksheet relationship for '{name}'")
        out[name] = by_id[rid]
    return out


def _strings(z: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    return ["".join(t.text or "" for t in si.findall(".//m:t", NS))
            for si in root.findall("m:si", NS)]


def _cell_value(c: ET.Element, shared: list[str], sheet: str, row: int):
    # Cannot use formulas, stale cached formula results, or Excel numeric dates.
    if c.find("m:f", NS) is not None:
        raise ManifestError(f"{sheet} row {row}: formulas are not permitted; paste values")
    typ = c.get("t", "n")
    if typ == "inlineStr":
        return "".join(t.text or "" for t in c.findall(".//m:t", NS)), "text"
    value = c.find("m:v", NS)
    content = value.text if value is not None and value.text is not None else ""
    if typ == "s":
        try:
            return shared[int(content)], "text"
        except (IndexError, TypeError, ValueError):
            raise ManifestError(f"{sheet} row {row}: invalid shared string index") from None
    if typ in {"str", "inlineStr"}:
        return content, "text"
    if typ == "b":
        return "TRUE" if content == "1" else "FALSE", "boolean"
    if typ == "e":
        raise ManifestError(f"{sheet} row {row}: workbook contains an Excel error")
    return content, "number" if content else "blank"


def _col_index(ref: str) -> int:
    letters = re.match(r"^([A-Z]+)", ref.upper())
    if not letters:
        raise ManifestError("Invalid Excel cell reference")
    idx = 0
    for ch in letters.group(1):
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1


def _rows(z: ZipFile, path: str, shared: list[str], sheet_name: str):
    if path not in z.namelist():
        raise ManifestError(f"Worksheet XML missing for '{sheet_name}'")
    root = ET.fromstring(z.read(path))
    for r in root.findall("m:sheetData/m:row", NS):
        number = int(r.get("r", "0"))
        vals = {}
        for cell in r.findall("m:c", NS):
            idx = _col_index(cell.get("r", ""))
            vals[idx] = _cell_value(cell, shared, sheet_name, number)
        yield number, vals


def _iso_datetime(value: str, loc: str) -> str:
    if not value or "T" not in value:
        raise ManifestError(f"{loc}: expected ISO 8601 date/time with UTC offset, e.g. 2026-11-19T09:00:00-05:00")
    try:
        d = datetime.fromisoformat(value)
    except ValueError:
        raise ManifestError(f"{loc}: invalid date/time") from None
    if d.tzinfo is None or d.utcoffset() is None:
        raise ManifestError(f"{loc}: time zone offset required")
    return d.isoformat(timespec="seconds")


def parse_manifest(path: Path) -> list[Record]:
    if path.suffix.lower() != ".xlsx" or not path.is_file():
        raise ManifestError("Input must be an existing .xlsx workbook")
    records = []
    try:
        with ZipFile(path) as z:
            mapped = _sheet_mapping(z)
            for sheet, headers in SHEETS.items():
                if sheet not in mapped:
                    raise ManifestError(f"Missing required worksheet '{sheet}'")
                rows = list(_rows(z, mapped[sheet], _strings(z), sheet))
                if not rows:
                    raise ManifestError(f"{sheet}: missing header row")
                head_r, hdr_cells = rows[0]
                if head_r != 1:
                    raise ManifestError(f"{sheet}: headers must appear on row 1")
                header = [hdr_cells.get(i, ("", "blank"))[0].strip() for i in range(len(headers))]
                if header != headers or any(i >= len(headers) and v[0].strip() for i, v in hdr_cells.items()):
                    raise ManifestError(f"{sheet}: header names/order do not match the provided template")
                for num, values in rows[1:]:
                    if not any(text.strip() for text, _ in values.values()):
                        continue
                    if any(i >= len(headers) and value.strip() for i, (value, _) in values.items()):
                        raise ManifestError(f"{sheet} row {num}: extra columns are not supported")
                    d = {}
                    for i, field in enumerate(headers):
                        value, cell_type = values.get(i, ("", "blank"))
                        value = str(value).strip()
                        if field in ID_FIELDS and value and cell_type not in {"text"}:
                            raise ManifestError(f"{sheet} row {num}: {field} must be Excel TEXT to preserve leading zeros")
                        if field in REQUIRED and not value:
                            raise ManifestError(f"{sheet} row {num}: required field {field} is blank")
                        d[field] = value
                    for field in ("departure_time", "boarding_time"):
                        if field in d and d[field]:
                            if values.get(headers.index(field), ("", "blank"))[1] not in {"text"}:
                                raise ManifestError(f"{sheet} row {num}: {field} must be ISO text, not Excel date numeric")
                            # A leading apostrophe is Excel's conventional text marker.
                            d[field] = _iso_datetime(d[field].removeprefix("'"), f"{sheet} row {num} {field}")
                    if sheet == "Air_Manifest" and not d['aircraft_tail_number']:
                        raise ManifestError(f"{sheet} row {num}: aircraft_tail_number required for in-transit manifest")
                    if sheet == "Rail_Manifest" and d['physical_ticket_evidence_reference'] and not d['ticket_id']:
                        raise ManifestError(f"{sheet} row {num}: physical_ticket_evidence_reference requires ticket_id")
                    records.append(Record("air" if sheet == "Air_Manifest" else "rail", sheet, num, d))
    except (BadZipFile, KeyError, ET.ParseError) as exc:
        raise ManifestError(f"Invalid or unsupported .xlsx workbook: {type(exc).__name__}") from exc
    if not records:
        raise ManifestError("Workbook contains no passenger manifest records")
    return records


def verify_records(records: list[Record]) -> None:
    person_names = {}
    entry_ids = set()
    tokens = set()
    trips = {}
    trip_airframes = {}
    stations = {}
    for r in records:
        d = r.values
        key = (r.mode, d['entry_id'])
        if key in entry_ids:
            raise ManifestError(f"{r.sheet} row {r.row}: duplicate entry_id within this mode")
        entry_ids.add(key)
        name_key = (d['first_name'].casefold(), d['last_name'].casefold())
        if d['person_id'] in person_names and person_names[d['person_id']] != name_key:
            raise ManifestError(f"{r.sheet} row {r.row}: conflicting names for same person_id")
        person_names[d['person_id']] = name_key
        for field in ('boarding_pass_id', 'ticket_id'):
            if d.get(field):
                token = (r.mode, field, d[field])
                if token in tokens:
                    raise ManifestError(f"{r.sheet} row {r.row}: duplicate {field}")
                tokens.add(token)
        trip_key = (r.mode, d['trip_id'])
        sig = tuple(d[x] for x in ('service_code', 'departure_id', 'departure_name',
                                    'arrival_id', 'arrival_name', 'departure_time'))
        if trip_key in trips and trips[trip_key] != sig:
            raise ManifestError(f"{r.sheet} row {r.row}: trip_id has conflicting route/time/status")
        trips[trip_key] = sig
        if r.mode == 'air' and d['aircraft_tail_number']:
            tail = d['aircraft_tail_number'].upper()
            if trip_key in trip_airframes and trip_airframes[trip_key] != tail:
                raise ManifestError(f"{r.sheet} row {r.row}: conflicting aircraft tails for same trip_id")
            trip_airframes[trip_key] = tail
        for field in ('departure', 'arrival'):
            loc_key = (r.mode, d[field+'_id'])
            loc_name = d[field+'_name'].casefold()
            if loc_key in stations and stations[loc_key] != loc_name:
                raise ManifestError(f"{r.sheet} row {r.row}: location ID has conflicting names")
            stations[loc_key] = loc_name


def verify_ontology(base: Graph, records: list[Record]) -> None:
    # Fail fast: no unsupported class/property IRIs are invented by ingestion.
    classes = [AGENT.person, AGENT.boarding_pass, AGENT.airline_reservation, AGENT.Flight_plan]
    props = [(AGENT.possesses, OWL.ObjectProperty),
             (AGENT.has_first_name, OWL.DatatypeProperty),
             (AGENT.has_last_name, OWL.DatatypeProperty),
             (AGENT.has_reservation_number, OWL.DatatypeProperty),
             (AGENT.has_departure_airport, OWL.DatatypeProperty),
             (AGENT.has_arrival_airport, OWL.DatatypeProperty)]
    if any(r.mode == 'air' for r in records):
        classes += [AGENT.Commcercial_airplane]
        props += [(AGENT.has_boarding_gate, OWL.DatatypeProperty),
                  (AGENT.has_boarding_group, OWL.DatatypeProperty),
                  (AGENT.has_boarding_time, OWL.DatatypeProperty),
                  (AGENT.has_tail_number, OWL.DatatypeProperty)]
    if any(r.mode == 'rail' for r in records):
        classes += [INTEL.rail_ticket, INTEL.rail_reservation_information,
                    INTEL.passenger_train, INTEL.railway_station, INTEL.rail_journey_plan,
                    INTEL.rail_journey]
        props += [(INTEL.departs_from_station, OWL.ObjectProperty),
                  (INTEL.arrives_at_station, OWL.ObjectProperty),
                  (INTEL.occurs_aboard_train, OWL.ObjectProperty),
                  (INTEL.specifies_rail_journey, OWL.ObjectProperty),
                  (INTEL.has_planned_departure_time, OWL.DatatypeProperty),
                  (INTEL.has_train_service_code, OWL.DatatypeProperty)]
    for cls in classes:
        if (cls, RDF.type, OWL.Class) not in base:
            raise ManifestError(f"Augmented ontology is missing required class: {cls}")
    for prop, kind in props:
        if (prop, RDF.type, kind) not in base:
            raise ManifestError(f"Augmented ontology is missing required {kind.split('#')[-1]}: {prop}")
    # These imported predicates already occur in original source assertions.
    for predicate in (BFO.BFO_0000101, BFO.IAO_0000136, BFO.BFO_0000056):
        if not any(base.triples((None, predicate, None))):
            raise ManifestError(f"Source does not establish the imported property {predicate}")


def build_abox(records: list[Record]) -> tuple[Graph, dict]:
    """Treat each row as a truthful onboard/in-transit manifest assertion."""
    graph = Graph()
    graph.bind('manifest', M)
    graph.bind('intel', INTEL)
    graph.bind('agent', AGENT)
    graph.bind('obo', BFO)
    counts = Counter()
    warnings = set()
    unique_air_trips, unique_rail_trips = set(), set()

    def person(d):
        node = _sid('person', d['person_id'])
        graph.add((node, RDF.type, AGENT.person))
        graph.add((node, RDFS.label, Literal(d['first_name']+' '+d['last_name'])))
        return node

    for record in records:
        d = record.values
        p = person(d)
        trip = _sid('trip', record.mode, d['trip_id'])
        reservation = _sid('reservation_information', record.mode, d['entry_id'])
        graph.add((reservation, RDF.type, AGENT.airline_reservation
                   if record.mode == 'air' else INTEL.rail_reservation_information))
        graph.add((reservation, RDFS.label, Literal(f"{record.mode.upper()} reservation information")))
        graph.add((reservation, AGENT.has_first_name, Literal(d['first_name'], datatype=XSD.string)))
        graph.add((reservation, AGENT.has_last_name, Literal(d['last_name'], datatype=XSD.string)))
        if d['reservation_id']:
            graph.add((reservation, AGENT.has_reservation_number,
                       Literal(d['reservation_id'], datatype=XSD.string)))
        # Manifest is an authoritative observation of the passenger aboard this
        # actual travel process; no passenger-role or threat classification.
        graph.add((p, BFO.BFO_0000056, trip))
        counts['onboard_passenger_rows'] += 1
        if record.mode == 'air':
            unique_air_trips.add(d['trip_id'])
            graph.add((trip, RDF.type, BFO.BFO_0000015))  # actual flight as BFO Process
            graph.add((trip, RDFS.label, Literal('In-transit flight '+d['service_code'])))
            pass_node = _sid('boarding_pass', d['boarding_pass_id'] or d['entry_id'])
            graph.add((pass_node, RDF.type, AGENT.boarding_pass))
            graph.add((pass_node, RDFS.label, Literal('Boarding pass for manifest entry '+d['entry_id'])))
            graph.add((p, AGENT.possesses, pass_node))
            graph.add((pass_node, BFO.BFO_0000101, reservation))
            for col, predicate in [('boarding_gate', AGENT.has_boarding_gate),
                                   ('boarding_group', AGENT.has_boarding_group)]:
                if d[col]:
                    graph.add((reservation, predicate, Literal(d[col], datatype=XSD.string)))
            if d['boarding_time']:
                graph.add((reservation, AGENT.has_boarding_time,
                           Literal(d['boarding_time'], datatype=XSD.dateTime)))
            for col, predicate in [('departure_name', AGENT.has_departure_airport),
                                   ('arrival_name', AGENT.has_arrival_airport)]:
                graph.add((reservation, predicate, Literal(d[col], datatype=XSD.string)))
            plan = _sid('flight_plan', d['trip_id'])
            graph.add((plan, RDF.type, AGENT.Flight_plan))
            graph.add((plan, RDFS.label, Literal('Flight plan for service '+d['service_code'])))
            graph.add((plan, AGENT.has_departure_airport,
                       Literal(d['departure_name'], datatype=XSD.string)))
            graph.add((plan, AGENT.has_arrival_airport,
                       Literal(d['arrival_name'], datatype=XSD.string)))
            aircraft = _sid('airframe', d['aircraft_tail_number'])
            graph.add((aircraft, RDF.type, AGENT.Commcercial_airplane))
            graph.add((aircraft, AGENT.has_tail_number,
                       Literal(d['aircraft_tail_number'], datatype=XSD.string)))
            graph.add((aircraft, RDFS.label,
                       Literal('Airplane tail '+d['aircraft_tail_number'])))
            graph.add((aircraft, BFO.BFO_0000056, trip))
            counts['air_passenger_rows'] += 1
        else:
            unique_rail_trips.add(d['trip_id'])
            if d['physical_ticket_evidence_reference']:
                ticket = _sid('rail_ticket', d['ticket_id'])
                graph.add((ticket, RDF.type, INTEL.rail_ticket))
                graph.add((ticket, RDFS.label,
                           Literal('Physical rail ticket for manifest entry '+d['entry_id'])))
                graph.add((ticket, BFO.BFO_0000101, reservation))
                graph.add((p, AGENT.possesses, ticket))
                counts['physical_rail_ticket_rows'] += 1
            elif d['ticket_id']:
                warnings.add('Ticket IDs do not automatically prove physical rail tickets exist.')
            train = _sid('passenger_train', d['trip_id'])
            graph.add((train, RDF.type, INTEL.passenger_train))
            graph.add((train, INTEL.has_train_service_code,
                       Literal(d['service_code'], datatype=XSD.string)))
            graph.add((train, RDFS.label, Literal('Passenger train service '+d['service_code'])))
            stations = []
            for side in ('departure', 'arrival'):
                station = _sid('railway_station', d[side+'_id'])
                graph.add((station, RDF.type, INTEL.railway_station))
                graph.add((station, RDFS.label, Literal(d[side+'_name'])))
                stations.append(station)
            plan = _sid('rail_journey_plan', d['entry_id'])
            graph.add((plan, RDF.type, INTEL.rail_journey_plan))
            graph.add((plan, RDFS.label,
                       Literal('Rail journey plan for manifest entry '+d['entry_id'])))
            graph.add((plan, INTEL.has_planned_departure_time,
                       Literal(d['departure_time'], datatype=XSD.dateTime)))
            graph.add((reservation, BFO.IAO_0000136, plan))
            graph.add((trip, RDF.type, INTEL.rail_journey))
            graph.add((trip, RDFS.label, Literal('In-transit rail journey '+d['trip_id'])))
            graph.add((trip, INTEL.departs_from_station, stations[0]))
            graph.add((trip, INTEL.arrives_at_station, stations[1]))
            graph.add((trip, INTEL.occurs_aboard_train, train))
            graph.add((plan, INTEL.specifies_rail_journey, trip))
            graph.add((train, BFO.BFO_0000056, trip))
            counts['rail_passenger_rows'] += 1

    counts['unique_air_flights'] = len(unique_air_trips)
    counts['unique_rail_journeys'] = len(unique_rail_trips)
    counts['rows'] = len(records)
    counts['people'] = len({r.values['person_id'] for r in records})
    instance_classes = {AGENT.person, AGENT.boarding_pass, AGENT.airline_reservation,
                        AGENT.Flight_plan, AGENT.Commcercial_airplane, BFO.BFO_0000015,
                        INTEL.rail_ticket, INTEL.rail_reservation_information,
                        INTEL.passenger_train, INTEL.railway_station,
                        INTEL.rail_journey_plan, INTEL.rail_journey}
    individuals = {s for s, typ in graph.subject_objects(RDF.type) if typ in instance_classes}
    for s in individuals:
        graph.add((s, RDF.type, OWL.NamedIndividual))
    counts['new_named_individuals'] = len(individuals)
    counts['abox_triples'] = len(graph)
    return graph, {'counts': dict(counts), 'warnings': sorted(warnings)}

def _turtle(graph: Graph) -> str:
    # There are no blank nodes; sorted N-Triples are also legal Turtle.
    return ("# Project 6 passenger-manifest ABox (local, potentially sensitive passenger data).\n"
            + "# Generated from a validated workbook. Keep access restricted.\n\n"
            + "\n".join(sorted(f"{s.n3()} {p.n3()} {o.n3()} ." for s,p,o in graph)) + "\n")


def _atomic_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='\n',
                                     dir=path.parent, prefix='.tmp_', delete=False) as f:
        tmp = Path(f.name)
        f.write(content)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, prefix='.tmp_', delete=False) as f:
        tmp = Path(f.name)
        f.write(content)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def ingest(manifest: Path, ontology: Path, abox_output: Path, combined_output: Path,
           report_output: Path, *, validate_only: bool = False) -> dict:
    records = parse_manifest(manifest)
    verify_records(records)
    if not ontology.is_file():
        raise ManifestError('Augmented ontology not found: '+str(ontology))
    ontology_hash = _sha_file(ontology)
    manifest_hash = _sha_file(manifest)
    base = Graph()
    try:
        base.parse(ontology, format='turtle')
    except Exception as exc:
        raise ManifestError(f'Cannot parse augmented ontology: {exc}') from exc
    verify_ontology(base, records)
    abox, extras = build_abox(records)
    if any(s in base for s in abox):
        raise ManifestError('Generated ABox overlaps original augmented TBox/source assertions')
    named = set(base.subjects(RDF.type, OWL.NamedIndividual))
    created = set(abox.subjects(RDF.type, OWL.NamedIndividual))
    if named & created:
        raise ManifestError('Generated IRI collides with an existing ontology individual')
    combined = base + abox
    if len(combined) != len(base) + len(abox):
        raise ManifestError('Combined output is not an exact additive union')
    if _sha_file(ontology) != ontology_hash or _sha_file(manifest) != manifest_hash:
        raise ManifestError('Input file changed during ingestion')
    result = {"version": VERSION, "status": "PASS_MANIFEST_VALIDATION_ONLY" if validate_only else "PASS_MANIFEST_ABOX_GENERATED",
              "input_sha256": manifest_hash, "ontology_sha256": ontology_hash,
              "source_and_augmented_ontology_unchanged": True,
              "ontology_triples": len(base), "abox_triples": len(abox), "combined_triples": len(combined),
              "generated_classes_or_properties": 0,
              "no_threat_or_person_of_interest_classification": True,
              "output_abox": str(abox_output), "output_combined": str(combined_output),
              "ontology_reasoner_executed": False,
              "travel_occurrence_policy": "MANIFEST_ONBOARD_IN_TRANSIT: all rows assert actual air/rail process and passenger participation",
              "manifest_truth_assumption": "For this assignment each input row is trusted as a current onboard passenger; no external occurrence or boarding verification",
              **extras}
    if validate_only:
        return result
    abox_text = _turtle(abox)
    _atomic_write(abox_output, abox_text)
    # Preserve the original ontology's exact textual content, including GCAs
    # and named blank-node layout; append only fully expanded ABox statements.
    original_bytes = ontology.read_bytes()
    _atomic_write_bytes(combined_output, original_bytes + (b'' if original_bytes.endswith(b'\n') else b'\n')
                        + b'\n' + abox_text.encode('utf-8'))
    # Verify generated combined output reparses with the expected counts.
    reread = Graph().parse(combined_output, format='turtle')
    if len(reread) != len(combined):
        raise RuntimeError('Combined TTL roundtrip changed triple count')
    if _sha_file(ontology) != ontology_hash:
        raise RuntimeError('Protected augmented ontology changed unexpectedly')
    _atomic_write(report_output, json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Structured .xlsx manifest')
    parser.add_argument('--ontology', type=Path, default=Path('generated/module_augmented.ttl'))
    parser.add_argument('--output', type=Path, default=Path('generated/manifest_instances.ttl'))
    parser.add_argument('--combined-output', type=Path, default=Path('generated/module_augmented_with_manifest.ttl'))
    parser.add_argument('--report', type=Path, default=Path('reports/manifest_ingestion_report.json'))
    parser.add_argument('--validate-only', action='store_true', help='No output files written')
    args = parser.parse_args(argv)
    try:
        report = ingest(args.input, args.ontology, args.output, args.combined_output, args.report,
                        validate_only=args.validate_only)
        print('MANIFEST INGEST — '+report['status'])
        print(json.dumps({k:report[k] for k in ['ontology_triples','abox_triples','combined_triples','counts','warnings']}, indent=2))
        return 0
    except (ManifestError, FileNotFoundError) as exc:
        print('MANIFEST INGEST — FAIL: '+str(exc), file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
