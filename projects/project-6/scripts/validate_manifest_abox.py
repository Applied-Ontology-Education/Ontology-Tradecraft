#!/usr/bin/env python3
"""Project 6: Validate the *enriched* manifest TBox/ABox overlay.

Uses the same six LOCAL import files as Project 6 Step 6, extracts a fresh
ROBOT STAR locality module, then runs HermiT on a flattened ontology-only
baseline and on that baseline plus the manifest ABox. No network imports.

Requires earlier scripts/ingest_passenger_manifest.py and
scripts/enrich_manifest_relationships.py, and their generated outputs.

    python scripts/validate_manifest_abox.py --workbook templates/passenger_manifest_template.xlsx --shacl-only
    python scripts/validate_manifest_abox.py --workbook templates/passenger_manifest_template.xlsx

The script NEVER changes the protected ontology, manifest, or Excel files.
OWL coherence checks consistency only; in-transit and aboard assertions are
accepted as facts of the synthetic assignment manifest, not separately proven.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.compare import isomorphic, to_canonical_graph
from rdflib.namespace import RDF, RDFS, OWL, XSD

from ingest_passenger_manifest import (
    AGENT, BFO, INTEL, _sid, _sha_file, build_abox,
    parse_manifest, verify_records, verify_ontology,
)
from enrich_manifest_relationships import (
    IS_ABOUT, IS_CARRIER_OF, PARTICIPATES_IN,
    NEW_DATA_PROPS, NEW_OBJECT_PROPS, _schema_graph,
)

VERSION = 'project6-manifest-semantic-validation-v3-in-transit'
SH = Namespace('http://www.w3.org/ns/shacl#')
TMP_NS = Namespace('https://example.org/ontologies/project6/manifest-validation#')
SKIP_VOCAB = (
    str(RDF), str(RDFS), str(OWL), str(XSD),
)
EXPECTED_IMPORTS = frozenset({
    'AgentOntology.ttl', 'bfo-core.ttl', 'core.ttl', 'EventOntology.ttl',
    'iao-edit.owl', 'InformationEntityOntology.ttl'
})

class ValidationError(Exception):
    pass


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(content, encoding='utf-8')
    temp.replace(path)


def atomic_json(path: Path, data: dict) -> None:
    atomic_text(path, json.dumps(data, indent=2, sort_keys=True) + '\n')


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValidationError(f'Missing upstream report: {path}')
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise ValidationError(f'Cannot read {path}: {exc}') from exc
    if not isinstance(data, dict):
        raise ValidationError(f'Report is not a JSON object: {path}')
    return data


def _only(graph: Graph, subject: URIRef, predicate: URIRef) -> Any:
    vals = set(graph.objects(subject, predicate))
    if len(vals) != 1:
        raise ValidationError(f'Expected exactly one {predicate} on {subject}; found {len(vals)}')
    return next(iter(vals))


def _require_triple(graph: Graph, s, p, o) -> None:
    if (s, p, o) not in graph:
        raise ValidationError(f'Missing expected assertion: {s} {p} {o}')


def load_inputs(root: Path, workbook: Path) -> dict:
    root = root.resolve()
    workbook = workbook.resolve()
    paths = {
        'source': root / 'src/module.ttl',
        'augmented': root / 'generated/module_augmented.ttl',
        'initial_abox': root / 'generated/manifest_instances.ttl',
        'schema': root / 'generated/manifest_tbox_extension.ttl',
        'enriched_abox': root / 'generated/manifest_instances_enriched.ttl',
        'combined': root / 'generated/module_augmented_with_manifest_enriched.ttl',
        'workbook': workbook,
        'ingest_report': root / 'reports/manifest_ingestion_report.json',
        'enrichment_report': root / 'reports/manifest_relationship_report.json',
    }
    for label, path in paths.items():
        if not path.is_file():
            raise ValidationError(f'Missing {label} file: {path}')
    hashes = {name: file_hash(path) for name, path in paths.items()}
    ingest = read_json(paths['ingest_report'])
    enrichment = read_json(paths['enrichment_report'])
    if ingest.get('status') != 'PASS_MANIFEST_ABOX_GENERATED':
        raise ValidationError('Initial ingestion was not completed successfully')
    if enrichment.get('status') != 'PASS_RELATIONSHIPS_ENRICHED_NOT_MERGED':
        raise ValidationError('Enrichment has not generated its validated output')
    # Validate all report hashes against actual protected files.
    for report, field, value in (
        (ingest, 'input_sha256', hashes['workbook']),
        (ingest, 'ontology_sha256', hashes['augmented']),
        (enrichment, 'workbook_sha256', hashes['workbook']),
        (enrichment, 'source_sha256', hashes['augmented']),
        (enrichment, 'original_abox_sha256', hashes['initial_abox']),
    ):
        if report.get(field) != value:
            raise ValidationError(f'Upstream report hash mismatch: {field}')
    if ingest.get('source_and_augmented_ontology_unchanged') is not True:
        raise ValidationError('Ingestion did not attest unchanged source/augmented inputs')
    if enrichment.get('protected_ontology_and_manifest_files_unchanged') is not True:
        raise ValidationError('Enrichment did not attest unchanged protected inputs')
    if not paths['augmented'].read_bytes().startswith(paths['source'].read_bytes()):
        raise ValidationError('Augmented ontology does not retain the source file as byte-prefix')
    if not paths['combined'].read_bytes().startswith(paths['augmented'].read_bytes()):
        raise ValidationError('Combined ontology does not retain the augmented ontology byte-prefix')
    graphs = {name: Graph().parse(paths[name], format='turtle')
              for name in ('source', 'augmented', 'initial_abox', 'schema', 'enriched_abox', 'combined')}
    if len(graphs['source']) != 421 or len(graphs['augmented']) != 484:
        raise ValidationError('Protected ontology sizes differ from reviewed Project 6 source/augmentation')
    if not isomorphic(graphs['schema'], _schema_graph(graphs['augmented'])):
        raise ValidationError('Property schema extension differs from the explicit reviewed enrichment code')
    if not set(graphs['initial_abox']).issubset(set(graphs['enriched_abox'])):
        raise ValidationError('Enriched ABox removed an original manifest assertion')
    union = graphs['augmented'] + graphs['schema'] + graphs['enriched_abox']
    if not isomorphic(union, graphs['combined']):
        raise ValidationError('Combined ontology is not the exact RDF union of approved inputs')
    if len(union) != len(graphs['augmented']) + len(graphs['schema']) + len(graphs['enriched_abox']):
        raise ValidationError('TBox/ABox union contains unexpected overlapping assertions')
    records = parse_manifest(workbook)
    verify_records(records)
    verify_ontology(graphs['augmented'], records)
    regenerated, _ = build_abox(records)
    if set(regenerated) != set(graphs['initial_abox']):
        raise ValidationError('Initial ABox does not match the workbook on this run')
    if ingest.get('abox_triples') != len(graphs['initial_abox']):
        raise ValidationError('Initial ABox triple count mismatches ingestion report')
    for key, actual in (
        ('new_tbox_triples', len(graphs['schema'])),
        ('new_abox_triples', len(graphs['enriched_abox']) - len(graphs['initial_abox'])),
        ('combined_triples', len(graphs['combined'])),
    ):
        if enrichment.get(key) != actual:
            raise ValidationError(f'Enrichment report {key} differs from generated output')
    return dict(root=root, workbook=workbook, paths=paths, hashes=hashes,
                graphs=graphs, records=records, ingest=ingest, enrichment=enrichment)


def validate_assertions(ctx: dict, *, strict_occurrence: bool = False) -> dict:
    """Verify all manifest passengers and vehicles participate in actual journeys."""
    g = ctx['graphs']['combined']
    abox = ctx['graphs']['enriched_abox']
    records = ctx['records']
    counters = defaultdict(int)
    air_trips = {r.values['trip_id']: r.values for r in records if r.mode=='air'}
    rail_trips = {r.values['trip_id']: r.values for r in records if r.mode=='rail'}
    for tid, d in sorted(air_trips.items()):
        process = _sid('trip', 'air', tid)
        aircraft = _sid('airframe', d['aircraft_tail_number'])
        plan = _sid('flight_plan', tid)
        act = _sid('act_of_flight_plan', tid)
        _require_triple(abox, process, RDF.type, BFO.BFO_0000015)
        _require_triple(abox, process, RDF.type, INTEL.air_flight)
        _require_triple(abox, aircraft, RDF.type, AGENT.Commcercial_airplane)
        _require_triple(abox, aircraft, PARTICIPATES_IN, process)
        _require_triple(abox, process, INTEL.occurs_aboard_airplane, aircraft)
        _require_triple(abox, process, INTEL.has_flight_plan_information, plan)
        _require_triple(abox, plan, RDF.type, AGENT.Flight_plan)
        _require_triple(abox, act, RDF.type, AGENT.Act_of_Flight_Plan)
        _require_triple(abox, plan, IS_ABOUT, act)
        _require_triple(abox, aircraft, PARTICIPATES_IN, act)
        _require_triple(abox, plan, INTEL.has_flight_service_code,
                        Literal(d['service_code'], datatype=XSD.string))
        _require_triple(abox, plan, INTEL.has_planned_air_departure_time,
                        Literal(d['departure_time'], datatype=XSD.dateTime))
        counters['actual_air_flights'] += 1
        counters['aircraft_actual_flight_participations'] += 1
        counters['air_planning_acts'] += 1
    for tid, d in sorted(rail_trips.items()):
        proc = _sid('trip','rail',tid)
        train = _sid('passenger_train',tid)
        dep = _sid('railway_station',d['departure_id'])
        arr = _sid('railway_station',d['arrival_id'])
        _require_triple(abox, proc, RDF.type, INTEL.rail_journey)
        _require_triple(abox, train, RDF.type, INTEL.passenger_train)
        _require_triple(abox, train, PARTICIPATES_IN, proc)
        _require_triple(abox, proc, INTEL.occurs_aboard_train, train)
        _require_triple(abox, proc, INTEL.departs_from_station, dep)
        _require_triple(abox, proc, INTEL.arrives_at_station, arr)
        _require_triple(abox, train, INTEL.has_train_departure_station_name,
                        Literal(d['departure_name'], datatype=XSD.string))
        _require_triple(abox, train, INTEL.has_train_arrival_station_name,
                        Literal(d['arrival_name'], datatype=XSD.string))
        _require_triple(abox, dep, RDFS.label, Literal(d['departure_name']))
        _require_triple(abox, arr, RDFS.label, Literal(d['arrival_name']))
        for r in (r for r in records if r.mode=='rail' and r.values['trip_id']==tid):
            plan = _sid('rail_journey_plan',r.values['entry_id'])
            _require_triple(abox, plan, INTEL.specifies_rail_journey, proc)
            _require_triple(abox, plan, INTEL.plans_to_use_train, train)
            _require_triple(abox, plan, INTEL.has_planned_departure_station, dep)
            _require_triple(abox, plan, INTEL.has_planned_arrival_station, arr)
            if r.values['physical_ticket_evidence_reference']:
                ticket = _sid('rail_ticket',r.values['ticket_id'])
                _require_triple(abox, ticket, INTEL.has_physical_rail_ticket_evidence_reference,
                                Literal(r.values['physical_ticket_evidence_reference'],datatype=XSD.string))
            counters['rail_plan_train_links'] += 1
            counters['rail_plan_station_pairs'] += 1
        counters['actual_rail_journeys'] += 1
        counters['rail_train_actual_participations'] += 1
    for r in records:
        d=r.values
        person = _sid('person',d['person_id'])
        process = _sid('trip',r.mode,d['trip_id'])
        _require_triple(abox, person, RDF.type, AGENT.person)
        _require_triple(abox, person, PARTICIPATES_IN, process)
        _require_triple(abox, process, RDF.type, BFO.BFO_0000015 if r.mode=='air' else INTEL.rail_journey)
        counters['manifest_passenger_participations'] += 1
    # All TBox-extension property assertions must have explicitly typed values.
    for prop, (dom, ran, *_rest) in NEW_OBJECT_PROPS.items():
        for subject, target in abox.subject_objects(prop):
            if not isinstance(target, URIRef) or (subject,RDF.type,dom) not in g or (target,RDF.type,ran) not in g:
                raise ValidationError(f'New object-property domain/range violation: {prop}')
            counters['verified_new_object_property_assertions'] += 1
    for prop, (dom, datatype, *_rest) in NEW_DATA_PROPS.items():
        for subject, value in abox.subject_objects(prop):
            if (subject,RDF.type,dom) not in g or not isinstance(value,Literal) or value.datatype != datatype:
                raise ValidationError(f'New datatype-property domain/range violation: {prop}')
            counters['verified_new_datatype_property_assertions'] += 1
    for subject in abox.subjects(RDF.type, BFO.IAO_0000178):
        if (subject, RDF.type, OWL.NamedIndividual) in abox:
            targets = set(abox.objects(subject, IS_CARRIER_OF))
            if len(targets)!=1 or any((target,RDF.type,AGENT.Flight_plan) not in abox for target in targets):
                raise ValidationError('Physical flight-plan document does not carry a typed Flight_plan ICE')
            counters['physical_flight_plan_documents'] += 1
    # strict_occurrence remains CLI-compatible; the same strict manifest rule
    # is enforced by default. The flag does not require external evidence.
    return {'status':'PASS_ONBOARD_IN_TRANSIT_ASSERTIONS', 'counts':dict(counters),
            'warnings':[], 'warning_count':0,
            'review_note': 'In-transit status and passenger presence are trusted manifest assertions for this assignment; not independently verified.'}

def _add_property_shape(shapes: Graph, parent: URIRef, path: URIRef,
                        *, minimum: int = 1, maximum: int | None = 1,
                        cls: URIRef | None = None, datatype: URIRef | None = None) -> None:
    s = BNode()
    shapes.add((parent, SH.property, s))
    shapes.add((s, RDF.type, SH.PropertyShape))
    shapes.add((s, SH.path, path))
    shapes.add((s, SH.minCount, Literal(minimum)))
    if maximum is not None:
        shapes.add((s, SH.maxCount, Literal(maximum)))
    if cls is not None:
        shapes.add((s, SH['class'], cls))
        shapes.add((s, SH.nodeKind, SH.IRI))
    if datatype is not None:
        shapes.add((s, SH.datatype, datatype))


def build_shapes(enriched_abox: Graph) -> Graph:
    """SHACL only targets manifest individuals, not older source individuals."""
    shapes = Graph()
    configs = [
        (AGENT.Flight_plan, [
            (IS_ABOUT, {'cls': AGENT.Act_of_Flight_Plan}),
            (INTEL.has_flight_service_code, {'datatype': XSD.string}),
            (INTEL.has_planned_air_departure_time, {'datatype': XSD.dateTime})]),
        (INTEL.passenger_train, [
            (INTEL.has_train_departure_station_name, {'datatype': XSD.string}),
            (INTEL.has_train_arrival_station_name, {'datatype': XSD.string})]),
        (INTEL.rail_journey_plan, [
            (INTEL.plans_to_use_train, {'cls': INTEL.passenger_train}),
            (INTEL.has_planned_departure_station, {'cls': INTEL.railway_station}),
            (INTEL.has_planned_arrival_station, {'cls': INTEL.railway_station})]),
        (INTEL.air_flight, [
            (INTEL.occurs_aboard_airplane, {'cls':AGENT.Commcercial_airplane}),
            (INTEL.has_flight_plan_information, {'cls':AGENT.Flight_plan})]),
        (INTEL.rail_journey, [
            (INTEL.occurs_aboard_train, {'cls': INTEL.passenger_train}),
            (INTEL.departs_from_station, {'cls': INTEL.railway_station}),
            (INTEL.arrives_at_station, {'cls': INTEL.railway_station})]),
        (AGENT.Act_of_Flight_Plan, []),
    ]
    for idx, (cls, props) in enumerate(configs):
        nodes = set(enriched_abox.subjects(RDF.type, cls))
        if not nodes:
            continue
        node_shape = TMP_NS[f'node-shape-{idx}']
        shapes.add((node_shape, RDF.type, SH.NodeShape))
        for node in nodes:
            shapes.add((node_shape, SH.targetNode, node))
        shapes.add((node_shape, SH['class'], cls))
        for prop, kwargs in props:
            _add_property_shape(shapes, node_shape, prop, **kwargs)
    # SHACL Shape for the new material flight-plan document individuals.
    documents = {s for s in enriched_abox.subjects(RDF.type, BFO.IAO_0000178)
                 if (s, RDF.type, OWL.NamedIndividual) in enriched_abox}
    if documents:
        shape = TMP_NS['material-flight-plan-document']
        shapes.add((shape, RDF.type, SH.NodeShape))
        for node in documents:
            shapes.add((shape, SH.targetNode, node))
        _add_property_shape(shapes, shape, IS_CARRIER_OF, cls=AGENT.Flight_plan)
    return shapes


def shacl_validate(combined: Graph, shapes: Graph) -> dict:
    try:
        from pyshacl import validate as shacl
    except ImportError as exc:
        raise ValidationError('pyshacl is not installed in the active .venv-mowl environment') from exc
    conforms, report_graph, report_text = shacl(
        data_graph=combined, shacl_graph=shapes,
        inference='none', advanced=False, abort_on_first=False)
    failures = list(report_graph.subjects(RDF.type, SH.ValidationResult))
    return {'conforms': bool(conforms), 'violation_count': len(failures),
            'report_text': str(report_text)}


def _stable(g: Graph, key: str) -> Graph:
    """Deterministic per-file BNode renaming to avoid import-vs-source collisions."""
    result = Graph()
    prefix = 'g' + hashlib.sha256(key.encode()).hexdigest()[:10] + '_'
    for s, p, o in to_canonical_graph(g):
        result.add(tuple(BNode(prefix + str(x)) if isinstance(x, BNode) else x
                         for x in (s, p, o)))
    return result


def _flat_parts(parts: list[tuple[str, Graph]]) -> Graph:
    out = Graph()
    for key, graph in parts:
        stable = _stable(graph, key)
        ontology_subjects = set(stable.subjects(RDF.type, OWL.Ontology))
        for s, p, o in stable:
            if p != OWL.imports and s not in ontology_subjects:
                out.add((s, p, o))
    out.add((TMP_NS['offline-flattened-ontology'], RDF.type, OWL.Ontology))
    return out


def _serial_flat(g: Graph) -> str:
    return ('# Offline ROBOT/HermiT input only; do not replace user ontology.\n'
            + '\n'.join(sorted(f'{s.n3()} {p.n3()} {o.n3()} .' for s, p, o in g)) + '\n')


def robot_command(robot: str | None, robot_jar: Path | None, java: str) -> list[str]:
    if robot_jar is not None:
        if not robot_jar.is_file():
            raise ValidationError(f'ROBOT JAR not found: {robot_jar}')
        if shutil.which(java) is None and not Path(java).is_file():
            raise ValidationError('Java executable not found')
        return [java, '-jar', str(robot_jar.resolve())]
    exe = shutil.which(robot or 'robot')
    if exe is None:
        raise ValidationError('ROBOT is not on PATH; use --robot-jar PATH_TO_ROBOT_JAR')
    return [exe]


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == 'nt':
        try:
            subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                           capture_output=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError, PermissionError):
            pass
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def run_robot(cmd: list[str], log: Path, timeout: int, output: Path) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    started = time.monotonic()
    rc = None
    timed_out = False
    print('MANIFEST VALIDATION — ROBOT:', ' '.join(cmd[1:4]), '->', log.name, flush=True)
    with log.open('w', encoding='utf-8') as stream:
        stream.write('COMMAND: ' + ' '.join(cmd) + '\n')
        stream.flush()
        kwargs = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
                  if os.name == 'nt' else {'start_new_session': True})
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=stream,
                                    stderr=subprocess.STDOUT, **kwargs)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate(proc)
        except OSError as exc:
            stream.write(f'\nROBOT launch failed: {exc}\n')
        stream.write(f'\nRETURN CODE: {rc}\nTIMEOUT: {timed_out}\n')
    result = {'command': cmd, 'return_code': rc, 'timeout': timed_out,
              'duration_seconds': round(time.monotonic() - started, 3),
              'log': str(log), 'output': str(output)}
    if rc != 0 or not output.is_file() or not output.stat().st_size:
        raise ValidationError('ROBOT execution failed; inspect ' + str(log))
    return result


def load_import(path: Path) -> Graph:
    fmt = 'xml' if path.suffix.lower() in {'.owl', '.rdf', '.xml'} else 'turtle'
    return Graph().parse(path, format=fmt)


def _scope_imports(ctx: dict, robot_cmd: list[str], timeout: int) -> tuple[Graph, dict]:
    root = ctx['root']
    imported_paths = sorted((root / 'imports').rglob('*'))
    imported_paths = [p for p in imported_paths if p.is_file() and p.name in EXPECTED_IMPORTS]
    names = [p.name for p in imported_paths]
    if len(imported_paths) != 6 or set(names) != EXPECTED_IMPORTS:
        raise ValidationError('Expected exactly the six Project 6 local imports in imports/')
    extra = [p for p in (root / 'imports').rglob('*')
             if p.is_file() and p.suffix.lower() in {'.ttl', '.owl', '.rdf', '.xml'} and p.name not in EXPECTED_IMPORTS]
    if extra:
        raise ValidationError('Unexpected additional ontology imports: ' + ', '.join(p.name for p in extra))
    print('MANIFEST VALIDATION — parsing six local imports and building STAR locality module', flush=True)
    imports = [(f'imports/{p.name}', load_import(p)) for p in imported_paths]
    imported = _flat_parts(imports)
    g = ctx['graphs']
    used = {str(item) for triple in g['combined'] for item in triple if isinstance(item, URIRef)}
    available = {str(item) for triple in imported for item in triple if isinstance(item, URIRef)}
    seeds = sorted(u for u in used & available if not u.startswith(SKIP_VOCAB))
    if not seeds:
        raise ValidationError('No imported ontology IRIs overlap with the manifest graph')
    directory = root / 'generated/manifest_validation'
    directory.mkdir(parents=True, exist_ok=True)
    corpus = directory / 'local_imports_corpus.ttl'
    terms = directory / 'import_seed_terms.txt'
    module_file = directory / 'scoped_imports_module.ttl'
    atomic_text(corpus, _serial_flat(imported))
    atomic_text(terms, '\n'.join(seeds) + '\n')
    extraction = run_robot(
        robot_cmd + ['extract', '--method', 'STAR', '--input', str(corpus),
                     '--term-file', str(terms), '--imports', 'exclude',
                     '--individuals', 'definitions', '--output', str(module_file)],
        root / 'reports/manifest_validation/import_extraction.log', timeout,
        module_file)
    result = load_import(module_file)
    if not result:
        raise ValidationError('ROBOT STAR module was empty')
    return result, {'scope': 'SOURCE_PLUS_STAR_LOCALITY_MODULE_OF_SIX_LOCAL_IMPORTS',
                    'import_files': {str(p.relative_to(root)): file_hash(p) for p in imported_paths},
                    'all_import_corpus_triples': len(imported), 'import_module_triples': len(result),
                    'seed_count': len(seeds), 'extract': extraction,
                    'not_full_import_closure': True}


def run_reasoning(ctx: dict, *, robot: str | None, robot_jar: Path | None,
                  java: str, timeout: int) -> dict:
    command = robot_command(robot, robot_jar, java)
    module, metadata = _scope_imports(ctx, command, timeout)
    root = ctx['root']
    g = ctx['graphs']
    baseline = _flat_parts([
        ('generated/module_augmented.ttl', g['augmented']),
        ('generated/manifest_tbox_extension.ttl', g['schema']),
        ('generated/manifest_validation/scoped_imports_module.ttl', module)])
    abox = _stable(g['enriched_abox'], 'generated/manifest_instances_enriched.ttl')
    candidate = Graph()
    candidate += baseline
    candidate += abox
    if len(candidate) != len(baseline) + len(abox):
        raise ValidationError('OWL reasoning overlay lost or overlapped ABox facts')
    work = root / 'generated/manifest_validation'
    a = work / 'reasoner_baseline.ttl'
    b = work / 'reasoner_candidate.ttl'
    atomic_text(a, _serial_flat(baseline))
    atomic_text(b, _serial_flat(candidate))
    if len(load_import(a)) != len(baseline) or len(load_import(b)) != len(candidate):
        raise ValidationError('Flattened ontology round-trip lost triples')
    report_dir = root / 'reports/manifest_validation'
    first = run_robot(command + ['reason', '--input', str(a), '--reasoner', 'hermit',
                                 '--output', str(work / 'reasoned_baseline.ttl')],
                      report_dir / 'baseline_hermit.log', timeout, work / 'reasoned_baseline.ttl')
    second = run_robot(command + ['reason', '--input', str(b), '--reasoner', 'hermit',
                                  '--output', str(work / 'reasoned_candidate.ttl')],
                       report_dir / 'candidate_hermit.log', timeout, work / 'reasoned_candidate.ttl')
    return {**metadata, 'baseline_triples': len(baseline),
            'candidate_triples': len(candidate), 'abox_delta_triples': len(abox),
            'baseline_reasoner': first, 'candidate_reasoner': second,
            'owl_reasoner_executed': True,
            'note': 'ROBOT exit 0 is evidence of scoped HermiT success, not global import-complete consistency or truth of asserted travel.'}


def validate(root: Path, workbook: Path, *, shacl_only: bool = False,
             strict_occurrence: bool = False,
             robot: str | None = None, robot_jar: Path | None = None,
             java: str = 'java', timeout: int = 300) -> dict:
    ctx = load_inputs(root, workbook)
    first = validate_assertions(ctx, strict_occurrence=strict_occurrence)
    shapes = build_shapes(ctx['graphs']['enriched_abox'])
    shacl_result = shacl_validate(ctx['graphs']['combined'], shapes)
    if not shacl_result['conforms']:
        raise ValidationError('SHACL reported ' + str(shacl_result['violation_count']) + ' violations:\n'
                              + shacl_result['report_text'][:2000])
    schema_sha = file_hash(ctx['paths']['schema'])
    print('MANIFEST VALIDATION — structural assertions and SHACL passed', flush=True)
    reasoned = None
    if not shacl_only:
        reasoned = run_reasoning(ctx, robot=robot, robot_jar=robot_jar, java=java, timeout=timeout)
    for key, path in ctx['paths'].items():
        if file_hash(path) != ctx['hashes'][key]:
            raise ValidationError(f'Protected input changed during validation: {key}')
    status = ('PASS_STRUCTURAL_SHACL_UNREASONED' if shacl_only else
              'PASS_SCOPED_HERMIT_AND_SHACL_WITH_WARNINGS' if first['warnings'] else
              'PASS_SCOPED_HERMIT_AND_SHACL')
    report = {
        'version': VERSION, 'status': status,
        'counts': {
            'augmented_tbox_triples': len(ctx['graphs']['augmented']),
            'new_property_declaration_triples': len(ctx['graphs']['schema']),
            'manifest_enriched_abox_triples': len(ctx['graphs']['enriched_abox']),
            'combined_triples': len(ctx['graphs']['combined']),
        },
        'protected_hashes': ctx['hashes'],
        'source_unchanged': True,
        'new_schema_sha256': schema_sha,
        'manifest_assertions': first,
        'shacl': {'conforms': True, 'target_shape_count': len(set(shapes.subjects(RDF.type, SH.NodeShape))),
                  'triple_count': len(shapes), 'violations': 0},
        'reasoning': reasoned if reasoned else {'owl_reasoner_executed': False},
        'limitations': [
            'Source-scoped import STAR module is not complete imported-ontology closure.',
            'OWL/SHACL validation does not independently prove journey occurrence, aircraft movement, or passenger ownership.',
            'In-transit travel and physical passenger presence are trusted assumptions from the manifest, not independently established by the reasoner.',
            'Physical-ticket evidence remains optional and is not independently verified.',
        ],
        'source_or_augmented_overwritten': False,
    }
    atomic_text(ctx['root'] / 'generated/manifest_validation/manifest_validation_shapes.ttl',
                shapes.serialize(format='turtle'))
    atomic_json(ctx['root'] / 'reports/manifest_semantic_validation.json', report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--workbook', type=Path, required=True)
    parser.add_argument('--shacl-only', action='store_true', help='Structural and SHACL; no ROBOT/HermiT')
    parser.add_argument('--strict-occurrence', action='store_true',
                        help='Compatibility flag: require every manifest passenger and vehicle to participate (already default)')
    parser.add_argument('--robot', help='Executable (defaults to robot on PATH)')
    parser.add_argument('--robot-jar', type=Path, help='Use java -jar instead of robot command')
    parser.add_argument('--java', default='java')
    parser.add_argument('--timeout', type=int, default=300, help='Seconds per ROBOT stage')
    parser.add_argument('--hard-timeout', type=int, default=1200, help='Overall watchdog limit in seconds')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.timeout < 30 or args.hard_timeout < 30:
        parser.error('--timeout and --hard-timeout must be at least 30 seconds')
    if not args._worker and not args.shacl_only:
        cmd = [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:] + ['--_worker']
        kwargs = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
                  if os.name == 'nt' else {'start_new_session': True})
        print('MANIFEST VALIDATION — supervising HermiT/ROBOT execution', flush=True)
        try:
            child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, **kwargs)
            try:
                return child.wait(timeout=args.hard_timeout)
            except subprocess.TimeoutExpired:
                _terminate(child)
                atomic_json(args.project_root.resolve() / 'reports/manifest_semantic_validation.json',
                            {'version': VERSION, 'status': 'FAIL_HARD_TIMEOUT',
                             'hard_timeout_seconds': args.hard_timeout, 'owl_reasoner_passed': False})
                print('MANIFEST VALIDATION — FAIL_HARD_TIMEOUT', file=sys.stderr, flush=True)
                return 1
        except OSError as exc:
            print('MANIFEST VALIDATION — FAIL to start worker: ' + str(exc), file=sys.stderr)
            return 1
    try:
        root = args.project_root.resolve()
        wb = args.workbook if args.workbook.is_absolute() else root / args.workbook
        report = validate(root, wb, shacl_only=args.shacl_only,
                          strict_occurrence=args.strict_occurrence,
                          robot=args.robot, robot_jar=args.robot_jar, java=args.java,
                          timeout=args.timeout)
        print('MANIFEST VALIDATION — ' + report['status'], flush=True)
        print(json.dumps({'counts': report['counts'], 'shacl': report['shacl'],
                          'semantic_warnings': report['manifest_assertions']['warnings'],
                          'reasoner_executed': report['reasoning']['owl_reasoner_executed']}, indent=2))
        return 0
    except (ValidationError, ValueError, OSError, RuntimeError) as exc:
        try:
            atomic_json(args.project_root.resolve() / 'reports/manifest_semantic_validation.json',
                        {'version': VERSION, 'status': 'FAIL_VALIDATION',
                         'error': str(exc), 'owl_reasoner_passed': False})
        except OSError:
            pass
        print('MANIFEST VALIDATION — FAIL: ' + str(exc), file=sys.stderr, flush=True)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
