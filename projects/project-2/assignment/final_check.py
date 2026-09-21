import signal
from rdflib import Graph
import glob

def handler(signum, frame):
    raise TimeoutError('timed out')
signal.signal(signal.SIGALRM, handler)

g = Graph()
g.parse('src/data/core.ttl', format='turtle')
g.parse('src/data/instances.ttl', format='turtle')
for f in glob.glob('src/data/modules/*.ttl'):
    g.parse(f, format='turtle')

from rdflib.namespace import RDF, RDFS, OWL, SKOS, XSD
INIT_NS = {'rdf': RDF, 'rdfs': RDFS, 'owl': OWL, 'skos': SKOS, 'xsd': XSD}

# 1. Untyped literal scan
q_untyped = """
SELECT ?s ?p ?o WHERE {
  ?s ?p ?o .
  FILTER(isLiteral(?o))
  BIND(DATATYPE(?o) AS ?dt)
  FILTER(!BOUND(?dt) && !langMatches(lang(?o), "*"))
} LIMIT 5
"""
signal.alarm(20)
try:
    rows = list(g.query(q_untyped, initNs=INIT_NS))
    signal.alarm(0)
    print('untyped literals found:', len(rows))
    for r in rows:
        print(' ', r)
except TimeoutError:
    print('untyped literal check TIMED OUT')

# 2. All key terms have label + definition
key_terms = [
    'Artifact Design', 'Jet Engine', 'Fixed Wing',
    'has continuant part', 'has Model BADA', 'has Model FAA',
    'has Length Ft', 'has Tail Height Ft', 'is about',
    'Test Process', 'is temporal part of', 'is measurement unit of',
    'prescribes', 'Measurement Unit',
    'Act of Aircraft Processing', 'Adaptability Evaluation',
    'quality', 'Act of Measuring', 'Total Employment',
    'has area', 'o group', 'own code',
]
missing = []
for term in key_terms:
    q = f"""
    ASK {{
      ?s rdfs:label ?l .
      FILTER(LCASE(STR(?l)) = LCASE("{term}"))
      ?s skos:definition ?d .
    }}
    """
    signal.alarm(10)
    try:
        ok = g.query(q, initNs=INIT_NS).askAnswer
        signal.alarm(0)
        if not ok:
            missing.append(term)
    except TimeoutError:
        missing.append(term + ' (TIMED OUT)')

print('Missing label+definition for:', missing if missing else 'NONE - all present')