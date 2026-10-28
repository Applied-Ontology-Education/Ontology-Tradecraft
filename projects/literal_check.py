from rdflib import Graph

g = Graph()
g.parse('/Users/brianavann/repos/Ontology-Tradecraft/projects/project-2/assignment/src/data/modules/InformationEntityOntology.ttl', format='turtle')

q = """
SELECT ?s ?p ?o WHERE {
  ?s ?p ?o .
  FILTER(isLiteral(?o))
  BIND(DATATYPE(?o) AS ?dt)
  FILTER(!BOUND(?dt) && !langMatches(lang(?o), "*"))
}
"""

results = list(g.query(q))
print(f'Untyped literals found: {len(results)}')
for r in results[:10]:
    print(r)