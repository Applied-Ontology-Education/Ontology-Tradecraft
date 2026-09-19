from rdflib import Graph

g = Graph()
g.parse('/tmp/iao.owl', format='xml')  # RDF/XML is the source format

print(f"Parsed {len(g)} triples")

g.serialize(destination='/tmp/iao.ttl', format='turtle')
print("Saved as Turtle at /tmp/iao.ttl")