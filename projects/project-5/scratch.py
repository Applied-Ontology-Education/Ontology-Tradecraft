from rdflib import Graph, URIRef

g = Graph()
g.add((URIRef("http://example.org/a"), URIRef("http://example.org/b"), URIRef("http://example.org/c")))
g.add((URIRef("http://example.org/a"), URIRef("http://example.org/b"), URIRef("http://example.org/c")))
g.serialize("test.ttl", format="turtle")
