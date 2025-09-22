Project 3 – Ontology Structural Matches

This project focused on working with several ontologies (BFO, IES, CCOM, QUDT, CCOT, and OWL-Time) and trying to identify structural overlaps between them. 
The workflow involved exporting class and axiom information into spreadsheets, running structural match queries, and then polishing the results into something consistent and readable.

I built out Excel files for each ontology with IRIs, labels, and axioms. Instead of leaving everything as “subClassOf owl:Thing,” I went back and aligned them with the actual ontology hierarchy — for example, making ProperInterval a subclass of Interval in OWL-Time, or Process a subclass of Occurrent in BFO. 
Labels were cleaned up and I added quick clarifications in plain English so they’re easier to understand (e.g., “Instant (a single moment in time)”). That makes the files a little more useful for a human reader, not just something for a reasoner to parse.

On the mapping side, I created TTL files that connect concepts across the ontology pairs (BFO ↔ IES, CCOM ↔ QUDT, CCOT ↔ Time). 
I used owl:equivalentClass when two things are basically the same idea (like Process in BFO and IES), and rdfs:subClassOf when one is broader or narrower (like ProperInterval under Interval).

The tests check that all the required deliverables are present, have the right structure, and parse correctly — everything passes now. 
What I learned from this is that structural matching is only a starting point: the real work is deciding which mappings actually make sense and writing them in a way that both machines and people can understand.