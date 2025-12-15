# Add a new OWL restriction to an existing OWL class

This template uses a Markdown format for human readability.

# Instructions

- Identify the phrases in single quotes in the Class definition in English.
- Verb phrases in the English definition refer to OWL properties.
- Noun phrases in the English definition refer to OWL classes.
- The combination of a verb phrase and a noun phrase is a possible OWL restriction.
- Use the Mapping of phrases to URIREF below to lookup the URIREF for the phrases.
- Using the URIRefs, see if a similar restriction is present in the Current OWL definition.
- If the restriction is not present, propose a new OWL restriction.
- To create this restriction, use the New OWL restriction template.

# Class definition in English

" This entity 'is part' of a 'Document'

# Mapping of verb and noun phrases to URIREF

```CSV
Label, URIREF
'is part of': <http://purl.obolibrary.org/obo/BFO_0000050>
'Document': <http://purl.obolibrary.org/obo/BFO_0000050>
...
```

# Current OWL definition

```Turtle
:entity_with_continuant_part a owl:Class ;
    rdfs:label "Entity with continuant part" ;
    rdfs:subClassOf [ a owl:Restriction ;
...
```

# New OWL restriction template

```Turtle
:{Class URIREF}} a owl:Class ;
    rdfs:label "{Property Label}" ;
    rdfs:subClassOf [ a owl:Restriction ;
    owl:onProperty {Property URIREF} ;
...
```
