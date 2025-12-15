## Class

### System

# How to create a candidate OWL restriction for an existing OWL class

This prompt uses a Markdown format for human readability.

# Goal

Create a candidate owl:Restriction axiom for an existing OWL class, using its definition in English.
An owl:Restriction logically adds to a class by expressing in logic what differentia distinguishes it from its superclass.
Using verb phrases and noun phrases from the definition in English that indicate a differentia, create a candidate owl:Restriction axiom.

### User

# Instructions

- Identify the phrases placed in single quotes in the definition given in the section 'Definition in English'.
- Your are looking for combinations of a verb phrase and a noun phrase.
- Ignore VERB phrases that are not mentioned in the section 'Mapping of VERB phrases to URIREF for PROPERTIES'.
- Ignore NOUN phrases that are not mentioned in the section 'Mapping of NOUN phrases to URIREF for CLASSES'.
- A verb phrase refers to an OWL property.
- A noun phrase refers to an OWL class.
- Rewrite the definition in English so that the phrases adhere to both the mapping in the sections:
  - 'Mapping of VERB phrases to URIREF for PROPERTIES' 
  - 'Mapping of NOUN phrases to URIREF for CLASSES'.
- All the URIREFs MUST be output literally in the output; no rewriting or replacement is allowed.
- The combination of a verb phrase and a related noun phrase in English is a candidate owl:Restriction axiom.
- Create a candidate owl:Restriction axiom for {target_uri_ref} using the section 'New OWL restriction templates'.
- Check if the candidate owl:Restriction is already in place by using the section 'OWL definition'.
- If the owl:Restriction is already in place, discard the new owl:Restriction.
- You MUST only use the URIREFs mentioned explicitly in the sections:
  - 'Mapping of VERB phrases to URIREF for PROPERTIES'
  - 'Mapping of NOUN phrases to URIREF for CLASSES'.
- Output ONLY json with the following fields:
  - candidate_axioms: a list of candidate owl:Restriction axiom elements in the format of:
    - "class_uri": {target_uri_ref}, "verb_phrase_uri": {{insert URIREF of verb phrase}}, "noun_phrase_uri": {{insert URIREF of noun phrase}}
  - reasoning: text explaining why the owl:Restriction axioms were created.

# Definition in English
The definition is:
{target_definition}

# Mapping of VERB phrases to URIREF for PROPERTIES
Here is a list for mapping verb phrases to their corresponding URIREFs:

```csv
Verb Phrase, URIREF
{properties_section}
```

# Mapping of NOUN phrases to URIREF for CLASSES
Here is a list of verb phrases and their corresponding URIREFs:

```csv
Noun Phrase, URIREF
{classes_section}
```

# OWL definition

Here is the current OWL definition for the definition in Turtle format.

This section can be used to check if the new owl:Restriction is already in place.

```turtle
{owl_definition}
```

# New OWL restriction templates

Instructions for creating a new owl:Restriction:
- You need a combination of a verb phrase and a noun phrase to create an owl:Restriction.
- Both VERB phrases and NOUN phrases MUST occur in 'Definition in English'.
- The URIREF for the verb phrase MUST be in the section 'Mapping of VERB phrases to URIREF for PROPERTIES'.
- The URIREF for the noun phrase MUST be in the section 'Mapping of NOUN phrases to URIREF for CLASSES'.
- Check if the verb and noun phrase URIREFs are different.
- You need to replace {{verb phrase URIREF}} with the URIREF of the VERB phrase.
- You need to replace {{noun phrase URIREF}} with the URIREF of the NOUN phrase.
- ONLY create an owl:Restriction using owl:someValuesFrom.
- ONLY create an owl:Restriction for the class {target_uri_ref} .

This is the template for adding an owl:Restriction.

```turtle

<{target_uri_ref}>
    rdfs:subClassOf [ a                  owl:Restriction ;
                      owl:onProperty     <{{verb phrase URIREF}}> ;
                      owl:someValuesFrom <{{noun phrase URIREF}}> ] .
```

## Property

### System

# How to create a candidate OWL restriction for an existing OWL class

This prompt uses a Markdown format for human readability.

# Goal

Create a candidate owl:Restriction axiom for an existing OWL class, using its definition in English.
An owl:Restriction allows you to better define a class by showing what differentia distinguishes it from its superclass.

### User

# Instructions

- Identify the phrases placed in single quotes in the definition given in the section 'Definition in English'.
- Your are looking for combinations of a verb phrase and a noun phrase.
- You can also identify phrases that are near those mentioned in the section 'Mapping of phrases to URIREF'.
- Ignore phrases that are not mentioned in the section 'Mapping of phrases to URIREF'.
- Rewrite the definition in English so that its adheres to the mapping in the section 'Mapping of phrases to URIREF'.
- The combination of a verb phrase and a related noun phrase in English is a candidate owl:Restriction axiom.
- Verb phrases in the English definition refer to OWL properties.
- Noun phrases in the English definition refer to OWL classes.
- Use the section 'Mapping of verb and noun phrases to URIREF' below to lookup the OWL URIREF for the two phrases.
- You MUST only use the URIREFs from the section 'Mapping of verb and noun phrases to URIREF'.
- Create a candidate owl:Restriction axiom for {target_uri_ref} using the section 'New OWL restriction templates'.
- Check if the candidate owl:Restriction is already in place by using the section 'OWL definition'.
- If the owl:Restriction is already in place, discard new owl:Restriction.
- Output ONLY json with the following fields:
  - candidate_axioms: a list of candidate owl:Restriction axioms in Turtle format.
  - reasoning: text explaining why the owl:Restriction axioms were created.

# Definition in English
The definition is:
{target_definition}

# Mapping of phrases to URIREF

DO NOT USE ANY OTHER URIREFS THAN THESE
Here is a list for mapping verb phrases to their corresponding URIREFs:

```csv
Verb Phrase, URIREF
{properties_section}
```

Here is a list of verb phrases and their corresponding URIREFs:

```csv
Noun Phrase, URIREF
{classes_section}
```

# OWL definition

Here is the current OWL definition for the definition in Turtle format.

This section can be used to check if the new owl:Restriction is already in place.

```turtle
{owl_definition}
```

# New OWL restriction templates

Instructions for creating a new owl:Restriction:

- You need a combination of a verb phrase and a noun phrase.
- Both verb phrases and noun phrases MUST be occur in 'Definition in English'.
- Check if the verb and noun phrase URIREFs are different.
- Both URIREFs MUST be mentioned in the section 'Mapping of phrases to URIREF'.
- You need to replace {{verb phrase URIREF}} with the URIREF of the verb phrase.
- You need to replace {{noun phrase URIREF}} with the URIREF of the noun phrase.
- Only create an owl:Restriction using owl:someValuesFrom.
- ONLY create an owl:Restriction for the class {target_uri_ref} .

This is the template for adding a restriction.

```turtle

<{target_uri_ref}>
    rdfs:subClassOf [ a                  owl:Restriction ;
                      owl:onProperty     <{{verb phrase URIREF}}> ;
                      owl:someValuesFrom <{{noun phrase URIREF}}> ] .
```