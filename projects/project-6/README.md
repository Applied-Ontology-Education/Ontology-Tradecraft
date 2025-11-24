# Week 6 Project: Enhancing your Semantic Pipeline

The final project for this seminar asks you to demonstrate how Large Language Models (LLMs) can be integrated with symbolic ontology tools to create a modern, hybrid semantic pipeline. Your goal is to show how LLMs can augment traditional semantic technologies—improving automation, scalability, and reasoning while maintaining correctness and trustworthiness.

You will select one of two project pathways and deliver a live presentation and demonstration during the final class meeting.

## Project

**Option 1: Data & Design-Pattern Expansion**

Extend your current semantic pipeline so that it can ingest and model a new domain, a new class of data, or a new family of design patterns.

Your project should demonstrate how LLMs assist with:
- Detecting and normalizing unfamiliar schemas
- Identifying new ontology design patterns
- Mapping new domain patterns to existing ones
- Auto-generating SPARQL QC queries and SHACL shape constraints
- Proposing candidate OWL axioms for the new domain

Tools you may use (not exhaustive):
- ROBOT + ELK for taxonomy and unit consistency
- SHACL for shape validation
- OWL reasoners to check logical coherence
- SPARQL QC to ensure data/ontology quality
- GitHub Actions for fully automated CI/CD execution
- Your favorite LLM(s)

**Option 2: Ontology Mapping Pipeline**

Develop or extend a pipeline that uses LLMs to generate, evaluate, and validate ontology mappings.

Your project should show how LLMs contribute by:
- Generating candidate mapping axioms
- Rewriting labels and definitions to aid matching
- Producing mapping explanations or rationales
- Scoring mapping plausibility
- Suggesting mapping-related SHACL or SPARQL QC constraints

Tools you may use (not exhaustive):
- OWL reasoners to test for inconsistencies introduced by mappings
- Cross-ontology inference tests to verify correct entailments
- SHACL to enforce mapping constraints
- SPARQL QC to identify mapping errors
- CI automation to validate the mapping layer end-to-end
- Your favorite LLM(s)

## Final Deliverable: Live Presentation

Your in-class presentation must include:
- Option selected and how you understood the task
- Explanation of your system architecture, including description of your inputs and expected outputs
- Live demonstration of your semantic pipeline
- Verifying correctness with reasoners, SHACL, SPARQL QC, and CI automation
- Display and description of your deliverables
