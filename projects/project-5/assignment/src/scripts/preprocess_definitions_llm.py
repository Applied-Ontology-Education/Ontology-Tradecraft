#!/usr/bin/env python3
"""
Preprocess and enrich ontology definitions using an LLM.
Normalizes definitions to canonical form and aligns with CCO style.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Try to import OpenAI, but provide alternatives if not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. Install with: pip install openai")

# Try to import Anthropic as an alternative
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic package not installed. Install with: pip install anthropic")


class DefinitionEnricher:
    """Enriches and normalizes ontology definitions using LLM."""
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "openai"):
        """
        Initialize the enricher with API credentials.
        
        Args:
            api_key: API key for the LLM provider
            provider: LLM provider to use ("openai" or "anthropic")
        """
        self.provider = provider.lower()
        
        if self.provider == "openai" and OPENAI_AVAILABLE:
            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model = "gpt-4-turbo-preview"  # or "gpt-3.5-turbo" for lower cost
        elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.model = "claude-3-haiku-20240307"  # or "claude-3-sonnet-20240229" for better quality
        else:
            raise ValueError(f"Provider '{provider}' not available or not installed")
    
    def create_prompt(self, label: str, definition: str, iri: str) -> str:
        """
        Create a prompt for the LLM to normalize and enrich a definition.
        
        Args:
            label: The entity label
            definition: The original definition
            iri: The entity IRI for context
        
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an ontology expert specializing in the Common Core Ontologies (CCO) style guide.

Task: Normalize and enrich the following ontology definition to match CCO standards.

Entity Label: {label}
Entity IRI: {iri}
Current Definition: {definition if definition else "No definition provided"}

Requirements:
1. Use the canonical form: "A/An [Entity] is a [Parent Class] that [distinguishing characteristics]"
2. Expand all abbreviations to their full forms
3. Remove ambiguity and clarify technical terms
4. Ensure terminology aligns with CCO style:
   - Use present tense
   - Be concise but complete
   - Focus on essential characteristics
   - Avoid circular definitions
5. If no definition exists, create one based on the entity label and common knowledge

Provide ONLY the improved definition text, nothing else. Do not include the entity name at the start."""
        
        return prompt
    
    def enrich_definition(self, label: str, definition: str, iri: str, max_retries: int = 3) -> str:
        """
        Enrich a single definition using the LLM.
        
        Args:
            label: The entity label
            definition: The original definition
            iri: The entity IRI
            max_retries: Maximum number of retry attempts
        
        Returns:
            Enriched definition text
        """
        prompt = self.create_prompt(label, definition, iri)
        
        for attempt in range(max_retries):
            try:
                if self.provider == "openai" and OPENAI_AVAILABLE:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are an expert in ontology engineering and the Common Core Ontologies."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,  # Lower temperature for more consistent output
                        max_tokens=200
                    )
                    enriched = response.choices[0].message.content.strip()
                    
                elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=200,
                        temperature=0.3,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    enriched = response.content[0].text.strip()
                else:
                    return definition  # Fallback to original
                
                # Validate the enriched definition
                if enriched and len(enriched) > 10:
                    # Ensure it doesn't start with the entity name (remove if present)
                    if enriched.lower().startswith(f"a {label.lower()} is") or \
                       enriched.lower().startswith(f"an {label.lower()} is") or \
                       enriched.lower().startswith(f"{label.lower()} is"):
                        # Extract just the part after "is"
                        enriched = enriched[enriched.lower().index(" is ") + 4:]
                        enriched = enriched[0].upper() + enriched[1:] if enriched else enriched
                    
                    # Ensure proper capitalization
                    if enriched and not enriched[0].isupper():
                        enriched = enriched[0].upper() + enriched[1:]
                    
                    # Ensure it ends with a period
                    if enriched and not enriched.endswith('.'):
                        enriched += '.'
                    
                    return enriched
                    
            except Exception as e:
                print(f"  Error enriching definition for '{label}' (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        # If all attempts failed, return original or a basic definition
        if not definition:
            return f"A concept in the facility domain related to {label.replace('_', ' ').lower()}."
        return definition
    
    def process_definitions(self, definitions: List[Dict[str, str]], batch_size: int = 10) -> List[Dict[str, str]]:
        """
        Process multiple definitions with rate limiting.
        
        Args:
            definitions: List of definition dictionaries
            batch_size: Number of definitions to process before pausing
        
        Returns:
            List of enriched definition dictionaries
        """
        enriched_definitions = []
        total = len(definitions)
        
        for i, defn in enumerate(definitions, 1):
            print(f"Processing {i}/{total}: {defn['label']}")
            
            enriched_def = self.enrich_definition(
                defn['label'],
                defn['definition'],
                defn['IRI']
            )
            
            enriched_definitions.append({
                'IRI': defn['IRI'],
                'label': defn['label'],
                'original_definition': defn['definition'],
                'enriched_definition': enriched_def,
                'modified': enriched_def != defn['definition']
            })
            
            # Rate limiting
            if i % batch_size == 0 and i < total:
                print(f"  Processed {i} definitions, pausing to avoid rate limits...")
                time.sleep(2)
        
        return enriched_definitions


def load_definitions(input_file: Path) -> List[Dict[str, str]]:
    """Load definitions from CSV file."""
    definitions = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            definitions.append(row)
    return definitions


def save_enriched_definitions(output_file: Path, definitions: List[Dict[str, str]]):
    """Save enriched definitions to CSV file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['IRI', 'label', 'original_definition', 'enriched_definition', 'modified']
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(definitions)


def main():
    """Main function to run the enrichment process."""
    
    # Define file paths
    script_path = Path(__file__).resolve()
    
    # The script is in src/scripts/, need to go up one level to src/
    scripts_dir = script_path.parent
    src_dir = scripts_dir.parent
    
    # Define input and output paths
    input_file = src_dir / 'data' / 'definitions.csv'
    output_file = src_dir / 'data' / 'definitions_enriched.csv'
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found!")
        print("Please run extract_definitions.py first to generate the definitions.csv file.")
        sys.exit(1)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key and (OPENAI_AVAILABLE or ANTHROPIC_AVAILABLE):
        print("Warning: No API key found in environment variables.")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.")
        print("\nUsing fallback mode - definitions will be minimally processed.")
        
        # Fallback processing without LLM
        definitions = load_definitions(input_file)
        enriched = []
        
        for defn in definitions:
            enriched_def = defn['definition']
            
            # Basic normalization without LLM
            if enriched_def:
                # Expand common abbreviations
                enriched_def = enriched_def.replace("govt.", "government")
                enriched_def = enriched_def.replace("Govt.", "Government")
                enriched_def = enriched_def.replace("dept.", "department")
                enriched_def = enriched_def.replace("Dept.", "Department")
                enriched_def = enriched_def.replace("org.", "organization")
                enriched_def = enriched_def.replace("Org.", "Organization")
                
                # Ensure proper ending
                if enriched_def and not enriched_def.endswith('.'):
                    enriched_def += '.'
                
                # Ensure proper capitalization
                if enriched_def and not enriched_def[0].isupper():
                    enriched_def = enriched_def[0].upper() + enriched_def[1:]
            else:
                # Create basic definition from label
                enriched_def = f"A facility-related concept associated with {defn['label'].replace('_', ' ').lower()}."
            
            enriched.append({
                'IRI': defn['IRI'],
                'label': defn['label'],
                'original_definition': defn['definition'],
                'enriched_definition': enriched_def,
                'modified': enriched_def != defn['definition']
            })
        
        save_enriched_definitions(output_file, enriched)
        print(f"\nDefinitions processed (fallback mode) and saved to: {output_file}")
        return
    
    # Determine which provider to use
    provider = "openai" if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY") else "anthropic"
    
    print(f"Using {provider.upper()} for definition enrichment...")
    
    # Initialize enricher
    try:
        enricher = DefinitionEnricher(api_key=api_key, provider=provider)
    except Exception as e:
        print(f"Error initializing LLM enricher: {e}")
        print("Please check your API key and internet connection.")
        sys.exit(1)
    
    # Load definitions
    print(f"Loading definitions from {input_file}...")
    definitions = load_definitions(input_file)
    print(f"Loaded {len(definitions)} definitions")
    
    # Process definitions
    print("\nEnriching definitions with LLM...")
    enriched_definitions = enricher.process_definitions(definitions)
    
    # Save enriched definitions
    save_enriched_definitions(output_file, enriched_definitions)
    
    # Print summary
    modified_count = sum(1 for d in enriched_definitions if d['modified'])
    print(f"\nEnrichment complete!")
    print(f"  Total definitions: {len(enriched_definitions)}")
    print(f"  Modified definitions: {modified_count}")
    print(f"  Unchanged definitions: {len(enriched_definitions) - modified_count}")
    print(f"\nEnriched definitions saved to: {output_file}")


if __name__ == "__main__":
    main()

