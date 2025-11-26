#!/usr/bin/env python3
"""
Preprocess and enrich ontology definitions using Anthropic Claude.
Normalizes definitions to canonical form and aligns with CCO style.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Import Anthropic package
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Error: anthropic package not installed. Install with: pip install anthropic")
    sys.exit(1)


class DefinitionEnricher:
    """Enriches and normalizes ontology definitions using Anthropic Claude."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the enricher with Anthropic API credentials.
        
        Args:
            api_key: Anthropic API key
        """
        if not ANTHROPIC_AVAILABLE:
            raise ValueError("Anthropic package not installed. Install with: pip install anthropic")
        
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = "claude-sonnet-4-20250514"
    
    def create_prompt(self, label: str, definition: str, iri: str) -> str:
        """
        Create a prompt for Claude to normalize and enrich a definition.
        
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
        Enrich a single definition using Claude.
        
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
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=200,
                    temperature=0.3,  # Lower temperature for more consistent output
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                enriched = message.content[0].text.strip()
                
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
            
            # Rate limiting (Anthropic has generous limits, but still be respectful)
            if i % batch_size == 0 and i < total:
                print(f"  Processed {i} definitions, brief pause...")
                time.sleep(1)
        
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
    
    # Check for Anthropic API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("Error: No Anthropic API key found!")
        print("Set the environment variable:")
        print("  PowerShell: $env:ANTHROPIC_API_KEY = 'your-key-here'")
        print("  Linux/Mac: export ANTHROPIC_API_KEY='your-key-here'")
        print("\nGet your API key from: https://console.anthropic.com/")
        sys.exit(1)
    
    print("Using Anthropic Claude for definition enrichment...")
    print(f"Model: claude-sonnet-4-20250514")
    
    # Initialize enricher
    try:
        enricher = DefinitionEnricher(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Claude enricher: {e}")
        print("Please check your API key and internet connection.")
        sys.exit(1)
    
    # Load definitions
    print(f"\nLoading definitions from {input_file}...")
    definitions = load_definitions(input_file)
    print(f"Loaded {len(definitions)} definitions")
    
    # Process definitions
    print("\nEnriching definitions with Claude...")
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
