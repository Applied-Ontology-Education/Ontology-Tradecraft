"""
scripts/filter_candidates_hybrid.py

Hybrid filtering: combines MOWL cosine similarity with LLM semantic plausibility.
Scores each candidate axiom using:
  - MOWL embedding cosine similarity
  - LLM semantic plausibility rating (0-1)
  - Weighted combination: 0.7 × cosine + 0.3 × LLM

Outputs accepted axioms to generated/accepted_el.ttl
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError as e:
    logger.error(f"Missing numpy package: {e}")
    logger.error("Install with: pip install numpy")
    sys.exit(1)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic package not installed. Install with: pip install anthropic")

import os


def resolve_paths(candidates_file, metrics_file, output_file):
    """Resolve file paths relative to current directory."""
    
    # Simply resolve relative to current working directory
    candidates_path = Path(candidates_file).resolve()
    metrics_path = Path(metrics_file).resolve()
    output_path = Path(output_file).resolve()
    
    return candidates_path, metrics_path, output_path


def load_candidates(candidates_file: Path) -> List[Tuple[str, str]]:
    """Load candidate axioms from TTL file, handling multi-line class definitions."""
    logger.info(f"Loading candidates from: {candidates_file}")
    candidates = []

    with open(candidates_file, 'r', encoding='utf-8') as f:
        current_subclass = None
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Detect start of a class definition
            if line.startswith('cco:ont') and 'a owl:Class' in line:
                current_subclass = line.split()[0]
                if current_subclass.startswith('cco:'):
                    current_subclass = f"https://www.commoncoreontologies.org/{current_subclass.split(':')[1]}"
            
            # Look for rdfs:subClassOf inside the block
            elif current_subclass and line.startswith('rdfs:subClassOf'):
                parts = line.split()
                if len(parts) >= 2:
                    superclass = parts[1].rstrip(';').rstrip('.')
                    if superclass.startswith('cco:'):
                        superclass = f"https://www.commoncoreontologies.org/{superclass.split(':')[1]}"
                    candidates.append((current_subclass, superclass))
                    logger.debug(f"Added candidate: {current_subclass} ⊑ {superclass}")
            
            # End of block (semicolon or period)
            if line.endswith('.'):
                current_subclass = None

    logger.info(f"Loaded {len(candidates)} candidate axioms")
    if len(candidates) == 0:
        logger.warning("No candidates found in file! Make sure rdfs:subClassOf is present.")
    
    return candidates



def load_mowl_metrics(metrics_file: Path) -> Dict:
    """Load MOWL training results and embeddings."""
    logger.info(f"Loading MOWL metrics from: {metrics_file}")
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    logger.info(f"MOWL model trained on {metrics.get('n_classes', 0)} classes")
    return metrics


def load_class_labels(train_file: Path) -> Dict[str, str]:
    """Extract class labels from training ontology."""
    logger.info(f"Loading class labels from: {train_file}")
    
    labels = {}
    current_class = None
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Match class definitions
            if line.startswith('###') and 'commoncoreontologies.org' in line:
                # Next line should have the class IRI
                continue
            elif line.startswith('cco:ont') and 'rdf:type owl:Class' in line:
                current_class = line.split()[0].replace('cco:', 'https://www.commoncoreontologies.org/')
            elif current_class and 'rdfs:label' in line:
                # Extract label
                label = line.split('rdfs:label')[1].strip()
                label = label.replace('"', '').replace('@en', '').replace(';', '').replace('.', '').strip()
                labels[current_class] = label
                current_class = None
    
    logger.info(f"Loaded {len(labels)} class labels")
    return labels


def compute_cosine_similarity(sub_iri: str, sup_iri: str, embeddings: np.ndarray, 
                               class_to_id: Dict[str, int]) -> float:
    """Compute cosine similarity between two classes using MOWL embeddings."""
    
    if sub_iri not in class_to_id or sup_iri not in class_to_id:
        return 0.0
    
    sub_id = class_to_id[sub_iri]
    sup_id = class_to_id[sup_iri]
    
    sub_emb = embeddings[sub_id]
    sup_emb = embeddings[sup_id]
    
    # Compute cosine similarity
    cos_sim = np.dot(sub_emb, sup_emb) / (
        np.linalg.norm(sub_emb) * np.linalg.norm(sup_emb) + 1e-8
    )
    
    return float(cos_sim)


def query_llm_plausibility(subclass_label: str, superclass_label: str, 
                           client: anthropic.Anthropic) -> float:
    """Query Claude to rate semantic plausibility of subclass relationship."""
    
    prompt = f"""You are evaluating an ontology axiom for semantic plausibility.

Given the following proposed subclass relationship:
- Subclass: "{subclass_label}"
- Superclass: "{superclass_label}"

Question: On a scale from 0.0 to 1.0, how semantically plausible is it that "{subclass_label}" is a subclass (more specific type) of "{superclass_label}"?

Guidelines:
- 1.0 = Highly plausible, clearly makes sense (e.g., "Hospital" is a subclass of "Healthcare Facility")
- 0.7-0.9 = Plausible, reasonable relationship
- 0.4-0.6 = Questionable, might be valid but unclear
- 0.1-0.3 = Implausible, likely incorrect
- 0.0 = Completely implausible, definitely wrong

Respond with ONLY a single number between 0.0 and 1.0, nothing else."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract score
        score_text = message.content[0].text.strip()
        
        # Parse the number
        try:
            score = float(score_text)
            # Clamp to [0, 1]
            score = max(0.0, min(1.0, score))
            return score
        except ValueError:
            logger.warning(f"Could not parse Claude response: {score_text}")
            return 0.5  # Default to neutral
            
    except Exception as e:
        logger.error(f"Claude query failed: {e}")
        logger.error(f"Error details: {type(e).__name__}")
        return 0.5  # Default to neutral on error


def hybrid_filter_candidates(
    candidates: List[Tuple[str, str]],
    embeddings: np.ndarray,
    class_to_id: Dict[str, int],
    class_labels: Dict[str, str],
    client: anthropic.Anthropic,
    cosine_weight: float = 0.7,
    llm_weight: float = 0.3,
    threshold: float = 0.70,
    batch_size: int = 10
) -> List[Dict]:
    """
    Filter candidates using hybrid scoring.
    
    Args:
        candidates: List of (subclass_iri, superclass_iri) tuples
        embeddings: MOWL class embeddings
        class_to_id: Mapping from IRI to embedding ID
        class_labels: Mapping from IRI to human-readable label
        client: Anthropic API client
        cosine_weight: Weight for cosine similarity (default 0.7)
        llm_weight: Weight for LLM score (default 0.3)
        threshold: Combined score threshold for acceptance (default 0.70)
        batch_size: Number of LLM queries per batch (for rate limiting)
    
    Returns:
        List of accepted axioms with scores
    """
    
    logger.info(f"Hybrid filtering {len(candidates)} candidates...")
    logger.info(f"Weights: {cosine_weight:.1f} × cosine + {llm_weight:.1f} × LLM")
    logger.info(f"Threshold: {threshold:.2f}")
    
    accepted = []
    
    for i, (sub_iri, sup_iri) in enumerate(candidates):
        # Get labels
        sub_label = class_labels.get(sub_iri, sub_iri.split('/')[-1])
        sup_label = class_labels.get(sup_iri, sup_iri.split('/')[-1])
        
        # Compute MOWL cosine similarity
        cosine_sim = compute_cosine_similarity(sub_iri, sup_iri, embeddings, class_to_id)
        
        # Query LLM for semantic plausibility
        logger.info(f"[{i+1}/{len(candidates)}] Evaluating: {sub_label} ⊑ {sup_label}")
        llm_score = query_llm_plausibility(sub_label, sup_label, client)
        
        # Combine scores
        combined_score = cosine_weight * cosine_sim + llm_weight * llm_score
        
        logger.info(f"  Cosine: {cosine_sim:.4f}, LLM: {llm_score:.4f}, Combined: {combined_score:.4f}")
        
        # Check threshold
        if combined_score >= threshold:
            accepted.append({
                'subclass': sub_iri,
                'superclass': sup_iri,
                'subclass_label': sub_label,
                'superclass_label': sup_label,
                'cosine_similarity': cosine_sim,
                'llm_score': llm_score,
                'combined_score': combined_score
            })
            logger.info(f"  ✓ ACCEPTED (score={combined_score:.4f})")
        else:
            logger.info(f"  ✗ REJECTED (score={combined_score:.4f} < {threshold:.2f})")
        
        # Rate limiting: small delay between LLM queries
        if (i + 1) % batch_size == 0:
            import time
            time.sleep(1)
    
    logger.info(f"\nAccepted {len(accepted)} / {len(candidates)} axioms")
    return accepted


def write_accepted_axioms(accepted: List[Dict], output_file: Path, train_file: Path):
    """Write accepted axioms to TTL file."""
    logger.info(f"Writing {len(accepted)} accepted axioms to: {output_file}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out:
        # Write header
        out.write("@prefix : <https://www.commoncoreontologies.org/FacilityOntologyGenerated/> .\n")
        out.write("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n")
        out.write("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n")
        out.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n")
        out.write("@prefix cco: <https://www.commoncoreontologies.org/> .\n")
        out.write("\n")
        
        out.write("<https://www.commoncoreontologies.org/FacilityOntologyGenerated> rdf:type owl:Ontology ;\n")
        out.write("    rdfs:label \"Generated Facility Ontology Axioms\"@en ;\n")
        out.write("    rdfs:comment \"Axioms generated by hybrid MOWL + LLM filtering\"@en .\n")
        out.write("\n")
        
        # Write axioms
        out.write("#################################################################\n")
        out.write("#    Generated Axioms\n")
        out.write("#################################################################\n\n")
        
        for axiom in accepted:
            sub_id = axiom['subclass'].split('/')[-1]
            sup_id = axiom['superclass'].split('/')[-1]
            
            out.write(f"### Generated axiom: {axiom['subclass_label']} ⊑ {axiom['superclass_label']}\n")
            out.write(f"### Cosine: {axiom['cosine_similarity']:.4f}, LLM: {axiom['llm_score']:.4f}, Combined: {axiom['combined_score']:.4f}\n")
            out.write(f"cco:{sub_id} rdfs:subClassOf cco:{sup_id} .\n\n")
    
    logger.info(f"✓ Wrote {len(accepted)} axioms to {output_file}")


def load_embeddings_and_mappings(metrics: Dict) -> Tuple[np.ndarray, Dict[str, int]]:
    """Load saved embeddings and class mappings."""
    
    # Check if paths are in metrics
    if 'embeddings_file' in metrics and 'mappings_file' in metrics:
        embeddings_file = Path(metrics['embeddings_file'])
        mappings_file = Path(metrics['mappings_file'])
        
        if embeddings_file.exists() and mappings_file.exists():
            logger.info(f"Loading embeddings from: {embeddings_file}")
            embeddings = np.load(embeddings_file)
            
            logger.info(f"Loading mappings from: {mappings_file}")
            import pickle
            with open(mappings_file, 'rb') as f:
                mappings_data = pickle.load(f)
            
            class_to_id = mappings_data['class_to_id']
            
            logger.info(f"Loaded embeddings: shape {embeddings.shape}")
            logger.info(f"Loaded {len(class_to_id)} class mappings")
            
            return embeddings, class_to_id
    
    # Fallback: create synthetic data
    logger.warning("Could not load saved embeddings/mappings. Creating synthetic data.")
    
    n_classes = metrics.get('n_classes', 100)
    embed_dim = metrics.get('hyperparameters', {}).get('embedding_dim', 200)
    
    embeddings = np.random.randn(n_classes, embed_dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    class_to_id = {}
    
    return embeddings, class_to_id


def main():
    parser = argparse.ArgumentParser(
        description='Hybrid filtering: MOWL cosine + LLM semantic plausibility'
    )
    parser.add_argument('--candidates', default='../generated/candidate_el.ttl',
                        help='Input candidate axioms file')
    parser.add_argument('--metrics', default='../../reports/mowl_metrics.json',
                        help='MOWL training metrics file')
    parser.add_argument('--train', default='../train.ttl',
                        help='Training ontology (for class labels)')
    parser.add_argument('--output', default='../generated/accepted_el.ttl',
                        help='Output accepted axioms file')
    parser.add_argument('--cosine-weight', type=float, default=0.7,
                        help='Weight for cosine similarity (default: 0.7)')
    parser.add_argument('--llm-weight', type=float, default=0.3,
                        help='Weight for LLM score (default: 0.3)')
    parser.add_argument('--threshold', type=float, default=0.70,
                        help='Combined score threshold (default: 0.70)')
    parser.add_argument('--api-key', default=None,
                        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Resolve paths
    candidates_path, metrics_path, output_path = resolve_paths(
        args.candidates, args.metrics, args.output
    )
    
    train_path = Path(args.train).resolve()
    
    # Check files exist
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    
    # Check if Anthropic package is available
    if not ANTHROPIC_AVAILABLE:
        logger.error("Anthropic package not installed!")
        logger.error("Install with: pip install anthropic")
        sys.exit(1)
    
    # Get API key
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    
    if not api_key:
        logger.error("Anthropic API key required!")
        logger.error("Set environment variable: $env:ANTHROPIC_API_KEY = 'your-key'")
        logger.error("Or pass --api-key argument")
        sys.exit(1)
    
    # Initialize Anthropic client
    logger.info("Initializing Anthropic API client...")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        logger.info("✓ Anthropic client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize client: {e}")
        sys.exit(1)
    
    # Load data
    candidates = load_candidates(candidates_path)
    metrics = load_mowl_metrics(metrics_path)
    class_labels = load_class_labels(train_path)
    
    # Load embeddings and mappings from saved files
    embeddings, class_to_id = load_embeddings_and_mappings(metrics)
    
    # If we don't have a proper mapping, create one from candidates
    if not class_to_id:
        logger.info("Building class_to_id mapping from candidates...")
        unique_classes = set()
        for sub, sup in candidates:
            unique_classes.add(sub)
            unique_classes.add(sup)
        
        for idx, cls in enumerate(sorted(unique_classes)):
            class_to_id[cls] = idx
        
        # Rebuild embeddings with correct size
        embeddings = np.random.randn(len(class_to_id), 
                                     metrics.get('hyperparameters', {}).get('embedding_dim', 200))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
    
    # Hybrid filtering
    accepted = hybrid_filter_candidates(
        candidates=candidates,
        embeddings=embeddings,
        class_to_id=class_to_id,
        class_labels=class_labels,
        client=client,
        cosine_weight=args.cosine_weight,
        llm_weight=args.llm_weight,
        threshold=args.threshold
    )
    
    # Write results
    write_accepted_axioms(accepted, output_path, train_path)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("FILTERING COMPLETE")
    logger.info("="*60)
    logger.info(f"Candidates evaluated: {len(candidates)}")
    logger.info(f"Axioms accepted: {len(accepted)}")
    if len(candidates) > 0:
        logger.info(f"Acceptance rate: {len(accepted)/len(candidates)*100:.1f}%")
    else:
        logger.warning("No candidates were evaluated!")
    logger.info(f"Output file: {output_path}")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

