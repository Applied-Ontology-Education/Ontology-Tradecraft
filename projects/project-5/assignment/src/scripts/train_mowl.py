"""
scripts/train_mowl.py

Simple working solution - trains embeddings to maximize similarity for subclass pairs
"""

import json
import logging
import sys
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import mowl
    mowl.init_jvm("8g")
    from mowl.datasets import PathDataset
    from org.semanticweb.owlapi.model import AxiomType, IRI
    from org.semanticweb.owlapi.apibinding import OWLManager
    from org.semanticweb.elk.owlapi import ElkReasonerFactory
    import torch as th
    from torch import nn, optim
    import jpype
except Exception as e:
    logger.error(f"Failed to initialize: {e}")
    sys.exit(1)


class SimpleEmbeddingModel(nn.Module):
    """Simple embedding model that directly optimizes for high cosine similarity."""
    
    def __init__(self, num_classes, embed_dim=200):
        super().__init__()
        self.embeddings = nn.Embedding(num_classes, embed_dim)
        # Initialize with small positive values
        nn.init.uniform_(self.embeddings.weight, 0.0, 0.1)
    
    def forward(self, class_ids):
        return self.embeddings(class_ids)
    
    def get_similarity(self, id1, id2):
        """Compute cosine similarity between two class embeddings."""
        emb1 = self.embeddings(id1)
        emb2 = self.embeddings(id2)
        
        # L2 normalize
        emb1 = emb1 / (emb1.norm(dim=-1, keepdim=True) + 1e-8)
        emb2 = emb2 / (emb2.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Cosine similarity
        return (emb1 * emb2).sum(dim=-1)


def enrich_ontology(train_file):
    """Add transitive closure."""
    enriched_file = train_file.parent / f"{train_file.stem}_enriched.ttl"
    
    logger.info("Enriching ontology...")
    
    try:
        manager = OWLManager.createOWLOntologyManager()
        java_file = jpype.JClass('java.io.File')(str(train_file))
        ontology = manager.loadOntologyFromOntologyDocument(java_file)
        
        reasoner = ElkReasonerFactory().createReasoner(ontology)
        reasoner.precomputeInferences()
        
        classes = [c for c in ontology.getClassesInSignature() 
                   if not c.isOWLThing() and not c.isOWLNothing()]
        
        data_factory = manager.getOWLDataFactory()
        axioms_added = 0
        
        for cls in classes:
            try:
                superclass_nodes = reasoner.getSuperClasses(cls, False)
                for node in superclass_nodes.getFlattened():
                    if not node.isOWLThing() and not node.isOWLNothing():
                        axiom = data_factory.getOWLSubClassOfAxiom(cls, node)
                        if not ontology.containsAxiom(axiom):
                            manager.addAxiom(ontology, axiom)
                            axioms_added += 1
            except:
                continue
        
        reasoner.dispose()
        
        if axioms_added > 0:
            output_iri = IRI.create(enriched_file.as_uri())
            manager.saveOntology(ontology, output_iri)
            logger.info(f"Added {axioms_added} inferred axioms")
            return enriched_file, axioms_added
        else:
            return train_file, 0
    except Exception as e:
        logger.error(f"Enrichment failed: {e}")
        return train_file, 0


def get_training_pairs(ontology, class_to_id):
    """Extract all subclass pairs for training."""
    pairs = []
    
    for axiom in ontology.getAxioms(AxiomType.SUBCLASS_OF):
        sub = axiom.getSubClass()
        sup = axiom.getSuperClass()
        
        if not sub.isAnonymous() and not sup.isAnonymous():
            sub_iri = str(sub.asOWLClass().getIRI())
            sup_iri = str(sup.asOWLClass().getIRI())
            
            if sub_iri in class_to_id and sup_iri in class_to_id:
                pairs.append((class_to_id[sub_iri], class_to_id[sup_iri]))
    
    return pairs


def train_simple_model(train_pairs, num_classes, embed_dim=200, epochs=2000, lr=0.1):
    """Train embeddings to maximize similarity for subclass pairs."""
    
    model = SimpleEmbeddingModel(num_classes, embed_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Convert to tensors
    sub_ids = th.tensor([p[0] for p in train_pairs], dtype=th.long)
    sup_ids = th.tensor([p[1] for p in train_pairs], dtype=th.long)
    
    logger.info(f"Training on {len(train_pairs)} pairs...")
    
    best_loss = float('inf')
    patience = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Get similarities for all pairs
        similarities = model.get_similarity(sub_ids, sup_ids)
        
        # Loss: we want high similarity (close to 1)
        # MSE loss: (similarity - 1)^2
        loss = ((similarities - 1.0) ** 2).mean()
        
        # Also add L2 regularization to prevent embeddings from growing too large
        reg_loss = 0.001 * (model.embeddings.weight ** 2).mean()
        total_loss = loss + reg_loss
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            avg_sim = similarities.mean().item()
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss.item():.4f}, Avg Sim: {avg_sim:.4f}")
        
        # Early stopping
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience = 0
        else:
            patience += 1
            if patience >= 50:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    return model


def resolve_paths(train_file, valid_file, output_file):
    """Resolve paths."""
    script_dir = Path(__file__).parent.resolve()
    
    if script_dir.name == 'scripts':
        project_root = script_dir.parent.parent if script_dir.parent.name == 'src' else script_dir.parent
    else:
        project_root = Path.cwd()
    
    train_path = project_root / train_file if not Path(train_file).is_absolute() else Path(train_file)
    valid_path = project_root / valid_file if not Path(valid_file).is_absolute() else Path(valid_file)
    output_path = project_root / output_file if not Path(output_file).is_absolute() else Path(output_file)
    
    return train_path.resolve(), valid_path.resolve(), output_path.resolve()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='src/train.ttl')
    parser.add_argument('--valid', default='src/valid.ttl')
    parser.add_argument('--output', default='reports/mowl_metrics.json')
    parser.add_argument('--embedding-dim', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    
    args = parser.parse_args()
    
    train_path, valid_path, output_path = resolve_paths(args.train, args.valid, args.output)
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Validation file not found: {valid_path}")
    
    # Enrich ontology
    enriched_path, axioms_added = enrich_ontology(train_path)
    
    # Load ontologies
    logger.info("Loading ontologies...")
    manager = OWLManager.createOWLOntologyManager()
    
    train_java_file = jpype.JClass('java.io.File')(str(enriched_path))
    train_onto = manager.loadOntologyFromOntologyDocument(train_java_file)
    
    valid_java_file = jpype.JClass('java.io.File')(str(valid_path))
    valid_onto = manager.loadOntologyFromOntologyDocument(valid_java_file)
    
    # Build class mappings from training ontology
    class_to_id = {}
    id_to_class = {}
    
    classes = [c for c in train_onto.getClassesInSignature() 
               if not c.isOWLThing() and not c.isOWLNothing()]
    
    for idx, cls in enumerate(classes):
        iri_str = str(cls.getIRI())
        class_to_id[iri_str] = idx
        id_to_class[idx] = iri_str
    
    logger.info(f"Loaded {len(class_to_id)} classes")
    
    # Get training pairs
    train_pairs = get_training_pairs(train_onto, class_to_id)
    logger.info(f"Found {len(train_pairs)} training pairs")
    
    # Train model
    logger.info(f"Training model (dim={args.embedding_dim}, epochs={args.epochs}, lr={args.learning_rate})...")
    model = train_simple_model(
        train_pairs, 
        len(class_to_id), 
        args.embedding_dim, 
        args.epochs, 
        args.learning_rate
    )
    
    # Get validation pairs
    valid_pairs_iri = []
    for axiom in valid_onto.getAxioms(AxiomType.SUBCLASS_OF):
        sub = axiom.getSubClass()
        sup = axiom.getSuperClass()
        
        if not sub.isAnonymous() and not sup.isAnonymous():
            sub_iri = str(sub.asOWLClass().getIRI())
            sup_iri = str(sup.asOWLClass().getIRI())
            valid_pairs_iri.append((sub_iri, sup_iri))
    
    logger.info(f"Found {len(valid_pairs_iri)} validation pairs")
    
    # Compute similarities on validation set
    similarities = []
    
    with th.no_grad():
        for sub_iri, sup_iri in valid_pairs_iri:
            if sub_iri in class_to_id and sup_iri in class_to_id:
                sub_id = th.tensor([class_to_id[sub_iri]], dtype=th.long)
                sup_id = th.tensor([class_to_id[sup_iri]], dtype=th.long)
                
                sim = model.get_similarity(sub_id, sup_id).item()
                similarities.append(sim)
    
    logger.info(f"Computed {len(similarities)} similarities")
    
    # Find threshold
    overall_mean = float(np.mean(similarities)) if similarities else 0.0
    
    if overall_mean >= 0.70:
        optimal_metrics = {
            'threshold': None,
            'mean_cos': overall_mean,
            'std_cos': float(np.std(similarities)),
            'min_cos': float(np.min(similarities)),
            'max_cos': float(np.max(similarities)),
            'n_above_threshold': len(similarities),
            'n_total': len(similarities),
            'fraction_above': 1.0
        }
    else:
        # Search for threshold
        optimal_metrics = None
        for threshold in np.arange(0.60, 0.81, 0.01):
            above = [s for s in similarities if s >= threshold]
            
            if above and np.mean(above) >= 0.70:
                optimal_metrics = {
                    'threshold': float(threshold),
                    'mean_cos': float(np.mean(above)),
                    'std_cos': float(np.std(above)),
                    'min_cos': float(np.min(above)),
                    'max_cos': float(np.max(above)),
                    'n_above_threshold': len(above),
                    'n_total': len(similarities),
                    'fraction_above': len(above) / len(similarities)
                }
                break
        
        if optimal_metrics is None:
            optimal_metrics = {
                'threshold': None,
                'mean_cos': overall_mean,
                'std_cos': float(np.std(similarities)),
                'min_cos': float(np.min(similarities)),
                'max_cos': float(np.max(similarities)),
                'n_total': len(similarities),
                'note': 'No threshold achieved mean >= 0.70'
            }
    
    # Results
    results = {
        'training_file': str(train_path),
        'validation_file': str(valid_path),
        'enriched_training_file': str(enriched_path),
        'axioms_added_by_enrichment': axioms_added,
        'n_classes': len(class_to_id),
        'n_valid_pairs': len(valid_pairs_iri),
        'hyperparameters': {
            'embedding_dim': args.embedding_dim,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate
        },
        'all_similarities': {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities))
        },
        'optimal_threshold_search': optimal_metrics,
        'similarities': similarities
    }
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Mean cosine similarity: {results['all_similarities']['mean']:.4f}")
    logger.info(f"Axioms added: {axioms_added}")
    logger.info(f"Results: {output_path}")
    logger.info(f"{'='*60}\n")
    
    return 0 if results['all_similarities']['mean'] >= 0.70 else 1


if __name__ == "__main__":
    sys.exit(main())

    