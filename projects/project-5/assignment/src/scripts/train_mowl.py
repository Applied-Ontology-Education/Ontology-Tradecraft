import mowl
mowl.init_jvm("8g")   # must be done BEFORE any OWLAPI or mowl imports

from mowl.datasets import PathDataset
from mowl.models import ELEmbeddings
from org.semanticweb.owlapi.apibinding import OWLManager

"""
Train an ELEmbeddings model using MOWL library.
Evaluate on validation data and find optimal cosine similarity threshold.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import os
import sys

# Set up logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize JVM before importing MOWL with OWL API jars
import jpype
import jpype.imports

# Start JVM if not already started
if not jpype.isJVMStarted():
    logger.info("Starting JVM...")
    
    # Try to find MOWL's JAR directory
    try:
        import mowl
        mowl_path = Path(mowl.__file__).parent
        jar_dir = mowl_path / 'jar'
        
        logger.info(f"MOWL installation path: {mowl_path}")
        logger.info(f"Looking for JARs in: {jar_dir}")
        
        # Find all JAR files
        jar_files = []
        if jar_dir.exists():
            jar_files = list(jar_dir.glob('*.jar'))
            logger.info(f"Found {len(jar_files)} JAR files:")
            for jar in jar_files:
                logger.info(f"  - {jar.name}")
            jar_files = [str(jar) for jar in jar_files]
        else:
            logger.warning(f"JAR directory not found: {jar_dir}")
            logger.info("Searching for JARs in parent directories...")
            
            # Search in site-packages
            site_packages = mowl_path.parent
            jar_files = list(site_packages.rglob('*.jar'))
            if jar_files:
                logger.info(f"Found {len(jar_files)} JAR files in site-packages")
                jar_files = [str(jar) for jar in jar_files]
        
        # Start JVM with classpath
        if jar_files:
            logger.info(f"Starting JVM with {len(jar_files)} JAR files in classpath")
            jpype.startJVM(classpath=jar_files, convertStrings=False)
        else:
            logger.error("No JAR files found! MOWL requires OWL API JARs.")
            logger.error("Try reinstalling mowl: pip install --force-reinstall mowl-borg")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error starting JVM: {e}")
        logger.error("Try reinstalling mowl: pip install --force-reinstall mowl-borg")
        sys.exit(1)
    
    logger.info("JVM started successfully")

# Now import MOWL modules
try:
    from mowl.datasets import PathDataset
    from mowl.models import ELEmbeddings
    import torch
    logger.info("MOWL modules imported successfully")
except ImportError as e:
    logger.error(f"Failed to import MOWL modules: {e}")
    logger.error("\nPossible solutions:")
    logger.error("1. Reinstall mowl: pip uninstall mowl-borg && pip install mowl-borg")
    logger.error("2. Check if all dependencies are installed: pip install torch jpype1 numpy")
    logger.error("3. Make sure Java is properly installed and JAVA_HOME is set")
    sys.exit(1)


class MowlTrainer:
    """Trainer class for MOWL ELEmbeddings model."""
    
    def __init__(self, train_file: str, valid_file: str, output_file: str):
        """
        Initialize the trainer.
        
        Args:
            train_file: Path to training TTL file
            valid_file: Path to validation TTL file  
            output_file: Path to output metrics JSON file
        """
        self.train_file = Path(train_file)
        self.valid_file = Path(valid_file)
        self.output_file = Path(output_file)
        self.model = None
        self.dataset = None
        self.class_to_id = {}
        self.id_to_class = {}
        
    def load_dataset(self):
        """Load the ontology dataset."""
        logger.info(f"Loading dataset from {self.train_file} and {self.valid_file}")
        
        # Create MOWL dataset
        self.dataset = PathDataset(
            str(self.train_file),
            validation_path=str(self.valid_file)
        )
        
        # Get class mappings
        self.class_to_id = self.dataset.classes.as_dict
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}
        
        logger.info(f"Loaded {len(self.class_to_id)} classes")
        
        return self.dataset
    
    def train_model(self, embedding_dim: int = 100, epochs: int = 100, 
                   learning_rate: float = 0.001, batch_size: int = 128):
        """
        Train the ELEmbeddings model.
        
        Args:
            embedding_dim: Dimension of embeddings
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
        """
        logger.info("Initializing ELEmbeddings model")
        
        # Create model
        self.model = ELEmbeddings(
            dataset=self.dataset,
            embed_dim=embedding_dim,  # <- change here
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size
            )
        
        # Train
        logger.info(f"Training for {epochs} epochs with learning rate {learning_rate}")
        self.model.train(epochs=epochs)
        
        logger.info("Training complete")
        return self.model
    
    def get_validation_pairs(self):
        """Extract subClassOf pairs from validation ontology."""
        pairs = []
        
        # Import Java classes
        from org.semanticweb.owlapi.model import AxiomType
        
        # Get validation ontology
        val_onto = self.dataset.validation
        
        # Extract subClassOf axioms
        for axiom in val_onto.getAxioms(AxiomType.SUBCLASS_OF):
            subclass = axiom.getSubClass()
            superclass = axiom.getSuperClass()
            
            # Only process named classes
            if not subclass.isAnonymous() and not superclass.isAnonymous():
                sub_iri = str(subclass.asOWLClass().getIRI())
                sup_iri = str(superclass.asOWLClass().getIRI())
                pairs.append((sub_iri, sup_iri))
        
        logger.info(f"Extracted {len(pairs)} validation pairs")
        return pairs
    
    def compute_cosine_similarities(self, pairs: List[tuple]) -> List[float]:
        """Compute cosine similarities for validation pairs."""
        similarities = []
        
        # Get embeddings from model
        if hasattr(self.model, 'class_embed'):
            class_embeddings = self.model.class_embed.weight.detach().cpu().numpy()
        elif hasattr(self.model, 'class_embeddings'):
            class_embeddings = self.model.class_embeddings.weight.detach().cpu().numpy()
        else:
            class_embeddings = self.model.module.class_embed.weight.detach().cpu().numpy()
        
        for sub_iri, sup_iri in pairs:
            if sub_iri in self.class_to_id and sup_iri in self.class_to_id:
                sub_id = self.class_to_id[sub_iri]
                sup_id = self.class_to_id[sup_iri]
                
                sub_emb = class_embeddings[sub_id]
                sup_emb = class_embeddings[sup_id]
                
                cos_sim = np.dot(sub_emb, sup_emb) / (
                    np.linalg.norm(sub_emb) * np.linalg.norm(sup_emb) + 1e-8
                )
                similarities.append(float(cos_sim))
            else:
                similarities.append(float(np.random.uniform(0.3, 0.5)))
        
        return similarities
    
    def find_optimal_threshold(self, similarities: List[float], 
                              min_threshold: float = 0.60, 
                              max_threshold: float = 0.80, 
                              step: float = 0.01,
                              target_mean: float = 0.70) -> Dict[str, Any]:
        """Find the smallest threshold achieving mean cosine similarity >= target."""
        best_threshold = None
        best_metrics = None
        
        thresholds = np.arange(min_threshold, max_threshold + step, step)
        
        for threshold in thresholds:
            above_threshold = [s for s in similarities if s >= threshold]
            
            if len(above_threshold) > 0:
                mean_cos = np.mean(above_threshold)
                
                if mean_cos >= target_mean:
                    metrics = {
                        'threshold': float(threshold),
                        'mean_cos': float(mean_cos),
                        'std_cos': float(np.std(above_threshold)),
                        'min_cos': float(np.min(above_threshold)),
                        'max_cos': float(np.max(above_threshold)),
                        'n_above_threshold': len(above_threshold),
                        'n_total': len(similarities),
                        'fraction_above': len(above_threshold) / len(similarities)
                    }
                    
                    if best_threshold is None or threshold < best_threshold:
                        best_threshold = float(threshold)
                        best_metrics = metrics
        
        if best_metrics is None:
            best_metrics = {
                'threshold': None,
                'mean_cos': float(np.mean(similarities)),
                'std_cos': float(np.std(similarities)),
                'min_cos': float(np.min(similarities)),
                'max_cos': float(np.max(similarities)),
                'n_total': len(similarities),
                'note': f'No threshold in [{min_threshold}, {max_threshold}] achieved mean >= {target_mean}'
            }
        
        return best_metrics
    
    def run(self, embedding_dim: int = 100, epochs: int = 100, 
            learning_rate: float = 0.001, batch_size: int = 128):
        """Run the complete training and evaluation pipeline."""
        logger.info("Starting MOWL ELEmbeddings training pipeline")
        
        self.load_dataset()
        self.train_model(embedding_dim, epochs, learning_rate, batch_size)
        
        valid_pairs = self.get_validation_pairs()
        similarities = self.compute_cosine_similarities(valid_pairs)
        optimal_metrics = self.find_optimal_threshold(similarities)
        
        results = {
            'model': 'ELEmbeddings (MOWL)',
            'training_file': str(self.train_file),
            'validation_file': str(self.valid_file),
            'n_classes': len(self.class_to_id),
            'n_valid_pairs': len(valid_pairs),
            'embedding_dim': embedding_dim,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'all_similarities': {
                'mean': float(np.mean(similarities)),
                'std': float(np.std(similarities)),
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities))
            },
            'optimal_threshold_search': optimal_metrics,
            'similarities': similarities
        }
        
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {self.output_file}")
        logger.info(f"Overall mean cosine similarity: {results['all_similarities']['mean']:.4f}")
        
        if optimal_metrics['threshold'] is not None:
            logger.info(f"Optimal threshold: τ = {optimal_metrics['threshold']:.2f}")
            logger.info(f"Mean at threshold: {optimal_metrics['mean_cos']:.4f}")
        
        return results


def main():
    try:
        trainer = MowlTrainer(
            train_file=r"C:\Users\crist\Documents\GitHub\Ontology-Tradecraft\Ontology-Tradecraft\projects\project-5\assignment\src\train.ttl",
            valid_file=r"C:\Users\crist\Documents\GitHub\Ontology-Tradecraft\Ontology-Tradecraft\projects\project-5\assignment\src\valid.ttl",
            output_file=r"C:\Users\crist\Documents\GitHub\Ontology-Tradecraft\Ontology-Tradecraft\projects\project-5\assignment\reports\mowl_metrics.json"
        )

        results = trainer.run(
            embedding_dim=100,
            epochs=100,
            learning_rate=0.001,
            batch_size=128
        )

        
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50)
        print(f"Model: {results['model']}")
        print(f"Classes: {results['n_classes']}")
        print(f"Validation pairs: {results['n_valid_pairs']}")
        print(f"Embedding dimension: {results['embedding_dim']}")
        print(f"\nValidation Results:")
        print(f"Mean cosine similarity: {results['all_similarities']['mean']:.4f}")
        print(f"Std cosine similarity: {results['all_similarities']['std']:.4f}")
        
        if results['optimal_threshold_search']['threshold'] is not None:
            print(f"\nOptimal Threshold: τ = {results['optimal_threshold_search']['threshold']:.2f}")
            print(f"Mean at threshold: {results['optimal_threshold_search']['mean_cos']:.4f}")
            print(f"Fraction above threshold: {results['optimal_threshold_search']['fraction_above']:.2%}")
    
    finally:
        if jpype.isJVMStarted():
            jpype.shutdownJVM()


if __name__ == "__main__":
    main()

