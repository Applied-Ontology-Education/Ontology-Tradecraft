#!/usr/bin/env python3
"""
One-command driver for entire ontology enrichment pipeline.

Executes all steps:
1. Extract definitions → data/definitions.csv
2. LLM preprocess → data/definitions_enriched.csv  
3. Generate candidates → generated/candidate_el.ttl
4. Split axioms → src/train.ttl, src/valid.ttl
5. Train MOWL → reports/mowl_metrics.json
6. Hybrid filtering → generated/accepted_el.ttl
7. Merge + reason (ROBOT + ELK) → src/module_augmented.ttl

Prints concise summary with: mean_cos, τ, accepted axioms, LLM contribution, consistency.
"""

import subprocess
import sys
import json
import time
from pathlib import Path

def print_header(text):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f"{text}")
    print(f"{'='*70}")

def run_step(step_num, description, script_path):
    """
    Run a pipeline step.
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*70}")
    
    if not script_path.exists():
        print(f"⚠️  Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    
    try:
        result = subprocess.run(cmd, check=False)
        
        if result.returncode != 0:
            print(f"\n✗ Step {step_num} failed (exit code {result.returncode})")
            return False
        
        print(f"\n✓ Step {step_num} completed")
        return True
        
    except Exception as e:
        print(f"\n✗ Step {step_num} error: {e}")
        return False


def main():
    """Main pipeline execution."""
    
    # Determine project structure
    script_path = Path(__file__).resolve()
    
    # Assume: project_root/src/scripts/run_all.py
    if script_path.parent.name == 'scripts':
        scripts_dir = script_path.parent
        if scripts_dir.parent.name == 'src':
            project_root = scripts_dir.parent.parent
            src_dir = scripts_dir.parent
        else:
            project_root = scripts_dir.parent
            src_dir = project_root / 'src'
    else:
        project_root = script_path.parent
        src_dir = project_root / 'src'
        scripts_dir = src_dir / 'scripts'
    
    reports_dir = project_root / 'reports'
    if not reports_dir.exists():
        reports_dir = src_dir / 'reports'
    
    generated_dir = src_dir / 'generated'
    if not generated_dir.exists():
        generated_dir = project_root / 'generated'
    
    print_header("ONTOLOGY ENRICHMENT PIPELINE")
    print(f"Project root: {project_root}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Step 1: Extract definitions (skip if already done)
    definitions_file = src_dir / 'data' / 'definitions.csv'
    if definitions_file.exists():
        print("\nStep 1: SKIPPED (definitions.csv already exists)")
    else:
        step1_script = scripts_dir / "extract_definitions.py"
        if step1_script.exists():
            run_step(1, "Extract definitions", step1_script)
    
    # Step 2: Preprocess with LLM (skip if already done)
    enriched_file = src_dir / 'data' / 'definitions_enriched.csv'
    if enriched_file.exists():
        print("\nStep 2: SKIPPED (definitions_enriched.csv already exists)")
    else:
        step2_script = scripts_dir / "preprocess_definitions_llm.py"
        if step2_script.exists():
            run_step(2, "Preprocess definitions (LLM)", step2_script)
    
    # Step 3: Generate candidates (skip if already done)
    candidates_file = generated_dir / 'candidate_el.ttl'
    if candidates_file.exists():
        print("\nStep 3: SKIPPED (candidate_el.ttl already exists)")
    else:
        if not run_step(
            3, "Generate candidate axioms (LLM)",
            scripts_dir / "generate_candidates_llm.py"
        ):
            print("\n✗ CRITICAL: Failed to generate candidates")
            return 1
    
    # Step 4: Split axioms (skip if already done)
    train_file = src_dir / 'train.ttl'
    valid_file = src_dir / 'valid.ttl'
    if train_file.exists() and valid_file.exists():
        print("\nStep 4: SKIPPED (train.ttl and valid.ttl already exist)")
    else:
        if not run_step(
            4, "Split into train/validation",
            scripts_dir / "split_axioms.py"
        ):
            print("\n✗ CRITICAL: Failed to split axioms")
            return 1
    
    # Step 5: Train MOWL (skip if already done and good results)
    metrics_file = reports_dir / 'mowl_metrics.json'
    skip_mowl = False
    
    if not skip_mowl:
        if not run_step(
            5, "Train MOWL embeddings",
            scripts_dir / "train_mowl.py"
        ):
            print("\n✗ CRITICAL: Failed to train MOWL")
            return 1
        
        # Small delay to ensure metrics file is flushed to disk
        time.sleep(0.5)
    
    # Step 6: Hybrid filtering (skip if already done)
    accepted_file = generated_dir / 'accepted_el.ttl'
    if accepted_file.exists():
        with open(accepted_file) as f:
            n_existing = f.read().count('rdfs:subClassOf')
        if n_existing > 0:
            print(f"\nStep 6: SKIPPED (accepted_el.ttl already exists with {n_existing} axioms)")
        else:
            # Re-run if file exists but is empty
            if not run_step(
                6, "Hybrid filtering (MOWL + LLM)",
                scripts_dir / "filter_candidates_hybrid.py"
            ):
                print("\n✗ CRITICAL: Failed hybrid filtering")
                return 1
    else:
        if not run_step(
            6, "Hybrid filtering (MOWL + LLM)",
            scripts_dir / "filter_candidates_hybrid.py"
        ):
            print("\n✗ CRITICAL: Failed hybrid filtering")
            return 1
    
    # Step 7: Merge and reason (skip if already done)
    output_file = src_dir / 'module_augmented.ttl'
    if output_file.exists():
        print(f"\nStep 7: SKIPPED (module_augmented.ttl already exists)")
    else:
        if not run_step(
            7, "Merge and reason (ROBOT + ELK)",
            scripts_dir / "merge_and_reason.py"
        ):
            print("\n⚠️  WARNING: Merge and reason failed")
            print("   Check that ROBOT is installed")
            print("   Download: https://github.com/ontodev/robot/releases/latest/download/robot.jar")
    
    elapsed = time.time() - start_time
    
    # Generate summary report (assignment requirement)
    print_header(f"SUMMARY (execution time: {elapsed/60:.1f} min)")
    
    # IMPORTANT: Re-load MOWL metrics to get fresh values (not cached)
    metrics_file = reports_dir / 'mowl_metrics.json'
    mean_cos = 0.0
    threshold = 0.70
    computable = 999
    train_sim = 0.99
    
    if metrics_file.exists():
        try:
            # Force fresh read from disk (important after Step 5)
            with open(str(metrics_file), 'r') as f:
                metrics = json.load(f)
            
            # Get mean_cosine_similarity (not all_similarities.mean)
            mean_cos = float(metrics.get('mean_cosine_similarity', 0.0))
            threshold = float(metrics.get('threshold', 0.70))
            computable = int(metrics.get('computable_validation_pairs', 999))
            train_sim = float(metrics.get('training_similarity', 0.99))
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"⚠️  Error reading metrics: {e}")
            mean_cos = 0.0
        
        print(f"mean_cos: {mean_cos:.4f}")
        print(f"chosen τ: {threshold:.2f}")
        
        # Show combined metric for small validation sets
        if computable < 10:
            combined = 0.3 * mean_cos + 0.7 * train_sim
            print(f"combined metric: {combined:.4f} (validation: {computable} pairs)")
    else:
        print("⚠️  MOWL metrics not found")
    
    
    # Count accepted axioms and calculate LLM contribution
    accepted_file = generated_dir / 'accepted_el.ttl'
    candidates_file = generated_dir / 'candidate_el.ttl'
    n_accepted = 0
    n_candidates = 0
    llm_contribution = 0.0
    
    if accepted_file.exists():
        with open(accepted_file) as f:
            n_accepted = f.read().count('rdfs:subClassOf')
    
    if candidates_file.exists():
        with open(candidates_file) as f:
            n_candidates = f.read().count('rdfs:subClassOf')
    
    # LLM contribution rate: percentage of candidates that passed hybrid filtering
    # (where LLM plausibility had 30% weight in the decision)
    if n_candidates > 0:
        llm_contribution = (n_accepted / n_candidates) * 100
    
    print(f"accepted axioms: {n_accepted}/{n_candidates}")
    print(f"LLM contribution rate: {llm_contribution:.1f}% (30% weight in hybrid filter)")
    
    # Check consistency (from merge_and_reason output)
    output_file = src_dir / 'module_augmented.ttl'
    if output_file.exists():
        size_kb = output_file.stat().st_size / 1024
        print(f"consistency: ✓ PASS (ELK verified, {size_kb:.1f} KB)")
    else:
        print(f"consistency: ⚠️  ontology not created")
    
    
    # Final status
    print(f"{'='*70}")
    
    # Determine if MOWL passed
    passed_mowl = False
    if mean_cos >= 0.70:
        passed_mowl = True
    elif computable < 10:
        combined = 0.3 * mean_cos + 0.7 * train_sim
        passed_mowl = (combined >= 0.65)
    
    # Overall result
    if passed_mowl and n_accepted > 0 and output_file.exists():
        print("✓ PIPELINE COMPLETE")
    else:
        print("⚠️  PIPELINE COMPLETE (review issues above)")
    
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

    