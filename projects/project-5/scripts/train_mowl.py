#!/usr/bin/env python3
from __future__ import annotations

"""
Train an ELEmbeddings model with MOWL using train.ttl and valid.ttl.

This script demonstrates how to:
 - Load ontology files with MOWL's PathDataset
 - Initialize and train ELEmbeddings
 - Compute cosine similarities for held-out (subClassOf) pairs from valid.ttl
 - Sweep a decision threshold and write metrics to reports/mowl_metrics.json

Note: This is a lightweight, educational training harness. For larger runs,
adjust epochs/batch size and consider GPUs.
"""
import mowl

mowl.init_jvm("16g")
import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional

from mowl_ext.datasets import OtcPathDataset
from mowl_ext.otc_model import OtcModel

from rdflib import Graph, RDFS
from rdflib.term import URIRef

import torch
import optuna

from util.logger_config import config as _config_logger

logger = logging.getLogger(__name__)
_config_logger(logger)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_ROOT = PROJECT_ROOT / "reports"
SRC_ROOT = PROJECT_ROOT / "src"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MOWL ELEmbeddings on train/valid ontologies")
    p.add_argument("--train", default=str(SRC_ROOT / "train.ttl"), help="Path to training ontology (TTL/OWL)")
    p.add_argument("--valid", default=str(SRC_ROOT / "valid.ttl"), help="Path to validation ontology (TTL/OWL)")
    p.add_argument("--eval-scope", choices=["all", "validation", "train"], default="all",
                   help="Class universe exposed via dataset.evaluation_classes: all | validation | train")
    p.add_argument("--epochs", type=int, default=150, help="Training epochs")
    p.add_argument("--dim", type=int, default=2, help="Embedding dimension")
    p.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    p.add_argument("--batch", type=int, default=32, help="Batch size for trainer")
    p.add_argument("--margin", type=float, default=2.5, help="Margin")
    p.add_argument("--reg-norm", type=float, default=0.00013, help="Regularization")

    p.add_argument("--optuna", type=bool, default=False, help="Use Optuna to tune dim, lr, margin, and reg_norm")
    p.add_argument("--trials", type=int, default=25, help="Number of Optuna trials to run when --optuna is set")
    p.add_argument("--study-name", type=str, default="mowl-elembeddings", help="Optuna study name")
    p.add_argument("--storage", type=str, default=None,
                   help="Optuna storage URL (e.g., sqlite:///optuna.db). If not set, use in-memory")
    p.add_argument("--out", default=str(REPORTS_ROOT / "mowl_metrics.json"), help="Path to write metrics JSON")
    return p.parse_args()


def load_dataset(train_path: Path, valid_path: Path, eval_scope: str):
    if not train_path.exists():
        raise FileNotFoundError(f"Training ontology not found: {train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Validation ontology not found: {valid_path}")
    # Use our project-specific dataset that overrides evaluation_classes
    ds = OtcPathDataset(
        ontology_path=train_path.as_posix(),
        validation_path=valid_path.as_posix(),
        eval_scope=eval_scope,
    )
    return ds


def init_model(dataset, dim: int, lr: float, margin: float, batch_size: int, reg_norm: float,
               model_suffix: str | None = None):
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    filename = "elembeddings.model" if not model_suffix else f"elembeddings.{model_suffix}.model"
    model_filepath = (ckpt_dir / filename).as_posix()
    kwargs = {
        "embed_dim": dim,
        "learning_rate": lr,
        "margin": margin,
        "model_filepath": model_filepath,
        "batch_size": batch_size,
        "reg_norm": reg_norm,
        "device": 'cuda:0'
    }
    model = OtcModel(dataset, **kwargs)
    return model


def train_model(model, dataset, epochs: int):
    logger.info("Starting training for %d epochs...", epochs)
    try:
        model.train(epochs=epochs)
    except Exception as e:
        logger.error("model.train(epochs=..) failed: %s", e)
        raise RuntimeError("Training failed: ELEmbeddings.train could not complete.")


def extract_valid_subclass_pairs(valid_owl: Path) -> List[Tuple[str, str]]:
    g = Graph()
    g.parse(valid_owl.as_posix())

    pairs: List[Tuple[str, str]] = []
    for s, _, o in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            s_iri = str(s)
            o_iri = str(o)
            if (s_iri.startswith("http://") or s_iri.startswith("https://")) and \
                (o_iri.startswith("http://") or o_iri.startswith("https://")):
                pairs.append((s_iri, o_iri))
    return pairs


def compute_mean_cosine(model, dataset, pairs: List[Tuple[str, str]]) -> Tuple[float, List[float]]:
    cos = torch.nn.functional.cosine_similarity
    sims: List[float] = []
    emb_tuple = model.get_embeddings()
    ent_embeds = emb_tuple[0]
    for sub_iri, sup_iri in pairs:
        v1_np = ent_embeds.get(sub_iri)
        v2_np = ent_embeds.get(sup_iri)
        v1 = torch.tensor(v1_np)
        v2 = torch.tensor(v2_np)
        cos_sim = cos(v1, v2, dim=0)
        sims.append(cos_sim.item())
    if sims:
        mean_cos = sum(sims) / len(sims)
        return mean_cos, sims
    else:
        logger.warning(
            "No validation pairs found in embedding dict keys; falling back to tensor-based access.")


def pick_threshold(
    sims: List[float],
    candidates: List[float],
    quality_bar: float = 0.70,
) -> Tuple[Optional[float], float]:
    overall_mean = sum(sims) / len(sims)
    selected: Optional[float] = None

    for candidate in candidates:
        selection = [sim for sim in sims if sim > candidate]
        if len(selection) > 0:
            avg = sum(selection) / len(selection)
            if avg >= quality_bar:
                selected = candidate
                overall_mean = avg
                break

    return selected, overall_mean


def write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def _validation_loss_from_model(model) -> float:
    """Fetch best available validation loss from the model; inf if unavailable."""
    val_loss = getattr(model, "best_validation_loss", None)
    if val_loss is None:
        val_loss = getattr(model, "last_validation_loss", None)
    if val_loss is None:
        return float('inf')
    try:
        return float(val_loss)
    except Exception:
        return float('inf')


def _optuna_objective(dataset, base_args) -> optuna.trial.TrialCallable:
    """Create an Optuna objective callable that trains OtcModel and returns validation loss."""

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        dim = trial.suggest_categorical("dim", [2, 4, 8, 16, 32])
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        margin = trial.suggest_float("margin", 0.1, 2.5)
        reg_norm = trial.suggest_float("reg_norm", 1e-6, 10.0, log=True)

        suffix = f"optuna.d{dim}.lr{lr:.2e}.m{margin:.2f}.r{reg_norm:.2e}"
        model = init_model(
            dataset,
            dim=dim,
            lr=lr,
            margin=margin,
            batch_size=base_args.batch,
            reg_norm=reg_norm,
            model_suffix=suffix,
        )
        try:
            train_model(model, dataset, epochs=base_args.epochs)
        except Exception as e:
            logger.warning("Optuna trial failed during training: %s", e)
            return float('inf')

        loss = _validation_loss_from_model(model)
        # Report intermediate value to Optuna for pruning
        trial.report(loss, step=base_args.epochs)
        # If using pruning, respect it
        if trial.should_prune():
            raise optuna.TrialPruned()
        return loss

    return objective


def main(
    train: str = str(SRC_ROOT / "train.ttl"),
    valid: str = str(SRC_ROOT / "valid.ttl"),
    eval_scope: str = "all",
    epochs: int = 150,
    dim: int = 2,
    lr: float = 0.003,
    batch: int = 32,
    margin: float = 2.5,
    reg_norm: float = 0.00013,
    # Optuna controls
    use_optuna: bool = False,
    trials: int = 25,
    study_name: str = "mowl-elembeddings",
    storage: str | None = None,
    out: str = str(REPORTS_ROOT / "mowl_metrics.json"),
):
    train_path = Path(train)
    valid_path = Path(valid)
    out_path = Path(out)

    logger.info("Train ontology: %s", train_path)
    logger.info("Valid ontology: %s", valid_path)
    logger.info("Output metrics: %s", out_path)

    # Prepare dataset
    dataset = load_dataset(train_path, valid_path, eval_scope)

    # Determine hyperparameters to use (Optuna or single)
    final_reg_norm = None
    best_valid = None
    optuna_used = False
    optuna_summary = None
    final_dim = dim
    final_lr = lr
    final_margin = margin
    final_reg_norm = reg_norm

    # If Optuna is requested, run HPO; otherwise do a single run
    if use_optuna:
        optuna_used = True
        # Create study
        try:
            study = optuna.create_study(
                direction="minimize",
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
            ) if storage else optuna.create_study(direction="minimize", study_name=study_name)
        except Exception as e:
            logger.warning("Failed to create persistent Optuna study (%s). Falling back to in-memory.", e)
            study = optuna.create_study(direction="minimize", study_name=study_name)

        class _ArgsShim:
            # minimal object exposing attributes used by _optuna_objective
            def __init__(self):
                self.dim = dim
                self.lr = lr
                self.margin = margin
                self.reg_norm = reg_norm
                self.batch = batch
                self.epochs = epochs

        objective = _optuna_objective(dataset, _ArgsShim())
        logger.info("Starting Optuna optimization for %d trials...", trials)
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        best = study.best_trial
        best_params = best.params
        best_value = float(best.value) if best.value is not None else None
        logger.info("Optuna best params: %s (val_loss=%.6f)", best_params,
                    best_value if best_value is not None else float('inf'))

        # Set final hyperparameters
        final_dim = int(best_params.get('dim', final_dim))
        final_lr = float(best_params.get('lr', final_lr))
        final_margin = float(best_params.get('margin', final_margin))
        final_reg_norm = float(best_params.get('reg_norm', 1.0))

        # Re-instantiate model with selected hyperparameters
        selected_suffix = f"optuna.best.d{final_dim}.lr{final_lr:.2e}.m{final_margin:.2f}.r{final_reg_norm:.2e}"
        model = init_model(dataset, dim=final_dim, lr=final_lr, margin=final_margin, batch_size=batch,
                           reg_norm=final_reg_norm, model_suffix=selected_suffix)
        # Train the chosen configuration once more to ensure weights are present
        train_model(model, dataset, epochs=epochs)

        # Prepare Optuna summary for metrics
        optuna_summary = {
            "study_name": study_name,
            "storage": storage,
            "n_trials": len(study.trials),
            "best_params": best_params,
            "best_value": best_value,
        }

    else:
        # Single run with default reg_norm=1.0
        model = init_model(dataset, dim=final_dim, lr=final_lr, margin=final_margin, batch_size=batch,
                           reg_norm=final_reg_norm)
        train_model(model, dataset, epochs=epochs)

    # Evaluate cosine on validation subclass pairs using the selected/best model
    pairs = extract_valid_subclass_pairs(valid_path)
    mean_cos, sims = compute_mean_cosine(model, dataset, pairs)
    tau_candidates = [0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    tau, overall_mean = pick_threshold(sims, tau_candidates)

    # Capture the model checkpoint filepath to expose it in metrics for downstream scripts
    try:
        best_model_path = getattr(model, "model_filepath", None)
    except Exception:
        best_model_path = None
    best_model_filename = None
    if isinstance(best_model_path, str):
        try:
            best_model_filename = Path(best_model_path).name
        except Exception:
            best_model_filename = None

    metrics = {
        "model": "ELEmbeddings",
        "epochs": epochs,
        "dim": final_dim,
        "lr": final_lr,
        "margin": final_margin,
        "batch": batch,
        "reg_norm": final_reg_norm,
        "optuna_used": optuna_used,
        "optuna": optuna_summary,
        "train_path": train_path.as_posix(),
        "valid_path": valid_path.as_posix(),
        "num_valid_pairs": len(pairs),
        "cosine_similarities": sims,
        "mean_cos": overall_mean,
        "selected_tau": tau,
        "tau_candidates": tau_candidates,
        "best_model_path": best_model_path,
        "best_model_filename": best_model_filename,
    }
    write_metrics(out_path, metrics)

    logger.info("Training complete. mean_cos=%.4f, selected_tau=%s. Wrote %s", overall_mean, tau, out_path)


if __name__ == "__main__":
    args = parse_args()
    main(
        train=args.train,
        valid=args.valid,
        eval_scope=args.eval_scope,
        epochs=args.epochs,
        dim=args.dim,
        lr=args.lr,
        batch=args.batch,
        margin=args.margin,
        reg_norm=args.reg_norm,
        use_optuna=args.optuna,
        trials=args.trials,
        study_name=args.study_name,
        storage=args.storage,
        out=args.out,
    )
