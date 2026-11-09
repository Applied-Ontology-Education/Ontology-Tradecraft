import mowl
mowl.init_jvm("4g")          # must run before the other mowl imports

import json
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
import rdflib
import torch
import yaml
from mowl.datasets import PathDataset
from rdflib import OWL, RDFS, URIRef
from mowl.models import ELEmbeddings

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config.yaml"

def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()
tcfg = cfg["train"]
train_path = PROJECT_DIR / cfg["paths"]["train"]
valid_path = PROJECT_DIR / cfg["paths"]["valid"]

seed = tcfg["seed"]
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

dataset = PathDataset(str(train_path), validation_path=str(valid_path))

# mowl 2.x on Windows crashes writing its normalization cache (backslash path),
# and only caches when it knows the file paths, so clear them after loading.
dataset.ontology_path = dataset.validation_path = dataset.testing_path = None

model = ELEmbeddings(dataset, embed_dim=tcfg["embed_dim"],
                     learning_rate=tcfg["learning_rate"])

print("Training datasets:")
for name, ds in model.training_datasets.items():
    print(f"  {name}: {len(ds)}")

model.eval_gci_name = "gci0"   # validate on plain subClassOf axioms
model.train(epochs=tcfg["epochs"])


def load_valid_pairs(path):
    g = rdflib.Graph()
    g.parse(path, format="turtle")
    return sorted((str(s), str(o)) for s, o in g.subject_objects(RDFS.subClassOf)
                  if isinstance(s, URIRef) and isinstance(o, URIRef))

def cosine(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

emb = model.class_embeddings          # property rebuilds the dict on each access, so grab once
valid_pairs = load_valid_pairs(valid_path)

valid_scores, missing = [], []
for sub, sup in valid_pairs:
    if sub in emb and sup in emb:
        valid_scores.append(cosine(emb[sub], emb[sup]))
    else:
        missing.append((sub, sup))

mean_cos_all = float(np.mean(valid_scores))
print(f"\nHeld-out pairs: {len(valid_pairs)}  scored: {len(valid_scores)}  missing: {len(missing)}")
print(f"mean_cos_all: {mean_cos_all:.3f}")


# Baseline: same subclasses, paired with a random class that is neither an
# ancestor nor a descendant, so a high mean_cos can be told apart from collapse.
g_all = rdflib.Graph()
g_all.parse(train_path, format="turtle")
g_all.parse(valid_path, format="turtle")

def related(cls):
    node = URIRef(cls)
    up = {str(x) for x in g_all.transitive_objects(node, RDFS.subClassOf)}
    down = {str(x) for x in g_all.transitive_subjects(RDFS.subClassOf, node)}
    return up | down          # includes cls itself

pool = sorted(k for k in emb if not k.startswith(str(OWL)))

neg_scores = []
for sub, _ in valid_pairs:
    bad = related(sub)
    neg = random.choice([c for c in pool if c not in bad])
    neg_scores.append(cosine(emb[sub], emb[neg]))

baseline_mean_cos = float(np.mean(neg_scores))
print(f"baseline_mean_cos: {baseline_mean_cos:.3f}  (gap: {mean_cos_all - baseline_mean_cos:+.3f})")

# Diagnostic: same score on the training subclass pairs the model actually saw.
train_pairs = load_valid_pairs(train_path)
train_scores = [cosine(emb[a], emb[b]) for a, b in train_pairs if a in emb and b in emb]
print(f"train_mean_cos (diagnostic): {np.mean(train_scores):.3f}  ({len(train_scores)} pairs)")


# Choose τ: smallest candidate where the mean cosine of pairs scoring ≥ τ
# reaches the target. Coverage = share of held-out pairs that clear τ.
scores = np.array(valid_scores)
tau_table = []
for tau in sorted(tcfg["tau_candidates"]):
    above = scores[scores >= tau]
    tau_table.append({
        "tau": tau,
        "n_above": int(len(above)),
        "coverage": len(above) / len(scores),
        "mean_cos_at_tau": float(above.mean()) if len(above) else None,
    })

print("\n  tau   n_above  coverage  mean_cos_at_tau")
for r in tau_table:
    m = f"{r['mean_cos_at_tau']:.3f}" if r["mean_cos_at_tau"] is not None else "  -"
    print(f"  {r['tau']:.2f}  {r['n_above']:>7}  {r['coverage']:>8.1%}  {m}")

passing = [r for r in tau_table
           if r["mean_cos_at_tau"] is not None and r["mean_cos_at_tau"] >= tcfg["target_mean_cos"]]
chosen = passing[0] if passing else None
print(f"\nchosen_tau: {chosen['tau'] if chosen else None}")


metrics = {
    "mean_cos_all": round(mean_cos_all, 4),
    "mean_cos_at_tau": round(chosen["mean_cos_at_tau"], 4) if chosen else None,
    "coverage_at_tau": round(chosen["coverage"], 4) if chosen else None,
    "chosen_tau": chosen["tau"] if chosen else None,
    "target_mean_cos": tcfg["target_mean_cos"],
    "baseline_mean_cos": round(baseline_mean_cos, 4),
    "train_mean_cos": round(float(np.mean(train_scores)), 4),
    "n_valid_pairs": len(valid_pairs),
    "n_missing_embeddings": len(missing),
    "tau_table": tau_table,
    "gci_counts": {name: len(ds) for name, ds in model.training_datasets.items()},
    "model": {
        "name": "ELEmbeddings",
        "mowl_version": version("mowl-borg"),
        "embed_dim": tcfg["embed_dim"],
        "learning_rate": tcfg["learning_rate"],
        "epochs": tcfg["epochs"],
        "seed": seed,
    },
    "hybrid": None,   # filled in by filter_candidates_hybrid.py (step 6)
}

# Save class and property vectors for step 6 (filter_candidates_hybrid.py).
rel = model.object_property_embeddings
emb_path = PROJECT_DIR / cfg["paths"]["embeddings"]
emb_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(emb_path,
         class_iris=np.array(list(emb.keys())), class_vecs=np.stack(list(emb.values())),
         prop_iris=np.array(list(rel.keys())), prop_vecs=np.stack(list(rel.values())))
print(f"\nWrote {emb_path.relative_to(PROJECT_DIR)}  ({len(emb)} classes, {len(rel)} properties)")

out = PROJECT_DIR / cfg["paths"]["mowl_metrics"]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"\nWrote {out.relative_to(PROJECT_DIR)}")
