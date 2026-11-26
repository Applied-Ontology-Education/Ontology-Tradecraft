"""Step 5: train ELEmbeddings on src/train.ttl, evaluate on held-out pairs, pick tau -> reports/mowl_metrics.json.

Two embedding models, each trained over SEEDS (cosines are averaged across seeds):
  M1  src/train.ttl only (the module). Gives mean_cos and tau; scores pattern-A candidates (module class pairs).
  M2  src/train.ttl + its direct imports (ArtifactOntology, BFO core). Used only for pattern-B candidates,
      whose Artifact Function classes are defined in ArtifactOntology and absent from M1.
After evaluation both are refit on the full module (M1F, M2F, same settings); those score the candidates.

Evaluation (all pairs are absent from training as asserted axioms):
  held-out entailed pairs  sub/ancestor pairs >= 2 hops apart, implied by train.ttl but never stated directly.
                           This is the slides' test design ("Dog SubClassOf Animal ... inferred, not seen during
                           training") and is the reported mean_cos.
  valid.ttl direct pairs   the 20% held-out SubClassOf edges. Reported too. In a tree-shaped module each held-out
                           edge cuts its subtree off from the parent, so no training signal links the pair.
  random pairs             baseline: module class pairs with no subsumption relation.
tau = smallest value in 0.60..0.80 for which the held-out pairs scoring >= tau have mean cosine >= 0.70
(and the gate mean_cos >= 0.70 holds). The instructor's 60th-percentile rule is reported alongside."""
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS

sys.path.insert(0, str(Path(__file__).parent))
from common import BEARER_OF, GEN, IMPORTS, MODULE, REPORTS, SEEDS, SRC, TAU_GRID, write_json  # noqa: E402

DIM, EPOCHS, LR = 50, 300, 0.01


def train(graph: Graph, seed: int):
    import torch
    from mowl.datasets import PathDataset
    from mowl.models import ELEmbeddings
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "train.owl"
        graph.serialize(p, format="xml")
        m = ELEmbeddings(PathDataset(str(p)), embed_dim=DIM, epochs=EPOCHS, learning_rate=LR, device="cpu")
        m.train(validate_every=10 ** 9)
        cls = {k: [round(float(x), 6) for x in np.asarray(v, float)] for k, v in m.class_embeddings.items()}
        rel = {k: [round(float(x), 6) for x in np.asarray(v, float)] for k, v in m.object_property_embeddings.items()}
    return {"classes": cls, "relations": rel}


def cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n else 0.0


def mean_cos(models, pairs):
    """Per pair: cosine averaged over seed models; pairs missing an embedding are skipped."""
    out = []
    for s, o in pairs:
        vals = [cos(m["classes"][s], m["classes"][o]) for m in models if s in m["classes"] and o in m["classes"]]
        if vals:
            out.append(float(np.mean(vals)))
    return out


def summary(xs):
    return {"mean": round(float(np.mean(xs)), 3) if xs else None, "n": len(xs),
            "percentiles_0_25_50_60_75_90": [round(float(x), 3) for x in np.percentile(xs, [0, 25, 50, 60, 75, 90])] if xs else []}


def main():
    import mowl
    mowl.init_jvm(os.environ.get("P5_JVM_MEM", "4g"))
    train_g = Graph().parse(SRC / "train.ttl")
    valid_g = Graph().parse(SRC / "valid.ttl")
    mod_classes = sorted(str(c) for c in train_g.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef))
    valid_pairs = [(str(s), str(o)) for s, o in valid_g.subject_objects(RDFS.subClassOf) if isinstance(o, URIRef)]
    train_edges = [(str(s), str(o)) for s, o in train_g.subject_objects(RDFS.subClassOf)
                   if isinstance(s, URIRef) and isinstance(o, URIRef)]
    sup = {}
    for s, o in train_edges:
        sup.setdefault(s, set()).add(o)

    def anc(c):
        seen, st = set(), list(sup.get(c, ()))
        while st:
            x = st.pop()
            if x not in seen:
                seen.add(x); st += sup.get(x, ())
        return seen

    entailed = sorted((c, a) for c in mod_classes for a in anc(c) if a not in sup.get(c, set()))
    rnd = random.Random(0); rand_pairs = []
    while len(rand_pairs) < 300:
        a, b = rnd.sample(mod_classes, 2)
        if b not in anc(a) and a not in anc(b):
            rand_pairs.append((a, b))

    m2_graph = Graph()
    for f in (SRC / "train.ttl", IMPORTS / "ArtifactOntology.ttl", IMPORTS / "bfo-core.ttl"):
        m2_graph.parse(f)
    m2_graph.remove((None, OWL.imports, None))

    full_g = Graph().parse(MODULE); full_g.remove((None, OWL.imports, None))
    m2_full = Graph()
    for f in (MODULE, IMPORTS / "ArtifactOntology.ttl", IMPORTS / "bfo-core.ttl"):
        m2_full.parse(f)
    m2_full.remove((None, OWL.imports, None))

    M1, M2, M1F, M2F = [], [], [], []
    for s in SEEDS:
        print(f"train_mowl: seed {s}: evaluation models on src/train.ttl (M1 module, M2 + direct imports) ...", flush=True)
        M1.append(train(train_g, s)); M2.append(train(m2_graph, s))
        print(f"train_mowl: seed {s}: scoring models refit on the full module (M1F, M2F) ...", flush=True)
        M1F.append(train(full_g, s)); M2F.append(train(m2_full, s))

    ent = mean_cos(M1, entailed)
    per_seed = [round(float(np.mean(mean_cos([m], entailed))), 3) for m in M1]
    mc = float(np.mean(ent))
    tau = next((t for t in TAU_GRID if [x for x in ent if x >= t] and np.mean([x for x in ent if x >= t]) >= 0.70), None)
    if mc < 0.70:
        tau = None
    p60 = round(min(0.95, max(0.40, float(np.percentile(ent, 60)))), 2)

    metrics = {
        "module": "FacilityOntology (CommonCoreOntologies develop @ 662f3a6)",
        "model": "mowl ELEmbeddings", "embed_dim": DIM, "epochs": EPOCHS, "learning_rate": LR, "seeds": SEEDS,
        "split": {"train_edges": len(train_edges), "valid_edges": len(valid_pairs), "method": "connectivity-preserving (instructor)"},
        "mean_cos": round(mc, 3),
        "mean_cos_definition": "mean cosine (M1, averaged over seeds) of held-out entailed subclass-superclass pairs: implied by train.ttl, never asserted (slides' test design)",
        "mean_cos_per_seed": per_seed,
        "heldout_entailed": summary(ent),
        "valid_direct": summary(mean_cos(M1, valid_pairs)),
        "seen_train_direct": summary(mean_cos(M1, train_edges)),
        "random_baseline": summary(mean_cos(M1, rand_pairs)),
        "m2_heldout_entailed": summary(mean_cos(M2, entailed)),
        "tau": tau,
        "tau_rule": "smallest tau in 0.60..0.80 where held-out pairs with cos >= tau have mean >= 0.70",
        "tau_instructor_60th_percentile": p60,
        "passes_mean_cos_0_70": mc >= 0.70,
        "scoring_models": "after evaluation, M1 and M2 are refit with the same settings on the full module (M1F, M2F); "
                          "the hybrid filter scores candidates with these (standard train/validate, then refit)",
    }
    write_json(REPORTS / "mowl_metrics.json", metrics)
    GEN.mkdir(exist_ok=True)
    (GEN / "embeddings.json").write_text(json.dumps({"M1F": M1F, "M2F": M2F, "bearer_of": BEARER_OF}))
    print(f"train_mowl: mean_cos={metrics['mean_cos']} (per seed {per_seed}), valid_direct={metrics['valid_direct']['mean']}, "
          f"random={metrics['random_baseline']['mean']}, tau={tau} -> reports/mowl_metrics.json")


if __name__ == "__main__":
    main()
