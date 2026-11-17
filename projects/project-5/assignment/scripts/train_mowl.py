#train_mowl.py
#Step 5 train an ELembeddings model (LOWL) on src/train.ttl file.
#check it on the held out links in src/valid.ttl file
#save the results to report/mowl_mtrics.json file
#
#run this from the assignment folder
#pythin/scripts/traiin_mowl.py

#1 Imports
import json
import random
from pathlib import Path

import numpy as np
import torch
import rdflib
from rdflib import RDFS, URIRef

import mowl 
mowl.init_jvm("4g") #mowl uses java in the background and the 4g is 4 gig of memory
from mowl.datasets import PathDataset
from mowl.models import ELEmbeddings

#2 Paths (from embedding file)
BASE = Path(__file__).resolve().parent.parent
TRAIN = BASE / "src" / "train.ttl"
VALID = BASE / "src" / "valid.ttl"
REPORT = BASE / "reports" / "mowl_metrics.json"
REPORT.parent.mkdir(exist_ok=True)

#3 settings if mean cosine is <.70 adjust these values and rerun. (MOWL default settings)
SEED = 42
EMBED_DIM = 50
EPOCHS = 3000
LEARNING_RATE = 0.01
MARGIN = -0.1
TAU_GRID =[0.60, 0.65, 0.70, 0.75, 0.80]
TARGET = 0.70

#fix the random numbers so that each run provides the same resulr
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#4 load the train and valid ttl files into mowl
dataset = PathDataset(str(TRAIN), validation_path=str(VALID))
print("Classes loaded:", len(dataset.classes))

#5 train the model
model = ELEmbeddings( #set the model up with the settings
    dataset,
    embed_dim=EMBED_DIM,
    margin=MARGIN,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    model_filepath=str(BASE / "data" / "embeddings.pt"), #location where mowl will store the model file
)
##model.train(validate_every=100)  #training

model.train(validate_every=EPOCHS +1) ##use epochs and plus one to prevent the validation from running and allows me to change my EPOCHS setting without impacting this

#train elembeddings on the train filr and get the cosine for the held out pairs

#6 vector per classs
class_vectors = model.class_embeddings

def vec(IRI):
    return np.array(class_vectors[IRI])

def cosine (a, b):
    return float(np.dot (a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

#read the held out links from valid.ttl
g = rdflib.Graph()
g.parse(VALID, format='turtle')

valid_pairs = []
for child, parent in g.subject_objects(RDFS.subClassOf):
    if isinstance(child, URIRef) and isinstance (parent, URIRef):
        valid_pairs.append((str(child), str(parent)))
valid_pairs.sort()
print("Held-out pairs:", len(valid_pairs))

#cosine for each held out pair
scores = []
skipped = []
for child, parent in valid_pairs:
    if child not in class_vectors or parent not in class_vectors:
        skipped.append([child, parent])
        continue
    c = cosine(vec(child), vec(parent))
    scores.append(c)
    print(f" {c:.3f} {child.split('/')[-1]} -> {parent.split('/')[-1]}")

#average over all held out pairs (after the loop, run once)
mean_cos = float(np.mean(scores)) if scores else 0.0
print(f"\nmean_cos (valid): {mean_cos:.3f} (target {TARGET})")

#sweep for the threshold value- pick the smallest tau that meets the .0 condition
#Rule -- for each Tau, keep each held out pair where the cosine >= Tau
#choose the smalles Tau where kept pairs avg at least .70

tau = None
sweep = []
for t in TAU_GRID:
    kept = [s for s in scores if s >= t]
    kept_mean = float(np.mean(kept)) if kept else 0.0
    sweep.append({"tau": t, "pairs_kept": len(kept), "mean_cos_kept": round(kept_mean, 3)})
    print(f" t={t:.2f}: kept {len(kept)}/{len(scores)}, mean_cos of kept = {kept_mean:.3f}")
    if tau is None and kept and kept_mean >= TARGET:
        tau = t

print("Chose t:", tau)
print("Passes (mean_cos >= 0.70):", mean_cos >= TARGET)

#save report
report = {
    "model": "ELEmbeddings (mowl-borg 1.0.2)",
    "train_file": "src/train.ttl",
    "valid_file": "src/valid.ttl",
    "settings": {
            "embed_dim": EMBED_DIM,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "margin": MARGIN,
            "seed": SEED,
    },
    "validPairs": len(valid_pairs),
    "pairs_scored": len(scores),
    "pairs_skipped": skipped,
    "mean_cos": round(mean_cos, 3),
    "passed": mean_cos >= TARGET,
    "tau": tau,
    "tau_rule": "smallest tau in [0.60, 0.65, 0.70, 0.75, 0.80] where held out pairs with cosine >= tau have mean cosine >= 0.70",
    "tau_sweep": sweep,
    }
REPORT.write_text(json.dumps(report, indent=2))
print("Saved:", REPORT)



#For step 6 - save the vectors for reuse.
# .tolist() turns each numpy vector into a plain list taht can be stored in json
class_out = {}
for iri, v in model.class_embeddings.items(): #vector per iRI
    class_out[iri] = v.tolist()  #list of numbers
property_out ={}
for iri, v in model.object_property_embeddings.items():   #one vector per object property to determine some shape capability
    property_out[iri] = v.tolist()

VECTORS = BASE / "data" / "mowl_embeddings.json"
VECTORS.write_text(json.dumps({"classes": class_out, "properties": property_out}))
print("Saved:", VECTORS, len(class_out), "classes,", len(property_out), "properties")
