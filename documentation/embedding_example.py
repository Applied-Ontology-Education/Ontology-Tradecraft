# --- Ontology Embeddings via Hybrid [PPMI + SVD] (connectivity-preserving split) ---
# Uses concatenated vectors: [TruncatedSVD(PPMI)  |  raw PPMI], L2-normalized, then cosine.
# This preserves fine-grained co-occurrence and usually lifts VALID/TEST above 0.

!pip -q install rdflib==7.0.0 networkx==3.2.1 matplotlib==3.8.4 requests==2.32.4

import random, json, requests, numpy as np, matplotlib.pyplot as plt, rdflib, networkx as nx
from rdflib.namespace import RDF, RDFS, OWL
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

random.seed(42); np.random.seed(42)

# 1) Download ontology
ARTI_URL = "https://raw.githubusercontent.com/CommonCoreOntology/CommonCoreOntologies/refs/heads/develop/src/cco-modules/ArtifactOntology.ttl"
ttl = requests.get(ARTI_URL, timeout=60).text

# 2) Parse and extract classes + rdfs:subClassOf edges
g = rdflib.Graph(); g.parse(data=ttl, format="turtle")

def iri(u):
    from rdflib.term import URIRef
    return str(u) if isinstance(u, URIRef) else None

# Labels
label_map = {}
for s, p, o in g.triples((None, RDFS.label, None)):
    if isinstance(o, rdflib.term.Literal):
        label_map[str(s)] = str(o)

def short_label(uri: str) -> str:
    if uri in label_map: return label_map[uri]
    frag = uri.split('#')[-1]
    return frag.split('/')[-1]

classes = set()
for s, p, o in g.triples((None, RDF.type, OWL.Class)):
    u = iri(s)
    if u: classes.add(u)

edges = []
for s, p, o in g.triples((None, RDFS.subClassOf, None)):
    u, v = iri(s), iri(o)
    if u and v:
        classes.add(u); classes.add(v)
        edges.append((u, v))

edges = [(u, v) for (u, v) in edges if u in classes and v in classes]
print(f"Classes: {len(classes):,}  SubClassOf edges: {len(edges):,}")

# 3) Connectivity-preserving split (80/10/10 target)
UG_full = nx.Graph(); UG_full.add_nodes_from(classes); UG_full.add_edges_from(edges)
target_train = int(0.8 * len(edges))
target_valid = int(0.1 * len(edges))
target_test  = len(edges) - target_train - target_valid

def can_remove_edge(edge_list, a, b):
    deg = {}
    for x, y in edge_list:
        deg[x] = deg.get(x, 0) + 1
        deg[y] = deg.get(y, 0) + 1
    return deg.get(a,0) > 1 and deg.get(b,0) > 1

train_set = set(edges)
deg_full = dict(UG_full.degree())
candidates = [(u,v) for (u,v) in edges if deg_full.get(u,0) >= 2 and deg_full.get(v,0) >= 2]
random.shuffle(candidates)

valid_edges, test_edges = [], []
for (u,v) in candidates:
    if len(valid_edges) < target_valid and (u,v) in train_set:
        tmp = list(train_set); tmp.remove((u,v))
        if can_remove_edge(tmp, u, v): train_set.remove((u,v)); valid_edges.append((u,v))
    elif len(test_edges) < target_test and (u,v) in train_set:
        tmp = list(train_set); tmp.remove((u,v))
        if can_remove_edge(tmp, u, v): train_set.remove((u,v)); test_edges.append((u,v))
    if len(valid_edges) >= target_valid and len(test_edges) >= target_test:
        break

train_edges = list(train_set)
print(f"Split sizes: train={len(train_edges)}, valid={len(valid_edges)}, test={len(test_edges)}")

# 4) TRAIN graph
G_train = nx.DiGraph(); G_train.add_nodes_from(classes); G_train.add_edges_from(train_edges)
UG_train = G_train.to_undirected()

# 5) Random walks
def random_walks(graph, num_walks=40, walk_len=20):
    nodes = list(graph.nodes())
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for n in nodes:
            walk = [n]; cur = n
            for _ in range(walk_len - 1):
                nbrs = list(graph.neighbors(cur))
                if not nbrs: break
                cur = random.choice(nbrs)
                walk.append(cur)
            walks.append(walk)
    return walks

walks = random_walks(UG_train, num_walks=40, walk_len=20)

# 6) Co-occurrence from walks + boosts
nodes = sorted({n for w in walks for n in w} | set(G_train.nodes()))
idx = {n: i for i, n in enumerate(nodes)}
V = len(nodes); window = 5
cooc = np.zeros((V, V), dtype=np.float32)

# 6a) Walk-based window co-occurrence (symmetric)
for w in walks:
    ids = [idx[n] for n in w if n in idx]
    L = len(ids)
    for i, c in enumerate(ids):
        left = max(0, i - window); right = min(L, i + window + 1)
        for j in range(left, right):
            if j == i: continue
            cooc[c, ids[j]] += 1.0
            cooc[ids[j], c] += 1.0

# 6b) Direct TRAIN edges boost
EDGE_BOOST = 8.0
for (u, v) in train_edges:
    if u in idx and v in idx:
        ui, vi = idx[u], idx[v]
        cooc[ui, vi] += EDGE_BOOST
        cooc[vi, ui] += EDGE_BOOST

# 6c) 2-hop ancestor boost
HOP2_BOOST = 3.5
for b in G_train.nodes():
    parents = list(G_train.successors(b))    # b -> parent
    children = list(G_train.predecessors(b)) # child -> b
    for a in children:
        for c in parents:
            if a in idx and c in idx:
                ai, ci = idx[a], idx[c]
                cooc[ai, ci] += HOP2_BOOST
                cooc[ci, ai] += HOP2_BOOST

# 6d) k-hop neighborhood boosts (1..3 hops with decay)
KHOP_WEIGHTS = {1: 5.0, 2: 2.5, 3: 1.2}
for src in UG_train.nodes():
    if src not in idx: continue
    si = idx[src]
    lengths = nx.single_source_shortest_path_length(UG_train, src, cutoff=3)
    for tgt, dist in lengths.items():
        if dist == 0 or tgt not in idx: continue
        w = KHOP_WEIGHTS.get(dist)
        if w is None: continue
        ti = idx[tgt]
        cooc[si, ti] += w
        cooc[ti, si] += w

# 7) PPMI
sum_c = cooc.sum()
if sum_c == 0:
    raise RuntimeError("No co-occurrences recorded; check graph/walks.")
row_sum = cooc.sum(axis=1, keepdims=True) + 1e-12
col_sum = cooc.sum(axis=0, keepdims=True) + 1e-12
Pij = cooc / sum_c
Pi  = row_sum / sum_c
Pj  = col_sum / sum_c
PMI = np.log((Pij + 1e-12) / (Pi @ Pj + 1e-12))
PPMI = np.maximum(PMI, 0.0)

# 8) HYBRID VECTORS: [SVD(PPMI) | raw PPMI] with L2 normalization
dim = 64  # smaller SVD part; raw PPMI keeps detail
n_comp = min(dim, max(2, min(PPMI.shape) - 1))
X_svd = TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(PPMI)
Vmat = np.hstack([X_svd, PPMI])             # shape: (V, n_comp + V)
Vmat = normalize(Vmat, norm='l2', axis=1)   # row-normalize so cosine = dot product

def vec(node: str):
    i = idx.get(node, None)
    return np.zeros((Vmat.shape[1],), dtype=np.float32) if i is None else Vmat[i]

# 9) Validation: cosine on held-out subclass edges; percentiles; τ = 60th pct clamped
valid_cos = [float(cosine_similarity(vec(a).reshape(1,-1), vec(b).reshape(1,-1))[0,0]) for a,b in valid_edges]
mean_cos_valid = float(np.mean(valid_cos)) if valid_cos else float('nan')
pcts = np.percentile(valid_cos, [0, 25, 50, 60, 75, 90, 95, 99]) if valid_cos else [float('nan')]*8

if valid_cos:
    tau = float(np.percentile(valid_cos, 60))
    tau = round(min(0.95, max(0.40, tau)), 2)  # allow a bit lower floor; hybrid lifts scores
else:
    tau = 0.65

print(f"\nVALIDATION — mean_cos: {mean_cos_valid:.3f}")
print(f"  percentiles [0,25,50,60,75,90,95,99]: {[round(x,3) for x in pcts]}")
print(f"  → chosen τ: {tau:.2f}")

# 10) Test: positives vs negatives
def eval_edges(edge_list, tau_value):
    cos = [float(cosine_similarity(vec(a).reshape(1,-1), vec(b).reshape(1,-1))[0,0]) for a,b in edge_list]
    above = sum(c >= tau_value for c in cos)
    return {"mean_cos": float(np.mean(cos)) if cos else float('nan'),
            "above_tau": int(above), "total": len(edge_list)}

test_pos = eval_edges(test_edges, tau)

existing = set(edges)
neg_edges = []
tries = 0
all_nodes = list(nodes)
target = len(test_edges)
while len(neg_edges) < target and tries < 20 * target:
    a, b = random.choice(all_nodes), random.choice(all_nodes)
    if a != b and (a, b) not in existing and (b, a) not in existing:
        neg_edges.append((a, b))
    tries += 1

test_neg = eval_edges(neg_edges, tau)

print(f"\nTEST — positives: mean_cos={test_pos['mean_cos']:.3f}, above_tau={test_pos['above_tau']}/{test_pos['total']}")
print(f"TEST — negatives: mean_cos={test_neg['mean_cos']:.3f}, above_tau={test_neg['above_tau']}/{test_neg['total']}")

# 11) Nearest neighbors (labels)
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
def top_neighbors(n: str, k=5):
    i = idx.get(n, None)
    if i is None: return []
    v = vec(n).reshape(1, -1)
    sims = cos_sim(v, Vmat)[0]
    order = np.argsort(-sims)
    out = []
    for j in order:
        if nodes[j] == n: continue
        out.append((nodes[j], float(sims[j])))
        if len(out) >= k: break
    return out

print("\nNearest neighbors (cosine) — sanity check:")
for n in random.sample(list(classes), k=min(5, len(classes))):
    print("•", short_label(n))
    for m, c in top_neighbors(n, 5):
        print(f"   → {short_label(m)}  ({c:.3f})")

# 12) 2D viz (PCA on hybrid vectors)
sample_ids = np.random.choice(len(nodes), size=min(180, len(nodes)), replace=False)
coords = PCA(2, random_state=42).fit_transform(Vmat[sample_ids])
plt.figure(figsize=(7,7))
plt.scatter(coords[:,0], coords[:,1], s=14, alpha=0.75)
plt.title("Ontology Embedding Space (ArtifactOntology — Hybrid [SVD|PPMI], connectivity-preserving)")
plt.show()

# 13) Report
report = {
    "classes": len(classes),
    "train_edges": len(train_edges),
    "valid_edges": len(valid_edges),
    "test_edges": len(test_edges),
    "mean_cos_valid": round(mean_cos_valid, 3),
    "tau": tau,
    "test_pos_mean_cos": round(test_pos["mean_cos"], 3),
    "test_pos_above_tau": f"{test_pos['above_tau']}/{test_pos['total']}",
    "test_neg_mean_cos": round(test_neg["mean_cos"], 3),
    "test_neg_above_tau": f"{test_neg['above_tau']}/{test_neg['total']}"
}
print("\n=== Live Demo Report ===")
print(json.dumps(report, indent=2))
