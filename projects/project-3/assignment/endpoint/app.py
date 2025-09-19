# endpoint/app.py
import os, sys, logging
from flask import Flask, request, jsonify, Response
from pathlib import Path
from rdflib import Graph
from rdflib.plugins.sparql.processor import SPARQLResult

# ---------- Loud logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("endpoint")
log.info("[BOOT] starting endpoint/app.py")

# ---------- Config: set real paths here ----------
ONTOLOGY_FILES = {
    "bfo":  "/Users/john/Repos/Ontology-Tradecraft/projects/project-3/assignment/src/bfo-core.ttl",   # BFO
    "ies":  "/Users/john/Repos/Ontology-Tradecraft/projects/project-3/assignment/src/ies.ttl",        # IES
    "qudt": "/Users/john/Repos/Ontology-Tradecraft/projects/project-3/assignment/src/qudt.ttl",       # QUDT  (update if different)
    "ccom": "/Users/john/Repos/Ontology-Tradecraft/projects/project-3/assignment/src/ccom.ttl",       # CCOM  (update if different)
    "ccot": "/Users/john/Repos/Ontology-Tradecraft/projects/project-3/assignment/src/ccot.ttl",      # CCO Time module
    "to":   "/Users/john/Repos/Ontology-Tradecraft/projects/project-3/assignment/src/time.ttl",      # W3C Time Ontology
}

# ---------- App ----------
app = Flask(__name__)
log.info("[BOOT] Flask app created")

# We load graphs lazily on first use so startup always logs.
GRAPHS: dict[str, Graph] = {}
LOADED: set[str] = set()

def _load_graph_if_needed(name: str) -> Graph:
    if name in LOADED:
        return GRAPHS[name]

    path = Path(ONTOLOGY_FILES[name])
    g = Graph()
    if not path.exists():
        log.error(f"[ERROR] File not found for '{name}': {path}")
        GRAPHS[name] = g
        LOADED.add(name)
        return g

    try:
        g.parse(path.as_posix())  # let rdflib auto-detect format
        log.info(f"[OK] Loaded {name}: {len(g)} triples")
    except Exception as e:
        log.exception(f"[FAIL] Could not parse {path}: {e}")

    GRAPHS[name] = g
    LOADED.add(name)
    return g

@app.get("/stats")
def stats():
    out = {}
    for name, path in ONTOLOGY_FILES.items():
        g = _load_graph_if_needed(name)
        out[name] = {
            "path": str(path),
            "triples": len(g)
        }
    return jsonify(out)

@app.get("/ping")
def ping():
    return jsonify({
        "status": "ok",
        "datasets": list(ONTOLOGY_FILES.keys()),
        "loaded": sorted(list(LOADED)),
    })

@app.route("/sparql/<name>", methods=["GET", "POST"])
def sparql_query(name: str):
    # 1) ensure graph exists (lazy load)
    try:
        g = _load_graph_if_needed(name)
    except KeyError as ke:
        return jsonify({"error": str(ke)}), 404

    # 2) get query string from GET or POST
    q = ""
    if request.method == "GET":
        q = request.args.get("query", "")
    else:  # POST
        ctype = (request.content_type or "").split(";", 1)[0].strip().lower()
        if ctype == "application/sparql-query":
            q = request.data.decode("utf-8", errors="replace")
        elif ctype in ("application/x-www-form-urlencoded", "multipart/form-data"):
            q = request.form.get("query", "")
        else:
            q = request.args.get("query", "")

    if not q.strip():
        return jsonify({"error": "Missing SPARQL query"}), 400

    log.info(f"[QRY] dataset={name} method={request.method} bytes={len(q)}")
    try:
        res: SPARQLResult = g.query(q)
        json_obj = res.serialize(format="json")
        return Response(json_obj, mimetype="application/sparql-results+json")
    except Exception as e:
        log.exception(f"[SPARQL ERROR] dataset={name}: {e}")
        return jsonify({"error": f"SPARQL error: {e}"}), 400

if __name__ == "__main__":
    log.info("[RUN] starting server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
