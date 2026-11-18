"""Project 6 Step 5B: mOWL ELEmbeddings cosine diagnostics and optional xAI plausibility.

Usage:
  python scripts/prepare_rail_mowl.py
  python scripts/evaluate_rail_mowl.py
  python scripts/evaluate_rail_mowl.py --with-llm --reuse-vectors

A rail candidate is never 'approved' by cosine or an LLM. Step 5A holds out
20% of independently supported rail subclass edges. Only those edges receive
an out-of-sample cosine score; other candidate edges remain in training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL, SKOS

from prepare_rail_mowl import (plan, file_hash, atomic_json, atomic_text,
                               RAIL_ROLE_PAIR, load_ontology, pairs)

VERSION="project6-elembeddings-hybrid-80-20-v2"
DEFAULT_MODEL="grok-4.7"


def cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float | None:
    a=np.asarray(vec_a,dtype=np.float64);b=np.asarray(vec_b,dtype=np.float64)
    if a.ndim != 1 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    if norm <= 1e-12: return None
    return round(float(np.clip(np.dot(a,b)/norm,-1.0,1.0)),6)


def score_holdouts(index: dict[str,int],vectors: np.ndarray, report: dict) -> list[dict]:
    rows=[]
    held = {(x["child"], x["parent"]) for x in report["rail_candidate_holdouts"]}
    for row in report["holdout_pairs"]:
        child,parent=row["child"],row["parent"]
        if child not in index or parent not in index:
            status="MISSING_EMBEDDING"; score=None
        else:
            score=cosine(vectors[index[child]],vectors[index[parent]])
            status="HELD_OUT_COSINE" if score is not None else "INVALID_EMBEDDING"
        rows.append({"child":child,"parent":parent,"cosine":score,
                     "status":status,"scope":"RAIL_NEW_DOMAIN" if (child,parent) in held else "SOURCE_IMPORT_REFERENCE"})
    return rows


def source_label(g: Graph, iri: str) -> str:
    labels=[str(o) for o in g.objects(URIRef(iri),RDFS.label)]
    return sorted(labels)[0] if labels else iri.rsplit("/",1)[-1].rsplit("#",1)[-1]


def build_llm_request(root: Path, split: dict, model: str) -> dict:
    refined=load_ontology(root/"generated/rail_axioms_refined.ttl")
    source=load_ontology(root/"src/module.ttl")
    proposal=json.loads((root/"generated/rail_pattern_proposal.json").read_text(encoding="utf-8"))
    g=Graph()
    for gr in (refined,source):
        for t in gr:g.add(t)
    requests=[]
    for i, pair in enumerate(split["candidate_pairs"]):
        child,parent=pair["child"],pair["parent"]
        defs=[str(x) for x in refined.objects(URIRef(child),SKOS.definition)]
        requests.append({"id":f"edge-{i:02d}","child_iri":child,"parent_iri":parent,
                         "child_label":source_label(g,child),"parent_label":source_label(g,parent),
                         "child_definition":sorted(defs)[0] if defs else "",
                         "cosine_status":pair["evaluation"]})
    return {"version":VERSION,"model":model,"candidates":requests,
            "scope":"Assess subclass plausibility only. Do not infer real-world individuals, a person's intentions, or threat status.",
            "constraint":"Ticket-as-bearer versus digital information remains a blocking human design review; source must remain unchanged."}


def validate_llm_reply(reply: dict, request:dict) -> dict[str,dict]:
    if set(reply) != {"ratings"} or not isinstance(reply["ratings"],list):
        raise ValueError("Unexpected xAI response shape")
    wanted={r["id"] for r in request["candidates"]}
    found={}
    for row in reply["ratings"]:
        if not isinstance(row,dict) or set(row)!={"id","plausibility","rationale","concern"}:
            raise ValueError("Malformed LLM rating row")
        if row["id"] in found or row["id"] not in wanted:
            raise ValueError("Duplicate/unexpected LLM rating ID")
        score=row["plausibility"]
        if (not isinstance(score,(int,float)) or isinstance(score,bool) or
            not np.isfinite(score) or not (0<=score<=1)):
            raise ValueError("Plausibility must be numeric in [0,1]")
        if not isinstance(row["rationale"],str) or len(row["rationale"].strip())<12:
            raise ValueError("Missing LLM rationale")
        if not isinstance(row["concern"],str):
            raise ValueError("Malformed concern")
        found[row["id"]]=row
    if set(found)!=wanted:
        raise ValueError("LLM returned missing ratings")
    return found


def call_llm(request:dict) -> dict:
    api_key=os.environ.get("XAI_API_KEY","").strip()
    if not api_key:
        raise ValueError("XAI_API_KEY missing; set it in this PowerShell session (do not share it)")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install openai in the SAME Python environment") from e
    schema={"type":"object","additionalProperties":False,"required":["ratings"],"properties":{
        "ratings":{"type":"array","items":{"type":"object","additionalProperties":False,
          "required":["id","plausibility","rationale","concern"],"properties":{
             "id":{"type":"string"},"plausibility":{"type":"number"},
             "rationale":{"type":"string"},"concern":{"type":"string"}}}}}}
    client=OpenAI(api_key=api_key,base_url="https://api.x.ai/v1",timeout=900.0,max_retries=0)
    response=client.chat.completions.create(
        model=request["model"],
        messages=[{"role":"system","content":(
            "You are reviewing OWL subclass axioms for an academic BFO/IAO ontology. "
            "Score semantic plausibility of each specific subclass relationship 0..1. "
            "Provide a grounded concise rationale and any caution. "
            "These are proposed TBox axioms, not known facts. "
            "Do not add persons, inferred individuals, or threat classifications. "
            "A score is a heuristic and does not prove consistency or accept any axiom. "
            "Return exactly one rating for each supplied id, with no other fields.")},
            {"role":"user","content":json.dumps(request,sort_keys=True)}],
        response_format={"type":"json_schema","json_schema":{
            "name":"rail_hybrid_review","strict":True,"schema":schema}})
    content=response.choices[0].message.content
    if not content:raise RuntimeError("xAI returned no response content")
    return json.loads(content)


def build_hybrid_report(split:dict, heldout_scores:list[dict], llm_ratings:dict[str,dict]|None) -> dict:
    held={(r["child"],r["parent"]):r for r in heldout_scores}
    rows=[]
    for i,edge in enumerate(split["candidate_pairs"]):
        p=(edge["child"],edge["parent"])
        validation=held.get(p)
        independently_scored = (edge["evaluation"]=="HELD_OUT" and validation is not None
                                and validation["status"]=="HELD_OUT_COSINE")
        llm=llm_ratings.get(f"edge-{i:02d}") if llm_ratings is not None else None
        cosine_score=validation["cosine"] if independently_scored else None
        hybrid_score=(round(0.70*max(0,cosine_score)+0.30*llm["plausibility"],6)
                      if cosine_score is not None and llm is not None else None)
        rows.append({"child":p[0],"parent":p[1],"cosine":cosine_score,
            "cosine_evidence":"HELD_OUT_INDEPENDENT" if cosine_score is not None else "INSUFFICIENT_INDEPENDENT_COSINE_EVIDENCE",
            "llm_plausibility":llm["plausibility"] if llm else None,
            "llm_rationale":llm["rationale"] if llm else "",
            "llm_concern":llm["concern"] if llm else "",
            "hybrid_score":hybrid_score,
            "review_priority": "STRONGER_EVIDENCE_FOR_HUMAN_REVIEW" if hybrid_score is not None and hybrid_score>=.60 and llm["plausibility"]>=.50 else
                               "HUMAN_REVIEW_REQUIRED",
            "approved":False})
    return {"version":VERSION,"status":"HYBRID_REVIEW_ONLY" if llm_ratings else "COSINE_ONLY_LLM_NOT_RUN",
      "method":"70% held-out cosine + 30% LLM only when held-out evidence exists; no score fabricated for cold-start/in-training candidate links",
      "weights":{"cosine":.70,"llm":.30},
      "llm_ratings_executed":llm_ratings is not None,"merges_performed":0,
      "candidate_relationships":rows,
      "hybrid_scored_pairs":sum(r["hybrid_score"] is not None for r in rows),
      "unscored_independent_cosine":sum(r["cosine"] is None for r in rows),
      "all_human_review_required":True,
      "note":"Cosine is not logical validity. Separate OWL reasoner and ticket-information scope review remain mandatory."}


def train_mowl(train_path: Path, *, epochs:int, dim:int, seed:int):
    try:
        import torch
        import mowl
    except ImportError as exc:
        raise RuntimeError("mOWL and torch are required. Use your working Python 3.12 mOWL environment, then run python -c 'import mowl, torch'") from exc
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Force CPU: deterministic-ish across machines and avoids silently picking GPU.
    mowl.init_jvm("2g")
    from mowl.datasets import Dataset, PathDataset
    from mowl.models import ELEmbeddings

# Load the ontology using OWLAPI, then create an in-memory
# dataset to avoid the Windows normalized-cache URI problem.
    loaded_dataset = PathDataset(str(train_path.resolve()))
    ds = Dataset(loaded_dataset.ontology)

    model=ELEmbeddings(ds,embed_dim=dim,margin=.1,reg_norm=1.0,
                       learning_rate=.001,batch_size=128,model_filepath=None,device="cpu")
    model.train(epochs=epochs,validate_every=epochs)
    index={str(iri):int(i) for iri,i in model.class_index_dict.items()}
    vectors=model.module.class_embed.weight.detach().cpu().numpy().copy()
    if len(index)!=vectors.shape[0] or not np.isfinite(vectors).all():
        raise RuntimeError("Bad mOWL embedding shape or nonfinite vector detected")
    return index,vectors,{"mowl_version":getattr(mowl,"__version__","unknown"),
                          "torch_version":torch.__version__}


def save_vectors(root:Path,index:dict,vectors:np.ndarray,manifest:dict) -> None:
    vec_path=root/"generated/mowl_rail_class_vectors.npz"
    vec_path.parent.mkdir(parents=True,exist_ok=True)
    ordered=sorted(index)
    # Strings rather than object arrays, so allow_pickle=False is safe.
    tmp=vec_path.with_name(vec_path.name+".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,iris=np.asarray(ordered,dtype="U"),
                            vectors=np.asarray([vectors[index[x]] for x in ordered],dtype=np.float32))
    tmp.replace(vec_path)
    manifest["vectors_sha256"]=file_hash(vec_path)
    atomic_json(root/"reports/step5_model_manifest.json",manifest)


def load_vectors(root:Path,manifest:dict) -> tuple[dict,np.ndarray,dict]:
    current=json.loads((root/"reports/step5_model_manifest.json").read_text(encoding="utf-8"))
    vec_path=root/"generated/mowl_rail_class_vectors.npz"
    if (any(current.get(k)!=v for k,v in manifest.items()) or
        file_hash(vec_path)!=current.get("vectors_sha256")):
        raise ValueError("Cached mOWL vectors are stale or mutated; rerun without --reuse-vectors")
    with np.load(vec_path,allow_pickle=False) as loaded:
        iris=list(loaded["iris"]);vectors=loaded["vectors"].copy()
    if len(set(iris))!=len(iris) or len(iris)!=len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("Invalid saved mOWL vectors")
    return {iri:i for i,iri in enumerate(iris)},vectors,current


def evaluate(root:Path,*,epochs:int=200,dim:int=50,seed:int=42,
             model:str=DEFAULT_MODEL,with_llm:bool=False,reuse_vectors:bool=False,
             prepare_only:bool=False, llm_function=None, train_function=None) -> dict:
    root=root.resolve()
    if epochs<1 or dim<2:
        raise ValueError("epochs must be >=1 and embedding dimension >=2")
    split=plan(root,seed=seed)
    if prepare_only:
        return {"status":"PASS_SPLIT_READY_FOR_MOWL","source_unchanged":True,
                "source_import_holdout_pairs":split["source_import_holdout_pairs"]}
    manifest={"version":VERSION,"train_sha256":split["train_sha256"],
              "valid_sha256":split["valid_sha256"],"source_sha256":split["source_sha256"],
              "refined_sha256":split["refined_sha256"],"seed":seed,"epochs":epochs,
              "embed_dim":dim,"device":"cpu","model_type":"mOWL ELEmbeddings", "model_path":"generated/mowl_rail_train.ttl"}
    if reuse_vectors:
        index,vectors,model_metadata=load_vectors(root,manifest)
    else:
        train_function=train_function or train_mowl
        index,vectors,details=train_function(root/"generated/mowl_rail_train.ttl",epochs=epochs,dim=dim,seed=seed)
        save_vectors(root,index,vectors,{**manifest,**details})
        model_metadata={**manifest,**details}
    scores=score_holdouts(index,vectors,split)
    rail_scores = [r for r in scores if r["scope"] == "RAIL_NEW_DOMAIN"]
    if (len(rail_scores) != split["rail_candidate_holdout_pairs"] or
        any(r["status"] != "HELD_OUT_COSINE" for r in rail_scores)):
        raise ValueError("Some independently held-out rail relationships lack valid embeddings; fail rather than fabricate a mean")
    rail_mean = round(float(np.mean([r["cosine"] for r in rail_scores])), 6)
    # Preserve a role-specific field solely for comparing this run with v1.
    role_score = next((r["cosine"] for r in rail_scores
                       if (r["child"], r["parent"]) == RAIL_ROLE_PAIR), None)
    threshold = .70
    meets_threshold = rail_mean >= threshold
    proposal=json.loads((root/"generated/rail_pattern_proposal.json").read_text(encoding="utf-8"))
    original_hash=split["source_sha256"]
    llm_ratings=None
    if with_llm:
        request=build_llm_request(root,split,model)
        atomic_json(root/"generated/step5_llm_review_request.json",request)
        llm_function=llm_function or call_llm
        reply=llm_function(request)
        llm_ratings=validate_llm_reply(reply,request)
        atomic_json(root/"generated/step5_llm_review_raw.json",reply)
    hybrid=build_hybrid_report(split,scores,llm_ratings)
    hybrid["source_sha256"]=original_hash
    hybrid["refined_sha256"]=split["refined_sha256"]
    hybrid["llm_model"]=model if with_llm else None
    heldref=[r["cosine"] for r in scores if r["scope"]=="SOURCE_IMPORT_REFERENCE" and r["cosine"] is not None]
    cosine_report={"version":VERSION,"status":"MOWL_TRAINED_HELDOUT_EVALUATED",
                   "train_sha256":split["train_sha256"],"valid_sha256":split["valid_sha256"],
                   "model":model_metadata,"heldout_source_import_pairs":len(heldref),
                   "mean_heldout_source_import_cosine":round(float(np.mean(heldref)),6) if heldref else None,
                   "heldout_rail_cosine":role_score,
                   "heldout_rail_pairs":len(rail_scores),
                   "mean_heldout_rail_cosine":rail_mean,
                   "cosine_threshold":threshold,
                   "cosine_requirement_met":meets_threshold,
                   "heldout_rail_pair_scores":rail_scores,
                   "scores":scores,
                   "warning":"Train set contains all other rail candidate superclass axioms. Do not use their in-sample scores as independent validation."}
    if (file_hash(root/"src/module.ttl")!=original_hash or
        file_hash(root/"generated/rail_axioms_refined.ttl")!=split["refined_sha256"]):
        raise RuntimeError("Ontology changed during Step 5; not saving scores")
    atomic_json(root/"reports/step5_cosine_scores.json",cosine_report)
    atomic_json(root/"reports/step5_hybrid_review.json",hybrid)
    summary={"version":VERSION,"status":"PASS_HYBRID_REVIEW_ONLY" if with_llm else "PASS_COSINE_ONLY",
             "mowl_training_completed":True,
              "cosine_heldout_rail_pairs":len(rail_scores),
              "mean_heldout_rail_cosine":rail_mean,
              "cosine_threshold":threshold,
              "cosine_requirement_met":meets_threshold,
             "cosine_source_reference_pairs":len(heldref),
             "hybrid_scored_pairs":hybrid["hybrid_scored_pairs"],
             "llm_executed":with_llm,"source_unchanged":True,"source_sha256":original_hash,
             "merged_into_source":False,"new_named_individuals":0,
             "semantic_approval":False,"owl_reasoner_executed":False,
             "next":"Human scope review + OWL reasoner + controlled merge; no candidate automatically accepted",
              "cosine_requirement_note": "Successful training does not imply 0.70 cosine mean; check cosine_requirement_met. The report is diagnostic and does not by itself block the runner."}
    atomic_json(root/"reports/step5_summary.json",summary)
    return summary


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--epochs",type=int,default=200)
    p.add_argument("--dim",type=int,default=50)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--model",default=os.getenv("XAI_MODEL",DEFAULT_MODEL))
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--with-llm",action="store_true")
    p.add_argument("--reuse-vectors",action="store_true")
    a=p.parse_args()
    if a.prepare_only and (a.with_llm or a.reuse_vectors):
        p.error("--prepare-only cannot be used with --with-llm or --reuse-vectors")
    try:
        result=evaluate(a.project_root,epochs=a.epochs,dim=a.dim,seed=a.seed,model=a.model,
                        with_llm=a.with_llm,reuse_vectors=a.reuse_vectors,
                        prepare_only=a.prepare_only)
    except Exception as e:
        print("STEP 5 — FAIL: " + str(e),file=sys.stderr)
        return 1
    print("STEP 5 — " + result["status"])
    print(json.dumps(result,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
