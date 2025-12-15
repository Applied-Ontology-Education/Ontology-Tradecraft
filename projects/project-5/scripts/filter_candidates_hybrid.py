from __future__ import annotations

import mowl
import torch
from rdflib import OWL, Graph, SKOS

from common.llm_communication import extract_json
from filter.filter_functions import _load_metrics, _load_definitions, filter_candidates, load_candidates, \
    LLMConfig, _build_langchain_ollama, _write_accepted, _init_model_for_embeddings, \
    _load_hybrid_prompt_template, _build_user_prompt

mowl.init_jvm("6g")
import argparse
import json
import logging
import os
from typing import Dict, Iterable, Tuple

import dotenv
from util.logger_config import config
from common.settings import build_settings

logger = logging.getLogger(__name__)
from pathlib import Path

config(logger)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
GENERATED_ROOT = PROJECT_ROOT / "generated"
REPORTS_ROOT = PROJECT_ROOT / "reports"
CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints"
SRC_ROOT = PROJECT_ROOT / "src"
PROMPTS_ROOT = PROJECT_ROOT / "prompts"


def _parse_args(settings: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter candidate EL axioms using hybrid MOWL cosine + LLM plausibility")
    p.add_argument("--input", default=str(settings["reference_ontology"]), help="Source ontology TTL/OWL path")
    p.add_argument("--candidates", default=str(GENERATED_ROOT / "candidates_el.ttl"),
                   help="Input TTL with candidate axioms (rdfs:subClassOf)")
    p.add_argument("--out", default=str(GENERATED_ROOT / "accepted_el.ttl"), help="Output TTL path for accepted axioms")
    p.add_argument("--metrics", default=str(REPORTS_ROOT / "mowl_metrics.json"),
                   help="Path to training metrics JSON (for tau, hyperparams)")
    p.add_argument("--w-cos", type=float, default=0.7,
                   help="Weight for cosine in hybrid score (final = w*cos + (1-w)*plaus)")
    p.add_argument("--tau", type=float, default=None,
                   help="Override threshold tau; defaults to selected_tau from metrics or 0.7 if missing")
    p.add_argument("--limit", type=int, default=0, help="Optional max number of candidates to process (0 = all)")
    p.add_argument("--dry-run", action="store_true", help="Skip LLM calls; use cosine only for filtering")
    # LLM options (LangChain + Ollama backend)
    env_host = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    p.add_argument("--ollama-host", default=env_host, help="Ollama server base URL (used via LangChain)")
    p.add_argument("--ollama-model", default=str(settings.get("model_name", "llama3:instruct")),
                   help="Ollama model name to use for plausibility scoring (via LangChain)")
    p.add_argument("--temperature", type=float, default=float(settings.get("temperature", 0.0)), help="LLM temperature")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="Per-call timeout seconds for LLM requests (best-effort)")
    return p.parse_args()


def _cosine_scores(
    model,
    pairs: Iterable[Tuple[str, str]],
    restriction_map: Dict[Tuple[str, str], object] | None = None,
    source_graph: object | None = None,
) -> Dict[Tuple[str, str], float]:
    cos = torch.nn.functional.cosine_similarity
    emb_tuple = model.get_embeddings()
    ent_map = emb_tuple[0]
    # Try to obtain relation embeddings if available from the model
    rel_map = None
    try:
        rel_map = emb_tuple[1]
    except Exception:
        rel_map = None

    restriction_map = restriction_map or {}
    scores: Dict[Tuple[str, str], float] = {}
    for sub, sup in pairs:
        v_sub = ent_map.get(sub)
        v_sup_or_fill = ent_map.get(sup)

        sim: float
        # If this pair corresponds to a restriction (C ⊑ ∃R.F) and we have relation embeddings,
        # compose the score using (sub + R) vs filler.
        if (
            source_graph is not None
            and (sub, sup) in restriction_map
        ):
            try:
                bnode = restriction_map[(sub, sup)]
                # Extract onProperty for the restriction
                props = list(source_graph.objects(bnode, OWL.onProperty))  # type: ignore[attr-defined]
                prop_iri = str(props[0]) if props else None
            except Exception:
                prop_iri = None

            v_rel = None
            if rel_map is not None and prop_iri is not None:
                v_rel = rel_map.get(prop_iri)

            if v_sub is not None and v_sup_or_fill is not None and v_rel is not None:
                t_comp = torch.tensor(v_sub) + torch.tensor(v_rel)
                t_fill = torch.tensor(v_sup_or_fill)
                sim = float(cos(t_comp, t_fill, dim=0).item())
            else:
                # Fallback to standard class-class cosine (sub vs filler)
                if v_sub is None or v_sup_or_fill is None:
                    logger.info("Missing embedding for %s or %s; assigning cosine=0.0", sub, sup)
                    sim = 0.0
                else:
                    sim = float(cos(torch.tensor(v_sub), torch.tensor(v_sup_or_fill), dim=0).item())
        else:
            # Non-restriction: standard class-class cosine
            if v_sub is None or v_sup_or_fill is None:
                logger.info("Missing embedding for %s or %s; assigning cosine=0.0", sub, sup)
                sim = 0.0
            else:
                sim = float(cos(torch.tensor(v_sub), torch.tensor(v_sup_or_fill), dim=0).item())
        # Clip to [0, 1] for hybrid combination
        sim = max(0.0, min(1.0, sim))
        logger.info(f"{sub}, {sup}, {sim}")
        scores[(sub, sup)] = sim
    return scores


def _score_plausibility(
    pairs: Iterable[Tuple[str, str]],
    llm_cfg: LLMConfig,
    defs: Dict[str, Dict[str, str]],
    dry_run: bool = False,
    restriction_map: Dict[Tuple[str, str], object] | None = None,
    source_graph: object | None = None,
    reference_ontology: object | None = None,
) -> Dict[Tuple[str, str], float]:
    results: Dict[Tuple[str, str], float] = {}
    if dry_run:
        # Return neutral plausibility
        for p in pairs:
            results[p] = 0.5
        return results

    chat = _build_langchain_ollama(llm_cfg)
    if chat is None:
        for p in pairs:
            results[p] = 0.5
        return results

    try:
        from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
    except Exception as e:
        logger.warning("langchain-core messages not available (%s). Falling back to plausibility=0.5.", e)
        for p in pairs:
            results[p] = 0.5
        return results

    # Load prompt templates (System and User) from markdown
    system_tmpl, user_tmpl = _load_hybrid_prompt_template(PROMPTS_ROOT)

    restriction_map = restriction_map or {}
    ref_g = reference_ontology

    from rdflib import RDFS, URIRef, Literal

    def _fallback_label_from_iri(iri: str) -> str:
        # Use fragment or last path segment
        if '#' in iri:
            return iri.rsplit('#', 1)[-1]
        return iri.rstrip('/').rsplit('/', 1)[-1]

    def _first_literal(g: Graph, s: URIRef, p) -> str:
        try:
            for lit in g.objects(s, p):
                if isinstance(lit, Literal):
                    return str(lit)
        except Exception:
            pass
        return ""

    def _label_and_def(g: Graph | None, iri: str) -> tuple[str, str]:
        if not isinstance(g, Graph):
            return _fallback_label_from_iri(iri), ""
        s = URIRef(iri)
        label = _first_literal(g, s, RDFS.label)
        definition = _first_literal(g, s, SKOS.definition)
        if not label:
            label = _fallback_label_from_iri(iri)
        return label, definition

    for sub, sup in pairs:
        # Extract property IRI from the restriction (all candidates are restrictions per assumption)
        rel_iri: str | None = None
        try:
            if source_graph is not None and (sub, sup) in restriction_map:
                bnode = restriction_map[(sub, sup)]
                props = list(source_graph.objects(bnode, OWL.onProperty))  # type: ignore[attr-defined]
                rel_iri = str(props[0]) if props else None
        except Exception:
            rel_iri = None

        # Look up labels/definitions in the provided reference ontology
        sub_label, sub_def = _label_and_def(ref_g, sub)
        sup_label, sup_def = _label_and_def(ref_g, sup)
        prop_label, prop_def = _label_and_def(ref_g, rel_iri) if rel_iri else ("", "")

        # Fill the User template with robust fallback if template placeholders mismatch
        try:
            user_content = user_tmpl.format(
                sub_label=sub_label,
                sub_iri=sub,
                sub_definition=sub_def,
                super_iri=sup,
                super_definition=sup_def,
                sup_label=sup_label,
                prop_iri=(rel_iri or ""),
                prop_definition=prop_def,
                prop_label=prop_label,
            )
        except Exception as fmt_err:
            logger.warning("Prompt template formatting failed: %s. Falling back to basic prompt.", fmt_err)
            # Use the generic builder that doesn't rely on the markdown template
            user_content = _build_user_prompt(
                sub_iri=sub,
                sup_iri=sup,
                defs=defs,
                reference_context="",
                rel_iri=rel_iri,
            )
        try:
            # Build chat messages for LangChain
            messages = [
                SystemMessage(content=system_tmpl),
                HumanMessage(content=user_content),
            ]

            resp = chat.invoke(messages)
            # resp is an AIMessage; extract the content
            text = getattr(resp, "content", None)
            logger.info(text)
            if text is None:
                text = str(resp)
            obj = extract_json(text)
            val = float(obj.get("plausibility", 0.0))
            val = max(0.0, min(1.0, val))
        except Exception as e:
            logger.warning("LLM scoring failed for %s ⊑ %s (LangChain/Ollama): %s", sub, sup, e)
            val = 0.5
        results[(sub, sup)] = val
    return results


def main():
    # Read settings
    dotenv.load_dotenv()
    settings = build_settings(PROJECT_ROOT, DATA_ROOT)

    args = _parse_args(settings)
    candidates_path = Path(args.candidates)
    out_path = Path(args.out)
    metrics_path = Path(args.metrics)
    input_path = Path(args.input)

    logger.info("Reading input ontology: %s", input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input ontology not found: {input_path}")
    reference_ontology = Graph()
    reference_ontology.parse(input_path.as_posix(), format="turtle")

    # Load information from metrics report
    metrics = _load_metrics(metrics_path)
    tau = float(args.tau) if args.tau is not None else float(metrics.get("selected_tau") or 0.70)
    w_cos = float(args.w_cos)
    if not (0.0 <= w_cos <= 1.0):
        raise ValueError("--w-cos must be in [0,1]")

    # Load the candidates from candidates_el.ttl
    pairs, source_graph, restriction_map = load_candidates(args, candidates_path)

    # Create the cosine similarities
    logger.info("Initializing ELEmbeddings model for cosine scoring…")
    model, dataset = _init_model_for_embeddings(metrics, CHECKPOINTS_ROOT, SRC_ROOT)
    cos_scores = _cosine_scores(model, pairs, restriction_map=restriction_map, source_graph=source_graph)

    # Create the LLM plausibility scores
    llm_cfg = LLMConfig(
        host=args.ollama_host,
        model=args.ollama_model,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    defs = _load_definitions(DATA_ROOT / "definitions_enriched.csv")
    plaus_scores = _score_plausibility(
        pairs,
        llm_cfg,
        defs,
        dry_run=bool(args.dry_run),
        restriction_map=restriction_map,
        source_graph=source_graph,
        reference_ontology=reference_ontology,
    )

    # Filter candidates to accept
    accepted, hybrid_details = filter_candidates(pairs, cos_scores, plaus_scores, tau, w_cos)

    # Output accepted candidates (write full restrictions when applicable)
    n_written = _write_accepted(out_path, accepted, source_graph=source_graph, restriction_map=restriction_map)
    logger.info("Accepted %d/%d candidates at tau=%.2f with w_cos=%.2f. Output: %s", n_written, len(pairs), tau, w_cos,
                out_path)

    # Write a small report next to the TTL for traceability
    try:
        report = {
            "tau": tau,
            "w_cos": w_cos,
            "num_candidates": len(pairs),
            "num_accepted": n_written,
            "ollama_model": args.ollama_model,
            "ollama_host": args.ollama_host,
            "dry_run": bool(args.dry_run),
            "details": hybrid_details[:200],  # cap for brevity
        }
        with (out_path.with_suffix(out_path.suffix + ".json")).open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    except Exception as e:
        logger.debug("Could not write hybrid report JSON: %s", e)


if __name__ == "__main__":
    main()
