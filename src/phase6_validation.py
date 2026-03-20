"""
Fase 6 — Validación del pipeline KGE + LLM.

Para cada muestra del qa_corpus.json:
  1. Se extrae la pregunta limpia (sin opciones a/b/c/d)
  2. Se construye el contexto KGE de la incidencia (subgrafo verbalizado)
  3. Se le hace la pregunta al LLM con ese contexto
  4. Se compara la respuesta generada con el campo "answer" del corpus

Métricas:
  - Token F1         : solapamiento de tokens (primera línea de la predicción)
  - BERTScore F1     : similitud semántica (bert-base-multilingual-cased)
  - Exact Match (EM) : la respuesta generada contiene exactamente el answer
  - Chain Accuracy   : fracción de cadenas donde TODOS los pasos son correctos
  - Hit@k (PyKEEN)   : métricas de ranking del modelo KGE sobre el test set

Salida:
  out/evaluation/results.json         (métricas globales)
  out/evaluation/predictions.jsonl    (detalle por muestra — 1hop + chains)

Uso:
  python src/phase6_validation.py [--n-samples 200] [--n-chains 100] [--model mistralai/Mistral-7B-Instruct-v0.2]
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_question(raw_question: str) -> str:
    """Elimina las opciones múltiples (\\n  a) ...) del enunciado."""
    return raw_question.split("\n")[0].strip()


def exact_match(prediction: str, reference: str) -> bool:
    """True si el answer exacto aparece en la predicción (case-insensitive)."""
    return reference.lower() in prediction.lower()


def token_f1(prediction: str, reference: str) -> float:
    """F1 sobre la primera línea de la predicción (el identificador extraído)."""
    first_line  = prediction.split("\n")[0].strip()
    pred_tokens = set(first_line.lower().split())
    ref_tokens  = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_bertscore(generated: list[str], references: list[str]) -> dict:
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(generated, references, lang="es", verbose=False)
        return {
            "precision": round(P.mean().item(), 4),
            "recall":    round(R.mean().item(), 4),
            "f1":        round(F1.mean().item(), 4),
        }
    except ImportError:
        print("  [Aviso] bert-score no instalado. Omitiendo BERTScore.")
        return {}


def load_pykeen_hit_metrics() -> dict:
    results_file = cfg.MODELS_DIR / "results.json"
    if not results_file.exists():
        return {}
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    realistic = data.get("metrics", {}).get("both", {}).get("realistic", {})
    out = {f"hit@{k}": realistic.get(f"hits_at_{k}") for k in cfg.HIT_K_VALUES}
    out["mrr"] = realistic.get("mean_reciprocal_rank")
    return {k: v for k, v in out.items() if v is not None}


# ---------------------------------------------------------------------------
# Contexto verbalizado para una incidencia
# ---------------------------------------------------------------------------

def _get_context(inc_id: str, incidents_map: dict, similarity_index, implicit_preds) -> list[str]:
    from phase4_llm_inference import get_verbalized_sentences, verbalize_props
    sentences = get_verbalized_sentences(inc_id)
    if not sentences:
        props = incidents_map.get(inc_id, {})
        if props:
            sentences = verbalize_props(inc_id, props)
    if not sentences:
        sentences = [f"La incidencia {inc_id} está registrada en el sistema."]
    if similarity_index is not None:
        try:
            from phase5_config_subgraph import build_session_subgraph, verbalize_session_subgraph
            sg = build_session_subgraph(inc_id, incidents_map, similarity_index, implicit_preds)
            sentences = verbalize_session_subgraph(sg)
        except Exception:
            pass
    return sentences


# ---------------------------------------------------------------------------
# Evaluación 1-hop
# ---------------------------------------------------------------------------

def run_1hop_evaluation(
    llm,
    incidents_map: dict,
    similarity_index,
    implicit_preds: Optional[dict],
    n_samples: int = cfg.EVAL_SAMPLE_N,
    seed: int      = cfg.RANDOM_SEED,
) -> tuple[dict, list[dict]]:
    """Evalúa preguntas 1-hop (excluye hf_paraphrase)."""
    with open(cfg.QA_CORPUS, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    pool = [
        item for item in corpus.get("1hop", [])
        if "hf_paraphrase" not in item.get("type", "")
    ]
    rng     = random.Random(seed)
    samples = rng.sample(pool, min(n_samples, len(pool)))
    print(f"  {len(samples)} preguntas 1-hop seleccionadas (sin hf_paraphrase, pool={len(pool)}).")

    log, generated, references = [], [], []

    for i, item in enumerate(samples):
        inc_id    = item.get("context_inc", "")
        question  = clean_question(item.get("question", ""))
        reference = item.get("answer", "")
        q_type    = item.get("type", "")

        sentences  = _get_context(inc_id, incidents_map, similarity_index, implicit_preds)
        prediction = llm.answer(sentences, question, do_extract=True)

        em  = exact_match(prediction, reference)
        tf1 = round(token_f1(prediction, reference), 4)
        generated.append(prediction)
        references.append(reference)

        log.append({
            "split": "1hop", "id": i + 1, "type": q_type,
            "incident": inc_id, "question": question,
            "expected": reference, "predicted": prediction,
            "exact_match": em, "token_f1": tf1,
        })
        status = "✓" if em else "✗"
        print(f"  [{i+1:>4}/{len(samples)}] {status}  F1={tf1:.2f}  "
              f"esperado={reference!r}  obtenido={prediction!r}")

    total    = len(log)
    em_score = sum(e["exact_match"] for e in log) / total
    mean_tf1 = sum(e["token_f1"]    for e in log) / total

    by_type: dict[str, dict] = {}
    for e in log:
        t = e["type"]
        by_type.setdefault(t, {"n": 0, "em": 0, "tf1_sum": 0.0})
        by_type[t]["n"]       += 1
        by_type[t]["em"]      += int(e["exact_match"])
        by_type[t]["tf1_sum"] += e["token_f1"]

    metrics = {
        "n_samples":      total,
        "exact_match":    round(em_score,  4),
        "mean_token_f1":  round(mean_tf1,  4),
        "bertscore":      evaluate_bertscore(generated, references),
        "by_type":        {
            t: {
                "n":           v["n"],
                "exact_match": round(v["em"] / v["n"], 4),
                "token_f1":    round(v["tf1_sum"] / v["n"], 4),
            }
            for t, v in sorted(by_type.items())
        },
    }
    return metrics, log


# ---------------------------------------------------------------------------
# Evaluación chains
# ---------------------------------------------------------------------------

def run_chain_evaluation(
    llm,
    incidents_map: dict,
    similarity_index,
    implicit_preds: Optional[dict],
    n_chains: int = cfg.EVAL_SAMPLE_N,
    seed: int     = cfg.RANDOM_SEED,
) -> tuple[dict, list[dict]]:
    """
    Evalúa cadenas multi-hop.
    - Cada paso se evalúa de forma independiente (el enunciado ya incluye
      la respuesta intermedia del paso anterior).
    - Chain accuracy = fracción de cadenas en que TODOS los pasos son correctos.
    """
    with open(cfg.QA_CORPUS, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    pool    = corpus.get("chains", [])
    rng     = random.Random(seed)
    chains  = rng.sample(pool, min(n_chains, len(pool)))
    n_steps = sum(len(c["steps"]) for c in chains)
    print(f"  {len(chains)} cadenas seleccionadas → {n_steps} pasos en total (pool={len(pool)}).")

    log, generated, references = [], [], []
    chain_correct: list[bool] = []
    global_idx = 0

    for chain in chains:
        inc_id     = chain.get("context_inc", "")
        chain_type = chain.get("chain_type", "")
        chain_id   = chain.get("chain_id", "")
        sentences  = _get_context(inc_id, incidents_map, similarity_index, implicit_preds)

        all_steps_ok = True
        for step in chain["steps"]:
            global_idx += 1
            question  = clean_question(step.get("question", ""))
            reference = step.get("answer", "")
            step_num  = step.get("step", "?")

            prediction = llm.answer(sentences, question, do_extract=True)

            em  = exact_match(prediction, reference)
            tf1 = round(token_f1(prediction, reference), 4)
            if not em:
                all_steps_ok = False

            generated.append(prediction)
            references.append(reference)

            log.append({
                "split": "chain", "id": global_idx,
                "chain_id": chain_id, "chain_type": chain_type,
                "step": step_num, "incident": inc_id, "question": question,
                "expected": reference, "predicted": prediction,
                "exact_match": em, "token_f1": tf1,
            })
            status = "✓" if em else "✗"
            print(f"  [{global_idx:>4}/{n_steps}] {status}  "
                  f"chain={chain_id} step={step_num}  "
                  f"F1={tf1:.2f}  esperado={reference!r}  obtenido={prediction!r}")

        chain_correct.append(all_steps_ok)

    total         = len(log)
    em_score      = sum(e["exact_match"] for e in log) / total
    mean_tf1      = sum(e["token_f1"]    for e in log) / total
    chain_acc     = sum(chain_correct) / len(chain_correct) if chain_correct else 0.0

    by_type: dict[str, dict] = {}
    for e in log:
        t = e["chain_type"]
        by_type.setdefault(t, {"n": 0, "em": 0, "tf1_sum": 0.0})
        by_type[t]["n"]       += 1
        by_type[t]["em"]      += int(e["exact_match"])
        by_type[t]["tf1_sum"] += e["token_f1"]

    metrics = {
        "n_chains":        len(chains),
        "n_steps":         total,
        "chain_accuracy":  round(chain_acc, 4),
        "step_exact_match": round(em_score, 4),
        "mean_token_f1":   round(mean_tf1,  4),
        "bertscore":       evaluate_bertscore(generated, references),
        "by_chain_type":   {
            t: {
                "n":           v["n"],
                "exact_match": round(v["em"] / v["n"], 4),
                "token_f1":    round(v["tf1_sum"] / v["n"], 4),
            }
            for t, v in sorted(by_type.items())
        },
    }
    return metrics, log


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(
    n_samples:  int = cfg.EVAL_SAMPLE_N,
    n_chains:   int = cfg.EVAL_SAMPLE_N,
    model_name: str = cfg.DEFAULT_MODEL,
    device:     str = cfg.DEVICE,
) -> dict:
    print("=" * 60)
    print("FASE 6 — Validación del pipeline KGE + LLM")
    print("=" * 60)

    print("[1/4] Cargando grafo de incidencias ...")
    import generate_corpus as gc
    g       = gc.load_graph(cfg.TTL_FILE)
    inc_map = gc.build_incident_map(g)

    print("[2/4] Cargando embeddings e índice de similitud ...")
    from phase5_config_subgraph import get_similarity_index, load_implicit_predictions
    sim_index  = get_similarity_index()
    impl_preds = load_implicit_predictions()
    if sim_index is None:
        print("  [Aviso] Embeddings no encontrados — se usará solo contexto directo.")

    print(f"[3/4] Cargando LLM: {model_name} (device={device}) ...")
    from phase4_llm_inference import KGEAugmentedLLM
    llm = KGEAugmentedLLM(model_name=model_name, device=device)

    if not cfg.QA_CORPUS.exists():
        raise FileNotFoundError(
            f"Corpus no encontrado: {cfg.QA_CORPUS}\n"
            "Ejecuta primero:  python src/generate_corpus.py"
        )

    # ── 1-hop ────────────────────────────────────────────────────────────
    print(f"\n[4/4a] Evaluando {n_samples} preguntas 1-hop ...\n")
    metrics_1hop, log_1hop = run_1hop_evaluation(
        llm, inc_map, sim_index, impl_preds, n_samples=n_samples
    )

    # ── Chains ───────────────────────────────────────────────────────────
    print(f"\n[4/4b] Evaluando {n_chains} cadenas multi-hop ...\n")
    metrics_chains, log_chains = run_chain_evaluation(
        llm, inc_map, sim_index, impl_preds, n_chains=n_chains
    )

    # ── Guardar ──────────────────────────────────────────────────────────
    cfg.EVAL_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "1hop":   metrics_1hop,
        "chains": metrics_chains,
        "hit_at_k_pykeen": load_pykeen_hit_metrics(),
    }
    with open(cfg.EVAL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    pred_file = cfg.EVAL_DIR / "predictions.jsonl"
    with open(pred_file, "w", encoding="utf-8") as f:
        for entry in log_1hop + log_chains:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Resumen ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTADOS — 1-hop")
    print("=" * 60)
    m = metrics_1hop
    n = m["n_samples"]
    print(f"  Muestras    : {n}")
    print(f"  Exact Match : {m['exact_match']:.4f}  ({int(m['exact_match']*n)}/{n})")
    print(f"  Token F1    : {m['mean_token_f1']:.4f}")
    if m["bertscore"]:
        bs = m["bertscore"]
        print(f"  BERTScore F1: {bs['f1']:.4f}  (P={bs['precision']:.4f}, R={bs['recall']:.4f})")
    print("\n  Desglose por tipo:")
    for t, v in m["by_type"].items():
        print(f"    {t:<40}  n={v['n']:>4}  EM={v['exact_match']:.3f}  F1={v['token_f1']:.3f}")

    print("\n" + "=" * 60)
    print("RESULTADOS — Chains (multi-hop)")
    print("=" * 60)
    mc = metrics_chains
    print(f"  Cadenas     : {mc['n_chains']}")
    print(f"  Pasos total : {mc['n_steps']}")
    print(f"  Chain Acc.  : {mc['chain_accuracy']:.4f}  "
          f"(cadenas completas correctas)")
    print(f"  Step EM     : {mc['step_exact_match']:.4f}  "
          f"({int(mc['step_exact_match']*mc['n_steps'])}/{mc['n_steps']} pasos)")
    print(f"  Token F1    : {mc['mean_token_f1']:.4f}")
    if mc["bertscore"]:
        bs = mc["bertscore"]
        print(f"  BERTScore F1: {bs['f1']:.4f}  (P={bs['precision']:.4f}, R={bs['recall']:.4f})")
    print("\n  Desglose por tipo de cadena:")
    for t, v in mc["by_chain_type"].items():
        print(f"    {t:<40}  n={v['n']:>4}  EM={v['exact_match']:.3f}  F1={v['token_f1']:.3f}")

    pykeen = results["hit_at_k_pykeen"]
    if pykeen:
        print("\n  Hit@k KGE (test set PyKEEN):")
        for k, v in pykeen.items():
            print(f"    {k}: {v:.4f}")

    print(f"\n  Métricas  → {cfg.EVAL_RESULTS_FILE}")
    print(f"  Detalle   → {pred_file}")
    print("\n✓ Fase 6 completada.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validación del pipeline KGE + LLM")
    parser.add_argument("--n-samples", type=int, default=cfg.EVAL_SAMPLE_N,
                        help=f"Preguntas 1-hop a evaluar (default: {cfg.EVAL_SAMPLE_N})")
    parser.add_argument("--n-chains",  type=int, default=cfg.EVAL_SAMPLE_N,
                        help=f"Cadenas multi-hop a evaluar (default: {cfg.EVAL_SAMPLE_N})")
    parser.add_argument("--model",     default=cfg.DEFAULT_MODEL)
    parser.add_argument("--device",    default=cfg.DEVICE)
    args = parser.parse_args()
    run(
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        model_name=args.model,
        device=args.device,
    )
