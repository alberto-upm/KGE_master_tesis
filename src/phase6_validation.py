"""
Fase 6 — Validación del pipeline KGE + LLM.

Para cada muestra del qa_corpus.json:
  1. Se extrae la pregunta limpia (sin opciones a/b/c/d)
  2. Se construye el contexto KGE de la incidencia (subgrafo verbalizado)
  3. Se le hace la pregunta al LLM con ese contexto
  4. Se compara la respuesta generada con el campo "answer" del corpus

Métricas:
  - Token F1        : solapamiento exacto de tokens (respuesta corta vs entidad)
  - BERTScore F1    : similitud semántica (usando bert-base-spanish-wwm-cased)
  - Exact Match (EM): si la respuesta generada contiene exactamente el answer
  - Hit@k (PyKEEN)  : métricas de ranking del modelo KGE sobre el test set

Salida:
  out/evaluation/results.json         (métricas globales)
  out/evaluation/predictions.jsonl    (detalle por muestra)

Uso:
  python src/phase6_validation.py [--n-samples 200] [--model google/flan-t5-base]
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
# Extracción de pregunta limpia
# ---------------------------------------------------------------------------

def clean_question(raw_question: str) -> str:
    """
    Elimina las opciones múltiples (a/b/c/d) que vienen después del \\n.
    Entrada:  "¿Quién es el cliente?\\n  a) company_X  b) company_Y ..."
    Salida:   "¿Quién es el cliente?"
    """
    return raw_question.split("\n")[0].strip()


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def exact_match(prediction: str, reference: str) -> bool:
    """True si el answer exacto aparece en la predicción (case-insensitive)."""
    return reference.lower() in prediction.lower()


def token_f1(prediction: str, reference: str) -> float:
    """
    F1 de solapamiento de tokens entre predicción y referencia.
    Usa solo la primera línea de la predicción para no penalizar salidas largas
    donde el identificador correcto aparece al principio.
    """
    # Tomar solo la primera línea significativa (el identificador extraído)
    first_line = prediction.split("\n")[0].strip()
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
    """Lee las métricas Hit@k calculadas por PyKEEN en phase2."""
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
# Bucle de evaluación principal
# ---------------------------------------------------------------------------

def run_evaluation(
    llm,
    incidents_map: dict,
    similarity_index,
    implicit_preds: Optional[dict],
    n_samples: int = cfg.EVAL_SAMPLE_N,
    seed: int      = cfg.RANDOM_SEED,
) -> tuple[dict, list[dict]]:
    """
    Retorna (métricas_globales, lista_de_predicciones_por_muestra).
    """
    from phase4_llm_inference import get_verbalized_sentences, verbalize_props
    from phase5_config_subgraph import build_session_subgraph, verbalize_session_subgraph

    if not cfg.QA_CORPUS.exists():
        raise FileNotFoundError(
            f"Corpus no encontrado: {cfg.QA_CORPUS}\n"
            "Ejecuta primero:  python src/generate_corpus.py"
        )

    with open(cfg.QA_CORPUS, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    samples_pool = corpus.get("1hop", [])
    rng = random.Random(seed)
    samples = rng.sample(samples_pool, min(n_samples, len(samples_pool)))
    print(f"  {len(samples)} preguntas seleccionadas del corpus (tipo 1-hop).")

    predictions_log = []
    generated_answers = []
    reference_answers = []

    for i, item in enumerate(samples):
        inc_id    = item.get("context_inc", "")
        question  = clean_question(item.get("question", ""))
        reference = item.get("answer", "")
        q_type    = item.get("type", "")

        # Construir contexto verbalizado del subgrafo
        sentences = get_verbalized_sentences(inc_id)
        if not sentences:
            props = incidents_map.get(inc_id, {})
            if props:
                sentences = verbalize_props(inc_id, props)
        if not sentences:
            sentences = [f"La incidencia {inc_id} está registrada en el sistema."]

        # Enriquecer con casos similares (CBR) si hay embeddings
        if similarity_index is not None:
            try:
                sg = build_session_subgraph(
                    inc_id, incidents_map, similarity_index, implicit_preds
                )
                sentences = verbalize_session_subgraph(sg)
            except Exception:
                pass

        # Generar respuesta libre (sin opciones); do_extract=True extrae el ID
        prediction = llm.answer(sentences, question, do_extract=True)

        em    = exact_match(prediction, reference)
        tf1   = round(token_f1(prediction, reference), 4)

        generated_answers.append(prediction)
        reference_answers.append(reference)

        entry = {
            "id":         i + 1,
            "type":       q_type,
            "incident":   inc_id,
            "question":   question,
            "expected":   reference,
            "predicted":  prediction,
            "exact_match": em,
            "token_f1":   tf1,
        }
        predictions_log.append(entry)

        # Progreso en consola
        status = "✓" if em else "✗"
        print(f"  [{i+1:>4}/{len(samples)}] {status}  "
              f"F1={tf1:.2f}  esperado={reference!r}  "
              f"obtenido={prediction!r}")

    # Métricas globales
    total = len(predictions_log)
    em_score     = sum(e["exact_match"] for e in predictions_log) / total
    mean_tf1     = sum(e["token_f1"]    for e in predictions_log) / total
    bertscore    = evaluate_bertscore(generated_answers, reference_answers)
    pykeen_hits  = load_pykeen_hit_metrics()

    # Desglose por tipo de pregunta
    by_type: dict[str, dict] = {}
    for e in predictions_log:
        t = e["type"]
        by_type.setdefault(t, {"n": 0, "em": 0, "tf1_sum": 0.0})
        by_type[t]["n"]       += 1
        by_type[t]["em"]      += int(e["exact_match"])
        by_type[t]["tf1_sum"] += e["token_f1"]
    breakdown = {
        t: {
            "n":          v["n"],
            "exact_match": round(v["em"]  / v["n"], 4),
            "token_f1":   round(v["tf1_sum"] / v["n"], 4),
        }
        for t, v in sorted(by_type.items())
    }

    metrics = {
        "n_samples":    total,
        "exact_match":  round(em_score,  4),
        "mean_token_f1": round(mean_tf1, 4),
        "bertscore":    bertscore,
        "by_type":      breakdown,
        "hit_at_k_pykeen": pykeen_hits,
    }
    return metrics, predictions_log


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(
    n_samples:  int = cfg.EVAL_SAMPLE_N,
    model_name: str = cfg.DEFAULT_MODEL,
    device:     str = cfg.DEVICE,
) -> dict:
    print("=" * 60)
    print("FASE 6 — Validación del pipeline KGE + LLM")
    print("=" * 60)

    # Cargar grafo
    print("[1/4] Cargando grafo de incidencias ...")
    import generate_corpus as gc
    g       = gc.load_graph(cfg.TTL_FILE)
    inc_map = gc.build_incident_map(g)

    # Cargar índice de similitud
    print("[2/4] Cargando embeddings e índice de similitud ...")
    from phase5_config_subgraph import get_similarity_index, load_implicit_predictions
    sim_index  = get_similarity_index()
    impl_preds = load_implicit_predictions()
    if sim_index is None:
        print("  [Aviso] Embeddings no encontrados — se usará solo contexto directo.")

    # Cargar LLM
    print(f"[3/4] Cargando LLM: {model_name} (device={device}) ...")
    from phase4_llm_inference import KGEAugmentedLLM
    llm = KGEAugmentedLLM(model_name=model_name, device=device)

    # Evaluar
    print(f"\n[4/4] Evaluando {n_samples} preguntas ...\n")
    metrics, predictions_log = run_evaluation(
        llm, inc_map, sim_index, impl_preds, n_samples=n_samples
    )

    # Guardar resultados
    cfg.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.EVAL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pred_file = cfg.EVAL_DIR / "predictions.jsonl"
    with open(pred_file, "w", encoding="utf-8") as f:
        for entry in predictions_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Resumen final
    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    print(f"  Muestras evaluadas : {metrics['n_samples']}")
    print(f"  Exact Match (EM)   : {metrics['exact_match']:.4f}  "
          f"({int(metrics['exact_match']*metrics['n_samples'])}/{metrics['n_samples']})")
    print(f"  Mean Token F1      : {metrics['mean_token_f1']:.4f}")
    if metrics["bertscore"]:
        bs = metrics["bertscore"]
        print(f"  BERTScore F1       : {bs['f1']:.4f}  "
              f"(P={bs['precision']:.4f}, R={bs['recall']:.4f})")

    print("\n  Desglose por tipo de pregunta:")
    for t, v in metrics["by_type"].items():
        print(f"    {t:<35}  n={v['n']:>4}  EM={v['exact_match']:.3f}  F1={v['token_f1']:.3f}")

    if metrics["hit_at_k_pykeen"]:
        print("\n  Hit@k KGE (test set PyKEEN):")
        for k, v in metrics["hit_at_k_pykeen"].items():
            print(f"    {k}: {v:.4f}")

    print(f"\n  Métricas  → {cfg.EVAL_RESULTS_FILE}")
    print(f"  Detalle   → {pred_file}")
    print("\n✓ Fase 6 completada.")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validación del pipeline KGE + LLM")
    parser.add_argument("--n-samples", type=int, default=cfg.EVAL_SAMPLE_N)
    parser.add_argument("--model",     default=cfg.DEFAULT_MODEL)
    parser.add_argument("--device",    default=cfg.DEVICE)
    args = parser.parse_args()
    run(n_samples=args.n_samples, model_name=args.model, device=args.device)
