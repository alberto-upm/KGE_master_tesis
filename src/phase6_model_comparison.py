"""
Comparación de modelos KGE mediante evaluación de link prediction.

Evalúa TransE, DistMult y ComplEx sobre el mismo corpus de test
(data/corpus/link_prediction_eval.json) y reporta Hit@k y MRR.

Opcionalmente verifica la integridad de la verbalización LLM:
comprueba que la respuesta del LLM contiene el identificador predicho.

Uso:
  python src/phase6_model_comparison.py
  python src/phase6_model_comparison.py --models TransE DistMult ComplEx
  python src/phase6_model_comparison.py --n-samples 100
  python src/phase6_model_comparison.py --verbalization-check --n-verb 50
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# ---------------------------------------------------------------------------
# Evaluación de un modelo
# ---------------------------------------------------------------------------

def evaluate_model_on_lp_corpus(
    model_name: str,
    eval_corpus: list[dict],
    top_k_values: list[int] = None,
) -> dict:
    """
    Carga el modelo KGE por nombre y evalúa link prediction sobre eval_corpus.

    Para cada entrada (subject, predicate, object_true):
      - Predice los top-max(k) objetos más probables
      - Calcula Hit@k (si object_true está en el top-k)
      - Calcula MRR (1/rank si encontrado, 0 si no)

    Las entidades desconocidas por la factory devuelven [] y se tratan
    como rank = infinito (contribución 0 a todos los métricas).

    Retorna dict con métricas globales y desglosadas por predicado.
    """
    from phase3_link_prediction import load_model_by_name, predict_tails

    if top_k_values is None:
        top_k_values = cfg.HIT_K_VALUES

    print(f"\n[{model_name}] Cargando modelo ...")
    model, factory = load_model_by_name(model_name)

    max_k = max(top_k_values)
    hits   = {k: 0 for k in top_k_values}
    rr_sum = 0.0
    per_rel: dict[str, dict] = defaultdict(lambda: {
        "n": 0,
        **{f"hit@{k}": 0 for k in top_k_values},
        "rr_sum": 0.0,
    })

    n = len(eval_corpus)
    print(f"[{model_name}] Evaluando {n} entradas ...")

    for i, entry in enumerate(eval_corpus):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n} ...")

        preds = predict_tails(
            model, factory,
            entry["subject"], entry["predicate"],
            top_k=max_k,
        )
        pred_entities = [e for e, _ in preds]
        true_obj = entry["object_true"]
        pred = entry["predicate"]

        rank = (pred_entities.index(true_obj) + 1) if true_obj in pred_entities else None
        rr   = (1.0 / rank) if rank else 0.0

        for k in top_k_values:
            hit = 1 if (rank is not None and rank <= k) else 0
            hits[k] += hit
            per_rel[pred][f"hit@{k}"] += hit

        rr_sum += rr
        per_rel[pred]["rr_sum"] += rr
        per_rel[pred]["n"] += 1

    # Métricas globales
    global_metrics = {
        "model":       model_name,
        "n_evaluated": n,
        "mrr":         round(rr_sum / n, 4) if n else 0.0,
        **{f"hit@{k}": round(hits[k] / n, 4) for k in top_k_values},
    }

    # Métricas por predicado
    per_relation = {}
    for rel, stats in per_rel.items():
        rel_n = stats["n"]
        per_relation[rel] = {
            "n":   rel_n,
            "mrr": round(stats["rr_sum"] / rel_n, 4) if rel_n else 0.0,
            **{f"hit@{k}": round(stats[f"hit@{k}"] / rel_n, 4) for k in top_k_values},
        }

    return {**global_metrics, "per_relation": per_relation}


# ---------------------------------------------------------------------------
# Comparación multi-modelo
# ---------------------------------------------------------------------------

def run_model_comparison(
    models: list[str] = None,
    n_samples: Optional[int] = None,
    top_k_values: list[int] = None,
) -> dict:
    """
    Evalúa todos los modelos sobre el mismo corpus y guarda los resultados.

    Retorna dict {model_name: metrics}.
    """
    if models is None:
        models = cfg.KGE_MODELS
    if top_k_values is None:
        top_k_values = cfg.HIT_K_VALUES

    print("=" * 60)
    print("COMPARACIÓN DE MODELOS KGE — Link Prediction")
    print("=" * 60)

    if not cfg.LP_EVAL_CORPUS.exists():
        raise FileNotFoundError(
            f"Corpus no encontrado: {cfg.LP_EVAL_CORPUS}\n"
            "Ejecuta primero:  python src/generate_corpus.py"
        )

    with open(cfg.LP_EVAL_CORPUS, encoding="utf-8") as f:
        corpus = json.load(f)

    if n_samples:
        corpus = corpus[:n_samples]

    print(f"Corpus: {len(corpus)} entradas")
    print(f"Modelos: {models}\n")

    results = {}
    for model_name in models:
        results[model_name] = evaluate_model_on_lp_corpus(
            model_name, corpus, top_k_values=top_k_values
        )

    _print_comparison_table(results, top_k_values)
    _save_results(results, top_k_values)

    return results


def _print_comparison_table(results: dict, top_k_values: list[int]) -> None:
    k_headers = "".join(f"  {'Hit@'+str(k):>8}" for k in top_k_values)
    print("\n" + "=" * (30 + 10 * len(top_k_values) + 10))
    print(f"  {'Modelo':<14}{k_headers}  {'MRR':>8}")
    print("  " + "-" * (26 + 10 * len(top_k_values) + 10))
    for model_name, m in results.items():
        k_vals = "".join(f"  {m[f'hit@{k}']:>8.4f}" for k in top_k_values)
        print(f"  {model_name:<14}{k_vals}  {m['mrr']:>8.4f}")
    print("=" * (30 + 10 * len(top_k_values) + 10))


def _save_results(results: dict, top_k_values: list[int]) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.MODEL_COMPARISON_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON completo
    json_path = out_dir / f"comparison_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(list(results.values()), f, ensure_ascii=False, indent=2)

    # CSV resumen
    csv_path = out_dir / f"comparison_{ts}.csv"
    fieldnames = ["model", "n_evaluated"] + [f"hit@{k}" for k in top_k_values] + ["mrr"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in results.values():
            writer.writerow({k: m[k] for k in fieldnames})

    print(f"\n  Resultados guardados en {out_dir}")
    print(f"  JSON: {json_path.name}")
    print(f"  CSV:  {csv_path.name}")


# ---------------------------------------------------------------------------
# Verificación de integridad de verbalización
# ---------------------------------------------------------------------------

def run_verbalization_integrity_check(
    model_name: str = 'DistMult',
    n_samples: int = 50,
    llm_base_url: str = cfg.VLLM_BASE_URL,
    llm_model_name: str = cfg.DEFAULT_MODEL,
) -> dict:
    """
    Verifica que el LLM preserva el identificador predicho al verbalizar.

    Flujo por muestra:
      1. Toma (subject, predicate, object_true) del corpus LP
      2. Predice top-1 con el modelo KGE
      3. Construye una verbalización con PRED_TEMPLATES_ES
      4. Pide al LLM que la reformule en lenguaje natural
      5. Extrae el identificador de la respuesta del LLM
      6. Comprueba que coincide con object_true

    Retorna:
      {
        "model":       model_name,
        "n_checked":   int,
        "integrity":   float,   # fracción de muestras donde se preserva el ID
        "details":     [...]
      }
    """
    from phase3_link_prediction import load_model_by_name, predict_tails
    from phase4_llm_inference import KGEAugmentedLLM, extract_answer
    from generate_corpus import PRED_TEMPLATES_ES

    print(f"\n[Integridad verbalización] Modelo: {model_name}, muestras: {n_samples}")

    with open(cfg.LP_EVAL_CORPUS, encoding="utf-8") as f:
        corpus = json.load(f)

    import random
    samples = random.sample(corpus, min(n_samples, len(corpus)))

    kge_model, factory = load_model_by_name(model_name)
    llm = KGEAugmentedLLM(model_name=llm_model_name, base_url=llm_base_url)

    template_map = {k: v for k, v in PRED_TEMPLATES_ES.items()}

    correct = 0
    details = []

    for entry in samples:
        preds = predict_tails(
            kge_model, factory,
            entry["subject"], entry["predicate"],
            top_k=1,
        )
        if not preds:
            details.append({**entry, "predicted": None, "llm_answer": None,
                             "integrity": False, "reason": "no_prediction"})
            continue

        predicted_entity = preds[0][0]

        # Construir frase de verbalización con el objeto predicho
        tmpl = template_map.get(entry["predicate"], "{s} — {p} → {o}.")
        verbalized = tmpl.format(
            s=entry["subject"],
            p=entry["predicate"],
            o=predicted_entity,
        )

        # Pedir al LLM que reformule
        context = [verbalized]
        question = (
            f"¿Cuál es el identificador exacto del {entry['predicate']} "
            f"de la incidencia {entry['subject']}?"
        )
        try:
            llm_raw = llm.answer(context, question, do_extract=False)
            llm_id  = extract_answer(llm_raw)
        except Exception as e:
            details.append({**entry, "predicted": predicted_entity,
                             "llm_answer": str(e), "integrity": False,
                             "reason": "llm_error"})
            continue

        # Integridad: el LLM devuelve el mismo identificador
        ok = predicted_entity.lower() in llm_id.lower()
        if ok:
            correct += 1

        details.append({
            "id":              entry["id"],
            "subject":         entry["subject"],
            "predicate":       entry["predicate"],
            "object_true":     entry["object_true"],
            "predicted":       predicted_entity,
            "llm_answer":      llm_id,
            "integrity":       ok,
        })

    n = len(samples)
    integrity_score = round(correct / n, 4) if n else 0.0

    result = {
        "model":     model_name,
        "n_checked": n,
        "integrity": integrity_score,
        "details":   details,
    }

    print(f"  Integridad verbalización: {integrity_score:.2%}  ({correct}/{n})")

    # Guardar
    cfg.MODEL_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = cfg.MODEL_COMPARISON_DIR / f"verbalization_integrity_{model_name.lower()}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Guardado: {out_path}")

    return result


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(
    models: list[str] = None,
    n_samples: int = None,
    verbalization_check: bool = False,
    n_verb: int = 50,
    verb_model: str = 'DistMult',
):
    results = run_model_comparison(models=models, n_samples=n_samples)
    if verbalization_check:
        run_verbalization_integrity_check(
            model_name=verb_model,
            n_samples=n_verb,
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparación de modelos KGE")
    parser.add_argument("--models", nargs="+", default=cfg.KGE_MODELS,
                        help="Modelos a comparar (default: todos en KGE_MODELS)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Nº de entradas del corpus a evaluar (default: todas)")
    parser.add_argument("--verbalization-check", action="store_true",
                        help="Ejecutar verificación de integridad de verbalización")
    parser.add_argument("--n-verb", type=int, default=50,
                        help="Muestras para verificación de verbalización (default: 50)")
    parser.add_argument("--verb-model", default="DistMult",
                        help="Modelo KGE para verificación de verbalización")
    args = parser.parse_args()
    run(
        models=args.models,
        n_samples=args.n_samples,
        verbalization_check=args.verbalization_check,
        n_verb=args.n_verb,
        verb_model=args.verb_model,
    )
