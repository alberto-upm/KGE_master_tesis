"""
Fase 6 — Evaluación entity-to-entity con KGE (2-hop).

Evalúa y compara modelos KGE en consultas de tipo:
  "Dado un cliente, ¿cuál es el técnico más probable?"
  "Dado un grupo de soporte, ¿cuál es la categoría más probable?"

A diferencia del LP eval clásico (incident → propiedad), aquí el punto de
partida es una entidad no-incidencia (empresa, grupo, tipo...) y la predicción
se hace mediante un camino de 2 saltos:
  source_entity → (proxies en el grafo) → predict_tails → target_entity

Los proxies son incidencias históricas que tienen source_prop == source_value.
Sólo se usan incidencias de test.tsv para las entradas del corpus (las dos
aristas del par no fueron vistas durante el entrenamiento).

Métricas: Hit@1, Hit@3, Hit@5, Hit@10, MRR — global y por par.

Salida:
  out/evaluation/entity_eval/<timestamp>/results.json
  out/evaluation/entity_eval/<timestamp>/comparison.csv   (tabla por modelo × par)
  out/evaluation/entity_eval/<timestamp>/predictions.csv  (detalle por entrada)

Uso:
  python src/phase6_entity_eval.py                        # todos los modelos
  python src/phase6_entity_eval.py --kge-model DistMult   # un solo modelo
  python src/phase6_entity_eval.py --kge-models DistMult ComplEx
  python src/phase6_entity_eval.py --n-samples 100
  python src/phase6_entity_eval.py --regen-corpus         # fuerza regeneración

Desde el pipeline:
  python src/run_pipeline.py --phase eval_entity
  python src/run_pipeline.py --phase eval_entity --kge-models DistMult TransE
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# ---------------------------------------------------------------------------
# Predicción 2-hop: source_entity → proxy incidents → predict_tails → target
# ---------------------------------------------------------------------------

def predict_entity_to_entity(
    model,
    factory,
    incidents_map: dict,
    source_prop: str,
    source_value: str,
    target_prop: str,
    top_k: int = 10,
    exclude_incident: str | None = None,
) -> list[tuple[str, int, float]]:
    """
    Predice el target_value dado source_value mediante 2 saltos en el grafo KG.

    Pasos:
      1. Busca incidencias proxy donde source_prop == source_value
         (excluye exclude_incident para evitar contaminación)
      2. Para cada proxy (hasta 30), llama predict_tails(proxy, target_prop)
      3. Agrega por (frecuencia DESC, score_medio DESC)

    Devuelve lista de (entity_label, frecuencia, score_medio) ordenada mejor primero.
    """
    from phase3_link_prediction import predict_tails

    proxies = [
        inc_id for inc_id, props in incidents_map.items()
        if inc_id != exclude_incident
        and source_value in props.get(source_prop, [])
    ]
    if not proxies:
        return []

    scores: dict[str, list[float]] = {}
    for proxy in proxies[:30]:
        for entity, score in predict_tails(model, factory, proxy, target_prop, top_k):
            scores.setdefault(entity, []).append(score)

    aggregated = [(ent, len(sc), sum(sc) / len(sc)) for ent, sc in scores.items()]
    aggregated.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return aggregated[:top_k]


# ---------------------------------------------------------------------------
# Evaluación de un modelo sobre el corpus entity-to-entity
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    eval_corpus: list,
    incidents_map: dict,
    top_k_values: list[int] | None = None,
) -> tuple[dict, list]:
    """
    Evalúa model_name en el corpus entity-to-entity.
    Devuelve (metrics_dict, prediction_rows).
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]
    top_k_fetch = max(top_k_values)

    from phase3_link_prediction import load_model_by_name
    print(f"\n  Evaluando {model_name} ({len(eval_corpus)} entradas) ...")
    model, factory = load_model_by_name(model_name)

    hits = {k: 0 for k in top_k_values}
    rr_sum = 0.0
    n_no_proxy = 0
    per_pair: dict[str, dict] = {}
    prediction_rows = []

    for entry in eval_corpus:
        pair_key = f"{entry['source_prop']}→{entry['target_prop']}"
        pp = per_pair.setdefault(pair_key, {
            "hits": {k: 0 for k in top_k_values},
            "rr": 0.0, "n": 0, "no_proxy": 0,
        })

        recs = predict_entity_to_entity(
            model, factory, incidents_map,
            source_prop=entry["source_prop"],
            source_value=entry["source_value"],
            target_prop=entry["target_prop"],
            top_k=top_k_fetch,
            exclude_incident=entry["incident_id"],
        )
        pred_entities = [ent for ent, _, _ in recs]
        true_value = entry["target_value"]

        if not recs:
            n_no_proxy += 1
            pp["no_proxy"] += 1

        rank = pred_entities.index(true_value) + 1 if true_value in pred_entities else None

        for k in top_k_values:
            if rank and rank <= k:
                hits[k] += 1
                pp["hits"][k] += 1
        if rank:
            rr_sum += 1.0 / rank
            pp["rr"] += 1.0 / rank
        pp["n"] += 1

        prediction_rows.append({
            "model":        model_name,
            "pair":         pair_key,
            "source_prop":  entry["source_prop"],
            "source_value": entry["source_value"],
            "target_prop":  entry["target_prop"],
            "target_value": true_value,
            "pred_top1":    pred_entities[0] if pred_entities else "",
            "rank":         rank if rank is not None else f">{top_k_fetch}",
            **{f"hit@{k}": 1 if (rank and rank <= k) else 0 for k in top_k_values},
            "no_proxy":     1 if not recs else 0,
        })

    n = len(eval_corpus)
    metrics = {
        "model":        model_name,
        "n_evaluated":  n,
        "no_proxy_pct": round(n_no_proxy / n, 4) if n else 0.0,
        **{f"hit@{k}": round(hits[k] / n, 4) for k in top_k_values},
        "mrr":          round(rr_sum / n, 4) if n else 0.0,
        "per_pair": {
            pk: {
                "n":          pp["n"],
                "no_proxy_pct": round(pp["no_proxy"] / pp["n"], 4) if pp["n"] else 0.0,
                **{f"hit@{k}": round(pp["hits"][k] / pp["n"], 4)
                   for k in top_k_values if pp["n"]},
                "mrr":        round(pp["rr"] / pp["n"], 4) if pp["n"] else 0.0,
            }
            for pk, pp in per_pair.items()
        },
    }
    return metrics, prediction_rows


# ---------------------------------------------------------------------------
# Impresión de tabla comparativa
# ---------------------------------------------------------------------------

def _print_comparison_table(all_results: dict) -> None:
    top_k_values = [1, 3, 5, 10]

    # Recoger todos los pares presentes
    all_pairs: list[str] = []
    for res in all_results.values():
        for pk in res.get("per_pair", {}):
            if pk not in all_pairs:
                all_pairs.append(pk)

    print(f"\n{'='*72}")
    print("  Evaluación Entity-to-Entity KGE")
    print(f"{'='*72}")

    # Tabla global
    print(f"\n  {'Modelo':<12} {'N':>5} {'H@1':>7} {'H@3':>7} {'H@5':>7} {'H@10':>7} {'MRR':>8}  {'SinProxy':>8}")
    print("  " + "-" * 68)
    for model_name, res in all_results.items():
        print(f"  {model_name:<12} {res['n_evaluated']:>5} "
              f"{res.get('hit@1',0):>7.4f} {res.get('hit@3',0):>7.4f} "
              f"{res.get('hit@5',0):>7.4f} {res.get('hit@10',0):>7.4f} "
              f"{res['mrr']:>8.4f}  {res.get('no_proxy_pct',0):>8.4f}")

    # Tabla por par
    for pair_key in all_pairs:
        print(f"\n  Par: {pair_key}")
        print(f"  {'Modelo':<12} {'N':>5} {'H@1':>7} {'H@3':>7} {'H@5':>7} {'H@10':>7} {'MRR':>8}")
        print("  " + "-" * 55)
        for model_name, res in all_results.items():
            pp = res.get("per_pair", {}).get(pair_key)
            if not pp:
                continue
            print(f"  {model_name:<12} {pp['n']:>5} "
                  f"{pp.get('hit@1',0):>7.4f} {pp.get('hit@3',0):>7.4f} "
                  f"{pp.get('hit@5',0):>7.4f} {pp.get('hit@10',0):>7.4f} "
                  f"{pp['mrr']:>8.4f}")

    print(f"\n{'='*72}\n")


# ---------------------------------------------------------------------------
# Guardado de resultados
# ---------------------------------------------------------------------------

def _save_results(out_dir: Path, all_results: dict, all_rows: list) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON completo
    json_path = out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # CSV de comparativa (modelo × par)
    comp_path = out_dir / "comparison.csv"
    top_k_values = [1, 3, 5, 10]
    fieldnames = ["model", "pair", "n", "hit@1", "hit@3", "hit@5", "hit@10", "mrr", "no_proxy_pct"]
    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, res in all_results.items():
            # Fila global
            writer.writerow({
                "model": model_name, "pair": "GLOBAL",
                "n": res["n_evaluated"],
                **{f"hit@{k}": res.get(f"hit@{k}", 0) for k in top_k_values},
                "mrr": res["mrr"],
                "no_proxy_pct": res.get("no_proxy_pct", 0),
            })
            # Filas por par
            for pair_key, pp in res.get("per_pair", {}).items():
                writer.writerow({
                    "model": model_name, "pair": pair_key,
                    "n": pp["n"],
                    **{f"hit@{k}": pp.get(f"hit@{k}", 0) for k in top_k_values},
                    "mrr": pp["mrr"],
                    "no_proxy_pct": pp.get("no_proxy_pct", 0),
                })

    # CSV de predicciones individuales
    pred_path = out_dir / "predictions.csv"
    if all_rows:
        with open(pred_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"  Resultados JSON  → {json_path}")
    print(f"  Comparativa CSV  → {comp_path}")
    print(f"  Predicciones CSV → {pred_path}")


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------

def run(
    models: list[str] | None = None,
    n_samples: int | None = None,
    regen_corpus: bool = False,
) -> dict:
    """
    Compara modelos KGE en el corpus entity-to-entity y guarda resultados.
    """
    models = models or cfg.KGE_MODELS

    # Generar corpus si no existe o se fuerza regeneración
    if regen_corpus or not cfg.ENTITY_EVAL_CORPUS.exists():
        print("Generando corpus entity-to-entity ...")
        from generate_corpus import generate_entity_to_entity_eval_corpus
        generate_entity_to_entity_eval_corpus()

    corpus = json.load(open(cfg.ENTITY_EVAL_CORPUS, encoding="utf-8"))
    print(f"Corpus cargado: {len(corpus)} entradas")

    if n_samples and n_samples < len(corpus):
        import random
        random.seed(cfg.RANDOM_SEED)
        corpus = random.sample(corpus, n_samples)
        print(f"Muestreo: {len(corpus)} entradas")

    # Cargar incidents_map (base de datos histórica completa)
    from rdflib import Graph
    from generate_corpus import build_incident_map
    print(f"Cargando grafo desde {cfg.TTL_FILE} ...")
    g = Graph()
    g.parse(str(cfg.TTL_FILE), format="turtle")
    incidents_map = build_incident_map(g)

    # Evaluar cada modelo
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.EVAL_DIR / "entity_eval" / ts

    all_results: dict = {}
    all_rows: list = []
    for model_name in models:
        metrics, rows = evaluate_model(model_name, corpus, incidents_map)
        all_results[model_name] = metrics
        all_rows.extend(rows)

    _print_comparison_table(all_results)
    _save_results(out_dir, all_results, all_rows)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación entity-to-entity con KGE (2-hop)"
    )
    parser.add_argument("--kge-model", default=None,
                        help="Un solo modelo KGE a evaluar (p.ej. DistMult)")
    parser.add_argument("--kge-models", nargs="+", default=None,
                        help=f"Lista de modelos a evaluar (default: todos {cfg.KGE_MODELS})")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Nº de entradas del corpus a evaluar (default: todas)")
    parser.add_argument("--regen-corpus", action="store_true",
                        help="Forzar regeneración del corpus entity_to_entity_eval.json")
    args = parser.parse_args()

    # Resolver lista de modelos
    if args.kge_model:
        models_to_eval = [args.kge_model]
    elif args.kge_models:
        models_to_eval = args.kge_models
    else:
        models_to_eval = cfg.KGE_MODELS

    run(
        models=models_to_eval,
        n_samples=args.n_samples,
        regen_corpus=args.regen_corpus,
    )
