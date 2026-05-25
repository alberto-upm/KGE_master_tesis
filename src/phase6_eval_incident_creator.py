"""
Fase 6 — Evaluación end-to-end del pipeline de creación guiada (phase4).

Simula la cascada REGLA → KGE+CBR de phase4_incident_creator.py sobre todas
las incidencias de data/test_eval.ttl (5% reservado por phase 0).

Flujo por incidencia:
  Para cada propiedad de INCIDENT_PROPS, en orden:
    0. Si la incidencia real NO tiene esa propiedad → skip (no cuenta).
    1. Consulta el motor de reglas con las propiedades ya conocidas:
         · Si la regla acierta (valor == ground truth)         → rule_hit
         · Si la regla devuelve algo distinto al ground truth  → rule_miss
                                                              → pasa a KGE+CBR
         · Si no hay regla aplicable                            → pasa a KGE+CBR
    2. KGE+CBR (recommend_property con top-K):
         · Si el ground truth está en las top-K                 → kge_hit (con rank)
         · Si no                                                → fail

  Tras evaluar el campo, se inyecta el valor REAL en known_props para
  acumular contexto realista en los pasos siguientes (golden path).

Métricas (por propiedad y globales):
  · n_evaluated, n_skipped, rule_hit, rule_miss
  · kge_hit@1/3/5/10, fail
  · accuracy@1/3/5/10 = (rule_hit + kge_hit@k) / n_evaluated
  · rule_coverage     = (rule_hit + rule_miss) / n_evaluated
  · rule_precision    = rule_hit / (rule_hit + rule_miss) si > 0

Salida:
  out/evaluation/incident_creator_full/<timestamp>/
    results.json
    per_property.csv
    predictions.csv

Uso:
  python src/phase6_eval_incident_creator.py
  python src/phase6_eval_incident_creator.py --kge-model TransE
  python src/phase6_eval_incident_creator.py --n-samples 200
  python src/phase6_eval_incident_creator.py --top-k 10
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from phase4_incident_creator import (
    INCIDENT_PROPS,
    MULTI_VALUE_PROPS,
    _build_incidents_map_from_tsv,
    recommend_property,
)
from rule_engine_pyclause import RuleEnginePyClause


# Campos del wizard que SÍ se evalúan en la cascada single-value.
# Se omiten los multi-valor (hasIntervention) porque sus IDs son únicos
# y no tiene sentido medirlos contra una recomendación single-target.
EVAL_PROPS = [p for p in INCIDENT_PROPS if p not in MULTI_VALUE_PROPS]


# ---------------------------------------------------------------------------
# Carga de incidencias de test desde TTL
# ---------------------------------------------------------------------------

def _load_test_incidents_from_ttl(ttl_path: Path) -> dict:
    """
    Lee data/test_eval.ttl con rdflib y devuelve
    {incident_id: {predicate: [values]}} filtrando sólo INCIDENT_PROPS.
    """
    from phase1b_generate_corpus import load_graph, build_incident_map

    if not ttl_path.exists():
        raise FileNotFoundError(
            f"No encontrado: {ttl_path}\n"
            "Ejecuta primero:  python src/run_pipeline.py --phase 0"
        )

    g       = load_graph(ttl_path)
    raw_map = build_incident_map(g)

    prop_set = set(EVAL_PROPS)
    incidents: dict = {}
    for inc_id, props in raw_map.items():
        filtered = {p: v for p, v in props.items() if p in prop_set}
        if filtered:
            incidents[inc_id] = filtered
    return incidents


# ---------------------------------------------------------------------------
# Pipeline de evaluación (cascada REGLA → KGE+CBR)
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    kge_model_name: str = "TransE",
    top_k_values: tuple[int, ...] = (1, 3, 5, 10),
    n_samples: int | None = None,
    random_seed: int = cfg.RANDOM_SEED,
) -> tuple[dict, list[dict]]:
    """
    Simula la cascada REGLA → KGE+CBR de phase4 sobre las incidencias de
    cfg.TEST_TTL y devuelve (per_prop_stats, prediction_rows).
    """
    from phase3_link_prediction import load_model_by_name

    max_k = max(top_k_values)

    print("=" * 60)
    print(f"  Evaluación pipeline incident_creator — {kge_model_name}")
    print("=" * 60)

    # --- Recursos ---
    print("\n[1/4] Cargando pool CBR desde train.tsv ...")
    incidents_map = _build_incidents_map_from_tsv()
    print(f"      Pool CBR: {len(incidents_map):,} incidencias")

    print(f"\n[2/4] Cargando incidencias de test desde {cfg.TEST_TTL.name} ...")
    test_incidents = _load_test_incidents_from_ttl(cfg.TEST_TTL)
    print(f"      Test set: {len(test_incidents):,} incidencias")

    print("\n[3/4] Cargando motor de reglas y modelo KGE ...")
    rule_engine = RuleEnginePyClause()
    rs = rule_engine.stats()
    print(f"      Reglas: {rs['total_rules']:,} ({rs['predicates']} predicados)")

    model, factory = load_model_by_name(kge_model_name)
    print(f"      Modelo KGE: {kge_model_name}")

    # --- Muestreo opcional ---
    inc_ids = sorted(test_incidents.keys())
    if n_samples and n_samples < len(inc_ids):
        rng = random.Random(random_seed)
        inc_ids = rng.sample(inc_ids, n_samples)

    print(f"\n[4/4] Evaluando {len(inc_ids):,} incidencias  (top_k={max_k}) ...")

    # --- Acumuladores por propiedad ---
    per_prop: dict[str, dict] = defaultdict(lambda: {
        "n_evaluated": 0,
        "n_skipped":   0,
        "rule_hit":    0,
        "rule_miss":   0,
        "kge_hit":     {k: 0 for k in top_k_values},
        "fail":        0,
        "rr_sum":      0.0,   # para MRR sobre los pasos que llegan a KGE+CBR
    })

    prediction_rows: list[dict] = []
    progress_every = max(50, len(inc_ids) // 20)

    # --- Bucle principal ---
    for i, inc_id in enumerate(inc_ids):
        if (i + 1) % progress_every == 0:
            print(f"      {i+1:,}/{len(inc_ids):,} ...")

        ground_truth = test_incidents[inc_id]
        # known_props sigue indexado por INCIDENT_PROPS porque el motor de
        # reglas y recommend_property se construyeron sobre esa lista.
        known_props: dict[str, str | None] = {p: None for p in INCIDENT_PROPS}

        # Excluir la incidencia objetivo del pool CBR para no contaminar
        cbr_pool = {k: v for k, v in incidents_map.items() if k != inc_id}

        for prop in EVAL_PROPS:
            stats = per_prop[prop]
            true_values = ground_truth.get(prop, [])

            if not true_values:
                stats["n_skipped"] += 1
                continue

            # Multi-valor (hasIntervention): evaluar contra el primer valor.
            # Los IDs de intervención son únicos → fallarán casi siempre, pero
            # se reporta igual para tener constancia.
            true_value = true_values[0] if isinstance(true_values, list) else true_values
            stats["n_evaluated"] += 1

            outcome:     str          = ""
            rank:        int | None   = None
            rule_id:     str | None   = None
            rule_value:  str | None   = None

            # --- Paso 1: motor de reglas ---
            rule_hit = rule_engine.query(known_props, prop)
            if rule_hit:
                rule_value = rule_hit.get("value")
                rule_id    = rule_hit.get("rule_id")
                if rule_value == true_value:
                    stats["rule_hit"] += 1
                    outcome = "rule_hit"
                else:
                    stats["rule_miss"] += 1
                    # No marcamos como hit; la cascada cae a KGE+CBR

            # --- Paso 2: KGE+CBR (si la regla no acertó) ---
            if outcome != "rule_hit":
                recs, _n_proxies = recommend_property(
                    known_props=known_props,
                    target_prop=prop,
                    incidents_map=cbr_pool,
                    model=model,
                    factory=factory,
                    top_k=max_k,
                )
                rec_entities = [ent for ent, *_ in recs]

                if true_value in rec_entities:
                    rank = rec_entities.index(true_value) + 1
                    for k in top_k_values:
                        if rank <= k:
                            stats["kge_hit"][k] += 1
                    stats["rr_sum"] += 1.0 / rank
                    outcome = "kge_hit"
                else:
                    stats["fail"] += 1
                    outcome = "fail"

            # --- Avanzar con el ground truth (golden path) ---
            known_props[prop] = true_value

            prediction_rows.append({
                "incident":   inc_id,
                "property":   prop,
                "true_value": true_value,
                "outcome":    outcome,
                "rule_id":    rule_id or "",
                "rule_value": rule_value or "",
                "rank":       rank if rank is not None else "",
                **{f"hit@{k}": 1 if (rank is not None and rank <= k) else 0
                   for k in top_k_values},
                "rule_correct": 1 if outcome == "rule_hit" else 0,
            })

    return per_prop, prediction_rows


# ---------------------------------------------------------------------------
# Cálculo de métricas derivadas
# ---------------------------------------------------------------------------

def _compute_metrics(per_prop: dict, top_k_values: tuple[int, ...]) -> dict:
    """
    Calcula métricas derivadas por propiedad y globales:
      accuracy@k = (rule_hit + kge_hit@k) / n_evaluated
      rule_coverage = (rule_hit + rule_miss) / n_evaluated
      rule_precision = rule_hit / (rule_hit + rule_miss)
      mrr_kge = rr_sum / (rule_miss + fail + kge_hit_total)
                (rank reciprocal entre los pasos que pasaron por KGE+CBR)
    """
    summary = {"per_property": {}, "global": {}}

    # Acumuladores globales
    g_eval     = 0
    g_skipped  = 0
    g_rule_hit = 0
    g_rule_miss = 0
    g_kge_hit  = {k: 0 for k in top_k_values}
    g_fail     = 0
    g_rr_sum   = 0.0

    for prop in EVAL_PROPS:
        s = per_prop.get(prop)
        if s is None or s["n_evaluated"] == 0:
            continue

        n = s["n_evaluated"]
        # Pasos que llegaron a KGE+CBR: cada uno termina en kge_hit@max_k o en fail
        kge_steps = s["kge_hit"][max(top_k_values)] + s["fail"]

        prop_metrics = {
            "n_evaluated":   n,
            "n_skipped":     s["n_skipped"],
            "rule_hit":      s["rule_hit"],
            "rule_miss":     s["rule_miss"],
            "fail":          s["fail"],
            "kge_hit":       {k: s["kge_hit"][k] for k in top_k_values},
            "accuracy":      {
                k: round((s["rule_hit"] + s["kge_hit"][k]) / n, 4) for k in top_k_values
            },
            "rule_coverage":  round((s["rule_hit"] + s["rule_miss"]) / n, 4),
            "rule_precision": (
                round(s["rule_hit"] / (s["rule_hit"] + s["rule_miss"]), 4)
                if (s["rule_hit"] + s["rule_miss"]) > 0 else None
            ),
            "mrr_kge":       round(s["rr_sum"] / kge_steps, 4) if kge_steps else 0.0,
        }
        summary["per_property"][prop] = prop_metrics

        g_eval     += n
        g_skipped  += s["n_skipped"]
        g_rule_hit += s["rule_hit"]
        g_rule_miss += s["rule_miss"]
        g_fail     += s["fail"]
        g_rr_sum   += s["rr_sum"]
        for k in top_k_values:
            g_kge_hit[k] += s["kge_hit"][k]

    if g_eval > 0:
        g_kge_steps = g_kge_hit[max(top_k_values)] + g_fail
        summary["global"] = {
            "n_evaluated":   g_eval,
            "n_skipped":     g_skipped,
            "rule_hit":      g_rule_hit,
            "rule_miss":     g_rule_miss,
            "fail":          g_fail,
            "kge_hit":       g_kge_hit,
            "accuracy":      {
                k: round((g_rule_hit + g_kge_hit[k]) / g_eval, 4) for k in top_k_values
            },
            "rule_coverage":  round((g_rule_hit + g_rule_miss) / g_eval, 4),
            "rule_precision": (
                round(g_rule_hit / (g_rule_hit + g_rule_miss), 4)
                if (g_rule_hit + g_rule_miss) > 0 else None
            ),
            "mrr_kge":       round(g_rr_sum / g_kge_steps, 4) if g_kge_steps else 0.0,
        }

    return summary


# ---------------------------------------------------------------------------
# Salida por consola
# ---------------------------------------------------------------------------

def _print_results(
    summary: dict,
    kge_model_name: str,
    top_k_values: tuple[int, ...],
) -> None:
    g = summary.get("global", {})
    if not g:
        print("\n[!] Sin resultados.")
        return

    print(f"\n{'='*70}")
    print(f"  Resultados — {kge_model_name}")
    print(f"  Campos evaluados: {g['n_evaluated']:,}   "
          f"saltados: {g['n_skipped']:,}")
    print(f"{'='*70}")
    print(f"  Cascada:")
    print(f"    rule_hit    : {g['rule_hit']:>7,}   ({g['rule_hit']/g['n_evaluated']:.2%})")
    print(f"    rule_miss   : {g['rule_miss']:>7,}   ({g['rule_miss']/g['n_evaluated']:.2%})")
    print(f"    fail        : {g['fail']:>7,}   ({g['fail']/g['n_evaluated']:.2%})")
    print(f"  Reglas:")
    print(f"    coverage    : {g['rule_coverage']:.4f}")
    if g["rule_precision"] is not None:
        print(f"    precision   : {g['rule_precision']:.4f}")
    print(f"  Pipeline completo (regla + KGE+CBR):")
    for k in top_k_values:
        print(f"    accuracy@{k:<3}: {g['accuracy'][k]:.4f}")
    print(f"    MRR (KGE)   : {g['mrr_kge']:.4f}")

    print(f"\n  Desglose por propiedad:")
    header = (f"  {'Propiedad':<26} {'N':>5} "
              f"{'RH':>5} {'RM':>5} {'FAIL':>5} "
              + " ".join(f"{'A@'+str(k):>6}" for k in top_k_values))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for prop in EVAL_PROPS:
        pm = summary["per_property"].get(prop)
        if not pm:
            continue
        acc_cols = " ".join(f"{pm['accuracy'][k]:>6.3f}" for k in top_k_values)
        print(f"  {prop:<26} {pm['n_evaluated']:>5} "
              f"{pm['rule_hit']:>5} {pm['rule_miss']:>5} {pm['fail']:>5} "
              f"{acc_cols}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------

def _save_results(
    out_dir: Path,
    summary: dict,
    prediction_rows: list[dict],
    kge_model_name: str,
    top_k_values: tuple[int, ...],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON con todo
    json_path = out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "kge_model":    kge_model_name,
            "top_k_values": list(top_k_values),
            **summary,
        }, f, ensure_ascii=False, indent=2)

    # CSV por propiedad
    pp_path = out_dir / "per_property.csv"
    fieldnames = (["property", "n_evaluated", "n_skipped", "rule_hit", "rule_miss",
                   "fail", "rule_coverage", "rule_precision", "mrr_kge"]
                  + [f"accuracy@{k}" for k in top_k_values]
                  + [f"kge_hit@{k}"  for k in top_k_values])
    with open(pp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for prop in EVAL_PROPS:
            pm = summary["per_property"].get(prop)
            if not pm:
                continue
            row = {
                "property":       prop,
                "n_evaluated":    pm["n_evaluated"],
                "n_skipped":      pm["n_skipped"],
                "rule_hit":       pm["rule_hit"],
                "rule_miss":      pm["rule_miss"],
                "fail":           pm["fail"],
                "rule_coverage":  pm["rule_coverage"],
                "rule_precision": pm["rule_precision"] if pm["rule_precision"] is not None else "",
                "mrr_kge":        pm["mrr_kge"],
            }
            for k in top_k_values:
                row[f"accuracy@{k}"] = pm["accuracy"][k]
                row[f"kge_hit@{k}"]  = pm["kge_hit"][k]
            writer.writerow(row)

    # CSV de predicciones individuales
    pred_path = out_dir / "predictions.csv"
    if prediction_rows:
        with open(pred_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()))
            writer.writeheader()
            writer.writerows(prediction_rows)

    print(f"  Resultados JSON  → {json_path}")
    print(f"  Por propiedad    → {pp_path}")
    print(f"  Predicciones     → {pred_path}")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(
    kge_model_name: str = "TransE",
    n_samples: int | None = None,
    top_k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.EVAL_DIR / "incident_creator_full" / ts

    per_prop, prediction_rows = evaluate_pipeline(
        kge_model_name=kge_model_name,
        top_k_values=top_k_values,
        n_samples=n_samples,
    )
    summary = _compute_metrics(per_prop, top_k_values)
    _print_results(summary, kge_model_name, top_k_values)
    _save_results(out_dir, summary, prediction_rows, kge_model_name, top_k_values)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación end-to-end del pipeline incident_creator "
                    "(REGLA → KGE+CBR) sobre test_eval.ttl"
    )
    parser.add_argument("--kge-model", default="TransE",
                        help=f"Modelo KGE (default: TransE). Opciones: {cfg.KGE_MODELS}")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Nº de incidencias a evaluar (default: todas)")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 10],
                        help="Valores de k para accuracy@k (default: 1 3 5 10)")
    args = parser.parse_args()

    run(
        kge_model_name=args.kge_model,
        n_samples=args.n_samples,
        top_k_values=tuple(args.top_k),
    )
