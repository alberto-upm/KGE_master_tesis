"""
Fase 6 — Evaluación del pipeline de creación guiada de incidencias (CBR + KGE + LLM).

Simula el wizard de phase4_incident_creator.py de forma automática sobre las incidencias
del conjunto de test, sin intervención humana.

Flujo por muestra:
  Para cada incidencia del test.tsv:
    1. Se revelan sus propiedades en el orden de INCIDENT_PROPS (una a una)
    2. En cada paso, se llaman recommend_property() con las propiedades ya conocidas
       (excluyendo la incidencia objetivo del pool CBR para evitar contaminación)
    3. Se comprueba si el valor real aparece en las top-K recomendaciones
    4. Opcionalmente, se genera el resumen LLM y se mide la fidelidad

Métricas:

  RECOMENDACIÓN KGE (CBR + predict_tails)
    hit@1, hit@3, hit@5  — fracción donde el valor correcto está en las top-K
    mrr                  — Mean Reciprocal Rank del valor correcto
    cbr_coverage         — fracción de pasos donde CBR encontró al menos 1 proxy
    (todo desglosado por propiedad)

  FIDELIDAD LLM (opcional, sobre n_llm_samples incidencias)
    faithfulness         — fracción de propiedades rellenadas mencionadas en el resumen
    full_faithfulness    — fracción de incidencias donde TODAS las props aparecen

Salida:
  out/evaluation/incident_creator/<timestamp>/results.json
  out/evaluation/incident_creator/<timestamp>/per_property.csv
  out/evaluation/incident_creator/<timestamp>/predictions.csv

Uso:
  python src/phase6_incident_creator_eval.py
  python src/phase6_incident_creator_eval.py --kge-model TransE
  python src/phase6_incident_creator_eval.py --n-samples 200
  python src/phase6_incident_creator_eval.py --no-llm
  python src/phase6_incident_creator_eval.py --n-llm-samples 50
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
from phase4_incident_creator import INCIDENT_PROPS, recommend_property


# ---------------------------------------------------------------------------
# Carga del conjunto de test
# ---------------------------------------------------------------------------

def _load_test_incidents(test_tsv: Path) -> dict:
    """
    Lee test.tsv (formato PyKEEN: head \\t relation \\t tail)
    y devuelve {incident_id: {predicate: [values]}} filtrando sólo
    las relaciones de INCIDENT_PROPS y sólo las entidades tipo incident_*.
    """
    incidents: dict[str, dict[str, list[str]]] = {}
    with open(test_tsv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            head, relation, tail = parts[0], parts[1], parts[2]
            if not head.startswith("incident_"):
                continue
            if relation not in INCIDENT_PROPS:
                continue
            incidents.setdefault(head, {}).setdefault(relation, []).append(tail)
    return incidents


# ---------------------------------------------------------------------------
# Evaluación KGE (CBR + predict_tails)
# ---------------------------------------------------------------------------

def evaluate_kge(
    model_name: str = "DistMult",
    n_samples: int | None = None,
    top_k_values: list[int] | None = None,
    random_seed: int = cfg.RANDOM_SEED,
) -> dict:
    """
    Para cada incidencia del test.tsv simula el wizard y evalúa la calidad
    de las recomendaciones de recommend_property().

    Excluye siempre la incidencia objetivo del pool CBR para evitar
    que el modelo se encuentre a sí mismo como proxy.
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]
    top_k_fetch = max(top_k_values)

    # --- cargar recursos ---
    print("\n[Evaluación KGE] Cargando recursos ...")

    from rdflib import Graph
    from generate_corpus import build_incident_map
    from phase3_link_prediction import load_model_by_name

    print(f"  Modelo KGE: {model_name}")
    model, factory = load_model_by_name(model_name)

    print(f"  Cargando grafo desde {cfg.TTL_FILE} ...")
    g = Graph()
    g.parse(str(cfg.TTL_FILE), format="turtle")
    full_incidents_map = build_incident_map(g)

    if not cfg.TEST_TSV.exists():
        raise FileNotFoundError(
            f"No encontrado: {cfg.TEST_TSV}\n"
            "Ejecuta primero: python src/phase1_triples.py"
        )
    test_incidents = _load_test_incidents(cfg.TEST_TSV)
    print(f"  Incidencias en test.tsv: {len(test_incidents):,}")

    # --- muestreo opcional ---
    incident_ids = sorted(test_incidents.keys())
    if n_samples and n_samples < len(incident_ids):
        random.seed(random_seed)
        incident_ids = random.sample(incident_ids, n_samples)
    print(f"  Evaluando: {len(incident_ids):,} incidencias\n")

    # --- acumuladores ---
    hits = {k: 0 for k in top_k_values}
    rr_sum = 0.0
    n_steps = 0
    cbr_with_proxy = 0

    per_prop: dict[str, dict] = {
        p: {"hits": {k: 0 for k in top_k_values}, "rr": 0.0, "n": 0, "cbr": 0}
        for p in INCIDENT_PROPS
    }

    prediction_rows = []  # para el CSV de detalle

    # --- simulación del wizard ---
    for inc_id in incident_ids:
        ground_truth = test_incidents[inc_id]   # {pred: [values]} del test set
        known_props: dict[str, str | None] = {p: None for p in INCIDENT_PROPS}

        # Pool CBR sin la incidencia objetivo
        incidents_map_no_target = {
            k: v for k, v in full_incidents_map.items() if k != inc_id
        }

        # Revelar propiedades en orden de prioridad
        for prop in INCIDENT_PROPS:
            true_values = ground_truth.get(prop)
            if not true_values:
                # Esta propiedad no está en el test set para esta incidencia → saltar
                # Pero si está en el grafo completo, tomar su valor para el contexto CBR
                full_values = full_incidents_map.get(inc_id, {}).get(prop)
                if full_values:
                    known_props[prop] = full_values[0]
                continue

            true_value = true_values[0]  # primer valor (la mayoría tienen uno solo)

            # Obtener recomendaciones (incluye n_proxies)
            recs, n_proxies = recommend_property(
                known_props=known_props,
                target_prop=prop,
                incidents_map=incidents_map_no_target,
                model=model,
                factory=factory,
                top_k=top_k_fetch,
            )
            has_proxy = n_proxies > 0
            rec_entities = [ent for ent, _freq, _score in recs]

            # Calcular rank
            rank = None
            if true_value in rec_entities:
                rank = rec_entities.index(true_value) + 1

            # Acumular métricas globales
            n_steps += 1
            if has_proxy:
                cbr_with_proxy += 1
            if rank is not None:
                for k in top_k_values:
                    if rank <= k:
                        hits[k] += 1
                rr_sum += 1.0 / rank

            # Acumular métricas por propiedad
            pp = per_prop[prop]
            pp["n"] += 1
            if has_proxy:
                pp["cbr"] += 1
            if rank is not None:
                for k in top_k_values:
                    if rank <= k:
                        pp["hits"][k] += 1
                pp["rr"] += 1.0 / rank

            # Fila de detalle
            prediction_rows.append({
                "incident":   inc_id,
                "property":   prop,
                "true_value": true_value,
                "pred_top1":  rec_entities[0] if rec_entities else "",
                "rank":       rank if rank is not None else ">top_k",
                "hit@1":      1 if (rank and rank <= 1) else 0,
                "hit@3":      1 if (rank and rank <= 3) else 0,
                "hit@5":      1 if (rank and rank <= 5) else 0,
                "hit@10":     1 if (rank and rank <= 10) else 0,
                "cbr_proxy":  1 if has_proxy else 0,
            })

            # Confirmar el valor real para el siguiente paso (simulación de usuario aceptando)
            known_props[prop] = true_value

    # --- métricas globales ---
    results = {
        "model":        model_name,
        "n_incidents":  len(incident_ids),
        "n_steps":      n_steps,
        "cbr_coverage": round(cbr_with_proxy / n_steps, 4) if n_steps else 0.0,
        **{f"hit@{k}": round(hits[k] / n_steps, 4) for k in top_k_values},
        "mrr":          round(rr_sum / n_steps, 4) if n_steps else 0.0,
        "per_property": {
            prop: {
                "n":            pp["n"],
                "cbr_coverage": round(pp["cbr"] / pp["n"], 4) if pp["n"] else 0.0,
                **{f"hit@{k}": round(pp["hits"][k] / pp["n"], 4)
                   for k in top_k_values if pp["n"]},
                "mrr":          round(pp["rr"] / pp["n"], 4) if pp["n"] else 0.0,
            }
            for prop, pp in per_prop.items() if pp["n"] > 0
        },
    }
    return results, prediction_rows


# ---------------------------------------------------------------------------
# Evaluación LLM (fidelidad del resumen)
# ---------------------------------------------------------------------------

def evaluate_llm_faithfulness(
    model_name: str = "DistMult",
    llm_model_name: str = cfg.DEFAULT_MODEL,
    n_samples: int = 50,
    random_seed: int = cfg.RANDOM_SEED,
) -> dict:
    """
    Para n_samples incidencias completas del test:
      1. Verbaliza sus propiedades con verbalize_props()
      2. Genera un resumen LLM
      3. Comprueba si cada propiedad rellenada aparece en el texto del resumen
    Devuelve faithfulness score (fracción de valores mencionados).
    """
    from rdflib import Graph
    from generate_corpus import build_incident_map
    from phase4_llm_inference import verbalize_props, KGEAugmentedLLM

    print("\n[Evaluación LLM] Cargando recursos ...")
    g = Graph()
    g.parse(str(cfg.TTL_FILE), format="turtle")
    incidents_map = build_incident_map(g)

    # Incidencias con ≥5 propiedades rellenas (casos ricos)
    rich = [(inc_id, props) for inc_id, props in incidents_map.items()
            if len(props) >= 5]
    random.seed(random_seed)
    sample = random.sample(rich, min(n_samples, len(rich)))
    print(f"  Evaluando fidelidad LLM en {len(sample)} incidencias ...")

    try:
        llm = KGEAugmentedLLM(model_name=llm_model_name, base_url=cfg.VLLM_BASE_URL)
    except Exception as e:
        print(f"  [!] LLM no disponible: {e}")
        return {"error": str(e)}

    total_props = 0
    mentioned_props = 0
    fully_faithful = 0
    rows = []

    for inc_id, props in sample:
        # Sólo propiedades en INCIDENT_PROPS
        filtered = {k: v[0] if isinstance(v, list) else v
                    for k, v in props.items()
                    if k in INCIDENT_PROPS and v}
        if not filtered:
            continue

        sentences = verbalize_props(inc_id, filtered)
        question = "Resume en una frase en español la incidencia con los datos anteriores."
        try:
            summary = llm.answer(sentences, question, do_extract=False)
        except Exception:
            summary = ""

        summary_lower = summary.lower()
        prop_results = {}
        for prop, val in filtered.items():
            mentioned = val.lower() in summary_lower
            prop_results[prop] = mentioned
            total_props += 1
            if mentioned:
                mentioned_props += 1

        all_mentioned = all(prop_results.values())
        if all_mentioned:
            fully_faithful += 1

        rows.append({
            "incident":       inc_id,
            "n_props":        len(filtered),
            "n_mentioned":    sum(prop_results.values()),
            "fully_faithful": int(all_mentioned),
            "summary":        summary[:200],
        })

    n = len(rows)
    return {
        "llm_model":       llm_model_name,
        "n_evaluated":     n,
        "faithfulness":    round(mentioned_props / total_props, 4) if total_props else 0.0,
        "full_faithfulness": round(fully_faithful / n, 4) if n else 0.0,
        "rows":            rows,
    }


# ---------------------------------------------------------------------------
# Impresión de tabla de resultados
# ---------------------------------------------------------------------------

def _print_results(kge_results: dict, llm_results: dict | None) -> None:
    model = kge_results["model"]
    n     = kge_results["n_incidents"]
    steps = kge_results["n_steps"]

    print(f"\n{'='*65}")
    print(f"  Evaluación Incident Creator — {model}")
    print(f"  Incidencias: {n}   Pasos totales: {steps}")
    print(f"{'='*65}")
    print(f"  Métricas globales KGE:")
    print(f"    CBR coverage : {kge_results['cbr_coverage']:.4f}")
    print(f"    Hit@1        : {kge_results.get('hit@1', 0):.4f}")
    print(f"    Hit@3        : {kge_results.get('hit@3', 0):.4f}")
    print(f"    Hit@5        : {kge_results.get('hit@5', 0):.4f}")
    print(f"    Hit@10       : {kge_results.get('hit@10', 0):.4f}")
    print(f"    MRR          : {kge_results['mrr']:.4f}")

    print(f"\n  Por propiedad:")
    header = f"  {'Propiedad':<26} {'N':>5} {'CBR':>6} {'H@1':>6} {'H@3':>6} {'H@5':>6} {'H@10':>6} {'MRR':>7}"
    print(header)
    print("  " + "-" * 70)
    for prop in INCIDENT_PROPS:
        pp = kge_results["per_property"].get(prop)
        if not pp:
            continue
        print(f"  {prop:<26} {pp['n']:>5} {pp['cbr_coverage']:>6.3f} "
              f"{pp.get('hit@1',0):>6.3f} {pp.get('hit@3',0):>6.3f} "
              f"{pp.get('hit@5',0):>6.3f} {pp.get('hit@10',0):>6.3f} {pp['mrr']:>7.4f}")

    if llm_results and "faithfulness" in llm_results:
        print(f"\n  Fidelidad LLM ({llm_results['n_evaluated']} incidencias):")
        print(f"    faithfulness      : {llm_results['faithfulness']:.4f}")
        print(f"    full_faithfulness : {llm_results['full_faithfulness']:.4f}")

    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Guardado de resultados
# ---------------------------------------------------------------------------

def _save_results(out_dir: Path, kge_results: dict,
                  prediction_rows: list, llm_results: dict | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON con todo
    combined = {
        "kge": {k: v for k, v in kge_results.items() if k != "per_property"},
        "kge_per_property": kge_results.get("per_property", {}),
        "llm": llm_results or {},
    }
    json_path = out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    # CSV por propiedad
    pp_path = out_dir / "per_property.csv"
    with open(pp_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["property", "n", "cbr_coverage", "hit@1", "hit@3", "hit@5", "hit@10", "mrr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for prop in INCIDENT_PROPS:
            pp = kge_results["per_property"].get(prop)
            if not pp:
                continue
            writer.writerow({
                "property":     prop,
                "n":            pp["n"],
                "cbr_coverage": pp["cbr_coverage"],
                "hit@1":        pp.get("hit@1", 0),
                "hit@3":        pp.get("hit@3", 0),
                "hit@5":        pp.get("hit@5", 0),
                "hit@10":       pp.get("hit@10", 0),
                "mrr":          pp["mrr"],
            })

    # CSV de predicciones individuales
    pred_path = out_dir / "predictions.csv"
    if prediction_rows:
        with open(pred_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()))
            writer.writeheader()
            writer.writerows(prediction_rows)

    # CSV fidelidad LLM
    if llm_results and "rows" in llm_results and llm_results["rows"]:
        llm_path = out_dir / "llm_faithfulness.csv"
        with open(llm_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(llm_results["rows"][0].keys()))
            writer.writeheader()
            writer.writerows(llm_results["rows"])
        print(f"  LLM fidelidad → {llm_path}")

    print(f"  Resultados JSON  → {json_path}")
    print(f"  Por propiedad    → {pp_path}")
    print(f"  Predicciones     → {pred_path}")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(
    kge_model_name: str = "DistMult",
    n_samples: int | None = None,
    use_llm: bool = False,
    llm_model_name: str = cfg.DEFAULT_MODEL,
    n_llm_samples: int = 50,
) -> dict:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.EVAL_DIR / "incident_creator" / ts

    # Evaluación KGE
    kge_results, prediction_rows = evaluate_kge(
        model_name=kge_model_name,
        n_samples=n_samples,
    )

    # Evaluación LLM (opcional)
    llm_results = None
    if use_llm:
        llm_results = evaluate_llm_faithfulness(
            model_name=kge_model_name,
            llm_model_name=llm_model_name,
            n_samples=n_llm_samples,
        )

    _print_results(kge_results, llm_results)
    _save_results(out_dir, kge_results, prediction_rows, llm_results)

    return {"kge": kge_results, "llm": llm_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación del pipeline de creación guiada de incidencias"
    )
    parser.add_argument("--kge-model", default="DistMult",
                        help=f"Modelo KGE (default: DistMult). Opciones: {cfg.KGE_MODELS}")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Nº de incidencias del test a evaluar (default: todas)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Omitir evaluación de fidelidad LLM")
    parser.add_argument("--llm-model", default=cfg.DEFAULT_MODEL,
                        help=f"Modelo LLM (default: {cfg.DEFAULT_MODEL})")
    parser.add_argument("--n-llm-samples", type=int, default=50,
                        help="Nº de incidencias para medir fidelidad LLM (default: 50)")
    args = parser.parse_args()

    run(
        kge_model_name=args.kge_model,
        n_samples=args.n_samples,
        use_llm=not args.no_llm,
        llm_model_name=args.llm_model,
        n_llm_samples=args.n_llm_samples,
    )
