"""
Fase 6 — Validación del pipeline completo: NL → GLiNER2 → KGE → LLM.

Flujo por muestra (usando link_prediction_eval.json como corpus de test):

  1. Pregunta en lenguaje natural  (e.g. "¿De qué tipo es la incidencia incident_X?")
  2. GLiNER2 extrae (sujeto, predicado)
  3. KGE hace link prediction → objeto predicho (e.g. typeIncident__2)
  4. LLM verbaliza la predicción en lenguaje natural
  5. Se comprueba que la verbalización del LLM preserva el objeto predicho (integridad)
  6. Se comprueba que el objeto predicho coincide con object_true (exactitud)

Métricas por sección:

  EXTRACCIÓN (GLiNER2)
    - subject_accuracy  : fracción donde el sujeto extraído == subject correcto
    - predicate_accuracy: fracción donde el predicado extraído == predicate correcto
    - full_accuracy     : ambos correctos a la vez

  LINK PREDICTION (KGE)
    - hit@1, hit@3, hit@10, mrr  (sobre los casos donde extracción fue correcta)
    - hit@1_full, ...            (sobre todos los casos, incluyendo fallos GLiNER)

  VERBALIZACIÓN (LLM)  — opcional, requiere servidor vLLM
    - integrity         : fracción donde LLM preserva el identificador predicho
    - end_to_end        : fracción donde LLM preserva object_true

Salida:
  out/evaluation/pipeline/<timestamp>/results.json
  out/evaluation/pipeline/<timestamp>/predictions.csv

Uso:
  python src/phase6_pipeline_validation.py
  python src/phase6_pipeline_validation.py --kge-model TransE
  python src/phase6_pipeline_validation.py --no-llm          (solo GLiNER + KGE)
  python src/phase6_pipeline_validation.py --n-samples 200
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from generate_corpus import PRED_TEMPLATES_ES


# ---------------------------------------------------------------------------
# Verbalización de predicción KGE
# ---------------------------------------------------------------------------

def _verbalize(subject: str, predicate: str, obj: str) -> str:
    tmpl = PRED_TEMPLATES_ES.get(predicate)
    if tmpl:
        return tmpl.format(s=subject, p=predicate, o=obj)
    return f"La incidencia {subject} tiene {predicate}: {obj}."


# ---------------------------------------------------------------------------
# Evaluación end-to-end
# ---------------------------------------------------------------------------

def run_pipeline_validation(
    kge_model_name: str  = 'DistMult',
    n_samples:      int  = None,
    use_llm:        bool = True,
    llm_base_url:   str  = cfg.VLLM_BASE_URL,
    llm_model_name: str  = cfg.DEFAULT_MODEL,
    seed:           int  = cfg.RANDOM_SEED,
    top_k_values:   list = None,
) -> dict:

    if top_k_values is None:
        top_k_values = cfg.HIT_K_VALUES

    print("=" * 60)
    print(f"FASE 6 — Validación pipeline completo ({kge_model_name})")
    print("=" * 60)

    # ── Cargar corpus ──────────────────────────────────────────────────────
    if not cfg.LP_EVAL_CORPUS.exists():
        raise FileNotFoundError(
            f"Corpus no encontrado: {cfg.LP_EVAL_CORPUS}\n"
            "Ejecuta primero:  python src/generate_corpus.py --lp-only"
        )
    with open(cfg.LP_EVAL_CORPUS, encoding="utf-8") as f:
        corpus = json.load(f)

    rng = random.Random(seed)
    if n_samples and n_samples < len(corpus):
        corpus = rng.sample(corpus, n_samples)

    print(f"  Corpus    : {len(corpus)} entradas")
    print(f"  Modelo KGE: {kge_model_name}")
    print(f"  LLM       : {'activado (' + llm_model_name + ')' if use_llm else 'desactivado'}\n")

    # ── Cargar componentes ─────────────────────────────────────────────────
    print("[1/3] Cargando GLiNER2 y modelo KGE ...")
    from gliner_extractor import GLiNERExtractor
    from phase3_link_prediction import load_model_by_name, predict_tails

    extractor = GLiNERExtractor()
    kge_model, factory = load_model_by_name(kge_model_name)

    llm = None
    if use_llm:
        try:
            from phase4_llm_inference import KGEAugmentedLLM, extract_answer
            llm = KGEAugmentedLLM(model_name=llm_model_name, base_url=llm_base_url)
            print(f"  LLM conectado a {llm_base_url}")
        except Exception as e:
            print(f"  [Aviso] No se pudo conectar al LLM: {e}  (continuando sin LLM)")
            use_llm = False

    # ── Contadores globales ────────────────────────────────────────────────
    max_k = max(top_k_values)

    # Extracción
    subj_ok = pred_ok = full_ok = 0

    # KGE (sobre todos los casos)
    hits_all  = {k: 0 for k in top_k_values}
    rr_all    = 0.0
    # KGE (solo cuando extracción fue correcta)
    hits_ext  = {k: 0 for k in top_k_values}
    rr_ext    = 0.0
    n_ext_ok  = 0

    # LLM
    integrity_ok  = 0   # LLM preserva objeto predicho
    end_to_end_ok = 0   # LLM preserva object_true
    n_llm_checked = 0

    # Por predicado
    per_pred: dict = defaultdict(lambda: {
        "n": 0, "subj_ok": 0, "pred_ok": 0,
        **{f"hit@{k}": 0 for k in top_k_values},
        "rr_sum": 0.0,
    })

    predictions_log = []

    print(f"\n[2/3] Evaluando {len(corpus)} muestras ...\n")

    for i, entry in enumerate(corpus):
        question   = entry["question"]
        true_subj  = entry["subject"]
        true_pred  = entry["predicate"]
        true_obj   = entry["object_true"]

        # ── PASO 1: Extracción GLiNER2 ─────────────────────────────────────
        extraction = extractor.extract(question)
        ext_subj   = extraction["head"]
        ext_pred   = extraction["relation"]

        s_ok = (ext_subj == true_subj)
        p_ok = (ext_pred == true_pred)
        f_ok = s_ok and p_ok

        subj_ok += int(s_ok)
        pred_ok += int(p_ok)
        full_ok += int(f_ok)

        per_pred[true_pred]["n"]       += 1
        per_pred[true_pred]["subj_ok"] += int(s_ok)
        per_pred[true_pred]["pred_ok"] += int(p_ok)

        # ── PASO 2: Link prediction ────────────────────────────────────────
        # Usamos la extracción real de GLiNER (puede ser incorrecta)
        # para simular el pipeline tal como lo verá el usuario.
        # Si la extracción falló usamos (true_subj, true_pred) para calcular
        # Hit@k_full y lo marcamos como fallo de extracción.
        head_for_lp     = ext_subj or true_subj
        relation_for_lp = ext_pred or true_pred

        preds = predict_tails(
            kge_model, factory,
            head_for_lp, relation_for_lp,
            top_k=max_k,
        )
        pred_entities = [e for e, _ in preds]
        top1_entity   = pred_entities[0] if pred_entities else None

        rank_all = (pred_entities.index(true_obj) + 1) if true_obj in pred_entities else None
        rr_v     = (1.0 / rank_all) if rank_all else 0.0
        rr_all  += rr_v

        for k in top_k_values:
            hit = 1 if (rank_all and rank_all <= k) else 0
            hits_all[k] += hit
            per_pred[true_pred][f"hit@{k}"] += hit
        per_pred[true_pred]["rr_sum"] += rr_v

        # KGE sobre extracción correcta
        if f_ok:
            n_ext_ok += 1
            rr_ext   += rr_v
            for k in top_k_values:
                hits_ext[k] += 1 if (rank_all and rank_all <= k) else 0

        # ── PASO 3: Verbalización LLM (opcional) ──────────────────────────
        verbalized   = _verbalize(true_subj, true_pred, top1_entity or "?")
        llm_answer   = None
        integrity    = None
        end_to_end   = None

        if use_llm and llm is not None and top1_entity:
            from phase4_llm_inference import extract_answer
            llm_question = (
                f"¿Cuál es el {true_pred} de la incidencia {true_subj}? "
                "Responde solo con el identificador exacto."
            )
            try:
                llm_raw    = llm.answer([verbalized], llm_question, do_extract=False)
                llm_answer = extract_answer(llm_raw)
                integrity  = top1_entity.lower() in llm_answer.lower()
                end_to_end = true_obj.lower()   in llm_answer.lower()
                integrity_ok  += int(integrity)
                end_to_end_ok += int(end_to_end)
                n_llm_checked += 1
            except Exception:
                pass

        # ── Log ──────────────────────────────────────────────────────────
        status = "✓" if (rank_all == 1) else "✗"
        print(f"  [{i+1:>4}/{len(corpus)}] {status}  "
              f"gliner={'✓' if f_ok else '✗'}  "
              f"rank={rank_all or '—':>3}  "
              f"pred={entry['predicate']:<24}  "
              f"top1={top1_entity or '—'}")

        predictions_log.append({
            "id":             entry["id"],
            "subject":        true_subj,
            "predicate":      true_pred,
            "object_true":    true_obj,
            "question":       question,
            # Extracción
            "ext_subject":    ext_subj,
            "ext_predicate":  ext_pred,
            "subject_ok":     s_ok,
            "predicate_ok":   p_ok,
            "extraction_ok":  f_ok,
            # KGE
            "top1_predicted": top1_entity,
            "rank":           rank_all,
            **{f"hit@{k}": int(bool(rank_all and rank_all <= k)) for k in top_k_values},
            # LLM
            "verbalized":     verbalized,
            "llm_answer":     llm_answer,
            "integrity":      integrity,
            "end_to_end":     end_to_end,
        })

    # ── Métricas globales ──────────────────────────────────────────────────
    n = len(corpus)

    extraction_metrics = {
        "subject_accuracy":   round(subj_ok / n, 4),
        "predicate_accuracy": round(pred_ok / n, 4),
        "full_accuracy":      round(full_ok / n, 4),
    }

    kge_metrics_all = {
        "note":  "link prediction usando extracción GLiNER (puede incluir fallos)",
        "n":     n,
        "mrr":   round(rr_all / n, 4),
        **{f"hit@{k}": round(hits_all[k] / n, 4) for k in top_k_values},
    }

    kge_metrics_ext = {
        "note":  "link prediction solo cuando GLiNER extrajo correctamente",
        "n":     n_ext_ok,
        "mrr":   round(rr_ext / n_ext_ok, 4) if n_ext_ok else 0.0,
        **{f"hit@{k}": round(hits_ext[k] / n_ext_ok, 4) if n_ext_ok else 0.0
           for k in top_k_values},
    }

    llm_metrics = {
        "n_checked":   n_llm_checked,
        "integrity":   round(integrity_ok  / n_llm_checked, 4) if n_llm_checked else None,
        "end_to_end":  round(end_to_end_ok / n_llm_checked, 4) if n_llm_checked else None,
    } if use_llm else {"note": "LLM desactivado"}

    per_pred_metrics = {}
    for pred, stats in sorted(per_pred.items()):
        pn = stats["n"]
        per_pred_metrics[pred] = {
            "n":                  pn,
            "subject_accuracy":   round(stats["subj_ok"] / pn, 4),
            "predicate_accuracy": round(stats["pred_ok"] / pn, 4),
            "mrr":                round(stats["rr_sum"]  / pn, 4),
            **{f"hit@{k}": round(stats[f"hit@{k}"] / pn, 4) for k in top_k_values},
        }

    results = {
        "kge_model":    kge_model_name,
        "n_evaluated":  n,
        "extraction":   extraction_metrics,
        "kge_all":      kge_metrics_all,
        "kge_exact":    kge_metrics_ext,
        "llm":          llm_metrics,
        "per_predicate": per_pred_metrics,
    }

    # ── Guardar ────────────────────────────────────────────────────────────
    print("\n[3/3] Guardando resultados ...")
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.EVAL_DIR / "pipeline" / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # CSV predicciones
    pred_csv = run_dir / "predictions.csv"
    if predictions_log:
        fieldnames = list(predictions_log[0].keys())
        with open(pred_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(predictions_log)

    # ── Imprimir resumen ───────────────────────────────────────────────────
    _print_summary(results, top_k_values)
    print(f"\n  Directorio → {run_dir}")
    print(f"  Métricas   → {run_dir / 'results.json'}")
    print(f"  Detalle    → {pred_csv}")
    print("\n✓ Validación completada.")

    return results


def _print_summary(results: dict, top_k_values: list) -> None:
    ext = results["extraction"]
    kge = results["kge_all"]
    kge_e = results["kge_exact"]
    llm = results["llm"]

    print("\n" + "=" * 60)
    print("  RESULTADOS — Extracción GLiNER2")
    print("=" * 60)
    print(f"  Sujeto correcto    : {ext['subject_accuracy']:.2%}")
    print(f"  Predicado correcto : {ext['predicate_accuracy']:.2%}")
    print(f"  Ambos correctos    : {ext['full_accuracy']:.2%}")

    k_header = "  ".join(f"Hit@{k}" for k in top_k_values)
    print(f"\n{'=' * 60}")
    print(f"  RESULTADOS — Link Prediction KGE (todos los casos, n={kge['n']})")
    print(f"{'=' * 60}")
    print(f"  {k_header}   MRR")
    vals = "  ".join(f"{kge[f'hit@{k}']:.4f}" for k in top_k_values)
    print(f"  {vals}   {kge['mrr']:.4f}")

    print(f"\n{'=' * 60}")
    print(f"  RESULTADOS — Link Prediction KGE (solo extracción correcta, n={kge_e['n']})")
    print(f"{'=' * 60}")
    vals_e = "  ".join(f"{kge_e[f'hit@{k}']:.4f}" for k in top_k_values)
    print(f"  {vals_e}   {kge_e['mrr']:.4f}")

    if llm.get("n_checked"):
        print(f"\n{'=' * 60}")
        print(f"  RESULTADOS — Verbalización LLM (n={llm['n_checked']})")
        print(f"{'=' * 60}")
        print(f"  Integridad (preserva objeto predicho) : {llm['integrity']:.2%}")
        print(f"  End-to-end (preserva object_true)     : {llm['end_to_end']:.2%}")

    print(f"\n{'=' * 60}")
    print("  DESGLOSE POR PREDICADO")
    print(f"{'=' * 60}")
    per_pred = results["per_predicate"]
    k_cols = "  ".join(f"{'H@'+str(k):>6}" for k in top_k_values)
    print(f"  {'Predicado':<28} {'n':>5}  {'SubjAcc':>7}  {'PredAcc':>7}  {k_cols}  {'MRR':>6}")
    print("  " + "-" * (28 + 5 + 7 + 7 + 8 * len(top_k_values) + 8))
    for pred, m in per_pred.items():
        k_vals = "  ".join(f"{m[f'hit@{k}']:>6.4f}" for k in top_k_values)
        print(f"  {pred:<28} {m['n']:>5}  "
              f"{m['subject_accuracy']:>7.4f}  {m['predicate_accuracy']:>7.4f}  "
              f"{k_vals}  {m['mrr']:>6.4f}")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(
    kge_model_name: str  = 'DistMult',
    n_samples:      int  = None,
    use_llm:        bool = True,
    llm_model_name: str  = cfg.DEFAULT_MODEL,
):
    return run_pipeline_validation(
        kge_model_name=kge_model_name,
        n_samples=n_samples,
        use_llm=use_llm,
        llm_model_name=llm_model_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validación end-to-end: GLiNER2 → KGE → LLM"
    )
    parser.add_argument("--kge-model", default="DistMult",
                        choices=cfg.KGE_MODELS,
                        help="Modelo KGE a usar (default: DistMult)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Nº de muestras del corpus LP (default: todas)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Desactivar verbalización LLM")
    parser.add_argument("--llm-model", default=cfg.DEFAULT_MODEL,
                        help=f"Modelo HuggingFace en vLLM (default: {cfg.DEFAULT_MODEL})")
    args = parser.parse_args()

    run_pipeline_validation(
        kge_model_name=args.kge_model,
        n_samples=args.n_samples,
        use_llm=not args.no_llm,
        llm_model_name=args.llm_model,
    )
