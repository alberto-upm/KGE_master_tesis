"""
Orquestador del pipeline KGE + LLM.

Ejecuta las fases del pipeline en orden o de forma individual.

Uso:
  # Pipeline completo (0 → 1 → 2 → 3 → create_incident)
  python src/run_pipeline.py --phase all

  # Preprocesado: incident_triplets → train_full.ttl + test_eval.ttl (split 95/5)
  python src/run_pipeline.py --phase 0

  # Parseo del grafo: train_full.ttl → train.tsv (+ mapas entidad/relación)
  python src/run_pipeline.py --phase 1

  # Entrenamiento KGE
  python src/run_pipeline.py --phase 2                         # TransE (por defecto)
  python src/run_pipeline.py --phase 2 --kge-model RotatE
  python src/run_pipeline.py --phase 2 --all-models            # entrena todos secuencialmente
  python src/run_pipeline.py --phase 2_plots --kge-model TransE  # regenera loss + t-SNE sin reentrenar

  # Link prediction
  python src/run_pipeline.py --phase 3
  python src/run_pipeline.py --phase 3 --kge-model ComplEx

  # Aprendizaje de reglas Horn con AnyBURL (OPCIONAL · requiere Java · fuera de --phase all)
  python src/run_pipeline.py --phase 4

  # Creación guiada de incidencias (CBR + KGE + LLM)
  python src/run_pipeline.py --phase create_incident
  python src/run_pipeline.py --phase create_incident --no-llm
  python src/run_pipeline.py --phase create_incident --kge-model TransE

  # Construcción del conjunto de evaluación (JSONL en data/evaluacion/)
  # Extrae N incidencias de test_eval.ttl. Campos ausentes → "skip".
  python src/run_pipeline.py --phase build_eval                 # 500 por defecto
  python src/run_pipeline.py --phase build_eval --n 1000

  # Evaluación end-to-end del incident creator (cascada REGLA → KGE+CBR sobre el JSONL)
  # Para cada incidencia del JSONL:
  #   1) intenta la regla: acierto = rule_hit (rank=1); valor distinto = rule_miss → KGE+CBR
  #   2) KGE+CBR: si el valor real está en top-K = kge_hit (con rank); si no = fail
  # Saltos: campos marcados como "skip" en el JSONL.
  # Resultados: out/evaluation/incident_creator_full/<ts>/{results.json, per_property.csv, predictions.csv}
  python src/run_pipeline.py --phase 6
  python src/run_pipeline.py --phase 6 --kge-model TransE
  python src/run_pipeline.py --phase 6 --eval-jsonl data/evaluacion/test_eval_500.jsonl

  # Opciones del modelo KGE
  python src/run_pipeline.py --phase 2 --epochs 50 --dim 64 --device cpu

Dependencias entre fases:
  Phase 0 → Phase 1 → Phase 2 → Phase 3
  Phase 0 → Phase 4 (reglas Horn, opcional)
                            → create_incident (CBR + KGE + LLM)
  build_eval → 6 (evaluación end-to-end)
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# ---------------------------------------------------------------------------
# Tee: duplica stdout+stderr al fichero de log
# ---------------------------------------------------------------------------

class _Tee:
    """Escribe simultáneamente en el stream original y en un fichero."""

    def __init__(self, stream, log_path: Path):
        self._stream   = stream
        self._log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def fileno(self):
        return self._stream.fileno()

    def close(self):
        self._file.close()

    # Delegar cualquier otro atributo al stream original
    def __getattr__(self, name):
        return getattr(self._stream, name)


def _start_logging(phase: str) -> "_Tee | None":
    """Redirige stdout y stderr a out/logs/pipeline_<fecha>_phase<N>.log."""
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = cfg.OUT_DIR / "logs" / f"pipeline_{ts}_phase{phase}.log"
    tee      = _Tee(sys.__stdout__, log_file)
    sys.stdout = tee
    sys.stderr = tee
    print(f"[Log] Guardando traza en: {log_file}")
    return tee


def _stop_logging(tee: "_Tee | None"):
    if tee is None:
        return
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    tee.close()


# ---------------------------------------------------------------------------
# Ejecución de fases
# ---------------------------------------------------------------------------

def run_phase0(test_ratio=None, seed=None):
    from phase0_split import run
    kwargs = {}
    if test_ratio is not None:
        kwargs["test_ratio"] = test_ratio
    if seed is not None:
        kwargs["seed"] = seed
    run(**kwargs)


def run_phase1():
    from phase1_triples import run
    run()


def run_phase2(epochs=None, dim=None, device=cfg.DEVICE, kge_model=None, all_models=False):
    from phase2_kge_train import run
    run(
        model_name=kge_model or 'TransE',
        epochs=epochs, dim=dim, device=device,
        all_models=all_models,
    )


def run_phase2_plots(kge_model=None):
    """Regenera loss curve + t-SNE de un modelo KGE ya entrenado."""
    from phase2_plots import run
    run(kge_model_name=kge_model or 'TransE')


def run_phase3(top_k=None, kge_model=None):
    from phase3_link_prediction import run
    run(top_k=top_k or cfg.TOP_K_PREDICT, model_name=kge_model or 'TransE')


def run_phase4_rules():
    """Aprende reglas Horn con AnyBURL (opcional; requiere Java). Fuera de --phase all."""
    from phase4_learn_rules import run
    run()


def run_create_incident(kge_model=None, llm_model=None, no_llm=False, top_k=10):
    from phase5_incident_creator import run
    run(
        kge_model_name=kge_model or 'TransE',
        use_llm=not no_llm,
        llm_model_name=llm_model or cfg.DEFAULT_MODEL,
        top_k=top_k,
    )


def run_phase6(kge_model=None, top_k=None, eval_jsonl=None):
    """Eval end-to-end del incident creator (cascada REGLA → KGE+CBR sobre data/evaluacion/test_eval_*.jsonl)."""
    from phase6_eval_incident_creator import run, DEFAULT_EVAL_JSONL
    run(
        kge_model_name=kge_model or 'TransE',
        top_k_values=tuple(top_k) if top_k else (1, 3, 5, 10),
        eval_jsonl=Path(eval_jsonl) if eval_jsonl else DEFAULT_EVAL_JSONL,
    )


def run_build_eval(n=500, seed=None, ttl=None, out=None):
    """Construye data/evaluacion/test_eval_<N>.jsonl + resumen de skips en
    out/evaluacion/test_eval_<N>_skips.txt"""
    from phase6_build_eval import build_and_save

    ttl_path = Path(ttl) if ttl else cfg.TEST_TTL
    out_dir  = Path(out) if out else cfg.DATA_DIR / "evaluacion"
    seed_val = seed if seed is not None else cfg.RANDOM_SEED

    build_and_save(ttl_path, n, seed_val, out_dir)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline KGE + LLM para gestión de incidencias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        default="all",
        choices=["all", "0", "1", "2", "2_plots", "3", "4", "6",
                 "build_eval", "create_incident"],
        help="Fase a ejecutar (default: all)",
    )
    parser.add_argument("--test-ratio", type=float, default=None,
                        help="Fracción de incidencias para test (default: 0.05, solo phase 0)")
    # Opciones Phase 2 — entrenamiento KGE
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Épocas de entrenamiento (default: {cfg.N_EPOCHS})")
    parser.add_argument("--dim",    type=int, default=None,
                        help=f"Dimensión de embeddings (default: {cfg.EMBEDDING_DIM})")
    parser.add_argument("--device", default=cfg.DEVICE, choices=["cpu", "cuda"],
                        help=f"Dispositivo PyTorch (default: auto-detectado → {cfg.DEVICE})")
    parser.add_argument("--kge-model", default=None,
                        help=f"Modelo KGE (default: TransE). Opciones: {cfg.KGE_MODELS}")
    parser.add_argument("--all-models", action="store_true",
                        help=f"Entrenar todos los modelos: {cfg.KGE_MODELS} (solo phase 2)")
    # Opciones Phase 3
    parser.add_argument("--top-k",  type=int, default=None,
                        help=f"Top-k en link prediction (default: {cfg.TOP_K_PREDICT})")
    # Opciones create_incident
    parser.add_argument("--model",       default=None,
                        help=f"Modelo HuggingFace para LLM (default: {cfg.DEFAULT_MODEL})")
    parser.add_argument("--no-llm", action="store_true",
                        help="Desactivar LLM (solo KGE)")
    # Opciones build_eval / 6
    parser.add_argument("--n",           type=int, default=500,
                        help="Nº de incidencias para build_eval (default: 500)")
    parser.add_argument("--seed",        type=int, default=None,
                        help=f"Semilla para build_eval (default: {cfg.RANDOM_SEED})")
    parser.add_argument("--eval-jsonl",  default=None,
                        help="JSONL de evaluación para --phase 6 "
                             "(default: data/evaluacion/test_eval_500.jsonl)")

    args = parser.parse_args()
    phase = args.phase

    tee = _start_logging(phase)
    start_total = time.time()
    print(f"\n{'='*60}")
    print(f"  Pipeline KGE + LLM  —  Fase(s): {phase.upper()}")
    print(f"{'='*60}\n")

    phases_to_run = (
        ["0", "1", "2", "3", "create_incident"] if phase == "all" else [phase]
    )

    for p in phases_to_run:
        t0 = time.time()
        if p == "0":
            run_phase0(test_ratio=args.test_ratio)
        elif p == "1":
            run_phase1()
        elif p == "2":
            run_phase2(
                epochs=args.epochs, dim=args.dim, device=args.device,
                kge_model=args.kge_model, all_models=args.all_models,
            )
        elif p == "2_plots":
            run_phase2_plots(kge_model=args.kge_model)
        elif p == "3":
            run_phase3(top_k=args.top_k, kge_model=args.kge_model)
        elif p == "4":
            run_phase4_rules()
        elif p == "6":
            run_phase6(
                kge_model=args.kge_model,
                top_k=args.top_k,
                eval_jsonl=args.eval_jsonl,
            )
        elif p == "build_eval":
            run_build_eval(n=args.n, seed=args.seed)
        elif p == "create_incident":
            run_create_incident(
                kge_model=args.kge_model,
                llm_model=args.model,
                no_llm=args.no_llm,
            )
        elapsed = time.time() - t0
        print(f"\n  [Fase {p}] completada en {elapsed:.1f}s\n")

    total = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"  Pipeline finalizado en {total:.1f}s")
    print(f"{'='*60}\n")
    _stop_logging(tee)


if __name__ == "__main__":
    main()
