"""
Orquestador del pipeline KGE + LLM.

Ejecuta las fases del pipeline en orden o de forma individual.

Uso:
  # Pipeline completo
  python src/run_pipeline.py --phase all

  # Solo una fase
  python src/run_pipeline.py --phase 1                # parseo TTL → TSV (split por incidencias)
  python src/run_pipeline.py --phase 2                # entrenamiento DistMult (por defecto)
  python src/run_pipeline.py --phase 2 --kge-model TransE
  python src/run_pipeline.py --phase 2 --all-models   # entrena TransE+DistMult+ComplEx
  python src/run_pipeline.py --phase 3                # link prediction (DistMult)
  python src/run_pipeline.py --phase 3 --kge-model ComplEx
  python src/run_pipeline.py --phase 5                # (sin ejecución standalone)
  python src/run_pipeline.py --phase compare          # comparación de modelos KGE

  # Creación guiada de incidencias (CBR + KGE + LLM)
  python src/run_pipeline.py --phase create_incident
  python src/run_pipeline.py --phase create_incident --no-llm
  python src/run_pipeline.py --phase create_incident --kge-model TransE

  # Evaluación del incident creator
  python src/run_pipeline.py --phase 6                          # eval completa
  python src/run_pipeline.py --phase 6 --n-samples 100          # menos muestras
  python src/run_pipeline.py --phase 6 --kge-model TransE

  # Comparación de modelos KGE
  python src/run_pipeline.py --phase compare --n-samples 200
  python src/run_pipeline.py --phase compare --verbalization-check

  # Opciones del modelo KGE
  python src/run_pipeline.py --phase 2 --epochs 50 --dim 64 --device cpu

Dependencias entre fases:
  Phase 1 → Phase 2 → Phase 3
                   → create_incident (CBR + KGE + LLM)
                   → Phase 6 (evaluación del incident creator)
                   → compare (requiere phase 2 para todos los modelos)
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


def run_phase3(top_k=None, kge_model=None):
    from phase3_link_prediction import run
    run(top_k=top_k or cfg.TOP_K_PREDICT, model_name=kge_model or 'TransE')


def run_model_comparison(models=None, n_samples=None,
                         verbalization_check=False, n_verb=50, verb_model='DistMult'):
    from phase6_model_comparison import run
    run(
        models=models,
        n_samples=n_samples,
        verbalization_check=verbalization_check,
        n_verb=n_verb,
        verb_model=verb_model,
    )


def run_create_incident(kge_model=None, llm_model=None, no_llm=False, top_k=5):
    from phase4_incident_creator import run
    run(
        kge_model_name=kge_model or 'TransE',
        use_llm=not no_llm,
        llm_model_name=llm_model or cfg.DEFAULT_MODEL,
        top_k=top_k,
    )


def run_phase5():
    """Phase 5 no tiene ejecución standalone; es una librería usada por otros módulos."""
    print("La fase 5 (subgrafo de configuración) es una librería.")
    print("No tiene ejecución standalone.")


def run_phase6(kge_model=None, n_samples=None, use_llm=False, llm_model=None):
    from phase6_incident_creator_eval import run
    run(
        kge_model_name=kge_model or 'TransE',
        n_samples=n_samples,
        use_llm=use_llm,
        llm_model_name=llm_model or cfg.DEFAULT_MODEL,
    )


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
        choices=["all", "1", "2", "3", "5", "6",
                 "compare", "create_incident"],
        help="Fase a ejecutar (default: all)",
    )
    # Opciones Phase 2 — entrenamiento KGE
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Épocas de entrenamiento (default: {cfg.N_EPOCHS})")
    parser.add_argument("--dim",    type=int, default=None,
                        help=f"Dimensión de embeddings (default: {cfg.EMBEDDING_DIM})")
    parser.add_argument("--device", default=cfg.DEVICE, choices=["cpu", "cuda"],
                        help=f"Dispositivo PyTorch (default: auto-detectado → {cfg.DEVICE})")
    parser.add_argument("--kge-model", default=None,
                        help=f"Modelo KGE (default: TransE). Opciones: {cfg.KGE_MODELS}")
    parser.add_argument("--kge-models", nargs="+", default=None,
                        help=f"Modelos KGE a comparar (default: todos). Ej: --kge-models TransE DistMult")
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
    # Opciones Phase 6
    parser.add_argument("--n-samples",   type=int, default=None,
                        help=f"Nº de incidencias a evaluar (default: {cfg.EVAL_SAMPLE_N})")
    # Opciones compare
    parser.add_argument("--verbalization-check", action="store_true",
                        help="Verificar integridad de verbalización (solo phase compare)")
    parser.add_argument("--n-verb", type=int, default=50,
                        help="Muestras para verificación de verbalización (default: 50)")

    args = parser.parse_args()
    phase = args.phase

    tee = _start_logging(phase)
    start_total = time.time()
    print(f"\n{'='*60}")
    print(f"  Pipeline KGE + LLM  —  Fase(s): {phase.upper()}")
    print(f"{'='*60}\n")

    phases_to_run = (
        ["1", "2", "3", "create_incident", "6"] if phase == "all" else [phase]
    )

    for p in phases_to_run:
        t0 = time.time()
        if p == "1":
            run_phase1()
        elif p == "2":
            run_phase2(
                epochs=args.epochs, dim=args.dim, device=args.device,
                kge_model=args.kge_model, all_models=args.all_models,
            )
        elif p == "3":
            run_phase3(top_k=args.top_k, kge_model=args.kge_model)
        elif p == "5":
            run_phase5()
        elif p == "6":
            run_phase6(
                kge_model=args.kge_model,
                n_samples=args.n_samples,
                use_llm=not args.no_llm,
                llm_model=args.model,
            )
        elif p == "compare":
            run_model_comparison(
                models=args.kge_models,
                n_samples=args.n_samples,
                verbalization_check=args.verbalization_check,
                n_verb=args.n_verb,
                verb_model=args.kge_model or 'TransE',
            )
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
