"""
Orquestador del pipeline KGE + LLM.

Ejecuta las fases del pipeline en orden o de forma individual.

Uso:
  # Pipeline completo
  python src/run_pipeline.py --phase all

  # Solo una fase
  python src/run_pipeline.py --phase 1             # parseo TTL → TSV
  python src/run_pipeline.py --phase 2             # entrenamiento DistMult (por defecto)
  python src/run_pipeline.py --phase 2 --kge-model TransE
  python src/run_pipeline.py --phase 2 --all-models   # entrena TransE+DistMult+ComplEx
  python src/run_pipeline.py --phase 3             # link prediction (DistMult)
  python src/run_pipeline.py --phase 3 --kge-model ComplEx
  python src/run_pipeline.py --phase 4             # demo LLM (no interactivo)
  python src/run_pipeline.py --phase 5             # (sin ejecución standalone)
  python src/run_pipeline.py --phase 6             # evaluación completa Q&A
  python src/run_pipeline.py --phase compare       # comparación de modelos KGE
  python src/run_pipeline.py --phase interactive_query  # sesión GLiNER+KGE+LLM

  # Sesión interactiva con LLM (phase4)
  python src/run_pipeline.py --phase 4 --interactive
  python src/run_pipeline.py --phase 4 --interactive --incident incident_XYZ

  # Sesión interactiva nueva (GLiNER + KGE + LLM)
  python src/run_pipeline.py --phase interactive_query --kge-model DistMult
  python src/run_pipeline.py --phase interactive_query --no-llm

  # Comparación de modelos KGE
  python src/run_pipeline.py --phase compare --n-samples 200
  python src/run_pipeline.py --phase compare --verbalization-check

  # Opciones del modelo KGE
  python src/run_pipeline.py --phase 2 --epochs 50 --dim 64 --device cpu

  # Opciones del LLM
  python src/run_pipeline.py --phase 6 --model google/flan-t5-large --n-samples 100

Dependencias entre fases:
  Phase 1 → Phase 2 → Phase 3
                   → Phase 4 (puede funcionar sin phase 3)
                   → Phase 5 (usa artefactos de phase 2)
                   → Phase 6 (usa phase 4 + phase 5)
                   → compare (requiere phase 2 para todos los modelos)
                   → interactive_query (requiere phase 2 para el modelo elegido)
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
        model_name=kge_model or 'DistMult',
        epochs=epochs, dim=dim, device=device,
        all_models=all_models,
    )


def run_phase3(top_k=None, kge_model=None):
    from phase3_link_prediction import run
    run(top_k=top_k or cfg.TOP_K_PREDICT, model_name=kge_model or 'DistMult')


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


def run_interactive_query(kge_model=None, llm_model=None, no_llm=False, log=None):
    from interactive_query import interactive_query_loop
    interactive_query_loop(
        kge_model_name=kge_model or 'DistMult',
        llm_model_name=llm_model or cfg.DEFAULT_MODEL,
        use_llm=not no_llm,
        log_path=Path(log) if log else None,
    )


def run_phase4(model_name=None, device=cfg.DEVICE, interactive=False, incident_id=""):
    from phase4_llm_inference import run
    run(
        model_name=model_name or cfg.DEFAULT_MODEL,
        device=device,
        interactive=interactive,
        incident_id=incident_id,
    )


def run_phase4_2(model_name=None, device=cfg.DEVICE, interactive=False, incident_id=""):
    from phase4_2_llm_inference import run
    run(
        model_name=model_name or cfg.DEFAULT_MODEL,
        device=device,
        interactive=interactive,
        incident_id=incident_id,
    )


def run_phase5():
    """Phase 5 no tiene ejecución standalone; es una librería usada por phase4 y phase6."""
    print("La fase 5 (subgrafo de configuración) es una librería usada por las fases 4 y 6.")
    print("No tiene ejecución standalone. Ejecuta las fases 4 o 6 para activarla.")


def run_phase6(n_samples=None, n_chains=None, model_name=None, device=cfg.DEVICE):
    from phase6_validation import run
    run(
        n_samples=n_samples  or cfg.EVAL_SAMPLE_N,
        n_chains=n_chains    or cfg.EVAL_SAMPLE_N,
        model_name=model_name or cfg.DEFAULT_MODEL,
        device=device,
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
        choices=["all", "1", "2", "3", "4", "4_2", "5", "6",
                 "compare", "interactive_query"],
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
                        help=f"Modelo KGE (default: DistMult). Opciones: {cfg.KGE_MODELS}")
    parser.add_argument("--all-models", action="store_true",
                        help=f"Entrenar todos los modelos: {cfg.KGE_MODELS} (solo phase 2)")
    # Opciones Phase 3
    parser.add_argument("--top-k",  type=int, default=None,
                        help=f"Top-k en link prediction (default: {cfg.TOP_K_PREDICT})")
    # Opciones Phase 4 / interactive_query
    parser.add_argument("--model",       default=None,
                        help=f"Modelo HuggingFace para LLM (default: {cfg.DEFAULT_MODEL})")
    parser.add_argument("--interactive", action="store_true",
                        help="Activar sesión interactiva Q&A (solo phase 4)")
    parser.add_argument("--incident",    default="",
                        help="ID de incidencia para sesión interactiva (phase 4)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Desactivar LLM en interactive_query (solo KGE)")
    # Opciones Phase 6
    parser.add_argument("--n-samples",   type=int, default=None,
                        help=f"Nº de preguntas a evaluar (default: {cfg.EVAL_SAMPLE_N})")
    parser.add_argument("--n-chains",    type=int, default=None,
                        help=f"Nº de cadenas multi-hop a evaluar (default: {cfg.EVAL_SAMPLE_N})")
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
        ["1", "2", "3", "4", "6"] if phase == "all" else [phase]
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
        elif p == "4":
            run_phase4(
                model_name=args.model,
                device=args.device,
                interactive=args.interactive,
                incident_id=args.incident,
            )
        elif p == "4_2":
            run_phase4_2(
                model_name=args.model,
                device=args.device,
                interactive=args.interactive,
                incident_id=args.incident,
            )
        elif p == "5":
            run_phase5()
        elif p == "6":
            run_phase6(
                n_samples=args.n_samples,
                n_chains=args.n_chains,
                model_name=args.model,
                device=args.device,
            )
        elif p == "compare":
            run_model_comparison(
                n_samples=args.n_samples,
                verbalization_check=args.verbalization_check,
                n_verb=args.n_verb,
                verb_model=args.kge_model or 'DistMult',
            )
        elif p == "interactive_query":
            run_interactive_query(
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
