"""
Orquestador del pipeline KGE + LLM.

Ejecuta las fases del pipeline en orden o de forma individual.

Uso:
  # Pipeline completo
  python src/run_pipeline.py --phase all

  # Solo una fase
  python src/run_pipeline.py --phase 1          # parseo TTL → TSV
  python src/run_pipeline.py --phase 2          # entrenamiento DistMult
  python src/run_pipeline.py --phase 3          # link prediction
  python src/run_pipeline.py --phase 4          # demo LLM (no interactivo)
  python src/run_pipeline.py --phase 5          # (sin ejecución standalone)
  python src/run_pipeline.py --phase 6          # evaluación completa

  # Sesión interactiva con LLM
  python src/run_pipeline.py --phase 4 --interactive
  python src/run_pipeline.py --phase 4 --interactive --incident incident_XYZ

  # Opciones del modelo KGE
  python src/run_pipeline.py --phase 2 --epochs 50 --dim 64 --device cpu

  # Opciones del LLM
  python src/run_pipeline.py --phase 6 --model google/flan-t5-large --n-samples 100

Dependencias entre fases:
  Phase 1 → Phase 2 → Phase 3
                   → Phase 4 (puede funcionar sin phase 3)
                   → Phase 5 (usa artefactos de phase 2)
                   → Phase 6 (usa phase 4 + phase 5)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# ---------------------------------------------------------------------------
# Ejecución de fases
# ---------------------------------------------------------------------------

def run_phase1():
    from phase1_triples import run
    run()


def run_phase2(epochs=None, dim=None, device="cpu"):
    from phase2_kge_train import run
    run(epochs=epochs, dim=dim, device=device)


def run_phase3(top_k=None):
    from phase3_link_prediction import run
    run(top_k=top_k or cfg.TOP_K_PREDICT)


def run_phase4(model_name=None, device="cpu", interactive=False, incident_id=""):
    from phase4_llm_inference import run
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


def run_phase6(n_samples=None, model_name=None, device="cpu"):
    from phase6_validation import run
    run(
        n_samples=n_samples or cfg.EVAL_SAMPLE_N,
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
        choices=["all", "1", "2", "3", "4", "5", "6"],
        help="Fase a ejecutar (default: all)",
    )
    # Opciones Phase 2
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Épocas de entrenamiento (default: {cfg.N_EPOCHS})")
    parser.add_argument("--dim",    type=int, default=None,
                        help=f"Dimensión de embeddings (default: {cfg.EMBEDDING_DIM})")
    parser.add_argument("--device", default=cfg.DEVICE, choices=["cpu", "cuda"],
                        help=f"Dispositivo PyTorch (default: auto-detectado → {cfg.DEVICE})")
    # Opciones Phase 3
    parser.add_argument("--top-k",  type=int, default=None,
                        help=f"Top-k en link prediction (default: {cfg.TOP_K_PREDICT})")
    # Opciones Phase 4
    parser.add_argument("--model",       default=None,
                        help=f"Modelo HuggingFace (default: {cfg.DEFAULT_MODEL})")
    parser.add_argument("--interactive", action="store_true",
                        help="Activar sesión interactiva Q&A (solo phase 4)")
    parser.add_argument("--incident",    default="",
                        help="ID de incidencia para sesión interactiva")
    # Opciones Phase 6
    parser.add_argument("--n-samples",   type=int, default=None,
                        help=f"Nº de muestras a evaluar (default: {cfg.EVAL_SAMPLE_N})")

    args = parser.parse_args()
    phase = args.phase

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
            run_phase2(epochs=args.epochs, dim=args.dim, device=args.device)
        elif p == "3":
            run_phase3(top_k=args.top_k)
        elif p == "4":
            run_phase4(
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
                model_name=args.model,
                device=args.device,
            )
        elapsed = time.time() - t0
        print(f"\n  [Fase {p}] completada en {elapsed:.1f}s\n")

    total = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"  Pipeline finalizado en {total:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
