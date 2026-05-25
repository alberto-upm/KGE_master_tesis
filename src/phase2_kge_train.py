"""
Fase 2 — Entrenamiento de modelos KGE con PyKEEN.

Modelos soportados:
  Traslacionales (buenos para jerarquías):
    TransE  — línea base traslacional
    RotatE  — rotaciones en espacio complejo, captura transitividad y asimetría
    TransH  — proyección en hiperplanos por relación
    HAKE    — coordenadas polares, diseñado específicamente para jerarquías
  Bilineales:
    DistMult — producto escalar, simétrico
    ComplEx  — espacio complejo, modela asimetría

Requisito previo: ejecutar phase1_triples.py para generar los TSV.

Salida por modelo (ej. RotatE):
  out/models/rotate/           (modelo PyKEEN completo)
  out/embeddings/rotate/entity_embeddings.pt
  out/embeddings/rotate/relation_embeddings.pt

Uso:
  python src/phase2_kge_train.py                        # DistMult (por defecto)
  python src/phase2_kge_train.py --model RotatE
  python src/phase2_kge_train.py --model HAKE
  python src/phase2_kge_train.py --all-models           # entrena todos secuencialmente
  python src/phase2_kge_train.py --epochs N --dim D --device cpu|cuda
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# ---------------------------------------------------------------------------
# Entrenamiento de un modelo
# ---------------------------------------------------------------------------

def _model_config(model_lower: str, dim: int, margin: float) -> dict:
    """
    Devuelve la configuración de loss, sampler y model_kwargs para cada modelo.

    Modelos traslacionales/rotacionales: NSSALoss + bernoulli.
      - Capturan asimetría y transitividad; clave para relaciones jerárquicas.
      - bernoulli pondera negativos según frecuencia de cabeza/cola (Bernoulli trick).
    Modelos bilineales: BCEWithLogitsLoss + basic.
      - DistMult/ComplEx funcionan mejor con BCE y muestreo uniforme.
    HAKE: NSSALoss + basic (modular + fase; bernoulli añade ruido en el módulo).
    """
    nssa = dict(loss="NSSALoss",
                loss_kwargs=dict(margin=margin, adversarial_temperature=1.0),
                sampler="bernoulli",
                eval_batch_size=32)
    bce  = dict(loss="BCEWithLogitsLoss",
                loss_kwargs={},
                sampler="basic",
                eval_batch_size=32)

    configs = {
        "transe":   {**nssa, "model_kwargs": dict(embedding_dim=dim, scoring_fct_norm=1)},
        # RotatE: embeddings complejos internamente; dim es la dimensión total real
        "rotate":   {**nssa, "model_kwargs": dict(embedding_dim=dim)},
        # TransH: proyecta entidades en hiperplanos específicos por relación
        "transh":   {**nssa, "model_kwargs": dict(embedding_dim=dim)},
        # HAKE: módulo captura nivel jerárquico, fase captura diferencia semántica
        "hake":     {**nssa, "model_kwargs": dict(embedding_dim=dim),
                     "sampler": "basic"},
        "distmult": {**bce,  "model_kwargs": dict(embedding_dim=dim)},
        # ComplEx: embeddings complejos (dim × 2 en RAM) → eval_batch más pequeño
        "complex":  {**bce,  "model_kwargs": dict(embedding_dim=dim),
                     "eval_batch_size": 8},
    }
    if model_lower not in configs:
        # Fallback genérico para cualquier otro modelo de PyKEEN
        return {**bce, "model_kwargs": dict(embedding_dim=dim)}
    return configs[model_lower]


def train(
    model_name:      str   = 'DistMult',
    epochs:          int   = cfg.N_EPOCHS,
    dim:             int   = cfg.EMBEDDING_DIM,
    batch:           int   = cfg.BATCH_SIZE,
    lr:              float = cfg.LEARNING_RATE,
    device:          str   = "cpu",
    eval_batch_size: int   = None,
    margin:          float = 9.0,
):
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    print("=" * 60)
    print(f"FASE 2 — Entrenamiento {model_name} con PyKEEN")
    print("=" * 60)

    if not cfg.TRAIN_TSV.exists():
        raise FileNotFoundError(
            f"No encontrado: {cfg.TRAIN_TSV}\n"
            "Ejecuta primero:  python src/phase1_triples.py"
        )

    print(f"[1/3] Cargando tripletas desde {cfg.TRAIN_TSV} ...")
    full = TriplesFactory.from_path(cfg.TRAIN_TSV)
    print(f"      Total de tripletas cargadas: {full.num_triples:,}")

    training, validation, testing = full.split(
        ratios=[cfg.TRAIN_RATIO, cfg.VALID_RATIO, 1.0 - cfg.TRAIN_RATIO - cfg.VALID_RATIO],
        random_state=cfg.RANDOM_SEED,
    )
    print(f"      Entidades:  {training.num_entities:,}")
    print(f"      Relaciones: {training.num_relations:,}")
    print(f"      Train / Valid / Test: "
          f"{training.num_triples:,} / {validation.num_triples:,} / {testing.num_triples:,}")

    model_lower = model_name.lower()
    mcfg = _model_config(model_lower, dim, margin)

    loss          = mcfg["loss"]
    loss_kwargs   = mcfg["loss_kwargs"]
    model_kwargs  = mcfg["model_kwargs"]
    sampler       = mcfg["sampler"]
    if eval_batch_size is None:
        eval_batch_size = mcfg["eval_batch_size"]

    print(f"\n[2/3] Entrenando {model_name}  "
          f"(dim={dim}, epochs={epochs}, loss={loss}, "
          f"sampler={sampler}, device={device}, eval_batch={eval_batch_size}) ...")
    # print(f"      Early stopping: metric={cfg.EARLY_STOP_METRIC}  "
    #       f"freq={cfg.EARLY_STOP_FREQUENCY}  patience={cfg.EARLY_STOP_PATIENCE}  "
    #       f"delta={cfg.EARLY_STOP_RELATIVE_DELTA}")

    pipeline_kwargs = dict(
        training=training,
        validation=validation,
        testing=testing,
        model=model_name,
        model_kwargs=model_kwargs,
        optimizer="Adam",
        optimizer_kwargs=dict(lr=lr),
        training_loop="sLCWA",
        training_loop_kwargs=dict(automatic_memory_optimization=False),
        training_kwargs=dict(num_epochs=epochs, batch_size=batch, sub_batch_size=batch),
        loss=loss,
        loss_kwargs=loss_kwargs if loss_kwargs else None,
        negative_sampler=sampler,
        negative_sampler_kwargs=dict(num_negs_per_pos=cfg.NEG_PER_POS),
        evaluator="RankBasedEvaluator",
        evaluator_kwargs=dict(filtered=True),
        evaluation_kwargs=dict(
            batch_size=cfg.BATCH_SIZE_EVAL,
            slice_size=cfg.SLICE_SIZE,
        ),
        # Early stopping desactivado temporalmente (bug NVML en evals
        # intermedias dentro del contenedor de Jupyter).
        # stopper="early",
        # stopper_kwargs=dict(
        #     frequency=cfg.EARLY_STOP_FREQUENCY,
        #     patience=cfg.EARLY_STOP_PATIENCE,
        #     relative_delta=cfg.EARLY_STOP_RELATIVE_DELTA,
        #     metric=cfg.EARLY_STOP_METRIC,
        #     evaluation_slice_size=cfg.SLICE_SIZE,
        # ),
        random_seed=cfg.RANDOM_SEED,
        device=device,
    )

    result = pipeline(**pipeline_kwargs)
    print(f"\n[3/3] Guardando modelo y embeddings ...")
    out_model_dir = cfg.model_dir(model_name)
    out_embed_dir = cfg.embed_dir(model_name)
    out_model_dir.mkdir(parents=True, exist_ok=True)
    out_embed_dir.mkdir(parents=True, exist_ok=True)

    result.save_to_directory(str(out_model_dir))
    print(f"      Modelo guardado en {out_model_dir}")

    entity_repr   = result.model.entity_representations[0]
    relation_repr = result.model.relation_representations[0]

    entity_embs   = entity_repr(indices=None).detach().cpu()
    relation_embs = relation_repr(indices=None).detach().cpu()

    torch.save(entity_embs,   cfg.entity_embeddings_path(model_name))
    torch.save(relation_embs, cfg.relation_embeddings_path(model_name))
    print(f"      Embeddings guardados en {out_embed_dir}")
    print(f"      entity_embeddings.pt  shape: {list(entity_embs.shape)}")
    print(f"      relation_embeddings.pt shape: {list(relation_embs.shape)}")

    # Los mapas entity_to_id y relation_to_id son compartidos entre modelos,
    # así que se guardan en MAPS_DIR (ya generados por fase 1)
    # No es necesario volver a guardarlos aquí

    # Resumen de métricas de test
    metrics = result.metric_results.to_dict()
    hits = metrics.get("both", {}).get("realistic", {})
    print(f"\n--- Métricas en test set ({model_name}) ---")
    for k in ("hits_at_1", "hits_at_3", "hits_at_10", "mean_reciprocal_rank"):
        v = hits.get(k)
        if v is not None:
            print(f"  {k}: {v:.4f}")

    # Añadir fila al control de versiones (CSV/JSON acumulativos)
    _append_to_comparison_table(
        model_name=model_name,
        result=result,
        dim=dim,
        epochs=epochs,
        lr=lr,
        device=device,
    )

    print(f"\n✓ Fase 2 completada para {model_name}.")
    return result


# ---------------------------------------------------------------------------
# Entrenamiento de todos los modelos + tabla comparativa
# ---------------------------------------------------------------------------

def train_all_models(
    epochs: int   = cfg.N_EPOCHS,
    dim:    int   = cfg.EMBEDDING_DIM,
    batch:  int   = cfg.BATCH_SIZE,
    lr:     float = cfg.LEARNING_RATE,
    device: str   = "cpu",
) -> dict:
    """
    Entrena todos los modelos en cfg.KGE_MODELS.

    Cada llamada a train() ya añade su fila al fichero acumulativo
    (training_comparison.csv / .json) vía _append_to_comparison_table.
    Aquí solo se imprime un resumen ASCII final con los modelos de esta
    sesión.
    """
    results = {}
    for model_name in cfg.KGE_MODELS:
        print(f"\n{'='*60}\nEntrenando {model_name}\n{'='*60}")
        results[model_name] = train(
            model_name=model_name,
            epochs=epochs, dim=dim, batch=batch, lr=lr, device=device,
        )
    _print_session_summary(results)
    return results


_COMPARISON_FIELDS = [
    "timestamp", "model", "dim", "epochs", "lr", "device",
    "hit@1", "hit@3", "hit@10", "mrr",
]


def _append_to_comparison_table(
    model_name: str,
    result,
    dim:    int,
    epochs: int,
    lr:     float,
    device: str,
) -> dict:
    """
    Añade una fila al fichero acumulativo de comparación de modelos.

    Cada entrenamiento (independientemente de si fue individual o vía
    --all-models) inserta una nueva fila con timestamp, hiperparámetros
    y métricas finales del test set. Es un log de versiones: si el
    mismo modelo se entrena varias veces, aparecerán varias filas y
    podrás comparar cómo evolucionan las métricas según los hiperparámetros.

    Salida:
      out/evaluation/model_comparison/training_comparison.csv  (append)
      out/evaluation/model_comparison/training_comparison.json (append)
    """
    metrics = result.metric_results.to_dict()
    hits    = metrics.get("both", {}).get("realistic", {})

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model":     model_name,
        "dim":       dim,
        "epochs":    epochs,
        "lr":        lr,
        "device":    device,
        "hit@1":     round(hits.get("hits_at_1",  0.0), 4),
        "hit@3":     round(hits.get("hits_at_3",  0.0), 4),
        "hit@10":    round(hits.get("hits_at_10", 0.0), 4),
        "mrr":       round(hits.get("mean_reciprocal_rank", 0.0), 4),
    }

    cfg.MODEL_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    csv_path  = cfg.MODEL_COMPARISON_DIR / "training_comparison.csv"
    json_path = cfg.MODEL_COMPARISON_DIR / "training_comparison.json"

    # CSV: append-mode con header si no existe
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COMPARISON_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    # JSON: leer, añadir, guardar (formato lista de dicts)
    existing: list = []
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                existing = data
        except json.JSONDecodeError:
            existing = []
    existing.append(row)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"\n  Tabla comparativa actualizada: {csv_path}")
    print(f"    Entradas totales: {len(existing)}  (entrada nueva: {row['timestamp']})")
    return row


def _print_session_summary(results: dict) -> None:
    """Imprime una tabla ASCII con los modelos entrenados en esta sesión."""
    rows = []
    for model_name, result in results.items():
        metrics = result.metric_results.to_dict()
        hits = metrics.get("both", {}).get("realistic", {})
        rows.append({
            "model":   model_name,
            "hit@1":   round(hits.get("hits_at_1",  0.0), 4),
            "hit@3":   round(hits.get("hits_at_3",  0.0), 4),
            "hit@10":  round(hits.get("hits_at_10", 0.0), 4),
            "mrr":     round(hits.get("mean_reciprocal_rank", 0.0), 4),
        })

    print("\n" + "=" * 55)
    print(f"  Resumen de sesión")
    print(f"  {'Modelo':<12} {'Hit@1':>8} {'Hit@3':>8} {'Hit@10':>8} {'MRR':>8}")
    print("  " + "-" * 51)
    for row in rows:
        print(f"  {row['model']:<12} {row['hit@1']:>8.4f} {row['hit@3']:>8.4f} "
              f"{row['hit@10']:>8.4f} {row['mrr']:>8.4f}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(model_name=None, epochs=None, dim=None, device=None, all_models=False, margin=9.0):
    epochs = epochs or cfg.N_EPOCHS
    dim    = dim    or cfg.EMBEDDING_DIM
    device = device or cfg.DEVICE
    if all_models:
        train_all_models(epochs=epochs, dim=dim, device=device)
    else:
        train(model_name=model_name or 'DistMult', epochs=epochs, dim=dim, device=device, margin=margin)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelos KGE con PyKEEN")
    parser.add_argument("--model",      default="DistMult",
                        help=f"Modelo KGE a entrenar. Opciones: {cfg.KGE_MODELS} (default: DistMult)")
    parser.add_argument("--all-models", action="store_true",
                        help=f"Entrenar todos los modelos: {cfg.KGE_MODELS}")
    parser.add_argument("--epochs", type=int, default=cfg.N_EPOCHS)
    parser.add_argument("--dim",    type=int, default=cfg.EMBEDDING_DIM)
    parser.add_argument("--device", default=cfg.DEVICE, choices=["cpu", "cuda"])
    parser.add_argument("--margin", type=float, default=9.0,
                        help="Margin para NSSALoss (solo TransE). Prueba 6, 12, 24.")
    args = parser.parse_args()
    run(
        model_name=args.model,
        epochs=args.epochs,
        dim=args.dim,
        device=args.device,
        all_models=args.all_models,
        margin=args.margin,
    )
