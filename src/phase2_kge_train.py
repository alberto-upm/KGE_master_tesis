"""
Fase 2 — Entrenamiento de modelos KGE con PyKEEN.

Modelos soportados: TransE, DistMult, ComplEx (y cualquier otro de PyKEEN).

Requisito previo: ejecutar phase1_triples.py para generar los TSV.

Salida por modelo (ej. DistMult):
  out/models/distmult/           (modelo PyKEEN completo)
  out/embeddings/distmult/entity_embeddings.pt
  out/embeddings/distmult/relation_embeddings.pt
  out/embeddings/distmult/entity_to_id.json
  out/embeddings/distmult/relation_to_id.json

Uso:
  python src/phase2_kge_train.py                        # DistMult (por defecto)
  python src/phase2_kge_train.py --model TransE
  python src/phase2_kge_train.py --model ComplEx
  python src/phase2_kge_train.py --all-models           # entrena los 3 secuencialmente
  python src/phase2_kge_train.py --epochs N --dim D --device cpu|cuda
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# ---------------------------------------------------------------------------
# Entrenamiento de un modelo
# ---------------------------------------------------------------------------

def train(
    model_name:      str   = 'DistMult',
    epochs:          int   = cfg.N_EPOCHS,
    dim:             int   = cfg.EMBEDDING_DIM,
    batch:           int   = cfg.BATCH_SIZE,
    lr:              float = cfg.LEARNING_RATE,
    device:          str   = "cpu",
    eval_batch_size: int   = None,
):
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    print("=" * 60)
    print(f"FASE 2 — Entrenamiento {model_name} con PyKEEN")
    print("=" * 60)

    # Verificar que existen los TSV
    if not cfg.TRAIN_TSV.exists():
        raise FileNotFoundError(
            f"No encontrado: {cfg.TRAIN_TSV}\n"
            "Ejecuta primero:  python src/phase1_triples.py"
        )

    print(f"[1/3] Cargando tripletas desde {cfg.TRAIN_TSV} ...")
    # Cargar TODAS las tripletas del train.tsv (que contiene todo el grafo)
    # PyKEEN hará el split internamente, garantizando cobertura de vocabulario
    full = TriplesFactory.from_path(cfg.TRAIN_TSV)
    print(f"      Total de tripletas cargadas: {full.num_triples:,}")

    # Split con garantía de PyKEEN: todas las entidades de valid/test están en train
    training, validation, testing = full.split(
        ratios=[cfg.TRAIN_RATIO, cfg.VALID_RATIO, 1.0 - cfg.TRAIN_RATIO - cfg.VALID_RATIO],
        random_state=cfg.RANDOM_SEED,
    )
    print(f"      Entidades:  {training.num_entities:,}")
    print(f"      Relaciones: {training.num_relations:,}")
    print(f"      Train / Valid / Test: "
          f"{training.num_triples:,} / {validation.num_triples:,} / {testing.num_triples:,}")

    # Configuración por modelo:
    # - TransE:   BCEWithLogitsLoss + LCWA + norm L2   (multi-clase, sin negative sampler)
    # - DistMult: BCEWithLogitsLoss + sLCWA            (bilineal, funciona bien con BCE)
    # - ComplEx:  BCEWithLogitsLoss + sLCWA            (igual que DistMult)
    model_lower = model_name.lower()

    if model_lower == "transe":
        loss            = "BCEWithLogitsLoss"
        loss_kwargs     = {}
        model_kwargs    = dict(embedding_dim=dim, scoring_fct_norm=2)
        training_loop   = "LCWA"
        transe_num_negs = None
        transe_sampler  = None
        train_batch     = 32
        train_slice     = 2048
    else:
        loss            = "BCEWithLogitsLoss"
        loss_kwargs     = {}
        model_kwargs    = dict(embedding_dim=dim)
        training_loop   = "sLCWA"
        transe_num_negs = cfg.NEG_PER_POS
        transe_sampler  = "basic"
        train_batch     = batch
        train_slice     = None

    # ComplEx usa embeddings complejos (dim real × 2) → necesita menos RAM en evaluación
    if eval_batch_size is None:
        eval_batch_size = 8 if model_lower == "complex" else 32

    print(f"\n[2/3] Entrenando {model_name}  "
          f"(dim={dim}, epochs={epochs}, loss={loss}, "
          f"loop={training_loop}, device={device}, eval_batch={eval_batch_size}) ...")

    # Construir kwargs de pipeline dinámicamente
    # LCWA no permite negative_sampler, mientras que sLCWA sí
    pipeline_kwargs = dict(
        training=training,
        validation=validation,
        testing=testing,
        model=model_name,
        model_kwargs=model_kwargs,
        optimizer="Adam",
        optimizer_kwargs=dict(lr=lr),
        training_loop=training_loop,
        training_loop_kwargs=dict(automatic_memory_optimization=False),
        training_kwargs={k: v for k, v in dict(
            num_epochs=epochs,
            batch_size=train_batch,
            sub_batch_size=train_batch,
            slice_size=train_slice,
        ).items() if v is not None},
        loss=loss,
        loss_kwargs=loss_kwargs if loss_kwargs else None,
        evaluator="RankBasedEvaluator",
        evaluator_kwargs=dict(filtered=True),
        # Entrenamiento en GPU, evaluación en CPU para evitar OOM de GPU.
        # ComplEx usa batch pequeño para evitar OOM de RAM (embeddings doble tamaño).
        evaluation_kwargs=dict(batch_size=eval_batch_size, device="cpu"),
        random_seed=cfg.RANDOM_SEED,
        device=device,
    )

    if training_loop != "LCWA":
        pipeline_kwargs["negative_sampler"]        = transe_sampler
        pipeline_kwargs["negative_sampler_kwargs"] = dict(num_negs_per_pos=transe_num_negs)

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
    """Entrena todos los modelos en cfg.KGE_MODELS y guarda tabla comparativa."""
    results = {}
    for model_name in cfg.KGE_MODELS:
        print(f"\n{'='*60}\nEntrenando {model_name}\n{'='*60}")
        results[model_name] = train(
            model_name=model_name,
            epochs=epochs, dim=dim, batch=batch, lr=lr, device=device,
        )
    _save_comparison_table(results)
    return results


def _save_comparison_table(results: dict) -> None:
    """Extrae métricas de cada resultado PyKEEN y guarda JSON + CSV."""
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

    cfg.MODEL_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    json_path = cfg.MODEL_COMPARISON_DIR / "training_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    csv_path = cfg.MODEL_COMPARISON_DIR / "training_comparison.csv"
    fieldnames = ["model", "hit@1", "hit@3", "hit@10", "mrr"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Tabla ASCII en consola
    print("\n" + "=" * 55)
    print(f"  {'Modelo':<12} {'Hit@1':>8} {'Hit@3':>8} {'Hit@10':>8} {'MRR':>8}")
    print("  " + "-" * 51)
    for row in rows:
        print(f"  {row['model']:<12} {row['hit@1']:>8.4f} {row['hit@3']:>8.4f} "
              f"{row['hit@10']:>8.4f} {row['mrr']:>8.4f}")
    print("=" * 55)
    print(f"\n  Tabla guardada en {csv_path}")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(model_name=None, epochs=None, dim=None, device=None, all_models=False):
    epochs = epochs or cfg.N_EPOCHS
    dim    = dim    or cfg.EMBEDDING_DIM
    device = device or cfg.DEVICE
    if all_models:
        train_all_models(epochs=epochs, dim=dim, device=device)
    else:
        train(model_name=model_name or 'DistMult', epochs=epochs, dim=dim, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelos KGE con PyKEEN")
    parser.add_argument("--model",      default="DistMult",
                        help="Modelo KGE a entrenar (default: DistMult)")
    parser.add_argument("--all-models", action="store_true",
                        help=f"Entrenar todos los modelos: {cfg.KGE_MODELS}")
    parser.add_argument("--epochs", type=int, default=cfg.N_EPOCHS)
    parser.add_argument("--dim",    type=int, default=cfg.EMBEDDING_DIM)
    parser.add_argument("--device", default=cfg.DEVICE, choices=["cpu", "cuda"])
    args = parser.parse_args()
    run(
        model_name=args.model,
        epochs=args.epochs,
        dim=args.dim,
        device=args.device,
        all_models=args.all_models,
    )
