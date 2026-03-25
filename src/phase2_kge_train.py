"""
Fase 2 — Entrenamiento del modelo KGE DistMult con PyKEEN.

Requisito previo: ejecutar phase1_triples.py para generar los TSV.

Salida:
  out/models/distmult/           (modelo PyKEEN completo)
  out/embeddings/entity_embeddings.pt
  out/embeddings/relation_embeddings.pt

Uso:
  python src/phase2_kge_train.py [--epochs N] [--dim D] [--device cpu|cuda]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train(
    epochs:  int = cfg.N_EPOCHS,
    dim:     int = cfg.EMBEDDING_DIM,
    batch:   int = cfg.BATCH_SIZE,
    lr:      float = cfg.LEARNING_RATE,
    device:  str = "cpu",
) -> None:
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    print("=" * 60)
    print("FASE 2 — Entrenamiento DistMult con PyKEEN")
    print("=" * 60)

    # Verificar que existen los TSV
    for tsv in (cfg.TRAIN_TSV, cfg.VALID_TSV, cfg.TEST_TSV):
        if not tsv.exists():
            raise FileNotFoundError(
                f"No encontrado: {tsv}\n"
                "Ejecuta primero:  python src/phase1_triples.py"
            )

    print(f"[1/3] Cargando tripletas desde {cfg.TRIPLES_DIR} ...")
    training = TriplesFactory.from_path(cfg.TRAIN_TSV)
    validation = TriplesFactory.from_path(
        cfg.VALID_TSV,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )
    testing = TriplesFactory.from_path(
        cfg.TEST_TSV,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )
    print(f"      Entidades:  {training.num_entities:,}")
    print(f"      Relaciones: {training.num_relations:,}")
    print(f"      Train / Valid / Test: "
          f"{training.num_triples:,} / {validation.num_triples:,} / {testing.num_triples:,}")

    print(f"\n[2/3] Entrenando DistMult  (dim={dim}, epochs={epochs}, device={device}) ...")
    result = pipeline(
        training=training,
        validation=validation,
        testing=testing,
        model="DistMult",
        model_kwargs=dict(embedding_dim=dim),
        optimizer="Adam",
        optimizer_kwargs=dict(lr=lr),
        training_loop="sLCWA",
        training_kwargs=dict(
            num_epochs=epochs,
            batch_size=batch,
        ),
        negative_sampler="basic",
        negative_sampler_kwargs=dict(num_negs_per_pos=cfg.NEG_PER_POS),
        loss="BCEWithLogitsLoss",
        evaluator="RankBasedEvaluator",
        evaluator_kwargs=dict(filtered=True),
        # Entrenamiento en GPU, evaluación en CPU para evitar OOM
        # (60K entidades × 45K tripletas desborda la GPU durante el ranking)
        evaluation_kwargs=dict(batch_size=32, device="cpu"),
        random_seed=cfg.RANDOM_SEED,
        device=device,
    )

    print("\n[3/3] Guardando modelo y embeddings ...")
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    result.save_to_directory(str(cfg.MODELS_DIR))
    print(f"      Modelo guardado en {cfg.MODELS_DIR}")

    # Exportar tensores de embeddings para uso en fases posteriores
    cfg.EMBED_DIR.mkdir(parents=True, exist_ok=True)

    entity_repr   = result.model.entity_representations[0]
    relation_repr = result.model.relation_representations[0]

    entity_embs   = entity_repr(indices=None).detach().cpu()
    relation_embs = relation_repr(indices=None).detach().cpu()

    torch.save(entity_embs,   cfg.ENTITY_EMBEDDINGS)
    torch.save(relation_embs, cfg.RELATION_EMBEDDINGS)
    print(f"      Embeddings guardados en {cfg.EMBED_DIR}")
    print(f"      entity_embeddings.pt  shape: {list(entity_embs.shape)}")
    print(f"      relation_embeddings.pt shape: {list(relation_embs.shape)}")

    # Guardar también los mapas id→entidad derivados de la factory
    # (complementarios a los generados en phase1)
    entity_to_id   = {str(k): int(v) for k, v in training.entity_to_id.items()}
    relation_to_id = {str(k): int(v) for k, v in training.relation_to_id.items()}
    with open(cfg.ENTITY_TO_ID,   "w", encoding="utf-8") as f:
        json.dump(entity_to_id, f, ensure_ascii=False, indent=2)
    with open(cfg.RELATION_TO_ID, "w", encoding="utf-8") as f:
        json.dump(relation_to_id, f, ensure_ascii=False, indent=2)
    print(f"      Mapas id actualizados en {cfg.EMBED_DIR}")

    # Resumen de métricas de test
    metrics = result.metric_results.to_dict()
    hits = metrics.get("both", {}).get("realistic", {})
    print("\n--- Métricas en test set ---")
    for k in ("hits_at_1", "hits_at_3", "hits_at_10", "mean_reciprocal_rank"):
        v = hits.get(k)
        if v is not None:
            print(f"  {k}: {v:.4f}")

    print("\n✓ Fase 2 completada.")
    return result


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(epochs=None, dim=None, device=None):
    epochs = epochs or cfg.N_EPOCHS
    dim    = dim    or cfg.EMBEDDING_DIM
    device = device or cfg.DEVICE
    train(epochs=epochs, dim=dim, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar DistMult con PyKEEN")
    parser.add_argument("--epochs", type=int, default=cfg.N_EPOCHS)
    parser.add_argument("--dim",    type=int, default=cfg.EMBEDDING_DIM)
    parser.add_argument("--device", default=cfg.DEVICE, choices=["cpu", "cuda"])
    args = parser.parse_args()
    run(epochs=args.epochs, dim=args.dim, device=args.device)
