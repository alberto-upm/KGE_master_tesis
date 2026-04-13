"""
Fase 3 — Inferencia de relaciones latentes (link prediction).

Usa cualquier modelo KGE entrenado en phase2 para predecir entidades
tail dada una cabeza y una relación, e inferir patrones implícitos
en el grafo de incidencias.

Salida:
  out/predictions/implicit_relations.json

Uso:
  python src/phase3_link_prediction.py [--top-k 10] [--model DistMult]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# ---------------------------------------------------------------------------
# Carga del modelo y fábrica de tripletas
# ---------------------------------------------------------------------------

def load_model_by_name(model_name: str = 'DistMult'):
    """
    Carga un modelo KGE entrenado por nombre desde out/models/<model_name>/.
    Retorna (model, training_factory).
    """
    from pykeen.triples import TriplesFactory

    model_path = cfg.model_dir(model_name) / "trained_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}\n"
            f"Ejecuta primero:  python src/phase2_kge_train.py --model {model_name}"
        )

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    training_factory = TriplesFactory.from_path(cfg.TRAIN_TSV)
    return model, training_factory


def load_model_and_factory():
    """Carga el modelo DistMult (backward compatibility)."""
    return load_model_by_name('DistMult')


# ---------------------------------------------------------------------------
# Predicción de cola (tail prediction)
# ---------------------------------------------------------------------------

def predict_tails(
    model,
    training_factory,
    head_label: str,
    relation_label: str,
    top_k: int = cfg.TOP_K_PREDICT,
) -> list[tuple[str, float]]:
    """
    Dado (head, relation, ?), devuelve las top_k entidades tail
    más probables según DistMult.

    Retorna: lista de (entity_label, score) ordenada de mayor a menor score.
    """
    from pykeen.predict import predict_target

    try:
        scored = predict_target(
            model=model,
            head=head_label,
            relation=relation_label,
            triples_factory=training_factory,
        )
        df = scored.df.head(top_k)
        return list(zip(df["tail_label"].tolist(), df["score"].tolist()))
    except KeyError:
        # Entidad o relación no conocida por la factory
        return []


def predict_heads(
    model,
    training_factory,
    relation_label: str,
    tail_label: str,
    top_k: int = cfg.TOP_K_PREDICT,
) -> list[tuple[str, float]]:
    """
    Dado (?, relation, tail), devuelve las top_k entidades head
    más probables (útil para "¿qué técnico resuelve tipo X?").

    Nota: DistMult es simétrico → score(h,r,t) == score(t,r,h).
    """
    from pykeen.predict import predict_target

    try:
        scored = predict_target(
            model=model,
            relation=relation_label,
            tail=tail_label,
            triples_factory=training_factory,
        )
        df = scored.df.head(top_k)
        return list(zip(df["head_label"].tolist(), df["score"].tolist()))
    except KeyError:
        return []


# ---------------------------------------------------------------------------
# Minería de relaciones implícitas
# ---------------------------------------------------------------------------

def mine_implicit_relations(
    model,
    training_factory,
    top_k: int = cfg.TOP_K_PREDICT,
    max_per_relation: int = 20,
) -> dict:
    """
    Para cada relación del grafo, muestrea entidades representativas y
    predice las entidades tail más probables.

    Esto permite descubrir patrones latentes como:
      - "¿Qué técnico suele resolver incidencias de typeIncident__1?"
      - "¿Qué grupo de soporte maneja más incidencias de company__X?"

    Retorna un dict estructurado por relación.
    """
    entity_to_id   = training_factory.entity_to_id
    relation_to_id = training_factory.relation_to_id
    id_to_entity   = {v: k for k, v in entity_to_id.items()}

    results = {}

    # Predicciones head→tail por relación
    for rel_label in relation_to_id:
        print(f"  Prediciendo tails para relación: {rel_label}")
        # Muestrear algunas entidades que sean cabeza de esta relación
        head_candidates = [
            e for e in entity_to_id
            if e.startswith("incident_")
        ][:max_per_relation]

        rel_predictions = []
        for head in head_candidates:
            preds = predict_tails(model, training_factory, head, rel_label, top_k=5)
            if preds:
                rel_predictions.append({
                    "head": head,
                    "relation": rel_label,
                    "top_tails": [{"entity": e, "score": round(s, 4)} for e, s in preds],
                })
        results[rel_label] = rel_predictions

    # Caso de uso clave: ¿qué técnico resuelve cada tipo de incidencia?
    print("  Prediciendo técnicos por tipo de incidencia ...")
    tech_by_type = {}
    for type_label in [e for e in entity_to_id if e.startswith("typeIncident__")]:
        preds = predict_heads(model, training_factory, "hasTechnician", type_label, top_k=top_k)
        # Filtrar solo empleados
        employee_preds = [(e, s) for e, s in preds if e.startswith("employee__")]
        tech_by_type[type_label] = [
            {"technician": e, "score": round(s, 4)} for e, s in employee_preds[:5]
        ]
    results["_techniciansByIncidentType"] = tech_by_type

    return results


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(top_k: int = cfg.TOP_K_PREDICT, model_name: str = 'DistMult') -> dict:
    print("=" * 60)
    print(f"FASE 3 — Link prediction ({model_name})")
    print("=" * 60)

    print(f"[1/3] Cargando modelo {model_name} y factory ...")
    model, training_factory = load_model_by_name(model_name)

    print(f"[2/3] Minando relaciones implícitas (top_k={top_k}) ...")
    predictions = mine_implicit_relations(model, training_factory, top_k=top_k)

    print("[3/3] Guardando predicciones ...")
    cfg.PRED_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.IMPLICIT_RELS_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"      Guardado: {cfg.IMPLICIT_RELS_FILE}")

    # Resumen
    n_rels = len([k for k in predictions if not k.startswith("_")])
    print(f"\n      Relaciones procesadas: {n_rels}")
    print("\n✓ Fase 3 completada.")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link prediction con modelos KGE")
    parser.add_argument("--top-k", type=int, default=cfg.TOP_K_PREDICT)
    parser.add_argument("--model", default="DistMult",
                        help="Modelo KGE a usar (default: DistMult)")
    args = parser.parse_args()
    run(top_k=args.top_k, model_name=args.model)
