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
# Caches por-factory (evitan reconstruir id_to_ent y diccionarios inversos
# en cada llamada a predict_tails / predict_heads).
# ---------------------------------------------------------------------------

_FACTORY_CACHE: dict[int, dict] = {}


def _factory_cache(training_factory) -> dict:
    key = id(training_factory)
    cache = _FACTORY_CACHE.get(key)
    if cache is None:
        cache = {
            "ent2id":   training_factory.entity_to_id,
            "rel2id":   training_factory.relation_to_id,
            "id_to_ent": {v: k for k, v in training_factory.entity_to_id.items()},
            "id_to_rel": {v: k for k, v in training_factory.relation_to_id.items()},
        }
        _FACTORY_CACHE[key] = cache
    return cache


# ---------------------------------------------------------------------------
# Carga del modelo y fábrica de tripletas
# ---------------------------------------------------------------------------

def load_model_by_name(model_name: str = 'DistMult'):
    """
    Carga un modelo KGE entrenado por nombre desde out/models/<model_name>/.
    Retorna (model, training_factory).

    El factory se carga del directorio del modelo (guardado por PyKEEN durante
    el entrenamiento), garantizando que entity_to_id coincide exactamente con
    el vocabulario con el que se entrenó el modelo.
    """
    import pickle
    from pykeen.triples import TriplesFactory

    model_dir = cfg.model_dir(model_name)
    model_path = model_dir / "trained_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}\n"
            f"Ejecuta primero:  python src/phase2_kge_train.py --model {model_name}"
        )

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    # Intentar cargar el factory guardado durante el entrenamiento.
    # PyKEEN lo guarda en training_triples_factory.pkl (o training.pkl).
    factory_loaded = False
    for fname in ("training_triples_factory.pkl", "training.pkl"):
        fpath = model_dir / fname
        if fpath.exists():
            with open(fpath, "rb") as fh:
                training_factory = pickle.load(fh)
            factory_loaded = True
            break

    if not factory_loaded:
        # Fallback: recrear desde TSV (puede haber mismatch de vocabulario)
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
    Dado (head, relation, ?), devuelve las top_k entidades tail más probables.

    Usa model.score_t directamente para evitar incompatibilidades de versión
    entre el factory guardado en el modelo y el factory reconstruido desde TSV.
    """
    try:
        cache    = _factory_cache(training_factory)
        head_id  = cache["ent2id"].get(head_label)
        rel_id   = cache["rel2id"].get(relation_label)
        if head_id is None or rel_id is None:
            return []

        hr = torch.tensor([[head_id, rel_id]], dtype=torch.long)
        with torch.no_grad():
            scores = model.score_t(hr).squeeze(0).cpu()  # [num_entities]

        n = min(top_k, scores.shape[0])
        top_scores, top_ids = torch.topk(scores, n)

        id_to_ent = cache["id_to_ent"]
        return [
            (id_to_ent[i.item()], s.item())
            for i, s in zip(top_ids, top_scores)
            if i.item() in id_to_ent
        ]
    except Exception:
        return []


def predict_heads(
    model,
    training_factory,
    relation_label: str,
    tail_label: str,
    top_k: int = cfg.TOP_K_PREDICT,
) -> list[tuple[str, float]]:
    """
    Dado (?, relation, tail), devuelve las top_k entidades head más probables.
    """
    try:
        cache    = _factory_cache(training_factory)
        tail_id  = cache["ent2id"].get(tail_label)
        rel_id   = cache["rel2id"].get(relation_label)
        if tail_id is None or rel_id is None:
            return []

        rt = torch.tensor([[rel_id, tail_id]], dtype=torch.long)
        with torch.no_grad():
            scores = model.score_h(rt).squeeze(0).cpu()  # [num_entities]

        n = min(top_k, scores.shape[0])
        top_scores, top_ids = torch.topk(scores, n)

        id_to_ent = cache["id_to_ent"]
        return [
            (id_to_ent[i.item()], s.item())
            for i, s in zip(top_ids, top_scores)
            if i.item() in id_to_ent
        ]
    except Exception:
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
