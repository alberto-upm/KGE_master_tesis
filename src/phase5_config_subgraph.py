"""
Fase 5 — Subgrafo de configuración por sesión (CBR-style).

Para cada incidencia nueva:
  1. Extrae sus tripletas directas del grafo KG
  2. Busca las incidencias históricas más similares usando similitud coseno
     en el espacio de embeddings de entidades (DistMult)
  3. Incluye predicciones implícitas de phase3 si están disponibles
  4. Devuelve el subgrafo completo listo para ser verbalizado por phase4

No hay punto de entrada standalone relevante; se importa desde phase4 y phase6.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from generate_corpus import PRED_TEMPLATES_ES


# ---------------------------------------------------------------------------
# Extracción del subgrafo directo
# ---------------------------------------------------------------------------

def extract_direct_subgraph(incident_id: str, incidents_map: dict) -> dict:
    """
    Devuelve las propiedades directas de la incidencia desde el mapa cargado
    por generate_corpus.build_incident_map().

    Retorna: {predicate_local: [object_labels], ...} o {} si no existe.
    """
    return dict(incidents_map.get(incident_id, {}))


# ---------------------------------------------------------------------------
# Verbalización del subgrafo
# ---------------------------------------------------------------------------

def verbalize_subgraph(incident_id: str, props: dict) -> list[str]:
    """
    Convierte las propiedades del subgrafo a frases en español.
    Reutiliza PRED_TEMPLATES_ES de generate_corpus.
    """
    sentences = []
    for pred, values in props.items():
        template = PRED_TEMPLATES_ES.get(pred)
        if not template:
            continue
        for val in (values if isinstance(values, list) else [values]):
            sentences.append(template.format(s=incident_id, o=val))
    return sentences


# ---------------------------------------------------------------------------
# Búsqueda de incidencias similares (CBR retrieval)
# ---------------------------------------------------------------------------

class SimilarityIndex:
    """
    Índice de similitud coseno sobre los embeddings de entidades DistMult.
    Pre-filtra sólo los índices de entidades tipo 'incident_' para
    eficiencia (O(n_incidents) en lugar de O(n_all_entities)).
    """

    def __init__(
        self,
        entity_embeddings: torch.Tensor,
        entity_to_id: dict,
    ):
        self.entity_to_id = entity_to_id
        self.id_to_entity = {v: k for k, v in entity_to_id.items()}

        # Pre-filtrar índices de incidencias
        self.incident_indices = [
            i for label, i in entity_to_id.items()
            if label.startswith("incident_")
        ]
        self.incident_labels = [self.id_to_entity[i] for i in self.incident_indices]

        # Submatriz de embeddings de incidencias: [n_incidents × dim]
        idx_tensor = torch.tensor(self.incident_indices, dtype=torch.long)
        self.incident_embs = F.normalize(
            entity_embeddings[idx_tensor], dim=1
        )  # normalizar para similitud coseno eficiente

    def find_similar(
        self,
        incident_id: str,
        top_k: int = cfg.TOP_K_SIMILAR,
    ) -> list[tuple[str, float]]:
        """
        Retorna las top_k incidencias más similares a incident_id
        (excluye la propia incidencia).

        Returns: lista de (incident_label, similarity_score)
        """
        idx = self.entity_to_id.get(incident_id)
        if idx is None:
            return []

        # Embedding de la incidencia query (normalizado)
        all_embs = self.incident_embs  # [n_incidents × dim]
        # Localizar posición en el subconjunto de incidencias
        try:
            pos_in_subset = self.incident_indices.index(idx)
        except ValueError:
            return []

        query_vec = all_embs[pos_in_subset].unsqueeze(0)  # [1 × dim]
        sims = torch.mv(all_embs, query_vec.squeeze())     # [n_incidents]

        # Top k+1 para luego descartar la propia incidencia
        top_vals, top_pos = sims.topk(min(top_k + 1, len(sims)))

        results = []
        for pos, val in zip(top_pos.tolist(), top_vals.tolist()):
            label = self.incident_labels[pos]
            if label != incident_id:
                results.append((label, round(float(val), 4)))
        return results[:top_k]


# ---------------------------------------------------------------------------
# Construcción del subgrafo de sesión completo
# ---------------------------------------------------------------------------

def build_session_subgraph(
    incident_id: str,
    incidents_map: dict,
    similarity_index: Optional[SimilarityIndex] = None,
    implicit_preds: Optional[dict] = None,
    top_k_similar: int = cfg.TOP_K_SIMILAR,
) -> dict:
    """
    Construye el subgrafo de sesión CBR para una incidencia concreta.

    Incluye:
      - Tripletas directas de la incidencia
      - Top-K incidencias históricas similares (con sus propiedades)
      - Predicciones implícitas de phase3 (si están disponibles)

    Retorna dict estructurado listo para verbalization y LLM.
    """
    # 1. Subgrafo directo
    direct_props = extract_direct_subgraph(incident_id, incidents_map)

    # 2. Incidencias similares
    similar_cases = []
    if similarity_index is not None:
        similar = similarity_index.find_similar(incident_id, top_k=top_k_similar)
        for sim_id, score in similar:
            sim_props = extract_direct_subgraph(sim_id, incidents_map)
            similar_cases.append({
                "incident_id": sim_id,
                "similarity":  score,
                "props":       sim_props,
            })

    # 3. Predicciones implícitas (si las hay en el JSON de phase3)
    imp_preds = []
    if implicit_preds:
        # Predicciones tail para esta incidencia como cabeza
        for rel_label, preds_list in implicit_preds.items():
            if rel_label.startswith("_"):
                continue
            for entry in preds_list:
                if entry.get("head") == incident_id:
                    imp_preds.extend(entry.get("top_tails", []))

    return {
        "incident_id":          incident_id,
        "direct":               direct_props,
        "similar_cases":        similar_cases,
        "implicit_predictions": imp_preds,
    }


def verbalize_session_subgraph(session_subgraph: dict) -> list[str]:
    """
    Convierte el subgrafo de sesión completo a una lista de frases españolas.
    """
    inc_id   = session_subgraph["incident_id"]
    sentences = verbalize_subgraph(inc_id, session_subgraph["direct"])

    # Casos similares
    for case in session_subgraph.get("similar_cases", []):
        sim_id   = case["incident_id"]
        sim_score = case["similarity"]
        sim_sentences = verbalize_subgraph(sim_id, case["props"])
        if sim_sentences:
            sentences.append(
                f"[Caso similar (similitud={sim_score})] {sim_id}:"
            )
            sentences.extend(f"  {s}" for s in sim_sentences[:3])

    # Predicciones implícitas
    for pred in session_subgraph.get("implicit_predictions", [])[:3]:
        entity = pred.get("entity", "")
        score  = pred.get("score", 0.0)
        if entity:
            sentences.append(
                f"[Predicción implícita] Entidad probable: {entity} (score={score:.3f})"
            )

    return sentences


# ---------------------------------------------------------------------------
# Carga de artefactos (embeddings + índice) — lazy singleton
# ---------------------------------------------------------------------------

_similarity_index: Optional[SimilarityIndex] = None


def get_similarity_index() -> Optional[SimilarityIndex]:
    """
    Carga el índice de similitud una sola vez (singleton).
    Retorna None si los embeddings no existen aún (phase2 no ejecutada).
    """
    global _similarity_index
    if _similarity_index is not None:
        return _similarity_index

    if not cfg.ENTITY_EMBEDDINGS.exists() or not cfg.ENTITY_TO_ID.exists():
        return None

    entity_embs = torch.load(cfg.ENTITY_EMBEDDINGS, map_location="cpu", weights_only=True)
    with open(cfg.ENTITY_TO_ID, "r", encoding="utf-8") as f:
        entity_to_id = json.load(f)
    # Convertir valores a int (pueden llegar como str desde JSON)
    entity_to_id = {k: int(v) for k, v in entity_to_id.items()}

    _similarity_index = SimilarityIndex(entity_embs, entity_to_id)
    n = len(_similarity_index.incident_labels)
    print(f"[Phase5] Índice de similitud listo: {n:,} incidencias indexadas.")
    return _similarity_index


def load_implicit_predictions() -> Optional[dict]:
    if not cfg.IMPLICIT_RELS_FILE.exists():
        return None
    with open(cfg.IMPLICIT_RELS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
