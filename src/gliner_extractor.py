"""
Capa de extracción de entidades y relaciones para el pipeline KGE + LLM.

Usa GLiNER2 (fastino/gliner2-base-v1) para extraer la entidad (sujeto)
de una consulta en lenguaje natural, y un mapa de palabras clave para
detectar el predicado (relación del grafo).

Instalación:
  pip install gliner2

Uso:
  from gliner_extractor import GLiNERExtractor

  extractor = GLiNERExtractor()
  result = extractor.extract("¿De qué tipo es la incidencia incident_1234?")
  # {'head': 'incident_1234', 'relation': 'hasTypeInc',
  #  'head_found_by': 'regex', 'relation_found_by': 'keyword_map'}

  # Pipeline completo con link prediction:
  from phase3_link_prediction import load_model_by_name
  model, factory = load_model_by_name('DistMult')
  result = extractor.extract_for_link_prediction(query, model, factory)
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# ---------------------------------------------------------------------------
# Mapa de palabras clave → predicado del grafo
# ---------------------------------------------------------------------------
# Se hace coincidir en orden de longitud descendente (mayor → menor) para
# preferir la frase más específica:
# "técnico externo" > "técnico", "grupo de soporte" > "grupo", etc.

RELATION_KEYWORDS: dict[str, str] = {
    # hasTypeInc
    "de qué tipo":           "hasTypeInc",
    "tipo de incidencia":    "hasTypeInc",
    "tipo":                  "hasTypeInc",
    # hasExternalTechnician
    "técnico externo":       "hasExternalTechnician",
    # hasTechnician
    "técnico asignado":      "hasTechnician",
    "técnico":               "hasTechnician",
    # hasSupportGroup
    "grupo de soporte":      "hasSupportGroup",
    "grupo":                 "hasSupportGroup",
    # hasSupportTeam
    "equipo de soporte":     "hasSupportTeam",
    "equipo":                "hasSupportTeam",
    # hasSupportCategory
    "categoría de soporte":  "hasSupportCategory",
    "categoría":             "hasSupportCategory",
    # hasStateIncident
    "estado":                "hasStateIncident",
    # incident_hasOrigin
    "origen":                "incident_hasOrigin",
    # int_hasCustomer
    "cliente":               "int_hasCustomer",
}

# Palabras clave ordenadas de mayor a menor longitud (se pre-computa una vez)
_SORTED_KEYWORDS = sorted(RELATION_KEYWORDS, key=len, reverse=True)

# Patrón regex para identificadores de incidencia en el grafo
_INCIDENT_RE = re.compile(r'\bincident_\d+\b')


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class GLiNERExtractor:
    """
    Extrae (head_entity, relation) de una consulta en lenguaje natural.

    Estrategia:
      - Entidad: regex → GLiNER2 (fallback)
      - Relación: diccionario de palabras clave (longest-match)
    """

    def __init__(
        self,
        gliner_model_name: str = cfg.GLINER_MODEL,
        entity_to_id_path: Path = cfg.ENTITY_TO_ID,
    ):
        self._gliner_model_name = gliner_model_name
        self._gliner = None  # carga diferida

        with open(entity_to_id_path, encoding="utf-8") as f:
            self._entity_to_id: dict[str, int] = json.load(f)

    # ------------------------------------------------------------------
    # Carga diferida del modelo GLiNER2
    # ------------------------------------------------------------------

    def _load_gliner(self) -> None:
        from gliner2 import GLiNER2
        print(f"[GLiNER2] Cargando modelo {self._gliner_model_name} ...")
        self._gliner = GLiNER2.from_pretrained(self._gliner_model_name)

    # ------------------------------------------------------------------
    # Extracción de entidad
    # ------------------------------------------------------------------

    def extract_entity(self, text: str) -> tuple[Optional[str], str]:
        """
        Devuelve (entity_label, method).
        - method = 'regex'  → encontrado por patrón incident_\\d+
        - method = 'gliner2' → encontrado por GLiNER2
        - method = 'none'   → no encontrado
        """
        # 1. Regex: detección directa de IDs tipo incident_1497...
        m = _INCIDENT_RE.search(text)
        if m and m.group(0) in self._entity_to_id:
            return m.group(0), "regex"

        # 2. GLiNER2: para consultas sin ID explícito
        if self._gliner is None:
            self._load_gliner()

        result = self._gliner.extract_entities(
            text,
            ["incidencia", "identificador de incidencia", "número de incidencia"],
        )
        # result = {'entities': {'incidencia': ['incident_1234'], ...}}
        for spans in result.get("entities", {}).values():
            for span in spans:
                span = span.strip()
                if span in self._entity_to_id:
                    return span, "gliner2"

        return None, "none"

    # ------------------------------------------------------------------
    # Extracción de relación
    # ------------------------------------------------------------------

    def extract_relation(self, text: str) -> Optional[str]:
        """
        Escanea el texto (case-insensitive) buscando palabras clave
        ordenadas de mayor a menor longitud (longest-match first).
        Devuelve el nombre del predicado del grafo o None.
        """
        lower = text.lower()
        for kw in _SORTED_KEYWORDS:
            if kw in lower:
                return RELATION_KEYWORDS[kw]
        return None

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def extract(self, text: str) -> dict:
        """
        Extrae sujeto y predicado de una consulta en lenguaje natural.

        Retorna:
          {
            "head":             str | None,   # entity label en el grafo
            "relation":         str | None,   # predicate label en el grafo
            "head_found_by":    str,          # 'regex' | 'gliner2' | 'none'
            "relation_found_by": str | None,  # 'keyword_map' | None
          }
        """
        entity, entity_method = self.extract_entity(text)
        relation = self.extract_relation(text)
        return {
            "head":              entity,
            "relation":          relation,
            "head_found_by":     entity_method,
            "relation_found_by": "keyword_map" if relation else None,
        }

    def extract_for_link_prediction(
        self,
        text: str,
        model,
        training_factory,
        top_k: int = cfg.TOP_K_PREDICT,
    ) -> dict:
        """
        Pipeline completo: extrae (head, relation) y ejecuta link prediction.

        Retorna el dict de extract() enriquecido con:
          {
            "query":       str,
            "predictions": [{"entity": str, "score": float}, ...]
          }
        """
        from phase3_link_prediction import predict_tails

        extraction = self.extract(text)
        predictions = []
        if extraction["head"] and extraction["relation"]:
            raw_preds = predict_tails(
                model, training_factory,
                extraction["head"], extraction["relation"],
                top_k=top_k,
            )
            predictions = [
                {"entity": e, "score": round(s, 4)} for e, s in raw_preds
            ]

        return {
            **extraction,
            "query":       text,
            "predictions": predictions,
        }


# ---------------------------------------------------------------------------
# CLI de prueba rápida
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prueba rápida de extracción GLiNER2"
    )
    parser.add_argument("query", nargs="?",
                        default="¿De qué tipo es la incidencia incident_1234?",
                        help="Consulta en lenguaje natural")
    parser.add_argument("--model", default="DistMult",
                        help="Modelo KGE para link prediction (default: DistMult)")
    parser.add_argument("--predict", action="store_true",
                        help="Ejecutar link prediction tras la extracción")
    args = parser.parse_args()

    extractor = GLiNERExtractor()

    if args.predict:
        from phase3_link_prediction import load_model_by_name
        kge_model, factory = load_model_by_name(args.model)
        result = extractor.extract_for_link_prediction(args.query, kge_model, factory)
    else:
        result = extractor.extract(args.query)

    import pprint
    pprint.pprint(result)
