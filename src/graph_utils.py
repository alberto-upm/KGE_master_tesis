"""
Utilidades de carga del grafo RDF de incidencias.

Librería transversal importada por el resto del pipeline (no es una fase):

  - load_graph()         : carga un .ttl/.n3 con rdflib
  - extract_label()      : parte local de un URI
  - build_incident_map() : {incident_label: {predicate: [valores]}}
  - PRED_TEMPLATES_ES    : plantillas de verbalización en español
"""

from pathlib import Path
from collections import defaultdict

from rdflib import Graph, URIRef


# ---------------------------------------------------------------------------
# 1. PARSEO DEL GRAFO
# ---------------------------------------------------------------------------

def load_graph(path: Path) -> Graph:
    print(f"Cargando grafo desde {path} ...")
    g = Graph()
    fmt = "n3" if str(path).endswith(".n3") else "turtle"
    g.parse(str(path), format=fmt)
    print(f"      {len(g)} tripletas cargadas.")
    return g


def extract_label(uri: URIRef) -> str:
    """Extrae la parte local del URI (ej. 'employee__233')."""
    s = str(uri)
    # Fragment (#) o último segmento (/)
    if "#" in s:
        return s.split("#")[-1]
    return s.split("/")[-1]


def build_incident_map(g: Graph) -> dict:
    """
    Devuelve un dict:
        incident_label → {predicate_local: [object_label, ...], ...}

    Detecta las incidencias por etiqueta local (predicado "type" → "incident"),
    de forma independiente del namespace.
    """
    print("Construyendo mapa de incidencias ...")

    # Identificar todos los sujetos cuyo predicado local "type" apunta a "incident"
    incident_subjects = set()
    for s, p, o in g:
        if extract_label(p) == "type" and extract_label(o) == "incident":
            incident_subjects.add(s)

    incidents = {}
    for subj in incident_subjects:
        label = extract_label(subj)
        props = defaultdict(list)
        for pred, obj in g.predicate_objects(subj):
            pred_local = extract_label(pred)
            if pred_local == "type":
                continue
            obj_label = extract_label(obj)
            props[pred_local].append(obj_label)
        incidents[label] = dict(props)

    print(f"      {len(incidents)} incidencias encontradas.")
    return incidents


# ---------------------------------------------------------------------------
# 2. VERBALIZACIÓN DE TRIPLETAS
# ---------------------------------------------------------------------------

PRED_TEMPLATES_ES = {
    "hasStateIncident":      "La incidencia {s} tiene el estado {o}.",
    "hasTechnician":         "El técnico asignado a la incidencia {s} es {o}.",
    "hasExternalTechnician": "El técnico externo asignado a la incidencia {s} es {o}.",
    "hasTypeInc":            "El tipo de la incidencia {s} es {o}.",
    "incident_hasOrigin":    "El origen de la incidencia {s} es {o}.",
    "int_hasCustomer":       "El cliente de la incidencia {s} es {o}.",
    "hasSupportGroup":       "El grupo de soporte de la incidencia {s} es {o}.",
    "hasSupportTeam":        "El equipo de soporte de la incidencia {s} es {o}.",
    "hasSupportCategory":    "La categoría de soporte de la incidencia {s} es {o}.",
    "createdOn":             "La incidencia {s} se creó el {o}.",
    "hasDedicationTimeMin":  "El tiempo dedicado a la incidencia {s} es de {o} minutos.",
    "hasIntervention":       "La incidencia {s} tiene la intervención {o}.",
}
