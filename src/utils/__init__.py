"""
Paquete de utilidades compartidas por las fases del pipeline.

Módulos:
  graph_utils    — carga de grafos RDF, extracción de labels, plantillas ES
  rule_engine    — motor de reglas simbólico (PyClause + AnyBURL)
  llm_inference  — cliente vLLM (KGEAugmentedLLM) y verbalización de propiedades

No son fases del pipeline, sino librerías que las fases importan.
"""
