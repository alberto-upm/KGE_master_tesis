"""
Fase 1 — Parseo del grafo RDF a tripletas TSV para PyKEEN.

Lee train_full.ttl (el 95% generado por la fase 0, con el 5% de test ya
reservado en test_eval.ttl) y vuelca todas las tripletas en un único
train.tsv, sin ningún split adicional.

Salida:
  data/triples/train.tsv      (todas las tripletas de train_full.ttl)
  out/maps/entity_to_id.json
  out/maps/relation_to_id.json

Uso:
  python src/phase1_triples.py
  python src/run_pipeline.py --phase 1
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rdflib import RDF

import config as cfg
from graph_utils import load_graph, extract_label


# ---------------------------------------------------------------------------
# Extracción de tripletas
# ---------------------------------------------------------------------------

def extract_all_triples(g) -> list[tuple[str, str, str]]:
    """
    Devuelve todas las tripletas (head, relation, tail) del grafo.
    Excluye rdf:type (estructura, no semántica útil para KGE).
    """
    triples = []
    skipped = 0
    for s, p, o in g:
        if p == RDF.type:
            skipped += 1
            continue
        head     = extract_label(s)
        relation = extract_label(p)
        tail     = extract_label(o)
        triples.append((head, relation, tail))
    print(f"      {len(triples):,} tripletas extraídas  ({skipped:,} rdf:type omitidas)")
    return triples


# ---------------------------------------------------------------------------
# Escritura de TSV
# ---------------------------------------------------------------------------

def save_tsv(triples: list[tuple[str, str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")
    print(f"      Guardado: {path}  ({len(triples):,} filas)")


# ---------------------------------------------------------------------------
# Mapas entidad/relación → id
# ---------------------------------------------------------------------------

def build_and_save_mappings(
    all_triples: list[tuple[str, str, str]],
) -> tuple[dict, dict]:
    """
    Construye entity_to_id y relation_to_id con índices enteros.
    Los guarda en out/maps/ como JSON (compartidos entre modelos).
    """
    entities  = sorted({t[0] for t in all_triples} | {t[2] for t in all_triples})
    relations = sorted({t[1] for t in all_triples})

    entity_to_id   = {e: i for i, e in enumerate(entities)}
    relation_to_id = {r: i for i, r in enumerate(relations)}

    cfg.MAPS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.ENTITY_TO_ID,   "w", encoding="utf-8") as f:
        json.dump(entity_to_id,   f, ensure_ascii=False, indent=2)
    with open(cfg.RELATION_TO_ID, "w", encoding="utf-8") as f:
        json.dump(relation_to_id, f, ensure_ascii=False, indent=2)

    print(f"      Entidades únicas:  {len(entity_to_id):,}")
    print(f"      Relaciones únicas: {len(relation_to_id):,}")
    print(f"      Mapas guardados en {cfg.MAPS_DIR}")
    return entity_to_id, relation_to_id


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run() -> None:
    print("=" * 60)
    print("FASE 1 — Parseo del grafo RDF a tripletas TSV")
    print("  train_full.ttl → train.tsv (sin split)")
    print("=" * 60)

    input_file = cfg.TRAIN_TTL

    if not input_file.exists():
        raise FileNotFoundError(
            f"No se encontró {input_file}\n"
            "  Ejecuta primero: python src/run_pipeline.py --phase 0"
        )

    # 1. Cargar grafo
    g = load_graph(input_file)

    # 2. Extraer tripletas
    print("[1/3] Extrayendo tripletas ...")
    all_triples = extract_all_triples(g)

    cfg.TRIPLES_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Guardar train.tsv (todas las tripletas, sin split)
    print("[2/3] Guardando train.tsv ...")
    save_tsv(all_triples, cfg.TRAIN_TSV)

    # 4. Mapas
    print("[3/3] Generando mapas entidad/relación → id ...")
    build_and_save_mappings(all_triples)

    print("\n✓ Fase 1 completada.")


if __name__ == "__main__":
    run()
