"""
Fase 1 — Parseo del grafo RDF a tripletas TSV para PyKEEN.

Modo normal (filtrado.ttl):
  Divide las incidencias en 80/10/10 y genera train/valid/test.tsv.

Modo no_split (train_full.ttl generado por fase 0):
  Vuelca todas las tripletas en un único train.tsv sin ningún split.
  Usar cuando el 5% de test ya fue reservado previamente en la fase 0.

Salida (modo normal):
  data/triples/train.tsv   (80% incidencias + auxiliares)
  data/triples/valid.tsv   (10% incidencias)
  data/triples/test.tsv    (10% incidencias)

Salida (modo no_split):
  data/triples/train.tsv   (todas las tripletas)

En ambos modos:
  out/maps/entity_to_id.json
  out/maps/relation_to_id.json

Uso:
  python src/phase1_triples.py              # modo normal (filtrado.ttl)
  python src/phase1_triples.py --no-split   # modo no_split (train_full.ttl)
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rdflib import RDF

import config as cfg
from phase1b_generate_corpus import load_graph, extract_label


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
# División estratificada por predicado
# ---------------------------------------------------------------------------

def split_by_incident(
    triples: list[tuple[str, str, str]],
    train_ratio: float = cfg.TRAIN_RATIO,
    valid_ratio: float = cfg.VALID_RATIO,
    seed: int          = cfg.RANDOM_SEED,
) -> tuple[list, list, list]:
    """
    Divide las tripletas en train/valid/test agrupando por incidencia.

    1. Identifica todos los IDs de incidencia únicos (head starts with 'incident_')
    2. Reparte los IDs de incidencias en 80/10/10
    3. Asigna TODAS las tripletas de cada incidencia al mismo split
    4. Tripletas cuyo head NO es una incidencia van a train (entidades auxiliares)
    """
    rng = random.Random(seed)

    # 1. Identificar IDs de incidencias únicos
    incident_ids = sorted({h for h, _, _ in triples if h.startswith("incident_")})
    print(f"      Incidencias únicas: {len(incident_ids):,}")

    # 2. Shuffle y split de IDs
    rng.shuffle(incident_ids)
    n  = len(incident_ids)
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + valid_ratio))
    train_ids = set(incident_ids[:i1])
    valid_ids = set(incident_ids[i1:i2])
    test_ids  = set(incident_ids[i2:])

    print(f"      Incidencias  →  train: {len(train_ids):,}  "
          f"valid: {len(valid_ids):,}  test: {len(test_ids):,}")

    # 3. Asignar tripletas según la incidencia del head
    train_set, valid_set, test_set = [], [], []
    non_incident = 0
    for triple in triples:
        head = triple[0]
        if head in train_ids:
            train_set.append(triple)
        elif head in valid_ids:
            valid_set.append(triple)
        elif head in test_ids:
            test_set.append(triple)
        else:
            # Head no es una incidencia → train (auxiliares)
            train_set.append(triple)
            non_incident += 1

    if non_incident:
        print(f"      Tripletas auxiliares (no-incidencia) añadidas a train: {non_incident:,}")
    print(f"      Tripletas  →  train: {len(train_set):,}  "
          f"valid: {len(valid_set):,}  test: {len(test_set):,}")
    return train_set, valid_set, test_set


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

def run(no_split: bool = False) -> None:
    print("=" * 60)
    print("FASE 1 — Parseo del grafo RDF a tripletas TSV")
    if no_split:
        print("  Modo: no_split  (train_full.ttl → train.tsv único)")
    else:
        print("  Modo: normal    (filtrado.ttl  → train/valid/test 80/10/10)")
    print("=" * 60)

    input_file = cfg.TRAIN_TTL if no_split else cfg.TTL_FILE

    if not input_file.exists():
        raise FileNotFoundError(
            f"No se encontró {input_file}\n"
            + ("  Ejecuta primero: python src/run_pipeline.py --phase 0"
               if no_split else "")
        )

    # 1. Cargar grafo
    g = load_graph(input_file)

    # 2. Extraer tripletas
    print("[1/3] Extrayendo tripletas ...")
    all_triples = extract_all_triples(g)

    cfg.TRIPLES_DIR.mkdir(parents=True, exist_ok=True)

    if no_split:
        # Modo no_split: todo va a train.tsv
        print("[2/3] Guardando train.tsv (sin split) ...")
        save_tsv(all_triples, cfg.TRAIN_TSV)
    else:
        # Modo normal: split 80/10/10
        print("[2/3] Dividiendo por incidencias en train/valid/test (80/10/10) ...")
        train_triples, valid_triples, test_triples = split_by_incident(all_triples)
        save_tsv(train_triples, cfg.TRAIN_TSV)
        save_tsv(valid_triples, cfg.VALID_TSV)
        save_tsv(test_triples,  cfg.TEST_TSV)

    # 3. Mapas
    print("[3/3] Generando mapas entidad/relación → id ...")
    build_and_save_mappings(all_triples)

    print("\n✓ Fase 1 completada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-split", action="store_true",
                        help="Usa train_full.ttl y vuelca todo en train.tsv (sin split)")
    args = parser.parse_args()
    run(no_split=args.no_split)
