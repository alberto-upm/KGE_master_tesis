"""
Fase 0 — Preprocesado: conversión N3→TTL + split 95/5

Pasos:
  1. Convierte incident_triplets.n3 → incident_triplets.ttl (si no existe ya)
  2. Extrae un 5% de incidencias como conjunto de evaluación
  3. El 95% restante queda para entrenamiento

Reglas de asignación:
  - Incidencias (95%)      → data/train_full.ttl
  - Incidencias (5%)       → data/test_eval.ttl
  - Intervenciones         → siguen a su incidencia padre
  - Employees / maestros   → siempre en train_full.ttl

Salida:
  data/incident_triplets.ttl  (conversión del N3 original)
  data/train_full.ttl         (95% incidencias + entidades auxiliares)
  data/test_eval.ttl          (5%  incidencias + sus intervenciones)
  data/test_eval_ids.json     (lista de IDs de incidencias de test)

Uso:
  python src/phase0_preprocess.py
  python src/run_pipeline.py --phase 0
"""

import json
import random
from pathlib import Path

BASE_DIR      = Path(__file__).parent.parent
DATA_DIR      = BASE_DIR / "data"

N3_SOURCE     = DATA_DIR / "incident_triplets.n3"
INPUT_FILE    = DATA_DIR / "incident_triplets.ttl"
TRAIN_TTL     = DATA_DIR / "train_full.ttl"
TEST_TTL      = DATA_DIR / "test_eval.ttl"
TEST_IDS_JSON = DATA_DIR / "test_eval_ids.json"

TEST_RATIO  = 0.05
RANDOM_SEED = 42

PREFIX = "repcon:"


# ---------------------------------------------------------------------------
# Parseo de bloques
# ---------------------------------------------------------------------------

def parse_blocks(n3_path: Path) -> tuple[str, dict[str, list[str]]]:
    """
    Parsea el N3 en bloques por entidad.

    Cada bloque empieza en una línea no indentada con 'repcon:' y termina
    cuando aparece el siguiente bloque o una línea vacía.

    Devuelve:
      prefix_line : todas las líneas '@prefix ...' del fichero (concatenadas)
      blocks      : {entity_id_sin_prefijo: [líneas_del_bloque]}
    """
    prefix_lines: list[str] = []
    blocks: dict[str, list[str]] = {}
    current_id: str | None = None
    current_lines: list[str] = []

    def _flush():
        if current_id and current_lines:
            blocks[current_id] = current_lines[:]

    with open(n3_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()

            if stripped.startswith("@prefix"):
                if line not in prefix_lines:
                    prefix_lines.append(line)
                continue

            if not stripped:
                _flush()
                current_id = None
                current_lines = []
                continue

            # Nueva entidad: línea no indentada que empieza con el prefijo
            if not raw[0].isspace() and stripped.startswith(PREFIX):
                _flush()
                entity_token = stripped.split()[0]           # repcon:entity_id
                current_id = entity_token[len(PREFIX):]      # entity_id
                current_lines = [line]
            else:
                if current_id is not None:
                    current_lines.append(line)

    _flush()  # último bloque
    return "\n".join(prefix_lines), blocks


# ---------------------------------------------------------------------------
# Extracción de IDs de intervención de un bloque de incidencia
# ---------------------------------------------------------------------------

def _intervention_ids_of(incident_lines: list[str]) -> list[str]:
    """
    Devuelve los IDs de intervención referenciados en el bloque de una incidencia.
    Busca tokens 'repcon:intervention_...' en las líneas que contienen hasIntervention.
    """
    ids = []
    for line in incident_lines:
        if "hasIntervention" not in line:
            continue
        for token in line.split():
            token = token.rstrip(",.;")
            if token.startswith(f"{PREFIX}intervention_"):
                ids.append(token[len(PREFIX):])
    return ids


# ---------------------------------------------------------------------------
# Escritura de N3
# ---------------------------------------------------------------------------

def _write_n3(path: Path, prefix_line: str,
              blocks: dict[str, list[str]], entity_ids: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write(prefix_line + "\n\n")
        for eid in entity_ids:
            if eid not in blocks:
                continue
            for line in blocks[eid]:
                f.write(line + "\n")
            f.write("\n")
            written += 1
    return written


# ---------------------------------------------------------------------------
# Conversión N3 → TTL
# ---------------------------------------------------------------------------

def convert_n3_to_ttl(src: Path, dst: Path) -> None:
    """Convierte un fichero N3 a Turtle usando rdflib."""
    from rdflib import Graph
    print(f"[0/4] Convirtiendo {src.name} → {dst.name} ...")
    g = Graph()
    g.parse(str(src), format="n3")
    g.serialize(destination=str(dst), format="turtle")
    print(f"      {len(g):,} tripletas serializadas como Turtle.")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(test_ratio: float = TEST_RATIO, seed: int = RANDOM_SEED) -> None:
    print("=" * 60)
    print("FASE 0 — Conversión N3→TTL + split train/test")
    print(f"  Test ratio : {test_ratio:.0%}")
    print(f"  Semilla    : {seed}")
    print("=" * 60)

    # Paso 0: convertir N3 → TTL si el TTL no existe todavía
    if not INPUT_FILE.exists():
        if N3_SOURCE.exists():
            convert_n3_to_ttl(N3_SOURCE, INPUT_FILE)
        else:
            raise FileNotFoundError(
                f"No se encontró ni {INPUT_FILE.name} ni {N3_SOURCE.name} en {DATA_DIR}"
            )
    else:
        print(f"[0/4] {INPUT_FILE.name} ya existe, se omite la conversión.")

    # 1. Parsear bloques
    print("\n[1/4] Parseando bloques del N3 ...")
    prefix_line, blocks = parse_blocks(INPUT_FILE)
    print(f"      Bloques totales: {len(blocks):,}")

    # 2. Clasificar entidades por tipo
    incident_ids     = sorted(k for k in blocks if k.startswith("incident_"))
    intervention_ids = set(k for k in blocks if k.startswith("intervention_"))
    other_ids        = sorted(k for k in blocks
                              if not k.startswith("incident_")
                              and not k.startswith("intervention_"))

    print(f"      Incidencias      : {len(incident_ids):,}")
    print(f"      Intervenciones   : {len(intervention_ids):,}")
    print(f"      Auxiliares       : {len(other_ids):,}  "
          f"(employees, maestros, etc.)")

    # 3. Split de incidencias 95 / 5
    print(f"\n[2/4] Dividiendo incidencias ...")
    rng = random.Random(seed)
    shuffled = incident_ids[:]
    rng.shuffle(shuffled)

    n_test = max(1, int(len(shuffled) * test_ratio))
    test_incident_ids  = set(shuffled[:n_test])
    train_incident_ids = set(shuffled[n_test:])

    print(f"      Train : {len(train_incident_ids):,} incidencias")
    print(f"      Test  : {len(test_incident_ids):,} incidencias  "
          f"({len(test_incident_ids)/len(incident_ids):.1%})")

    # 4. Asignar intervenciones a su split según la incidencia padre
    print("\n[3/4] Asignando intervenciones a su split ...")
    test_int_ids  : set[str] = set()
    train_int_ids : set[str] = set()

    for inc_id in test_incident_ids:
        for int_id in _intervention_ids_of(blocks[inc_id]):
            if int_id in intervention_ids:
                test_int_ids.add(int_id)

    for inc_id in train_incident_ids:
        for int_id in _intervention_ids_of(blocks[inc_id]):
            if int_id in intervention_ids:
                train_int_ids.add(int_id)

    orphan = intervention_ids - test_int_ids - train_int_ids
    if orphan:
        train_int_ids.update(orphan)
        print(f"      Intervenciones huérfanas → train: {len(orphan):,}")

    print(f"      Intervenciones train : {len(train_int_ids):,}")
    print(f"      Intervenciones test  : {len(test_int_ids):,}")

    # 5. Escribir ficheros
    print("\n[4/4] Escribiendo ficheros ...")

    train_entities = other_ids + sorted(train_incident_ids) + sorted(train_int_ids)
    n_train = _write_n3(TRAIN_TTL, prefix_line, blocks, train_entities)
    print(f"      train_full.ttl  : {n_train:,} bloques")

    test_entities = sorted(test_incident_ids) + sorted(test_int_ids)
    n_test_w = _write_n3(TEST_TTL, prefix_line, blocks, test_entities)
    print(f"      test_eval.ttl   : {n_test_w:,} bloques")

    with open(TEST_IDS_JSON, "w", encoding="utf-8") as f:
        json.dump(sorted(test_incident_ids), f, ensure_ascii=False, indent=2)
    print(f"      test_eval_ids.json : {len(test_incident_ids):,} IDs")

    print("\n✓ Fase 0 completada.")
    print(f"  Conversión    : {INPUT_FILE}")
    print(f"  Entrenamiento : {TRAIN_TTL}")
    print(f"  Evaluación    : {TEST_TTL}")
    print(f"  IDs de test   : {TEST_IDS_JSON}")


if __name__ == "__main__":
    run()
