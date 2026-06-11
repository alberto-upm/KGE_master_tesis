#!/usr/bin/env python3
"""Divide data/train_full.ttl en subconjuntos por tipo de entidad.

Cada bloque Turtle (sujeto + sus triples, terminado en ' .') se clasifica
segun el prefijo de su sujeto:
    repcon:incident_      -> incident
    repcon:intervention_  -> intervention
    repcon:employee_      -> employee

Genera en data/train_splits/:
    train_full_incidents.ttl                  (incidents)
    train_full_interventions.ttl              (interventions)
    train_full_incidents_interventions.ttl    (incidents + interventions)
    train_full_interventions_employees.ttl    (interventions + employees)
    train_full_employees.ttl                  (employees)

Los bloques se copian intactos; solo se filtra por el tipo del sujeto.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # src/rules/ → src/ → raíz del repo
SRC = ROOT / "data" / "train_full.ttl"
OUT_DIR = ROOT / "data" / "train_splits"

# Prefijo del sujeto -> tipo de entidad
SUBJECT_TYPES = {
    "repcon:incident_": "incident",
    "repcon:intervention_": "intervention",
    "repcon:employee_": "employee",
}

# Fichero de salida -> tipos de entidad que incluye
OUTPUTS = {
    "train_full_incidents.ttl": {"incident"},
    "train_full_interventions.ttl": {"intervention"},
    "train_full_incidents_interventions.ttl": {"incident", "intervention"},
    "train_full_interventions_employees.ttl": {"intervention", "employee"},
    "train_full_employees.ttl": {"employee"},
}


def classify(subject_line: str):
    """Devuelve el tipo de entidad del sujeto, o None si no coincide."""
    for prefix, etype in SUBJECT_TYPES.items():
        if subject_line.startswith(prefix):
            return etype
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Recoge las cabeceras @prefix para replicarlas en cada salida.
    prefixes = []
    with SRC.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("@prefix") or stripped.startswith("@base"):
                prefixes.append(line)
            elif stripped:
                break

    handles = {}
    counts = {name: 0 for name in OUTPUTS}
    try:
        for name in OUTPUTS:
            fh = (OUT_DIR / name).open("w", encoding="utf-8")
            for p in prefixes:
                fh.write(p)
            if prefixes:
                fh.write("\n")
            handles[name] = fh

        block = []
        current_type = None
        with SRC.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("@prefix") or stripped.startswith("@base"):
                    continue

                if not block:
                    # Primera linea del bloque: el sujeto.
                    current_type = classify(stripped)
                    block.append(line)
                else:
                    block.append(line)

                # Fin de bloque: la linea termina con '.'
                if stripped.endswith("."):
                    if current_type is not None:
                        for name, types in OUTPUTS.items():
                            if current_type in types:
                                handles[name].writelines(block)
                                counts[name] += 1
                    block = []
                    current_type = None
    finally:
        for fh in handles.values():
            fh.close()

    print("Bloques escritos por fichero:")
    for name in OUTPUTS:
        print(f"  {name}: {counts[name]}")


if __name__ == "__main__":
    main()
