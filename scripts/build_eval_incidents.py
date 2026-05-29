"""
Construye el conjunto de evaluación para phase6_eval_incident_creator.

Lee data/test_eval.ttl, selecciona N incidencias (default 500) y exporta cada
una como un objeto JSONL con la misma forma que `created_incidents.jsonl` del
wizard de phase4_incident_creator. Los campos ausentes se marcan con "skip"
para que la evaluación los pueda saltar sin ambigüedad.

Salida:
    data/evaluacion/test_eval_<N>.jsonl

Formato por línea:
    {
      "incident_id": "incident__xxx",
      "source_file": "test_eval.ttl",
      "incident": {
          "int_hasCustomer":       "company__9G1G3MV0P",
          "hasTypeInc":            "typeIncident__1",
          ...
          "hasExternalTechnician": "skip",
          "hasIntervention":       "skip"
      }
    }

Uso:
    python scripts/build_eval_incidents.py
    python scripts/build_eval_incidents.py --n 500 --seed 42
    python scripts/build_eval_incidents.py --ttl data/test_eval.ttl --out data/evaluacion
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config as cfg
from phase4_incident_creator import INCIDENT_PROPS, MULTI_VALUE_PROPS
from phase1b_generate_corpus import load_graph, build_incident_map


SKIP_MARK = "skip"


def _flatten(values, multi: bool):
    """Devuelve el valor a guardar para una propiedad."""
    if not values:
        return SKIP_MARK
    if multi:
        return list(values)
    return values[0] if isinstance(values, list) else values


def build_eval_set(
    ttl_path: Path,
    n: int,
    seed: int,
) -> list[dict]:
    if not ttl_path.exists():
        raise FileNotFoundError(
            f"No encontrado: {ttl_path}\n"
            "Ejecuta antes:  python src/run_pipeline.py --phase 0"
        )

    print(f"[1/3] Cargando grafo desde {ttl_path} ...")
    g = load_graph(ttl_path)
    raw = build_incident_map(g)
    print(f"      {len(raw):,} incidencias en el TTL")

    prop_set = set(INCIDENT_PROPS)

    # Solo conservamos incidencias con al menos una propiedad relevante.
    candidates: list[str] = []
    for inc_id, props in raw.items():
        if any(p in prop_set and props.get(p) for p in INCIDENT_PROPS):
            candidates.append(inc_id)

    print(f"[2/3] {len(candidates):,} incidencias con al menos un campo evaluable")
    if not candidates:
        raise RuntimeError("No hay incidencias evaluables en el TTL.")

    rng = random.Random(seed)
    if n < len(candidates):
        chosen_ids = rng.sample(candidates, n)
    else:
        chosen_ids = candidates
        print(f"      [!] N solicitado ({n}) >= disponibles ({len(candidates)}). "
              f"Se exportan todas.")

    chosen_ids.sort()

    print(f"[3/3] Construyendo JSONL para {len(chosen_ids):,} incidencias ...")
    out_rows: list[dict] = []
    for inc_id in chosen_ids:
        props = raw[inc_id]
        incident_dict = {}
        for p in INCIDENT_PROPS:
            multi = p in MULTI_VALUE_PROPS
            incident_dict[p] = _flatten(props.get(p), multi)
        out_rows.append({
            "incident_id": inc_id,
            "source_file": ttl_path.name,
            "incident":    incident_dict,
        })
    return out_rows


def main():
    parser = argparse.ArgumentParser(
        description="Construye el JSONL de incidencias de evaluación desde test_eval.ttl"
    )
    parser.add_argument("--ttl",  type=Path, default=cfg.TEST_TTL,
                        help=f"TTL de entrada (default: {cfg.TEST_TTL})")
    parser.add_argument("--out",  type=Path,
                        default=cfg.DATA_DIR / "evaluacion",
                        help="Carpeta destino (default: data/evaluacion/)")
    parser.add_argument("--n",    type=int, default=500,
                        help="Nº de incidencias a exportar (default: 500)")
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED,
                        help=f"Semilla para muestreo (default: {cfg.RANDOM_SEED})")
    args = parser.parse_args()

    rows = build_eval_set(args.ttl, args.n, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"test_eval_{len(rows)}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Estadísticas
    n_skips_per_prop = {p: 0 for p in INCIDENT_PROPS}
    for row in rows:
        for p, v in row["incident"].items():
            if v == SKIP_MARK:
                n_skips_per_prop[p] += 1

    print(f"\n✓ Guardado: {out_path}  ({len(rows):,} incidencias)")
    print("\n  Skips por propiedad:")
    for p in INCIDENT_PROPS:
        n_skip = n_skips_per_prop[p]
        n_fill = len(rows) - n_skip
        print(f"    {p:<24}  filled={n_fill:>4}   skip={n_skip:>4}")


if __name__ == "__main__":
    main()
