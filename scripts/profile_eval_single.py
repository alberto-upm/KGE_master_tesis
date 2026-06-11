"""
Profile rápido de una sola incidencia para identificar dónde se va el tiempo
real en la evaluación. Imprime tiempo por componente (CBR matching, KGE batched,
PyClause query, aggregation) en cada uno de los campos de la cascada.

Uso:
    python scripts/profile_eval_single.py
    python scripts/profile_eval_single.py --eval-jsonl data/evaluacion/test_eval_500.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config as cfg
from phase5_incident_creator import (
    INCIDENT_PROPS, MULTI_VALUE_PROPS,
    _build_incidents_map_from_tsv, build_incidents_index,
    find_matching_incidents,
)
from phase3_link_prediction import (
    load_model_by_name, predict_tails_batch, _factory_cache,
)
from utils.rule_engine import RuleEnginePyClause


SKIP = "skip"
USER_PROVIDED = {"int_hasCustomer"}
EVAL_PROPS = [p for p in INCIDENT_PROPS if p not in MULTI_VALUE_PROPS and p not in USER_PROVIDED]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", type=Path,
                    default=cfg.DATA_DIR / "evaluacion" / "test_eval_500.jsonl")
    ap.add_argument("--kge-model", default="TransE")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    def lap(label, t0):
        dt = time.perf_counter() - t0
        print(f"  {label:<40} {dt*1000:>8.1f} ms")
        return time.perf_counter()

    print(f"\n=== Carga de recursos ===")
    t0 = time.perf_counter()
    incidents_map = _build_incidents_map_from_tsv()
    t0 = lap(f"_build_incidents_map_from_tsv (N={len(incidents_map):,})", t0)

    cbr_index = build_incidents_index(incidents_map)
    t0 = lap("build_incidents_index", t0)

    rule_engine = RuleEnginePyClause()
    t0 = lap("RuleEnginePyClause.__init__", t0)

    model, factory = load_model_by_name(args.kge_model)
    t0 = lap(f"load_model_by_name({args.kge_model})", t0)

    # warm cache id_to_ent
    _ = _factory_cache(factory)
    t0 = lap("_factory_cache (id_to_ent)", t0)

    # primera incidencia del JSONL
    with open(args.eval_jsonl) as f:
        row = json.loads(f.readline())
    inc_id = row["incident_id"]
    gt     = row["incident"]
    print(f"\n=== Incidencia: {inc_id} ===")

    known = {p: None for p in INCIDENT_PROPS}
    cust = gt.get("int_hasCustomer")
    if cust and cust != SKIP:
        known["int_hasCustomer"] = cust

    incident_t0 = time.perf_counter()

    for prop in EVAL_PROPS:
        v = gt.get(prop)
        if v is None or v == SKIP or not v:
            continue
        if isinstance(v, list):
            v = v[0]

        print(f"\n--- {prop} (gt={v}) ---")
        t0 = time.perf_counter()
        rule = rule_engine.query(known, prop)
        t0 = lap(f"rule_engine.query[{prop}]", t0)

        if rule and rule.get("value") == v:
            print(f"  → rule_hit (rank=1)")
            known[prop] = v
            continue

        proxies = find_matching_incidents(
            known, incidents_map, index=cbr_index, exclude_id=inc_id
        )
        t0 = lap(f"find_matching_incidents (#proxies={len(proxies)})", t0)

        per_proxy = predict_tails_batch(
            model, factory, proxies[:30], prop, args.top_k
        )
        t0 = lap(f"predict_tails_batch (B={min(30, len(proxies))})", t0)

        # agregación WRRF (la parte rápida)
        scores: dict[str, list[float]] = {}
        for ranked in per_proxy:
            for ent, sc in ranked:
                scores.setdefault(ent, []).append(sc)
        t0 = lap("aggregate scores", t0)

        known[prop] = v

    total = time.perf_counter() - incident_t0
    print(f"\n=== Total incidencia: {total*1000:.1f} ms ({total:.2f} s) ===\n")


if __name__ == "__main__":
    main()
