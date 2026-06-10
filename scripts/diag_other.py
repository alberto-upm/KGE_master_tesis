"""Diagnóstico: ¿qué entidades caen en la categoría 'other' del meta-grafo?

Recorre el TTL en streaming (sin rdflib) replicando la lógica de
_entity_type / subject_type de scripts/analisis_previo.py, y lista las
entidades clasificadas como 'other' agrupadas por su prefijo real.
"""
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT = ROOT / "data" / "incident_triplets.ttl"

# --- misma clasificación que analisis_previo.py ---------------------------
_TYPE_PREFIXES = [
    ("incident_",       "incident"),
    ("intervention_",   "intervention"),
    ("employee",        "employee"),
    ("company",         "company"),
    ("supportGroup",    "supportGroup"),
    ("supportTeam",     "supportTeam"),
    ("supportCategory", "supportCategory"),
    ("statusIncident",  "status"),
    ("typeIncident",    "type"),
    ("incidentOrigin",  "origin"),
    ("person_",         "person"),
    ("bu_",             "businessUnit"),
]
_KNOWN = {
    "incident", "intervention", "employee", "company", "supportGroup",
    "supportTeam", "supportCategory", "status", "type", "origin",
    "person", "businessUnit", "other",
}


def _entity_type(label: str) -> str:
    for pref, etype in _TYPE_PREFIXES:
        if label.startswith(pref):
            return etype
    return "other"


# token repcon:localname  (localname permite letras, dígitos, _, -)
TOK = re.compile(r"repcon:([A-Za-z0-9_\-.]+)")


def main(path: Path) -> None:
    subject_type: dict[str, str] = {}
    entity_freq: Counter = Counter()   # nº de tripletas en que aparece la entidad
    cur_subj: str | None = None

    print(f"Leyendo {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("@"):
                continue
            toks = TOK.findall(line)
            if not toks:
                continue
            # ¿línea de sujeto nuevo? (no empieza con espacio)
            if not line[0].isspace():
                cur_subj = toks[0]
                preds_objs = toks[1:]
            else:
                preds_objs = toks
            entity_freq[cur_subj] += 1
            # pares (pred, obj)
            for i in range(0, len(preds_objs) - 1, 2):
                pred, obj = preds_objs[i], preds_objs[i + 1]
                if pred == "type":
                    subject_type[cur_subj] = obj
                else:
                    entity_freq[obj] += 1

    # clasificar cada entidad distinta
    other_freq: Counter = Counter()
    type_count: Counter = Counter()
    for ent, freq in entity_freq.items():
        etype = subject_type.get(ent) or _entity_type(ent)
        if etype not in _KNOWN:
            etype = _entity_type(ent)
        type_count[etype] += 1
        if etype == "other":
            other_freq[ent] += freq

    print(f"\nEntidades distintas: {len(entity_freq):,}")
    print("\n--- Conteo de entidades por tipo ---")
    for t, n in type_count.most_common():
        print(f"  {t:16s} {n:>8,}")

    print(f"\n--- 'other': {type_count.get('other', 0):,} entidades distintas ---")
    # agrupar por raíz (parte sin dígitos finales) para ver patrones
    pattern: Counter = Counter()
    for ent in other_freq:
        root = re.sub(r"[0-9]+$", "<N>", ent)
        root = re.sub(r"^[0-9]+$", "<NUM>", root)
        pattern[root] += 1
    print("\nPatrones de label (raíz → nº de labels distintos):")
    for pat, n in pattern.most_common(40):
        print(f"  {pat:30s} {n:>8,}")

    print("\nTop 40 entidades 'other' por frecuencia de aparición:")
    for ent, n in other_freq.most_common(40):
        declared = subject_type.get(ent)
        extra = f"  (type={declared})" if declared else ""
        print(f"  {ent:30s} {n:>8,}{extra}")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    main(p)
