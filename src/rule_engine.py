"""
RuleEngine: carga y aplica las reglas Datalog de rules-1000-3.

Formato del fichero (una regla por línea):
  total  true  confidence  head_pred(X,head_val) <= body1(X,val1)[, body2(X,val2), ...]

Uso:
  engine = RuleEngine()
  result = engine.query(known_props, "hasTypeInc")
  # → {"value": "typeIncident__2", "source": "RULE", "rule_id": "r_0004", "confidence": 0.7965}
  # → None  si ninguna regla aplica
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_RULES_FILE = Path(__file__).parent.parent / "data" / "reglas" / "rules-1000-3"
_ATOM_RE = re.compile(r'(\w+)\(X,([^)]+)\)')


@dataclass(slots=True)
class Rule:
    rule_id: str
    head_pred: str
    head_val: str
    body: list[tuple[str, str]]   # [(pred, val), ...]
    confidence: float
    total: int
    true_groundings: int


def _parse_line(line_no: int, line: str) -> Optional[Rule]:
    parts = line.strip().split(None, 3)
    if len(parts) < 4:
        return None
    try:
        total = int(parts[0])
        true_g = int(parts[1])
        conf = float(parts[2])
    except ValueError:
        return None
    rule_text = parts[3]
    if "<=" not in rule_text:
        return None
    head_str, body_str = rule_text.split("<=", 1)
    head_match = _ATOM_RE.match(head_str.strip())
    if not head_match:
        return None
    body_atoms = _ATOM_RE.findall(body_str)
    if not body_atoms:
        return None
    return Rule(
        rule_id=f"r_{line_no:04d}",
        head_pred=head_match.group(1),
        head_val=head_match.group(2),
        body=[(pred, val) for pred, val in body_atoms],
        confidence=conf,
        total=total,
        true_groundings=true_g,
    )


class RuleEngine:
    """
    Motor de reglas Datalog para inferencia en cascada.

    Indexa las reglas por predicado de cabeza para lookup O(1).
    Dentro de cada bucket las reglas están ordenadas por confianza desc,
    por lo que `query` devuelve siempre la mejor regla aplicable.
    """

    def __init__(self, rules_path: Path = _RULES_FILE):
        # head_pred → [Rule]  (ordenado por confidence DESC)
        self._index: dict[str, list[Rule]] = {}
        self._load(rules_path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            print(f"  [!] RuleEngine: fichero de reglas no encontrado: {path}")
            return
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                rule = _parse_line(i, line)
                if rule:
                    self._index.setdefault(rule.head_pred, []).append(rule)
        for bucket in self._index.values():
            bucket.sort(key=lambda r: r.confidence, reverse=True)

    def query(
        self,
        known_props: dict[str, str | None],
        target_pred: str,
    ) -> Optional[dict]:
        """
        Busca la regla de mayor confianza cuya cabeza sea `target_pred` y cuyo
        cuerpo esté completamente satisfecho por `known_props`.

        Devuelve un objeto de trazabilidad o None si ninguna regla aplica:
          {
            "value":      str,    # valor sugerido por la regla
            "source":     "RULE",
            "rule_id":    str,    # p.ej. "r_0004"
            "confidence": float,
          }
        """
        for rule in self._index.get(target_pred, []):
            if all(known_props.get(pred) == val for pred, val in rule.body):
                return {
                    "value":      rule.head_val,
                    "source":     "RULE",
                    "rule_id":    rule.rule_id,
                    "confidence": rule.confidence,
                }
        return None

    def stats(self) -> dict:
        total = sum(len(v) for v in self._index.values())
        return {"predicates": len(self._index), "total_rules": total}
