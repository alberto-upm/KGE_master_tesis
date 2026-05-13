"""
Motor de reglas simbólico basado en PyClause para inferencia sobre el KG de incidencias.

Capa 3 — Inferencia en cascada (parte REGLA):
  - Carga las reglas AnyBURL de rules-1000-3 una sola vez al inicio.
  - query(incident, prop) construye un KG temporal con los hechos conocidos
    de la incidencia en construcción y usa PyClause QAHandler para comprobar
    si alguna regla puede inferir el valor de prop.
  - Devuelve un objeto de trazabilidad completo:
      {"value": "typeIncident__1", "source": "RULE", "rule_id": "r_0042", "confidence": 0.91}
    o None si no hay regla aplicable → la cascada pasa al KGE+CBR.

Interfaz pública (compatible con el antiguo RuleEngine):
    engine = RuleEnginePyClause()
    engine.stats()          → {"total_rules": N, "predicates": M}
    engine.query(incident, prop) → dict | None
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import sys

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

RULES_FILE = cfg.DATA_DIR / "reglas" / "rules-1000-3"
_QUERY_ENTITY = "incident__QUERY"


# ---------------------------------------------------------------------------
# Parseo del fichero de reglas AnyBURL
# ---------------------------------------------------------------------------

def _extract_body_relations(body_part: str) -> set[str]:
    """Extrae los nombres de predicado de todos los átomos del cuerpo de una regla."""
    rels: set[str] = set()
    for atom in body_part.split("),"):
        atom = atom.strip().rstrip(")")
        pred = atom.partition("(")[0].strip()
        if pred:
            rels.add(pred)
    return rels


def _parse_rules_file(
    path: Path,
) -> tuple[list[str], list[list[int]], list[float], list[set[str]], list[str]]:
    """
    Parsea el fichero de reglas AnyBURL.
    Formato por línea: num_preds<TAB>support<TAB>confidence<TAB>rule_string

    Devuelve (rules, stats, confidences, body_relations, head_preds) paralelas.
    - body_relations[i]: predicados del cuerpo de la regla i (filtrado por relaciones conocidas)
    - head_preds[i]: predicado de la cabeza de la regla i (filtrado por target_prop)
    """
    rules: list[str] = []
    stats: list[list[int]] = []
    confidences: list[float] = []
    body_rels: list[set[str]] = []
    head_preds: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 3)
            if len(parts) < 4:
                continue
            try:
                num_preds = int(parts[0])
                support   = int(parts[1])
                conf      = float(parts[2])
                rule_str  = parts[3]
            except (ValueError, IndexError):
                continue
            head_part, sep, body_part = rule_str.partition(" <= ")
            if not sep:
                continue
            h_pred = head_part.partition("(")[0].strip()
            rules.append(rule_str)
            stats.append([num_preds, support])
            confidences.append(conf)
            body_rels.append(_extract_body_relations(body_part))
            head_preds.append(h_pred)
    return rules, stats, confidences, body_rels, head_preds


# ---------------------------------------------------------------------------
# Trazabilidad: qué regla produjo la predicción
# ---------------------------------------------------------------------------

def _find_rule(
    rules: list[str],
    confidences: list[float],
    target_prop: str,
    predicted_val: str,
    known: dict[str, str],
) -> tuple[str, float]:
    """
    Busca la regla de mayor confianza que:
      - tiene target_prop(X, predicted_val) como cabeza, y
      - cuyo cuerpo está satisfecho por los hechos conocidos.

    Retorna (rule_id, confidence). Sólo soporta reglas de cuerpo simple (un átomo)
    con valores constantes en la cabeza, que son la mayoría en AnyBURL.
    """
    best_id = "r_unknown"
    best_conf = 0.0

    for idx, rule_str in enumerate(rules):
        head_part, sep, body_part = rule_str.partition(" <= ")
        if not sep:
            continue

        # Parsear cabeza: predicate(X,value)
        h_pred, _, h_rest = head_part.partition("(")
        if h_pred != target_prop:
            continue
        h_args = h_rest.rstrip(")").split(",", 1)
        if len(h_args) < 2:
            continue
        h_val = h_args[1].strip()
        # Sólo reglas con valor constante en la cabeza (no variable Y)
        if h_val != predicted_val:
            continue

        # Parsear cuerpo: un átomo o cadena de átomos separados por coma
        # Para reglas de cuerpo simple: pred(X,const)
        satisfied = _body_satisfied(body_part, known)
        if satisfied:
            conf = confidences[idx]
            if conf > best_conf:
                best_conf = conf
                best_id = f"r_{idx:04d}"

    return best_id, best_conf


def _body_satisfied(body_part: str, known: dict[str, str]) -> bool:
    """
    Comprueba si el cuerpo de una regla está satisfecho por los hechos conocidos.
    Trata cada átomo pred(X,val) donde val es una constante.
    Ignora átomos con variables (A, B, Y...) en los argumentos — cadenas.
    """
    # Dividir en átomos (terminan en ")")
    # Ejemplo: "hasSupportCategory(X,supportCategory_1) , hasSupportGroup(X,sg_2)"
    raw_atoms = body_part.split("),")
    for atom in raw_atoms:
        atom = atom.strip().rstrip(")")
        if not atom:
            continue
        b_pred, _, b_rest = atom.partition("(")
        b_args = b_rest.split(",", 1)
        if len(b_args) < 2:
            return False
        b_subj = b_args[0].strip()
        b_val  = b_args[1].strip()
        # Omitir si el valor es una variable (letra mayúscula sola o A,B,...)
        if len(b_val) <= 2 and b_val[0].isupper():
            continue
        # El sujeto siempre debería ser X (la incidencia); si no, regla compleja
        if b_subj != "X":
            return False
        # Comprobar que el hecho está en los conocidos
        if known.get(b_pred) != b_val:
            return False
    return True


# ---------------------------------------------------------------------------
# Motor de reglas PyClause
# ---------------------------------------------------------------------------

class RuleEnginePyClause:
    """
    Motor de reglas simbólico con PyClause — Capa REGLA de la inferencia en cascada.

    Prioridad: REGLA → KGE+CBR (si la regla falla o no existe).
    Cada respuesta incluye trazabilidad completa con rule_id y confidence.
    """

    def __init__(self, rules_path: Path = RULES_FILE):
        try:
            from c_clause import QAHandler, Loader
            from clause import Options
            self._pyclause_available = True
            self._Loader   = Loader
            self._QAHandler = QAHandler
            opts = Options()
            opts.set("qa_handler.aggregation_function", "maxplus")
            self._loader_opts = opts.get("loader")
            self._qa_opts     = opts.get("qa_handler")
        except ImportError:
            self._pyclause_available = False

        self._rules_path = rules_path
        self._rules, self._stats, self._confidences, self._body_rels, self._head_preds = _parse_rules_file(rules_path)

        self._n_rules   = len(self._rules)
        self._predicates = len({
            r.partition("(")[0] for r in self._rules if "(" in r
        })

        # Pre-construir el dataset "mundo" con todos los pares (relación, entidad)
        # que aparecen en las reglas. Esto permite que PyClause reconozca todas
        # las entidades y relaciones al cargar las reglas, sin necesitar el KG completo.
        # Se usa la entidad especial "__world__" como sujeto para no interferir con
        # las consultas sobre _QUERY_ENTITY.
        self._world_triples = self._build_world_triples()

    # ------------------------------------------------------------------

    def _build_world_triples(self) -> list[tuple[str, str, str]]:
        """
        Extrae todos los pares (relación, entidad) que aparecen en las reglas
        y los convierte en triples semilla con sujeto "__world__".

        PyClause exige que todas las entidades y relaciones de las reglas cargadas
        estén presentes en los datos. Este "mundo" permite cargar todas las reglas
        sin necesitar el KG histórico completo.
        """
        seen: set[tuple[str, str]] = set()
        triples: list[tuple[str, str, str]] = []
        for rule_str in self._rules:
            head_part, sep, body_part = rule_str.partition(" <= ")
            if not sep:
                continue
            # Cabeza
            h_pred, _, h_rest = head_part.partition("(")
            h_args = h_rest.rstrip(")").split(",", 1)
            if len(h_args) == 2:
                h_val = h_args[1].strip()
                if h_val and not (len(h_val) <= 2 and h_val[0].isupper()):
                    key = (h_pred, h_val)
                    if key not in seen:
                        seen.add(key)
                        triples.append(("__world__", h_pred, h_val))
            # Cuerpo
            for atom in body_part.split("),"):
                atom = atom.strip().rstrip(")")
                b_pred, _, b_rest = atom.partition("(")
                b_args = b_rest.split(",", 1)
                if len(b_args) == 2:
                    b_val = b_args[1].strip()
                    if b_val and not (len(b_val) <= 2 and b_val[0].isupper()):
                        key = (b_pred, b_val)
                        if key not in seen:
                            seen.add(key)
                            triples.append(("__world__", b_pred, b_val))
        return triples

    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {"total_rules": self._n_rules, "predicates": self._predicates}

    # ------------------------------------------------------------------

    def query(self, incident: dict, target_prop: str) -> Optional[dict]:
        """
        Intenta inferir target_prop usando las reglas simbólicas (PyClause).

        Construye un KG temporal = [mundo semilla] + [hechos conocidos del incidente]
        y consulta PyClause. Usa el fallback Python si PyClause no está disponible.

        Devuelve:
            {"value": "...", "source": "RULE", "rule_id": "r_NNNN", "confidence": 0.91}
        o None si no hay regla aplicable.
        """
        known = {k: v for k, v in incident.items() if v is not None and k != target_prop}
        if not known:
            return None

        if not self._pyclause_available:
            return self._fallback_query(known, target_prop)

        # Filtrar a reglas con cabeza == target_prop cuyo cuerpo esté en los hechos conocidos.
        # El mundo semilla ya garantiza que todos los nodos/relaciones son reconocidos.
        known_rels = set(known.keys())
        indices = [
            i for i, (h_pred, brels) in enumerate(zip(self._head_preds, self._body_rels))
            if h_pred == target_prop and brels.issubset(known_rels)
        ]
        if not indices:
            return None

        filtered_rules = [self._rules[i] for i in indices]
        filtered_stats = [self._stats[i] for i in indices]

        # KG temporal = mundo semilla (todas las entidades/relaciones de las reglas)
        #             + hechos conocidos del incidente en construcción
        data = list(self._world_triples)
        for rel, val in known.items():
            data.append((_QUERY_ENTITY, rel, val))

        try:
            loader = self._Loader(options=self._loader_opts)
            loader.load_data(data)
            loader.load_rules(rules=filtered_rules, stats=filtered_stats)

            qa = self._QAHandler(options=self._qa_opts)
            qa.calculate_answers(
                queries=[(_QUERY_ENTITY, target_prop)],
                loader=loader,
                direction="tail",
            )

            answers = qa.get_answers(as_string=True)
        except Exception:
            return self._fallback_query(known, target_prop)

        if not answers or not answers[0]:
            return None

        # Filtrar respuestas artificiales del mundo semilla (__world__, __seed_val__, etc.)
        real_answers = [
            (ent, score) for ent, score in answers[0]
            if not ent.startswith("__")
        ]
        if not real_answers:
            return None

        best_entity, best_score = real_answers[0]

        # Trazabilidad: buscar entre las reglas filtradas cuál disparó la predicción
        filtered_confs = [self._confidences[i] for i in indices]
        rule_id, confidence = _find_rule(
            filtered_rules, filtered_confs,
            target_prop, best_entity, known,
        )
        # Mapear el índice local al índice global para el rule_id
        if rule_id != "r_unknown":
            local_idx = int(rule_id[2:])
            rule_id = f"r_{indices[local_idx]:04d}"

        return {
            "value":      best_entity,
            "source":     "RULE",
            "rule_id":    rule_id,
            "confidence": confidence if confidence > 0.0 else float(best_score),
        }

    # ------------------------------------------------------------------

    def _fallback_query(self, known: dict[str, str], target_prop: str) -> Optional[dict]:
        """
        Fallback sin PyClause: comprueba directamente las reglas de cuerpo simple.
        Devuelve la regla de mayor confianza que se satisfaga, o None.
        """
        best: Optional[dict] = None
        for idx, rule_str in enumerate(self._rules):
            head_part, sep, body_part = rule_str.partition(" <= ")
            if not sep:
                continue
            h_pred, _, h_rest = head_part.partition("(")
            if h_pred != target_prop:
                continue
            h_args = h_rest.rstrip(")").split(",", 1)
            if len(h_args) < 2:
                continue
            h_val = h_args[1].strip()
            if len(h_val) <= 2 and h_val[0].isupper():
                continue  # variable en cabeza — regla de cadena
            if _body_satisfied(body_part, known):
                conf = self._confidences[idx]
                if best is None or conf > best["confidence"]:
                    best = {
                        "value":      h_val,
                        "source":     "RULE",
                        "rule_id":    f"r_{idx:04d}",
                        "confidence": conf,
                    }
        return best
