"""
Pipeline de generación de corpus sintético de evaluación Q&A
a partir de un grafo de conocimiento RDF (filtrado.ttl).

Genera preguntas 1-hop y multi-hop en formato opción múltiple,
manteniendo las etiquetas originales de las entidades.

Salida: data/corpus/qa_corpus.json  (al menos 500 pares Q&A)
        data/corpus/qa_corpus.csv
        data/corpus/triples_verbalized.json
"""

import json
import random
import csv
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

from rdflib import Graph, Namespace, RDF, URIRef

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
TARGET_QA = 500          # Mínimo de pares Q&A a generar
N_DISTRACTORS = 3        # Número de opciones incorrectas en opción múltiple
N_MULTIHOP_MAX = 300     # Cuántas preguntas multi-hop generar como máximo

DATA_DIR = Path(__file__).parent.parent / "data"
TTL_FILE = DATA_DIR / "filtrado.ttl"
CORPUS_DIR = DATA_DIR / "corpus"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

REPCON = Namespace("http://repcon.org/schema#")

random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. PARSEO DEL GRAFO
# ---------------------------------------------------------------------------

def load_graph(ttl_path: Path) -> Graph:
    print(f"Cargando grafo desde {ttl_path} ...")
    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    print(f"      {len(g)} tripletas cargadas.")
    return g


def extract_label(uri: URIRef) -> str:
    """Extrae la parte local del URI (ej. 'employee__233')."""
    s = str(uri)
    # Fragment (#) o último segmento (/)
    if "#" in s:
        return s.split("#")[-1]
    return s.split("/")[-1]


def build_incident_map(g: Graph) -> dict:
    """
    Devuelve un dict:
        incident_label → {predicate_local: [object_label, ...], ...}
    Sólo incluye instancias de tipo repcon:incident.
    """
    print("Construyendo mapa de incidencias ...")
    incidents = {}
    for subj in g.subjects(RDF.type, REPCON.incident):
        label = extract_label(subj)
        props = defaultdict(list)
        for pred, obj in g.predicate_objects(subj):
            pred_local = extract_label(pred)
            if pred_local == "type":          # rdf:type ya lo usamos para filtrar
                continue
            obj_label = extract_label(obj)
            props[pred_local].append(obj_label)
        incidents[label] = dict(props)
    print(f"      {len(incidents)} incidencias encontradas.")
    return incidents


def build_entity_pools(incidents: dict) -> dict:
    """
    Reúne todos los valores únicos por tipo de predicado
    (sirven como pool de distractores).
    """
    pools = defaultdict(set)
    for props in incidents.values():
        for pred, values in props.items():
            for v in values:
                pools[pred].add(v)
    return {k: list(v) for k, v in pools.items()}


# ---------------------------------------------------------------------------
# 2. VERBALIZACIÓN DE TRIPLETAS
# ---------------------------------------------------------------------------

PRED_TEMPLATES_ES = {
    "hasStateIncident":      "La incidencia {s} tiene el estado {o}.",
    "hasTechnician":         "El técnico asignado a la incidencia {s} es {o}.",
    "hasExternalTechnician": "El técnico externo asignado a la incidencia {s} es {o}.",
    "hasTypeInc":            "El tipo de la incidencia {s} es {o}.",
    "incident_hasOrigin":    "El origen de la incidencia {s} es {o}.",
    "int_hasCustomer":       "El cliente de la incidencia {s} es {o}.",
    "hasSupportGroup":       "El grupo de soporte de la incidencia {s} es {o}.",
    "hasSupportTeam":        "El equipo de soporte de la incidencia {s} es {o}.",
    "hasSupportCategory":    "La categoría de soporte de la incidencia {s} es {o}.",
}


def verbalize_triples(incidents: dict) -> list[dict]:
    """
    Devuelve lista de dicts con la verbalización de cada tripleta.
    """
    print("Verbalizando tripletas ...")
    verbalized = []
    for inc_label, props in incidents.items():
        for pred, values in props.items():
            template = PRED_TEMPLATES_ES.get(pred)
            if template is None:
                continue
            for obj in values:
                verbalized.append({
                    "subject":    inc_label,
                    "predicate":  pred,
                    "object":     obj,
                    "verbalized": template.format(s=inc_label, o=obj),
                })
    print(f"      {len(verbalized)} tripletas verbalizadas.")
    return verbalized


# ---------------------------------------------------------------------------
# 3. GENERACIÓN DE PREGUNTAS
# ---------------------------------------------------------------------------

def pick_distractors(pool: list, correct: str, n: int) -> list:
    """Elige n distractores del pool distintos del valor correcto."""
    candidates = [x for x in pool if x != correct]
    if len(candidates) < n:
        return candidates
    return random.sample(candidates, n)


def format_mc_question(question: str, correct: str, distractors: list) -> dict:
    """
    Formatea la pregunta como opción múltiple con letras a/b/c/d.
    La opción correcta se inserta en posición aleatoria.
    """
    options = distractors[:N_DISTRACTORS] + [correct]
    random.shuffle(options)
    labels = ["a", "b", "c", "d"]
    options_dict = {labels[i]: options[i] for i in range(len(options))}
    correct_letter = [k for k, v in options_dict.items() if v == correct][0]
    options_str = "  ".join(f"{k}) {v}" for k, v in options_dict.items())
    full_question = f"{question}\n  {options_str}"
    return {
        "question":      full_question,
        "answer":        correct,
        "answer_letter": correct_letter,
        "options":       options_dict,
        "hop":           None,   # se rellena en el llamador
        "type":          None,
    }


# ── Plantillas 1-HOP ────────────────────────────────────────────────────────

Q1HOP_TEMPLATES = [
    # (predicado, plantilla de pregunta, etiqueta de tipo)
    ("hasTechnician",
     "¿Qué técnico está asignado a la incidencia {inc}?",
     "1hop_technician"),

    ("hasExternalTechnician",
     "¿Qué técnico externo está asignado a la incidencia {inc}?",
     "1hop_external_technician"),

    ("hasTypeInc",
     "¿Cuál es el tipo de la incidencia {inc}?",
     "1hop_type"),

    ("hasStateIncident",
     "¿Cuál es el estado de la incidencia {inc}?",
     "1hop_state"),

    ("incident_hasOrigin",
     "¿Cuál es el origen de la incidencia {inc}?",
     "1hop_origin"),

    ("int_hasCustomer",
     "¿A qué cliente corresponde la incidencia {inc}?",
     "1hop_customer"),

    ("hasSupportGroup",
     "¿A qué grupo de soporte pertenece la incidencia {inc}?",
     "1hop_support_group"),

    ("hasSupportTeam",
     "¿A qué equipo de soporte pertenece la incidencia {inc}?",
     "1hop_support_team"),

    ("hasSupportCategory",
     "¿Cuál es la categoría de soporte de la incidencia {inc}?",
     "1hop_support_category"),
]

# Variantes adicionales para aumentar volumen (sinónimos de pregunta)
Q1HOP_VARIANTS = {
    "hasTechnician": [
        "¿Quién es el técnico responsable de la incidencia {inc}?",
        "¿Qué empleado tiene asignada la incidencia {inc}?",
        "Indica el técnico que atiende la incidencia {inc}.",
    ],
    "hasTypeInc": [
        "¿De qué tipo es la incidencia {inc}?",
        "Indica el tipo al que corresponde la incidencia {inc}.",
    ],
    "int_hasCustomer": [
        "¿Qué empresa es cliente en la incidencia {inc}?",
        "¿Quién es el cliente de la incidencia {inc}?",
    ],
    "hasSupportGroup": [
        "¿Qué grupo gestiona la incidencia {inc}?",
    ],
}


def generate_1hop_qa(incidents: dict, pools: dict) -> list[dict]:
    """Genera preguntas 1-hop para cada incidencia × predicado disponible."""
    qa_list = []
    inc_labels = list(incidents.keys())

    for inc_label, props in incidents.items():
        for pred, q_template, q_type in Q1HOP_TEMPLATES:
            if pred not in props:
                continue
            for correct in props[pred]:
                distractors = pick_distractors(pools.get(pred, []), correct, N_DISTRACTORS)
                if not distractors:
                    continue

                # Pregunta base
                q_text = q_template.format(inc=inc_label)
                qa = format_mc_question(q_text, correct, distractors)
                qa["hop"] = 1
                qa["type"] = q_type
                qa["context_inc"] = inc_label
                qa_list.append(qa)

                # Variantes
                for variant_template in Q1HOP_VARIANTS.get(pred, []):
                    q_text_v = variant_template.format(inc=inc_label)
                    qa_v = format_mc_question(q_text_v, correct, distractors)
                    qa_v["hop"] = 1
                    qa_v["type"] = q_type + "_variant"
                    qa_v["context_inc"] = inc_label
                    qa_list.append(qa_v)

    return qa_list


# ── Cadenas MULTI-HOP ────────────────────────────────────────────────────────
#
# Cada cadena es una secuencia de pasos Q&A donde la respuesta del paso N
# se usa como contexto explícito en la pregunta del paso N+1.
# Esto replica el flujo conversacional del sistema "Akinator de ascensores":
#   paso 1: el sistema pregunta el tipo de incidencia
#   paso 2: con el tipo confirmado, pregunta el técnico recomendado
#   paso 3: con el técnico confirmado, pregunta el grupo de soporte
#
# Estructura de cada cadena:
# {
#   "chain_id": int,
#   "context_inc": str,
#   "n_hops": int,
#   "chain_type": str,
#   "steps": [
#     {"step": 1, "question": str, "answer": str,
#      "answer_letter": str, "options": dict},
#     {"step": 2, ...},
#     ...
#   ]
# }


def make_step(question_text: str, correct: str, pool: list) -> Optional[dict]:
    """Crea un paso de cadena con opción múltiple. Devuelve None si no hay distractores."""
    distractors = pick_distractors(pool, correct, N_DISTRACTORS)
    if not distractors:
        return None
    mc = format_mc_question(question_text, correct, distractors)
    return {
        "question":      mc["question"],
        "answer":        mc["answer"],
        "answer_letter": mc["answer_letter"],
        "options":       mc["options"],
    }


def generate_chains(incidents: dict, pools: dict) -> list[dict]:
    """
    Genera cadenas conversacionales multi-hop.
    Cada cadena tiene steps secuenciales donde step[i+1].question
    referencia explícitamente la respuesta de step[i].

    Patrones de 2 saltos (5 variantes):
      A: inc → tipo → técnico
      B: inc → cliente → grupo de soporte
      C: inc → tipo → grupo de soporte
      D: inc → origen → técnico
      E: inc → cliente → técnico

    Patrones de 3 saltos (2 variantes):
      F: inc → tipo → técnico → equipo de soporte
      G: inc → origen → tipo → técnico
    """
    chains = []
    chain_id = 0
    inc_items = list(incidents.items())
    random.shuffle(inc_items)

    counts = {p: 0 for p in ["A", "B", "C", "D", "E", "F", "G"]}
    per_pattern = N_MULTIHOP_MAX // 7  # cuántas cadenas por patrón

    for inc_label, props in inc_items:
        type_inc  = props.get("hasTypeInc",          [None])[0]
        emp       = props.get("hasTechnician",        [None])[0]
        customer  = props.get("int_hasCustomer",      [None])[0]
        group     = props.get("hasSupportGroup",      [None])[0]
        team      = props.get("hasSupportTeam",       [None])[0]
        origin    = props.get("incident_hasOrigin",   [None])[0]

        # ── Patrón A (2-hop): inc → tipo → técnico ────────────────────────
        # Flujo real: "¿qué tipo tiene la incidencia?" → confirma tipo →
        #             "¿qué técnico recomendamos para ese tipo?"
        if counts["A"] < per_pattern and type_inc and emp:
            s1 = make_step(
                f"¿Cuál es el tipo de la incidencia {inc_label}?",
                type_inc, pools.get("hasTypeInc", [])
            )
            s2 = make_step(
                f"Sabiendo que el tipo de incidencia es {type_inc}, "
                f"¿qué técnico está asignado a la incidencia {inc_label}?",
                emp, pools.get("hasTechnician", [])
            )
            if s1 and s2:
                chains.append({
                    "chain_id":   chain_id, "context_inc": inc_label,
                    "n_hops": 2,  "chain_type": "2hop_inc→type→tech",
                    "steps": [{**s1, "step": 1}, {**s2, "step": 2}],
                })
                chain_id += 1; counts["A"] += 1

        # ── Patrón B (2-hop): inc → cliente → grupo de soporte ────────────
        if counts["B"] < per_pattern and customer and group:
            s1 = make_step(
                f"¿A qué cliente corresponde la incidencia {inc_label}?",
                customer, pools.get("int_hasCustomer", [])
            )
            s2 = make_step(
                f"Dado que el cliente es {customer}, "
                f"¿qué grupo de soporte gestiona la incidencia {inc_label}?",
                group, pools.get("hasSupportGroup", [])
            )
            if s1 and s2:
                chains.append({
                    "chain_id":   chain_id, "context_inc": inc_label,
                    "n_hops": 2,  "chain_type": "2hop_inc→customer→group",
                    "steps": [{**s1, "step": 1}, {**s2, "step": 2}],
                })
                chain_id += 1; counts["B"] += 1

        # ── Patrón C (2-hop): inc → tipo → grupo de soporte ───────────────
        if counts["C"] < per_pattern and type_inc and group:
            s1 = make_step(
                f"¿Cuál es el tipo de la incidencia {inc_label}?",
                type_inc, pools.get("hasTypeInc", [])
            )
            s2 = make_step(
                f"Para una incidencia de tipo {type_inc}, "
                f"¿a qué grupo de soporte pertenece la incidencia {inc_label}?",
                group, pools.get("hasSupportGroup", [])
            )
            if s1 and s2:
                chains.append({
                    "chain_id":   chain_id, "context_inc": inc_label,
                    "n_hops": 2,  "chain_type": "2hop_inc→type→group",
                    "steps": [{**s1, "step": 1}, {**s2, "step": 2}],
                })
                chain_id += 1; counts["C"] += 1

        # ── Patrón D (2-hop): inc → origen → técnico ──────────────────────
        if counts["D"] < per_pattern and origin and emp:
            s1 = make_step(
                f"¿Cuál es el origen de la incidencia {inc_label}?",
                origin, pools.get("incident_hasOrigin", [])
            )
            s2 = make_step(
                f"Sabiendo que el origen de la incidencia es {origin}, "
                f"¿qué técnico está asignado a la incidencia {inc_label}?",
                emp, pools.get("hasTechnician", [])
            )
            if s1 and s2:
                chains.append({
                    "chain_id":   chain_id, "context_inc": inc_label,
                    "n_hops": 2,  "chain_type": "2hop_inc→origin→tech",
                    "steps": [{**s1, "step": 1}, {**s2, "step": 2}],
                })
                chain_id += 1; counts["D"] += 1

        # ── Patrón E (2-hop): inc → cliente → técnico ─────────────────────
        if counts["E"] < per_pattern and customer and emp:
            s1 = make_step(
                f"¿Qué cliente tiene la incidencia {inc_label}?",
                customer, pools.get("int_hasCustomer", [])
            )
            s2 = make_step(
                f"Dado que el cliente es {customer}, "
                f"¿qué técnico atiende la incidencia {inc_label}?",
                emp, pools.get("hasTechnician", [])
            )
            if s1 and s2:
                chains.append({
                    "chain_id":   chain_id, "context_inc": inc_label,
                    "n_hops": 2,  "chain_type": "2hop_inc→customer→tech",
                    "steps": [{**s1, "step": 1}, {**s2, "step": 2}],
                })
                chain_id += 1; counts["E"] += 1

        # ── Patrón F (3-hop): inc → tipo → técnico → equipo ───────────────
        # Flujo real del sistema: confirma tipo → asigna técnico → informa equipo
        if counts["F"] < per_pattern and type_inc and emp and team:
            s1 = make_step(
                f"¿Cuál es el tipo de la incidencia {inc_label}?",
                type_inc, pools.get("hasTypeInc", [])
            )
            s2 = make_step(
                f"Sabiendo que el tipo de incidencia es {type_inc}, "
                f"¿qué técnico está asignado a la incidencia {inc_label}?",
                emp, pools.get("hasTechnician", [])
            )
            s3 = make_step(
                f"Sabiendo que el técnico asignado es {emp}, "
                f"¿en qué equipo de soporte se gestiona la incidencia {inc_label}?",
                team, pools.get("hasSupportTeam", [])
            )
            if s1 and s2 and s3:
                chains.append({
                    "chain_id":   chain_id, "context_inc": inc_label,
                    "n_hops": 3,  "chain_type": "3hop_inc→type→tech→team",
                    "steps": [{**s1, "step": 1}, {**s2, "step": 2}, {**s3, "step": 3}],
                })
                chain_id += 1; counts["F"] += 1

        # ── Patrón G (3-hop): inc → origen → tipo → técnico ───────────────
        if counts["G"] < per_pattern and origin and type_inc and emp:
            s1 = make_step(
                f"¿Cuál es el origen de la incidencia {inc_label}?",
                origin, pools.get("incident_hasOrigin", [])
            )
            s2 = make_step(
                f"Con origen {origin}, ¿cuál es el tipo de la incidencia {inc_label}?",
                type_inc, pools.get("hasTypeInc", [])
            )
            s3 = make_step(
                f"Con tipo de incidencia {type_inc}, "
                f"¿qué técnico está asignado a la incidencia {inc_label}?",
                emp, pools.get("hasTechnician", [])
            )
            if s1 and s2 and s3:
                chains.append({
                    "chain_id":   chain_id, "context_inc": inc_label,
                    "n_hops": 3,  "chain_type": "3hop_inc→origin→type→tech",
                    "steps": [{**s1, "step": 1}, {**s2, "step": 2}, {**s3, "step": 3}],
                })
                chain_id += 1; counts["G"] += 1

        if all(v >= per_pattern for v in counts.values()):
            break

    return chains


# ---------------------------------------------------------------------------
# (OPCIONAL) 4. PARAFRASEO VÍA SERVIDOR vLLM
# ---------------------------------------------------------------------------

_vllm_paraphraser_client = None


def _load_paraphraser():
    """
    Devuelve un cliente OpenAI apuntando al servidor vLLM local.
    Requiere que vLLM esté corriendo (ver config.VLLM_BASE_URL).
    """
    global _vllm_paraphraser_client
    if _vllm_paraphraser_client is not None:
        return _vllm_paraphraser_client
    try:
        from openai import OpenAI
        _vllm_paraphraser_client = OpenAI(
            base_url=cfg.VLLM_BASE_URL, api_key="EMPTY"
        )
        return _vllm_paraphraser_client
    except Exception as e:
        print(f"[vLLM] No se pudo crear el cliente: {e}")
        return None


def _paraphrase_question(client, question_text: str) -> Optional[str]:
    """
    Pide al servidor vLLM que reformule el enunciado en español
    manteniendo los identificadores de entidad intactos.
    Devuelve el texto reformulado o None si falla / no cambia nada.
    """
    prompt = (
        "Reformula la siguiente pregunta en español de forma diferente, "
        "manteniendo EXACTAMENTE los mismos identificadores de entidad "
        "(no los traduzcas ni los modifiques). "
        "Responde SOLO con la pregunta reformulada, sin explicaciones.\n\n"
        f"Pregunta: {question_text}"
    )
    try:
        resp = client.chat.completions.create(
            model=cfg.DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.7,
        )
        result = resp.choices[0].message.content.strip()
        return result if result and result != question_text else None
    except Exception:
        return None


def paraphrase_1hop_with_hf(qa_list: list[dict], n_to_paraphrase: int = 0) -> list[dict]:
    """
    Parafrasea n_to_paraphrase pares 1-hop con flan-t5-small.
    Cada entrada nueva hereda las opciones/respuesta originales;
    solo cambia el enunciado de la pregunta y el tipo queda marcado
    con el sufijo '_paraphrase'.

    Activa con n_to_paraphrase > 0 (requiere RAM/GPU suficiente).
    """
    if n_to_paraphrase <= 0:
        return []
    paraphraser = _load_paraphraser()
    if paraphraser is None:
        return []

    print(f"[HF-1hop] Parafraseando {n_to_paraphrase} preguntas 1-hop ...")
    new_entries = []
    for item in random.sample(qa_list, min(n_to_paraphrase, len(qa_list))):
        original_q = item["question"].split("\n")[0]
        paraphrased = _paraphrase_question(paraphraser, original_q)
        if paraphrased:
            new_item = dict(item)
            opts_str = item["question"].split("\n", 1)[1] if "\n" in item["question"] else ""
            new_item["question"] = paraphrased + ("\n" + opts_str if opts_str else "")
            new_item["type"] = item["type"] + "_paraphrase"
            new_entries.append(new_item)

    print(f"[HF-1hop] {len(new_entries)} preguntas añadidas.")
    return new_entries


def paraphrase_chains_with_hf(chains: list[dict], n_to_paraphrase: int = 0) -> list[dict]:
    """
    Parafrasea n_to_paraphrase cadenas multi-hop con flan-t5-small.
    Para cada cadena seleccionada, reformula el enunciado de CADA paso
    de forma independiente manteniendo los nombres de entidad intactos
    (incluyendo las referencias al resultado del paso anterior, p.ej.
    "Sabiendo que el tipo es typeIncident__1, ...").
    Las opciones y respuestas de cada paso no cambian.
    Las cadenas nuevas reciben un chain_id nuevo y el tipo queda marcado
    con el sufijo '_paraphrase'.

    Activa con n_to_paraphrase > 0 (requiere RAM/GPU suficiente).
    """
    if n_to_paraphrase <= 0:
        return []
    paraphraser = _load_paraphraser()
    if paraphraser is None:
        return []

    print(f"[HF-chains] Parafraseando {n_to_paraphrase} cadenas ...")
    # chain_id de las nuevas cadenas: continúa desde el máximo existente
    next_id = max(c["chain_id"] for c in chains) + 1 if chains else 0
    new_chains = []

    for chain in random.sample(chains, min(n_to_paraphrase, len(chains))):
        new_steps = []
        for step in chain["steps"]:
            original_q = step["question"].split("\n")[0]
            paraphrased = _paraphrase_question(paraphraser, original_q)
            if paraphrased is None:
                paraphrased = original_q   # si falla, conserva el original
            opts_str = step["question"].split("\n", 1)[1] if "\n" in step["question"] else ""
            new_step = dict(step)
            new_step["question"] = paraphrased + ("\n" + opts_str if opts_str else "")
            new_steps.append(new_step)

        new_chains.append({
            "chain_id":   next_id,
            "context_inc": chain["context_inc"],
            "n_hops":     chain["n_hops"],
            "chain_type": chain["chain_type"] + "_paraphrase",
            "steps":      new_steps,
        })
        next_id += 1

    print(f"[HF-chains] {len(new_chains)} cadenas añadidas.")
    return new_chains


# ---------------------------------------------------------------------------
# 5. GUARDADO
# ---------------------------------------------------------------------------

def save_verbalized(verbalized: list[dict]) -> None:
    out_path = CORPUS_DIR / "triples_verbalized.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(verbalized, f, ensure_ascii=False, indent=2)
    print(f"      Verbalizaciones guardadas en {out_path}")


def save_qa_corpus(qa_1hop: list[dict], chains: list[dict]) -> None:
    # ── JSON unificado ──────────────────────────────────────────────────────
    unified = {
        "1hop":   qa_1hop,
        "chains": chains,
    }
    json_path = CORPUS_DIR / "qa_corpus.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(unified, f, ensure_ascii=False, indent=2)
    print(f"      Corpus JSON guardado en {json_path}")

    # ── CSV 1-hop (aplanado, una fila = un par Q&A) ─────────────────────────
    csv_1hop_path = CORPUS_DIR / "qa_1hop.csv"
    fieldnames_1hop = ["id", "hop", "type", "context_inc",
                       "question_text", "answer", "answer_letter",
                       "opt_a", "opt_b", "opt_c", "opt_d"]
    with open(csv_1hop_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_1hop)
        writer.writeheader()
        for i, item in enumerate(qa_1hop):
            opts = item.get("options", {})
            q_text = item["question"].split("\n")[0]
            writer.writerow({
                "id":            i,
                "hop":           item.get("hop"),
                "type":          item.get("type"),
                "context_inc":   item.get("context_inc"),
                "question_text": q_text,
                "answer":        item.get("answer"),
                "answer_letter": item.get("answer_letter"),
                "opt_a":         opts.get("a", ""),
                "opt_b":         opts.get("b", ""),
                "opt_c":         opts.get("c", ""),
                "opt_d":         opts.get("d", ""),
            })
    print(f"      1-hop CSV guardado en {csv_1hop_path}")

    # ── CSV cadenas aplanado (una fila = un paso dentro de una cadena) ───────
    # chain_id + step permite reconstruir la cadena completa agrupando por chain_id
    csv_chains_path = CORPUS_DIR / "qa_chains_flat.csv"
    fieldnames_chains = ["chain_id", "n_hops", "chain_type", "context_inc",
                         "step", "question_text", "answer", "answer_letter",
                         "opt_a", "opt_b", "opt_c", "opt_d"]
    with open(csv_chains_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_chains)
        writer.writeheader()
        for chain in chains:
            for step in chain["steps"]:
                opts = step.get("options", {})
                q_text = step["question"].split("\n")[0]
                writer.writerow({
                    "chain_id":      chain["chain_id"],
                    "n_hops":        chain["n_hops"],
                    "chain_type":    chain["chain_type"],
                    "context_inc":   chain["context_inc"],
                    "step":          step["step"],
                    "question_text": q_text,
                    "answer":        step.get("answer"),
                    "answer_letter": step.get("answer_letter"),
                    "opt_a":         opts.get("a", ""),
                    "opt_b":         opts.get("b", ""),
                    "opt_c":         opts.get("c", ""),
                    "opt_d":         opts.get("d", ""),
                })
    print(f"      Cadenas CSV guardado en {csv_chains_path}")


def print_stats(qa_1hop: list[dict], chains: list[dict]) -> None:
    by_type_1hop = defaultdict(int)
    for item in qa_1hop:
        by_type_1hop[item.get("type", "?")] += 1

    by_chain_type = defaultdict(int)
    by_nhops = defaultdict(int)
    for chain in chains:
        by_chain_type[chain["chain_type"]] += 1
        by_nhops[chain["n_hops"]] += 1

    total_steps = sum(len(c["steps"]) for c in chains)

    print("\n" + "=" * 65)
    print(f"  CORPUS FINAL")
    print("=" * 65)
    print(f"  1-hop Q&A pares:   {len(qa_1hop)}")
    print(f"  Cadenas multi-hop: {len(chains)}  ({total_steps} pasos totales)")
    print(f"    2-hop cadenas:   {by_nhops[2]}")
    print(f"    3-hop cadenas:   {by_nhops[3]}")
    print("-" * 65)
    print("  Tipos 1-hop:")
    for t, n in sorted(by_type_1hop.items(), key=lambda x: -x[1]):
        print(f"    {t:<48} {n:>5}")
    print("  Tipos de cadenas:")
    for t, n in sorted(by_chain_type.items(), key=lambda x: -x[1]):
        print(f"    {t:<48} {n:>5}")
    print("=" * 65)


def print_sample_1hop(qa_1hop: list[dict], n: int = 3) -> None:
    print(f"\n── Ejemplos 1-hop ({n}) ──")
    for item in random.sample(qa_1hop, min(n, len(qa_1hop))):
        print(f"\n  [{item['type']}]  incidencia: {item.get('context_inc')}")
        print(f"  P: {item['question']}")
        print(f"  R: {item['answer']} ({item['answer_letter']})")


def print_sample_chains(chains: list[dict], n: int = 2) -> None:
    print(f"\n── Ejemplos de cadenas ({n}) ──")
    for chain in random.sample(chains, min(n, len(chains))):
        print(f"\n  chain_id={chain['chain_id']}  [{chain['chain_type']}]"
              f"  incidencia: {chain['context_inc']}")
        for step in chain["steps"]:
            q_text = step["question"].split("\n")[0]
            opts_str = step["question"].split("\n")[1] if "\n" in step["question"] else ""
            print(f"  Paso {step['step']}: {q_text}")
            if opts_str:
                print(f"          {opts_str.strip()}")
            print(f"          → {step['answer']} ({step['answer_letter']})")


# ---------------------------------------------------------------------------
# 6. CORPUS DE EVALUACIÓN LINK PREDICTION (por modelo KGE)
# ---------------------------------------------------------------------------

# Todas las relaciones del grafo con sus plantillas de pregunta (sin opciones múltiples)
_LP_RELATIONS = {
    "hasTypeInc":            "¿Cuál es el tipo de la incidencia {inc}?",
    "hasStateIncident":      "¿Cuál es el estado de la incidencia {inc}?",
    "hasTechnician":         "¿Qué técnico está asignado a la incidencia {inc}?",
    "hasExternalTechnician": "¿Qué técnico externo está asignado a la incidencia {inc}?",
    "incident_hasOrigin":    "¿Cuál es el origen de la incidencia {inc}?",
    "int_hasCustomer":       "¿A qué cliente corresponde la incidencia {inc}?",
    "hasSupportGroup":       "¿Cuál es el grupo de soporte de la incidencia {inc}?",
    "hasSupportTeam":        "¿Cuál es el equipo de soporte de la incidencia {inc}?",
    "hasSupportCategory":    "¿Cuál es la categoría de soporte de la incidencia {inc}?",
}
_LP_SAMPLES_PER_REL = 200


def generate_link_prediction_eval_corpus(
    ttl_path: Path = None,
    out_path: Path = None,
) -> list[dict]:
    """
    Genera data/corpus/link_prediction_eval.json para comparar modelos KGE.

    Cada entrada tiene:
      {
        "id":          "lp_0001",
        "subject":     "incident_X",
        "predicate":   "hasTypeInc",
        "object_true": "typeIncident__1",
        "question":    "¿Cuál es el tipo de la incidencia incident_X?"
      }

    Se generan hasta _LP_SAMPLES_PER_REL entradas por relación
    (~600 entradas en total para 3 relaciones).
    """
    if ttl_path is None:
        ttl_path = TTL_FILE
    if out_path is None:
        out_path = cfg.LP_EVAL_CORPUS

    g = load_graph(ttl_path)
    incidents = build_incident_map(g)

    entries = []
    entry_id = 0

    for predicate, question_tmpl in _LP_RELATIONS.items():
        # Recoger todos los pares (incidencia, objeto) para este predicado
        pairs = [
            (inc_label, values[0])
            for inc_label, props in incidents.items()
            if predicate in props and props[predicate]
            for values in [props[predicate]]
        ]
        # Muestrear hasta _LP_SAMPLES_PER_REL pares de forma reproducible
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(pairs)
        pairs = pairs[:_LP_SAMPLES_PER_REL]

        for inc_label, obj in pairs:
            entries.append({
                "id":          f"lp_{entry_id:04d}",
                "subject":     inc_label,
                "predicate":   predicate,
                "object_true": obj,
                "question":    question_tmpl.format(inc=inc_label),
            })
            entry_id += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"      LP eval corpus: {len(entries)} entradas → {out_path}")
    by_pred = defaultdict(int)
    for e in entries:
        by_pred[e["predicate"]] += 1
    for pred, n in sorted(by_pred.items()):
        print(f"        {pred}: {n}")

    return entries


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # 1. Cargar grafo
    g = load_graph(TTL_FILE)

    # 2. Extraer mapa de incidencias
    incidents = build_incident_map(g)
    pools = build_entity_pools(incidents)

    # 3. Verbalizar tripletas
    verbalized = verbalize_triples(incidents)
    save_verbalized(verbalized)

    # 4. Generar Q&A
    print("Generando preguntas Q&A ...")

    # 1-hop: pares pregunta-respuesta directos
    qa_1hop = generate_1hop_qa(incidents, pools)
    print(f"      1-hop generadas: {len(qa_1hop)}")

    # Limitar 1-hop a 3500 de forma estratificada para no inflar el corpus
    if len(qa_1hop) > 3500:
        by_type: dict[str, list] = defaultdict(list)
        for item in qa_1hop:
            by_type[item["type"]].append(item)
        qa_1hop = []
        per_type = max(1, 3500 // len(by_type))
        for items in by_type.values():
            qa_1hop.extend(random.sample(items, min(per_type, len(items))))
        random.shuffle(qa_1hop)
        print(f"      1-hop tras muestreo: {len(qa_1hop)}")

    # (Opcional) Parafraseo con HuggingFace — activa poniendo n > 0
    # Requiere RAM/GPU suficiente para cargar flan-t5-small (~300 MB)
    qa_1hop += paraphrase_1hop_with_hf(qa_1hop, n_to_paraphrase=200)

    # Multi-hop: cadenas conversacionales secuenciales
    chains = generate_chains(incidents, pools)
    print(f"      cadenas multi-hop: {len(chains)}")

    chains += paraphrase_chains_with_hf(chains, n_to_paraphrase=200)

    total_qa = len(qa_1hop) + sum(len(c["steps"]) for c in chains)
    if total_qa < TARGET_QA:
        print(f"[WARNING] Solo se generaron {total_qa} items Q&A (objetivo: {TARGET_QA}). "
              "El grafo puede ser demasiado pequeño.")

    # 5. Guardar
    print("Guardando corpus ...")
    save_qa_corpus(qa_1hop, chains)

    print_stats(qa_1hop, chains)
    print_sample_1hop(qa_1hop, n=3)
    print_sample_chains(chains, n=2)

    # 6. Corpus de evaluación link prediction
    print("\nGenerando corpus de evaluación link prediction ...")
    generate_link_prediction_eval_corpus()

    print(f"\n✓ Pipeline completado. Ficheros en: {CORPUS_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generación de corpus sintético Q&A y LP eval")
    parser.add_argument("--lp-only", action="store_true",
                        help="Generar solo link_prediction_eval.json (rápido, sin Q&A)")
    args = parser.parse_args()

    if args.lp_only:
        print("Generando solo corpus de evaluación link prediction ...")
        generate_link_prediction_eval_corpus()
        print(f"\n✓ Completado. Fichero en: {cfg.LP_EVAL_CORPUS}")
    else:
        main()
