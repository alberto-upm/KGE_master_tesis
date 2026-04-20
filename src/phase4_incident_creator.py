"""
Creador guiado de incidencias con CBR + KGE + LLM conversacional.

Flujo:
  1. El usuario describe el problema en texto libre → se extraen entidades conocidas
     mediante lookup contra el grafo (company__X, employee__N, typeIncident__N, ...)
  2. Para cada propiedad pendiente:
     a. KGE (CBR + predict_tails) genera recomendaciones basadas en incidencias similares
     b. Si LLM disponible: el LLM formula una pregunta natural con las opciones KGE
        como contexto. El usuario responde libremente y el LLM extrae el identificador.
     c. Si LLM no disponible: menú numerado clásico.
  3. Al completar todos los campos, el LLM genera un resumen final y se guarda en JSONL.

El KGE es el motor de recomendación (anti-alucinación).
El LLM es solo la interfaz conversacional, NO la fuente de conocimiento.

Uso:
  python src/incident_creator.py                    # DistMult + LLM
  python src/incident_creator.py --kge-model TransE
  python src/incident_creator.py --no-llm           # menú numerado sin LLM
  python src/incident_creator.py --top-k 5

Desde el pipeline:
  python src/run_pipeline.py --phase create_incident --kge-model DistMult
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# ---------------------------------------------------------------------------
# Orden y etiquetas de propiedades
# ---------------------------------------------------------------------------

INCIDENT_PROPS = [
    "int_hasCustomer",        # 1: cliente — ancla el contexto
    "hasTypeInc",             # 2: tipo de incidencia
    "incident_hasOrigin",     # 3: canal de origen
    "hasSupportGroup",        # 4: grupo de soporte
    "hasTechnician",          # 5: técnico asignado
    "hasSupportCategory",     # 6: categoría de soporte
    "hasSupportTeam",         # 7: equipo de soporte
    "hasStateIncident",       # 8: estado
    "hasExternalTechnician",  # 9: técnico externo (opcional)
]

_PROP_LABELS = {
    "int_hasCustomer":       "cliente",
    "hasTypeInc":            "tipo de incidencia",
    "incident_hasOrigin":    "origen",
    "hasSupportGroup":       "grupo de soporte",
    "hasTechnician":         "técnico asignado",
    "hasSupportCategory":    "categoría de soporte",
    "hasSupportTeam":        "equipo de soporte",
    "hasStateIncident":      "estado",
    "hasExternalTechnician": "técnico externo",
}


# ---------------------------------------------------------------------------
# Prompts LLM
# ---------------------------------------------------------------------------

_ASK_SYSTEM_PROMPT = (
    "Eres un asistente que formula preguntas para completar una ficha de incidencia. "
    "Tu tarea es hacer UNA pregunta corta y natural en español invitando al usuario "
    "a elegir entre las opciones que se le van a mostrar (por número o por identificador exacto). "
    "NO inventes opciones, NO sugieras valores que no estén en la lista, "
    "NO interpretes los identificadores (son códigos opacos, no categorías). "
    "Responde SOLO con la pregunta, sin explicaciones ni encabezados."
)

_EXTRACT_SYSTEM_PROMPT = (
    "Eres un asistente de extracción de datos de incidencias. "
    "El usuario respondió a una pregunta sobre un campo de incidencia. "
    "Basándote en su respuesta y las opciones válidas, devuelve ÚNICAMENTE "
    "el identificador exacto elegido (p.ej. employee__259, company__GB3782FUB, "
    "typeIncident__1, supportGroup_149762...). "
    "Si la respuesta no corresponde a ninguna opción o no es clara, devuelve: UNCLEAR"
)


# ---------------------------------------------------------------------------
# Extracción de texto libre (lookup contra el grafo)
# ---------------------------------------------------------------------------

def extract_from_free_text(text: str, incidents_map: dict) -> dict[str, str]:
    """
    Escanea el texto libre buscando valores conocidos del grafo.

    Construye un índice inverso {valor_entidad → propiedad} desde incidents_map
    y busca coincidencias directas en el texto (case-sensitive, los IDs son opacos).

    Detecta principalmente: company__XXXX, employee__N, typeIncident__N,
    incidentOrigin__N, statusIncident__N. Los hashes largos (supportGroup_...)
    raramente aparecen en texto libre escrito por un usuario.

    Devuelve {propiedad: primer_valor_encontrado}.
    """
    # Índice inverso: valor → propiedad (el primero que mapee gana)
    value_to_prop: dict[str, str] = {}
    for props in incidents_map.values():
        for prop, values in props.items():
            for v in (values if isinstance(values, list) else [values]):
                if v and v not in value_to_prop:
                    value_to_prop[v] = prop

    found: dict[str, str] = {}
    for value, prop in value_to_prop.items():
        if len(value) >= 4 and value in text and prop not in found:
            found[prop] = value
    return found


# ---------------------------------------------------------------------------
# CBR: búsqueda de incidencias históricas similares
# ---------------------------------------------------------------------------

def find_matching_incidents(known_props: dict, incidents_map: dict) -> list[str]:
    """
    Devuelve las incident_ids cuyas propiedades coinciden con known_props.
    Empieza exigiendo que TODAS las propiedades conocidas coincidan; si
    obtiene menos de 3 resultados, relaja el umbral en 1 hasta mínimo 1.
    """
    filled = {k: v for k, v in known_props.items() if v is not None}
    if not filled:
        return []
    matches = []
    for threshold in range(len(filled), 0, -1):
        matches = [
            inc_id for inc_id, props in incidents_map.items()
            if sum(1 for k, v in filled.items() if v in props.get(k, [])) >= threshold
        ]
        if len(matches) >= 3:
            return matches
    return matches


# ---------------------------------------------------------------------------
# Recomendación KGE vía proxies CBR
# ---------------------------------------------------------------------------

# Reciprocal Rank Fusion: constante de suavizado (estándar IR).
RRF_K = 60


def recommend_property(
    known_props: dict,
    target_prop: str,
    incidents_map: dict,
    model,
    factory,
    top_k: int = 5,
) -> tuple[list[tuple[str, int, float]], int]:
    """
    Genera recomendaciones para target_prop combinando CBR + KGE mediante
    Reciprocal Rank Fusion (RRF).

    1. Encuentra proxies CBR (incidencias históricas con propiedades similares)
    2. Para cada proxy, llama predict_tails(proxy, target_prop, top_k)
    3. Para cada candidato calcula:
         - rank_freq: posición al ordenar por frecuencia CBR (DESC)
         - rank_kge:  posición al ordenar por score KGE medio (DESC)
       y fusiona: RRF(o) = 1/(K + rank_freq) + 1/(K + rank_kge)
    4. Fallback si no hay proxies: predict_heads sobre la primera prop conocida

    Devuelve (lista de (entity_label, frecuencia, score_medio), n_proxies)
    ordenada por RRF DESC.
    """
    from phase3_link_prediction import predict_tails, predict_heads

    proxies = find_matching_incidents(known_props, incidents_map)

    if not proxies:
        first_prop = next((k for k, v in known_props.items() if v is not None), None)
        if first_prop:
            heads = predict_heads(model, factory, first_prop,
                                  known_props[first_prop], top_k=20)
            proxies = [h for h, _ in heads if h in incidents_map]

    if not proxies:
        return [], 0

    n_proxies = len(proxies)
    scores: dict[str, list[float]] = {}
    for proxy in proxies[:30]:
        for entity, score in predict_tails(model, factory, proxy, target_prop, top_k):
            scores.setdefault(entity, []).append(score)

    aggregated = [
        (ent, len(sc), sum(sc) / len(sc))
        for ent, sc in scores.items()
    ]

    # Ranking 1: por frecuencia CBR (más votos = mejor rank)
    by_freq = sorted(aggregated, key=lambda x: x[1], reverse=True)
    rank_freq = {ent: i + 1 for i, (ent, _, _) in enumerate(by_freq)}

    # Ranking 2: por score KGE medio (mayor score = mejor rank)
    by_kge = sorted(aggregated, key=lambda x: x[2], reverse=True)
    rank_kge = {ent: i + 1 for i, (ent, _, _) in enumerate(by_kge)}

    # Fusión RRF
    aggregated.sort(
        key=lambda x: 1.0 / (RRF_K + rank_freq[x[0]])
                    + 1.0 / (RRF_K + rank_kge[x[0]]),
        reverse=True,
    )
    return aggregated[:top_k], n_proxies


# ---------------------------------------------------------------------------
# Carga rápida de incidencias desde TSV (sin rdflib)
# ---------------------------------------------------------------------------

def _build_incidents_map_from_tsv() -> dict:
    """
    Construye incidents_map = {incident_id: {predicate: [values]}} leyendo
    train.tsv + valid.tsv directamente, sin pasar por rdflib.

    Solo incluye triples cuya cabeza sea una incidencia (empieza por 'incident_')
    y cuya relación sea una de las propiedades relevantes (INCIDENT_PROPS).
    """
    prop_set = set(INCIDENT_PROPS)
    incidents: dict = {}
    for tsv_path in (cfg.TRAIN_TSV, cfg.VALID_TSV):
        if not tsv_path.exists():
            continue
        with open(tsv_path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                head, rel, tail = parts
                if not head.startswith("incident_"):
                    continue
                if rel not in prop_set:
                    continue
                incidents.setdefault(head, {}).setdefault(rel, []).append(tail)
    return incidents


# ---------------------------------------------------------------------------
# Sesión interactiva de creación de incidencia
# ---------------------------------------------------------------------------

class IncidentCreatorSession:
    """
    Creador guiado de incidencias con CBR + KGE + conversación LLM.

    Si el LLM está disponible:
      - El LLM genera preguntas naturales con las recomendaciones KGE como contexto.
      - El usuario responde libremente y el LLM extrae el identificador elegido.
    Si el LLM no está disponible:
      - Menú numerado clásico (fallback).
    """

    def __init__(
        self,
        kge_model_name: str = 'TransE',
        use_llm: bool = True,
        llm_model_name: str = cfg.DEFAULT_MODEL,
        top_k: int = 5,
    ):
        self.kge_model_name  = kge_model_name
        self.llm_model_name  = llm_model_name
        self.top_k           = top_k
        self._openai_client  = None
        self.llm             = None  # KGEAugmentedLLM para el resumen final

        print(f"\n{'='*60}")
        print("  Cargando recursos ...")
        print(f"{'='*60}")

        # Mapa de incidencias históricas (desde TSV, sin rdflib)
        print(f"  [1/3] Cargando incidencias desde TSV ...")
        self.incidents_map = _build_incidents_map_from_tsv()
        print(f"        {len(self.incidents_map):,} incidencias históricas cargadas.")

        # Modelo KGE
        from phase3_link_prediction import load_model_by_name
        print(f"  [2/3] Cargando modelo KGE: {kge_model_name} ...")
        self.model, self.factory = load_model_by_name(kge_model_name)

        # LLM (opcional)
        if use_llm:
            print(f"  [3/3] Conectando con LLM: {llm_model_name} ...")
            try:
                from openai import OpenAI
                from phase4_llm_inference import KGEAugmentedLLM
                self._openai_client = OpenAI(
                    base_url=cfg.VLLM_BASE_URL, api_key="EMPTY"
                )
                self.llm = KGEAugmentedLLM(
                    model_name=llm_model_name,
                    base_url=cfg.VLLM_BASE_URL,
                )
                print("        LLM listo (modo conversacional).")
            except Exception as e:
                print(f"  [!] LLM no disponible: {e}. Usando menú numerado.")
                self._openai_client = None
                self.llm = None
        else:
            print("  [3/3] Modo sin LLM (menú numerado).")

        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Métodos LLM
    # ------------------------------------------------------------------

    def _llm_ask(self, prop: str, recs: list, incident: dict) -> str:
        """Genera una pregunta natural invitando al usuario a elegir entre las
        opciones que el KGE ha recomendado. NO menciona valores fuera de recs."""
        from phase4_llm_inference import verbalize_props
        label = _PROP_LABELS.get(prop, prop)

        filled = {k: v for k, v in incident.items() if v is not None}
        known_block = ""
        if filled:
            known_block = "Propiedades ya conocidas:\n" + "\n".join(
                f"- {s}" for s in verbalize_props("la nueva incidencia", filled)
            ) + "\n\n"

        ids_line = ", ".join(ent for ent, _, _ in recs)
        user_prompt = (
            f"{known_block}"
            f"Campo a completar: '{label}'.\n"
            f"Opciones recomendadas por el sistema (únicos valores válidos): {ids_line}.\n\n"
            f"Formula una pregunta corta para que el usuario elija una de estas opciones. "
            f"Indícale que puede responder con el número de la opción o con el identificador exacto."
        )
        messages = [
            {"role": "system", "content": _ASK_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
        resp = self._openai_client.chat.completions.create(
            model=self.llm_model_name,
            messages=messages,
            max_tokens=80,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    def _llm_extract(self, question_asked: str, recs: list, user_response: str) -> str | None:
        """
        Extrae el identificador elegido de la respuesta libre del usuario.
        Devuelve el identificador o None si no está claro (LLM responde UNCLEAR).
        """
        options_text = "\n".join(
            f"- {ent}  (frecuencia: {freq}, score: {score:.3f})"
            for ent, freq, score in recs
        )
        messages = [
            {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
            {"role": "user",   "content": (
                f"Pregunta realizada: {question_asked}\n\n"
                f"Opciones válidas:\n{options_text}\n\n"
                f"Respuesta del usuario: {user_response}\n\n"
                "Identificador elegido:"
            )},
        ]
        resp = self._openai_client.chat.completions.create(
            model=self.llm_model_name,
            messages=messages,
            max_tokens=40,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        if not raw or raw.upper() == "UNCLEAR":
            return None
        # Validar que lo devuelto por el LLM es uno de los IDs del KGE.
        # Si el LLM se inventa algo, tratarlo como UNCLEAR.
        known_ids = {ent for ent, _, _ in recs}
        return raw if raw in known_ids else None

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Ejecuta la sesión. Devuelve el dict de la incidencia completada."""
        incident: dict[str, str | None] = {p: None for p in INCIDENT_PROPS}

        print("=== Creación de nueva incidencia ===\n")

        # ------------------------------------------------------------------
        # Fase 0: texto libre inicial
        # ------------------------------------------------------------------
        print("Describe el problema (o pulsa Enter para empezar desde cero):")
        try:
            free_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            free_text = ""

        if free_text:
            pre_filled = extract_from_free_text(free_text, self.incidents_map)
            if pre_filled:
                print()
                for prop, val in pre_filled.items():
                    incident[prop] = val
                    label = _PROP_LABELS.get(prop, prop)
                    print(f"  [Detectado] {label} = {val}")
            else:
                print("  (No se detectaron entidades conocidas en el texto)")
            print()

        if not self._openai_client:
            print("Comandos: s/si/y = aceptar #1 | 2..N = elegir nº | "
                  "texto = valor propio | skip = saltar | exit = salir\n")

        # ------------------------------------------------------------------
        # Fase 1: completar propiedad a propiedad
        # ------------------------------------------------------------------
        total    = len(INCIDENT_PROPS)
        prop_idx = 0
        recs: list = []
        n_proxies: int = 0
        last_question: str = ""

        while prop_idx < total:
            prop  = INCIDENT_PROPS[prop_idx]
            label = _PROP_LABELS.get(prop, prop)

            # Saltar si ya está relleno (por texto libre o paso anterior)
            if incident[prop] is not None:
                prop_idx += 1
                recs = []
                continue

            # Calcular recomendaciones una sola vez por campo
            if not recs:
                recs, n_proxies = recommend_property(
                    known_props=incident,
                    target_prop=prop,
                    incidents_map=self.incidents_map,
                    model=self.model,
                    factory=self.factory,
                    top_k=self.top_k,
                )

            # ---- Rama LLM conversacional ----
            if self._openai_client and recs:
                try:
                    last_question = self._llm_ask(prop, recs, incident)
                except Exception as e:
                    print(f"  [!] LLM falló generando pregunta: {e}")
                    last_question = f"¿Qué valor eliges para {label}?"

                print(f"\n[Asistente] {last_question}\n")
                print("Opciones recomendadas (KGE):")
                for i, (ent, freq, score) in enumerate(recs, 1):
                    marker = "►" if i == 1 else " "
                    print(f"  {marker}{i}. {ent}  (freq: {freq}, score: {score:.3f})")
                print("(responde con número, ID exacto, s/si = #1, skip, exit)\n")

                try:
                    user_input = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n[Interrupción — guardando lo completado]")
                    break

                # 1) Comandos deterministas (número, s/si, skip, exit, ID exacto)
                chosen = self._pick_from_menu(user_input, recs, prop, label, incident)
                if chosen == "__exit__":
                    break
                if chosen == "__skip__":
                    prop_idx += 1; recs = []; continue

                # Si _pick_from_menu devolvió el input textual tal cual, y NO es un
                # ID conocido, delegamos al LLM para interpretar la respuesta.
                known_ids = {ent for ent, _, _ in recs}
                if chosen is not None and chosen in known_ids:
                    incident[prop] = chosen
                    print(f"  ✓ {label} = {chosen}")
                    prop_idx += 1; recs = []
                    continue

                # 2) Intentar extracción LLM contra las opciones KGE
                extracted = None
                if user_input:
                    try:
                        extracted = self._llm_extract(last_question, recs, user_input)
                    except Exception as e:
                        print(f"  [!] LLM falló extrayendo respuesta: {e}")

                if extracted:
                    incident[prop] = extracted
                    print(f"  ✓ {label} = {extracted}")
                    prop_idx += 1; recs = []
                else:
                    # 3) UNCLEAR → re-preguntar (mantener recs, no avanzar prop_idx)
                    print("  [No he entendido la respuesta. Vuelve a intentarlo "
                          "respondiendo con el número o el identificador exacto.]")

            # ---- Rama menú numerado (sin LLM o sin recomendaciones) ----
            else:
                filled_str = ", ".join(
                    f"{_PROP_LABELS.get(k, k)}={v}"
                    for k, v in incident.items() if v is not None
                ) or "ninguna"
                print(f"\n[{prop_idx + 1}/{total}] Completando: {label}")
                print(f"Conocidas: {filled_str}")
                if n_proxies:
                    print(f"[CBR] {n_proxies} incidencias similares.")
                if recs:
                    for i, (ent, freq, score) in enumerate(recs):
                        marker = "►" if i == 0 else " "
                        print(f"  {marker}{i+1}. {ent}  (score: {score:.4f}, freq: {freq})")
                else:
                    print("[KGE] Sin recomendaciones. Escribe un valor manualmente.")

                try:
                    user_input = input("\nRespuesta > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                chosen = self._pick_from_menu(user_input, recs, prop, label, incident)
                if chosen == "__exit__":
                    break
                elif chosen == "__skip__":
                    prop_idx += 1; recs = []
                elif chosen is not None:
                    known_ids = {ent for ent, _, _ in recs}
                    if chosen in known_ids or not recs:
                        incident[prop] = chosen
                        print(f"  ✓ {label} = {chosen}")
                        prop_idx += 1; recs = []
                    else:
                        print(f"  [!] '{chosen}' no está entre las opciones válidas. "
                              "Elige un número o un identificador de la lista.")

        self._finish(incident)
        return incident

    # ------------------------------------------------------------------

    def _pick_from_menu(
        self,
        user_input: str,
        recs: list,
        prop: str,
        label: str,
        incident: dict,
    ) -> str | None:
        """
        Interpreta la entrada del usuario en el menú numerado.
        Devuelve: valor elegido | "__exit__" | "__skip__" | None (no reconocido)
        """
        cmd = user_input.lower().strip()

        if cmd in ("salir", "exit", "quit"):
            return "__exit__"
        if cmd in ("saltar", "skip"):
            print(f"  ⟳ {label} dejado sin rellenar.")
            return "__skip__"
        if cmd in ("s", "si", "y", "yes"):
            if recs:
                return recs[0][0]
            print("  [!] No hay recomendación. Escribe un valor.")
            return None
        if cmd.isdigit():
            choice = int(cmd) - 1
            if recs and 0 <= choice < len(recs):
                return recs[choice][0]
            print(f"  [!] Número fuera de rango (1–{len(recs)}).")
            return None
        if user_input:
            return user_input
        return None

    # ------------------------------------------------------------------

    def _finish(self, incident: dict) -> None:
        """Verbaliza, genera resumen LLM y guarda en JSONL."""
        from phase4_llm_inference import verbalize_props

        filled = {k: v for k, v in incident.items() if v is not None}
        n_filled = len(filled)

        print(f"\n{'='*60}")
        print(f"  Incidencia completada ({n_filled}/{len(INCIDENT_PROPS)} campos)")
        print(f"{'='*60}")

        if not filled:
            print("  [!] No se completó ningún campo. No se guardará.")
            return

        sentences = verbalize_props("nueva_incidencia", filled)
        print("\n  Propiedades:")
        for s in sentences:
            print(f"    · {s}")

        llm_summary = ""
        if self.llm and sentences:
            try:
                llm_summary = self.llm.answer(
                    context_sentences=sentences,
                    question="Resume en una frase en español la incidencia creada.",
                    do_extract=False,
                )
                print(f"\n  [LLM] Resumen: \"{llm_summary}\"")
            except Exception as e:
                print(f"\n  [!] LLM no pudo generar resumen: {e}")

        out_dir  = cfg.OUT_DIR / "sessions"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "created_incidents.jsonl"
        record   = {
            "timestamp":   datetime.now().isoformat(timespec="seconds"),
            "kge_model":   self.kge_model_name,
            "incident":    incident,
            "llm_summary": llm_summary,
        }
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"\n  ✓ Guardado en {out_path}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(
    kge_model_name: str = 'TransE',
    use_llm: bool = True,
    llm_model_name: str = cfg.DEFAULT_MODEL,
    top_k: int = 5,
) -> dict:
    session = IncidentCreatorSession(
        kge_model_name=kge_model_name,
        use_llm=use_llm,
        llm_model_name=llm_model_name,
        top_k=top_k,
    )
    return session.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creador guiado de incidencias con CBR + KGE + LLM"
    )
    parser.add_argument("--kge-model", default="TransE",
                        help=f"Modelo KGE (default: DistMult). Opciones: {cfg.KGE_MODELS}")
    parser.add_argument("--no-llm", action="store_true",
                        help="Desactivar LLM (menú numerado clásico)")
    parser.add_argument("--model", default=cfg.DEFAULT_MODEL,
                        help=f"Modelo LLM (default: {cfg.DEFAULT_MODEL})")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Recomendaciones KGE por propiedad (default: 5)")
    args = parser.parse_args()

    run(
        kge_model_name=args.kge_model,
        use_llm=not args.no_llm,
        llm_model_name=args.model,
        top_k=args.top_k,
    )
