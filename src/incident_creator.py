"""
Creador guiado de incidencias con CBR + KGE.

Para cada propiedad de la nueva incidencia, el sistema:
  1. Busca incidencias históricas que coincidan con las propiedades ya conocidas (CBR)
  2. Usa esas incidencias como proxies para predecir el siguiente valor vía KGE (predict_tails)
  3. Presenta al usuario las top-N recomendaciones y le pide confirmación
  4. Al completar todas las propiedades, verbaliza el resultado con el LLM y guarda en JSONL

Uso:
  python src/incident_creator.py                         # DistMult, con LLM
  python src/incident_creator.py --kge-model TransE
  python src/incident_creator.py --no-llm                # sin verbalización LLM
  python src/incident_creator.py --top-k 5               # más recomendaciones por propiedad

O desde el pipeline:
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
# Orden de propiedades y etiquetas en español
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
    for threshold in range(len(filled), 0, -1):
        matches = []
        for inc_id, props in incidents_map.items():
            n_match = sum(
                1 for k, v in filled.items()
                if v in props.get(k, [])
            )
            if n_match >= threshold:
                matches.append(inc_id)
        if len(matches) >= 3:
            return matches
    return matches


# ---------------------------------------------------------------------------
# Recomendación KGE vía proxies CBR
# ---------------------------------------------------------------------------

def recommend_property(
    known_props: dict,
    target_prop: str,
    incidents_map: dict,
    model,
    factory,
    top_k: int = 5,
) -> list[tuple[str, int, float]]:
    """
    Genera recomendaciones para target_prop combinando CBR + KGE.

    Estrategia:
      1. Encuentra proxies CBR (incidencias históricas con propiedades similares)
      2. Para cada proxy, llama predict_tails(proxy, target_prop, top_k)
      3. Agrega resultados: ordena por (frecuencia DESC, score_medio DESC)
      4. Si no hay proxies: fallback con predict_heads sobre la primera prop conocida

    Devuelve lista de (entity_label, frecuencia, score_medio) ordenada mejor primero.
    """
    from phase3_link_prediction import predict_tails, predict_heads

    proxies = find_matching_incidents(known_props, incidents_map)

    if not proxies:
        # Fallback: buscar incidencias vía predicción inversa sobre la primera prop conocida
        first_prop = next((k for k, v in known_props.items() if v is not None), None)
        if first_prop:
            heads = predict_heads(model, factory, first_prop,
                                  known_props[first_prop], top_k=20)
            proxies = [h for h, _ in heads if h in incidents_map]

    if not proxies:
        return []

    # Recoger predicciones de hasta 30 proxies (límite de velocidad)
    scores: dict[str, list[float]] = {}
    for proxy in proxies[:30]:
        for entity, score in predict_tails(model, factory, proxy, target_prop, top_k):
            scores.setdefault(entity, []).append(score)

    # Agregar: frecuencia y score medio
    aggregated = [
        (ent, len(sc), sum(sc) / len(sc))
        for ent, sc in scores.items()
    ]
    aggregated.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return aggregated[:top_k]


# ---------------------------------------------------------------------------
# Sesión interactiva de creación de incidencia
# ---------------------------------------------------------------------------

class IncidentCreatorSession:
    """
    Wizard guiado que rellena una nueva incidencia propiedad a propiedad
    usando CBR + KGE y confirmación del usuario en cada paso.
    """

    def __init__(
        self,
        kge_model_name: str = 'DistMult',
        use_llm: bool = True,
        llm_model_name: str = cfg.DEFAULT_MODEL,
        top_k: int = 5,
    ):
        self.kge_model_name = kge_model_name
        self.use_llm = use_llm
        self.llm_model_name = llm_model_name
        self.top_k = top_k

        print(f"\n{'='*60}")
        print("  Cargando recursos ...")
        print(f"{'='*60}")

        # Mapa de incidencias históricas
        from rdflib import Graph
        from generate_corpus import build_incident_map
        print(f"  [1/3] Cargando grafo desde {cfg.TTL_FILE} ...")
        g = Graph()
        g.parse(str(cfg.TTL_FILE), format="turtle")
        self.incidents_map = build_incident_map(g)
        print(f"        {len(self.incidents_map):,} incidencias históricas cargadas.")

        # Modelo KGE
        from phase3_link_prediction import load_model_by_name
        print(f"  [2/3] Cargando modelo KGE: {kge_model_name} ...")
        self.model, self.factory = load_model_by_name(kge_model_name)

        # LLM (opcional)
        self.llm = None
        if use_llm:
            from phase4_llm_inference import KGEAugmentedLLM
            print(f"  [3/3] Conectando con LLM: {llm_model_name} ...")
            try:
                self.llm = KGEAugmentedLLM(
                    model_name=llm_model_name,
                    base_url=cfg.VLLM_BASE_URL,
                )
            except Exception as e:
                print(f"  [!] LLM no disponible: {e}. Continuando sin LLM.")
                self.llm = None
        else:
            print("  [3/3] Modo sin LLM.")

        print(f"{'='*60}\n")

    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Ejecuta el wizard. Devuelve el dict de la incidencia completada.
        """
        incident: dict[str, str | None] = {p: None for p in INCIDENT_PROPS}

        print("=== Creación de nueva incidencia ===\n")
        print("Comandos: s/si/y = aceptar #1 | 2..N = elegir nº |")
        print("          texto  = valor propio       | skip = saltar campo")
        print("          exit   = salir (guarda lo completado hasta ahora)\n")

        total = len(INCIDENT_PROPS)
        current_idx = 0   # posición en la lista de recomendaciones (para "n/no")
        recs: list = []   # caché de recomendaciones para el campo actual

        prop_idx = 0
        while prop_idx < total:
            prop = INCIDENT_PROPS[prop_idx]
            label = _PROP_LABELS.get(prop, prop)

            # Estado actual
            filled_str = ", ".join(
                f"{_PROP_LABELS.get(k, k)}={v}"
                for k, v in incident.items()
                if v is not None
            ) or "ninguna"

            print(f"\n[{prop_idx + 1}/{total}] Completando: {label}  ({prop})")
            print(f"Propiedades conocidas: {filled_str}")

            # Calcular recomendaciones (solo si no las tenemos ya)
            if not recs:
                recs = recommend_property(
                    known_props=incident,
                    target_prop=prop,
                    incidents_map=self.incidents_map,
                    model=self.model,
                    factory=self.factory,
                    top_k=self.top_k,
                )
                current_idx = 0
                n_proxies = len(find_matching_incidents(incident, self.incidents_map))
                if n_proxies:
                    print(f"[CBR] {n_proxies} incidencias históricas similares encontradas.")
                else:
                    print("[CBR] Sin coincidencias exactas — usando predicción KGE directa.")

            if recs:
                print(f"[KGE] Recomendaciones para '{label}':")
                for i, (ent, freq, score) in enumerate(recs):
                    marker = "►" if i == current_idx else " "
                    print(f"  {marker}{i + 1}. {ent}  (score: {score:.4f}, freq: {freq})")
            else:
                print("[KGE] No se encontraron recomendaciones. Introduce un valor manualmente.")

            # Leer respuesta del usuario
            try:
                user_input = input("\nRespuesta > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Interrupción — guardando lo completado]")
                break

            cmd = user_input.lower()

            # Salir
            if cmd in ("salir", "exit", "quit"):
                print("[Saliendo — guardando lo completado]")
                break

            # Saltar campo
            if cmd in ("saltar", "skip"):
                print(f"  ⟳ {label} dejado sin rellenar.")
                prop_idx += 1
                recs = []
                continue

            # Aceptar la recomendación marcada (►)
            if cmd in ("s", "si", "y", "yes", ""):
                if recs:
                    chosen = recs[current_idx][0]
                    incident[prop] = chosen
                    print(f"  ✓ {prop} = {chosen}")
                    prop_idx += 1
                    recs = []
                else:
                    print("  [!] No hay recomendación disponible. Escribe un valor.")
                continue

            # Siguiente recomendación
            if cmd in ("n", "no"):
                if recs:
                    current_idx = (current_idx + 1) % len(recs)
                    print(f"  → Siguiente recomendación: {recs[current_idx][0]}")
                else:
                    print("  [!] No hay más recomendaciones.")
                continue

            # Selección por número
            if cmd.isdigit():
                choice = int(cmd) - 1
                if recs and 0 <= choice < len(recs):
                    chosen = recs[choice][0]
                    incident[prop] = chosen
                    print(f"  ✓ {prop} = {chosen}")
                    prop_idx += 1
                    recs = []
                else:
                    print(f"  [!] Número fuera de rango (1–{len(recs)}).")
                continue

            # Valor manual libre
            if user_input:
                incident[prop] = user_input
                print(f"  ✓ {prop} = {user_input}")
                prop_idx += 1
                recs = []
                continue

        # ------------------------------------------------------------------
        # Verbalización y guardado
        # ------------------------------------------------------------------
        self._finish(incident)
        return incident

    # ------------------------------------------------------------------

    def _finish(self, incident: dict) -> None:
        """Verbaliza el incidente completado, muestra el resumen y guarda en JSONL."""
        from phase4_llm_inference import verbalize_props

        filled = {k: v for k, v in incident.items() if v is not None}
        n_filled = len(filled)

        print(f"\n{'='*60}")
        print(f"  Incidencia completada ({n_filled}/{len(INCIDENT_PROPS)} campos rellenos)")
        print(f"{'='*60}")

        if not filled:
            print("  [!] No se completó ningún campo. No se guardará.")
            return

        # Verbalización de propiedades
        sentences = verbalize_props("nueva_incidencia", filled)
        print("\n  Propiedades:")
        for s in sentences:
            print(f"    · {s}")

        # Resumen con LLM
        llm_summary = ""
        if self.llm and sentences:
            try:
                question = ("Resume en una frase en español la incidencia creada "
                            "con los datos anteriores.")
                llm_summary = self.llm.answer(
                    context_sentences=sentences,
                    question=question,
                    do_extract=False,
                )
                print(f"\n  [LLM] Resumen: \"{llm_summary}\"")
            except Exception as e:
                print(f"\n  [!] LLM no pudo generar resumen: {e}")

        # Guardar en JSONL
        out_dir = cfg.OUT_DIR / "sessions"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "created_incidents.jsonl"

        record = {
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
    kge_model_name: str = 'DistMult',
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
        description="Creador guiado de incidencias con CBR + KGE"
    )
    parser.add_argument("--kge-model", default="DistMult",
                        help=f"Modelo KGE a usar (default: DistMult). Opciones: {cfg.KGE_MODELS}")
    parser.add_argument("--no-llm", action="store_true",
                        help="Desactivar verbalización LLM al finalizar")
    parser.add_argument("--model", default=cfg.DEFAULT_MODEL,
                        help=f"Modelo LLM (default: {cfg.DEFAULT_MODEL})")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Número de recomendaciones a mostrar por propiedad (default: 5)")
    args = parser.parse_args()

    run(
        kge_model_name=args.kge_model,
        use_llm=not args.no_llm,
        llm_model_name=args.model,
        top_k=args.top_k,
    )
