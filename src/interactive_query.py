"""
Sesión interactiva: consulta en lenguaje natural → GLiNER2 → KGE → LLM.

Flujo por consulta:
  1. El usuario escribe una pregunta en español.
  2. GLiNERExtractor detecta (sujeto, predicado) en la consulta.
  3. El modelo KGE hace link prediction y devuelve los objetos más probables.
  4. Se verbaliza la predicción top-1 usando PRED_TEMPLATES_ES.
  5. El LLM reformula la respuesta en lenguaje natural.
  6. Se comprueba que la respuesta del LLM preserva el identificador predicho.
  7. El usuario confirma o rechaza → si confirma, se registra la tripleta.

Uso:
  python src/interactive_query.py
  python src/interactive_query.py --kge-model TransE
  python src/interactive_query.py --no-llm        (solo link prediction, sin LLM)

Comandos durante la sesión:
  salir / exit / quit  → terminar
  modelos              → cambiar modelo KGE en tiempo de ejecución
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from generate_corpus import PRED_TEMPLATES_ES


# ---------------------------------------------------------------------------
# Verbalizador de predicciones KGE
# ---------------------------------------------------------------------------

def verbalize_prediction(subject: str, predicate: str, obj: str) -> str:
    """
    Construye una frase en español describiendo la predicción KGE.
    Usa PRED_TEMPLATES_ES si existe plantilla para el predicado,
    de lo contrario usa un formato genérico.
    """
    tmpl = PRED_TEMPLATES_ES.get(predicate)
    if tmpl:
        return tmpl.format(s=subject, p=predicate, o=obj)
    return f"La incidencia {subject} tiene {predicate}: {obj}."


def verify_verbalization_integrity(llm_answer: str, expected_entity: str) -> bool:
    """
    Comprueba que el identificador esperado aparece en la respuesta del LLM.
    Comparación case-insensitive.
    """
    return expected_entity.lower() in llm_answer.lower()


# ---------------------------------------------------------------------------
# Bucle interactivo principal
# ---------------------------------------------------------------------------

def interactive_query_loop(
    kge_model_name: str = 'DistMult',
    llm_base_url: str = cfg.VLLM_BASE_URL,
    llm_model_name: str = cfg.DEFAULT_MODEL,
    use_llm: bool = True,
    log_path: Optional[Path] = None,
) -> None:
    """
    REPL interactivo que integra GLiNER2 + KGE + LLM.

    Parámetros
    ----------
    kge_model_name : str
        Modelo KGE a usar para link prediction (TransE, DistMult, ComplEx).
    llm_base_url : str
        URL del servidor vLLM (OpenAI-compatible).
    llm_model_name : str
        Nombre del modelo HuggingFace en el servidor vLLM.
    use_llm : bool
        Si False, omite la verbalización por LLM (útil sin servidor vLLM).
    log_path : Path, optional
        Ruta donde guardar las tripletas confirmadas (JSON Lines).
    """
    from gliner_extractor import GLiNERExtractor
    from phase3_link_prediction import load_model_by_name, predict_tails

    print("=" * 60)
    print("  Sesión interactiva KGE + GLiNER2 + LLM")
    print("=" * 60)
    print(f"  Modelo KGE : {kge_model_name}")
    print(f"  LLM        : {'activado (' + llm_model_name + ')' if use_llm else 'desactivado'}")
    print(f"  Escribe 'salir' para terminar, 'modelos' para ver los disponibles.\n")

    # Cargar componentes
    extractor = GLiNERExtractor()
    print(f"[KGE] Cargando modelo {kge_model_name} ...")
    kge_model, factory = load_model_by_name(kge_model_name)

    llm = None
    if use_llm:
        try:
            from phase4_llm_inference import KGEAugmentedLLM, extract_answer as _extract_answer
            llm = KGEAugmentedLLM(model_name=llm_model_name, base_url=llm_base_url)
            print(f"[LLM] Conectado a {llm_base_url}\n")
        except Exception as e:
            print(f"[LLM] No se pudo conectar: {e}  (continuando sin LLM)\n")
            use_llm = False

    if log_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = cfg.OUT_DIR / "sessions" / f"interactive_query_{ts}.jsonl"

    confirmed_triples: list[dict] = []

    print("─" * 60)

    while True:
        try:
            query = input("\nConsulta: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue

        low = query.lower()
        if low in ("salir", "exit", "quit"):
            break
        if low == "modelos":
            print(f"  Modelos disponibles: {cfg.KGE_MODELS}")
            print(f"  Modelo actual: {kge_model_name}")
            continue

        # ── Extracción GLiNER2 ──────────────────────────────────────────
        extraction = extractor.extract(query)
        head     = extraction["head"]
        relation = extraction["relation"]

        print(f"\n[Extractor]")
        print(f"  Entidad   : {head or '(no detectada)'}  [{extraction['head_found_by']}]")
        print(f"  Relación  : {relation or '(no detectada)'}  [{extraction['relation_found_by'] or '—'}]")

        if not head or not relation:
            print("  No se pudo extraer la entidad o relación. "
                  "Prueba a incluir el ID de la incidencia y una palabra clave "
                  "(tipo, técnico, grupo, estado, etc.).")
            continue

        # ── Link prediction ─────────────────────────────────────────────
        predictions = predict_tails(kge_model, factory, head, relation, top_k=cfg.TOP_K_PREDICT)

        if not predictions:
            print(f"  [KGE] No se encontraron predicciones para ({head}, {relation}, ?).")
            continue

        print(f"\n[KGE] Top-{min(3, len(predictions))} predicciones:")
        for i, (entity, score) in enumerate(predictions[:3], 1):
            print(f"  {i}. {entity}  (score: {score:.4f})")

        top_entity = predictions[0][0]

        # ── Verbalización ───────────────────────────────────────────────
        verbalized = verbalize_prediction(head, relation, top_entity)
        print(f"\n[Predicción] {verbalized}")

        # ── LLM (opcional) ──────────────────────────────────────────────
        llm_answer = None
        integrity_ok = None

        if use_llm and llm is not None:
            context = [verbalized]
            llm_question = (
                f"¿Cuál es el {relation} de la incidencia {head}? "
                f"Responde solo con el identificador exacto."
            )
            try:
                from phase4_llm_inference import extract_answer as _extract_answer
                llm_raw    = llm.answer(context, llm_question, do_extract=False)
                llm_answer = _extract_answer(llm_raw)
                integrity_ok = verify_verbalization_integrity(llm_answer, top_entity)
                print(f"\n[LLM] Respuesta: {llm_answer}")
                integrity_icon = "✓" if integrity_ok else "✗"
                print(f"      Integridad: {integrity_icon}  "
                      f"(esperado: {top_entity})")
            except Exception as e:
                print(f"\n[LLM] Error: {e}")

        # ── Confirmación del usuario ────────────────────────────────────
        try:
            confirm = input(
                f"\n¿Es correcta la predicción '{top_entity}'? "
                f"[s/n/corrección]: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            break

        if confirm.lower() in ("s", "si", "sí", "y", "yes"):
            triple = {
                "subject":    head,
                "predicate":  relation,
                "object":     top_entity,
                "confirmed":  True,
                "source":     "interactive_query",
                "timestamp":  datetime.now().isoformat(),
            }
            confirmed_triples.append(triple)
            print(f"  ✓ Tripleta registrada: ({head}, {relation}, {top_entity})")
        elif confirm.lower() in ("n", "no"):
            print("  ✗ Predicción rechazada.")
        elif confirm:
            # El usuario proporcionó la respuesta correcta
            corrected_obj = confirm.strip()
            triple = {
                "subject":    head,
                "predicate":  relation,
                "object":     corrected_obj,
                "confirmed":  True,
                "corrected":  True,
                "predicted":  top_entity,
                "source":     "interactive_query",
                "timestamp":  datetime.now().isoformat(),
            }
            confirmed_triples.append(triple)
            print(f"  ✓ Corrección registrada: ({head}, {relation}, {corrected_obj})")

    # ── Guardar tripletas confirmadas ───────────────────────────────────────
    if confirmed_triples:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            for triple in confirmed_triples:
                f.write(json.dumps(triple, ensure_ascii=False) + "\n")
        print(f"\n✓ {len(confirmed_triples)} tripleta(s) confirmada(s) guardadas en:")
        print(f"  {log_path}")
    else:
        print("\nSesión finalizada. No se confirmaron tripletas.")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sesión interactiva GLiNER2 + KGE + LLM"
    )
    parser.add_argument("--kge-model", default="DistMult",
                        choices=cfg.KGE_MODELS,
                        help="Modelo KGE a usar (default: DistMult)")
    parser.add_argument("--base-url", default=cfg.VLLM_BASE_URL,
                        help=f"URL del servidor vLLM (default: {cfg.VLLM_BASE_URL})")
    parser.add_argument("--llm-model", default=cfg.DEFAULT_MODEL,
                        help=f"Modelo HuggingFace en vLLM (default: {cfg.DEFAULT_MODEL})")
    parser.add_argument("--no-llm", action="store_true",
                        help="Desactivar verbalización LLM (solo KGE)")
    parser.add_argument("--log", default=None,
                        help="Ruta para guardar tripletas confirmadas (.jsonl)")
    args = parser.parse_args()

    interactive_query_loop(
        kge_model_name=args.kge_model,
        llm_base_url=args.base_url,
        llm_model_name=args.llm_model,
        use_llm=not args.no_llm,
        log_path=Path(args.log) if args.log else None,
    )
