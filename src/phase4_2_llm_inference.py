"""
Fase 4.2 — Inferencia con LLM aumentado con contexto KGE enriquecido.

Extiende phase4_llm_inference.py integrando:
  - Casos similares por cosine similarity en el espacio de embeddings KGE (fase 5)
  - Predicciones implícitas de fase 3 (implicit_relations.json)
  - Tripletas directas del grafo (como en fase 4)

El LLM se sirve externamente con vLLM (API OpenAI-compatible).
Arrancar antes de ejecutar este módulo:

  vllm serve meta-llama/Meta-Llama-3-8B-Instruct \\
      --port 8000 --dtype float16 --max-model-len 4096 \\
      --tool-call-parser llama3_json

Prerequisitos:
  - Fase 2 ejecutada: out/embeddings/entity_embeddings.pt
  - Fase 3 ejecutada: out/predictions/implicit_relations.json

Uso:
  python src/phase4_2_llm_inference.py
  python src/phase4_2_llm_inference.py --interactive --incident incident_XXXXX
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
import phase5_config_subgraph as p5
from generate_corpus import PRED_TEMPLATES_ES

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Eres un asistente experto en gestión de incidencias técnicas. "
    "Responde ÚNICAMENTE con el identificador exacto de la entidad (por ejemplo: "
    "employee__986, supportGroup_149763..., incidentOrigin__2). "
    "No escribas frases, ni explicaciones, ni el contexto. Solo el identificador."
)

_USER_TEMPLATE = (
    "Dado el siguiente contexto del grafo de conocimiento, responde a la pregunta.\n"
    "IMPORTANTE: Responde ÚNICAMENTE con el identificador exacto de la entidad "
    "(sin frases, sin explicaciones, sin puntuación adicional).\n\n"
    "Contexto:\n{context_block}\n\n"
    "Pregunta: {question}\n"
    "Identificador:"
)


def _build_messages(context_sentences: list[str], question: str) -> list[dict]:
    """Construye la lista de mensajes para la API de chat."""
    context_block = "\n".join(f"- {s}" for s in context_sentences)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": _USER_TEMPLATE.format(
            context_block=context_block,
            question=question,
        )},
    ]


def extract_answer(raw: str) -> str:
    """
    Extrae el identificador de entidad de la salida cruda del LLM.
    Estrategia:
      1. Busca texto tras "Identificador:" o "Respuesta:" y toma la 1ª línea.
      2. Si no, toma la 1ª línea no vacía que no empiece por '-' ni '['.
      3. Fallback: devuelve el texto tal cual (sin el contexto repetido).
    """
    for marker in ("Identificador:", "Respuesta:", "[/INST]"):
        if marker in raw:
            raw = raw.split(marker)[-1]

    for line in raw.split("\n"):
        line = line.strip(" .-·\t")
        if line and not line.startswith("[") and not line.startswith("Contexto"):
            return line

    return raw.strip()


# ---------------------------------------------------------------------------
# Verbalización rápida de un subgrafo (fallback sin KGE)
# ---------------------------------------------------------------------------

def verbalize_props(incident_id: str, props: dict) -> list[str]:
    """
    Convierte el dict de propiedades de una incidencia en frases españolas
    usando PRED_TEMPLATES_ES importado de generate_corpus.
    """
    sentences = []
    for pred, values in props.items():
        template = PRED_TEMPLATES_ES.get(pred)
        if not template:
            continue
        for val in (values if isinstance(values, list) else [values]):
            sentences.append(template.format(s=incident_id, o=val))
    return sentences


# ---------------------------------------------------------------------------
# Construcción de contexto enriquecido con KGE
# ---------------------------------------------------------------------------

def _build_kge_context(incident_id: str, inc_map: dict) -> list[str]:
    """
    Construye el contexto enriquecido usando fase 5:
      - Tripletas directas del grafo
      - Casos similares (cosine similarity sobre embeddings KGE)
      - Predicciones implícitas (fase 3)

    Si los embeddings o las predicciones no están disponibles, phase5 devuelve
    solo las tripletas directas sin lanzar excepción.
    """
    subgraph = p5.build_session_subgraph(incident_id, inc_map)
    return p5.verbalize_session_subgraph(subgraph)


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class KGEAugmentedLLM:
    """
    LLM aumentado con contexto KGE enriquecido (fase 4.2).
    Delega la inferencia en un servidor vLLM (API OpenAI-compatible).
    No carga ningún peso en memoria local.

    Diferencia respecto a phase4_llm_inference.KGEAugmentedLLM:
      - El contexto incluye casos similares y predicciones implícitas (KGE).
    """

    def __init__(
        self,
        model_name: str = cfg.DEFAULT_MODEL,
        base_url:   str = cfg.VLLM_BASE_URL,
        device:       str  = "cpu",
        load_in_4bit: bool = False,
    ):
        from openai import OpenAI

        self.model_name = model_name
        self.base_url   = base_url
        self._client    = OpenAI(base_url=base_url, api_key="EMPTY")
        print(f"[Phase4.2] Cliente vLLM → {base_url}  modelo={model_name}")

    def answer(
        self,
        context_sentences: list[str],
        question: str,
        max_new_tokens: int = cfg.MAX_NEW_TOKENS,
        do_extract: bool = True,
    ) -> str:
        """
        Envía el contexto + pregunta al servidor vLLM y devuelve
        el identificador de entidad extraído de la respuesta.
        """
        messages = _build_messages(context_sentences, question)
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        if do_extract:
            return extract_answer(raw)
        return raw

    # ------------------------------------------------------------------
    # Sesión interactiva con contexto KGE enriquecido
    # ------------------------------------------------------------------

    def interactive_session(
        self,
        incident_id: str,
        props: dict,
        session_log_path: Optional[Path] = None,
        inc_map: Optional[dict] = None,
        initial_sentences: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Bucle interactivo de Q&A con contexto KGE enriquecido.
        El usuario puede confirmar, corregir o salir ('salir').
        Escribiendo 'incidencia <id>' se cambia la incidencia activa sin salir.

        Args:
            incident_id:        Etiqueta de la incidencia (ej. "incident_XYZ")
            props:              Dict {predicate: [values]} de la incidencia
            session_log_path:   Ruta donde guardar el log de la sesión
            inc_map:            Mapa completo {incident_id: props}
            initial_sentences:  Contexto pre-construido (tripletas + KGE).
                                Si None, se construye llamando a fase 5.

        Returns:
            Lista de dicts con las interacciones de la sesión.
        """
        def _print_context(inc_id: str, sents: list[str]) -> None:
            print(f"\n{'='*60}")
            print(f"Incidencia activa: {inc_id}")
            print("Contexto disponible (KGE enriquecido):")
            for s in sents:
                print(f"  {s}")
            print("(escribe 'incidencia <id>' para cambiar, 'salir' para terminar)\n")

        # Usar contexto pre-construido o construirlo ahora
        sentences = initial_sentences if initial_sentences is not None else verbalize_props(incident_id, props)

        _print_context(incident_id, sentences)

        session_log = []
        while True:
            try:
                question = input("Pregunta: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question or question.lower() == "salir":
                break

            # Cambio de incidencia dentro de la sesión
            if question.lower().startswith("incidencia "):
                new_id = question.split(None, 1)[1].strip()
                if inc_map and new_id in inc_map:
                    incident_id = new_id
                    try:
                        sentences = _build_kge_context(new_id, inc_map)
                    except Exception:
                        sentences = verbalize_props(new_id, inc_map[new_id])
                    _print_context(incident_id, sentences)
                else:
                    print(f"Incidencia no encontrada: {new_id}")
                continue

            print("\n[DEBUG] Contexto enviado al LLM:")
            for _s in sentences:
                print(f"  {_s}")
            print()
            answer = self.answer(sentences, question)
            print(f"Respuesta: {answer}")

            try:
                feedback = input("¿Correcto? (s / n / corrección): ").strip()
            except (EOFError, KeyboardInterrupt):
                feedback = ""

            entry = {
                "incident":  incident_id,
                "question":  question,
                "answer":    answer,
                "feedback":  feedback,
            }
            session_log.append(entry)

            # Si el usuario aporta corrección, añadirla al contexto
            if feedback.lower() not in ("s", "si", "sí", "yes", "y", "n", "no", ""):
                sentences.append(f"[Corrección del usuario] {feedback}")

        if session_log_path:
            session_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(session_log_path, "w", encoding="utf-8") as f:
                json.dump(session_log, f, ensure_ascii=False, indent=2)
            print(f"\nSesión guardada en {session_log_path}")

        return session_log


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(
    model_name:  str  = cfg.DEFAULT_MODEL,
    base_url:    str  = cfg.VLLM_BASE_URL,
    interactive: bool = False,
    incident_id: str  = "",
    device:      str  = "cpu",
) -> None:
    print("=" * 60)
    print("FASE 4.2 — Inferencia LLM con contexto KGE enriquecido")
    print("=" * 60)

    llm = KGEAugmentedLLM(model_name=model_name, base_url=base_url)

    import generate_corpus as gc
    g       = gc.load_graph(cfg.TTL_FILE)
    inc_map = gc.build_incident_map(g)

    if interactive:
        if not incident_id:
            incident_id = next(iter(inc_map))

        props = inc_map.get(incident_id, {})
        if not props:
            print(f"Incidencia no encontrada: {incident_id}")
            return

        # Construir contexto enriquecido con KGE
        try:
            initial_sentences = _build_kge_context(incident_id, inc_map)
        except Exception as e:
            print(f"[WARN] KGE context unavailable ({e}), using direct triples only.")
            initial_sentences = verbalize_props(incident_id, props)

        log_path = cfg.OUT_DIR / "sessions" / f"{incident_id}_kge_session.json"
        llm.interactive_session(
            incident_id, props,
            session_log_path=log_path,
            inc_map=inc_map,
            initial_sentences=initial_sentences,
        )
    else:
        # Demo rápido con un ejemplo del corpus
        if cfg.QA_CORPUS.exists():
            with open(cfg.QA_CORPUS, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            sample        = corpus["1hop"][0]
            inc_id        = sample["context_inc"]
            question_text = sample["question"].split("\n")[0]
            reference     = sample["answer"]

            try:
                sentences = _build_kge_context(inc_id, inc_map)
            except Exception as e:
                print(f"[WARN] KGE context unavailable ({e}), using direct triples only.")
                sentences = verbalize_props(inc_id, inc_map.get(inc_id, {}))
                if not sentences:
                    sentences = [f"La incidencia {inc_id} está registrada en el sistema."]

            answer = llm.answer(sentences, question_text)
            print(f"\nEjemplo demo (contexto KGE enriquecido):")
            print(f"  Incidencia:  {inc_id}")
            print(f"  Pregunta:    {question_text}")
            print(f"  Referencia:  {reference}")
            print(f"  LLM:         {answer}")
            print(f"\nContexto enviado ({len(sentences)} frases):")
            for s in sentences:
                print(f"  {s}")
        else:
            print("QA corpus no encontrado. Ejecuta generate_corpus.py primero.")

    print("\n✓ Fase 4.2 completada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM con contexto KGE enriquecido (vLLM)")
    parser.add_argument("--model",       default=cfg.DEFAULT_MODEL)
    parser.add_argument("--base-url",    default=cfg.VLLM_BASE_URL)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--incident",    default="", help="ID de incidencia para sesión interactiva")
    args = parser.parse_args()
    run(
        model_name=args.model,
        base_url=args.base_url,
        interactive=args.interactive,
        incident_id=args.incident,
    )
