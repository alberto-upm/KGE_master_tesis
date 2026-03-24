"""
Fase 4 — Inferencia con LLM aumentado con contexto KGE.

Estrategia de inyección: Graph RAG
  1. Extraer subgrafo de la incidencia (fase 5)
  2. Verbalizar tripletas a frases en español
  3. Enviar el contexto verbalizado + pregunta al servidor vLLM
  4. El LLM genera la respuesta; el usuario confirma o corrige

El LLM se sirve externamente con vLLM (API OpenAI-compatible).
Arrancar antes de ejecutar este módulo:

  vllm serve meta-llama/Meta-Llama-3-8B-Instruct \\
      --port 8000 --dtype float16 --max-model-len 4096 \\
      --tool-call-parser llama3_json

Uso:
  python src/phase4_llm_inference.py
  python src/phase4_llm_inference.py --interactive --incident incident_XXXXX
  python src/phase4_llm_inference.py --base-url http://localhost:8000/v1 \\
      --model meta-llama/Meta-Llama-3-8B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
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
    # Eliminar posibles ecos del prompt (Mistral a veces repite el [INST])
    for marker in ("Identificador:", "Respuesta:", "[/INST]"):
        if marker in raw:
            raw = raw.split(marker)[-1]

    for line in raw.split("\n"):
        line = line.strip(" .-·\t")
        if line and not line.startswith("[") and not line.startswith("Contexto"):
            return line

    return raw.strip()


# ---------------------------------------------------------------------------
# Verbalización rápida de un subgrafo
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
# Índice de tripletas verbalizadas (O(1) lookup por sujeto)
# ---------------------------------------------------------------------------

_verbalized_index: Optional[dict] = None


def _load_verbalized_index() -> dict:
    global _verbalized_index
    if _verbalized_index is not None:
        return _verbalized_index

    if not cfg.TRIPLES_VRB.exists():
        _verbalized_index = {}
        return _verbalized_index

    print(f"[Phase4] Cargando índice de tripletas verbalizadas desde {cfg.TRIPLES_VRB} ...")
    with open(cfg.TRIPLES_VRB, "r", encoding="utf-8") as f:
        data = json.load(f)

    index: dict[str, list[str]] = {}
    for item in data:
        subj = item.get("subject", "")
        text = item.get("verbalized", "")
        if subj and text:
            index.setdefault(subj, []).append(text)
    _verbalized_index = index
    print(f"[Phase4] Índice listo: {len(index):,} sujetos indexados.")
    return index


def get_verbalized_sentences(incident_id: str) -> list[str]:
    """
    Devuelve las frases verbalizadas para incident_id usando el índice
    precargado de triples_verbalized.json. Si no existe, cae al verbalizer
    basado en PRED_TEMPLATES_ES.
    """
    idx = _load_verbalized_index()
    if incident_id in idx:
        return idx[incident_id]
    return []


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class KGEAugmentedLLM:
    """
    LLM aumentado con contexto KGE que delega la inferencia en un servidor
    vLLM (API OpenAI-compatible).  No carga ningún peso en memoria.

    Requiere que vLLM esté corriendo antes de instanciar esta clase:
      vllm serve meta-llama/Meta-Llama-3-8B-Instruct \\
          --port 8000 --dtype float16 --max-model-len 4096 \\
          --tool-call-parser llama3_json
    """

    def __init__(
        self,
        model_name: str = cfg.DEFAULT_MODEL,
        base_url:   str = cfg.VLLM_BASE_URL,
        # Los siguientes parámetros se aceptan por compatibilidad con llamadas
        # anteriores pero se ignoran (el servidor vLLM gestiona el dispositivo).
        device:       str  = "cpu",
        load_in_4bit: bool = False,
    ):
        from openai import OpenAI

        self.model_name = model_name
        self.base_url   = base_url
        self._client    = OpenAI(base_url=base_url, api_key="EMPTY")
        print(f"[Phase4] Cliente vLLM → {base_url}  modelo={model_name}")

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
    # Sesión interactiva (Fase 4 + Fase 5 combinadas)
    # ------------------------------------------------------------------

    def interactive_session(
        self,
        incident_id: str,
        props: dict,
        implicit_preds: Optional[list] = None,
        session_log_path: Optional[Path] = None,
    ) -> list[dict]:
        """
        Bucle interactivo de Q&A para una incidencia concreta.
        El usuario puede confirmar, corregir o salir ('salir').

        Args:
            incident_id:      Etiqueta de la incidencia (ej. "incident_XYZ")
            props:            Dict {predicate: [values]} de la incidencia
            implicit_preds:   Predicciones implícitas de phase3 (opcional)
            session_log_path: Ruta donde guardar el log de la sesión

        Returns:
            Lista de dicts con las interacciones de la sesión.
        """
        # Verbalizar subgrafo directo
        sentences = verbalize_props(incident_id, props)

        # Añadir predicciones implícitas si las hay
        if implicit_preds:
            for pred in implicit_preds[:3]:
                sentences.append(
                    f"[Implícito] {pred.get('description', str(pred))}"
                )

        print(f"\n{'='*60}")
        print(f"Sesión para incidencia: {incident_id}")
        print("Contexto disponible:")
        for s in sentences:
            print(f"  {s}")
        print("(escribe 'salir' para terminar)\n")

        session_log = []
        while True:
            try:
                question = input("Pregunta: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question or question.lower() == "salir":
                break

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
    # device se acepta por compatibilidad con run_pipeline.py pero se ignora
    device:      str  = "cpu",
) -> None:
    print("=" * 60)
    print("FASE 4 — Inferencia LLM con contexto KGE")
    print("=" * 60)

    llm = KGEAugmentedLLM(model_name=model_name, base_url=base_url)

    if interactive:
        # Cargar mapa de incidencias para obtener las propiedades
        import generate_corpus as gc
        g        = gc.load_graph(cfg.TTL_FILE)
        inc_map  = gc.build_incident_map(g)

        if not incident_id:
            # Tomar una incidencia de ejemplo
            incident_id = next(iter(inc_map))

        props = inc_map.get(incident_id, {})
        if not props:
            print(f"Incidencia no encontrada: {incident_id}")
            return

        log_path = cfg.OUT_DIR / "sessions" / f"{incident_id}_session.json"
        llm.interactive_session(incident_id, props, session_log_path=log_path)
    else:
        # Demo rápido con un ejemplo del corpus
        if cfg.QA_CORPUS.exists():
            with open(cfg.QA_CORPUS, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            sample = corpus["1hop"][0]
            inc_id  = sample["context_inc"]
            question_text = sample["question"].split("\n")[0]
            reference     = sample["answer"]

            sentences = get_verbalized_sentences(inc_id)
            if not sentences:
                sentences = [f"La incidencia {inc_id} está registrada en el sistema."]

            answer = llm.answer(sentences, question_text)
            print(f"\nEjemplo demo:")
            print(f"  Incidencia:  {inc_id}")
            print(f"  Pregunta:    {question_text}")
            print(f"  Referencia:  {reference}")
            print(f"  LLM:         {answer}")
        else:
            print("QA corpus no encontrado. Ejecuta generate_corpus.py primero.")

    print("\n✓ Fase 4 completada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM con contexto KGE (vLLM)")
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
