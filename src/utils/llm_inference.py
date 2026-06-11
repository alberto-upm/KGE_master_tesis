"""
Fase 4 — Inferencia con LLM aumentado con contexto KGE.

Librería usada por phase5_incident_creator para generar el resumen final de
la incidencia. Expone:

  - KGEAugmentedLLM : cliente del servidor vLLM (API OpenAI-compatible)
  - verbalize_props : convierte {predicado: valores} en frases en español

El LLM se sirve externamente con vLLM. Arrancar antes de usar este módulo:

  vllm serve meta-llama/Meta-Llama-3-8B-Instruct \\
      --port 8000 --dtype float16 --max-model-len 4096 \\
      --tool-call-parser llama3_json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from utils.graph_utils import PRED_TEMPLATES_ES

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
    usando PRED_TEMPLATES_ES importado de graph_utils.
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
