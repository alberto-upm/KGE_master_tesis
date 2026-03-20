"""
Fase 4 — Inferencia con LLM aumentado con contexto KGE.

Estrategia de inyección: Graph RAG
  1. Extraer subgrafo de la incidencia (fase 5)
  2. Verbalizar tripletas a frases en español
  3. Anteponer el contexto verbalizado al prompt del LLM
  4. El LLM genera la respuesta; el usuario confirma o corrige

Modelos soportados:
  - google/flan-t5-base   (CPU, ~250 MB, por defecto)
  - google/flan-t5-large  (CPU, ~800 MB, mejor calidad)
  - mistralai/Mistral-7B-Instruct-v0.2  (GPU + cuantización 4-bit)

Uso:
  python src/phase4_llm_inference.py --model google/flan-t5-base
  python src/phase4_llm_inference.py --interactive --incident incident_XXXXX
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
    "Responde únicamente con la información contenida en el contexto del grafo. "
    "Si la respuesta no está en el contexto, responde: 'No tengo información suficiente'."
)


def build_prompt(context_sentences: list[str], question: str) -> str:
    context_block = "\n".join(f"- {s}" for s in context_sentences)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Contexto del grafo de conocimiento:\n{context_block}\n\n"
        f"Pregunta: {question}\n"
        f"Respuesta:"
    )


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
    LLM de HuggingFace aumentado con contexto de grafo de conocimiento.

    Soporta arquitecturas seq2seq (T5) y decoder-only (Mistral/Llama)
    de forma transparente.
    """

    def __init__(
        self,
        model_name: str = cfg.DEFAULT_MODEL,
        device: str = "cpu",
        load_in_4bit: bool = False,
    ):
        import torch
        from transformers import AutoTokenizer

        print(f"[Phase4] Cargando modelo: {model_name} ...")
        self.model_name  = model_name
        self.device      = device
        self.is_seq2seq  = self._is_seq2seq(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict = {}
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"]   = "auto"

        if self.is_seq2seq:
            from transformers import AutoModelForSeq2SeqLM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, **model_kwargs
            )
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )

        if not load_in_4bit:
            self.model = self.model.to(device)
        self.model.eval()
        print(f"[Phase4] Modelo listo (seq2seq={self.is_seq2seq}).")

    @staticmethod
    def _is_seq2seq(model_name: str) -> bool:
        name_lower = model_name.lower()
        return any(k in name_lower for k in ["t5", "bart", "mbart", "pegasus"])

    def answer(
        self,
        context_sentences: list[str],
        question: str,
        max_new_tokens: int = cfg.MAX_NEW_TOKENS,
    ) -> str:
        """
        Genera una respuesta dado el contexto verbalizado y la pregunta.
        """
        import torch

        prompt = build_prompt(context_sentences, question)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.MAX_CTX_LEN,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
            )

        # Para decoder-only: eliminar los tokens del prompt de la salida
        if not self.is_seq2seq:
            input_len = inputs["input_ids"].shape[1]
            outputs   = outputs[:, input_len:]

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

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
    model_name: str    = cfg.DEFAULT_MODEL,
    device: str        = "cpu",
    interactive: bool  = False,
    incident_id: str   = "",
) -> None:
    print("=" * 60)
    print("FASE 4 — Inferencia LLM con contexto KGE")
    print("=" * 60)

    llm = KGEAugmentedLLM(model_name=model_name, device=device)

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
    parser = argparse.ArgumentParser(description="LLM con contexto KGE")
    parser.add_argument("--model",       default=cfg.DEFAULT_MODEL)
    parser.add_argument("--device",      default=cfg.DEVICE)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--incident",    default="", help="ID de incidencia para sesión interactiva")
    args = parser.parse_args()
    run(
        model_name=args.model,
        device=args.device,
        interactive=args.interactive,
        incident_id=args.incident,
    )
