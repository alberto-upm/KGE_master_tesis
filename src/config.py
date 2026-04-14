"""
Configuración compartida del pipeline KGE + LLM.
Importar desde cualquier módulo src/phaseN_*.py para usar rutas y parámetros.
"""

from pathlib import Path

# Auto-detección de dispositivo (CUDA si está disponible, si no CPU)
try:
    import torch as _torch
    DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
TTL_FILE    = DATA_DIR / "filtrado.ttl"

TRIPLES_DIR = DATA_DIR / "triples"
CORPUS_DIR  = DATA_DIR / "corpus"

OUT_DIR     = BASE_DIR / "out"
PRED_DIR    = OUT_DIR / "predictions"
EVAL_DIR    = OUT_DIR / "evaluation"

# Corpus generado por generate_corpus.py
QA_CORPUS   = CORPUS_DIR / "qa_corpus.json"
TRIPLES_VRB = CORPUS_DIR / "triples_verbalized.json"

# Splits PyKEEN
TRAIN_TSV   = TRIPLES_DIR / "train.tsv"
VALID_TSV   = TRIPLES_DIR / "valid.tsv"
TEST_TSV    = TRIPLES_DIR / "test.tsv"

# Predicciones y evaluación
IMPLICIT_RELS_FILE  = PRED_DIR / "implicit_relations.json"
EVAL_RESULTS_FILE   = EVAL_DIR / "results.json"

# ---------------------------------------------------------------------------
# Multi-model KGE
# ---------------------------------------------------------------------------

KGE_MODELS = ['TransE', 'DistMult', 'ComplEx']


def model_dir(model_name: str) -> Path:
    return OUT_DIR / "models" / model_name.lower()


def embed_dir(model_name: str) -> Path:
    return OUT_DIR / "embeddings" / model_name.lower()


def entity_embeddings_path(model_name: str) -> Path:
    return embed_dir(model_name) / "entity_embeddings.pt"


def relation_embeddings_path(model_name: str) -> Path:
    return embed_dir(model_name) / "relation_embeddings.pt"


def entity_to_id_path(model_name: str) -> Path:
    return embed_dir(model_name) / "entity_to_id.json"


def relation_to_id_path(model_name: str) -> Path:
    return embed_dir(model_name) / "relation_to_id.json"


# Rutas por defecto apuntan a DistMult (backward compatibility con phase4/5/6)
MODELS_DIR          = model_dir('distmult')
EMBED_DIR           = embed_dir('distmult')
ENTITY_EMBEDDINGS   = entity_embeddings_path('distmult')
RELATION_EMBEDDINGS = relation_embeddings_path('distmult')
ENTITY_TO_ID        = entity_to_id_path('distmult')
RELATION_TO_ID      = relation_to_id_path('distmult')

# ---------------------------------------------------------------------------
# GLiNER2
# ---------------------------------------------------------------------------

GLINER_MODEL = "fastino/gliner2-base-v1"

# ---------------------------------------------------------------------------
# Corpus de evaluación link prediction (por modelo)
# ---------------------------------------------------------------------------

LP_EVAL_CORPUS       = CORPUS_DIR / "link_prediction_eval.json"
MODEL_COMPARISON_DIR = EVAL_DIR / "model_comparison"

# Entity-to-entity evaluation corpus
ENTITY_EVAL_CORPUS = CORPUS_DIR / "entity_to_entity_eval.json"

# Pares a evaluar: (source_prop, target_prop)
# Lectura: "dado el valor de source_prop, predice el valor de target_prop"
ENTITY_EVAL_PAIRS = [
    ("int_hasCustomer",  "hasTechnician"),        # empresa → técnico
    ("hasSupportGroup",  "hasSupportCategory"),   # grupo de soporte → categoría
    ("hasTypeInc",       "hasTechnician"),         # tipo de incidencia → técnico
    ("int_hasCustomer",  "hasSupportGroup"),       # empresa → grupo de soporte
    ("hasSupportGroup",  "hasTechnician"),         # grupo de soporte → técnico
]

# ---------------------------------------------------------------------------
# Hiperparámetros KGE (DistMult)
# ---------------------------------------------------------------------------

EMBEDDING_DIM  = 256      # 256-dim aprovecha bien la A100 (128 para CPU)
N_EPOCHS       = 600      # más épocas → mejor convergencia en GPU
BATCH_SIZE     = 2048     # A100 40GB puede manejar batches grandes (512 para CPU)
LEARNING_RATE  = 1e-3
NEG_PER_POS    = 50       # más negativos → mejor calibración (10 para CPU)
RANDOM_SEED    = 42

TRAIN_RATIO    = 0.80
VALID_RATIO    = 0.10
# TEST_RATIO = 0.10 (implícito)

# ---------------------------------------------------------------------------
# LLM — vLLM (servidor local OpenAI-compatible)
#
# Arrancar con:
#   vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
#       --port 8000 --dtype float16 --max-model-len 4096 \
#       --tool-call-parser llama3_json
# ---------------------------------------------------------------------------

VLLM_BASE_URL   = "http://localhost:8000/v1"
DEFAULT_MODEL   = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_NEW_TOKENS  = 128

# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

EVAL_SAMPLE_N   = 200          # nº de Q&A a evaluar en phase6
HIT_K_VALUES    = [1, 3, 10]   # valores de k para Hit@k
TOP_K_PREDICT   = 10           # top-k en link prediction (phase3)
TOP_K_SIMILAR   = 5            # incidencias similares en CBR (phase5)
