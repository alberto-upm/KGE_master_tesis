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
TRAIN_TTL   = DATA_DIR / "train_full.ttl"   # generado por phase0_preprocess
TEST_TTL    = DATA_DIR / "test_eval.ttl"

TRIPLES_DIR = DATA_DIR / "triples"

OUT_DIR     = BASE_DIR / "out"
MAPS_DIR    = OUT_DIR / "maps"  # Mapas entity_to_id / relation_to_id (compartidos)
PRED_DIR    = OUT_DIR / "predictions"
EVAL_DIR    = OUT_DIR / "evaluation"

# Tripletas PyKEEN (flujo incident_triplets: todo en train.tsv, sin split)
TRAIN_TSV   = TRIPLES_DIR / "train.tsv"

# Predicciones
IMPLICIT_RELS_FILE  = PRED_DIR / "implicit_relations.json"

# ---------------------------------------------------------------------------
# Multi-model KGE
# ---------------------------------------------------------------------------

KGE_MODELS = ['TransE', 'RotatE', 'TransH', 'HAKE', 'DistMult', 'ComplEx', 'TorusE', 'PairRE']


def model_dir(model_name: str) -> Path:
    return OUT_DIR / "models" / model_name.lower()


def embed_dir(model_name: str) -> Path:
    return OUT_DIR / "embeddings" / model_name.lower()


def entity_embeddings_path(model_name: str) -> Path:
    return embed_dir(model_name) / "entity_embeddings.pt"


def relation_embeddings_path(model_name: str) -> Path:
    return embed_dir(model_name) / "relation_embeddings.pt"


# Mapas compartidos (independientes del modelo KGE)
ENTITY_TO_ID        = MAPS_DIR / "entity_to_id.json"
RELATION_TO_ID      = MAPS_DIR / "relation_to_id.json"

# Salida de la comparación de entrenamiento multi-modelo (phase2 --all-models)
MODEL_COMPARISON_DIR = EVAL_DIR / "model_comparison"

# ---------------------------------------------------------------------------
# Hiperparámetros KGE
# ---------------------------------------------------------------------------

EMBEDDING_DIM  = 256
N_EPOCHS       = 100
BATCH_SIZE     = 5500
LEARNING_RATE  = 1e-3
NEG_PER_POS    = 10 
RANDOM_SEED    = 42

LR_FACTOR      = 0.5
LR_PATIENCE    = 10
LR_MIN         = 1e-5

BATCH_SIZE_EVAL = 1024
SLICE_SIZE      = 5000

# t-SNE: muestreo estratificado por tipo (hasta N entidades de cada tipo).
# Coste aproximado del t-SNE en función del total muestreado (≈ 11 tipos):
#   n_per_type=500   →   ~3-5k puntos   →  1-2 min
#   n_per_type=2000  →  ~15-20k puntos  →  5-10 min
#   n_per_type=5000  →  ~40-50k puntos  →  20-40 min
TSNE_N_PER_TYPE = 2000

# Early stopping (PyKEEN EarlyStopper). Activado en phase2.
EARLY_STOP_FREQUENCY      = 1         # evaluar en validación cada N épocas
EARLY_STOP_PATIENCE       = 3         # nº de evaluaciones consecutivas sin mejora antes de parar
EARLY_STOP_RELATIVE_DELTA = 0.002     # mejora relativa mínima para considerar progreso
EARLY_STOP_METRIC         = "inverse_harmonic_mean_rank"  # MRR (mayor = mejor)

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
# Weighted Reciprocal Rank Fusion (CBR + KGE)
# ---------------------------------------------------------------------------

RRF_K   = 60   # constante de suavizado (estándar IR)
W_KGE   = 0.7  # peso del ranking KGE
W_CBR   = 0.3  # peso del ranking CBR  (W_KGE + W_CBR debe sumar 1)

# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

TOP_K_PREDICT   = 10           # top-k en link prediction (phase3)
