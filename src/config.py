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
MODELS_DIR  = OUT_DIR / "models" / "distmult"
EMBED_DIR   = OUT_DIR / "embeddings"
PRED_DIR    = OUT_DIR / "predictions"
EVAL_DIR    = OUT_DIR / "evaluation"

# Corpus generado por generate_corpus.py
QA_CORPUS   = CORPUS_DIR / "qa_corpus.json"
TRIPLES_VRB = CORPUS_DIR / "triples_verbalized.json"

# Splits PyKEEN
TRAIN_TSV   = TRIPLES_DIR / "train.tsv"
VALID_TSV   = TRIPLES_DIR / "valid.tsv"
TEST_TSV    = TRIPLES_DIR / "test.tsv"

# Embeddings exportados
ENTITY_EMBEDDINGS   = EMBED_DIR / "entity_embeddings.pt"
RELATION_EMBEDDINGS = EMBED_DIR / "relation_embeddings.pt"
ENTITY_TO_ID        = EMBED_DIR / "entity_to_id.json"
RELATION_TO_ID      = EMBED_DIR / "relation_to_id.json"

# Predicciones y evaluación
IMPLICIT_RELS_FILE  = PRED_DIR / "implicit_relations.json"
EVAL_RESULTS_FILE   = EVAL_DIR / "results.json"

# ---------------------------------------------------------------------------
# Hiperparámetros KGE (DistMult)
# ---------------------------------------------------------------------------

EMBEDDING_DIM  = 256      # 256-dim aprovecha bien la A100 (128 para CPU)
N_EPOCHS       = 200      # más épocas → mejor convergencia en GPU
BATCH_SIZE     = 2048     # A100 40GB puede manejar batches grandes (512 para CPU)
LEARNING_RATE  = 1e-3
NEG_PER_POS    = 50       # más negativos → mejor calibración (10 para CPU)
RANDOM_SEED    = 42

TRAIN_RATIO    = 0.80
VALID_RATIO    = 0.10
# TEST_RATIO = 0.10 (implícito)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

DEFAULT_MODEL   = "mistralai/Mistral-7B-Instruct-v0.2"   # alternativa: flan-t5-large mistralai/Mistral-7B-Instruct-v0.2
MAX_CTX_LEN     = 512                      # tokens máximos de entrada
MAX_NEW_TOKENS  = 128

# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

EVAL_SAMPLE_N   = 200          # nº de Q&A a evaluar en phase6
HIT_K_VALUES    = [1, 3, 10]   # valores de k para Hit@k
TOP_K_PREDICT   = 10           # top-k en link prediction (phase3)
TOP_K_SIMILAR   = 5            # incidencias similares en CBR (phase5)
