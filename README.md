# KGE-Augmented LLM: Reducción de Alucinaciones mediante Grafos de Conocimiento

## Descripción

Este proyecto implementa un sistema end-to-end que combina **Knowledge Graph Embeddings (KGE)** con **modelos de lenguaje grandes (LLMs)** para reducir alucinaciones mediante inyección de conocimiento estructurado. El sistema transforma un grafo RDF de gestión de incidencias en representaciones entrenables que sirven como contexto verificable para guiar las respuestas del LLM.

---

## Arquitectura del Pipeline

```
data/filtrado.ttl  (grafo RDF con ~60K incidencias)
        │
        ▼
  Fase 1 — Parseo RDF → tripletas TSV (train/valid/test)
        │
        ▼
  Fase 2 — Entrenamiento KGE TransE (PyKEEN, GPU A100)
        │
        ▼
  Fase 3 — Link prediction: inferencia de relaciones latentes
        │
        ▼
  Fase 4 — Creación guiada de incidencias (CBR + KGE + LLM conversacional)
        │
        ▼
  Fase 6 — Evaluación: EM, Token F1, BERTScore, Hit@k
```

**Dominio**: Sistema de gestión de incidencias técnicas en español.
**Entidades**: incidencias, técnicos (internos/externos), clientes, grupos/equipos/categorías de soporte, estados, tipos, orígenes.

---

## Requisitos Previos

- Python 3.11
- pip
- Git
- Hardware compatible con VLLM y Ollama (GPU NVIDIA recomendada)

---

## Instalación

### 1. Crear un entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar Hugging Face 🤗

Instala y configura el CLI de Hugging Face:

```bash
pip install huggingface-hub
hf auth login
```

**Obtener tu token de Hugging Face:**

- Ve a https://huggingface.co/settings/tokens
- Crea un nuevo token con permisos de lectura 🔑
- Usa la configuración mostrada en la imagen:

![HuggingFace Token Configuration](figuras/hugginface_token.png)

- Introduce el token cuando se te solicite ✍️

---

## Configuración de Servidores

### Servidor VLLM

Arranca el servidor en una terminal separada (requerido para las fases que usan LLM):

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000 \
    --dtype float16 \
    --max-model-len 4096 \
    --tool-call-parser llama3_json
```

---

---

## Estructura del Repositorio

```
KGE_master_tesis/
├── data/
│   ├── filtrado.ttl              # Grafo RDF fuente (~30 MB, 573K líneas)
│   ├── triples/                  # TSV generados por fase 1
│   │   ├── train.tsv             # 80% del grafo
│   │   ├── valid.tsv             # 10%
│   │   └── test.tsv              # 10%
│   └── corpus/                   # Corpus sintético de evaluación
│       ├── qa_corpus.json        # Preguntas 1-hop + cadenas multi-hop
│       ├── qa_1hop.csv           # Preguntas 1-hop (CSV)
│       ├── qa_chains_flat.csv    # Cadenas multi-hop (CSV)
│       └── triples_verbalized.json  # Tripletas verbalizadas en español
├── src/
│   ├── config.py                 # Parámetros globales y rutas
│   ├── generate_corpus.py        # Generación del corpus de evaluación
│   ├── phase1_triples.py         # Parseo RDF → TSV
│   ├── phase2_kge_train.py       # Entrenamiento KGE (TransE/DistMult/ComplEx)
│   ├── phase3_link_prediction.py # Link prediction (relaciones latentes)
│   ├── phase4_llm_inference.py   # Inferencia LLM vía vLLM
│   ├── phase4_incident_creator.py # Creador guiado de incidencias (CBR + KGE + LLM)
│   ├── phase5_config_subgraph.py # Subgrafo de sesión (CBR)
│   ├── phase6_incident_creator_eval.py # Evaluación del incident creator
│   ├── phase6_model_comparison.py    # Comparación de modelos KGE
│   └── run_pipeline.py           # Orquestador del pipeline
├── figuras/
│   └── hugginface_token.png      # Guía de configuración de token HuggingFace
├── out/
│   ├── maps/                     # Mapas entity_to_id / relation_to_id (compartidos)
│   ├── models/
│   │   ├── transe/               # Modelo TransE entrenado (PyKEEN)
│   │   ├── distmult/             # Modelo DistMult entrenado (PyKEEN)
│   │   └── complex/              # Modelo ComplEx entrenado (PyKEEN)
│   ├── embeddings/
│   │   ├── transe/               # Embeddings TransE (.pt)
│   │   ├── distmult/             # Embeddings DistMult (.pt)
│   │   └── complex/              # Embeddings ComplEx (.pt)
│   ├── predictions/              # Relaciones latentes inferidas
│   ├── evaluation/
│   │   ├── incident_creator/     # Resultados evaluación incident creator
│   │   └── model_comparison/     # Comparación de modelos
│   ├── sessions/                 # Sesiones interactivas guardadas
│   └── logs/                     # Trazas de ejecución con timestamp
├── requirements.txt
├── README.md                     # Este archivo
└── .gitignore
```

---

## Ejecución del Pipeline paso a paso

### Paso 0 — Generar el corpus de evaluación (solo si no existe)

```bash
python src/generate_corpus.py
```

Genera `data/corpus/qa_corpus.json` con ~3.700 preguntas 1-hop y ~490 cadenas multi-hop en español. Este corpus se usa para evaluar la calidad de las respuestas del LLM en las fases posteriores.

**Salida**:
- `data/corpus/qa_corpus.json` — Preguntas y respuestas para evaluación
- `data/corpus/qa_1hop.csv` — Formato CSV para análisis
- `data/corpus/qa_chains_flat.csv` — Cadenas multi-hop en CSV
- `data/corpus/triples_verbalized.json` — Tripletas verbalizadas en español

---

### Paso 1 — Parsear el grafo RDF a tripletas TSV

```bash
python src/run_pipeline.py --phase 1
```

**Entrada**: `data/filtrado.ttl`  
**Salida**:
- `data/triples/train.tsv` (80%)
- `data/triples/valid.tsv` (10%)
- `data/triples/test.tsv` (10%)
- `out/maps/entity_to_id.json` (mapas compartidos entre modelos)
- `out/maps/relation_to_id.json`

---

### Paso 2 — Entrenar el modelo KGE (TransE por defecto)

```bash
python src/run_pipeline.py --phase 2
```

Entrena **TransE** con CUDA automáticamente. Parámetros en `src/config.py`:
- `EMBEDDING_DIM = 256`
- `N_EPOCHS = 600`
- `BATCH_SIZE = 2048`
- `NEG_PER_POS = 50`

**Salida**:
- `out/models/transe/` (modelo completo PyKEEN)
- `out/embeddings/entity_embeddings.pt`
- `out/embeddings/relation_embeddings.pt`

**Otras opciones**:

Elegir otro modelo:
```bash
python src/run_pipeline.py --phase 2 --kge-model DistMult
python src/run_pipeline.py --phase 2 --kge-model ComplEx
```

Entrenar los tres modelos a la vez:
```bash
python src/run_pipeline.py --phase 2 --all-models
```

Ajustar hiperparámetros:
```bash
python src/run_pipeline.py --phase 2 --epochs 300 --dim 128 --device cuda
```

---

### Paso 3 — Inferencia de relaciones latentes (link prediction)

```bash
python src/run_pipeline.py --phase 3
```

**Salida**: `out/predictions/implicit_relations.json`

Top-K predicciones implícitas por entidad (defecto: top-10).

---

### Paso 4 — Crear una incidencia guiada (CBR + KGE + LLM)

**Sin LLM** (menú numerado, no requiere vLLM):
```bash
python src/run_pipeline.py --phase create_incident --no-llm
```

**Con LLM conversacional** (requiere vLLM corriendo en `localhost:8000`):
```bash
python src/run_pipeline.py --phase create_incident
```

**Cambiar el modelo KGE**:
```bash
python src/run_pipeline.py --phase create_incident --kge-model DistMult
```

---

### Paso 5 — Evaluación del incident creator

```bash
python src/run_pipeline.py --phase 6
```

Evalúa el creador de incidencias contra un conjunto de test. El LLM responde sobre propiedades de incidencias y se comparan con los valores esperados.

**Métricas calculadas:**
- **Hit@1, Hit@3, Hit@10**: incidencias cuyo valor verdadero está en el top-K de recomendaciones
- **Presencia de proxies CBR**: fracción de iteraciones con proxies similares
- **Integridad de recomendaciones**: correctitud del ranking KGE

**Salida**:
- `out/evaluation/incident_creator/` — resultados por timestamp
- `out/logs/pipeline_<fecha>_phase6.log` — traza completa de ejecución

**Opciones**:

Evaluar con menos muestras (prueba rápida):
```bash
python src/run_pipeline.py --phase 6 --n-samples 50
```

Cambiar el modelo KGE:
```bash
python src/run_pipeline.py --phase 6 --kge-model DistMult
```

---

### Pipeline completo (fases 1 → 2 → 3 → create_incident → 6)

```bash
python src/run_pipeline.py --phase all
```

---

## Configuración

Todos los parámetros centralizados en `src/config.py`:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `EMBEDDING_DIM` | 256 | Dimensión embeddings KGE |
| `N_EPOCHS` | 600 | Épocas de entrenamiento |
| `BATCH_SIZE` | 512 | Batch size (reduce si hay OOM en GPU) |
| `NEG_PER_POS` | 10 | Negativos por tripleta positiva |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | Endpoint del servidor vLLM |
| `DEFAULT_MODEL` | `meta-llama/Meta-Llama-3-8B-Instruct` | Modelo LLM |
| `MAX_NEW_TOKENS` | 128 | Tokens máximos de respuesta |
| `EVAL_SAMPLE_N` | 200 | Muestras por split en validación |
| `TOP_K_PREDICT` | 10 | Top-K en link prediction |

---

## Logs de ejecución

Cada ejecución de `run_pipeline.py` genera automáticamente un fichero de log en `out/logs/` con el timestamp y la fase ejecutada:

```
out/logs/pipeline_20260324_143022_phase6.log
out/logs/pipeline_20260324_090011_phaseall.log
```

El fichero captura toda la salida de la terminal (progreso, métricas, errores).

---

## Literatura de Referencia

| Paper | Autores |
|-------|---------|
| *Let Your Graph Do the Talking: Encoding Structured Data for LLMs* | Perozzi et al. (2024) |
| *Injecting Knowledge Graphs into Large Language Models* | Coppolillo (2024) |
| *Talk Like a Graph: Encoding Graphs for Large Language Models* | Fatemi et al. (2024) |
| *Can Knowledge Graphs Reduce Hallucinations in LLMs? A Survey* | Agrawal et al. (2024) |
| *Neurosymbolic AI for Enhancing Instructability in Generative AI* | Sheth et al. (2023) |
