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
  Fase 2 — Entrenamiento KGE DistMult (PyKEEN, GPU A100)
        │
        ▼
  Fase 3 — Link prediction: inferencia de relaciones latentes
        │
        ▼
  Fase 4 — Inferencia LLM aumentada con contexto KGE (vLLM)
        │
        ▼
  Fase 5 — Subgrafo de configuración por sesión (CBR)
        │
        ▼
  Fase 6 — Validación: EM, Token F1, BERTScore, Hit@k
```

**Dominio**: Sistema de gestión de incidencias técnicas en español.
**Entidades**: incidencias, técnicos (internos/externos), clientes, grupos/equipos/categorías de soporte, estados, tipos, orígenes.

---

## Requisitos

```bash
pip install -r requirements.txt
pip install openai bert-score
```

**GPU**: NVIDIA A100-40GB (o similar). El entrenamiento KGE usa CUDA automáticamente si está disponible.

---

## Estructura del Repositorio

```
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
│   ├── phase2_kge_train.py       # Entrenamiento DistMult (PyKEEN)
│   ├── phase3_link_prediction.py # Link prediction (relaciones latentes)
│   ├── phase4_llm_inference.py   # Inferencia LLM vía vLLM
│   ├── phase5_config_subgraph.py # Subgrafo de sesión (CBR)
│   ├── phase6_validation.py      # Evaluación completa
│   └── run_pipeline.py           # Orquestador del pipeline
├── out/
│   ├── models/distmult/          # Modelo KGE entrenado (PyKEEN)
│   ├── embeddings/               # Embeddings exportados (.pt + .json)
│   ├── predictions/              # Relaciones latentes inferidas
│   ├── evaluation/               # Métricas y detalle por muestra
│   └── logs/                     # Trazas de ejecución con timestamp
└── requirements.txt
```

---

## Ejecución paso a paso

### Paso 0 — Generar el corpus de evaluación (solo si no existe)

```bash
python src/generate_corpus.py
```

Genera `data/corpus/qa_corpus.json` con ~3.700 preguntas 1-hop y ~490 cadenas multi-hop en español.

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
- `out/embeddings/entity_to_id.json`
- `out/embeddings/relation_to_id.json`

---

### Paso 2 — Entrenar el modelo KGE (DistMult)

```bash
python src/run_pipeline.py --phase 2
```

Usa CUDA automáticamente si está disponible. Parámetros en `src/config.py`:
- `EMBEDDING_DIM = 256`
- `N_EPOCHS = 200`
- `BATCH_SIZE = 2048`
- `NEG_PER_POS = 100`

**Salida**:
- `out/models/distmult/` (modelo completo PyKEEN)
- `out/embeddings/entity_embeddings.pt`
- `out/embeddings/relation_embeddings.pt`

Para ajustar hiperparámetros:
```bash
python src/run_pipeline.py --phase 2 --epochs 300 --dim 128
```

---

### Paso 3 — Inferencia de relaciones latentes (link prediction)

```bash
python src/run_pipeline.py --phase 3
```

**Salida**: `out/predictions/implicit_relations.json`

Top-K predicciones implícitas por entidad (defecto: top-10).

---

### Paso 4 — Arrancar el servidor LLM (vLLM)

El LLM se sirve localmente con vLLM, exponiendo una API compatible con OpenAI.
**Ejecutar en una terminal separada antes de las fases 4 y 6:**

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000 \
    --dtype float16 \
    --max-model-len 4096 \
    --tool-call-parser llama3_json
```

El cliente en `phase4_llm_inference.py` conecta a `http://localhost:8000/v1` (configurable en `config.py` → `VLLM_BASE_URL`).

---

### Paso 4 — Demo de inferencia LLM

```bash
python src/run_pipeline.py --phase 4
```

Ejemplo rápido: toma la primera entrada del corpus, construye el contexto del subgrafo y lanza la pregunta al LLM.

**Sesión interactiva** con una incidencia concreta:
```bash
python src/run_pipeline.py --phase 4 --interactive
python src/run_pipeline.py --phase 4 --interactive --incident incident_1497610128711762304007
```

---

### Paso 6 — Validación completa

```bash
python src/run_pipeline.py --phase 6
```

Evalúa 200 preguntas 1-hop + 200 cadenas multi-hop del corpus (sin opciones múltiples).
El LLM responde en texto libre y se compara con el `answer` del corpus.

**Métricas calculadas:**
- **Exact Match (EM)**: la respuesta contiene exactamente el identificador esperado
- **Token F1**: solapamiento de tokens (primera línea de la predicción)
- **BERTScore F1**: similitud semántica (`xlm-roberta-base`)
- **Chain Accuracy**: fracción de cadenas con todos los pasos correctos
- **Hit@k**: métricas de ranking del KGE sobre el test set (PyKEEN)

**Salida**:
- `out/evaluation/results.json` — métricas globales + desglose por tipo
- `out/evaluation/predictions.jsonl` — detalle de cada predicción
- `out/logs/pipeline_<fecha>_phase6.log` — traza completa de ejecución

Para evaluar menos muestras:
```bash
python src/run_pipeline.py --phase 6 --n-samples 100 --n-chains 50
```

---

### Pipeline completo (fases 1 → 2 → 3 → 4 → 6)

```bash
python src/run_pipeline.py --phase all
```

---

## Configuración

Todos los parámetros centralizados en `src/config.py`:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `EMBEDDING_DIM` | 256 | Dimensión embeddings KGE |
| `N_EPOCHS` | 200 | Épocas de entrenamiento |
| `BATCH_SIZE` | 2048 | Batch size (A100 40GB) |
| `NEG_PER_POS` | 100 | Negativos por tripleta positiva |
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
