# KGE-Augmented LLM: Reducción de Alucinaciones mediante Grafos de Conocimiento

## Descripción

Este proyecto investiga y desarrolla un sistema que combina **Knowledge Graph Embeddings (KGE)** con **modelos de lenguaje grandes (LLMs)** para reducir las alucinaciones mediante la inyección de conocimiento estructurado. El enfoque consiste en transformar un grafo RDF en representaciones entrenables que sirven como fuente de contexto verificable para guiar las respuestas del modelo de lenguaje.

---

## Motivación

Los LLMs son propensos a generar información incorrecta o inventada (alucinaciones), especialmente en dominios donde la precisión factual es crítica. Los grafos de conocimiento, al ser fuentes de información estructurada y verificable, representan un complemento natural para anclar las respuestas del modelo a hechos concretos.

Este trabajo explora cómo integrar ambas tecnologías de forma eficiente en un flujo end-to-end.

---

## Literatura de Referencia

Los siguientes trabajos constituyen la base teórica del proyecto:

| Paper | Autores |
|-------|---------|
| *Let Your Graph Do the Talking: Encoding Structured Data for LLMs* | Bryan Perozzi, Bahare Fatemi, Dustin Zelle, Anton Tsitsulin, Mehran Kazemi, Rami Al-Rfou, Jonathan Halcrow |
| *Injecting Knowledge Graphs into Large Language Models* | Erica Coppolillo |
| *Talk Like a Graph: Encoding Graphs for Large Language Models* | Bahare Fatemi, Jonathan Halcrow, Bryan Perozzi |
| *Can Knowledge Graphs Reduce Hallucinations in LLMs? A Survey* | Garima Agrawal, Tharindu Kumarage, Zeyad Alghamdi, Huan Liu |
| *Neurosymbolic AI for Enhancing Instructability in Generative AI* | Amit Sheth, Vishal Pallagani, Kaushik Roy |

---

## Arquitectura del Sistema

### Prueba de Concepto (PoC) End-to-End

El pipeline actual implementa los siguientes pasos:

```
Grafo RDF
    │
    ▼
Extracción de tripletas (sujeto, predicado, objeto)
    │
    ▼
Entrenamiento del modelo KGE (TransE)
    │
    ▼
Recuperación de contexto relevante desde el grafo
    │
    ▼
Evaluación del LLM en tareas de preguntas y respuestas
```

### Flujo Iterativo (en desarrollo)

Se está construyendo un sistema conversacional iterativo donde:

1. El grafo actúa como base de conocimiento estructurado.
2. El sistema propone hipótesis relevantes a partir de la información existente.
3. El usuario confirma o refuta la información mediante preguntas sucesivas.
4. El conocimiento confirmado se incorpora dinámicamente al contexto.

---

## Estado del Proyecto

### Completado

- [x] Revisión del estado del arte en KGE e inyección de conocimiento en LLMs

### En Progreso

- [ ] Pipeline end-to-end: RDF → tripletas → KGE (TransE) → contexto para QA
- [ ] Verbalización de tripletas mediante plantillas o LLM ligero de Hugging Face
- [ ] Generación de corpus sintético de evaluación (preguntas y respuestas) con modelo ligero de Hugging Face
  - Consultas directas (1-hop)
  - Consultas multi-hop
- [ ] Validación de que el sistema recupera información del grafo sin introducir alucinaciones
- [ ] Desarrollo del flujo iterativo de hipótesis y confirmación

---

## Metodología de Evaluación

El sistema se evalúa sobre un corpus sintético generado con un modelo ligero de Hugging Face, diseñado para comprobar:

- **Recuperación 1-hop**: El modelo responde correctamente a preguntas que requieren una única relación del grafo.
- **Recuperación multi-hop**: El modelo encadena correctamente varias relaciones para responder preguntas complejas.
- **Ausencia de alucinaciones**: Las respuestas generadas no introducen información que no esté presente en el grafo.

---

## Tecnologías

- **Representación del conocimiento**: RDF, tripletas (sujeto, predicado, objeto)
- **Modelo KGE**: TransE
- **Modelos de lenguaje**: LLMs de Hugging Face (modelo ligero para generación del corpus)
- **Paradigma**: Neurosimbólico — combinación de razonamiento simbólico (grafo) con aprendizaje profundo (LLM)

---

## Estructura del Repositorio

```
├── data/
│   ├── raw/             # Grafo RDF original
│   ├── triples/         # Tripletas extraídas para entrenamiento KGE
│   └── corpus/          # Corpus sintético de evaluación (QA)
├── models/
│   ├── kge/             # Modelo TransE entrenado
│   └── verbalization/   # Plantillas de verbalización de tripletas
├── src/
│   ├── extraction/      # Extracción de tripletas desde RDF
│   ├── training/        # Entrenamiento del modelo KGE
│   ├── retrieval/       # Recuperación de contexto desde el grafo
│   ├── qa/              # Pipeline de preguntas y respuestas
│   └── evaluation/      # Scripts de evaluación
├── notebooks/           # Experimentos y análisis exploratorio
└── README.md
```

---

## Referencias

- Perozzi et al. (2024). *Let Your Graph Do the Talking*.
- Coppolillo, E. (2024). *Injecting Knowledge Graphs into Large Language Models*.
- Fatemi et al. (2024). *Talk Like a Graph*.
- Agrawal et al. (2024). *Can Knowledge Graphs Reduce Hallucinations in LLMs? A Survey*.
- Sheth et al. (2023). *Neurosymbolic AI for Enhancing Instructability in Generative AI*.
