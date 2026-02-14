# AI Engineering Lab

Repositorio de aprendizaje práctico de AI/ML para consultoría. Cubre desde fundamentos de ML hasta proyectos de portfolio listos para presentar a clientes.

**Stack**: Python + PyTorch + scikit-learn + HuggingFace + XGBoost + FastAPI

## Estructura

```
docs/           → Teoría y conceptos (lectura + referencia)
notebooks/      → Código interactivo (aprender haciendo)
portfolio/      → Proyectos completos (demos para clientes)
```

## Roadmap de estudio

### Fase 1: Fundamentos (Semana 1-2)
- [ ] Setup del entorno (`docs/00-environment-setup`)
- [ ] Python para AI: NumPy, Pandas, Matplotlib (`docs/01-python-for-ai`)
- [ ] Notebook: `01-numpy-pandas-essentials.ipynb`
- [ ] ML fundamentals: supervised, unsupervised, métricas (`docs/02-ml-fundamentals`)
- [ ] Notebook: `02-sklearn-ml-pipeline.ipynb`

### Fase 2: Deep Learning (Semana 3-4)
- [ ] Deep Learning: redes neuronales, backprop, optimizers (`docs/03-deep-learning`)
- [ ] Notebook: `03-pytorch-fundamentals.ipynb`
- [ ] Computer Vision: CNNs, transfer learning (`docs/04-computer-vision`)
- [ ] Notebooks: `04-cnn-image-classification.ipynb`, `05-transfer-learning.ipynb`
- [ ] Notebook: `06-object-detection-yolo.ipynb`

### Fase 3: NLP y Tabular (Semana 5-6)
- [ ] NLP: tokenización, transformers, fine-tuning (`docs/05-nlp`)
- [ ] Notebooks: `07-nlp-transformers.ipynb`, `09-fine-tuning-huggingface.ipynb`
- [ ] Tabular ML: XGBoost, LightGBM, feature engineering (`docs/06-tabular-ml`)
- [ ] Notebook: `08-tabular-xgboost-lightgbm.ipynb`

### Fase 4: Deployment y MLOps (Semana 7-8)
- [ ] Model deployment: FastAPI, Docker (`docs/07-deployment`)
- [ ] Notebook: `10-model-serving-fastapi.ipynb`
- [ ] MLOps: experiment tracking, pipelines (`docs/08-mlops`)

### Fase 5: Portfolio (Semana 9-12)
- [ ] Proyecto 1: Detección de defectos (Computer Vision)
- [ ] Proyecto 2: Document AI (OCR + extracción)
- [ ] Proyecto 3: Churn prediction (Tabular ML)
- [ ] Proyecto 4: Motor de recomendaciones (Embeddings + ML)
- [ ] Proyecto 5: Búsqueda semántica (NLP + Vector DB)

## Prerequisitos

- Python 3.10+
- GPU recomendada para Deep Learning (o usar Google Colab / cloud GPU)
- Conocimientos básicos de programación en Python

## Setup rápido

```bash
# Clonar repo
git clone <repo-url>
cd ai-engineering-lab

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Lanzar Jupyter
jupyter lab
```

## Para qué sirve cada parte

| Sección | Para qué | Cuándo usarla |
|---------|----------|---------------|
| `docs/` | Entender la teoría y conceptos | Antes de escribir código |
| `notebooks/` | Aprender con código interactivo | Mientras estudias cada tema |
| `portfolio/` | Proyectos completos de demo | Para presentar a clientes |

## Qué falta por cubrir

Ver `docs/09-whats-next/README.md` para un roadmap de temas avanzados que podrías necesitar según el tipo de proyecto.
