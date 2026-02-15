# AI Engineering Lab

## Contexto
Repositorio de aprendizaje de AI/ML para consultoría IT. El objetivo es poder desarrollar y entregar proyectos de AI puro (no solo LLMs) a clientes de diferentes industrias.

## Stack principal
- **Lenguaje**: Python 3.10+
- **Deep Learning**: PyTorch
- **ML clásico**: scikit-learn
- **Gradient Boosting**: XGBoost, LightGBM
- **NLP**: HuggingFace Transformers, Sentence-Transformers
- **CV**: torchvision, ultralytics (YOLO), albumentations
- **Deployment**: FastAPI, Docker, ONNX Runtime
- **MLOps**: MLflow, Optuna
- **Documentación**: Español
- **Código**: Python con comments en inglés

## Estructura
```
docs/       → Teoría (00-environment a 09-whats-next)
notebooks/  → Jupyter notebooks interactivos (01-10)
portfolio/  → 5 proyectos completos para demos a clientes
```

## Convenciones de código Python
- Type hints siempre en funciones públicas
- Docstrings en clases y funciones principales
- Imports organizados: stdlib → third-party → local
- Usar pathlib para paths, no os.path
- Logging con logging module, no print() en producción
- Seeds para reproducibilidad (torch.manual_seed, np.random.seed)
- Device detection: CUDA → MPS → CPU (en ese orden)

## Convenciones de notebooks
- Markdown cells en español
- Code cells con comments en inglés
- Cada notebook es autocontenido (imports propios)
- Visualizaciones con matplotlib/seaborn
- Al final de cada notebook: resumen y ejercicios

## Convenciones de proyectos portfolio
- Cada proyecto tiene: README.md, src/, requirements.txt, Dockerfile
- README incluye: problema de negocio, solución, resultados, pitch para cliente, ROI estimado
- API siempre con FastAPI + Pydantic + health check
- Modelos servidos con lifespan (cargar al inicio, no por request)

## Patrones de diseño ML
- Siempre empezar con un baseline simple antes de modelos complejos
- Transfer learning antes de entrenar de cero
- SHAP para interpretabilidad en proyectos tabular (el cliente pregunta "por qué")
- Optuna para hyperparameter tuning
- Pipeline de sklearn para preprocesamiento reproducible
- Early stopping en deep learning
- Data augmentation antes de buscar más datos

## Relación con otros repos
- `llm-playbook`: para cuando la solución es un LLM (prompting, RAG, agentes)
- `aws-solutions-architect-lab`: para desplegar modelos en AWS
