# AI Engineering Lab

## Context
AI/ML learning repository for IT consulting. The goal is to be able to develop and deliver pure AI projects (not just LLMs) to clients across different industries.

## Main Stack
- **Language**: Python 3.10+
- **Deep Learning**: PyTorch
- **Classical ML**: scikit-learn
- **Gradient Boosting**: XGBoost, LightGBM
- **NLP**: HuggingFace Transformers, Sentence-Transformers
- **CV**: torchvision, ultralytics (YOLO), albumentations
- **Deployment**: FastAPI, Docker, ONNX Runtime
- **MLOps**: MLflow, Optuna
- **Documentation**: English
- **Code**: Python with comments in English

## Structure
```
docs/       → Theory (00-environment to 09-whats-next)
notebooks/  → Interactive Jupyter notebooks (01-10)
portfolio/  → 5 complete projects for client demos
```

## Python Code Conventions
- Type hints always on public functions
- Docstrings on classes and main functions
- Imports organized: stdlib → third-party → local
- Use pathlib for paths, not os.path
- Logging with logging module, not print() in production
- Seeds for reproducibility (torch.manual_seed, np.random.seed)
- Device detection: CUDA → MPS → CPU (in that order)

## Notebook Conventions
- Markdown cells in English
- Code cells with comments in English
- Each notebook is self-contained (own imports)
- Visualizations with matplotlib/seaborn
- At the end of each notebook: summary and exercises

## Portfolio Project Conventions
- Each project has: README.md, src/, requirements.txt, Dockerfile
- README includes: business problem, solution, results, client pitch, estimated ROI
- API always with FastAPI + Pydantic + health check
- Models served with lifespan (load at startup, not per request)

## ML Design Patterns
- Always start with a simple baseline before complex models
- Transfer learning before training from scratch
- SHAP for interpretability in tabular projects (the client asks "why")
- Optuna for hyperparameter tuning
- sklearn Pipeline for reproducible preprocessing
- Early stopping in deep learning
- Data augmentation before looking for more data

## Relationship with Other Repos
- `llm-playbook`: for when the solution is an LLM (prompting, RAG, agents)
- `aws-solutions-architect-lab`: for deploying models on AWS
