# AI Engineering Lab

Hands-on AI/ML learning repository for consulting. Covers from ML fundamentals to portfolio projects ready to present to clients.

**Stack**: Python + PyTorch + scikit-learn + HuggingFace + XGBoost + FastAPI

## Structure

```
docs/           → Theory and concepts (reading + reference)
notebooks/      → Interactive code (learn by doing)
portfolio/      → Complete projects (client demos)
```

## Study Roadmap

### Phase 1: Fundamentals (Week 1-2)
- [ ] Environment setup (`docs/00-environment-setup`)
- [ ] Python for AI: NumPy, Pandas, Matplotlib (`docs/01-python-for-ai`)
- [ ] Notebook: `01-numpy-pandas-essentials.ipynb`
- [ ] ML fundamentals: supervised, unsupervised, metrics (`docs/02-ml-fundamentals`)
- [ ] Notebook: `02-sklearn-ml-pipeline.ipynb`

### Phase 2: Deep Learning (Week 3-4)
- [ ] Deep Learning: neural networks, backprop, optimizers (`docs/03-deep-learning`)
- [ ] Notebook: `03-pytorch-fundamentals.ipynb`
- [ ] Computer Vision: CNNs, transfer learning (`docs/04-computer-vision`)
- [ ] Notebooks: `04-cnn-image-classification.ipynb`, `05-transfer-learning.ipynb`
- [ ] Notebook: `06-object-detection-yolo.ipynb`

### Phase 3: NLP and Tabular (Week 5-6)
- [ ] NLP: tokenization, transformers, fine-tuning (`docs/05-nlp`)
- [ ] Notebooks: `07-nlp-transformers.ipynb`, `09-fine-tuning-huggingface.ipynb`
- [ ] Tabular ML: XGBoost, LightGBM, feature engineering (`docs/06-tabular-ml`)
- [ ] Notebook: `08-tabular-xgboost-lightgbm.ipynb`

### Phase 4: Deployment and MLOps (Week 7-8)
- [ ] Model deployment: FastAPI, Docker (`docs/07-deployment`)
- [ ] Notebook: `10-model-serving-fastapi.ipynb`
- [ ] MLOps: experiment tracking, pipelines (`docs/08-mlops`)

### Phase 5: Portfolio (Week 9-12)
- [ ] Project 1: Defect detection (Computer Vision)
- [ ] Project 2: Document AI (OCR + extraction)
- [ ] Project 3: Churn prediction (Tabular ML)
- [ ] Project 4: Recommendation engine (Embeddings + ML)
- [ ] Project 5: Semantic search (NLP + Vector DB)

## Prerequisites

- Python 3.10+
- GPU recommended for Deep Learning (or use Google Colab / cloud GPU)
- Basic Python programming knowledge

## Quick Setup

```bash
# Clone repo
git clone <repo-url>
cd ai-engineering-lab

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
```

## What Each Section Is For

| Section | Purpose | When to use it |
|---------|---------|----------------|
| `docs/` | Understand theory and concepts | Before writing code |
| `notebooks/` | Learn with interactive code | While studying each topic |
| `portfolio/` | Complete demo projects | To present to clients |

## What's Not Covered Yet

See `docs/09-whats-next/README.md` for a roadmap of advanced topics you might need depending on the type of project.
