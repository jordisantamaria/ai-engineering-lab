# Development Environment for AI/ML

## Table of Contents

- [Python Setup](#python-setup)
- [Dependency Management](#dependency-management)
- [Jupyter Lab](#jupyter-lab)
- [Local GPU Setup](#local-gpu-setup)
- [Cloud GPU](#cloud-gpu)
- [Docker for AI](#docker-for-ai)
- [ML Project Structure](#ml-project-structure)
- [IDEs and Tools](#ides-and-tools)
- [Productivity Tips](#productivity-tips)

---

## Python Setup

### Version management with pyenv

pyenv lets you have multiple Python versions installed without them interfering with each other.

```bash
# Install pyenv
curl https://pyenv.run | bash

# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"

# Install a Python version
pyenv install 3.11.7
pyenv install 3.12.3

# Set global version
pyenv global 3.11.7

# Set local version (per project)
cd my-project/
pyenv local 3.12.3  # Creates .python-version
```

### Virtual environments: venv vs conda

| Feature | venv | conda |
|---|---|---|
| Comes with Python | Yes | No (install separately) |
| Python only | Yes | No (R, Julia, etc.) |
| Speed | Fast | Slower |
| Package management | pip | conda / pip |
| Precompiled binaries | PyPI wheels only | Yes (own channel) |
| Reproducibility | requirements.txt | environment.yml |
| **Recommendation** | **Simple projects/production** | **Heavy data science** |

**Recommendation: use venv for simplicity.** conda is useful when you need packages with complex C dependencies (like GDAL, cartopy), but for most AI/ML projects, venv + pip is sufficient.

```bash
# Create virtual environment with venv (RECOMMENDED)
python -m venv .venv

# Activate
source .venv/bin/activate     # Linux/Mac
.venv\Scripts\activate        # Windows

# Deactivate
deactivate
```

```bash
# Alternative with conda
conda create -n my-project python=3.11
conda activate my-project
conda deactivate
```

> **Key point:** Always use a virtual environment per project. Never install packages in the system Python.

---

## Dependency Management

### Basic requirements.txt

```txt
# requirements.txt
torch==2.2.0
transformers==4.38.0
numpy==1.26.4
pandas==2.2.0
scikit-learn==1.4.0
matplotlib==3.8.3
jupyter==1.0.0
```

### Recommended workflow with pip-compile

pip-compile (part of pip-tools) generates a requirements.txt with exact versions and all sub-dependencies. This guarantees total reproducibility.

```bash
# Install pip-tools
pip install pip-tools

# Create requirements.in (what YOU need)
# requirements.in
torch
transformers
pandas
scikit-learn

# Generate requirements.txt with pinned versions
pip-compile requirements.in -o requirements.txt

# Install from the lock file
pip-sync requirements.txt

# Update dependencies
pip-compile --upgrade requirements.in
```

### Separate dependencies by environment

```
requirements/
├── base.txt          # Core dependencies
├── dev.txt           # Testing, linting (-r base.txt)
├── gpu.txt           # PyTorch with CUDA (-r base.txt)
└── notebook.txt      # Jupyter + visualization (-r base.txt)
```

```txt
# requirements/dev.txt
-r base.txt
pytest==8.0.0
black==24.1.0
ruff==0.2.0
mypy==1.8.0
```

---

## Jupyter Lab

### Installation and configuration

```bash
pip install jupyterlab

# Start
jupyter lab --port 8888

# With a specific virtual environment
python -m ipykernel install --user --name my-project --display-name "My Project"
```

### Useful extensions

| Extension | Purpose | Installation |
|---|---|---|
| jupyterlab-git | Integrated Git | `pip install jupyterlab-git` |
| jupyterlab-code-formatter | Auto-format with black | `pip install jupyterlab-code-formatter` |
| jupyterlab-lsp | Intelligent autocompletion | `pip install jupyterlab-lsp python-lsp-server` |
| ipywidgets | Interactive widgets | `pip install ipywidgets` |
| nbstripout | Clean outputs before commit | `pip install nbstripout && nbstripout --install` |

### Recommended configuration

```python
# At the beginning of each notebook
%load_ext autoreload
%autoreload 2  # Automatically reloads modules when changed

# For inline plots
%matplotlib inline

# To see all columns of a DataFrame
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 50)
```

> **Key point:** Use notebooks for exploration and experimentation. Move stable code to `.py` modules in `src/`.

---

## Local GPU Setup

### NVIDIA (Linux/Windows)

NVIDIA GPUs are the standard in AI/ML. You need three components:

```
NVIDIA Driver  -->  CUDA Toolkit  -->  cuDNN  -->  PyTorch with CUDA
```

#### Step 1: Install NVIDIA driver

```bash
# Ubuntu
sudo apt update
sudo apt install nvidia-driver-545

# Verify
nvidia-smi
# Should show your GPU, driver version, and supported CUDA version
```

#### Step 2: Install CUDA Toolkit

```bash
# Option A: System installation (for Docker, C++ custom kernels)
# Download from https://developer.nvidia.com/cuda-downloads

# Option B: Via conda (simpler, isolated)
conda install cuda-toolkit -c nvidia

# Option C: PyTorch already includes its own CUDA (the simplest)
# You only need the NVIDIA driver installed
```

#### Step 3: Install PyTorch with CUDA

```bash
# PyTorch with CUDA 12.1 (check current version at pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Verification

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Quick test
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = x @ y  # Matrix multiplication on GPU
print(f"Result on: {z.device}")
```

### Apple Silicon (M1/M2/M3/M4)

Apple Silicon uses the MPS (Metal Performance Shaders) backend in PyTorch. It's not as fast as NVIDIA CUDA but much better than CPU.

```bash
# Install PyTorch (comes with MPS support by default)
pip install torch torchvision torchaudio
```

```python
import torch

# Verify MPS
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Use MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
z = x @ y
print(f"Result on: {z.device}")
```

#### Universal pattern for device selection

```python
def get_device():
    """Select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using: {device}")
```

> **Key point for Apple Silicon:**
> - Works well for medium-sized models and fine-tuning
> - Some operators are not supported on MPS (automatic CPU fallback)
> - Unified memory is an advantage: the GPU can use all the RAM
> - Don't expect A100 performance, but it's enough for development

---

## Cloud GPU

### When to use cloud GPU

- Your local GPU doesn't have enough VRAM (large models)
- You need multiple GPUs
- Training that takes many hours (leave it running)
- You don't have an NVIDIA GPU (only Apple Silicon)

### Provider comparison table

| Provider | GPU | Cost/hour (USD) | VRAM | Ease of use | Best for |
|---|---|---|---|---|---|
| **Google Colab** (free) | T4 | Free (limited) | 15 GB | Very easy | Experimenting, learning |
| **Google Colab Pro** | A100, V100 | ~10/month | 40-80 GB | Very easy | Medium projects |
| **AWS SageMaker** | Various | 1.50-40+ | 16-80 GB | Medium | Production, teams |
| **Lambda Labs** | A100, H100 | 1.10-3.50 | 40-80 GB | Easy | Intensive training |
| **Vast.ai** | Various | 0.20-2.00 | 8-80 GB | Medium | Tight budget |
| **RunPod** | Various | 0.40-3.50 | 16-80 GB | Easy | Cost/ease balance |
| **Modal** | A100, H100 | 0.80-4.00 | 40-80 GB | Easy (serverless) | One-off jobs |

### Google Colab: connect to your repository

```python
# In a Colab cell
from google.colab import drive
drive.mount('/content/drive')

# Clone your repo
!git clone https://github.com/your-user/your-project.git
%cd your-project

# Install dependencies
!pip install -r requirements.txt

# Verify GPU
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### AWS SageMaker (quick setup)

```python
# From a SageMaker notebook
import sagemaker
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    source_dir='src/',
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.g5.xlarge',  # A10G GPU, ~1.50/hour
    framework_version='2.2.0',
    py_version='py311',
    hyperparameters={
        'epochs': 10,
        'batch-size': 32,
        'lr': 1e-4,
    }
)

estimator.fit({'training': 's3://my-bucket/data/'})
```

### Lambda Labs / RunPod (SSH access)

```bash
# Connect via SSH
ssh ubuntu@<server-ip>

# Clone repo and start working
git clone https://github.com/your-user/your-project.git
cd your-project
pip install -r requirements.txt
python train.py
```

> **Key point:** Start with free Colab. When you need more, use RunPod or Vast.ai for cost/ease balance. SageMaker for enterprise production.

---

## Docker for AI

### Why Docker for ML

- **Reproducibility:** The same environment on your machine, in CI/CD, and in production
- **GPU:** NVIDIA Container Toolkit allows using GPUs inside Docker
- **Dependencies:** CUDA, cuDNN, and system libraries packaged together

### Base Dockerfile for ML projects

```dockerfile
# Dockerfile
# Base image with CUDA + Python
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system tools
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create alias
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Working directory
WORKDIR /app

# Copy and install dependencies first (Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Port for API or Jupyter
EXPOSE 8000

# Default command
CMD ["python", "src/train.py"]
```

### Docker Compose for development

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-dev:
    build: .
    volumes:
      - ./src:/app/src          # Hot reload of code
      - ./data:/app/data        # Local data
      - ./notebooks:/app/notebooks
      - ./models:/app/models
    ports:
      - "8888:8888"            # Jupyter
      - "8000:8000"            # API
      - "6006:6006"            # TensorBoard
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: jupyter lab --ip=0.0.0.0 --allow-root --no-browser

  tensorboard:
    image: tensorflow/tensorflow:latest
    volumes:
      - ./logs:/logs
    ports:
      - "6006:6006"
    command: tensorboard --logdir=/logs --host=0.0.0.0
```

### NVIDIA Container Toolkit

```bash
# Install (Ubuntu)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Dockerfile for CPU (Apple Silicon / no GPU)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/

EXPOSE 8000
CMD ["python", "src/train.py"]
```

---

## ML Project Structure

```
project/
├── data/               # Data - DO NOT commit (use .gitignore)
│   ├── raw/            # Original data, immutable
│   ├── processed/      # Clean data, ready for model
│   └── external/       # Data from external sources
├── notebooks/          # Jupyter notebooks for exploration
│   ├── 01-eda.ipynb
│   ├── 02-feature-eng.ipynb
│   └── 03-modeling.ipynb
├── src/                # Production code
│   ├── __init__.py
│   ├── data/           # Data loading and processing
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/         # Model definitions
│   │   ├── __init__.py
│   │   └── model.py
│   ├── training/       # Training logic
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── evaluation/     # Metrics and evaluation
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── utils.py
├── models/             # Trained models - DO NOT commit
│   └── .gitkeep
├── configs/            # Configuration and hyperparameters
│   ├── train_config.yaml
│   └── model_config.yaml
├── tests/              # Tests
│   ├── test_data.py
│   ├── test_model.py
│   └── test_training.py
├── logs/               # Training logs
├── scripts/            # Utility scripts
│   ├── train.sh
│   └── evaluate.sh
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── Makefile            # Frequent commands
├── pyproject.toml      # Project configuration
├── requirements.txt    # Pinned dependencies
└── README.md
```

### .gitignore for ML projects

```gitignore
# Data
data/
*.csv
*.parquet
*.h5
*.hdf5

# Models
models/
*.pt
*.pth
*.onnx
*.bin
*.safetensors

# Environments
.venv/
__pycache__/
*.pyc

# Jupyter
.ipynb_checkpoints/

# Logs
logs/
wandb/
mlruns/

# IDE
.vscode/
.idea/

# OS
.DS_Store
```

### Makefile for frequent commands

```makefile
.PHONY: setup train test clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

train:
	python src/training/trainer.py --config configs/train_config.yaml

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	rm -rf __pycache__ .pytest_cache logs/*
	find . -name "*.pyc" -delete
```

---

## IDEs and Tools

### VSCode: essential extensions

| Extension | Purpose |
|---|---|
| **Python** (Microsoft) | Basic Python support |
| **Pylance** | Advanced IntelliSense, type checking |
| **Jupyter** | Notebooks inside VSCode |
| **GitHub Copilot** | AI autocompletion (very useful for boilerplate) |
| **Ruff** | Ultra-fast linting |
| **Remote - SSH** | Develop on remote servers |
| **Dev Containers** | Develop inside Docker |
| **GitLens** | Advanced Git history |
| **YAML** | For hyperparameter configs |

### Recommended settings (VSCode)

```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "editor.formatOnSave": true,
  "python.formatting.provider": "black",
  "editor.rulers": [88],
  "files.exclude": {
    "**/__pycache__": true,
    "**/.ipynb_checkpoints": true
  }
}
```

---

## Productivity Tips

### tmux / screen for long training runs

When training on a remote server, if the SSH connection drops, you lose the process. tmux prevents this.

```bash
# Install tmux
sudo apt install tmux

# Create session
tmux new -s training

# Inside tmux: run your training
python train.py --epochs 100

# Detach from tmux (without killing the process)
# Ctrl+B, then D

# Reattach to the session
tmux attach -t training

# List sessions
tmux ls

# Useful commands inside tmux:
# Ctrl+B, %    -> Vertical split
# Ctrl+B, "    -> Horizontal split
# Ctrl+B, arrow -> Move between panes
```

### Monitor GPU

```bash
# See GPU usage in real time (updates every 1 second)
watch -n 1 nvidia-smi

# Nicer alternative
pip install gpustat
gpustat --watch

# Monitor from Python
import torch
print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### Other tips

```bash
# Run in background with nohup (simple alternative to tmux)
nohup python train.py > training.log 2>&1 &

# View log in real time
tail -f training.log

# Notification when training finishes (Mac)
python train.py && osascript -e 'display notification "Training finished" with title "ML"'

# Notification when training finishes (Linux)
python train.py && notify-send "Training finished"

# Measure execution time
time python train.py
```

### Checklist before starting a project

- [ ] Create Git repository
- [ ] Configure .gitignore for ML
- [ ] Create virtual environment
- [ ] Install base dependencies
- [ ] Verify GPU (if applicable)
- [ ] Create folder structure
- [ ] Configure IDE
- [ ] First EDA notebook

---

> **Summary:** The development environment is the foundation of everything. Invest time in setting it up well at the beginning and it will save you hours of headaches later. Use venv for environments, pip-compile for dependencies, and Docker when you need total reproducibility.
