# Entorno de Desarrollo para AI/ML

## Tabla de Contenidos

- [Setup de Python](#setup-de-python)
- [Gestion de Dependencias](#gestion-de-dependencias)
- [Jupyter Lab](#jupyter-lab)
- [GPU Setup Local](#gpu-setup-local)
- [GPU en la Nube](#gpu-en-la-nube)
- [Docker para AI](#docker-para-ai)
- [Estructura de Proyecto ML](#estructura-de-proyecto-ml)
- [IDEs y Herramientas](#ides-y-herramientas)
- [Tips de Productividad](#tips-de-productividad)

---

## Setup de Python

### Gestion de versiones con pyenv

pyenv te permite tener multiples versiones de Python instaladas sin que interfieran entre si.

```bash
# Instalar pyenv
curl https://pyenv.run | bash

# Agregar a ~/.bashrc o ~/.zshrc
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"

# Instalar una version de Python
pyenv install 3.11.7
pyenv install 3.12.3

# Establecer version global
pyenv global 3.11.7

# Establecer version local (por proyecto)
cd mi-proyecto/
pyenv local 3.12.3  # Crea .python-version
```

### Entornos virtuales: venv vs conda

| Caracteristica | venv | conda |
|---|---|---|
| Viene con Python | Si | No (instalar aparte) |
| Solo Python | Si | No (R, Julia, etc.) |
| Velocidad | Rapido | Mas lento |
| Gestion de paquetes | pip | conda / pip |
| Binarios precompilados | Solo PyPI wheels | Si (propio canal) |
| Reproducibilidad | requirements.txt | environment.yml |
| **Recomendacion** | **Proyectos simples/produccion** | **Ciencia de datos pesada** |

**Recomendacion: usar venv por simplicidad.** conda es util cuando necesitas paquetes con dependencias C complejas (como GDAL, cartopy), pero para la mayoria de proyectos de AI/ML, venv + pip es suficiente.

```bash
# Crear entorno virtual con venv (RECOMENDADO)
python -m venv .venv

# Activar
source .venv/bin/activate     # Linux/Mac
.venv\Scripts\activate        # Windows

# Desactivar
deactivate
```

```bash
# Alternativa con conda
conda create -n mi-proyecto python=3.11
conda activate mi-proyecto
conda deactivate
```

> **Punto clave:** Siempre usa un entorno virtual por proyecto. Nunca instales paquetes en el Python del sistema.

---

## Gestion de Dependencias

### requirements.txt basico

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

### Flujo recomendado con pip-compile

pip-compile (parte de pip-tools) genera un requirements.txt con versiones exactas y todas las subdependencias. Esto garantiza reproducibilidad total.

```bash
# Instalar pip-tools
pip install pip-tools

# Crear requirements.in (lo que TU necesitas)
# requirements.in
torch
transformers
pandas
scikit-learn

# Generar requirements.txt con versiones pinneadas
pip-compile requirements.in -o requirements.txt

# Instalar desde el lock file
pip-sync requirements.txt

# Actualizar dependencias
pip-compile --upgrade requirements.in
```

### Separar dependencias por entorno

```
requirements/
├── base.txt          # Dependencias core
├── dev.txt           # Testing, linting (-r base.txt)
├── gpu.txt           # PyTorch con CUDA (-r base.txt)
└── notebook.txt      # Jupyter + visualizacion (-r base.txt)
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

### Instalacion y configuracion

```bash
pip install jupyterlab

# Iniciar
jupyter lab --port 8888

# Con un entorno virtual especifico
python -m ipykernel install --user --name mi-proyecto --display-name "Mi Proyecto"
```

### Extensiones utiles

| Extension | Proposito | Instalacion |
|---|---|---|
| jupyterlab-git | Git integrado | `pip install jupyterlab-git` |
| jupyterlab-code-formatter | Auto-format con black | `pip install jupyterlab-code-formatter` |
| jupyterlab-lsp | Autocompletado inteligente | `pip install jupyterlab-lsp python-lsp-server` |
| ipywidgets | Widgets interactivos | `pip install ipywidgets` |
| nbstripout | Limpiar outputs antes de commit | `pip install nbstripout && nbstripout --install` |

### Configuracion recomendada

```python
# Al inicio de cada notebook
%load_ext autoreload
%autoreload 2  # Recarga modulos automaticamente al cambiar

# Para graficas inline
%matplotlib inline

# Para ver todas las columnas de un DataFrame
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 50)
```

> **Punto clave:** Usa notebooks para exploracion y experimentacion. Mueve codigo estable a modulos `.py` en `src/`.

---

## GPU Setup Local

### NVIDIA (Linux/Windows)

Las GPUs NVIDIA son el estandar en AI/ML. Necesitas tres componentes:

```
Driver NVIDIA  -->  CUDA Toolkit  -->  cuDNN  -->  PyTorch con CUDA
```

#### Paso 1: Instalar driver NVIDIA

```bash
# Ubuntu
sudo apt update
sudo apt install nvidia-driver-545

# Verificar
nvidia-smi
# Deberia mostrar tu GPU, driver version, y CUDA version soportada
```

#### Paso 2: Instalar CUDA Toolkit

```bash
# Opcion A: Instalacion del sistema (para Docker, C++ custom kernels)
# Descargar de https://developer.nvidia.com/cuda-downloads

# Opcion B: Via conda (mas simple, aislado)
conda install cuda-toolkit -c nvidia

# Opcion C: PyTorch ya incluye su propia CUDA (la mas simple)
# Solo necesitas el driver NVIDIA instalado
```

#### Paso 3: Instalar PyTorch con CUDA

```bash
# PyTorch con CUDA 12.1 (verificar version actual en pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Verificacion

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Test rapido
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = x @ y  # Multiplicacion de matrices en GPU
print(f"Resultado en: {z.device}")
```

### Apple Silicon (M1/M2/M3/M4)

Apple Silicon usa el backend MPS (Metal Performance Shaders) en PyTorch. No es tan rapido como NVIDIA CUDA pero es mucho mejor que CPU.

```bash
# Instalar PyTorch (viene con soporte MPS por defecto)
pip install torch torchvision torchaudio
```

```python
import torch

# Verificar MPS
print(f"MPS disponible: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Usar MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
z = x @ y
print(f"Resultado en: {z.device}")
```

#### Patron universal para seleccionar device

```python
def get_device():
    """Selecciona el mejor device disponible."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Usando: {device}")
```

> **Punto clave para Apple Silicon:**
> - Funciona bien para modelos medianos y fine-tuning
> - Algunos operadores no estan soportados en MPS (fallback a CPU automatico)
> - La memoria unificada es una ventaja: la GPU puede usar toda la RAM
> - No esperes rendimiento de una A100, pero es suficiente para desarrollo

---

## GPU en la Nube

### Cuando usar GPU en la nube

- Tu GPU local no tiene suficiente VRAM (modelos grandes)
- Necesitas multiples GPUs
- Training que tarda muchas horas (dejarlo corriendo)
- No tienes GPU NVIDIA (solo Apple Silicon)

### Tabla comparativa de proveedores

| Proveedor | GPU | Coste/hora (USD) | VRAM | Facilidad | Mejor para |
|---|---|---|---|---|---|
| **Google Colab** (gratis) | T4 | Gratis (limitado) | 15 GB | Muy facil | Experimentar, aprender |
| **Google Colab Pro** | A100, V100 | ~10/mes | 40-80 GB | Muy facil | Proyectos medianos |
| **AWS SageMaker** | Varias | 1.50-40+ | 16-80 GB | Media | Produccion, equipos |
| **Lambda Labs** | A100, H100 | 1.10-3.50 | 40-80 GB | Facil | Training intensivo |
| **Vast.ai** | Varias | 0.20-2.00 | 8-80 GB | Media | Presupuesto ajustado |
| **RunPod** | Varias | 0.40-3.50 | 16-80 GB | Facil | Balance coste/facilidad |
| **Modal** | A100, H100 | 0.80-4.00 | 40-80 GB | Facil (serverless) | Jobs puntuales |

### Google Colab: conectar a tu repositorio

```python
# En una celda de Colab
from google.colab import drive
drive.mount('/content/drive')

# Clonar tu repo
!git clone https://github.com/tu-usuario/tu-proyecto.git
%cd tu-proyecto

# Instalar dependencias
!pip install -r requirements.txt

# Verificar GPU
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### AWS SageMaker (setup rapido)

```python
# Desde un notebook de SageMaker
import sagemaker
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    source_dir='src/',
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.g5.xlarge',  # GPU A10G, ~1.50/hora
    framework_version='2.2.0',
    py_version='py311',
    hyperparameters={
        'epochs': 10,
        'batch-size': 32,
        'lr': 1e-4,
    }
)

estimator.fit({'training': 's3://mi-bucket/datos/'})
```

### Lambda Labs / RunPod (acceso SSH)

```bash
# Conectar por SSH
ssh ubuntu@<ip-del-servidor>

# Clonar repo y empezar a trabajar
git clone https://github.com/tu-usuario/tu-proyecto.git
cd tu-proyecto
pip install -r requirements.txt
python train.py
```

> **Punto clave:** Empieza con Colab gratis. Cuando necesites mas, usa RunPod o Vast.ai para balance coste/facilidad. SageMaker para produccion empresarial.

---

## Docker para AI

### Por que Docker para ML

- **Reproducibilidad:** El mismo entorno en tu maquina, en CI/CD, y en produccion
- **GPU:** NVIDIA Container Toolkit permite usar GPUs dentro de Docker
- **Dependencias:** CUDA, cuDNN, y librerias del sistema empaquetadas

### Dockerfile base para proyectos ML

```dockerfile
# Dockerfile
# Base image con CUDA + Python
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Instalar Python y herramientas del sistema
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear alias
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Directorio de trabajo
WORKDIR /app

# Copiar e instalar dependencias primero (cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar codigo fuente
COPY src/ ./src/
COPY configs/ ./configs/

# Puerto para API o Jupyter
EXPOSE 8000

# Comando por defecto
CMD ["python", "src/train.py"]
```

### Docker Compose para desarrollo

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-dev:
    build: .
    volumes:
      - ./src:/app/src          # Hot reload del codigo
      - ./data:/app/data        # Datos locales
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
# Instalar (Ubuntu)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verificar GPU en Docker
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Dockerfile para CPU (Apple Silicon / sin GPU)

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

## Estructura de Proyecto ML

```
project/
├── data/               # Datos - NO commitear (usar .gitignore)
│   ├── raw/            # Datos originales, inmutables
│   ├── processed/      # Datos limpios, listos para modelo
│   └── external/       # Datos de fuentes externas
├── notebooks/          # Jupyter notebooks para exploracion
│   ├── 01-eda.ipynb
│   ├── 02-feature-eng.ipynb
│   └── 03-modeling.ipynb
├── src/                # Codigo de produccion
│   ├── __init__.py
│   ├── data/           # Carga y procesamiento de datos
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/         # Definiciones de modelos
│   │   ├── __init__.py
│   │   └── model.py
│   ├── training/       # Logica de entrenamiento
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── evaluation/     # Metricas y evaluacion
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── utils.py
├── models/             # Modelos entrenados - NO commitear
│   └── .gitkeep
├── configs/            # Configuracion e hiperparametros
│   ├── train_config.yaml
│   └── model_config.yaml
├── tests/              # Tests
│   ├── test_data.py
│   ├── test_model.py
│   └── test_training.py
├── logs/               # Logs de entrenamiento
├── scripts/            # Scripts de utilidad
│   ├── train.sh
│   └── evaluate.sh
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── Makefile            # Comandos frecuentes
├── pyproject.toml      # Configuracion del proyecto
├── requirements.txt    # Dependencias pinneadas
└── README.md
```

### .gitignore para proyectos ML

```gitignore
# Datos
data/
*.csv
*.parquet
*.h5
*.hdf5

# Modelos
models/
*.pt
*.pth
*.onnx
*.bin
*.safetensors

# Entornos
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

### Makefile para comandos frecuentes

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

## IDEs y Herramientas

### VSCode: extensiones esenciales

| Extension | Proposito |
|---|---|
| **Python** (Microsoft) | Soporte basico de Python |
| **Pylance** | IntelliSense avanzado, type checking |
| **Jupyter** | Notebooks dentro de VSCode |
| **GitHub Copilot** | AI autocompletado (muy util para boilerplate) |
| **Ruff** | Linting ultrarapido |
| **Remote - SSH** | Desarrollar en servidores remotos |
| **Dev Containers** | Desarrollar dentro de Docker |
| **GitLens** | Historial de Git avanzado |
| **YAML** | Para configs de hiperparametros |

### Settings recomendados (VSCode)

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

## Tips de Productividad

### tmux / screen para training largos

Cuando entrenas en un servidor remoto, si se corta la conexion SSH, pierdes el proceso. tmux lo evita.

```bash
# Instalar tmux
sudo apt install tmux

# Crear sesion
tmux new -s training

# Dentro de tmux: ejecutar tu training
python train.py --epochs 100

# Desconectar de tmux (sin matar el proceso)
# Ctrl+B, luego D

# Reconectar a la sesion
tmux attach -t training

# Listar sesiones
tmux ls

# Comandos utiles dentro de tmux:
# Ctrl+B, %    -> Split vertical
# Ctrl+B, "    -> Split horizontal
# Ctrl+B, arrow -> Mover entre paneles
```

### Monitorizar GPU

```bash
# Ver uso de GPU en tiempo real (se actualiza cada 1 segundo)
watch -n 1 nvidia-smi

# Alternativa mas bonita
pip install gpustat
gpustat --watch

# Monitorizar desde Python
import torch
print(f"Memoria asignada: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Memoria reservada: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### Otros tips

```bash
# Ejecutar en background con nohup (alternativa simple a tmux)
nohup python train.py > training.log 2>&1 &

# Ver el log en tiempo real
tail -f training.log

# Notificacion cuando termina el training (Mac)
python train.py && osascript -e 'display notification "Training terminado" with title "ML"'

# Notificacion cuando termina el training (Linux)
python train.py && notify-send "Training terminado"

# Medir tiempo de ejecucion
time python train.py
```

### Checklist antes de empezar un proyecto

- [ ] Crear repositorio Git
- [ ] Configurar .gitignore para ML
- [ ] Crear entorno virtual
- [ ] Instalar dependencias base
- [ ] Verificar GPU (si aplica)
- [ ] Crear estructura de carpetas
- [ ] Configurar IDE
- [ ] Primer notebook de EDA

---

> **Resumen:** El entorno de desarrollo es la base de todo. Invierte tiempo en configurarlo bien al principio y te ahorrara horas de dolores de cabeza despues. Usa venv para entornos, pip-compile para dependencias, y Docker cuando necesites reproducibilidad total.
