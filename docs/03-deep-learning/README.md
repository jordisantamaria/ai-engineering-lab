# Deep Learning

## Tabla de Contenidos

- [Por que Deep Learning](#por-que-deep-learning)
- [Neurona Artificial](#neurona-artificial)
- [Funciones de Activacion](#funciones-de-activacion)
- [Arquitectura de una Red Neuronal](#arquitectura-de-una-red-neuronal)
- [Backpropagation](#backpropagation)
- [Loss Functions](#loss-functions)
- [Optimizers](#optimizers)
- [Regularizacion en Deep Learning](#regularizacion-en-deep-learning)
- [PyTorch Fundamentals](#pytorch-fundamentals)
- [Hiperparametros Clave](#hiperparametros-clave)
- [Transfer Learning](#transfer-learning)
- [Mixed Precision Training](#mixed-precision-training)
- [Tips Practicos](#tips-practicos)

---

## Por que Deep Learning

### Cuando usar DL vs ML clasico

| Criterio | ML Clasico (XGBoost, RF) | Deep Learning |
|---|---|---|
| **Datos tabulares** | Casi siempre mejor | Raramente justificado |
| **Imagenes** | No competitivo | El estandar |
| **Texto/NLP** | TF-IDF + LR para cosas simples | El estandar (Transformers) |
| **Audio** | Features manuales + ML | El estandar |
| **< 10K muestras** | Generalmente mejor | Riesgo de overfitting |
| **> 100K muestras** | Funciona bien | Empieza a brillar |
| **Necesitas explicar** | Facil (feature importance) | Dificil (caja negra) |
| **Tiempo de desarrollo** | Rapido (horas-dias) | Lento (dias-semanas) |
| **Infraestructura** | CPU suficiente | GPU necesaria |
| **Produccion** | Ligero, rapido | Pesado, mas latencia |

### Tabla de decision rapida

```
Tengo datos tabulares?
  ├─ Si --> Usa XGBoost/LightGBM (DL raramente gana en tabular)
  └─ No --> Que tipo de datos?
              ├─ Imagenes     --> CNN o Vision Transformer
              ├─ Texto        --> Transformer (BERT, GPT, etc.)
              ├─ Audio        --> Transformer (Whisper, etc.)
              ├─ Video        --> CNN 3D o Video Transformer
              ├─ Grafos       --> GNN
              └─ Multimodal   --> Modelos multimodales (CLIP, etc.)

Tengo pocos datos (<1K)?
  ├─ Si --> Transfer learning (modelo preentrenado + fine-tune)
  └─ No --> Entrenar desde cero es una opcion
```

---

## Neurona Artificial

### Analogia intuitiva

Una neurona artificial funciona como una funcion de decision simple:

```
Entradas (features)     Pesos (importancia)       Suma ponderada        Activacion     Salida
     x1 ──── w1 ────┐
                      ├──── z = w1*x1 + w2*x2 + w3*x3 + b ──── f(z) ──── output
     x2 ──── w2 ────┤                       ↑
                      │                    bias (sesgo)
     x3 ──── w3 ────┘

Ejemplo: predecir si aprobaras un examen
  x1 = horas de estudio    w1 = 0.6  (muy importante)
  x2 = horas de sueno      w2 = 0.3  (importante)
  x3 = cafe bebido          w3 = 0.1  (poco importante)
  b  = -5.0                           (umbral base)

  z = 0.6*8 + 0.3*7 + 0.1*3 + (-5.0) = 4.8 + 2.1 + 0.3 - 5.0 = 2.2
  output = sigmoid(2.2) = 0.90  --> 90% probabilidad de aprobar
```

**Lo que el modelo "aprende"** son los pesos (w1, w2, w3) y el bias (b). El entrenamiento ajusta estos valores para minimizar el error.

---

## Funciones de Activacion

Las funciones de activacion introducen no-linealidad. Sin ellas, una red neuronal de muchas capas seria equivalente a una sola capa lineal (multiplicar matrices da otra matriz).

### Comparativa

| Funcion | Formula intuitiva | Rango | Uso principal | Ventaja |
|---|---|---|---|---|
| **ReLU** | max(0, x) | [0, inf) | Hidden layers (default) | Simple, rapida, evita vanishing gradient |
| **Sigmoid** | 1/(1+e^-x) | (0, 1) | Output binaria | Salida interpretable como probabilidad |
| **Tanh** | (e^x - e^-x)/(e^x + e^-x) | (-1, 1) | Hidden layers (alternativa a ReLU) | Centrada en 0 |
| **GELU** | x * P(X <= x) ~= x * sigmoid(1.7*x) | (-0.17, inf) | Transformers | Suave, mejor en NLP |
| **Softmax** | e^xi / sum(e^xj) | (0, 1), suma=1 | Output multiclase | Distribucion de probabilidad |
| **Leaky ReLU** | max(0.01*x, x) | (-inf, inf) | Evitar "dying ReLU" | No mata neuronas |

### Graficas ASCII

```
ReLU: max(0, x)              Sigmoid: 1/(1+e^-x)          Tanh
      │                            │                            │
  y   │      /                 1.0 ┤·········──────         1.0 ┤·········──────
      │     /                      │       /                    │      /
      │    /                  0.5  ┤──── /                 0.0  ┤────/
      │   /                        │   /                        │  /
      │  /                    0.0  ┤──                    -1.0  ┤──
  ────┼──────── x             ─────┼──────── x             ─────┼──────── x
      │                            │                            │
  "Si es negativo, muere"    "Comprime todo               "Como sigmoid pero
   "Si es positivo, pasa"     entre 0 y 1"                 centrada en 0"

GELU                          Leaky ReLU                   Softmax
      │                            │                       No es una curva, es
  y   │      /                 y   │      /                una funcion sobre un
      │    _/                      │    /                   vector que convierte
      │  _/                        │  /                     numeros en probabilidades:
      │_/                      ────┼/─────── x
  ────┼──────── x                 /│                       [2.0, 1.0, 0.1]
     ~│                          / │                            ↓ softmax
  "ReLU suave"               "ReLU con pendiente           [0.659, 0.243, 0.098]
                               para negativos"              Suma = 1.0
```

### Cuando usar cada una

```python
import torch.nn as nn

# Hidden layers: ReLU (default seguro)
nn.ReLU()

# Hidden layers en Transformers: GELU
nn.GELU()

# Output para clasificacion binaria: Sigmoid
nn.Sigmoid()  # Salida entre 0 y 1

# Output para clasificacion multiclase: Softmax
nn.Softmax(dim=-1)  # Probabilidades que suman 1

# Si tienes problemas de "dying ReLU": Leaky ReLU
nn.LeakyReLU(negative_slope=0.01)
```

---

## Arquitectura de una Red Neuronal

### Componentes

```
Input Layer          Hidden Layers            Output Layer
(features)           (aprendizaje)            (prediccion)

   x1 ─────┐     ┌── h1 ──┐     ┌── h4 ──┐     ┌── y1
            ├─────┤         ├─────┤         ├─────┤
   x2 ─────┤     ├── h2 ──┤     ├── h5 ──┤     └── y2
            ├─────┤         ├─────┤         │
   x3 ─────┤     └── h3 ──┘     └── h6 ──┘
            │
   x4 ─────┘

  4 features     3 neuronas       3 neuronas       2 clases
                 + ReLU           + ReLU           + Softmax

  Cada flecha (─) es un peso (weight) que se aprende.
  Total de pesos: (4*3) + (3*3) + (3*2) = 12 + 9 + 6 = 27 pesos
  Total de biases: 3 + 3 + 2 = 8
  Total de parametros: 35
```

### Forward pass (propagacion hacia adelante)

Paso a paso, como fluyen los datos por la red:

```
Datos de entrada: x = [0.5, 0.3, 0.8, 0.1]

Capa 1 (input -> hidden 1):
  z1 = W1 @ x + b1           # Multiplicacion de matrices + bias
  # z1 = [0.72, -0.15, 1.23]
  h1 = ReLU(z1)              # Activacion
  # h1 = [0.72, 0.00, 1.23]  # Nota: -0.15 se convierte en 0

Capa 2 (hidden 1 -> hidden 2):
  z2 = W2 @ h1 + b2
  h2 = ReLU(z2)

Capa 3 (hidden 2 -> output):
  z3 = W3 @ h2 + b3
  output = Softmax(z3)
  # output = [0.85, 0.15]    # 85% clase A, 15% clase B
```

```python
import torch
import torch.nn as nn

# Definir la red del diagrama
class RedSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.capa1 = nn.Linear(4, 3)    # 4 inputs -> 3 neuronas
        self.capa2 = nn.Linear(3, 3)    # 3 -> 3 neuronas
        self.capa3 = nn.Linear(3, 2)    # 3 -> 2 outputs (clases)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.capa1(x))    # Capa 1 + ReLU
        x = self.relu(self.capa2(x))    # Capa 2 + ReLU
        x = self.capa3(x)               # Capa 3 (sin activacion - la loss la aplica)
        return x

modelo = RedSimple()
print(f"Parametros totales: {sum(p.numel() for p in modelo.parameters()):,}")
# 35 parametros
```

---

## Backpropagation

### Intuicion (sin derivaciones formales)

Backpropagation es el algoritmo que "ensena" a la red. Funciona en dos fases:

```
FORWARD PASS (izquierda a derecha):
  Input → Capa 1 → Capa 2 → Output → Loss
  "Hacemos una prediccion y calculamos cuanto nos equivocamos"

BACKWARD PASS (derecha a izquierda):
  Loss → Capa 2 → Capa 1
  "Propagamos el error hacia atras, calculando cuanto
   contribuyo cada peso al error"

ACTUALIZAR PESOS:
  peso_nuevo = peso_viejo - learning_rate * gradiente
  "Ajustamos cada peso en la direccion que reduce el error"
```

### Chain rule: la idea central

La chain rule dice: si un cambio en un peso afecta a la capa siguiente, que afecta a la siguiente, que afecta al loss, podemos calcular cuanto afecta ese peso al loss multiplicando los efectos en cadena.

```
Analogia: una cadena de dominos

  Peso w  -->  Neurona h  -->  Output o  -->  Loss L

  ¿Cuanto afecta w al Loss?
  dL/dw = dL/do * do/dh * dh/dw

  Es como preguntar: "si muevo este domino 1mm,
  cuanto se mueve el ultimo domino?"
  = (efecto del ultimo) * (efecto del medio) * (efecto del primero)
```

### Vanishing y Exploding Gradients

```
VANISHING GRADIENTS (gradientes que desaparecen):

  Capas:    [1] ── [2] ── [3] ── [4] ── [5] ── [Loss]

  Gradientes: 0.001  0.01   0.1    0.5    1.0
              ←────────────────────────────────
              Cada capa multiplica por un numero < 1
              Las primeras capas apenas aprenden

EXPLODING GRADIENTS (gradientes que explotan):

  Gradientes: 1000   100    10     2      1.0
              ←────────────────────────────────
              Cada capa multiplica por un numero > 1
              Los pesos saltan descontroladamente
```

#### Soluciones

| Problema | Solucion | Como |
|---|---|---|
| **Vanishing** | ReLU en vez de sigmoid/tanh | `nn.ReLU()` |
| **Vanishing** | Residual connections (skip connections) | ResNet, Transformers |
| **Vanishing** | Batch Normalization | `nn.BatchNorm1d()` |
| **Vanishing** | Better initialization | `nn.init.kaiming_normal_()` |
| **Exploding** | Gradient clipping | `torch.nn.utils.clip_grad_norm_()` |
| **Exploding** | Learning rate mas bajo | `lr=1e-4` en vez de `lr=1e-2` |
| **Ambos** | Layer Normalization | `nn.LayerNorm()` (usado en Transformers) |

```python
# Gradient clipping (previene exploding gradients)
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## Loss Functions

La loss function mide cuanto se equivoca el modelo. El training consiste en minimizar esta funcion.

### Cuando usar cada una

| Loss | Tipo de problema | Que mide | PyTorch |
|---|---|---|---|
| **CrossEntropyLoss** | Clasificacion multiclase | Diferencia entre distribucion predicha y real | `nn.CrossEntropyLoss()` |
| **BCEWithLogitsLoss** | Clasificacion binaria | Diferencia binaria (con estabilidad numerica) | `nn.BCEWithLogitsLoss()` |
| **MSELoss** | Regresion | Error cuadratico medio | `nn.MSELoss()` |
| **L1Loss (MAE)** | Regresion robusta a outliers | Error absoluto medio | `nn.L1Loss()` |
| **HuberLoss** | Regresion (mix MSE + MAE) | MSE cerca de 0, MAE lejos | `nn.HuberLoss()` |

```python
import torch.nn as nn

# Clasificacion multiclase (la mas comun)
# NOTA: CrossEntropyLoss ya incluye Softmax internamente
# NO pongas Softmax en la ultima capa si usas esta loss
loss_fn = nn.CrossEntropyLoss()

logits = model(x)       # Shape: (batch_size, num_clases) - SIN softmax
target = labels          # Shape: (batch_size,) - indices de clase (0, 1, 2...)
loss = loss_fn(logits, target)

# Clasificacion binaria
loss_fn = nn.BCEWithLogitsLoss()  # Incluye sigmoid internamente
logits = model(x)       # Shape: (batch_size, 1) - SIN sigmoid
target = labels.float() # Shape: (batch_size, 1) - 0.0 o 1.0
loss = loss_fn(logits, target)

# Regresion
loss_fn = nn.MSELoss()
predictions = model(x)  # Shape: (batch_size, 1)
target = values          # Shape: (batch_size, 1)
loss = loss_fn(predictions, target)
```

### Clases desbalanceadas

```python
# Opcion 1: Pesos por clase en la loss
# Si tienes 90% clase 0 y 10% clase 1
weights = torch.tensor([1.0, 9.0])  # Dar mas peso a la clase minoritaria
loss_fn = nn.CrossEntropyLoss(weight=weights)

# Opcion 2: Calcular pesos automaticamente
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights = torch.tensor(class_weights, dtype=torch.float32)
loss_fn = nn.CrossEntropyLoss(weight=weights)
```

---

## Optimizers

El optimizer es el algoritmo que actualiza los pesos usando los gradientes calculados por backpropagation.

### Intuicion

```
Imagina que estas en una montana con niebla y quieres llegar al valle (minimo de la loss):

SGD:         Caminas cuesta abajo. Simple pero lento.
             En pendientes suaves, apenas avanzas.

SGD+Momentum: Caminas cuesta abajo pero con inercia (como una bola rodando).
             Atraviesas zonas planas sin detenerte.

Adam:        Caminas cuesta abajo con inercia Y ajustas
             el tamano del paso automaticamente.
             Pasos grandes en zonas planas, pequeños en zonas empinadas.
             ES EL DEFAULT. Usa Adam si no sabes que elegir.
```

### Tabla comparativa

| Optimizer | Descripcion | Cuando usarlo | Learning rate tipico |
|---|---|---|---|
| **SGD** | El mas basico | Casi nunca solo | 0.01 - 0.1 |
| **SGD + Momentum** | SGD con inercia | CNNs, cuando necesitas convergencia fina | 0.01 - 0.1 |
| **Adam** | Momentum + LR adaptativo | **Default para todo** | 1e-3 - 1e-4 |
| **AdamW** | Adam + weight decay correcto | **Default para Transformers, fine-tuning** | 1e-4 - 1e-5 |
| **Adagrad** | LR adaptativo por parametro | Sparse features (NLP antiguo) | 0.01 |
| **RMSprop** | Version mejorada de Adagrad | RNNs | 1e-3 |

```python
import torch.optim as optim

# Adam (default recomendado)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# AdamW (para Transformers y fine-tuning)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# SGD con momentum (para CNNs en entrenamiento largo)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
```

### Learning Rate

El learning rate es el hiperparametro MAS IMPORTANTE. Controla el tamano del paso en cada actualizacion.

```
LR demasiado alto:           LR demasiado bajo:          LR correcto:
      Loss                        Loss                       Loss
      │\  /\                      │\                         │\
      │ \/  \  /\                 │ \                        │ \
      │      \/  \  /             │  \                       │  \
      │           \/              │   \                      │   \___
      │                           │    \___________          │
      └──────────── epoch         └──────────── epoch        └──────────── epoch
  "Salta demasiado,            "Converge pero              "Converge rapido
   nunca converge"              tardara 10x mas"             y bien"
```

### Learning Rate Schedulers

Cambiar el learning rate durante el entrenamiento. Generalmente: empezar alto (explorar) y bajar gradualmente (refinar).

```python
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, OneCycleLR, LinearLR, SequentialLR
)

# Step decay: reducir LR cada N epochs
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# LR: 0.01 -> (epoch 30) -> 0.001 -> (epoch 60) -> 0.0001

# Cosine annealing: reducir LR siguiendo una curva coseno
scheduler = CosineAnnealingLR(optimizer, T_max=100)
# LR baja suavemente de max a min siguiendo coseno

# OneCycleLR: warmup + decay (muy popular, suele dar buenos resultados)
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=num_epochs * len(train_loader),
    pct_start=0.1,  # 10% warmup
)

# Warmup lineal + cosine decay (tipico en Transformers)
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

# En el training loop:
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = train_step(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()  # Actualizar LR
```

### Warmup: por que empezar con LR bajo

```
Sin warmup:                      Con warmup:
  LR                               LR
  │────────\                       │    /──────\
  │         \                      │   /        \
  │          \                     │  /          \
  │           \___                 │ /            \___
  └──────────── step               └──────────── step

Al inicio los pesos son random.         Primero calentamos con LR bajo
Un LR alto puede hacer que el          (los pesos se estabilizan), luego
modelo diverja inmediatamente.         subimos el LR para entrenar rapido.
```

---

## Regularizacion en Deep Learning

### Dropout

Dropout apaga neuronas aleatorias durante el entrenamiento. Esto obliga a la red a no depender de ninguna neurona individual y aprender representaciones mas robustas.

```
Training (dropout=0.3):           Inference (sin dropout):

  [x1]──[h1]──[h4]──[y]           [x1]──[h1]──[h4]──[y]
  [x2]──[XX]──[h5]──               [x2]──[h2]──[h5]──
  [x3]──[h3]──[XX]──               [x3]──[h3]──[h6]──

  XX = neurona apagada              Todas activas, pesos
  (30% apagadas al azar)           escalados por (1-p)
  Diferente en cada batch!
```

```python
class MiModelo(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(0.3)   # 30% de neuronas apagadas
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)              # Dropout DESPUES de activacion
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)                    # NO dropout en la ultima capa
        return x
```

> **Punto clave:** Dropout se usa entre capas ocultas, nunca en la capa de salida. Valores tipicos: 0.1-0.5. En Transformers se usa 0.1 tipicamente.

### Batch Normalization

Normaliza las activaciones de cada capa para que tengan media 0 y varianza 1. Esto estabiliza el entrenamiento y permite usar learning rates mas altos.

```
Sin BatchNorm:                     Con BatchNorm:
Las activaciones de cada capa      Se normalizan antes de cada capa
pueden tener rangos muy distintos   -> entrenamiento mas estable

Capa 1 output: [100, -50, 200]     Capa 1 output: [0.2, -1.1, 1.3]
Capa 2 output: [0.01, 0.003]       Capa 2 output: [-0.5, 0.8]
```

```python
class MiModelo(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)    # BatchNorm despues de linear
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))  # Linear -> BN -> ReLU
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
```

### Weight Decay (regularizacion L2)

Penaliza pesos grandes. En AdamW esta integrado directamente.

```python
# Weight decay en el optimizer (la forma correcta con AdamW)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

### Data Augmentation

Crear variaciones de los datos de entrenamiento. Muy efectivo en imagenes.

```python
from torchvision import transforms

# Augmentation para imagenes
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Sin augmentation para validacion/test
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

### Early Stopping

Detener el entrenamiento cuando la loss de validacion deja de mejorar.

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

# Uso
early_stop = EarlyStopping(patience=10)

for epoch in range(max_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()

    early_stop(val_loss)
    if early_stop.should_stop:
        print(f"Early stopping en epoch {epoch}")
        break
```

### Resumen de regularizacion

| Tecnica | Efecto | Donde | Valor tipico |
|---|---|---|---|
| **Dropout** | Apaga neuronas | Entre hidden layers | 0.1-0.5 |
| **Batch Norm** | Normaliza activaciones | Despues de linear/conv | - |
| **Weight Decay** | Penaliza pesos grandes | En optimizer | 0.01-0.1 |
| **Data Augmentation** | Mas datos sinteticos | En DataLoader | Varias |
| **Early Stopping** | Para entrenamiento | En training loop | patience=5-20 |

---

## PyTorch Fundamentals

### Tensors

Los tensors de PyTorch son como arrays de NumPy pero con dos superpoderes: pueden vivir en GPU y registran operaciones para calcular gradientes automaticamente.

```python
import torch

# Crear tensors (similar a NumPy)
x = torch.tensor([1.0, 2.0, 3.0])                 # Desde lista
x = torch.zeros(3, 4)                               # Tensor de ceros
x = torch.ones(3, 4)                                # Tensor de unos
x = torch.randn(3, 4)                               # Normal(0, 1)
x = torch.arange(0, 10, 2)                          # [0, 2, 4, 6, 8]
x = torch.linspace(0, 1, 5)                         # [0, 0.25, 0.5, 0.75, 1.0]

# Propiedades
print(x.shape)     # torch.Size([3, 4])
print(x.dtype)     # torch.float32
print(x.device)    # cpu o cuda:0

# Conversion NumPy <-> PyTorch (comparten memoria si en CPU)
numpy_array = x.numpy()
tensor = torch.from_numpy(numpy_array)

# Operaciones (igual que NumPy)
a = torch.randn(3, 4)
b = torch.randn(3, 4)
c = a + b                    # Suma
d = a @ b.T                  # Multiplicacion de matrices
e = a * b                    # Multiplicacion elemento a elemento
f = a.mean(), a.std()        # Estadisticas
```

### GPU support

```python
# Mover a GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.randn(1000, 1000, device=device)    # Crear directamente en GPU
y = torch.randn(1000, 1000).to(device)         # Mover a GPU

# IMPORTANTE: todos los tensors deben estar en el mismo device
z = x @ y  # OK: ambos en GPU
# z = x @ torch.randn(1000, 1000)  # ERROR: x en GPU, tensor nuevo en CPU
```

### Autograd (diferenciacion automatica)

```python
# PyTorch registra operaciones para calcular gradientes automaticamente
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Forward pass
y = x ** 2 + 3 * x      # y = x^2 + 3x
loss = y.sum()           # Escalar necesario para backward

# Backward pass (calcula gradientes)
loss.backward()

# Gradientes: dy/dx = 2x + 3
print(x.grad)            # tensor([7., 9.])  ->  2*2+3=7, 2*3+3=9
```

### nn.Module: crear modelos

```python
import torch.nn as nn

class ClasificadorImagenes(nn.Module):
    def __init__(self, num_clases=10):
        super().__init__()

        # Capas convolucionales
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),               # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),               # 112 -> 56
        )

        # Capas fully connected
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_clases),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Instanciar
model = ClasificadorImagenes(num_clases=10)
print(f"Parametros: {sum(p.numel() for p in model.parameters()):,}")
```

### Dataset y DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class MiDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# Crear datasets
train_dataset = MiDataset(X_train, y_train)
val_dataset = MiDataset(X_val, y_val)

# DataLoaders (manejan batching, shuffling, paralelismo)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,         # Shuffle en train
    num_workers=4,        # Cargar datos en paralelo
    pin_memory=True,      # Mas rapido para GPU
    drop_last=True,       # Descartar ultimo batch incompleto
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,        # Puede ser mas grande en val (no hay backward)
    shuffle=False,        # NO shuffle en val
    num_workers=4,
    pin_memory=True,
)
```

### Training loop completo

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    """Entrena el modelo por una epoch."""
    model.train()  # Modo entrenamiento (activa dropout, batchnorm)
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Mover datos a device (GPU/CPU)
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Backward pass
        optimizer.zero_grad()     # Limpiar gradientes anteriores
        loss.backward()           # Calcular gradientes
        optimizer.step()          # Actualizar pesos

        # Metricas
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()  # No calcular gradientes (mas rapido, menos memoria)
def validate(model, val_loader, loss_fn, device):
    """Evalua el modelo en el set de validacion."""
    model.eval()  # Modo evaluacion (desactiva dropout, batchnorm usa stats globales)
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


# === SETUP ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ClasificadorImagenes(num_clases=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# === TRAINING LOOP ===
best_val_loss = float('inf')
early_stop = EarlyStopping(patience=10)

for epoch in range(50):
    train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, loss_fn, device)
    scheduler.step()

    # Log
    print(f"Epoch {epoch+1:3d} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # Guardar mejor modelo
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/best_model.pt')
        print(f"  -> Mejor modelo guardado (val_loss: {val_loss:.4f})")

    # Early stopping
    early_stop(val_loss)
    if early_stop.should_stop:
        print(f"Early stopping en epoch {epoch+1}")
        break

print(f"\nMejor val_loss: {best_val_loss:.4f}")
```

### Guardar y cargar modelos

```python
# GUARDAR: solo el state_dict (pesos) - RECOMENDADO
torch.save(model.state_dict(), 'models/modelo_v1.pt')

# CARGAR:
model = ClasificadorImagenes(num_clases=10)  # Crear la arquitectura
model.load_state_dict(torch.load('models/modelo_v1.pt', map_location=device))
model.to(device)
model.eval()  # Modo evaluacion para inferencia

# Guardar checkpoint completo (para resumir entrenamiento)
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss': best_val_loss,
    'train_loss': train_loss,
}
torch.save(checkpoint, 'models/checkpoint_epoch10.pt')

# Cargar checkpoint
checkpoint = torch.load('models/checkpoint_epoch10.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Device management

```python
# Patron universal
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

device = get_device()
print(f"Usando: {device}")

# Mover modelo a device
model = model.to(device)

# Mover datos a device (en el training loop)
inputs = inputs.to(device)
targets = targets.to(device)

# Mover resultado a CPU para NumPy/Pandas
predictions = model(inputs).cpu().numpy()

# Multi-GPU (DataParallel - basico)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)
```

---

## Hiperparametros Clave

### Learning rate (el mas importante)

```python
# Encontrar el LR optimo con LR Finder (concepto de Leslie Smith)
from torch.optim.lr_scheduler import ExponentialLR

# Idea: empezar con LR muy bajo e ir subiendo exponencialmente.
# Graficar loss vs LR. El mejor LR esta justo antes de que la loss explote.

lrs = []
losses = []
model_copy = copy.deepcopy(model)
optimizer = optim.Adam(model_copy.parameters(), lr=1e-7)
scheduler = ExponentialLR(optimizer, gamma=1.1)

for batch in train_loader:
    loss = train_step(batch)
    lrs.append(optimizer.param_groups[0]['lr'])
    losses.append(loss.item())
    scheduler.step()
    if loss.item() > 4 * min(losses):
        break

plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('LR Finder')
plt.show()
# Elegir LR donde la loss baja mas rapido (pendiente mas pronunciada)
```

### Batch size

| Batch size | Ventaja | Desventaja |
|---|---|---|
| **Pequeño** (16-32) | Mejor generalizacion, menos memoria | Mas lento, mas ruidoso |
| **Medio** (64-128) | Buen balance | - |
| **Grande** (256-1024) | Mas rapido, GPU mas eficiente | Puede generalizar peor, necesita LR alto |

```
Regla practica:
- Usa el batch size mas grande que quepa en tu GPU
- Si subes batch size, sube LR proporcionalmente
  (batch 32, lr=1e-3) -> (batch 64, lr=2e-3)
- Para fine-tuning de LLMs: batch size 4-16 (modelos grandes)
```

### Cuantas capas y neuronas

```
Regla general:
- Empezar pequeno e ir creciendo
- Mas datos -> puedes usar modelos mas grandes
- La primera hidden layer suele ser la mas grande
- Ir reduciendo tamano: 512 -> 256 -> 128

Ejemplo tipico para datos tabulares:
  Input (20 features)
  -> Linear(20, 128) + ReLU + Dropout(0.3)
  -> Linear(128, 64) + ReLU + Dropout(0.3)
  -> Linear(64, num_clases)
```

### Hyperparameter tuning

```python
# Optuna (RECOMENDADO - el mejor framework de hyperparameter tuning)
import optuna

def objective(trial):
    # Sugerir hiperparametros
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    n_layers = trial.suggest_int('n_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])

    # Crear modelo con estos hiperparametros
    model = crear_modelo(n_layers, hidden_size, dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Entrenar y evaluar
    for epoch in range(20):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)

        # Reportar metrica intermedia (permite poda de trials malos)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_acc

# Ejecutar busqueda
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, timeout=3600)  # 50 trials o 1 hora

# Mejores hiperparametros
print(f"Mejor accuracy: {study.best_value:.4f}")
print(f"Mejores params: {study.best_params}")

# Visualizar
optuna.visualization.plot_param_importances(study)
optuna.visualization.plot_optimization_history(study)
```

| Metodo | Descripcion | Cuando usarlo |
|---|---|---|
| **Grid search** | Probar todas las combinaciones | Pocos hiperparametros (2-3) |
| **Random search** | Probar combinaciones al azar | Mejor que grid search en general |
| **Optuna (Bayesian)** | Busqueda inteligente guiada | **Siempre que puedas, es el mejor** |

---

## Transfer Learning

### Que es y por que funciona

En vez de entrenar un modelo desde cero, partimos de un modelo que ya fue entrenado en un dataset grande (como ImageNet con 1M de imagenes). Las primeras capas ya aprendieron patrones generales (bordes, texturas, formas) que son utiles para cualquier tarea visual.

```
Modelo preentrenado (ImageNet):

  [Bordes] -> [Texturas] -> [Partes] -> [Objetos] -> [1000 clases ImageNet]
   Capa 1      Capa 2       Capa 3      Capa 4        Head

Tu tarea (clasificar rayos X):

  [Bordes] -> [Texturas] -> [Partes] -> [Objetos] -> [2 clases: normal/anormal]
   Reusar     Reusar       Reusar      Reusar        NUEVA head
   (congelar) (congelar)   (congelar)  (congelar)    (entrenar)
```

### Fine-tuning vs Feature extraction

| Estrategia | Que haces | Cuando usarlo |
|---|---|---|
| **Feature extraction** | Congelas todo, solo entrenas la head nueva | Pocos datos (<1K), tarea muy similar al preentrenamiento |
| **Fine-tuning parcial** | Congelas primeras capas, entrenas las ultimas + head | Datos moderados (1K-10K) |
| **Fine-tuning completo** | Entrenas todo el modelo con LR bajo | Muchos datos (>10K), tarea distinta al preentrenamiento |

```python
import torchvision.models as models

# Cargar modelo preentrenado
model = models.resnet50(weights='IMAGENET1K_V2')

# === Estrategia 1: Feature Extraction ===
# Congelar TODOS los parametros
for param in model.parameters():
    param.requires_grad = False

# Reemplazar la ultima capa (head) para tu tarea
num_clases = 5
model.fc = nn.Linear(model.fc.in_features, num_clases)
# Solo model.fc tiene requires_grad=True

# Optimizer solo para parametros que requieren gradiente
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)


# === Estrategia 2: Fine-tuning parcial ===
model = models.resnet50(weights='IMAGENET1K_V2')

# Congelar todo
for param in model.parameters():
    param.requires_grad = False

# Descongelar las ultimas capas (layer4) + head
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, num_clases)

# LR distinto para capas preentrenadas vs head nueva
optimizer = optim.AdamW([
    {'params': model.layer4.parameters(), 'lr': 1e-5},   # LR bajo para preentrenadas
    {'params': model.fc.parameters(), 'lr': 1e-3},        # LR alto para head nueva
], weight_decay=0.01)


# === Estrategia 3: Fine-tuning completo ===
model = models.resnet50(weights='IMAGENET1K_V2')
model.fc = nn.Linear(model.fc.in_features, num_clases)

# LR bajo para todo (no destruir lo aprendido)
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
```

### Modelos preentrenados populares

| Modelo | Tipo | Parametros | Uso principal | PyTorch |
|---|---|---|---|---|
| **ResNet-50** | CNN | 25M | Imagenes (baseline) | `models.resnet50()` |
| **EfficientNet-B0** | CNN | 5M | Imagenes (eficiente) | `models.efficientnet_b0()` |
| **ViT-B/16** | Transformer | 86M | Imagenes (SOTA) | `models.vit_b_16()` |
| **BERT** | Transformer | 110M | Texto (clasificacion, NER) | HuggingFace `transformers` |
| **GPT-2** | Transformer | 124M-1.5B | Texto (generacion) | HuggingFace `transformers` |
| **Whisper** | Transformer | 39M-1.5B | Audio (transcripcion) | HuggingFace `transformers` |
| **CLIP** | Multimodal | 400M | Imagenes + Texto | OpenAI / HuggingFace |

---

## Mixed Precision Training

### Que es

Normalmente los modelos usan float32 (32 bits por numero). Mixed precision usa float16 para la mayoria de operaciones y float32 solo donde es necesario. Esto reduce la memoria a la mitad y acelera el entrenamiento en GPUs modernas.

```
Float32 (precision completa):    Float16 (media precision):
  32 bits por peso                 16 bits por peso
  Mas memoria                      Menos memoria (~50%)
  Mas lento en GPUs nuevas         Mas rapido (Tensor Cores)
  Siempre estable                  Puede ser inestable (loss scaling)
```

### Cuando usarlo

- Tienes GPU NVIDIA con Tensor Cores (V100, A100, RTX 3000+, RTX 4000+)
- Tu modelo no cabe en GPU con float32
- Quieres entrenar mas rapido sin sacrificar calidad

### Implementacion con torch.amp

```python
from torch.amp import autocast, GradScaler

# Crear scaler (previene underflow en float16)
scaler = GradScaler('cuda')

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass en mixed precision
        with autocast('cuda'):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        # Backward pass con scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # El loss para logging debe ser .item() (float32)
        print(f"Loss: {loss.item():.4f}")
```

> **Punto clave:** Mixed precision es "gratis": casi no afecta la precision del modelo, reduce memoria ~50%, y acelera entrenamiento 1.5-3x. Usalo siempre que tu GPU lo soporte.

---

## Tips Practicos

### 1. Empezar siempre con un modelo pequeno

```python
# Primero: modelo minimo para verificar que el pipeline funciona
model = nn.Sequential(
    nn.Linear(input_size, 32),
    nn.ReLU(),
    nn.Linear(32, num_clases)
)
# Entrenar 5 epochs
# Si funciona, ir aumentando complejidad
```

### 2. Overfittear un batch pequeno (sanity check)

Antes de entrenar en todo el dataset, verifica que tu modelo PUEDE aprender intentando memorizar un solo batch. Si no puede overfittear un batch, algo esta mal.

```python
# Tomar un solo batch
batch = next(iter(train_loader))
inputs, targets = batch
inputs, targets = inputs.to(device), targets.to(device)

model = MiModelo().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Intentar overfittear este unico batch
for i in range(200):
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 20 == 0:
        acc = (outputs.argmax(1) == targets).float().mean()
        print(f"Step {i}: Loss={loss.item():.4f}, Acc={acc:.4f}")

# Si la loss llega a ~0 y acc a ~1.0 -> el modelo puede aprender -> OK
# Si no baja -> bug en el codigo, modelo muy pequeno, o LR incorrecto
```

### 3. Monitorizar loss y metricas

```python
# Con Weights & Biases (wandb) - el estandar en la industria
import wandb

wandb.init(project="mi-proyecto", config={
    "lr": 1e-3,
    "batch_size": 32,
    "model": "resnet50",
    "epochs": 50,
})

for epoch in range(50):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = validate(...)

    wandb.log({
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "lr": optimizer.param_groups[0]['lr'],
        "epoch": epoch,
    })

wandb.finish()

# Con TensorBoard (alternativa gratuita)
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs/experiment_1')
for epoch in range(50):
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
writer.close()

# Lanzar TensorBoard
# tensorboard --logdir=logs/
```

### 4. Reproducibilidad

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    """Establece seed para reproducibilidad completa."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Para reproducibilidad total (puede ser mas lento)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Checklist antes de entrenar

```
Antes de lanzar un entrenamiento largo, verifica:

[ ] Sanity check: overfittear un batch pequeno
[ ] Data: verificar shapes, dtypes, rangos de valores
[ ] Labels: verificar distribucion de clases
[ ] Model: contar parametros, verificar output shape
[ ] Loss: verificar que es la correcta para el problema
[ ] Optimizer: verificar LR y weight decay
[ ] Device: modelo y datos en el mismo device
[ ] Reproducibilidad: seed establecida
[ ] Logging: wandb/tensorboard configurado
[ ] Checkpointing: guardar mejor modelo
[ ] Early stopping: configurado
[ ] Memoria GPU: verificar que cabe con nvidia-smi
```

### Debugging comun

| Problema | Posible causa | Solucion |
|---|---|---|
| Loss no baja | LR muy bajo o muy alto | Probar LR finder |
| Loss es NaN | LR demasiado alto, overflow | Bajar LR, gradient clipping |
| Loss sube | Bug en el codigo, LR alto | Revisar labels, bajar LR |
| Val loss sube (train baja) | Overfitting | Dropout, weight decay, mas datos |
| Train/val loss altos | Underfitting | Modelo mas grande, mas epochs |
| OOM (Out of Memory) | Modelo/batch muy grande | Reducir batch size, mixed precision |
| Training muy lento | CPU bottleneck en data loading | Mas num_workers, pin_memory |
| Accuracy no sube de X% | Clase desbalanceada, metrica incorrecta | Class weights, cambiar metrica |

---

> **Resumen:** Deep Learning es una herramienta poderosa pero no siempre la mejor. Usalo cuando tengas muchos datos no tabulares (imagenes, texto, audio). Empieza simple, verifica que tu pipeline funciona con un sanity check, y luego escala. Usa transfer learning siempre que puedas. Adam/AdamW con cosine LR schedule es un setup robusto para empezar.
