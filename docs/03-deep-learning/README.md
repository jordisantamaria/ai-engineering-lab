# Deep Learning

## Table of Contents

- [Why Deep Learning](#why-deep-learning)
- [Artificial Neuron](#artificial-neuron)
- [Activation Functions](#activation-functions)
- [Neural Network Architecture](#neural-network-architecture)
- [Backpropagation](#backpropagation)
- [Loss Functions](#loss-functions)
- [Optimizers](#optimizers)
- [Regularization in Deep Learning](#regularization-in-deep-learning)
- [PyTorch Fundamentals](#pytorch-fundamentals)
- [Key Hyperparameters](#key-hyperparameters)
- [Transfer Learning](#transfer-learning)
- [Mixed Precision Training](#mixed-precision-training)
- [Practical Tips](#practical-tips)

---

## Why Deep Learning

### When to use DL vs classical ML

| Criterion | Classical ML (XGBoost, RF) | Deep Learning |
|---|---|---|
| **Tabular data** | Almost always better | Rarely justified |
| **Images** | Not competitive | The standard |
| **Text/NLP** | TF-IDF + LR for simple things | The standard (Transformers) |
| **Audio** | Manual features + ML | The standard |
| **< 10K samples** | Generally better | Risk of overfitting |
| **> 100K samples** | Works well | Starts to shine |
| **Need to explain** | Easy (feature importance) | Difficult (black box) |
| **Development time** | Fast (hours-days) | Slow (days-weeks) |
| **Infrastructure** | CPU sufficient | GPU required |
| **Production** | Lightweight, fast | Heavy, more latency |

### Quick decision table

```
Do I have tabular data?
  +- Yes --> Use XGBoost/LightGBM (DL rarely wins on tabular)
  +- No  --> What type of data?
              +- Images     --> CNN or Vision Transformer
              +- Text       --> Transformer (BERT, GPT, etc.)
              +- Audio      --> Transformer (Whisper, etc.)
              +- Video      --> 3D CNN or Video Transformer
              +- Graphs     --> GNN
              +- Multimodal --> Multimodal models (CLIP, etc.)

Do I have few data (<1K)?
  +- Yes --> Transfer learning (pretrained model + fine-tune)
  +- No  --> Training from scratch is an option
```

---

## Artificial Neuron

### Intuitive analogy

An artificial neuron works like a simple decision function:

```
Inputs (features)       Weights (importance)       Weighted sum          Activation     Output
     x1 ---- w1 ----+
                      +---- z = w1*x1 + w2*x2 + w3*x3 + b ---- f(z) ---- output
     x2 ---- w2 ----|                       ^
                      |                    bias
     x3 ---- w3 ----+

Example: predict if you will pass an exam
  x1 = hours of study       w1 = 0.6  (very important)
  x2 = hours of sleep       w2 = 0.3  (important)
  x3 = coffee consumed      w3 = 0.1  (not very important)
  b  = -5.0                           (base threshold)

  z = 0.6*8 + 0.3*7 + 0.1*3 + (-5.0) = 4.8 + 2.1 + 0.3 - 5.0 = 2.2
  output = sigmoid(2.2) = 0.90  --> 90% probability of passing
```

**What the model "learns"** are the weights (w1, w2, w3) and the bias (b). Training adjusts these values to minimize the error.

---

## Activation Functions

Activation functions introduce non-linearity. Without them, a multi-layer neural network would be equivalent to a single linear layer (multiplying matrices gives another matrix).

### Comparison

| Function | Intuitive formula | Range | Main use | Advantage |
|---|---|---|---|---|
| **ReLU** | max(0, x) | [0, inf) | Hidden layers (default) | Simple, fast, avoids vanishing gradient |
| **Sigmoid** | 1/(1+e^-x) | (0, 1) | Binary output | Output interpretable as probability |
| **Tanh** | (e^x - e^-x)/(e^x + e^-x) | (-1, 1) | Hidden layers (alternative to ReLU) | Centered at 0 |
| **GELU** | x * P(X <= x) ~= x * sigmoid(1.7*x) | (-0.17, inf) | Transformers | Smooth, better for NLP |
| **Softmax** | e^xi / sum(e^xj) | (0, 1), sum=1 | Multiclass output | Probability distribution |
| **Leaky ReLU** | max(0.01*x, x) | (-inf, inf) | Avoid "dying ReLU" | Doesn't kill neurons |

### ASCII Graphs

```
ReLU: max(0, x)              Sigmoid: 1/(1+e^-x)          Tanh
      |                            |                            |
  y   |      /                 1.0 |.........------         1.0 |.........------
      |     /                      |       /                    |      /
      |    /                  0.5  |---- /                 0.0  |----/
      |   /                        |   /                        |  /
      |  /                    0.0  |--                    -1.0  |--
  ----+-------- x             -----+-------- x             -----+-------- x
      |                            |                            |
  "If negative, dies"        "Compresses everything        "Like sigmoid but
   "If positive, passes"      between 0 and 1"              centered at 0"

GELU                          Leaky ReLU                   Softmax
      |                            |                       Not a curve, it's a
  y   |      /                 y   |      /                function over a vector
      |    _/                      |    /                   that converts numbers
      |  _/                        |  /                     into probabilities:
      |_/                      ----+/------- x
  ----+-------- x                 /|                       [2.0, 1.0, 0.1]
     ~|                          / |                            v softmax
  "Smooth ReLU"              "ReLU with slope              [0.659, 0.243, 0.098]
                               for negatives"               Sum = 1.0
```

### When to use each one

```python
import torch.nn as nn

# Hidden layers: ReLU (safe default)
nn.ReLU()

# Hidden layers in Transformers: GELU
nn.GELU()

# Output for binary classification: Sigmoid
nn.Sigmoid()  # Output between 0 and 1

# Output for multiclass classification: Softmax
nn.Softmax(dim=-1)  # Probabilities that sum to 1

# If you have "dying ReLU" problems: Leaky ReLU
nn.LeakyReLU(negative_slope=0.01)
```

---

## Neural Network Architecture

### Components

```
Input Layer          Hidden Layers            Output Layer
(features)           (learning)               (prediction)

   x1 ---------+     +-- h1 --+     +-- h4 --+     +-- y1
            +---+-----+         +-----+         +-----+
   x2 ---------+     +-- h2 --+     +-- h5 --+     +-- y2
            +---+-----+         +-----+         |
   x3 ---------+     +-- h3 --+     +-- h6 --+
            |
   x4 ------+

  4 features     3 neurons       3 neurons       2 classes
                 + ReLU           + ReLU           + Softmax

  Each arrow (-) is a weight that is learned.
  Total weights: (4*3) + (3*3) + (3*2) = 12 + 9 + 6 = 27 weights
  Total biases: 3 + 3 + 2 = 8
  Total parameters: 35
```

### Forward pass

Step by step, how data flows through the network:

```
Input data: x = [0.5, 0.3, 0.8, 0.1]

Layer 1 (input -> hidden 1):
  z1 = W1 @ x + b1           # Matrix multiplication + bias
  # z1 = [0.72, -0.15, 1.23]
  h1 = ReLU(z1)              # Activation
  # h1 = [0.72, 0.00, 1.23]  # Note: -0.15 becomes 0

Layer 2 (hidden 1 -> hidden 2):
  z2 = W2 @ h1 + b2
  h2 = ReLU(z2)

Layer 3 (hidden 2 -> output):
  z3 = W3 @ h2 + b3
  output = Softmax(z3)
  # output = [0.85, 0.15]    # 85% class A, 15% class B
```

```python
import torch
import torch.nn as nn

# Define the network from the diagram
class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 3)    # 4 inputs -> 3 neurons
        self.layer2 = nn.Linear(3, 3)    # 3 -> 3 neurons
        self.layer3 = nn.Linear(3, 2)    # 3 -> 2 outputs (classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))    # Layer 1 + ReLU
        x = self.relu(self.layer2(x))    # Layer 2 + ReLU
        x = self.layer3(x)               # Layer 3 (no activation - the loss applies it)
        return x

model = SimpleNetwork()
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
# 35 parameters
```

---

## Backpropagation

### Intuition (no formal derivations)

Backpropagation is the algorithm that "teaches" the network. It works in two phases:

```
FORWARD PASS (left to right):
  Input -> Layer 1 -> Layer 2 -> Output -> Loss
  "We make a prediction and calculate how wrong we are"

BACKWARD PASS (right to left):
  Loss -> Layer 2 -> Layer 1
  "We propagate the error backwards, calculating how much
   each weight contributed to the error"

UPDATE WEIGHTS:
  new_weight = old_weight - learning_rate * gradient
  "We adjust each weight in the direction that reduces the error"
```

### Chain rule: the central idea

The chain rule says: if a change in a weight affects the next layer, which affects the next layer, which affects the loss, we can calculate how much that weight affects the loss by multiplying the effects in chain.

```
Analogy: a chain of dominoes

  Weight w  -->  Neuron h  -->  Output o  -->  Loss L

  How much does w affect the Loss?
  dL/dw = dL/do * do/dh * dh/dw

  It's like asking: "if I move this domino 1mm,
  how much does the last domino move?"
  = (effect of the last) * (effect of the middle) * (effect of the first)
```

### Vanishing and Exploding Gradients

```
VANISHING GRADIENTS (gradients that disappear):

  Layers:    [1] -- [2] -- [3] -- [4] -- [5] -- [Loss]

  Gradients: 0.001  0.01   0.1    0.5    1.0
              <------------------------------------
              Each layer multiplies by a number < 1
              The first layers barely learn

EXPLODING GRADIENTS (gradients that explode):

  Gradients: 1000   100    10     2      1.0
              <------------------------------------
              Each layer multiplies by a number > 1
              Weights jump uncontrollably
```

#### Solutions

| Problem | Solution | How |
|---|---|---|
| **Vanishing** | ReLU instead of sigmoid/tanh | `nn.ReLU()` |
| **Vanishing** | Residual connections (skip connections) | ResNet, Transformers |
| **Vanishing** | Batch Normalization | `nn.BatchNorm1d()` |
| **Vanishing** | Better initialization | `nn.init.kaiming_normal_()` |
| **Exploding** | Gradient clipping | `torch.nn.utils.clip_grad_norm_()` |
| **Exploding** | Lower learning rate | `lr=1e-4` instead of `lr=1e-2` |
| **Both** | Layer Normalization | `nn.LayerNorm()` (used in Transformers) |

```python
# Gradient clipping (prevents exploding gradients)
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## Loss Functions

The loss function measures how wrong the model is. Training consists of minimizing this function.

### When to use each one

| Loss | Problem type | What it measures | PyTorch |
|---|---|---|---|
| **CrossEntropyLoss** | Multiclass classification | Difference between predicted and real distribution | `nn.CrossEntropyLoss()` |
| **BCEWithLogitsLoss** | Binary classification | Binary difference (with numerical stability) | `nn.BCEWithLogitsLoss()` |
| **MSELoss** | Regression | Mean squared error | `nn.MSELoss()` |
| **L1Loss (MAE)** | Regression robust to outliers | Mean absolute error | `nn.L1Loss()` |
| **HuberLoss** | Regression (mix MSE + MAE) | MSE near 0, MAE far away | `nn.HuberLoss()` |

```python
import torch.nn as nn

# Multiclass classification (the most common)
# NOTE: CrossEntropyLoss already includes Softmax internally
# Do NOT put Softmax in the last layer if you use this loss
loss_fn = nn.CrossEntropyLoss()

logits = model(x)       # Shape: (batch_size, num_classes) - WITHOUT softmax
target = labels          # Shape: (batch_size,) - class indices (0, 1, 2...)
loss = loss_fn(logits, target)

# Binary classification
loss_fn = nn.BCEWithLogitsLoss()  # Includes sigmoid internally
logits = model(x)       # Shape: (batch_size, 1) - WITHOUT sigmoid
target = labels.float() # Shape: (batch_size, 1) - 0.0 or 1.0
loss = loss_fn(logits, target)

# Regression
loss_fn = nn.MSELoss()
predictions = model(x)  # Shape: (batch_size, 1)
target = values          # Shape: (batch_size, 1)
loss = loss_fn(predictions, target)
```

### Imbalanced classes

```python
# Option 1: Class weights in the loss
# If you have 90% class 0 and 10% class 1
weights = torch.tensor([1.0, 9.0])  # Give more weight to the minority class
loss_fn = nn.CrossEntropyLoss(weight=weights)

# Option 2: Calculate weights automatically
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights = torch.tensor(class_weights, dtype=torch.float32)
loss_fn = nn.CrossEntropyLoss(weight=weights)
```

---

## Optimizers

The optimizer is the algorithm that updates the weights using the gradients calculated by backpropagation.

### Intuition

```
Imagine you are on a mountain with fog and want to reach the valley (loss minimum):

SGD:         You walk downhill. Simple but slow.
             On gentle slopes, you barely advance.

SGD+Momentum: You walk downhill but with inertia (like a ball rolling).
             You cross flat zones without stopping.

Adam:        You walk downhill with inertia AND adjust
             the step size automatically.
             Large steps in flat zones, small steps in steep zones.
             THIS IS THE DEFAULT. Use Adam if you don't know what to choose.
```

### Comparison table

| Optimizer | Description | When to use | Typical learning rate |
|---|---|---|---|
| **SGD** | The most basic | Almost never alone | 0.01 - 0.1 |
| **SGD + Momentum** | SGD with inertia | CNNs, when you need fine convergence | 0.01 - 0.1 |
| **Adam** | Momentum + adaptive LR | **Default for everything** | 1e-3 - 1e-4 |
| **AdamW** | Adam + correct weight decay | **Default for Transformers, fine-tuning** | 1e-4 - 1e-5 |
| **Adagrad** | Adaptive LR per parameter | Sparse features (old NLP) | 0.01 |
| **RMSprop** | Improved version of Adagrad | RNNs | 1e-3 |

```python
import torch.optim as optim

# Adam (recommended default)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# AdamW (for Transformers and fine-tuning)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# SGD with momentum (for CNNs in long training)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
```

### Learning Rate

The learning rate is the MOST IMPORTANT hyperparameter. It controls the step size in each update.

```
LR too high:                 LR too low:                 LR correct:
      Loss                        Loss                       Loss
      |\  /\                      |\                         |\
      | \/  \  /\                 | \                        | \
      |      \/  \  /             |  \                       |  \
      |           \/              |   \                      |   \___
      |                           |    \___________          |
      +------------ epoch         +------------ epoch        +------------ epoch
  "Jumps too much,             "Converges but               "Converges fast
   never converges"             will take 10x longer"         and well"
```

### Learning Rate Schedulers

Change the learning rate during training. Generally: start high (explore) and gradually decrease (refine).

```python
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, OneCycleLR, LinearLR, SequentialLR
)

# Step decay: reduce LR every N epochs
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# LR: 0.01 -> (epoch 30) -> 0.001 -> (epoch 60) -> 0.0001

# Cosine annealing: reduce LR following a cosine curve
scheduler = CosineAnnealingLR(optimizer, T_max=100)
# LR smoothly decreases from max to min following cosine

# OneCycleLR: warmup + decay (very popular, usually gives good results)
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=num_epochs * len(train_loader),
    pct_start=0.1,  # 10% warmup
)

# Linear warmup + cosine decay (typical in Transformers)
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

# In the training loop:
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = train_step(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()  # Update LR
```

### Warmup: why start with low LR

```
Without warmup:                  With warmup:
  LR                               LR
  |--------\                       |    /------\
  |         \                      |   /        \
  |          \                     |  /          \
  |           \___                 | /            \___
  +------------ step               +------------ step

At the start weights are random.         First we warm up with low LR
A high LR can cause the model           (weights stabilize), then we
to diverge immediately.                  increase LR to train fast.
```

---

## Regularization in Deep Learning

### Dropout

Dropout turns off random neurons during training. This forces the network not to depend on any individual neuron and learn more robust representations.

```
Training (dropout=0.3):           Inference (no dropout):

  [x1]--[h1]--[h4]--[y]           [x1]--[h1]--[h4]--[y]
  [x2]--[XX]--[h5]--               [x2]--[h2]--[h5]--
  [x3]--[h3]--[XX]--               [x3]--[h3]--[h6]--

  XX = turned off neuron           All active, weights
  (30% turned off at random)       scaled by (1-p)
  Different in each batch!
```

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(0.3)   # 30% of neurons turned off
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)              # Dropout AFTER activation
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)                    # NO dropout on the last layer
        return x
```

> **Key point:** Dropout is used between hidden layers, never on the output layer. Typical values: 0.1-0.5. In Transformers, 0.1 is typically used.

### Batch Normalization

Normalizes the activations of each layer so they have mean 0 and variance 1. This stabilizes training and allows using higher learning rates.

```
Without BatchNorm:                With BatchNorm:
Activations of each layer         Normalized before each layer
can have very different ranges     -> more stable training

Layer 1 output: [100, -50, 200]     Layer 1 output: [0.2, -1.1, 1.3]
Layer 2 output: [0.01, 0.003]       Layer 2 output: [-0.5, 0.8]
```

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)    # BatchNorm after linear
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))  # Linear -> BN -> ReLU
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
```

### Weight Decay (L2 regularization)

Penalizes large weights. In AdamW it is integrated directly.

```python
# Weight decay in the optimizer (the correct way with AdamW)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

### Data Augmentation

Create variations of the training data. Very effective for images.

```python
from torchvision import transforms

# Augmentation for images
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# No augmentation for validation/test
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

### Early Stopping

Stop training when validation loss stops improving.

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

# Usage
early_stop = EarlyStopping(patience=10)

for epoch in range(max_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()

    early_stop(val_loss)
    if early_stop.should_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

### Regularization summary

| Technique | Effect | Where | Typical value |
|---|---|---|---|
| **Dropout** | Turns off neurons | Between hidden layers | 0.1-0.5 |
| **Batch Norm** | Normalizes activations | After linear/conv | - |
| **Weight Decay** | Penalizes large weights | In optimizer | 0.01-0.1 |
| **Data Augmentation** | More synthetic data | In DataLoader | Various |
| **Early Stopping** | Stops training | In training loop | patience=5-20 |

---

## PyTorch Fundamentals

### Tensors

PyTorch tensors are like NumPy arrays but with two superpowers: they can live on GPU and they record operations to calculate gradients automatically.

```python
import torch

# Create tensors (similar to NumPy)
x = torch.tensor([1.0, 2.0, 3.0])                 # From list
x = torch.zeros(3, 4)                               # Tensor of zeros
x = torch.ones(3, 4)                                # Tensor of ones
x = torch.randn(3, 4)                               # Normal(0, 1)
x = torch.arange(0, 10, 2)                          # [0, 2, 4, 6, 8]
x = torch.linspace(0, 1, 5)                         # [0, 0.25, 0.5, 0.75, 1.0]

# Properties
print(x.shape)     # torch.Size([3, 4])
print(x.dtype)     # torch.float32
print(x.device)    # cpu or cuda:0

# NumPy <-> PyTorch conversion (share memory if on CPU)
numpy_array = x.numpy()
tensor = torch.from_numpy(numpy_array)

# Operations (same as NumPy)
a = torch.randn(3, 4)
b = torch.randn(3, 4)
c = a + b                    # Addition
d = a @ b.T                  # Matrix multiplication
e = a * b                    # Element-wise multiplication
f = a.mean(), a.std()        # Statistics
```

### GPU support

```python
# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.randn(1000, 1000, device=device)    # Create directly on GPU
y = torch.randn(1000, 1000).to(device)         # Move to GPU

# IMPORTANT: all tensors must be on the same device
z = x @ y  # OK: both on GPU
# z = x @ torch.randn(1000, 1000)  # ERROR: x on GPU, new tensor on CPU
```

### Autograd (automatic differentiation)

```python
# PyTorch records operations to calculate gradients automatically
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Forward pass
y = x ** 2 + 3 * x      # y = x^2 + 3x
loss = y.sum()           # Scalar needed for backward

# Backward pass (calculates gradients)
loss.backward()

# Gradients: dy/dx = 2x + 3
print(x.grad)            # tensor([7., 9.])  ->  2*2+3=7, 2*3+3=9
```

### nn.Module: creating models

```python
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Convolutional layers
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

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Instantiate
model = ImageClassifier(num_classes=10)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Dataset and DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
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

# Create datasets
train_dataset = MyDataset(X_train, y_train)
val_dataset = MyDataset(X_val, y_val)

# DataLoaders (handle batching, shuffling, parallelism)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,         # Shuffle on train
    num_workers=4,        # Load data in parallel
    pin_memory=True,      # Faster for GPU
    drop_last=True,       # Discard last incomplete batch
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,        # Can be larger on val (no backward)
    shuffle=False,        # NO shuffle on val
    num_workers=4,
    pin_memory=True,
)
```

### Complete training loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    """Train the model for one epoch."""
    model.train()  # Training mode (activates dropout, batchnorm)
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move data to device (GPU/CPU)
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Backward pass
        optimizer.zero_grad()     # Clear previous gradients
        loss.backward()           # Calculate gradients
        optimizer.step()          # Update weights

        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()  # Don't calculate gradients (faster, less memory)
def validate(model, val_loader, loss_fn, device):
    """Evaluate the model on the validation set."""
    model.eval()  # Evaluation mode (deactivates dropout, batchnorm uses global stats)
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
model = ImageClassifier(num_classes=10).to(device)
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

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/best_model.pt')
        print(f"  -> Best model saved (val_loss: {val_loss:.4f})")

    # Early stopping
    early_stop(val_loss)
    if early_stop.should_stop:
        print(f"Early stopping at epoch {epoch+1}")
        break

print(f"\nBest val_loss: {best_val_loss:.4f}")
```

### Saving and loading models

```python
# SAVE: only the state_dict (weights) - RECOMMENDED
torch.save(model.state_dict(), 'models/model_v1.pt')

# LOAD:
model = ImageClassifier(num_classes=10)  # Create the architecture
model.load_state_dict(torch.load('models/model_v1.pt', map_location=device))
model.to(device)
model.eval()  # Evaluation mode for inference

# Save complete checkpoint (to resume training)
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss': best_val_loss,
    'train_loss': train_loss,
}
torch.save(checkpoint, 'models/checkpoint_epoch10.pt')

# Load checkpoint
checkpoint = torch.load('models/checkpoint_epoch10.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Device management

```python
# Universal pattern
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

device = get_device()
print(f"Using: {device}")

# Move model to device
model = model.to(device)

# Move data to device (in the training loop)
inputs = inputs.to(device)
targets = targets.to(device)

# Move result to CPU for NumPy/Pandas
predictions = model(inputs).cpu().numpy()

# Multi-GPU (DataParallel - basic)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)
```

---

## Key Hyperparameters

### Learning rate (the most important)

```python
# Find the optimal LR with LR Finder (Leslie Smith's concept)
from torch.optim.lr_scheduler import ExponentialLR

# Idea: start with a very low LR and increase exponentially.
# Plot loss vs LR. The best LR is just before the loss explodes.

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
# Choose LR where the loss drops fastest (steepest slope)
```

### Batch size

| Batch size | Advantage | Disadvantage |
|---|---|---|
| **Small** (16-32) | Better generalization, less memory | Slower, noisier |
| **Medium** (64-128) | Good balance | - |
| **Large** (256-1024) | Faster, more GPU efficient | May generalize worse, needs high LR |

```
Rule of thumb:
- Use the largest batch size that fits in your GPU
- If you increase batch size, increase LR proportionally
  (batch 32, lr=1e-3) -> (batch 64, lr=2e-3)
- For LLM fine-tuning: batch size 4-16 (large models)
```

### How many layers and neurons

```
General rule:
- Start small and grow
- More data -> you can use larger models
- The first hidden layer is usually the largest
- Gradually reduce size: 512 -> 256 -> 128

Typical example for tabular data:
  Input (20 features)
  -> Linear(20, 128) + ReLU + Dropout(0.3)
  -> Linear(128, 64) + ReLU + Dropout(0.3)
  -> Linear(64, num_classes)
```

### Hyperparameter tuning

```python
# Optuna (RECOMMENDED - the best hyperparameter tuning framework)
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    n_layers = trial.suggest_int('n_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])

    # Create model with these hyperparameters
    model = create_model(n_layers, hidden_size, dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train and evaluate
    for epoch in range(20):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)

        # Report intermediate metric (allows pruning of bad trials)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_acc

# Run search
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, timeout=3600)  # 50 trials or 1 hour

# Best hyperparameters
print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Visualize
optuna.visualization.plot_param_importances(study)
optuna.visualization.plot_optimization_history(study)
```

| Method | Description | When to use |
|---|---|---|
| **Grid search** | Try all combinations | Few hyperparameters (2-3) |
| **Random search** | Try random combinations | Better than grid search in general |
| **Optuna (Bayesian)** | Intelligent guided search | **Whenever you can, it's the best** |

---

## Transfer Learning

### What it is and why it works

Instead of training a model from scratch, we start from a model that was already trained on a large dataset (like ImageNet with 1M images). The first layers already learned general patterns (edges, textures, shapes) that are useful for any visual task.

```
Pretrained model (ImageNet):

  [Edges] -> [Textures] -> [Parts] -> [Objects] -> [1000 ImageNet classes]
   Layer 1     Layer 2      Layer 3     Layer 4       Head

Your task (classify X-rays):

  [Edges] -> [Textures] -> [Parts] -> [Objects] -> [2 classes: normal/abnormal]
   Reuse      Reuse        Reuse      Reuse         NEW head
   (freeze)   (freeze)     (freeze)   (freeze)      (train)
```

### Fine-tuning vs Feature extraction

| Strategy | What you do | When to use |
|---|---|---|
| **Feature extraction** | Freeze everything, only train the new head | Few data (<1K), task very similar to pretraining |
| **Partial fine-tuning** | Freeze first layers, train last layers + head | Moderate data (1K-10K) |
| **Full fine-tuning** | Train the entire model with low LR | Lots of data (>10K), task different from pretraining |

```python
import torchvision.models as models

# Load pretrained model
model = models.resnet50(weights='IMAGENET1K_V2')

# === Strategy 1: Feature Extraction ===
# Freeze ALL parameters
for param in model.parameters():
    param.requires_grad = False

# Replace the last layer (head) for your task
num_classes = 5
model.fc = nn.Linear(model.fc.in_features, num_classes)
# Only model.fc has requires_grad=True

# Optimizer only for parameters that require gradient
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)


# === Strategy 2: Partial fine-tuning ===
model = models.resnet50(weights='IMAGENET1K_V2')

# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last layers (layer4) + head
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, num_classes)

# Different LR for pretrained layers vs new head
optimizer = optim.AdamW([
    {'params': model.layer4.parameters(), 'lr': 1e-5},   # Low LR for pretrained
    {'params': model.fc.parameters(), 'lr': 1e-3},        # High LR for new head
], weight_decay=0.01)


# === Strategy 3: Full fine-tuning ===
model = models.resnet50(weights='IMAGENET1K_V2')
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Low LR for everything (don't destroy what was learned)
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
```

### Popular pretrained models

| Model | Type | Parameters | Main use | PyTorch |
|---|---|---|---|---|
| **ResNet-50** | CNN | 25M | Images (baseline) | `models.resnet50()` |
| **EfficientNet-B0** | CNN | 5M | Images (efficient) | `models.efficientnet_b0()` |
| **ViT-B/16** | Transformer | 86M | Images (SOTA) | `models.vit_b_16()` |
| **BERT** | Transformer | 110M | Text (classification, NER) | HuggingFace `transformers` |
| **GPT-2** | Transformer | 124M-1.5B | Text (generation) | HuggingFace `transformers` |
| **Whisper** | Transformer | 39M-1.5B | Audio (transcription) | HuggingFace `transformers` |
| **CLIP** | Multimodal | 400M | Images + Text | OpenAI / HuggingFace |

---

## Mixed Precision Training

### What it is

Normally models use float32 (32 bits per number). Mixed precision uses float16 for most operations and float32 only where necessary. This reduces memory by half and speeds up training on modern GPUs.

```
Float32 (full precision):       Float16 (half precision):
  32 bits per weight              16 bits per weight
  More memory                     Less memory (~50%)
  Slower on new GPUs              Faster (Tensor Cores)
  Always stable                   Can be unstable (loss scaling)
```

### When to use it

- You have an NVIDIA GPU with Tensor Cores (V100, A100, RTX 3000+, RTX 4000+)
- Your model doesn't fit on GPU with float32
- You want to train faster without sacrificing quality

### Implementation with torch.amp

```python
from torch.amp import autocast, GradScaler

# Create scaler (prevents underflow in float16)
scaler = GradScaler('cuda')

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass in mixed precision
        with autocast('cuda'):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        # Backward pass with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # The loss for logging must be .item() (float32)
        print(f"Loss: {loss.item():.4f}")
```

> **Key point:** Mixed precision is "free": it barely affects model precision, reduces memory ~50%, and speeds up training 1.5-3x. Use it whenever your GPU supports it.

---

## Practical Tips

### 1. Always start with a small model

```python
# First: minimal model to verify the pipeline works
model = nn.Sequential(
    nn.Linear(input_size, 32),
    nn.ReLU(),
    nn.Linear(32, num_classes)
)
# Train for 5 epochs
# If it works, start increasing complexity
```

### 2. Overfit a small batch (sanity check)

Before training on the entire dataset, verify that your model CAN learn by trying to memorize a single batch. If it can't overfit a batch, something is wrong.

```python
# Take a single batch
batch = next(iter(train_loader))
inputs, targets = batch
inputs, targets = inputs.to(device), targets.to(device)

model = MyModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Try to overfit this single batch
for i in range(200):
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 20 == 0:
        acc = (outputs.argmax(1) == targets).float().mean()
        print(f"Step {i}: Loss={loss.item():.4f}, Acc={acc:.4f}")

# If loss reaches ~0 and acc ~1.0 -> the model can learn -> OK
# If it doesn't go down -> bug in code, model too small, or incorrect LR
```

### 3. Monitor loss and metrics

```python
# With Weights & Biases (wandb) - the industry standard
import wandb

wandb.init(project="my-project", config={
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

# With TensorBoard (free alternative)
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs/experiment_1')
for epoch in range(50):
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
writer.close()

# Launch TensorBoard
# tensorboard --logdir=logs/
```

### 4. Reproducibility

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    """Set seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For total reproducibility (may be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Checklist before training

```
Before launching a long training run, verify:

[ ] Sanity check: overfit a small batch
[ ] Data: verify shapes, dtypes, value ranges
[ ] Labels: verify class distribution
[ ] Model: count parameters, verify output shape
[ ] Loss: verify it's the correct one for the problem
[ ] Optimizer: verify LR and weight decay
[ ] Device: model and data on the same device
[ ] Reproducibility: seed set
[ ] Logging: wandb/tensorboard configured
[ ] Checkpointing: save best model
[ ] Early stopping: configured
[ ] GPU memory: verify it fits with nvidia-smi
```

### Common debugging

| Problem | Possible cause | Solution |
|---|---|---|
| Loss doesn't go down | LR too low or too high | Try LR finder |
| Loss is NaN | LR too high, overflow | Lower LR, gradient clipping |
| Loss goes up | Bug in code, high LR | Check labels, lower LR |
| Val loss goes up (train goes down) | Overfitting | Dropout, weight decay, more data |
| Train/val loss both high | Underfitting | Larger model, more epochs |
| OOM (Out of Memory) | Model/batch too large | Reduce batch size, mixed precision |
| Training very slow | CPU bottleneck in data loading | More num_workers, pin_memory |
| Accuracy doesn't go above X% | Imbalanced class, incorrect metric | Class weights, change metric |

---

> **Summary:** Deep Learning is a powerful tool but not always the best one. Use it when you have lots of non-tabular data (images, text, audio). Start simple, verify your pipeline works with a sanity check, and then scale up. Use transfer learning whenever you can. Adam/AdamW with cosine LR schedule is a robust setup to start with.
