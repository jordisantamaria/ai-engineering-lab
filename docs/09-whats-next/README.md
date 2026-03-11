# What's Left to Learn

> This repository covers the essentials for getting started with AI/ML consulting. But the field is enormous. Here are the advanced topics you might need depending on the type of project that comes your way.

---

## Table of Contents

- [Advanced Topics Map](#advanced-topics-map)
- [1. Time Series Forecasting](#1-time-series-forecasting)
- [2. Reinforcement Learning](#2-reinforcement-learning)
- [3. Generative AI (beyond LLMs)](#3-generative-ai-beyond-llms)
- [4. Audio / Speech](#4-audio--speech)
- [5. Multimodal AI](#5-multimodal-ai)
- [6. Graph Neural Networks](#6-graph-neural-networks)
- [7. Federated Learning](#7-federated-learning)
- [8. Edge AI / TinyML](#8-edge-ai--tinyml)
- [9. AutoML](#9-automl)
- [10. Responsible AI / Fairness](#10-responsible-ai--fairness)
- [Suggested Priority for Consulting](#suggested-priority-for-consulting)
- [Quick Summary](#quick-summary)

---

## Advanced Topics Map

```
                         ┌──────────────────┐
                         │  WHAT YOU KNOW   │
                         │   (this repo)    │
                         │                  │
                         │ Classical ML, DL,│
                         │ NLP, LLMs, RAG,  │
                         │ CV, MLOps,       │
                         │ Deployment       │
                         └────────┬─────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
     ┌────────▼────────┐  ┌──────▼───────┐  ┌───────▼────────┐
     │    NEW DATA     │  │     NEW      │  │   ADVANCED     │
     │    TYPES        │  │  MODALITIES  │  │   DEPLOYMENT   │
     │                 │  │              │  │                │
     │ Time Series     │  │ Audio/Speech │  │ Edge AI/TinyML │
     │ Graph Data      │  │ Multimodal   │  │ Federated      │
     │                 │  │ Gen AI       │  │ Learning       │
     └────────┬────────┘  └──────┬───────┘  └───────┬────────┘
              │                  │                   │
     ┌────────▼────────┐  ┌─────▼────────┐  ┌──────▼─────────┐
     │  AUTOMATION     │  │  ETHICS &    │  │  SPECIALIZED   │
     │                 │  │  REGULATION  │  │                │
     │ AutoML          │  │              │  │ Reinforcement  │
     │                 │  │ Responsible  │  │ Learning       │
     │                 │  │ AI/Fairness  │  │                │
     └─────────────────┘  └──────────────┘  └────────────────┘
```

---

## 1. Time Series Forecasting

### What It Is

Prediction of future values based on historically ordered temporal data. Product demand, sales, stock prices, energy consumption, sensor data.

### When You'll Need It

- **Retail:** Demand and inventory forecasting
- **Finance:** Revenue, cashflow, and price prediction
- **Manufacturing:** Predictive maintenance, production
- **Energy:** Consumption, renewable generation
- **Logistics:** Shipping volumes, delivery times

### Technology Stack

| Tool | Type | Best For | Complexity |
|-------------|------|-----------|-------------|
| **Prophet** (Meta) | Statistical | Series with seasonality, quick to implement | Low |
| **NeuralProphet** | Hybrid | Prophet + neural networks | Low-Medium |
| **statsmodels** | Statistical | ARIMA, SARIMA, classical models | Medium |
| **N-BEATS** | Deep Learning | Multiple series, high accuracy | Medium |
| **Temporal Fusion Transformer** | Deep Learning | Complex series with exogenous variables | High |
| **Darts** (Unit8) | Framework | Unifies multiple models under one API | Medium |
| **GluonTS** (Amazon) | Framework | Probabilistic models, AWS integration | Medium-High |

### Quick Example: Prophet

```python
from prophet import Prophet
import pandas as pd

# Data: columns "ds" (date) and "y" (value)
df = pd.read_csv("ventas.csv")
df.columns = ["ds", "y"]

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    changepoint_prior_scale=0.05,  # Trend flexibility
)
model.fit(df)

# Predict 90 days
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Visualize
model.plot(forecast)
model.plot_components(forecast)  # Trend + seasonality
```

### Difficulty: Medium

The statistical methods (Prophet, ARIMA) are accessible. Deep learning models (TFT, N-BEATS) require more experience. The real challenge is not the model but understanding seasonality, special events, and evaluating correctly (time-series cross-validation, not random split).

### Recommended Resources

- **Book:** "Forecasting: Principles and Practice" (Hyndman & Athanasopoulos) - free online
- **Course:** Kaggle Time Series course
- **Practice:** Kaggle forecasting competitions (Store Sales, M5)
- **Documentation:** Prophet docs, Darts docs

---

## 2. Reinforcement Learning

### What It Is

An ML paradigm where an agent learns to make decisions through trial and error, receiving rewards or penalties for its actions. The agent optimizes its behavior to maximize cumulative reward.

```
Agent observes state → Takes action → Receives reward → Updates policy
         ↑                                                        │
         └────────────────────────────────────────────────────────┘
```

### When You'll Need It

- **Robotics:** Robotic arm control, navigation
- **Games:** Strategy, intelligent NPCs
- **Process optimization:** Scheduling, resource allocation
- **Dynamic pricing:** Adjusting prices in real time
- **Recommendation systems:** Recommendation sequences
- **Industrial control:** HVAC, supply chains

### Technology Stack

| Tool | Type | Best For |
|-------------|------|-----------|
| **Stable-Baselines3** | RL algorithms | Quick implementation of PPO, SAC, DQN |
| **Gymnasium** (formerly Gym) | Environments | Standard environments for experimentation |
| **RLlib** (Ray) | Distributed | RL at scale, multi-agent |
| **CleanRL** | Educational | Simple and readable implementations |
| **PettingZoo** | Multi-agent | Multi-agent environments |

### Quick Example

```python
import gymnasium as gym
from stable_baselines3 import PPO

# Create environment
env = gym.make("CartPole-v1")

# Train agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)

# Evaluate
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
```

### Difficulty: High

RL is conceptually different from supervised ML. Training agents is unstable, hyperparameters are sensitive, and reward engineering (designing the reward function) is an art. Debugging is hard because errors manifest as "the agent doesn't learn" without a clear error message.

> **Important note:** Very few consulting projects need RL. Most problems are better solved with classical optimization or supervised ML. RL shines when the decision space is huge and actions have long-term consequences.

### Recommended Resources

- **Book:** "Reinforcement Learning: An Introduction" (Sutton & Barto) - free online
- **Course:** David Silver's RL Course (UCL/DeepMind) on YouTube
- **Course:** Hugging Face Deep RL Course
- **Practice:** Gymnasium environments, then custom environments

---

## 3. Generative AI (beyond LLMs)

### GANs (Generative Adversarial Networks)

Two networks compete: a generator creates fake data, a discriminator tries to distinguish it from real data. Over time, the generator produces data indistinguishable from real data.

```
Random noise → [Generator] → Fake image ──┐
                                          ├─→ [Discriminator] → Real/Fake
                            Real image ───┘
```

**Use cases:**
- Data augmentation (generating synthetic data to train other models)
- Image generation (faces, products, art)
- Image super-resolution
- Inpainting (filling in missing parts of images)

### Diffusion Models

The models behind Stable Diffusion, DALL-E, Midjourney. They learn to generate images by gradually removing noise.

```python
# Stable Diffusion with diffusers (Hugging Face)
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

image = pipe(
    "A professional photograph of a modern office building, 4k, detailed",
    num_inference_steps=50,
    guidance_scale=7.5,
).images[0]

image.save("generated_office.png")
```

**Use cases:**
- Visual content generation (marketing, e-commerce)
- Rapid design prototyping
- Image personalization at scale

### Variational Autoencoders (VAEs)

A generative model that learns a compressed latent representation of the data. It allows generating new samples and interpolating between examples.

**Use cases:**
- Anomaly detection (poor reconstruction = anomaly)
- Synthetic tabular data generation
- Data compression

### When You'll Need Gen AI (non-LLM)

| Case | Tool | Difficulty |
|------|-------------|-----------|
| Generate images from text | Stable Diffusion, DALL-E API | Low (API) / High (fine-tune) |
| Data augmentation (images) | GANs, Diffusion | Medium |
| Synthetic tabular data | CTGAN, SDV | Medium |
| Anomaly detection | VAE | Medium |
| Super-resolution | ESRGAN, Real-ESRGAN | Low (pre-trained) |

### Recommended Resources

- **Course:** Hugging Face Diffusion Models Course
- **Book:** "Generative Deep Learning" (David Foster) - O'Reilly
- **Practice:** Fine-tune Stable Diffusion with DreamBooth, generate data with CTGAN

---

## 4. Audio / Speech

### Speech-to-Text (STT)

**Whisper (OpenAI)** has democratized STT. It is open-source, multilingual (99 languages), and works surprisingly well even with noisy audio.

```python
import whisper

# Load model (tiny, base, small, medium, large)
model = whisper.load_model("medium")

# Transcribe
result = model.transcribe(
    "audio.mp3",
    language="es",     # Language (optional, auto-detects)
    task="transcribe",  # or "translate" to translate to English
)

print(result["text"])

# With timestamps per segment
for segment in result["segments"]:
    print(f"[{segment['start']:.1f}s - {segment['end']:.1f}s] {segment['text']}")
```

Hosted alternatives: Google Speech-to-Text, AWS Transcribe, Azure Speech.

### Text-to-Speech (TTS)

| Tool | Quality | Latency | Open Source | Multilingual |
|-------------|---------|----------|-------------|-------------|
| **Bark** (Suno) | High | Slow | Yes | Yes |
| **XTTS** (Coqui) | Very high | Medium | Yes | Yes |
| **ElevenLabs** | Excellent | Low | No (API) | Yes |
| **Google TTS** | Good | Low | No (API) | Yes |
| **Edge TTS** | Good | Low | Free (API) | Yes |

```python
# XTTS - Voice cloning with 6 seconds of reference
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

tts.tts_to_file(
    text="Hola, esto es una prueba de síntesis de voz.",
    speaker_wav="referencia_voz.wav",  # 6s of reference audio
    language="es",
    file_path="output.wav",
)
```

### Audio Classification

Classify audio into categories: music vs speech, emotions, sound types, event detection.

```python
from transformers import pipeline

classifier = pipeline(
    "audio-classification",
    model="MIT/ast-finetuned-audioset-10-10-0.4593",
)

result = classifier("audio.wav")
# [{"label": "Speech", "score": 0.95}, {"label": "Music", "score": 0.03}, ...]
```

### When You'll Need It

- **Call centers:** Call transcription, audio sentiment analysis
- **Accessibility:** STT for hearing-impaired users, TTS for visually impaired users
- **Content:** Podcasts to text, automatic subtitling
- **Healthcare:** Voice analysis for diagnosis (research)
- **Legal:** Hearing and deposition transcription

### Difficulty: Low-Medium

Whisper for STT is extremely easy to use. TTS requires more work if you need high quality and voice cloning. Audio classification with pre-trained models is accessible.

### Recommended Resources

- **Documentation:** Whisper GitHub, Coqui TTS docs
- **Course:** Hugging Face Audio Course
- **Practice:** Transcribe meetings with Whisper, build a call analysis pipeline

---

## 5. Multimodal AI

### What It Is

Models that process and combine multiple data types (modalities): text + image, text + audio, image + tabular data, etc.

### Key Models

| Model | Modalities | What It Does | Access |
|--------|------------|----------|--------|
| **CLIP** (OpenAI) | Text + Image | Shared embeddings, similarity search | Open source |
| **LLaVA** | Text + Image | Visual understanding, answering questions about images | Open source |
| **GPT-4o** | Text + Image + Audio | Full multimodal understanding | API |
| **Gemini** | Text + Image + Audio + Video | Multimodal understanding | API |
| **ImageBind** (Meta) | 6 modalities | Unified embeddings | Open source |

### Example: CLIP for Text-Based Image Search

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Search images with text
images = [Image.open(f"img_{i}.jpg") for i in range(100)]
text_query = "un gato durmiendo en un sofá"

inputs = processor(
    text=[text_query],
    images=images,
    return_tensors="pt",
    padding=True,
)

outputs = model(**inputs)
similarities = outputs.logits_per_text.softmax(dim=-1)

# Top 5 most relevant images
top5 = similarities[0].topk(5)
for score, idx in zip(top5.values, top5.indices):
    print(f"Image {idx}: {score:.3f}")
```

### When You'll Need It

- **E-commerce:** Visual search (upload photo, find similar products)
- **E-commerce:** Product classification = image + title + description + price
- **Medical AI:** Combining medical images + clinical history + lab data
- **Real estate:** Valuation = photos + description + numerical data
- **Content moderation:** Image + post text to detect inappropriate content

### Difficulty: Medium-High

Using pre-trained models (CLIP, GPT-4o) is easy via API. Multimodal fine-tuning and training custom models is significantly more complex due to cross-modal alignment.

### Recommended Resources

- **Documentation:** Hugging Face Transformers (multimodal section)
- **Papers:** CLIP, LLaVA, Flamingo
- **Practice:** Build an image search engine with CLIP, use GPT-4o Vision for classification

---

## 6. Graph Neural Networks

### What It Is

ML on data with graph structure: nodes connected by edges. GNNs learn representations by considering the connection structure, not just each node's features.

```
Tabular data:  Each row is independent
Graph data:    Each node depends on its neighbors

Example - Social network:
  User A ─── friends with ─── User B
      │                          │
  friends with               friends with
      │                          │
  User C ─── friends with ─── User D

  Question: Will A and D become friends? (link prediction)
  Question: Is C a bot? (node classification)
```

### When You'll Need It

| Industry | Use Case | Graph Type |
|-----------|-------------|---------------|
| **Social networks** | Friend recommendation, bot detection | User network |
| **Finance** | Fraud detection, transaction network | Account/transaction network |
| **Pharmaceuticals** | Drug discovery, molecular interaction | Molecular graphs |
| **Retail** | Recommendation systems (user-item graph) | Bipartite graph |
| **Infrastructure** | Network optimization, fault detection | Network graphs |
| **Knowledge graphs** | Semantic search, reasoning | Knowledge graphs |

### Technology Stack

| Tool | Purpose | Ease of Use |
|-------------|----------|-----------|
| **PyTorch Geometric (PyG)** | Main GNN framework | Medium |
| **DGL (Deep Graph Library)** | Alternative to PyG, more flexible | Medium |
| **NetworkX** | Graph analysis (not ML) | Low |
| **Neo4j** + **GDS** | Graph database + native ML | Medium |
| **Stellargraph** | Simplified GNN (Keras-like) | Low |

### Example: Node Classification with PyG

```python
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Dataset
dataset = Planetoid(root="data", name="Cora")
data = dataset[0]

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

### Difficulty: Medium-High

Conceptually different from traditional ML. Requires understanding graph theory and message passing. Datasets and evaluations have their own quirks (transductive vs inductive, train/val/test splits on graphs). The tooling (PyG, DGL) is mature but has a learning curve.

### Recommended Resources

- **Course:** Stanford CS224W (Machine Learning with Graphs) on YouTube
- **Book:** "Graph Representation Learning" (Hamilton) - free online
- **Tutorial:** PyTorch Geometric tutorials
- **Practice:** Classification on Cora/Citeseer, then fraud detection

---

## 7. Federated Learning

### What It Is

A technique for collaboratively training ML models without sharing data between participants. Each participant trains a local model with their data, and only shares gradients or model weights (not the data itself).

```
Participant A (sensitive data) → Trains local model → Sends weights ──┐
Participant B (sensitive data) → Trains local model → Sends weights ──┤
Participant C (sensitive data) → Trains local model → Sends weights ──┤
                                                                       │
                     ┌─────────────────────────────────────────────────┘
                     ▼
            [Central server]
            Aggregates weights (FedAvg)
            Sends global model
                     │
                     ├──→ Participant A (updates local model)
                     ├──→ Participant B (updates local model)
                     └──→ Participant C (updates local model)
```

### When You'll Need It

- **Healthcare:** Hospitals that cannot share patient data (HIPAA, GDPR)
- **Finance:** Banks collaborating on fraud detection without sharing transactions
- **Telecommunications:** Models on user devices (on-device)
- **Manufacturing:** Factories with proprietary data collaborating on quality

### Technology Stack

| Tool | Type | Best For |
|-------------|------|-----------|
| **Flower** | Framework | General purpose, flexible, production |
| **PySyft** (OpenMined) | Framework + privacy | Differential privacy, secure computation |
| **TensorFlow Federated** | Framework | TensorFlow integration |
| **NVIDIA FLARE** | Enterprise | Healthcare, enterprise |

### Difficulty: High

It is not just the technical implementation (which frameworks like Flower simplify considerably), but the inherent challenges:
- Non-IID data (each participant has a different distribution)
- Efficient communication (compressing updates)
- Attacks and robustness (malicious participants)
- Organizational coordination between participants

### Recommended Resources

- **Course:** Flower Federated Learning tutorials
- **Paper:** "Communication-Efficient Learning of Deep Networks" (McMahan et al.)
- **Practice:** Flower tutorial with distributed CIFAR-10

---

## 8. Edge AI / TinyML

### What It Is

Deploying ML models on edge devices: mobile phones, IoT, microcontrollers, cameras, sensors. The model runs directly on the device, without needing to send data to the cloud.

```
Cloud AI:     Device → [Internet] → Server (inference) → [Internet] → Result
Edge AI:      Device → [Local model] → Result (immediate, no internet)
```

### Edge AI Advantages

| Advantage | Why |
|---------|---------|
| **Latency** | No round trip to the cloud, immediate response |
| **Privacy** | Data never leaves the device |
| **Cost** | No cloud inference cost |
| **Offline** | Works without internet |
| **Bandwidth** | No need to send heavy data (video, audio) |

### When You'll Need It

- **Manufacturing:** Visual inspection on production lines (cameras with local model)
- **Retail:** Smart cameras (people counting, analytics)
- **Automotive:** ADAS, real-time object detection
- **Agriculture:** Drones with on-device pest detection
- **Wearables:** Fall detection, health monitoring

### Technology Stack

| Tool | Target | Model Size | Speed |
|-------------|--------|--------------|-----------|
| **TensorFlow Lite** | Mobile, IoT | Very small | Fast |
| **ONNX Runtime Mobile** | Mobile, Edge | Small | Fast |
| **OpenVINO** (Intel) | Intel hardware | Medium | Very fast |
| **TensorRT** (NVIDIA) | NVIDIA GPUs/Jetson | Medium | Very fast |
| **Core ML** (Apple) | iOS/macOS | Variable | Fast |
| **MediaPipe** (Google) | Mobile, web | Very small | Fast |
| **Apache TVM** | Any hardware | Variable | Optimized |

### Example: Convert to TensorFlow Lite

```python
import tensorflow as tf

# Trained model
model = tf.keras.models.load_model("my_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantization
converter.target_spec.supported_types = [tf.float16]   # FP16

tflite_model = converter.convert()

# Save (much smaller than the original)
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Size: {len(tflite_model) / 1024:.1f} KB")
```

### Difficulty: Medium-High

The model itself can be easy to convert. The challenges are:
- Optimizing for specific hardware (every device is different)
- Maintaining acceptable accuracy with aggressive quantization
- Integrating with device firmware/apps
- Updating models on deployed devices (OTA updates)
- Testing on real hardware (emulators are not always faithful)

### Recommended Resources

- **Book:** "TinyML" (Pete Warden & Daniel Situnayake) - O'Reilly
- **Course:** Harvard CS249r - TinyML and Efficient Deep Learning
- **Documentation:** TensorFlow Lite, ONNX Runtime, OpenVINO
- **Hardware for learning:** Raspberry Pi, NVIDIA Jetson Nano, Arduino Nano 33 BLE

---

## 9. AutoML

### What It Is

Automation of the ML pipeline: feature selection, model selection, hyperparameter tuning, and in some cases, automatic feature engineering. The goal is to get a good model with minimal human effort.

```
AutoML:
  Data → [Try 50+ combinations of models and parameters] → Best model

  What it automates:
  - Data preprocessing
  - Feature selection
  - Algorithm selection
  - Hyperparameter tuning
  - Model ensembling
```

### When You'll Need It

- **Rapid prototyping:** Get a strong baseline in hours, not weeks
- **Clients without ML team:** Deliver models without needing senior data scientists
- **Benchmarking:** Compare your manual model against AutoML as a baseline
- **Tabular data:** AutoML especially shines with tabular data

### Tools

| Tool | Open Source | Type | Best For | Ease of Use |
|-------------|-----------|------|-----------|-----------|
| **AutoGluon** (Amazon) | Yes | Framework | Tabular, text, image, multimodal | Very easy |
| **H2O AutoML** | Yes | Framework | Tabular data, enterprise | Easy |
| **FLAML** (Microsoft) | Yes | Framework | Fast, low resource | Easy |
| **Auto-sklearn** | Yes | Wrapper sklearn | Tabular data | Easy |
| **TPOT** | Yes | Genetic programming | sklearn pipelines | Easy |
| **Google Cloud AutoML** | No | Managed | Vision, NLP, tabular | Very easy |
| **Azure AutoML** | No | Managed | Integrated with Azure ML | Easy |

### Example: AutoGluon

```python
from autogluon.tabular import TabularPredictor
import pandas as pd

# Load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Train (AutoGluon decides everything: models, features, tuning)
predictor = TabularPredictor(
    label="target",           # Column to predict
    eval_metric="f1_weighted",
    path="autogluon_models",
).fit(
    train_data,
    time_limit=3600,          # 1 hour maximum
    presets="best_quality",   # or "medium_quality" for faster
)

# Evaluate
results = predictor.evaluate(test_data)
print(results)

# Model leaderboard
leaderboard = predictor.leaderboard(test_data)
print(leaderboard)

# Predict
predictions = predictor.predict(test_data)
```

### Difficulty: Low

AutoML is easy to use by design. The challenge lies in knowing when to trust its results and when you need manual control. Also in avoiding overfitting (AutoML can overfit if you don't have a properly separated test set).

> **Consulting tip:** Use AutoML as your first baseline. If the client needs more, improve manually. Many times the AutoML model is already good enough, especially for tabular data.

### Recommended Resources

- **Documentation:** AutoGluon docs (excellent), H2O docs
- **Practice:** Kaggle competitions with AutoGluon as baseline
- **Articles:** AutoGluon, Auto-sklearn papers

---

## 10. Responsible AI / Fairness

### What It Is

A set of practices to ensure that ML models are fair, transparent, explainable, and non-discriminatory. Includes bias detection, fairness metrics, explainability, and regulatory compliance.

### When You'll Need It (increasingly mandatory)

| Regulation | Where | Impact |
|------------|-------|---------|
| **EU AI Act** | Europe | Risk classification, mandatory audits for "high risk" |
| **NYC Local Law 144** | New York | Bias audit in hiring tools |
| **GDPR Art. 22** | Europe | Right to explanation in automated decisions |
| **ECOA / Fair Lending** | USA | Non-discrimination in credit |
| **Sector-specific** | Global | Healthcare, insurance, criminal justice |

**Sectors where it's critical:**
- **Hiring / HR:** CV screening, candidate scoring
- **Finance:** Credit scoring, loan approval
- **Healthcare:** Diagnosis, triage, resource allocation
- **Justice:** Risk assessment, recidivism prediction
- **Insurance:** Pricing, risk evaluation

### Technology Stack

| Tool | What It Does | Author |
|-------------|----------|-------|
| **Fairlearn** | Fairness metrics, mitigation algorithms | Microsoft |
| **AI Fairness 360 (AIF360)** | 70+ fairness metrics, mitigation | IBM |
| **SHAP** | Explainability (why the model predicted X) | Lundberg |
| **LIME** | Local interpretable explanations | Ribeiro |
| **Aequitas** | Bias audit | U. Chicago |
| **InterpretML** | Interpretable models (EBM) | Microsoft |
| **What-If Tool** | Interactive fairness exploration | Google |

### Example: Bias Detection with Fairlearn

```python
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from sklearn.metrics import accuracy_score, recall_score

# Suppose you have predictions and a sensitive attribute (gender, race, etc.)
y_true = [...]        # Actual labels
y_pred = [...]        # Model predictions
sensitive = [...]      # Sensitive attribute (e.g., "male", "female")

# Metrics by group
metric_frame = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "recall": recall_score,
    },
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive,
)

print("Metrics by group:")
print(metric_frame.by_group)

print("\nDifference between groups:")
print(metric_frame.difference())

# Demographic parity: is the positive prediction rate equal across groups?
dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
print(f"\nDemographic Parity Difference: {dpd:.4f}")
# If close to 0, the model treats groups similarly
# If far from 0, there is potential bias
```

### Example: Explainability with SHAP

```python
import shap

# Trained model
model = trained_xgboost_model
X_test = test_features

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global plot: which features are most important
shap.summary_plot(shap_values, X_test)

# Local explanation: why this specific prediction
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=X_test.columns.tolist(),
))
```

### Difficulty: Medium

The tools are accessible. The real difficulty is understanding what type of fairness to pursue (there are multiple definitions that can be mutually exclusive) and navigating the specific legal requirements of each jurisdiction and sector.

### Recommended Resources

- **Book:** "Fairness and Machine Learning" (Barocas, Hardt, Narayanan) - free online
- **Course:** Google Responsible AI Practices
- **Documentation:** Fairlearn docs, AIF360 docs
- **Regulation:** EU AI Act text, NIST AI Risk Management Framework

---

## Suggested Priority for Consulting

Not all topics have the same demand. This is a prioritization based on what clients ask for most frequently:

### High Priority (learn it soon)

| # | Topic | Why | Demand |
|---|------|---------|---------|
| 1 | **Time Series Forecasting** | Almost every client with historical data asks for it. Retail, finance, manufacturing, energy. Prophet allows delivering results quickly. | Very high |
| 2 | **Audio/Speech** | Whisper makes transcription trivial. Call centers, accessibility, content. Immediate ROI. | High |
| 3 | **AutoML** | Dramatically accelerates prototyping. AutoGluon generates competitive baselines in hours. Perfect for first meetings with a client. | High |

### Medium Priority (learn when a project comes up)

| # | Topic | Why | Demand |
|---|------|---------|---------|
| 4 | **Edge AI** | Manufacturing and retail clients are increasingly asking for it. Requires specific hardware. | Medium-High |
| 5 | **Responsible AI** | Regulation (EU AI Act) is making it mandatory. Competitive differentiator as a consultant. | Medium (growing) |
| 6 | **Multimodal AI** | E-commerce and medical AI are the main drivers. GPT-4o simplifies it via API. | Medium |

### Low Priority (learn if you need it)

| # | Topic | Why | Demand |
|---|------|---------|---------|
| 7 | **Gen AI (non-LLM)** | Specific cases (data augmentation, visual content). Stable Diffusion via API covers most needs. | Low-Medium |
| 8 | **Graph Neural Networks** | Niche but powerful. Fraud detection and recommendation are the most common cases. | Low |
| 9 | **Federated Learning** | Very few projects need it. Only when regulation prevents sharing data and there are multiple participants. | Low |
| 10 | **Reinforcement Learning** | Almost never in consulting. Optimization problems are usually better solved with classical methods. | Very low |

### Prioritization Diagram

```
Impact on consulting
        ▲
  High  │  Time Series ★       Audio/Speech ★
        │                          AutoML ★
        │
  Med   │  Edge AI              Responsible AI
        │                       Multimodal AI
        │
  Low   │  Gen AI (non-LLM)     Graph NN
        │  Federated Learning
        │  Reinforcement Learning
        └──────────────────────────────────────────▶
              Low               Medium             High
                        Ease of learning

★ = Learn first
```

---

## Quick Summary

| Topic | Difficulty | Consulting Demand | Time to Become Productive |
|------|-----------|--------------------|-----------------------------|
| Time Series | Medium | Very high | 2-4 weeks |
| Reinforcement Learning | High | Very low | 2-3 months |
| Gen AI (non-LLM) | Medium | Low-Medium | 2-4 weeks |
| Audio/Speech | Low-Medium | High | 1-2 weeks |
| Multimodal AI | Medium-High | Medium | 3-4 weeks |
| Graph Neural Networks | Medium-High | Low | 1-2 months |
| Federated Learning | High | Low | 1-2 months |
| Edge AI / TinyML | Medium-High | Medium-High | 1-2 months |
| AutoML | Low | High | 1 week |
| Responsible AI | Medium | Medium (growing) | 2-3 weeks |

---

> **Key takeaway:** Don't try to learn everything at once. With what this repository covers (classical ML, deep learning, NLP, LLMs, RAG, Computer Vision, deployment, MLOps) you can already tackle the majority of AI consulting projects. Expand into Time Series, Audio, and AutoML first. The rest, learn it when a project requires it. Breadth comes with projects, not courses.
