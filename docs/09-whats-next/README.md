# Qué Falta por Aprender

> Este repositorio cubre lo esencial para empezar a hacer consultoría AI/ML. Pero el campo es enorme. Aquí están los temas avanzados que podrías necesitar según el tipo de proyecto que te llegue.

---

## Tabla de Contenidos

- [Mapa de Temas Avanzados](#mapa-de-temas-avanzados)
- [1. Time Series Forecasting](#1-time-series-forecasting)
- [2. Reinforcement Learning](#2-reinforcement-learning)
- [3. Generative AI (más allá de LLMs)](#3-generative-ai-más-allá-de-llms)
- [4. Audio / Speech](#4-audio--speech)
- [5. Multimodal AI](#5-multimodal-ai)
- [6. Graph Neural Networks](#6-graph-neural-networks)
- [7. Federated Learning](#7-federated-learning)
- [8. Edge AI / TinyML](#8-edge-ai--tinyml)
- [9. AutoML](#9-automl)
- [10. Responsible AI / Fairness](#10-responsible-ai--fairness)
- [Prioridad Sugerida para Consultoría](#prioridad-sugerida-para-consultoría)
- [Resumen Rápido](#resumen-rápido)

---

## Mapa de Temas Avanzados

```
                         ┌──────────────────┐
                         │   LO QUE SABES   │
                         │   (este repo)    │
                         │                  │
                         │ ML clásico, DL,  │
                         │ NLP, LLMs, RAG,  │
                         │ CV, MLOps,       │
                         │ Deployment       │
                         └────────┬─────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
     ┌────────▼────────┐  ┌──────▼───────┐  ┌───────▼────────┐
     │  DATOS NUEVOS   │  │  MODALIDADES │  │  DEPLOYMENT    │
     │                 │  │  NUEVAS      │  │  AVANZADO      │
     │ Time Series     │  │              │  │                │
     │ Graph Data      │  │ Audio/Speech │  │ Edge AI/TinyML │
     │                 │  │ Multimodal   │  │ Federated      │
     └────────┬────────┘  │ Gen AI       │  │ Learning       │
              │           └──────┬───────┘  └───────┬────────┘
              │                  │                   │
     ┌────────▼────────┐  ┌─────▼────────┐  ┌──────▼─────────┐
     │  AUTOMATIZACIÓN │  │  ÉTICA &     │  │  ESPECIALIZADO │
     │                 │  │  REGULACIÓN  │  │                │
     │ AutoML          │  │              │  │ Reinforcement  │
     │                 │  │ Responsible  │  │ Learning       │
     │                 │  │ AI/Fairness  │  │                │
     └─────────────────┘  └──────────────┘  └────────────────┘
```

---

## 1. Time Series Forecasting

### Qué es

Predicción de valores futuros basándose en datos históricos ordenados temporalmente. Demanda de productos, ventas, precio de acciones, consumo energético, datos de sensores.

### Cuándo lo necesitarás

- **Retail:** Forecasting de demanda e inventario
- **Finanzas:** Predicción de ingresos, cashflow, precios
- **Manufactura:** Mantenimiento predictivo, producción
- **Energía:** Consumo, generación renovable
- **Logística:** Volúmenes de envío, tiempos de entrega

### Stack Tecnológico

| Herramienta | Tipo | Mejor Para | Complejidad |
|-------------|------|-----------|-------------|
| **Prophet** (Meta) | Estadístico | Series con estacionalidad, rápido de implementar | Baja |
| **NeuralProphet** | Híbrido | Prophet + redes neuronales | Baja-Media |
| **statsmodels** | Estadístico | ARIMA, SARIMA, modelos clásicos | Media |
| **N-BEATS** | Deep Learning | Múltiples series, alta accuracy | Media |
| **Temporal Fusion Transformer** | Deep Learning | Series complejas con variables exógenas | Alta |
| **Darts** (Unit8) | Framework | Unifica múltiples modelos en una API | Media |
| **GluonTS** (Amazon) | Framework | Modelos probabilísticos, AWS integration | Media-Alta |

### Ejemplo Rápido: Prophet

```python
from prophet import Prophet
import pandas as pd

# Datos: columnas "ds" (fecha) y "y" (valor)
df = pd.read_csv("ventas.csv")
df.columns = ["ds", "y"]

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    changepoint_prior_scale=0.05,  # Flexibilidad de la tendencia
)
model.fit(df)

# Predecir 90 días
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Visualizar
model.plot(forecast)
model.plot_components(forecast)  # Tendencia + estacionalidad
```

### Dificultad: Media

Lo estadístico (Prophet, ARIMA) es accesible. Los modelos deep learning (TFT, N-BEATS) requieren más experiencia. El reto real no es el modelo sino entender la estacionalidad, eventos especiales, y evaluar correctamente (time-series cross-validation, no random split).

### Recursos Recomendados

- **Libro:** "Forecasting: Principles and Practice" (Hyndman & Athanasopoulos) - gratis online
- **Curso:** Kaggle Time Series course
- **Práctica:** Kaggle competitions de forecasting (Store Sales, M5)
- **Documentación:** Prophet docs, Darts docs

---

## 2. Reinforcement Learning

### Qué es

Paradigma de ML donde un agente aprende a tomar decisiones mediante prueba y error, recibiendo recompensas o penalizaciones por sus acciones. El agente optimiza su comportamiento para maximizar la recompensa acumulada.

```
Agente observa estado → Toma acción → Recibe recompensa → Actualiza política
         ↑                                                        │
         └────────────────────────────────────────────────────────┘
```

### Cuándo lo necesitarás

- **Robótica:** Control de brazos robóticos, navegación
- **Juegos:** Estrategia, NPCs inteligentes
- **Optimización de procesos:** Scheduling, asignación de recursos
- **Pricing dinámico:** Ajustar precios en tiempo real
- **Sistemas de recomendación:** Secuencias de recomendaciones
- **Control industrial:** HVAC, cadenas de suministro

### Stack Tecnológico

| Herramienta | Tipo | Mejor Para |
|-------------|------|-----------|
| **Stable-Baselines3** | Algoritmos RL | Implementación rápida de PPO, SAC, DQN |
| **Gymnasium** (antes Gym) | Entornos | Entornos estándar para experimentar |
| **RLlib** (Ray) | Distribuido | RL a escala, multi-agente |
| **CleanRL** | Educativo | Implementaciones simples y legibles |
| **PettingZoo** | Multi-agente | Entornos multi-agente |

### Ejemplo Rápido

```python
import gymnasium as gym
from stable_baselines3 import PPO

# Crear entorno
env = gym.make("CartPole-v1")

# Entrenar agente
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)

# Evaluar
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
```

### Dificultad: Alta

RL es conceptualmente diferente al ML supervisado. Entrenar agentes es inestable, los hiperparámetros son sensibles, y la reward engineering (disenar la función de recompensa) es un arte. Debugging es difícil porque los errores se manifiestan como "el agente no aprende" sin mensaje de error claro.

> **Nota importante:** Muy pocos proyectos de consultoría necesitan RL. La mayoría de problemas se resuelven mejor con optimización clásica o ML supervisado. RL brilla cuando el espacio de decisiones es enorme y las acciones tienen consecuencias a largo plazo.

### Recursos Recomendados

- **Libro:** "Reinforcement Learning: An Introduction" (Sutton & Barto) - gratis online
- **Curso:** David Silver's RL Course (UCL/DeepMind) en YouTube
- **Curso:** Hugging Face Deep RL Course
- **Práctica:** Gymnasium environments, luego entornos custom

---

## 3. Generative AI (más allá de LLMs)

### GANs (Generative Adversarial Networks)

Dos redes compiten: un generador crea datos falsos, un discriminador intenta distinguirlos de los reales. Con el tiempo, el generador produce datos indistinguibles de los reales.

```
Ruido aleatorio → [Generador] → Imagen falsa ──┐
                                                ├─→ [Discriminador] → Real/Falso
                              Imagen real ──────┘
```

**Casos de uso:**
- Data augmentation (generar datos sintéticos para entrenar otros modelos)
- Generación de imágenes (caras, productos, arte)
- Super-resolución de imágenes
- Inpainting (rellenar partes faltantes de imágenes)

### Diffusion Models

Los modelos que están detrás de Stable Diffusion, DALL-E, Midjourney. Aprenden a generar imágenes eliminando ruido gradualmente.

```python
# Stable Diffusion con diffusers (Hugging Face)
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

**Casos de uso:**
- Generación de contenido visual (marketing, e-commerce)
- Prototipado rápido de diseños
- Personalización de imágenes a escala

### Variational Autoencoders (VAEs)

Modelo generativo que aprende una representación latente comprimida de los datos. Permite generar nuevas muestras y hacer interpolación entre ejemplos.

**Casos de uso:**
- Detección de anomalías (reconstrucción mala = anomalía)
- Generación de datos tabulares sintéticos
- Compresión de datos

### Cuándo necesitarás Gen AI (no-LLM)

| Caso | Herramienta | Dificultad |
|------|-------------|-----------|
| Generar imágenes desde texto | Stable Diffusion, DALL-E API | Baja (API) / Alta (fine-tune) |
| Data augmentation (imágenes) | GANs, Diffusion | Media |
| Datos tabulares sintéticos | CTGAN, SDV | Media |
| Detección de anomalías | VAE | Media |
| Super-resolución | ESRGAN, Real-ESRGAN | Baja (pre-entrenado) |

### Recursos Recomendados

- **Curso:** Hugging Face Diffusion Models Course
- **Libro:** "Generative Deep Learning" (David Foster) - O'Reilly
- **Práctica:** Fine-tune Stable Diffusion con DreamBooth, generar datos con CTGAN

---

## 4. Audio / Speech

### Speech-to-Text (STT)

**Whisper (OpenAI)** ha democratizado el STT. Es open-source, multilingual (99 idiomas), y funciona sorprendentemente bien incluso en audio con ruido.

```python
import whisper

# Cargar modelo (tiny, base, small, medium, large)
model = whisper.load_model("medium")

# Transcribir
result = model.transcribe(
    "audio.mp3",
    language="es",     # Idioma (opcional, auto-detecta)
    task="transcribe",  # o "translate" para traducir a inglés
)

print(result["text"])

# Con timestamps por segmento
for segment in result["segments"]:
    print(f"[{segment['start']:.1f}s - {segment['end']:.1f}s] {segment['text']}")
```

Alternativas hosted: Google Speech-to-Text, AWS Transcribe, Azure Speech.

### Text-to-Speech (TTS)

| Herramienta | Calidad | Latencia | Open Source | Multilingue |
|-------------|---------|----------|-------------|-------------|
| **Bark** (Suno) | Alta | Lenta | Si | Si |
| **XTTS** (Coqui) | Muy alta | Media | Si | Si |
| **ElevenLabs** | Excelente | Baja | No (API) | Si |
| **Google TTS** | Buena | Baja | No (API) | Si |
| **Edge TTS** | Buena | Baja | Gratis (API) | Si |

```python
# XTTS - Clonación de voz con 6 segundos de referencia
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

tts.tts_to_file(
    text="Hola, esto es una prueba de síntesis de voz.",
    speaker_wav="referencia_voz.wav",  # 6s de audio de referencia
    language="es",
    file_path="output.wav",
)
```

### Audio Classification

Clasificar audio en categorías: música vs habla, emociones, tipos de sonido, detección de eventos.

```python
from transformers import pipeline

classifier = pipeline(
    "audio-classification",
    model="MIT/ast-finetuned-audioset-10-10-0.4593",
)

result = classifier("audio.wav")
# [{"label": "Speech", "score": 0.95}, {"label": "Music", "score": 0.03}, ...]
```

### Cuándo lo necesitarás

- **Call centers:** Transcripción de llamadas, análisis de sentimiento en audio
- **Accesibilidad:** STT para personas con discapacidad auditiva, TTS para visual
- **Contenido:** Podcasts a texto, subtitulado automático
- **Salud:** Análisis de voz para diagnóstico (investigación)
- **Legal:** Transcripción de audiencias, deposiciones

### Dificultad: Baja-Media

Whisper para STT es extremadamente fácil de usar. TTS requiere más trabajo si necesitas calidad alta y clonación de voz. Audio classification con modelos pre-entrenados es accesible.

### Recursos Recomendados

- **Documentación:** Whisper GitHub, Coqui TTS docs
- **Curso:** Hugging Face Audio Course
- **Práctica:** Transcribir reuniones con Whisper, construir un pipeline de análisis de llamadas

---

## 5. Multimodal AI

### Qué es

Modelos que procesan y combinan múltiples tipos de datos (modalidades): texto + imagen, texto + audio, imagen + datos tabulares, etc.

### Modelos Clave

| Modelo | Modalidades | Qué Hace | Acceso |
|--------|------------|----------|--------|
| **CLIP** (OpenAI) | Texto + Imagen | Embeddings compartidos, búsqueda por similitud | Open source |
| **LLaVA** | Texto + Imagen | Comprensión visual, responder preguntas sobre imágenes | Open source |
| **GPT-4o** | Texto + Imagen + Audio | Comprensión multimodal completa | API |
| **Gemini** | Texto + Imagen + Audio + Video | Comprensión multimodal | API |
| **ImageBind** (Meta) | 6 modalidades | Embeddings unificados | Open source |

### Ejemplo: CLIP para Búsqueda de Imágenes por Texto

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Buscar imágenes con texto
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

# Top 5 imágenes más relevantes
top5 = similarities[0].topk(5)
for score, idx in zip(top5.values, top5.indices):
    print(f"Imagen {idx}: {score:.3f}")
```

### Cuándo lo necesitarás

- **E-commerce:** Búsqueda visual (subir foto, encontrar productos similares)
- **E-commerce:** Clasificación producto = imagen + título + descripción + precio
- **Medical AI:** Combinar imagen médica + historial clínico + datos lab
- **Real estate:** Valoración = fotos + descripción + datos numéricos
- **Content moderation:** Imagen + texto del post para detectar contenido inapropiado

### Dificultad: Media-Alta

Usar modelos pre-entrenados (CLIP, GPT-4o) es fácil via API. Fine-tuning multimodal y entrenar modelos propios es significativamente más complejo por la alineación entre modalidades.

### Recursos Recomendados

- **Documentación:** Hugging Face Transformers (sección multimodal)
- **Papers:** CLIP, LLaVA, Flamingo
- **Práctica:** Construir un buscador de imágenes con CLIP, usar GPT-4o Vision para clasificación

---

## 6. Graph Neural Networks

### Qué es

ML sobre datos que tienen estructura de grafo: nodos conectados por aristas. Los GNNs aprenden representaciones considerando la estructura de conexiones, no solo las features de cada nodo.

```
Datos tabulares:  Cada fila es independiente
Datos de grafo:   Cada nodo depende de sus vecinos

Ejemplo - Red social:
  Usuario A ─── amigo de ─── Usuario B
      │                          │
  amigo de                   amigo de
      │                          │
  Usuario C ─── amigo de ─── Usuario D

  Pregunta: ¿A y D se harán amigos? (link prediction)
  Pregunta: ¿C es un bot? (node classification)
```

### Cuándo lo necesitarás

| Industria | Caso de Uso | Tipo de Grafo |
|-----------|-------------|---------------|
| **Redes sociales** | Recomendación de amigos, detección de bots | Red de usuarios |
| **Finanzas** | Detección de fraude, red de transacciones | Red de cuentas/transacciones |
| **Farmacéutica** | Descubrimiento de fármacos, interacción molecular | Grafos moleculares |
| **Retail** | Sistemas de recomendación (user-item graph) | Grafo bipartito |
| **Infraestructura** | Optimización de redes, detección de fallos | Grafos de red |
| **Knowledge graphs** | Búsqueda semántica, razonamiento | Grafos de conocimiento |

### Stack Tecnológico

| Herramienta | Para Qué | Facilidad |
|-------------|----------|-----------|
| **PyTorch Geometric (PyG)** | Framework principal de GNN | Media |
| **DGL (Deep Graph Library)** | Alternativa a PyG, más flexible | Media |
| **NetworkX** | Análisis de grafos (no ML) | Baja |
| **Neo4j** + **GDS** | Base de datos de grafos + ML nativo | Media |
| **Stellargraph** | GNN simplificado (Keras-like) | Baja |

### Ejemplo: Node Classification con PyG

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

### Dificultad: Media-Alta

Conceptualmente diferente al ML tradicional. Requiere entender teoría de grafos y message passing. Los datasets y evaluaciones tienen sus propias peculiaridades (transductive vs inductive, train/val/test splits en grafos). El tooling (PyG, DGL) es maduro pero con curva de aprendizaje.

### Recursos Recomendados

- **Curso:** Stanford CS224W (Machine Learning with Graphs) en YouTube
- **Libro:** "Graph Representation Learning" (Hamilton) - gratis online
- **Tutorial:** PyTorch Geometric tutorials
- **Práctica:** Clasificación en Cora/Citeseer, luego detección de fraude

---

## 7. Federated Learning

### Qué es

Técnica para entrenar modelos ML de forma colaborativa sin compartir los datos entre participantes. Cada participante entrena un modelo local con sus datos, y solo comparte los gradientes o pesos del modelo (no los datos en sí).

```
Participante A (datos sensibles) → Entrena modelo local → Envía pesos ──┐
Participante B (datos sensibles) → Entrena modelo local → Envía pesos ──┤
Participante C (datos sensibles) → Entrena modelo local → Envía pesos ──┤
                                                                         │
                     ┌───────────────────────────────────────────────────┘
                     ▼
            [Servidor central]
            Agrega pesos (FedAvg)
            Envía modelo global
                     │
                     ├──→ Participante A (actualiza modelo local)
                     ├──→ Participante B (actualiza modelo local)
                     └──→ Participante C (actualiza modelo local)
```

### Cuándo lo necesitarás

- **Healthcare:** Hospitales que no pueden compartir datos de pacientes (HIPAA, GDPR)
- **Finanzas:** Bancos colaborando en detección de fraude sin compartir transacciones
- **Telecomunicaciones:** Modelos en dispositivos de usuarios (on-device)
- **Manufactura:** Fábricas con datos propietarios colaborando en calidad

### Stack Tecnológico

| Herramienta | Tipo | Mejor Para |
|-------------|------|-----------|
| **Flower** | Framework | General purpose, flexible, producción |
| **PySyft** (OpenMined) | Framework + privacidad | Privacidad diferencial, computación segura |
| **TensorFlow Federated** | Framework | Integración con TensorFlow |
| **NVIDIA FLARE** | Enterprise | Healthcare, enterprise |

### Dificultad: Alta

No solo es la implementación técnica (que frameworks como Flower simplifican bastante), sino los retos inherentes:
- Datos no-IID (cada participante tiene distribución diferente)
- Comunicación eficiente (comprimir actualizaciones)
- Ataques y robustez (participantes maliciosos)
- Coordinación organizacional entre participantes

### Recursos Recomendados

- **Curso:** Flower Federated Learning tutorials
- **Paper:** "Communication-Efficient Learning of Deep Networks" (McMahan et al.)
- **Práctica:** Tutorial de Flower con CIFAR-10 distribuido

---

## 8. Edge AI / TinyML

### Qué es

Desplegar modelos ML en dispositivos edge: móviles, IoT, microcontroladores, cámaras, sensores. El modelo se ejecuta directamente en el dispositivo, sin necesidad de enviar datos a la nube.

```
Cloud AI:     Dispositivo → [Internet] → Servidor (inference) → [Internet] → Resultado
Edge AI:      Dispositivo → [Modelo local] → Resultado (inmediato, sin internet)
```

### Ventajas de Edge AI

| Ventaja | Por qué |
|---------|---------|
| **Latencia** | Sin viaje a la nube, respuesta inmediata |
| **Privacidad** | Los datos nunca salen del dispositivo |
| **Coste** | Sin coste de cloud inference |
| **Offline** | Funciona sin internet |
| **Ancho de banda** | No necesita enviar datos pesados (video, audio) |

### Cuándo lo necesitarás

- **Manufactura:** Inspección visual en línea de producción (cámaras con modelo local)
- **Retail:** Cámaras inteligentes (conteo de personas, analytics)
- **Automotive:** ADAS, detección de objetos en tiempo real
- **Agriculture:** Drones con detección de plagas on-device
- **Wearables:** Detección de caídas, monitorización de salud

### Stack Tecnológico

| Herramienta | Target | Tamaño Modelo | Velocidad |
|-------------|--------|--------------|-----------|
| **TensorFlow Lite** | Mobile, IoT | Muy pequeño | Rápida |
| **ONNX Runtime Mobile** | Mobile, Edge | Pequeño | Rápida |
| **OpenVINO** (Intel) | Intel hardware | Medio | Muy rápida |
| **TensorRT** (NVIDIA) | NVIDIA GPUs/Jetson | Medio | Muy rápida |
| **Core ML** (Apple) | iOS/macOS | Variable | Rápida |
| **MediaPipe** (Google) | Mobile, web | Muy pequeño | Rápida |
| **Apache TVM** | Cualquier hardware | Variable | Optimizada |

### Ejemplo: Convertir a TensorFlow Lite

```python
import tensorflow as tf

# Modelo entrenado
model = tf.keras.models.load_model("my_model.h5")

# Convertir a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimizaciones
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantization
converter.target_spec.supported_types = [tf.float16]   # FP16

tflite_model = converter.convert()

# Guardar (mucho más pequeño que el original)
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Tamaño: {len(tflite_model) / 1024:.1f} KB")
```

### Dificultad: Media-Alta

El modelo en sí puede ser fácil de convertir. Los retos son:
- Optimizar para el hardware específico (cada dispositivo es diferente)
- Mantener accuracy aceptable con quantization agresiva
- Integrar con firmware/apps del dispositivo
- Actualizar modelos en dispositivos desplegados (OTA updates)
- Testing en hardware real (emuladores no siempre son fieles)

### Recursos Recomendados

- **Libro:** "TinyML" (Pete Warden & Daniel Situnayake) - O'Reilly
- **Curso:** Harvard CS249r - TinyML and Efficient Deep Learning
- **Documentación:** TensorFlow Lite, ONNX Runtime, OpenVINO
- **Hardware para aprender:** Raspberry Pi, NVIDIA Jetson Nano, Arduino Nano 33 BLE

---

## 9. AutoML

### Qué es

Automatización del pipeline de ML: selección de features, selección de modelo, tuning de hiperparámetros, y en algunos casos, feature engineering automático. El objetivo es obtener un buen modelo con mínimo esfuerzo humano.

```
AutoML:
  Datos → [Probar 50+ combinaciones de modelos y parámetros] → Mejor modelo

  Lo que automatiza:
  - Preprocesamiento de datos
  - Selección de features
  - Selección de algoritmo
  - Tuning de hiperparámetros
  - Ensamblado de modelos
```

### Cuándo lo necesitarás

- **Prototipado rápido:** Obtener un baseline fuerte en horas, no semanas
- **Clientes sin equipo ML:** Entregar modelos sin necesitar data scientists senior
- **Benchmarking:** Comparar tu modelo manual contra AutoML como baseline
- **Tabular data:** AutoML brilla especialmente en datos tabulares

### Herramientas

| Herramienta | Open Source | Tipo | Mejor Para | Facilidad |
|-------------|-----------|------|-----------|-----------|
| **AutoGluon** (Amazon) | Si | Framework | Tabular, texto, imagen, multimodal | Muy fácil |
| **H2O AutoML** | Si | Framework | Datos tabulares, enterprise | Fácil |
| **FLAML** (Microsoft) | Si | Framework | Rápido, bajo recurso | Fácil |
| **Auto-sklearn** | Si | Wrapper sklearn | Datos tabulares | Fácil |
| **TPOT** | Si | Genetic programming | Pipelines sklearn | Fácil |
| **Google Cloud AutoML** | No | Managed | Vision, NLP, tabular | Muy fácil |
| **Azure AutoML** | No | Managed | Integrado con Azure ML | Fácil |

### Ejemplo: AutoGluon

```python
from autogluon.tabular import TabularPredictor
import pandas as pd

# Cargar datos
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Entrenar (AutoGluon decide todo: modelos, features, tuning)
predictor = TabularPredictor(
    label="target",           # Columna a predecir
    eval_metric="f1_weighted",
    path="autogluon_models",
).fit(
    train_data,
    time_limit=3600,          # 1 hora máximo
    presets="best_quality",   # o "medium_quality" para más rápido
)

# Evaluar
results = predictor.evaluate(test_data)
print(results)

# Leaderboard de modelos
leaderboard = predictor.leaderboard(test_data)
print(leaderboard)

# Predecir
predictions = predictor.predict(test_data)
```

### Dificultad: Baja

AutoML es fácil de usar por diseño. El reto está en saber cuándo confiar en sus resultados y cuándo necesitas control manual. También en evitar overfitting (AutoML puede sobreajustar si no tienes un test set bien separado).

> **Consejo para consultoría:** Usa AutoML como primer baseline. Si el cliente necesita más, mejora manualmente. Muchas veces el modelo de AutoML ya es suficientemente bueno, especialmente para datos tabulares.

### Recursos Recomendados

- **Documentación:** AutoGluon docs (excelentes), H2O docs
- **Práctica:** Kaggle competitions con AutoGluon como baseline
- **Artículos:** Papers de AutoGluon, Auto-sklearn

---

## 10. Responsible AI / Fairness

### Qué es

Conjunto de prácticas para asegurar que los modelos ML son justos, transparentes, explicables y no discriminan. Incluye detección de sesgos, fairness metrics, explainability, y cumplimiento regulatorio.

### Cuándo lo necesitarás (cada vez más obligatorio)

| Regulación | Dónde | Impacto |
|------------|-------|---------|
| **EU AI Act** | Europa | Clasificación de riesgo, auditorías obligatorias para "high risk" |
| **NYC Local Law 144** | Nueva York | Auditoría de bias en herramientas de hiring |
| **GDPR Art. 22** | Europa | Derecho a explicación en decisiones automatizadas |
| **ECOA / Fair Lending** | EEUU | No discriminación en crédito |
| **Sector-specific** | Global | Healthcare, seguros, justicia penal |

**Sectores donde es crítico:**
- **Hiring / RRHH:** Screening de CVs, scoring de candidatos
- **Finanzas:** Scoring crediticio, aprobación de préstamos
- **Healthcare:** Diagnóstico, triaje, asignación de recursos
- **Justicia:** Risk assessment, predicción de reincidencia
- **Seguros:** Pricing, evaluación de riesgo

### Stack Tecnológico

| Herramienta | Qué Hace | Autor |
|-------------|----------|-------|
| **Fairlearn** | Métricas de fairness, algoritmos de mitigación | Microsoft |
| **AI Fairness 360 (AIF360)** | 70+ métricas de fairness, mitigación | IBM |
| **SHAP** | Explainability (por qué el modelo predijo X) | Lundberg |
| **LIME** | Explicaciones locales interpretables | Ribeiro |
| **Aequitas** | Auditoría de bias | U. Chicago |
| **InterpretML** | Modelos interpretables (EBM) | Microsoft |
| **What-If Tool** | Exploración interactiva de fairness | Google |

### Ejemplo: Detección de Bias con Fairlearn

```python
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from sklearn.metrics import accuracy_score, recall_score

# Supón que tienes predicciones y un atributo sensible (género, raza, etc.)
y_true = [...]        # Labels reales
y_pred = [...]        # Predicciones del modelo
sensitive = [...]      # Atributo sensible (e.g., "male", "female")

# Métricas por grupo
metric_frame = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "recall": recall_score,
    },
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive,
)

print("Métricas por grupo:")
print(metric_frame.by_group)

print("\nDiferencia entre grupos:")
print(metric_frame.difference())

# Demographic parity: ¿la tasa de predicción positiva es igual entre grupos?
dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
print(f"\nDemographic Parity Difference: {dpd:.4f}")
# Si es cercano a 0, el modelo trata a los grupos de forma similar
# Si es lejano de 0, hay sesgo potencial
```

### Ejemplo: Explicabilidad con SHAP

```python
import shap

# Modelo entrenado
model = trained_xgboost_model
X_test = test_features

# Calcular SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Gráfico global: qué features son más importantes
shap.summary_plot(shap_values, X_test)

# Explicación local: por qué esta predicción específica
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=X_test.columns.tolist(),
))
```

### Dificultad: Media

Las herramientas son accesibles. La dificultad real es entender qué tipo de fairness buscar (hay múltiples definiciones que pueden ser mutuamente excluyentes) y navegar los requisitos legales específicos de cada jurisdicción y sector.

### Recursos Recomendados

- **Libro:** "Fairness and Machine Learning" (Barocas, Hardt, Narayanan) - gratis online
- **Curso:** Google Responsible AI Practices
- **Documentación:** Fairlearn docs, AIF360 docs
- **Regulación:** EU AI Act text, NIST AI Risk Management Framework

---

## Prioridad Sugerida para Consultoría

No todos los temas tienen la misma demanda. Esta es una priorización basada en lo que los clientes piden más frecuentemente:

### Prioridad Alta (aprenderlo pronto)

| # | Tema | Por Qué | Demanda |
|---|------|---------|---------|
| 1 | **Time Series Forecasting** | Casi todos los clientes con datos históricos lo piden. Retail, finanzas, manufactura, energía. Prophet permite entregar resultados rápido. | Muy alta |
| 2 | **Audio/Speech** | Whisper hace que transcripción sea trivial. Call centers, accesibilidad, contenido. ROI inmediato. | Alta |
| 3 | **AutoML** | Acelera prototipado drásticamente. AutoGluon genera baselines competitivos en horas. Perfecto para primeras reuniones con cliente. | Alta |

### Prioridad Media (aprender cuando surja un proyecto)

| # | Tema | Por Qué | Demanda |
|---|------|---------|---------|
| 4 | **Edge AI** | Clientes de manufactura y retail lo piden cada vez más. Requiere hardware específico. | Media-Alta |
| 5 | **Responsible AI** | La regulación (EU AI Act) lo está convirtiendo en obligatorio. Diferenciador competitivo como consultor. | Media (creciendo) |
| 6 | **Multimodal AI** | E-commerce y medical AI son los drivers principales. GPT-4o lo simplifica vía API. | Media |

### Prioridad Baja (aprender si lo necesitas)

| # | Tema | Por Qué | Demanda |
|---|------|---------|---------|
| 7 | **Gen AI (no-LLM)** | Casos específicos (data augmentation, contenido visual). Stable Diffusion vía API cubre la mayoría de necesidades. | Baja-Media |
| 8 | **Graph Neural Networks** | Nicho pero poderoso. Fraud detection y recomendación son los casos más comunes. | Baja |
| 9 | **Federated Learning** | Muy pocos proyectos lo necesitan. Solo cuando la regulación impide compartir datos y hay múltiples participantes. | Baja |
| 10 | **Reinforcement Learning** | Casi nunca en consultoría. Problemas de optimización suelen resolverse mejor con métodos clásicos. | Muy baja |

### Diagrama de Priorización

```
Impacto en consultoría
        ▲
  Alto  │  Time Series ★       Audio/Speech ★
        │                          AutoML ★
        │
  Medio │  Edge AI              Responsible AI
        │                       Multimodal AI
        │
  Bajo  │  Gen AI (no-LLM)     Graph NN
        │  Federated Learning
        │  Reinforcement Learning
        └──────────────────────────────────────────▶
              Baja              Media              Alta
                        Facilidad de aprender

★ = Aprender primero
```

---

## Resumen Rápido

| Tema | Dificultad | Demanda Consultoría | Tiempo para ser Productivo |
|------|-----------|--------------------|-----------------------------|
| Time Series | Media | Muy alta | 2-4 semanas |
| Reinforcement Learning | Alta | Muy baja | 2-3 meses |
| Gen AI (no-LLM) | Media | Baja-Media | 2-4 semanas |
| Audio/Speech | Baja-Media | Alta | 1-2 semanas |
| Multimodal AI | Media-Alta | Media | 3-4 semanas |
| Graph Neural Networks | Media-Alta | Baja | 1-2 meses |
| Federated Learning | Alta | Baja | 1-2 meses |
| Edge AI / TinyML | Media-Alta | Media-Alta | 1-2 meses |
| AutoML | Baja | Alta | 1 semana |
| Responsible AI | Media | Media (creciendo) | 2-3 semanas |

---

> **Resumen para llevar:** No intentes aprender todo a la vez. Con lo que cubre este repositorio (ML clásico, deep learning, NLP, LLMs, RAG, Computer Vision, deployment, MLOps) ya puedes abordar la mayoría de proyectos de consultoría AI. Expande hacia Time Series, Audio, y AutoML primero. El resto, apréndelo cuando un proyecto lo requiera. La amplitud viene con los proyectos, no con los cursos.
