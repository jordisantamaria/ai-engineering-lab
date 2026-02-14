# NLP (Natural Language Processing)

## NLP en la Era de los LLMs

Con la llegada de ChatGPT y los LLMs, el panorama de NLP ha cambiado drasticamente. Pero no todo se resuelve con un LLM.

### Que Sigue Siendo Relevante vs Que Cubren los LLMs

| Area | Sigue siendo relevante? | Por que |
|---|---|---|
| **Tokenizacion** | Si | Necesitas entender como los modelos procesan texto |
| **Word Embeddings (Word2Vec, GloVe)** | Parcialmente | Reemplazados por embeddings contextuales, pero la intuicion es fundamental |
| **Transformers** | Absolutamente | Es la base de TODO (BERT, GPT, T5, LLMs) |
| **Fine-tuning BERT/RoBERTa** | Si | Para clasificacion con datos propios, mas barato que un LLM |
| **NER con modelos pequenos** | Si | A veces no necesitas un LLM para extraer entidades |
| **Sentence embeddings** | Si | Base de RAG y busqueda semantica |
| **Preprocesamiento clasico** | Poco | Los LLMs manejan texto crudo, pero util para modelos pequenos |
| **LLMs con prompting** | Si | Para generacion, resumen, QA general |

> **Regla practica:** Si tienes datos etiquetados y necesitas clasificar miles de textos al dia, fine-tunea un modelo pequeno (BERT). Si necesitas flexibilidad y la tarea es compleja/cambiante, usa un LLM con prompting. Si el costo importa mucho, modelos pequenos siempre ganan.

---

## Preprocesamiento de Texto

### Tokenizacion

Tokenizar = dividir texto en unidades que el modelo puede procesar.

**Tipos de Tokenizacion:**

```
Texto original: "Los transformers revolucionaron el NLP"

Word-level:     ["Los", "transformers", "revolucionaron", "el", "NLP"]
                Problema: vocabulario enorme, palabras desconocidas (OOV)

Character-level: ["L","o","s"," ","t","r","a","n","s","f","o","r","m","e","r","s",...]
                 Problema: secuencias muy largas, pierde significado

Subword (BPE):  ["Los", "transform", "##ers", "revolucion", "##aron", "el", "NLP"]
                Lo mejor de ambos mundos: vocabulario manejable, sin OOV
```

**Algoritmos de Subword Tokenization:**

| Algoritmo | Usado por | Como funciona (intuicion) |
|---|---|---|
| **BPE** (Byte Pair Encoding) | GPT, RoBERTa | Empieza con caracteres, va juntando pares mas frecuentes |
| **WordPiece** | BERT | Similar a BPE, pero usa likelihood en lugar de frecuencia |
| **SentencePiece** | T5, mBERT, LLaMA | Trata el texto como bytes crudos, no necesita pre-tokenizar |
| **Tiktoken** | GPT-3.5/4 | BPE optimizado de OpenAI |

```python
# Tokenizacion con HuggingFace
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer("Los transformers revolucionaron el NLP")

print(tokens)
# {'input_ids': [101, 2469, 19081, 2015, 13028, ...],
#  'attention_mask': [1, 1, 1, 1, 1, ...]}

# Ver los tokens
print(tokenizer.tokenize("Los transformers revolucionaron el NLP"))
# ['los', 'transform', '##ers', 'revolucion', '##aron', 'el', 'nl', '##p']
```

### Limpieza de Texto

| Tecnica | Descripcion | Cuando usarla |
|---|---|---|
| **Lowercasing** | Convertir a minusculas | Modelos clasicos. BERT uncased ya lo hace |
| **Stopwords** | Eliminar palabras comunes (el, la, de) | TF-IDF, modelos clasicos. NO para Transformers |
| **Stemming** | Reducir a raiz (corriendo -> corr) | Busqueda, modelos clasicos. Agresivo |
| **Lemmatization** | Reducir a lema (corriendo -> correr) | Mejor que stemming, mas lento |
| **Regex cleanup** | Eliminar URLs, emails, HTML, caracteres especiales | Casi siempre util |

```python
import re
import spacy

# Limpieza basica con regex
def clean_text(text):
    text = re.sub(r'http\S+', '', text)           # URLs
    text = re.sub(r'<.*?>', '', text)              # HTML tags
    text = re.sub(r'[^\w\s]', '', text)            # Puntuacion
    text = re.sub(r'\s+', ' ', text).strip()       # Espacios multiples
    return text

# Lemmatizacion con spaCy
nlp = spacy.load("es_core_news_sm")  # Modelo en espanol
doc = nlp("Los trabajadores estaban corriendo rapidamente")
lemmas = [token.lemma_ for token in doc]
# ['el', 'trabajador', 'estar', 'correr', 'rapidamente']
```

### Cuando Preprocesar vs Cuando Dejar al Modelo

```
Modelo clasico (TF-IDF + Logistic Regression):
  -> Preprocesar TODO: lowercase, stopwords, lemmatize, limpiar

Modelo Transformer (BERT, RoBERTa):
  -> Limpieza basica: URLs, HTML, caracteres rotos
  -> NO quitar stopwords, NO lemmatizar
  -> El tokenizer del modelo se encarga del resto

LLM (GPT-4, Claude):
  -> Minima limpieza: solo formatear bien el prompt
  -> El modelo maneja texto crudo perfectamente
```

---

## Word Embeddings

### El Problema de One-Hot Encoding

```
Vocabulario: ["gato", "perro", "casa", "coche"] (4 palabras)

One-hot encoding:
gato  = [1, 0, 0, 0]
perro = [0, 1, 0, 0]
casa  = [0, 0, 1, 0]
coche = [0, 0, 0, 1]

Problemas:
1. Dimension = tamano del vocabulario (50,000+ palabras = vectores de 50,000)
2. Todos los vectores son ortogonales (distancia identica)
3. "gato" esta tan lejos de "perro" como de "coche"
   -> No captura NINGUNA relacion semantica
```

### Word2Vec: La Revolucion

Idea central: **una palabra se define por las palabras que la rodean** (hipotesis distribucional).

```
"El gato se sento en la alfombra"
"El perro se sento en el sofa"

"gato" y "perro" aparecen en contextos similares
-> Sus vectores deberian ser similares

Word2Vec entrena una red neuronal simple para predecir:
- CBOW: dadas las palabras del contexto, predecir la palabra central
- Skip-gram: dada la palabra central, predecir las del contexto

Resultado: vectores densos de 100-300 dimensiones que capturan significado

Ejemplo de relaciones aprendidas:
  vector("rey") - vector("hombre") + vector("mujer") ≈ vector("reina")
  vector("Madrid") - vector("Espana") + vector("Francia") ≈ vector("Paris")
```

### GloVe: Global Vectors

Intuicion: en lugar de ventanas locales como Word2Vec, GloVe usa la **matriz global de co-ocurrencias** de todo el corpus. Si dos palabras aparecen juntas frecuentemente en el corpus, sus vectores seran cercanos.

Resultado similar a Word2Vec pero captura mejor relaciones globales.

### Limitacion Critica

```
"El banco del parque estaba mojado"     -> banco = asiento
"Fui al banco a sacar dinero"           -> banco = institucion financiera

Word2Vec/GloVe: UN solo vector para "banco"
No capturan el contexto -> misma representacion para significados diferentes

Solucion: Embeddings CONTEXTUALES (BERT, GPT)
Cada aparicion de "banco" tiene un vector DIFERENTE segun su contexto.
```

---

## Transformers: La Arquitectura Clave

### Attention Mechanism

La atencion es el mecanismo central que hace a los Transformers tan poderosos.

**Intuicion con Analogia:**

```
Imagina una base de datos:

Query (Q):  "Que busco?"       -> Lo que la palabra actual necesita saber
Key (K):    "Que tengo?"       -> Lo que cada palabra ofrece
Value (V):  "Que devuelvo?"    -> La informacion real que se pasa

Proceso:
1. Cada palabra genera sus vectores Q, K, V (multiplicando por matrices aprendidas)
2. Se calcula la "compatibilidad" entre el Query de una palabra y los Keys de todas
3. Se normalizan los scores con softmax (suman 1)
4. Se usa el score como peso para combinar los Values

Matematicamente:
  Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

  donde d_k es la dimension de los Keys (para estabilidad numerica)
```

**Self-Attention Paso a Paso:**

```
Frase: "El gato se sento en la alfombra"

Para la palabra "sento":
  Q_sento se compara con K de cada palabra:

  "El"        -> score: 0.02  (poco relevante)
  "gato"      -> score: 0.45  (quien se sento? el gato!)
  "se"        -> score: 0.05
  "sento"     -> score: 0.15  (atencion a si misma)
  "en"        -> score: 0.08
  "la"        -> score: 0.03
  "alfombra"  -> score: 0.22  (donde se sento? en la alfombra!)
                        -----
                         1.00  (softmax: suman 1)

  La representacion de "sento" se construye ponderando
  los Values de todas las palabras con estos scores.
  -> "sento" ahora "sabe" que el gato se sento en la alfombra.
```

### Multi-Head Attention

En lugar de una sola atencion, usar multiples "cabezas" en paralelo. Cada cabeza aprende a atender a diferentes aspectos:

```
Cabeza 1: Atiende a relaciones sintacticas (sujeto-verbo)
Cabeza 2: Atiende a relaciones semanticas (sinonimos)
Cabeza 3: Atiende a posiciones cercanas
Cabeza 4: Atiende a dependencias lejanas
...

Multi-Head Attention = Concatenar(cabeza_1, cabeza_2, ..., cabeza_h) * W_o

Tipicamente: 8-16 cabezas en modelos base, 32-96 en modelos grandes
```

### Arquitectura Transformer

```
                   ENCODER                              DECODER
                   (BERT-style)                         (GPT-style)

             [Input Embeddings]                    [Output Embeddings]
                     +                                      +
            [Positional Encoding]                 [Positional Encoding]
                     |                                      |
              .------+------.                        .------+------.
              |             |                        |             |
              | Multi-Head  |                        | Masked      |
              | Self-Attn   |                        | Self-Attn   |
              |             |                        |             |
              '------+------'                        '------+------'
                     |                                      |
              [Add & LayerNorm]                      [Add & LayerNorm]
                     |                                      |
              .------+------.                        .------+------.
              |             |                        | Cross-Attn  |
              | Feed Forward|                        | (al encoder)|
              |  Network    |                        '------+------'
              |             |                        [Add & LayerNorm]
              '------+------'                        .------+------.
                     |                               | Feed Forward|
              [Add & LayerNorm]                      '------+------'
                     |                               [Add & LayerNorm]
                     |                                      |
              (repetir Nx)                           (repetir Nx)
                     |                                      |
              [Representaciones]                     [Linear + Softmax]
              contextuales                           -> siguiente token
```

**Tres Variantes Principales:**

| Tipo | Modelo ejemplo | Que ve | Mejor para |
|---|---|---|---|
| **Encoder-only** | BERT, RoBERTa | Todas las palabras (bidireccional) | Clasificacion, NER, embeddings |
| **Decoder-only** | GPT, LLaMA | Solo palabras anteriores (causal) | Generacion de texto |
| **Encoder-Decoder** | T5, BART | Encoder bidireccional, decoder causal | Traduccion, resumen, text-to-text |

### Positional Encoding

Los Transformers procesan todas las palabras en paralelo (no secuencialmente como RNNs). Para que "sepan" el orden, se anaden positional encodings:

```
Embedding de "gato" en posicion 2:

embedding_final = embedding_palabra("gato") + positional_encoding(posicion=2)

Original Transformer: funciones seno/coseno
BERT: positional embeddings aprendidos
RoPE (LLaMA, modelos modernos): rotary position embeddings
```

### Por Que los Transformers Ganaron

```
RNNs/LSTMs (pre-2017):
- Procesan secuencialmente: lento, no paralelizable
- Dependencias lejanas se "olvidan" (vanishing gradients)
- Training: O(n) secuencial

Transformers:
- Procesan TODO en paralelo: GPU-friendly
- Self-attention conecta cualquier par de palabras directamente
- Training: O(1) paralelo para todas las posiciones
- Escalan enormemente bien con mas datos y compute
```

---

## Modelos Preentrenados Clave

### Tabla Comparativa

| Modelo | Tipo | Parametros | Uso tipico | Notas |
|---|---|---|---|---|
| **BERT-base** | Encoder | 110M | Clasificacion, NER, QA | El pionero, bidireccional |
| **RoBERTa** | Encoder | 125M | Igual que BERT, mejor rendimiento | BERT mejor entrenado |
| **DistilBERT** | Encoder | 66M | Lo mismo, pero mas rapido | 97% de BERT, 60% tamano |
| **DeBERTa-v3** | Encoder | 86-304M | SOTA en muchas tareas de entendimiento | El mejor encoder actual |
| **GPT-2** | Decoder | 124M-1.5B | Generacion de texto | El abuelo de ChatGPT |
| **T5-base** | Enc-Dec | 220M | Cualquier tarea como text-to-text | "Classify: ...", "Summarize: ..." |
| **FLAN-T5** | Enc-Dec | 80M-11B | Text-to-text con instruction tuning | T5 + fine-tuning con instrucciones |
| **Sentence-BERT** | Encoder | ~110M | Embeddings de frases, similitud | Optimizado para comparar textos |
| **BGE / E5** | Encoder | ~110-335M | Embeddings de frases (SOTA) | Los mejores embeddings actuales |

> **Consejo practico:** Para clasificacion, empieza con DeBERTa-v3. Para embeddings, usa BGE o E5. Para generacion, usa un LLM (GPT-4, Claude) o fine-tunea un modelo open source.

---

## HuggingFace Ecosystem

HuggingFace es el ecosistema central de NLP moderno. Tres componentes principales:

### transformers library

```python
# Forma rapida: Pipeline (inference en 3 lineas)
from transformers import pipeline

# Clasificacion de sentimiento
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
result = classifier("Este producto es excelente, lo recomiendo mucho")
# [{'label': '5 stars', 'score': 0.87}]

# NER
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
result = ner("Carlos trabaja en Google en Madrid")
# [{'entity_group': 'PER', 'word': 'Carlos', 'score': 0.99},
#  {'entity_group': 'ORG', 'word': 'Google', 'score': 0.98},
#  {'entity_group': 'LOC', 'word': 'Madrid', 'score': 0.99}]

# Question Answering
qa = pipeline("question-answering")
result = qa(question="Donde trabaja Carlos?",
            context="Carlos trabaja en Google en la oficina de Madrid")
# {'answer': 'Google', 'score': 0.95, 'start': 21, 'end': 27}
```

```python
# Forma completa: AutoModel + AutoTokenizer
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Tokenizar
inputs = tokenizer("Hello world", return_tensors="pt", padding=True, truncation=True)
print(inputs.keys())  # dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# outputs.last_hidden_state: (batch, seq_len, hidden_dim)
print(outputs.last_hidden_state.shape)  # torch.Size([1, 4, 768])
```

### datasets library

```python
from datasets import load_dataset

# Cargar dataset del Hub
dataset = load_dataset("imdb")
print(dataset)
# DatasetDict({
#     train: Dataset({features: ['text', 'label'], num_rows: 25000})
#     test: Dataset({features: ['text', 'label'], num_rows: 25000})
# })

# Procesar con map (eficiente, con cache)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length",
                     truncation=True, max_length=512)

tokenized = dataset.map(tokenize_function, batched=True)

# Cargar dataset local (CSV, JSON, Parquet)
dataset = load_dataset("csv", data_files="mis_datos.csv")
dataset = load_dataset("json", data_files="mis_datos.jsonl")
```

### Trainer API vs Training Loop Manual

```python
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Cargar modelo para clasificacion
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Definir metricas
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }

# Configurar entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,  # Mixed precision (si tienes GPU compatible)
)

# Crear Trainer y entrenar
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluar
results = trainer.evaluate()
print(results)
```

---

## Fine-Tuning de Modelos

### Cuando Fine-Tunear vs Usar un LLM con Prompting

| Factor | Fine-tuning modelo pequeno | LLM con prompting |
|---|---|---|
| **Datos etiquetados** | Necesitas 500+ ejemplos | 0-10 ejemplos (few-shot) |
| **Costo por prediccion** | Muy bajo (~0.001 USD/1K) | Alto (~0.01-0.10 USD/1K) |
| **Latencia** | Rapido (~10-50ms) | Lento (~500-2000ms) |
| **Personalizacion** | Alta (tu dominio) | Media (general) |
| **Mantenimiento** | Necesitas infra de ML | Solo API |
| **Volumen** | Miles-millones/dia | Cientos-miles/dia |
| **Tarea nueva** | Necesitas re-entrenar | Cambias el prompt |

> **Regla de oro:** Si tienes datos etiquetados + alto volumen + latencia importa, fine-tunea. Si la tarea cambia frecuentemente o tienes poco dato, usa un LLM.

### Fine-Tuning para Clasificacion

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Cargar modelo preentrenado con head de clasificacion
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5,                     # Numero de clases
    problem_type="single_label_classification"
)

# 2. El modelo ya tiene:
#    - Backbone preentrenado (DeBERTa)
#    - Classification head nuevo (inicializado random)

# 3. Entrenar con Trainer API (ver seccion anterior)
# El backbone se ajusta suavemente, el head aprende de cero
```

### Fine-Tuning para NER (Named Entity Recognition)

```python
from transformers import AutoModelForTokenClassification

# Labels NER tipicos (formato BIO)
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_list)
)

# Tokenizacion especial para NER:
# Si una palabra se divide en sub-tokens, alinear los labels
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(examples["tokens"], truncation=True,
                          is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignorar tokens especiales
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # Ignorar sub-tokens
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized
```

### LoRA y PEFT: Fine-Tuning Eficiente

Para modelos grandes, fine-tunear TODOS los parametros es costoso. LoRA (Low-Rank Adaptation) congela el modelo original y entrena pequenas matrices adicionales.

```
Modelo original:        LoRA:
W (768 x 768)          W (768 x 768) [CONGELADO]
= 589,824 params       + A (768 x 8) * B (8 x 768) [ENTRENABLES]
                        = 589,824 + 12,288 params
                        Solo entrenas ~2% de parametros!

Intuicion: los cambios necesarios para adaptar el modelo
a tu tarea se pueden representar con matrices de bajo rango.
```

```python
from peft import LoraConfig, get_peft_model, TaskType

# Configurar LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                        # Rango de las matrices (8-64)
    lora_alpha=32,              # Factor de escalado
    lora_dropout=0.1,
    target_modules=["query", "value"],  # Capas donde aplicar LoRA
)

# Aplicar LoRA al modelo
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 294,912 || all params: 109,482,240 || trainable%: 0.27%

# Entrenar normalmente con Trainer
# Solo se actualizan los parametros LoRA
```

---

## Tareas NLP Comunes en Consultoria

### Tabla de Tareas y Recomendaciones

| Tarea | Descripcion | Modelo recomendado | Complejidad |
|---|---|---|---|
| **Clasificacion de sentimiento** | Positivo/Negativo/Neutro | DeBERTa fine-tuned / LLM | Baja |
| **Categorizacion de texto** | Asignar categoria (ticket, email) | DeBERTa / BERT fine-tuned | Baja-Media |
| **Deteccion de spam** | Filtrar mensajes no deseados | DistilBERT fine-tuned | Baja |
| **NER** | Extraer nombres, organizaciones, fechas | BERT-NER / spaCy | Media |
| **Similitud semantica** | Comparar si dos textos dicen lo mismo | Sentence-BERT / BGE | Baja |
| **Question Answering** | Responder preguntas sobre un contexto | DeBERTa / LLM + RAG | Media-Alta |
| **Resumen** | Resumir documentos largos | T5 / BART / LLM | Media |
| **Traduccion** | Traducir entre idiomas | mBART / LLM | Media |
| **Extraccion de informacion** | Extraer campos de documentos | LLM con prompting | Media |

### Clasificacion de Texto

El caso mas comun en consultoria. Ejemplo: clasificar tickets de soporte.

```python
from transformers import pipeline

# Opcion 1: Zero-shot con LLM (sin datos etiquetados)
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

text = "La factura no refleja el descuento acordado"
labels = ["facturacion", "envio", "producto", "atencion"]

result = classifier(text, labels)
# {'labels': ['facturacion', 'atencion', 'producto', 'envio'],
#  'scores': [0.82, 0.09, 0.05, 0.04]}

# Opcion 2: Fine-tuning con datos (si tienes 500+ ejemplos etiquetados)
# Usar Trainer API (ver seccion anterior)
```

### NER (Named Entity Recognition)

```python
import spacy

# spaCy para NER rapido
nlp = spacy.load("es_core_news_lg")
doc = nlp("Apple lanzo el iPhone 15 en septiembre de 2023 en California")

for ent in doc.ents:
    print(f"{ent.text:20} -> {ent.label_}")
# Apple                -> ORG
# iPhone 15            -> MISC
# septiembre de 2023   -> DATE
# California           -> LOC
```

### Similitud Semantica

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

sentences = [
    "El gato esta en el tejado",
    "Hay un felino sobre la casa",
    "Python es un lenguaje de programacion",
]

embeddings = model.encode(sentences)
similarities = util.cos_sim(embeddings, embeddings)
print(similarities)
# tensor([[1.00, 0.82, 0.05],    gato-felino: alta similitud
#         [0.82, 1.00, 0.04],    ambas vs python: baja
#         [0.05, 0.04, 1.00]])
```

---

## Embeddings y Busqueda Semantica

### Sentence Embeddings

Los sentence embeddings convierten frases completas en vectores densos que capturan su significado.

```
"El cliente quiere devolver el producto"  ->  [0.12, -0.34, 0.87, ..., 0.23]  (768 dims)
"Quiero hacer una devolucion"             ->  [0.11, -0.31, 0.85, ..., 0.25]  (768 dims)

Cosine similarity entre estos dos: 0.89 (muy similares!)
```

### Vector Similarity

```python
import numpy as np

def cosine_similarity(a, b):
    """Mide la similitud del angulo entre dos vectores. Rango: [-1, 1]"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dot_product(a, b):
    """Producto punto. Mas rapido, pero depende de la magnitud."""
    return np.dot(a, b)

# Cosine similarity es la mas usada porque es invariante a la magnitud
# Dot product es mas rapido y funciona bien si los embeddings estan normalizados
```

### Vector Databases

Para buscar entre millones de embeddings, necesitas una base de datos vectorial con busqueda aproximada (ANN).

| Base de datos | Tipo | Mejor para | Costo |
|---|---|---|---|
| **FAISS** | Libreria (Meta) | Busqueda rapida en memoria | Gratis |
| **ChromaDB** | Embebida / Servidor | Prototipos, proyectos pequenos | Gratis |
| **Pinecone** | Cloud managed | Produccion, escalabilidad | Pago |
| **Weaviate** | Self-hosted / Cloud | Busqueda hibrida (vector + keyword) | Gratis / Pago |
| **Qdrant** | Self-hosted / Cloud | Alto rendimiento, filtros | Gratis / Pago |
| **pgvector** | Extension PostgreSQL | Si ya usas PostgreSQL | Gratis |

```python
# Ejemplo con ChromaDB
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
client = chromadb.Client()

collection = client.create_collection("documentos")

# Indexar documentos
docs = ["Politica de devoluciones...", "Horario de atencion...", "Precios..."]
embeddings = model.encode(docs).tolist()

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=["doc1", "doc2", "doc3"]
)

# Buscar
query = "Como devuelvo un producto?"
query_embedding = model.encode([query]).tolist()

results = collection.query(query_embeddings=query_embedding, n_results=2)
print(results["documents"])
# [['Politica de devoluciones...']]
```

### RAG (Retrieval-Augmented Generation)

Concepto basico: combinar busqueda semantica con un LLM.

```
1. Usuario pregunta: "Cual es la politica de devoluciones?"
2. Buscar en vector DB los documentos mas relevantes
3. Pasar los documentos + pregunta al LLM como contexto
4. El LLM genera una respuesta basada en los documentos

Ventajas:
- El LLM tiene informacion actualizada (no solo su entrenamiento)
- Respuestas basadas en TUS datos
- Reduccion de alucinaciones
```

> **Nota:** RAG se cubre en mayor profundidad en el LLM Playbook (seccion de RAG).

---

## Metricas NLP

### Clasificacion

Las metricas son las mismas de ML clasico:

| Metrica | Formula (intuicion) | Cuando usarla |
|---|---|---|
| **Accuracy** | Correctas / Total | Solo si clases balanceadas |
| **Precision** | TP / (TP + FP) | Cuando los falsos positivos son costosos |
| **Recall** | TP / (TP + FN) | Cuando los falsos negativos son costosos |
| **F1** | Media armonica de Precision y Recall | Metrica balanceada, la mas usada |
| **AUC-ROC** | Area bajo curva ROC | Comparar modelos, clasificacion binaria |

### NER: Entity-Level F1

```
Para NER se evalua a nivel de ENTIDAD, no de token:

Texto:     "Carlos Garcia trabaja en Microsoft"
Gold:      [B-PER, I-PER, O, O, B-ORG]
Predicted: [B-PER, I-PER, O, O, B-ORG]

Entidades gold:      {"Carlos Garcia": PER, "Microsoft": ORG}
Entidades predicted: {"Carlos Garcia": PER, "Microsoft": ORG}

Entity-level F1: ambas entidades correctas -> F1 = 1.0

Cuidado: si predices "Carlos" como PER (sin "Garcia"),
NO cuenta como match parcial. Es incorrecto.
```

### Metricas de Generacion de Texto

| Metrica | Que mide | Uso |
|---|---|---|
| **BLEU** | N-gram overlap con referencia | Traduccion (menos usado ahora) |
| **ROUGE** | Recall de n-grams de la referencia | Resumen (ROUGE-L es el mas usado) |
| **BERTScore** | Similitud semantica con embeddings | Mas robusto que BLEU/ROUGE |
| **Human evaluation** | Juicio humano | El gold standard, pero caro |

> **En la practica con LLMs:** las metricas automaticas (BLEU, ROUGE) son poco fiables para evaluar LLMs. Se usa mas evaluacion humana o LLM-as-a-judge (usar otro LLM para evaluar).

---

## Arbol de Decision: NLP Clasico vs LLM vs Fine-Tuning

```
                     Tienes datos etiquetados?
                    /                          \
                  Si                            No
                  |                              |
           Cuantos?                     La tarea es compleja?
          /        \                    /                  \
       <500       500+               Si                    No
        |           |                 |                     |
   LLM con      Volumen alto?    LLM con              Reglas /
   few-shot     /         \      prompting             Regex
     |        Si           No        |
     |         |            |        |
     |   Fine-tune     LLM con      |
     |   modelo        prompting    |
     |   pequeno           |        |
     |      |              |        |
     v      v              v        v

   Evaluar: el fine-tuned es mejor que el LLM?
   Si -> deploy modelo pequeno (mas barato, rapido)
   No -> usar LLM (mas flexible)

   Consideraciones adicionales:
   - Latencia critica (<100ms)? -> Modelo pequeno
   - Privacidad de datos? -> Modelo local/fine-tuned
   - Presupuesto limitado? -> Modelo pequeno
   - Tarea cambia frecuentemente? -> LLM con prompting
```

---

## Tips Practicos para NLP en Consultoria

1. **Empieza siempre con un baseline simple** - TF-IDF + Logistic Regression o zero-shot con un LLM. Te sorprendera lo lejos que llegan.

2. **La calidad de datos > tamano del modelo** - 500 ejemplos bien etiquetados con un BERT valen mas que 50,000 ejemplos ruidosos.

3. **Multilingue?** Usa modelos multilingues (mBERT, XLM-RoBERTa) o LLMs que ya son multilingues.

4. **Siempre evalua en datos reales del cliente**, no en benchmarks academicos. La distribucion del mundo real es diferente.

5. **Versionado de datos y modelos** desde el dia 1. Usa herramientas como DVC o MLflow.

6. **Para demos rapidas al cliente**, usa `pipeline()` de HuggingFace o un LLM via API. Luego optimiza.

7. **El preprocesamiento en espanol** tiene particularidades: acentos, caracteres especiales, modelos especificos (BETO, RoBERTa-BNE para espanol).
