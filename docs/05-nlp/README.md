# NLP (Natural Language Processing)

## NLP in the Era of LLMs

With the arrival of ChatGPT and LLMs, the NLP landscape has changed drastically. But not everything is solved with an LLM.

### What Remains Relevant vs What LLMs Cover

| Area | Still relevant? | Why |
|---|---|---|
| **Tokenization** | Yes | You need to understand how models process text |
| **Word Embeddings (Word2Vec, GloVe)** | Partially | Replaced by contextual embeddings, but the intuition is fundamental |
| **Transformers** | Absolutely | It's the foundation of EVERYTHING (BERT, GPT, T5, LLMs) |
| **Fine-tuning BERT/RoBERTa** | Yes | For classification with your own data, cheaper than an LLM |
| **NER with small models** | Yes | Sometimes you don't need an LLM to extract entities |
| **Sentence embeddings** | Yes | Foundation of RAG and semantic search |
| **Classic preprocessing** | Little | LLMs handle raw text, but useful for small models |
| **LLMs with prompting** | Yes | For generation, summarization, general QA |

> **Practical rule:** If you have labeled data and need to classify thousands of texts per day, fine-tune a small model (BERT). If you need flexibility and the task is complex/changing, use an LLM with prompting. If cost matters a lot, small models always win.

---

## Text Preprocessing

### Tokenization

Tokenize = split text into units that the model can process.

**Tokenization Types:**

```
Original text: "Los transformers revolucionaron el NLP"

Word-level:     ["Los", "transformers", "revolucionaron", "el", "NLP"]
                Problem: huge vocabulary, unknown words (OOV)

Character-level: ["L","o","s"," ","t","r","a","n","s","f","o","r","m","e","r","s",...]
                 Problem: very long sequences, loses meaning

Subword (BPE):  ["Los", "transform", "##ers", "revolucion", "##aron", "el", "NLP"]
                Best of both worlds: manageable vocabulary, no OOV
```

**Subword Tokenization Algorithms:**

| Algorithm | Used by | How it works (intuition) |
|---|---|---|
| **BPE** (Byte Pair Encoding) | GPT, RoBERTa | Starts with characters, merges most frequent pairs |
| **WordPiece** | BERT | Similar to BPE, but uses likelihood instead of frequency |
| **SentencePiece** | T5, mBERT, LLaMA | Treats text as raw bytes, doesn't need pre-tokenization |
| **Tiktoken** | GPT-3.5/4 | OpenAI's optimized BPE |

```python
# Tokenization with HuggingFace
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer("Los transformers revolucionaron el NLP")

print(tokens)
# {'input_ids': [101, 2469, 19081, 2015, 13028, ...],
#  'attention_mask': [1, 1, 1, 1, 1, ...]}

# View the tokens
print(tokenizer.tokenize("Los transformers revolucionaron el NLP"))
# ['los', 'transform', '##ers', 'revolucion', '##aron', 'el', 'nl', '##p']
```

### Text Cleaning

| Technique | Description | When to use |
|---|---|---|
| **Lowercasing** | Convert to lowercase | Classic models. BERT uncased already does it |
| **Stopwords** | Remove common words (the, a, of) | TF-IDF, classic models. NOT for Transformers |
| **Stemming** | Reduce to root (running -> run) | Search, classic models. Aggressive |
| **Lemmatization** | Reduce to lemma (running -> run) | Better than stemming, slower |
| **Regex cleanup** | Remove URLs, emails, HTML, special characters | Almost always useful |

```python
import re
import spacy

# Basic cleaning with regex
def clean_text(text):
    text = re.sub(r'http\S+', '', text)           # URLs
    text = re.sub(r'<.*?>', '', text)              # HTML tags
    text = re.sub(r'[^\w\s]', '', text)            # Punctuation
    text = re.sub(r'\s+', ' ', text).strip()       # Multiple spaces
    return text

# Lemmatization with spaCy
nlp = spacy.load("es_core_news_sm")  # Spanish model
doc = nlp("Los trabajadores estaban corriendo rapidamente")
lemmas = [token.lemma_ for token in doc]
# ['el', 'trabajador', 'estar', 'correr', 'rapidamente']
```

### When to Preprocess vs When to Let the Model Handle It

```
Classic model (TF-IDF + Logistic Regression):
  -> Preprocess EVERYTHING: lowercase, stopwords, lemmatize, clean

Transformer model (BERT, RoBERTa):
  -> Basic cleaning: URLs, HTML, broken characters
  -> DO NOT remove stopwords, DO NOT lemmatize
  -> The model's tokenizer handles the rest

LLM (GPT-4, Claude):
  -> Minimal cleaning: just format the prompt well
  -> The model handles raw text perfectly
```

---

## Word Embeddings

### The One-Hot Encoding Problem

```
Vocabulary: ["cat", "dog", "house", "car"] (4 words)

One-hot encoding:
cat   = [1, 0, 0, 0]
dog   = [0, 1, 0, 0]
house = [0, 0, 1, 0]
car   = [0, 0, 0, 1]

Problems:
1. Dimension = vocabulary size (50,000+ words = 50,000 vectors)
2. All vectors are orthogonal (identical distance)
3. "cat" is as far from "dog" as from "car"
   -> Captures NO semantic relationship
```

### Word2Vec: The Revolution

Central idea: **a word is defined by the words that surround it** (distributional hypothesis).

```
"The cat sat on the rug"
"The dog sat on the sofa"

"cat" and "dog" appear in similar contexts
-> Their vectors should be similar

Word2Vec trains a simple neural network to predict:
- CBOW: given context words, predict the central word
- Skip-gram: given the central word, predict the context words

Result: dense vectors of 100-300 dimensions that capture meaning

Example of learned relationships:
  vector("king") - vector("man") + vector("woman") ≈ vector("queen")
  vector("Madrid") - vector("Spain") + vector("France") ≈ vector("Paris")
```

### GloVe: Global Vectors

Intuition: instead of local windows like Word2Vec, GloVe uses the **global co-occurrence matrix** of the entire corpus. If two words appear together frequently in the corpus, their vectors will be close.

Similar result to Word2Vec but captures global relationships better.

### Critical Limitation

```
"The bank of the park was wet"              -> bank = bench
"I went to the bank to withdraw money"      -> bank = financial institution

Word2Vec/GloVe: ONE single vector for "bank"
They don't capture context -> same representation for different meanings

Solution: CONTEXTUAL embeddings (BERT, GPT)
Each occurrence of "bank" has a DIFFERENT vector depending on its context.
```

---

## Transformers: The Key Architecture

### Attention Mechanism

Attention is the central mechanism that makes Transformers so powerful.

**Intuition with Analogy:**

```
Imagine a database:

Query (Q):  "What am I looking for?"  -> What the current word needs to know
Key (K):    "What do I have?"         -> What each word offers
Value (V):  "What do I return?"       -> The actual information that gets passed

Process:
1. Each word generates its Q, K, V vectors (by multiplying with learned matrices)
2. The "compatibility" between one word's Query and all Keys is computed
3. The scores are normalized with softmax (sum to 1)
4. The score is used as a weight to combine the Values

Mathematically:
  Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

  where d_k is the dimension of the Keys (for numerical stability)
```

**Self-Attention Step by Step:**

```
Sentence: "The cat sat on the rug"

For the word "sat":
  Q_sat is compared with K of each word:

  "The"    -> score: 0.02  (not very relevant)
  "cat"    -> score: 0.45  (who sat? the cat!)
  "sat"    -> score: 0.15  (attention to itself)
  "on"     -> score: 0.08
  "the"    -> score: 0.03
  "rug"    -> score: 0.22  (where did it sit? on the rug!)
                      -----
                       1.00  (softmax: sum to 1)

  The representation of "sat" is built by weighting
  the Values of all words with these scores.
  -> "sat" now "knows" that the cat sat on the rug.
```

### Multi-Head Attention

Instead of a single attention, use multiple "heads" in parallel. Each head learns to attend to different aspects:

```
Head 1: Attends to syntactic relationships (subject-verb)
Head 2: Attends to semantic relationships (synonyms)
Head 3: Attends to nearby positions
Head 4: Attends to long-range dependencies
...

Multi-Head Attention = Concatenate(head_1, head_2, ..., head_h) * W_o

Typically: 8-16 heads in base models, 32-96 in large models
```

### Transformer Architecture

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
              | Feed Forward|                        | (to encoder)|
              |  Network    |                        '------+------'
              |             |                        [Add & LayerNorm]
              '------+------'                        .------+------.
                     |                               | Feed Forward|
              [Add & LayerNorm]                      '------+------'
                     |                               [Add & LayerNorm]
                     |                                      |
              (repeat Nx)                            (repeat Nx)
                     |                                      |
              [Contextual                            [Linear + Softmax]
              representations]                       -> next token
```

**Three Main Variants:**

| Type | Example Model | What it sees | Best for |
|---|---|---|---|
| **Encoder-only** | BERT, RoBERTa | All words (bidirectional) | Classification, NER, embeddings |
| **Decoder-only** | GPT, LLaMA | Only previous words (causal) | Text generation |
| **Encoder-Decoder** | T5, BART | Encoder bidirectional, decoder causal | Translation, summarization, text-to-text |

### Positional Encoding

Transformers process all words in parallel (not sequentially like RNNs). So they "know" the order, positional encodings are added:

```
Embedding of "cat" at position 2:

final_embedding = word_embedding("cat") + positional_encoding(position=2)

Original Transformer: sine/cosine functions
BERT: learned positional embeddings
RoPE (LLaMA, modern models): rotary position embeddings
```

### Why Transformers Won

```
RNNs/LSTMs (pre-2017):
- Process sequentially: slow, not parallelizable
- Long-range dependencies are "forgotten" (vanishing gradients)
- Training: O(n) sequential

Transformers:
- Process EVERYTHING in parallel: GPU-friendly
- Self-attention connects any pair of words directly
- Training: O(1) parallel for all positions
- Scale enormously well with more data and compute
```

---

## Key Pretrained Models

### Comparison Table

| Model | Type | Parameters | Typical Use | Notes |
|---|---|---|---|---|
| **BERT-base** | Encoder | 110M | Classification, NER, QA | The pioneer, bidirectional |
| **RoBERTa** | Encoder | 125M | Same as BERT, better performance | Better-trained BERT |
| **DistilBERT** | Encoder | 66M | Same, but faster | 97% of BERT, 60% size |
| **DeBERTa-v3** | Encoder | 86-304M | SOTA on many understanding tasks | Best current encoder |
| **GPT-2** | Decoder | 124M-1.5B | Text generation | The grandfather of ChatGPT |
| **T5-base** | Enc-Dec | 220M | Any task as text-to-text | "Classify: ...", "Summarize: ..." |
| **FLAN-T5** | Enc-Dec | 80M-11B | Text-to-text with instruction tuning | T5 + fine-tuning with instructions |
| **Sentence-BERT** | Encoder | ~110M | Sentence embeddings, similarity | Optimized for comparing texts |
| **BGE / E5** | Encoder | ~110-335M | Sentence embeddings (SOTA) | Best current embeddings |

> **Practical advice:** For classification, start with DeBERTa-v3. For embeddings, use BGE or E5. For generation, use an LLM (GPT-4, Claude) or fine-tune an open source model.

---

## HuggingFace Ecosystem

HuggingFace is the central ecosystem of modern NLP. Three main components:

### transformers library

```python
# Quick way: Pipeline (inference in 3 lines)
from transformers import pipeline

# Sentiment classification
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
# Full approach: AutoModel + AutoTokenizer
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Tokenize
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

# Load dataset from the Hub
dataset = load_dataset("imdb")
print(dataset)
# DatasetDict({
#     train: Dataset({features: ['text', 'label'], num_rows: 25000})
#     test: Dataset({features: ['text', 'label'], num_rows: 25000})
# })

# Process with map (efficient, with cache)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length",
                     truncation=True, max_length=512)

tokenized = dataset.map(tokenize_function, batched=True)

# Load local dataset (CSV, JSON, Parquet)
dataset = load_dataset("csv", data_files="my_data.csv")
dataset = load_dataset("json", data_files="my_data.jsonl")
```

### Trainer API vs Manual Training Loop

```python
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Load model for classification
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }

# Configure training
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
    fp16=True,  # Mixed precision (if you have a compatible GPU)
)

# Create Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluate
results = trainer.evaluate()
print(results)
```

---

## Fine-Tuning Models

### When to Fine-Tune vs Use an LLM with Prompting

| Factor | Fine-tuning small model | LLM with prompting |
|---|---|---|
| **Labeled data** | Need 500+ examples | 0-10 examples (few-shot) |
| **Cost per prediction** | Very low (~0.001 USD/1K) | High (~0.01-0.10 USD/1K) |
| **Latency** | Fast (~10-50ms) | Slow (~500-2000ms) |
| **Customization** | High (your domain) | Medium (general) |
| **Maintenance** | Need ML infra | Just API |
| **Volume** | Thousands-millions/day | Hundreds-thousands/day |
| **New task** | Need to retrain | Change the prompt |

> **Rule of thumb:** If you have labeled data + high volume + latency matters, fine-tune. If the task changes frequently or you have little data, use an LLM.

### Fine-Tuning for Classification

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Load pretrained model with classification head
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5,                     # Number of classes
    problem_type="single_label_classification"
)

# 2. The model already has:
#    - Pretrained backbone (DeBERTa)
#    - New classification head (randomly initialized)

# 3. Train with Trainer API (see previous section)
# The backbone is gently adjusted, the head learns from scratch
```

### Fine-Tuning for NER (Named Entity Recognition)

```python
from transformers import AutoModelForTokenClassification

# Typical NER labels (BIO format)
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_list)
)

# Special tokenization for NER:
# If a word is split into sub-tokens, align the labels
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
                label_ids.append(-100)  # Ignore special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # Ignore sub-tokens
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized
```

### LoRA and PEFT: Efficient Fine-Tuning

For large models, fine-tuning ALL parameters is expensive. LoRA (Low-Rank Adaptation) freezes the original model and trains small additional matrices.

```
Original model:         LoRA:
W (768 x 768)          W (768 x 768) [FROZEN]
= 589,824 params       + A (768 x 8) * B (8 x 768) [TRAINABLE]
                        = 589,824 + 12,288 params
                        You only train ~2% of parameters!

Intuition: the changes needed to adapt the model
to your task can be represented with low-rank matrices.
```

```python
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                        # Rank of the matrices (8-64)
    lora_alpha=32,              # Scaling factor
    lora_dropout=0.1,
    target_modules=["query", "value"],  # Layers where to apply LoRA
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 294,912 || all params: 109,482,240 || trainable%: 0.27%

# Train normally with Trainer
# Only LoRA parameters are updated
```

---

## Common NLP Tasks in Consulting

### Task and Recommendation Table

| Task | Description | Recommended Model | Complexity |
|---|---|---|---|
| **Sentiment classification** | Positive/Negative/Neutral | DeBERTa fine-tuned / LLM | Low |
| **Text categorization** | Assign category (ticket, email) | DeBERTa / BERT fine-tuned | Low-Medium |
| **Spam detection** | Filter unwanted messages | DistilBERT fine-tuned | Low |
| **NER** | Extract names, organizations, dates | BERT-NER / spaCy | Medium |
| **Semantic similarity** | Compare if two texts say the same thing | Sentence-BERT / BGE | Low |
| **Question Answering** | Answer questions about a context | DeBERTa / LLM + RAG | Medium-High |
| **Summarization** | Summarize long documents | T5 / BART / LLM | Medium |
| **Translation** | Translate between languages | mBART / LLM | Medium |
| **Information extraction** | Extract fields from documents | LLM with prompting | Medium |

### Text Classification

The most common case in consulting. Example: classify support tickets.

```python
from transformers import pipeline

# Option 1: Zero-shot with LLM (no labeled data)
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

text = "La factura no refleja el descuento acordado"
labels = ["billing", "shipping", "product", "customer service"]

result = classifier(text, labels)
# {'labels': ['billing', 'customer service', 'product', 'shipping'],
#  'scores': [0.82, 0.09, 0.05, 0.04]}

# Option 2: Fine-tuning with data (if you have 500+ labeled examples)
# Use Trainer API (see previous section)
```

### NER (Named Entity Recognition)

```python
import spacy

# spaCy for quick NER
nlp = spacy.load("es_core_news_lg")
doc = nlp("Apple lanzo el iPhone 15 en septiembre de 2023 en California")

for ent in doc.ents:
    print(f"{ent.text:20} -> {ent.label_}")
# Apple                -> ORG
# iPhone 15            -> MISC
# septiembre de 2023   -> DATE
# California           -> LOC
```

### Semantic Similarity

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

sentences = [
    "The cat is on the roof",
    "There is a feline on top of the house",
    "Python is a programming language",
]

embeddings = model.encode(sentences)
similarities = util.cos_sim(embeddings, embeddings)
print(similarities)
# tensor([[1.00, 0.82, 0.05],    cat-feline: high similarity
#         [0.82, 1.00, 0.04],    both vs python: low
#         [0.05, 0.04, 1.00]])
```

---

## Embeddings and Semantic Search

### Sentence Embeddings

Sentence embeddings convert complete sentences into dense vectors that capture their meaning.

```
"The customer wants to return the product"   ->  [0.12, -0.34, 0.87, ..., 0.23]  (768 dims)
"I want to make a return"                    ->  [0.11, -0.31, 0.85, ..., 0.25]  (768 dims)

Cosine similarity between these two: 0.89 (very similar!)
```

### Vector Similarity

```python
import numpy as np

def cosine_similarity(a, b):
    """Measures the angle similarity between two vectors. Range: [-1, 1]"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dot_product(a, b):
    """Dot product. Faster, but depends on magnitude."""
    return np.dot(a, b)

# Cosine similarity is the most used because it's invariant to magnitude
# Dot product is faster and works well if embeddings are normalized
```

### Vector Databases

To search among millions of embeddings, you need a vector database with approximate search (ANN).

| Database | Type | Best for | Cost |
|---|---|---|---|
| **FAISS** | Library (Meta) | Fast in-memory search | Free |
| **ChromaDB** | Embedded / Server | Prototypes, small projects | Free |
| **Pinecone** | Cloud managed | Production, scalability | Paid |
| **Weaviate** | Self-hosted / Cloud | Hybrid search (vector + keyword) | Free / Paid |
| **Qdrant** | Self-hosted / Cloud | High performance, filters | Free / Paid |
| **pgvector** | PostgreSQL extension | If you already use PostgreSQL | Free |

```python
# Example with ChromaDB
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
client = chromadb.Client()

collection = client.create_collection("documents")

# Index documents
docs = ["Return policy...", "Business hours...", "Pricing..."]
embeddings = model.encode(docs).tolist()

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=["doc1", "doc2", "doc3"]
)

# Search
query = "How do I return a product?"
query_embedding = model.encode([query]).tolist()

results = collection.query(query_embeddings=query_embedding, n_results=2)
print(results["documents"])
# [['Return policy...']]
```

### RAG (Retrieval-Augmented Generation)

Basic concept: combine semantic search with an LLM.

```
1. User asks: "What is the return policy?"
2. Search the vector DB for the most relevant documents
3. Pass the documents + question to the LLM as context
4. The LLM generates an answer based on the documents

Advantages:
- The LLM has up-to-date information (not just its training)
- Answers based on YOUR data
- Reduction of hallucinations
```

> **Note:** RAG is covered in greater depth in the LLM Playbook (RAG section).

---

## NLP Metrics

### Classification

The metrics are the same as classic ML:

| Metric | Formula (intuition) | When to use |
|---|---|---|
| **Accuracy** | Correct / Total | Only if classes are balanced |
| **Precision** | TP / (TP + FP) | When false positives are costly |
| **Recall** | TP / (TP + FN) | When false negatives are costly |
| **F1** | Harmonic mean of Precision and Recall | Balanced metric, most commonly used |
| **AUC-ROC** | Area under ROC curve | Compare models, binary classification |

### NER: Entity-Level F1

```
For NER, evaluation is at the ENTITY level, not the token level:

Text:      "Carlos Garcia works at Microsoft"
Gold:      [B-PER, I-PER, O, O, B-ORG]
Predicted: [B-PER, I-PER, O, O, B-ORG]

Gold entities:      {"Carlos Garcia": PER, "Microsoft": ORG}
Predicted entities: {"Carlos Garcia": PER, "Microsoft": ORG}

Entity-level F1: both entities correct -> F1 = 1.0

Beware: if you predict "Carlos" as PER (without "Garcia"),
it does NOT count as a partial match. It's incorrect.
```

### Text Generation Metrics

| Metric | What it measures | Use |
|---|---|---|
| **BLEU** | N-gram overlap with reference | Translation (less used now) |
| **ROUGE** | Recall of n-grams from the reference | Summarization (ROUGE-L is the most used) |
| **BERTScore** | Semantic similarity with embeddings | More robust than BLEU/ROUGE |
| **Human evaluation** | Human judgment | The gold standard, but expensive |

> **In practice with LLMs:** automatic metrics (BLEU, ROUGE) are unreliable for evaluating LLMs. Human evaluation or LLM-as-a-judge (using another LLM to evaluate) is used more.

---

## Decision Tree: Classic NLP vs LLM vs Fine-Tuning

```
                     Do you have labeled data?
                    /                          \
                  Yes                           No
                  |                              |
           How much?                     Is the task complex?
          /        \                    /                  \
       <500       500+               Yes                   No
        |           |                 |                     |
   LLM with      High volume?    LLM with              Rules /
   few-shot     /         \      prompting              Regex
     |        Yes          No        |
     |         |            |        |
     |   Fine-tune     LLM with     |
     |   small         prompting    |
     |   model              |       |
     |      |               |       |
     v      v               v       v

   Evaluate: is the fine-tuned better than the LLM?
   Yes -> deploy small model (cheaper, faster)
   No  -> use LLM (more flexible)

   Additional considerations:
   - Critical latency (<100ms)? -> Small model
   - Data privacy? -> Local/fine-tuned model
   - Limited budget? -> Small model
   - Task changes frequently? -> LLM with prompting
```

---

## Practical Tips for NLP in Consulting

1. **Always start with a simple baseline** - TF-IDF + Logistic Regression or zero-shot with an LLM. You'll be surprised how far they go.

2. **Data quality > model size** - 500 well-labeled examples with a BERT are worth more than 50,000 noisy examples.

3. **Multilingual?** Use multilingual models (mBERT, XLM-RoBERTa) or LLMs that are already multilingual.

4. **Always evaluate on real client data**, not on academic benchmarks. The real-world distribution is different.

5. **Data and model versioning** from day 1. Use tools like DVC or MLflow.

6. **For quick client demos**, use HuggingFace `pipeline()` or an LLM via API. Then optimize.

7. **Spanish text preprocessing** has particularities: accents, special characters, specific models (BETO, RoBERTa-BNE for Spanish).
