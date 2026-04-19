# =============================================================================
# TRANSFORMER WORKFLOWS CHEATSHEET (guided-style tasks)
#   Word2Vec → BERT tokenization → embeddings → pipelines (NER, QA, classification)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from pandas import DataFrame
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel, BertTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util

# -----------------------------------------------------------------------------
# 1. WORD2VEC — train word embeddings from scratch on your own corpus
#    Produces a fixed vector per word (static, not context-aware)
# -----------------------------------------------------------------------------
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["a", "dog", "barked", "at", "the", "stranger"],
    ["i", "had", "an", "apple", "for", "lunch"],
    ["i", "ate", "an", "orange"],
]

w2v = Word2Vec(
    sentences,
    vector_size=50,   # dimensions of each word vector
    window=3,         # context window size (words to left/right)
    min_count=1,      # ignore words appearing < this many times
    workers=1,
)

vector_cat    = w2v.wv["cat"]
vector_dog    = w2v.wv["dog"]
vector_orange = w2v.wv["orange"]
vector_apple  = w2v.wv["apple"]
print("cat vector (first 5):", vector_cat[:5])

# Cluster the word vectors — animals should group separately from fruits
vectors = [vector_cat, vector_dog, vector_orange, vector_apple]
kmeans  = KMeans(n_clusters=2, random_state=142).fit(DataFrame(vectors))
print("Labels [cat, dog, orange, apple]:", kmeans.labels_)

# -----------------------------------------------------------------------------
# 2. BERT TOKENIZATION
#    Splits text into subword tokens + adds [CLS] start and [SEP] end
# -----------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained("bert-base-uncased")

encoded   = tok("I live in Mississauga", add_special_tokens=True)
input_ids = encoded["input_ids"]
tokens    = tok.convert_ids_to_tokens(input_ids)

for i in range(len(tokens)):
    print(tokens[i] + ": " + str(input_ids[i]))
# Mississauga gets split into subwords like "mississ", "##auga"
# ## prefix = continuation of previous subword

# -----------------------------------------------------------------------------
# 3. STATIC vs CONTEXTUAL EMBEDDINGS
#    Static   : same vector no matter the context (like Word2Vec)
#    Contextual: BERT's output — vector changes based on surrounding words
#    Example: "board" in "snow board" vs "board of directors" → different vectors
# -----------------------------------------------------------------------------
model = AutoModel.from_pretrained("bert-base-uncased")

text    = "snow board"
encoded = tok(text, return_tensors="pt", add_special_tokens=True)
ids     = encoded["input_ids"][0]
toks    = tok.convert_ids_to_tokens(ids)

with torch.no_grad():
    # Static — lookup table, no context
    static_emb = model.embeddings.word_embeddings(ids)

    # Contextual — full BERT forward pass
    outputs     = model(**encoded)
    contextual  = outputs.last_hidden_state[0]   # shape: [seq_len, 768]

print("Static shape:",      static_emb.shape)
print("Contextual shape:",  contextual.shape)

# -----------------------------------------------------------------------------
# 4. PADDING & TRUNCATION — align sequences to same length for batching
# -----------------------------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoded   = tokenizer(
    ["I love transformers", "Transformers are powerful models"],
    padding="max_length",
    max_length=8,
    truncation=True,
    return_tensors="pt",
)
print("Input IDs:\n",       encoded["input_ids"])
print("Attention Mask:\n",  encoded["attention_mask"])
# attention_mask: 1 for real tokens, 0 for padding — tells BERT what to ignore

# -----------------------------------------------------------------------------
# 5. SENTENCE EMBEDDINGS  →  CLUSTERING
#    Use [CLS] (first token) from BERT's last hidden state as a sentence vector
# -----------------------------------------------------------------------------
reviews = [
    "The movie was absolutely wonderful!",
    "I really enjoyed the film.",
    "The movie was terrible.",
]
inputs = tokenizer(reviews, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

sentence_vectors = outputs.last_hidden_state[:, 0, :]   # [CLS] token per sentence
kmeans = KMeans(n_clusters=2, random_state=142).fit(DataFrame(sentence_vectors))
print("Review cluster labels:", kmeans.labels_)

# -----------------------------------------------------------------------------
# 6. SENTENCE SIMILARITY  (sentence-transformers, ready to use)
#    Cosine similarity: 1=identical meaning, 0=unrelated
# -----------------------------------------------------------------------------
st_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = st_model.encode([
    "The student passed the exam.",
    "The learner successfully completed the test.",
    "I had pop tarts for dessert.",
])

print("sim(1,2) — paraphrase:", util.cos_sim(embeddings[0], embeddings[1]))
print("sim(1,3) — unrelated :", util.cos_sim(embeddings[0], embeddings[2]))

# -----------------------------------------------------------------------------
# 7. TEXT CLASSIFICATION PIPELINE (sentiment analysis)
# -----------------------------------------------------------------------------
classifier = pipeline("text-classification",
                       model="distilbert-base-uncased-finetuned-sst-2-english")
print(classifier("The movie was absolutely wonderful!"))
# → [{'label': 'POSITIVE', 'score': 0.99...}]

# -----------------------------------------------------------------------------
# 8. NAMED ENTITY RECOGNITION (NER) PIPELINE
#    Identifies PERSON, ORG, LOCATION, etc. in text
# -----------------------------------------------------------------------------
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
entities = ner("Pat McGee teaches at BCIT in Vancouver.")
for e in entities:
    print(e["entity_group"], ":", e["word"])
# → PER: Pat McGee  |  ORG: BCIT  |  LOC: Vancouver

# -----------------------------------------------------------------------------
# 9. QUESTION ANSWERING PIPELINE
#    Extracts an answer span from a given context paragraph
# -----------------------------------------------------------------------------
qa = pipeline("question-answering",
               model="distilbert-base-cased-distilled-squad")
result = qa(
    question="When will the exam take place?",
    context="The exam will take place on Friday at 10 a.m. in room DTC-545.",
)
print(result)
# → {'answer': 'Friday at 10 a.m.', 'score': ..., 'start': ..., 'end': ...}

# -----------------------------------------------------------------------------
# KEY CONCEPTS (exam reminders)
# -----------------------------------------------------------------------------
# Word2Vec       : train your own static word vectors from a corpus
# Static emb     : same vector regardless of context (Word2Vec, BERT input layer)
# Contextual emb : BERT output — vector changes based on surrounding words
# [CLS] token    : first token — summary vector for whole sentence
# [SEP] token    : separator between sentences or end of input
# ## prefix      : subword continuation (e.g. "mississ", "##auga")
# Attention mask : 1 = real token, 0 = padding (tells model to ignore)
# Padding        : add zeros to make all sequences equal length for batching
# Truncation     : cut sequences longer than max_length
# cos_sim        : cosine similarity between embeddings — 1=same, 0=unrelated
# pipeline()     : high-level API for common tasks (classification, NER, QA)
