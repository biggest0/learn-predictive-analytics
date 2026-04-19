# =============================================================================
# TEXT PREPROCESSING CHEATSHEET
# =============================================================================

import re
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Natural language processing is a field of artificial intelligence.",
    "Text preprocessing cleans and prepares raw text for analysis.",
    "Tokenization splits text into words or sentences.",
]

# -----------------------------------------------------------------------------
# 1. LOWERCASING & PUNCTUATION REMOVAL
# -----------------------------------------------------------------------------
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)   # remove punctuation/numbers
    return text

cleaned = [clean(t) for t in corpus]
print("Cleaned:\n", cleaned)

# -----------------------------------------------------------------------------
# 2. TOKENIZATION
#    word_tokenize  : splits into individual words
#    sent_tokenize  : splits into sentences
# -----------------------------------------------------------------------------
tokens = [word_tokenize(t) for t in cleaned]
print("\nWord Tokens:\n", tokens[0])

sentences = sent_tokenize(corpus[1])
print("\nSentence Tokens:\n", sentences)

# -----------------------------------------------------------------------------
# 3. STOPWORD REMOVAL
#    Remove common words (the, is, a, ...) that carry little meaning
# -----------------------------------------------------------------------------
stop_words = set(stopwords.words('english'))
filtered = [[w for w in t if w not in stop_words] for t in tokens]
print("\nAfter Stopword Removal:\n", filtered[0])

# -----------------------------------------------------------------------------
# 4. STEMMING  (chops suffix — fast but crude)
#    running → run  |  studies → studi
# -----------------------------------------------------------------------------
stemmer = PorterStemmer()
stemmed = [[stemmer.stem(w) for w in t] for t in filtered]
print("\nStemmed:\n", stemmed[0])

# -----------------------------------------------------------------------------
# 6. N-GRAMS  (sequences of n consecutive words)
#    Bigram (n=2) : ("quick", "brown"), ("brown", "fox") ...
#    Trigram (n=3): ("quick", "brown", "fox") ...
# -----------------------------------------------------------------------------
bigrams  = list(ngrams(filtered[0], 2))
trigrams = list(ngrams(filtered[0], 3))
print("\nBigrams:\n",  bigrams)
print("\nTrigrams:\n", trigrams)

# -----------------------------------------------------------------------------
# 7. BAG OF WORDS  (CountVectorizer)
#    Counts how many times each word appears per document
# -----------------------------------------------------------------------------
cv = CountVectorizer()
bow = cv.fit_transform(corpus)
print("\nBag of Words (vocab):\n", cv.get_feature_names_out())
print(pd.DataFrame(bow.toarray(), columns=cv.get_feature_names_out()))

# With n-grams via CountVectorizer
cv_ngram = CountVectorizer(ngram_range=(1, 2))   # unigrams + bigrams
bow_ngram = cv_ngram.fit_transform(corpus)
print("\nBoW with Bigrams (vocab):\n", cv_ngram.get_feature_names_out())


# -----------------------------------------------------------------------------
# KEY CONCEPTS (exam reminders)
# -----------------------------------------------------------------------------
# Tokenization    : split text into words (word_tokenize) or sentences (sent_tokenize)
# Stopwords       : remove common low-info words (the, is, a ...)
# Stemming        : chop suffix → fast, may produce non-words (studi)
# Lemmatization   : map to base dictionary form → slower, cleaner (study)
# N-gram          : sequence of n words — captures context (bigram, trigram)
# Bag of Words    : word count matrix — ignores word order
# TF-IDF          : weighted BoW — rare words get higher scores
# ngram_range     : (1,1)=unigrams only, (1,2)=unigrams+bigrams, (2,2)=bigrams only
