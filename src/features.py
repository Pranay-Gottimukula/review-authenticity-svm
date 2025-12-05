import numpy as np
import pandas as pd
import re
from collections import Counter
from tfidf import build_tfidf   # ← your existing function


# 1. Helper functions for stylometric features

def avg_word_length(text):
    words = text.split()
    if not words:
        return 0
    return sum(len(w) for w in words) / len(words)


def sentence_count(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)


def punctuation_ratio(text):
    punct = sum(1 for ch in text if ch in ".,!?;:")
    return punct / max(len(text), 1)


def uppercase_ratio(text):
    upper = sum(1 for ch in text if ch.isupper())
    return upper / max(len(text), 1)


def lexical_diversity(text):
    words = text.split()
    if not words:
        return 0
    return len(set(words)) / len(words)


def repeated_word_ratio(text):
    words = text.split()
    if not words:
        return 0
    counts = Counter(words)
    repeated = sum(1 for w, c in counts.items() if c > 1)
    return repeated / len(words)


def exclamation_count(text):
    return text.count("!")


# 2. Main feature builder

def build_features(cleaned_csv_path):
    """
    Loads cleaned dataset and builds:
    - TF-IDF features (from clean_text)
    - Stylometric features (from original text)

    Returns:
        X_final  -> NumPy array of features
        y        -> labels (0/1 for deceptive)
        vocab    -> vocabulary used for TF-IDF
    """

    # Load cleaned dataset
    df = pd.read_csv(cleaned_csv_path)

    # Extract columns
    corpus = df["clean_text"].astype(str).tolist()
    original_texts = df["text"].astype(str).tolist()
    label_map = {"deceptive": 1, "truthful": -1}
    labels = df["deceptive"].map(label_map).values

    # Build TF-IDF matrix
    tfidf_matrix, vocab = build_tfidf(corpus)
    tfidf_matrix = np.array(tfidf_matrix)

    # Build stylometric features
    feature_list = []

    for text in original_texts:
        feature_list.append([
            avg_word_length(text),
            sentence_count(text),
            punctuation_ratio(text),
            uppercase_ratio(text),
            lexical_diversity(text),
            repeated_word_ratio(text),
            exclamation_count(text),
        ])

    style_matrix = np.array(feature_list, dtype=np.float32)

    # Combine TF-IDF + Stylometric features
    X_final = np.hstack([tfidf_matrix, style_matrix])

    return X_final, labels, vocab

# Shape of X_final --> (number of reviews, vocab size + 7(stylometric features))

X, y, vocab = build_features("../data/clean/deceptive-opinion-clean.csv")

print(X)
print(y)