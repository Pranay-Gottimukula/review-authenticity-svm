import numpy as np
import pandas as pd
from tfidf import build_tfidf
from .stylometry import *

# ---------------- Stylometric Features ---------------- #

def remove_punctuation(text):
    return re.sub(r"[^\w\s]", "", text)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_extra_spaces(text)
    return text

def build_all_features_from_df(df):
    """
    Takes already-loaded dataframe:
        df["text"]
    Creates:
        df["clean_text"] → cleaned version
        Stylometric features from RAW text
        TF-IDF from cleaned text
    Returns:
        X, y, vocab
    """

    # Clean text
    df["clean_text"] = df["text"].apply(preprocess_text)

    raw_texts = df["text"].astype(str).tolist()
    clean_texts = df["clean_text"].astype(str).tolist()

    label_map = {"deceptive": 1, "truthful": -1}
    y = df["deceptive"].map(label_map).values

    # --- Stylometric features ---
    style = []
    for text in raw_texts:
        style.append([
            avg_word_length(text),
            sentence_count(text),
            punctuation_ratio(text),
            uppercase_ratio(text),
            lexical_diversity(text),
            repeated_word_ratio(text),
            exclamation_count(text)
        ])
    style_matrix = np.array(style, dtype=np.float32)

    # --- TF-IDF ---
    tfidf_matrix, vocab = build_tfidf(clean_texts)
    tfidf_matrix = np.array(tfidf_matrix, dtype=np.float32)

    return tfidf_matrix, style_matrix, y, vocab
