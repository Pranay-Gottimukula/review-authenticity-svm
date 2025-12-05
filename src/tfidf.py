import numpy as np
import math
from collections import Counter


# Build TF vector efficiently
def get_tf_vector(doc, word_to_index, vocab_size):
    vec = np.zeros(vocab_size, dtype=np.float32)
    
    for word in doc.split():
        if word in word_to_index:
            vec[word_to_index[word]] += 1
    
    return vec


# Build IDF vector efficiently
def compute_idf_vector(corpus, word_to_index, vocab_size):
    df = np.zeros(vocab_size, dtype=np.int32)

    # Count in how many docs each word appears
    for doc in corpus:
        unique_words = set(doc.split())
        for word in unique_words:
            if word in word_to_index:
                df[word_to_index[word]] += 1

    num_docs = len(corpus)
    idf = np.zeros(vocab_size, dtype=np.float32)

    for word, idx in word_to_index.items():
        idf[idx] = math.log((num_docs + 1) / (df[idx] + 1)) + 1

    return idf


# Main TF-IDF builder
def build_tfidf(corpus):
    # Build vocabulary and index map
    vocab = []
    word_to_index = {}

    for doc in corpus:
        for word in doc.split():
            if word not in word_to_index:
                word_to_index[word] = len(vocab)
                vocab.append(word)

    vocab_size = len(vocab)
    num_docs = len(corpus)

    # Build TF matrix
    tf_matrix = np.zeros((num_docs, vocab_size), dtype=np.float32)

    for i, doc in enumerate(corpus):
        tf_matrix[i] = get_tf_vector(doc, word_to_index, vocab_size)

    # Build IDF vector
    idf_vector = compute_idf_vector(corpus, word_to_index, vocab_size)

    # Compute TF-IDF (full vectorized multiply)
    tfidf_matrix = tf_matrix * idf_vector

    return tfidf_matrix, vocab
