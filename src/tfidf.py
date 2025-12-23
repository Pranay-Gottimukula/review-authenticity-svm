import numpy as np
import math
from collections import Counter


def get_unigrams_and_bigrams(tokens):
    terms = []

    # unigrams
    terms.extend(tokens)

    # bigrams
    for i in range(len(tokens) - 1):
        terms.append(tokens[i] + "_" + tokens[i + 1])

    return terms

def get_tf_vector(doc, word_to_index, vocab_size):
    vec = np.zeros(vocab_size, dtype=np.float32)
    tokens = doc.split()
    # Add bigrams
    terms = get_unigrams_and_bigrams(tokens)

    for term in terms:
        if term in word_to_index:
            vec[word_to_index[term]] += 1

    return vec


def compute_idf_vector(corpus, word_to_index, vocab_size):
    df = np.zeros(vocab_size, dtype=np.int32)

    for doc in corpus:
        tokens = doc.split()
        unique_terms = set(get_unigrams_and_bigrams(tokens))
        for term in unique_terms:
          if term in word_to_index:
              df[word_to_index[term]] += 1

    num_docs = len(corpus)
    idf = np.zeros(vocab_size, dtype=np.float32)

    for word, idx in word_to_index.items():
        idf[idx] = math.log((num_docs + 1) / (df[idx] + 1)) + 1

    return idf


def build_tfidf(corpus):
    min_df = 3
    max_df = 0.85 

    vocab = []
    word_to_index = {}
    term_df = {}

    for term, df in term_df.items():
        if df >= min_df and df <= max_df * num_docs:
            word_to_index[term] = len(vocab)
            vocab.append(term)

    for doc in corpus:
      tokens = doc.split()
      terms = get_unigrams_and_bigrams(tokens)

      for term in terms:
          if term not in word_to_index:
              word_to_index[term] = len(vocab)
              vocab.append(term)

    vocab_size = len(vocab)
    num_docs = len(corpus)

    tf_matrix = np.zeros((num_docs, vocab_size), dtype=np.float32)
    for i, doc in enumerate(corpus):
        tf_matrix[i] = get_tf_vector(doc, word_to_index, vocab_size)

    idf_vector = compute_idf_vector(corpus, word_to_index, vocab_size)

    tfidf_matrix = tf_matrix * idf_vector

    return tfidf_matrix, vocab
