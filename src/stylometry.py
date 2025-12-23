import re
import numpy as np
from collections import Counter

def avg_word_length(text):
    words = text.split()
    return (sum(len(w) for w in words) / len(words)) if words else 0

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
    return len(set(words)) / len(words) if words else 0

def repeated_word_ratio(text):
    words = text.split()
    counts = Counter(words)
    rep = sum(1 for w, c in counts.items() if c > 1)
    return rep / len(words) if words else 0

def exclamation_count(text):
    return text.count("!")