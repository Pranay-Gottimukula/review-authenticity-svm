# review-authenticity-svm
Project based on institute course CS215: Mathematics for AI ML, it segregates fake and real reviews

# Deceptive Review Detection (From Scratch)

This project detects deceptive vs truthful reviews using a linear SVM trained
from scratch with TF-IDF (unigrams + bigrams) and stylometric features.

## Implemented from Scratch
- TF-IDF with n-grams and document-frequency pruning
- Primal SVM with hinge loss and SGD
- Threshold tuning for F1 optimization

## Dataset
OTT Deceptive Opinion Dataset

## Final Metrics
F1 ≈ 0.68 (balanced precision–recall)

## Notes
Focus is on correct ML pipeline, feature engineering, and interpretability.
