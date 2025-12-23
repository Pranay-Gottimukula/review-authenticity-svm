# Review Authenticity SVM
Project based on institute course CS215: Mathematics for AI ML

# Deceptive Review Detection (From Scratch)

This project detects deceptive vs truthful reviews using a linear SVM trained
from scratch with TF-IDF (unigrams + bigrams) and stylometric features.

## Experiments & Analysis

The full experimental workflow, including feature engineering decisions,
threshold tuning, and detailed analysis, is documented in the **notebook**:

`notebook/review_authenticity_svm.ipynb`

It is good to start from there for a step-by-step explaination
of the model development.

## Implemented from Scratch
- TF-IDF with n-grams and document-frequency pruning
- Primal SVM with hinge loss and SGD
- Threshold tuning for F1 optimization

## Dataset
OTT Deceptive Opinion Dataset

## Final Metrics
F1        ≈ 0.68 (balanced precision–recall)

Precision ≈ 0.55

Recall    ≈ 0.90

## Notes
Focus is on correct ML pipeline, feature engineering, and interpretability.
