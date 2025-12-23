import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from features import build_all_features_from_df
from svm import PrimalSVM

# load data
df = pd.read_csv("data/deceptive-opinion.csv")

# build features
X_tfidf, X_style, y, vocab = build_all_features_from_df(df)

# split
X_tfidf_train, X_tfidf_test, X_style_train, X_style_test, y_train, y_test = train_test_split(
    X_tfidf, X_style, y, test_size=0.2, stratify=y, random_state=42
)

# normalize
X_tfidf_train = normalize(X_tfidf_train)
X_tfidf_test = normalize(X_tfidf_test)

scaler = StandardScaler()
X_style_train = scaler.fit_transform(X_style_train)
X_style_test = scaler.transform(X_style_test)

X_train = np.hstack([X_tfidf_train, X_style_train])
X_test = np.hstack([X_tfidf_test, X_style_test])

# train
svm = PrimalSVM(C=1.0, lr=0.01, epochs=50, lr_decay=1e-4)
svm.fit(X_train, y_train)

# evaluate
y_pred = svm.predict(X_test, threshold=0.0)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
