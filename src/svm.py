import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class PrimalSVM:
  def __init__(self, C=1.0, lr=0.01, epochs=50, shuffle=True, lr_decay=1e-4, verbose=False) -> None:
      "Declare all params"
      self.C = float(C)
      self.lr = float(lr)
      self.epochs = int(epochs)
      self.shuffle = shuffle
      self.lr_decay = float(lr_decay)
      self.verbose = verbose
      self.w = None
      self.b = 0.0
      self.t = 0

  def _get_lr(self):
    # Adding decay learning rate
    return self.lr / (1 + self.lr_decay * self.t)

  def fit(self, X, y):
    n_samples, n_features = X.shape

    self.w = np.zeros(n_features)
    self.b = 0.0
    self.t = 0

    for epoch in range(self.epochs):
      if self.shuffle:
        perm = np.random.permutation(n_samples)
        X_perm = X[perm]
        y_perm = y[perm]
      else:
        X_perm = X
        y_perm = y

      for i in range(n_samples):
        xi = X_perm[i]
        yi = y_perm[i]
        lr = self._get_lr()
        # Dataset has perfectly balanced classes
        margin = y_perm[i] * (np.dot(self.w, X_perm[i]) + self.b)

        if margin >= 1.0:
          # w = w - lr * w  (equivalently: (1 - lr) * w)
          self.w = self.w - lr * self.w
          # b unchanged

        else:
          # w = w - lr * (w - C*y_i*x_i) = (1 - lr)w + lr*C*y_i*x_i
          self.w = self.w - lr * self.w + lr * self.C * yi * xi
          # b = b - lr * (-C*y_i) = b + lr * C * y_i
          self.b = self.b + lr * self.C * yi

        self.t += 1

    if self.verbose:
      # quick epoch summary (train accuracy)
      y_pred = self.predict(X)
      acc = accuracy_score(y, y_pred)
      print(f"Epoch {epoch+1}/{self.epochs}, lr={lr:.5f}, train_acc={acc:.4f}")

    return self

  def decision_function(self, X):
      return np.dot(X, self.w) + self.b

  # Added threshold to increase recall because the model is hard to accept something is right
  def predict(self, X, threshold=0.0):
      scores = self.decision_function(X)
      return np.where(scores >= threshold, 1, -1)

  def score(self, X, y):
      y_pred = self.predict(X)
      return accuracy_score(y, y_pred)

  def get_params(self):
      return {'w': self.w, 'b': self.b}
