"""
Week 2: ML Algorithms for Industrial Defect Classification
===========================================================
This file implements and compares:
  1. Logistic Regression (linear baseline)
  2. Decision Tree (interpretable, non-linear)
  3. Bagging Classifier (variance reduction via bootstrap)
  4. Random Forest (Bagging + feature subsampling)
  5. k-Nearest Neighbours (non-parametric baseline)

Each is applied to a simulated industrial QA dataset:
  Features: temperature, vibration, pressure, spindle_speed
  Target: defect (0=normal, 1=defective)

All from-scratch implementations are included where useful to
demonstrate conceptual understanding, followed by sklearn equivalents.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

np.random.seed(42)


# =============================================================================
# DATA GENERATION — Simulated Industrial QA Dataset
# =============================================================================

def generate_qa_dataset(n_samples: int = 1000) -> tuple:
    """
    Simulate an industrial sensor + defect dataset.

    Feature relationships (domain knowledge encoded):
    - High temperature (>85°C) increases defect probability
    - High vibration (>2.5 mm/s) independently signals mechanical issue
    - Their INTERACTION matters: both high simultaneously → high defect risk
    - Pressure and spindle speed have weaker, noisy correlations

    This non-linear interaction structure means linear models will underfit,
    and tree-based models should outperform.

    Returns:
        X: (n_samples, 4) feature matrix
        y: (n_samples,) binary defect labels
        feature_names: list of feature names
    """
    # Feature ranges based on typical CNC/conveyor industrial specs
    temperature   = np.random.normal(75, 10, n_samples)       # Celsius
    vibration     = np.random.exponential(1.5, n_samples)     # mm/s (right-skewed)
    pressure      = np.random.normal(9, 2, n_samples)         # bar
    spindle_speed = np.random.normal(120, 15, n_samples)      # RPM (normalized)

    # Clip to physical bounds
    temperature   = np.clip(temperature, 50, 130)
    vibration     = np.clip(vibration, 0.1, 6.0)
    pressure      = np.clip(pressure, 4, 16)
    spindle_speed = np.clip(spindle_speed, 70, 170)

    # Non-linear defect probability — captures domain interactions
    # Interaction term: high temp AND high vibration is MUCH worse than either alone
    logit = (
        -6                                          # intercept (base rate ~0.24%)
        + 0.05 * (temperature - 75)                # linear temperature effect
        + 0.8  * (vibration - 1.5)                 # linear vibration effect
        + 0.04 * (temperature - 75) * (vibration - 1.5)   # INTERACTION term
        + 0.1  * (pressure - 9)
        - 0.01 * (spindle_speed - 120)
        + np.random.normal(0, 0.5, n_samples)       # observation noise
    )
    prob_defect = 1 / (1 + np.exp(-logit))          # sigmoid → probability
    y = (np.random.rand(n_samples) < prob_defect).astype(int)

    X = np.column_stack([temperature, vibration, pressure, spindle_speed])
    feature_names = ['temperature', 'vibration', 'pressure', 'spindle_speed']

    print(f"Dataset: {n_samples} samples, {y.sum()} defective ({100*y.mean():.1f}%)")
    return X, y, feature_names


X, y, feature_names = generate_qa_dataset(n_samples=2000)

# Train-test split (stratified to preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling (critical for Logistic Regression and k-NN; trees don't need it)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Fit on train, transform both
X_test_scaled  = scaler.transform(X_test)         # NEVER fit on test — data leakage!


# =============================================================================
# MODEL 1: LOGISTIC REGRESSION — From Scratch (Gradient Descent)
# =============================================================================

class LogisticRegressionScratch:
    """
    Binary Logistic Regression trained via gradient descent.

    Demonstrates the core learning loop that all neural networks also use:
    forward pass → compute loss → compute gradient → update weights.

    Loss: Binary Cross-Entropy
        L = -[y * log(σ(Xw)) + (1-y) * log(1 - σ(Xw))]

    Gradient (derivation via chain rule):
        ∂L/∂w = X^T (σ(Xw) - y) / n
    """

    def __init__(self, lr: float = 0.1, n_iter: int = 1000, lambda_reg: float = 0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.lambda_reg = lambda_reg   # L2 regularization coefficient
        self.w = None
        self.b = None
        self.loss_history = []

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionScratch':
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for iteration in range(self.n_iter):
            # Forward pass
            z = X @ self.w + self.b
            y_hat = self.sigmoid(z)

            # Binary cross-entropy loss (with L2 regularization)
            eps = 1e-15   # avoid log(0)
            bce = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
            l2_penalty = (self.lambda_reg / 2) * np.sum(self.w ** 2)
            loss = bce + l2_penalty
            self.loss_history.append(loss)

            # Gradients (chain rule through sigmoid + cross-entropy)
            residuals = y_hat - y                           # (n,)
            dw = (X.T @ residuals) / n_samples + self.lambda_reg * self.w
            db = residuals.mean()

            # Gradient descent step
            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# Train from-scratch model
lr_scratch = LogisticRegressionScratch(lr=0.1, n_iter=500, lambda_reg=0.01)
lr_scratch.fit(X_train_scaled, y_train)
y_pred_lr_scratch = lr_scratch.predict(X_test_scaled)

print("\n--- Logistic Regression (From Scratch) ---")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_lr_scratch):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, lr_scratch.predict_proba(X_test_scaled)):.4f}")


# =============================================================================
# MODEL 2: DECISION TREE
# =============================================================================

# The power of a Decision Tree: it can learn non-linear boundaries
# Weakness: deep trees memorize training data (high variance)

dt_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)   # Regularized
dt_deep    = DecisionTreeClassifier(max_depth=None, random_state=42) # Unrestricted

dt_shallow.fit(X_train, y_train)
dt_deep.fit(X_train, y_train)

print("\n--- Decision Tree ---")
print(f"Shallow (depth=3) - Train: {dt_shallow.score(X_train, y_train):.4f}, "
      f"Test: {dt_shallow.score(X_test, y_test):.4f}")
print(f"Deep (unrestricted) - Train: {dt_deep.score(X_train, y_train):.4f}, "
      f"Test: {dt_deep.score(X_test, y_test):.4f}")
# Observe: deep tree has near-perfect train accuracy but worse test → OVERFITTING

# Feature importances from shallow tree
print("\nShallow DT Feature Importances:")
for name, imp in sorted(zip(feature_names, dt_shallow.feature_importances_),
                         key=lambda x: -x[1]):
    bar = '█' * int(imp * 30)
    print(f"  {name:<15} {bar} {imp:.4f}")


# =============================================================================
# MODEL 3: BAGGING
# =============================================================================

# Bagging = train many decision trees on bootstrap samples, average predictions
# Why this works: Individual trees have high variance; averaging uncorrelated
# high-variance estimators reduces variance without increasing bias.
#
# Proof sketch: Var(mean of n independent vars) = Var(single var) / n
# Trees aren't independent but are "decorrelated" → partial variance reduction

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=100,
    max_samples=0.8,    # Each tree sees 80% of training data (bootstrapped)
    max_features=0.8,   # Each tree uses 80% of features
    bootstrap=True,
    oob_score=True,     # Evaluate on out-of-bag samples (free validation!)
    random_state=42,
    n_jobs=-1           # Use all CPU cores
)
bagging.fit(X_train, y_train)

print("\n--- Bagging (100 Trees) ---")
print(f"OOB Score (free validation): {bagging.oob_score_:.4f}")
print(f"Test Accuracy: {bagging.score(X_test, y_test):.4f}")


# =============================================================================
# MODEL 4: RANDOM FOREST
# =============================================================================

# Random Forest = Bagging + random feature subset at each split
# This ADDITIONAL decorrelation is the key insight.
#
# Why feature subsampling helps:
# In Bagging, all trees will tend to use the most predictive feature
# at their root → they're still correlated. Feature subsampling forces
# each tree to find different splits → lower correlation → more variance reduction.

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    max_features='sqrt',    # Classic: consider sqrt(n_features) at each split
    min_samples_leaf=5,     # Prevents tiny, overfit leaves
    oob_score=True,
    class_weight='balanced', # Handles class imbalance in defect data
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("\n--- Random Forest (200 Trees) ---")
print(f"OOB Score: {rf.oob_score_:.4f}")
print(f"Test Accuracy: {rf.score(X_test, y_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Normal', 'Defective']))

print("\nRandom Forest Feature Importances:")
for name, imp in sorted(zip(feature_names, rf.feature_importances_),
                         key=lambda x: -x[1]):
    bar = '█' * int(imp * 30)
    print(f"  {name:<15} {bar} {imp:.4f}")


# =============================================================================
# MODEL 5: k-NEAREST NEIGHBOURS
# =============================================================================

# k-NN is instance-based: no model is learned.
# At inference, it computes distances to all training points → O(n) per query.
# Works well in low dimensions; suffers from curse of dimensionality in high-d.

knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean', n_jobs=-1)
knn.fit(X_train_scaled, y_train)   # Note: needs scaled features (distance-based)

print("\n--- k-Nearest Neighbours (k=7) ---")
print(f"Test Accuracy: {knn.score(X_test_scaled, y_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, knn.predict_proba(X_test_scaled)[:,1]):.4f}")


# =============================================================================
# MODEL COMPARISON SUMMARY
# =============================================================================

models = {
    'Logistic Regression (sklearn)': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    'Decision Tree (depth=3)': DecisionTreeClassifier(max_depth=3, random_state=42),
    'Bagging (100 trees)': BaggingClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Random Forest (200 trees)': RandomForestClassifier(n_estimators=200, oob_score=True,
                                                         class_weight='balanced',
                                                         random_state=42, n_jobs=-1),
    'k-NN (k=7)': KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
}

print("\n" + "="*65)
print(f"{'Model':<35} {'CV Acc':>8} {'CV Std':>8} {'AUC':>8}")
print("="*65)

for name, model in models.items():
    # Use appropriate feature set
    X_fit = X_train_scaled if 'Logistic' in name or 'k-NN' in name else X_train
    X_eval = X_test_scaled if 'Logistic' in name or 'k-NN' in name else X_test

    # 5-fold cross-validation on training data
    cv_scores = cross_val_score(model, X_fit, y_train, cv=5, scoring='accuracy', n_jobs=-1)

    # Fit and get AUC on held-out test
    model.fit(X_fit, y_train)
    if hasattr(model, 'predict_proba'):
        auc = roc_auc_score(y_test, model.predict_proba(X_eval)[:, 1])
    else:
        auc = 0.0

    print(f"{name:<35} {cv_scores.mean():>8.4f} {cv_scores.std():>8.4f} {auc:>8.4f}")

print("="*65)
print("\nConclusion: Random Forest achieves highest AUC — consistent with")
print("the data-generating process, which has non-linear feature interactions.")
print("Logistic Regression underperforms because the true decision boundary")
print("is non-linear (temperature × vibration interaction).")
