"""
Week 3: Neural Network from Scratch — Backpropagation Implementation
======================================================================
This file implements a fully connected neural network (Multi-Layer Perceptron)
from scratch using only NumPy. The goal is to understand exactly what
happens during training — forward pass, loss computation, backward pass,
and weight update.

This is the foundation for understanding CNNs, which add convolutional
layers but use the SAME backpropagation algorithm for learning.

Architecture for QA task (sensor data):
  Input (4 features) → Hidden(64, ReLU) → Hidden(32, ReLU) → Output(1, Sigmoid)
  Loss: Binary Cross-Entropy
  Optimizer: Mini-batch SGD with momentum
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

np.random.seed(42)


# =============================================================================
# ACTIVATION FUNCTIONS AND THEIR DERIVATIVES
# =============================================================================

def relu(z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit: max(0, z)"""
    return np.maximum(0, z)

def relu_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU:
      d/dz max(0,z) = 1 if z > 0, else 0
    Used in backward pass (chain rule)
    """
    return (z > 0).astype(float)

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: 1 / (1 + e^{-z})"""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid:
      d/dz σ(z) = σ(z)(1 - σ(z))
    This saturation (→ 0 for large |z|) causes vanishing gradients in deep nets.
    """
    s = sigmoid(z)
    return s * (1 - s)

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    BCE = -mean[ y*log(ŷ) + (1-y)*log(1-ŷ) ]

    This is the appropriate loss for binary classification.
    When y=1: loss = -log(ŷ) — penalizes low confidence in positive prediction
    When y=0: loss = -log(1-ŷ) — penalizes high confidence in negative prediction
    """
    eps = 1e-12   # Clip to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# =============================================================================
# WEIGHT INITIALIZATION
# =============================================================================

def he_initialization(n_in: int, n_out: int) -> np.ndarray:
    """
    He (Kaiming) Initialization — designed for ReLU activations.

    Standard deviation: sqrt(2 / n_in)

    Derivation insight: For a ReLU network, roughly half the neurons are
    "off" (output 0) at any time. He init compensates for this by using
    sqrt(2/n_in) instead of sqrt(1/n_in), maintaining variance through layers.

    Wrong initialization leads to:
    - Too large: activations explode → NaN gradients
    - Too small: activations shrink to 0 → vanishing gradients
    """
    return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)


# =============================================================================
# NEURAL NETWORK CLASS
# =============================================================================

class NeuralNetwork:
    """
    Fully connected neural network with configurable architecture.

    Supports:
    - Arbitrary depth and width
    - ReLU hidden activations + Sigmoid output
    - Mini-batch gradient descent with momentum
    - L2 weight regularization
    - Training history tracking

    Architecture is defined by `layer_dims`:
    e.g., [4, 64, 32, 1] → 4 inputs, 2 hidden layers, 1 output
    """

    def __init__(self, layer_dims: list[int], lr: float = 0.01,
                 lambda_reg: float = 0.001, momentum: float = 0.9):
        """
        Initialize network weights using He initialization.

        Args:
            layer_dims: List of layer sizes [input_dim, hidden1, ..., output_dim]
            lr: Learning rate
            lambda_reg: L2 regularization coefficient
            momentum: Momentum coefficient for SGD
        """
        self.layer_dims = layer_dims
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.momentum = momentum
        self.n_layers = len(layer_dims) - 1
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        # Initialize weights and biases
        self.W = {}   # Weight matrices
        self.b = {}   # Bias vectors
        self.vW = {}  # Momentum for weights
        self.vb = {}  # Momentum for biases

        for l in range(1, self.n_layers + 1):
            self.W[l] = he_initialization(layer_dims[l-1], layer_dims[l])
            self.b[l] = np.zeros((1, layer_dims[l]))
            self.vW[l] = np.zeros_like(self.W[l])
            self.vb[l] = np.zeros_like(self.b[l])

    def _forward(self, X: np.ndarray) -> tuple[dict, dict]:
        """
        Forward pass: compute activations at every layer.

        Returns:
            Z: pre-activations (before activation function) — needed for backprop
            A: post-activations (after activation function)
        """
        Z, A = {}, {}
        A[0] = X   # Input is the "zeroth activation"

        for l in range(1, self.n_layers + 1):
            Z[l] = A[l-1] @ self.W[l] + self.b[l]   # Linear step
            if l < self.n_layers:
                A[l] = relu(Z[l])                     # Hidden layers: ReLU
            else:
                A[l] = sigmoid(Z[l])                  # Output: Sigmoid → probability

        return Z, A

    def _backward(self, X: np.ndarray, y: np.ndarray,
                  Z: dict, A: dict) -> tuple[dict, dict]:
        """
        Backward pass: compute gradients via the chain rule.

        Key insight: we propagate the error (delta) backwards through the network.
        The gradient at each layer depends on the gradient of all later layers.

        For binary cross-entropy + sigmoid output:
          δ^[L] = A^[L] - y    (elegant simplification of chain rule)

        For hidden layers (ReLU):
          δ^[l] = (δ^[l+1] @ W^[l+1].T) * relu'(Z^[l])

        Returns:
            dW: Gradients for weight matrices
            db: Gradients for bias vectors
        """
        n = X.shape[0]
        dW, db = {}, {}
        delta = {}

        # Output layer: BCE + Sigmoid gradient simplifies to (A[L] - y)
        delta[self.n_layers] = A[self.n_layers] - y.reshape(-1, 1)

        # Hidden layers (going backward)
        for l in range(self.n_layers - 1, 0, -1):
            delta[l] = (delta[l+1] @ self.W[l+1].T) * relu_derivative(Z[l])

        # Compute weight gradients from deltas
        for l in range(1, self.n_layers + 1):
            dW[l] = (A[l-1].T @ delta[l]) / n + self.lambda_reg * self.W[l]
            db[l] = delta[l].mean(axis=0, keepdims=True)

        return dW, db

    def _update_weights(self, dW: dict, db: dict):
        """
        SGD with momentum update:
          v = momentum * v_prev - lr * gradient
          w = w + v

        Momentum accumulates gradient in directions of consistent improvement,
        dampens oscillations in directions with sign changes.
        Conceptually: gradient descent with inertia, like a ball rolling downhill.
        """
        for l in range(1, self.n_layers + 1):
            self.vW[l] = self.momentum * self.vW[l] - self.lr * dW[l]
            self.vb[l] = self.momentum * self.vb[l] - self.lr * db[l]
            self.W[l] += self.vW[l]
            self.b[l] += self.vb[l]

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            n_epochs: int = 200, batch_size: int = 64,
            verbose: bool = True) -> 'NeuralNetwork':
        """
        Mini-batch gradient descent training loop.

        Why mini-batches?
        - Batch GD: precise gradients, but one slow update per epoch
        - SGD (batch=1): fast updates, but very noisy → doesn't converge cleanly
        - Mini-batch (32–256): balances speed and gradient quality → industry standard
        """
        n = X_train.shape[0]

        for epoch in range(n_epochs):
            # Shuffle training data each epoch → prevents pathological repetitions
            perm = np.random.permutation(n)
            X_shuf, y_shuf = X_train[perm], y_train[perm]

            # Mini-batch loop
            for start in range(0, n, batch_size):
                X_batch = X_shuf[start:start + batch_size]
                y_batch = y_shuf[start:start + batch_size]

                Z, A = self._forward(X_batch)
                dW, db = self._backward(X_batch, y_batch, Z, A)
                self._update_weights(dW, db)

            # Track metrics at end of each epoch (on full train + val)
            _, A_train = self._forward(X_train)
            _, A_val   = self._forward(X_val)

            train_loss = binary_cross_entropy(y_train, A_train[self.n_layers].ravel())
            val_loss   = binary_cross_entropy(y_val,   A_val[self.n_layers].ravel())
            train_acc  = accuracy_score(y_train, (A_train[self.n_layers].ravel() > 0.5).astype(int))
            val_acc    = accuracy_score(y_val,   (A_val[self.n_layers].ravel() > 0.5).astype(int))

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, A = self._forward(X)
        return A[self.n_layers].ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def plot_training_curves(self):
        """Visualize training dynamics — essential for diagnosing training issues."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        epochs = range(1, len(self.train_losses) + 1)

        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r--', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Binary Cross-Entropy Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, self.train_accs, 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(epochs, self.val_accs, 'r--', label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training & Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Detect overfitting: val loss increases while train loss decreases
        train_final = self.train_losses[-1]
        val_final   = self.val_losses[-1]
        gap = val_final - train_final
        if gap > 0.05:
            ax1.text(0.6, 0.8, f'⚠ Generalization gap: {gap:.3f}',
                     transform=ax1.transAxes, color='red', fontsize=10)

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()


# =============================================================================
# GENERATE DATA & TRAIN
# =============================================================================

def generate_qa_data(n=2000):
    """Reuse from Week 2 — non-linear QA classification problem."""
    temp = np.random.normal(75, 10, n)
    vib  = np.abs(np.random.normal(1.5, 0.8, n))
    pres = np.random.normal(9, 2, n)
    spd  = np.random.normal(120, 15, n)

    logit = (-5 + 0.04*(temp-75) + 0.7*(vib-1.5) + 0.03*(temp-75)*(vib-1.5) + 0.08*(pres-9))
    prob  = 1 / (1 + np.exp(-logit))
    y     = (np.random.rand(n) < prob).astype(int)
    X     = np.column_stack([temp, vib, pres, spd])
    return X, y

X, y = generate_qa_data(2000)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

scaler = StandardScaler()
X_tr   = scaler.fit_transform(X_tr)
X_val  = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Defect rates — Train: {y_tr.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")


# Train the network
net = NeuralNetwork(
    layer_dims=[4, 64, 32, 1],
    lr=0.02,
    lambda_reg=0.001,
    momentum=0.9
)

print("\n--- Training Neural Network [4 → 64 → 32 → 1] ---")
net.fit(X_tr, y_tr, X_val, y_val, n_epochs=200, batch_size=64)

# Final evaluation
y_prob_test = net.predict_proba(X_test)
y_pred_test = net.predict(X_test)

print("\n--- Final Test Set Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob_test):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=['Normal', 'Defective']))

net.plot_training_curves()

print("\n[Key Insight] This MLP is applied to tabular sensor data.")
print("In Week 3 (cnn_keras.py), we apply the SAME principles to image data")
print("but with convolutional layers replacing fully connected layers for the")
print("feature extraction stage — exploiting spatial locality and weight sharing.")
