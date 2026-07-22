# ML Theory Notes — Week 1
## Machine Learning from First Principles

### What is a Model?

A model is a **parameterized function** that maps inputs to outputs:

```
f(x; θ) → y
```

- `x` = input (e.g., sensor readings, pixel values)
- `θ` = parameters (weights and biases — learned from data)
- `y` = output (e.g., "defective" or "normal")

Training = finding the `θ` that minimizes prediction error on labelled data.

---

### Loss Functions

The **loss function** (also called cost or objective function) measures how wrong the model is. Gradient descent moves `θ` in the direction that reduces this loss.

| Loss | Formula | Used For |
|------|---------|----------|
| Mean Squared Error | `(1/n) Σ(ŷ - y)²` | Regression (continuous output) |
| Binary Cross-Entropy | `-[y log(ŷ) + (1-y) log(1-ŷ)]` | Binary classification (defect/no-defect) |
| Categorical Cross-Entropy | `-Σ y_i log(ŷ_i)` | Multi-class (type of defect) |

For our QA problem: we'll use **binary cross-entropy** (defective vs. normal) and **categorical cross-entropy** (which defect type).

---

### Gradient Descent — Intuition

Imagine you're blindfolded on a hilly landscape and want to reach the lowest valley (minimum loss). You can feel the slope under your feet. Each step, you move **downhill** (negative gradient direction).

```
θ_new = θ_old - α * ∇L(θ)
```

- `α` = learning rate (step size — too large → overshoot, too small → slow)
- `∇L(θ)` = gradient of the loss with respect to parameters

**Variants:**
- **Batch GD**: Uses ALL training data per step → stable but slow
- **SGD**: Uses ONE sample per step → noisy but fast
- **Mini-batch GD**: Uses a batch of ~32–256 samples → best of both

---

### The Three Types of ML

**1. Supervised Learning**
- Input: labelled data `(x_i, y_i)`
- Goal: Learn `f` such that `f(x_i) ≈ y_i`
- QA application: Learn to classify product images as defective/normal from historical labelled inspections

**2. Unsupervised Learning**
- Input: unlabelled data `{x_i}`
- Goal: Find structure (clusters, compressed representations)
- QA application: Cluster sensor readings to discover natural operating states without human labels

**3. Semi-Supervised Learning**
- Combines small labelled + large unlabelled dataset
- Very relevant in industrial QA: labelled defect images are expensive to obtain (requires expert annotation), but unlabelled "normal" production images are plentiful

---

### Bias-Variance Tradeoff — Deep Dive

Total prediction error = **Bias² + Variance + Irreducible Noise**

| | Bias | Variance |
|-|------|----------|
| Definition | Error from wrong model assumptions | Error from sensitivity to training data fluctuations |
| Cause | Model too simple | Model too complex |
| Symptom | High train error AND high test error | Low train error BUT high test error |
| Example | Linear classifier on non-linear data | 100-leaf decision tree on small dataset |
| Fix | More complex model, more features | Regularization, dropout, more data |

**Why this matters for QA:**
- A biased model will consistently miss certain defect types (systematic failure)
- A high-variance model will perform well in the test lab but fail on the factory floor with slightly different lighting conditions

The goal: find the **sweet spot** where both are minimized. Ensemble methods (Random Forest, XGBoost) are specifically designed to reduce variance while keeping bias manageable.

---

### Cross-Validation

Instead of one train/test split, we do **k-fold cross-validation**:

1. Split data into k folds (typically k=5 or 10)
2. Train on k-1 folds, evaluate on the remaining fold
3. Repeat k times, rotating which fold is the test set
4. Average the k evaluation scores

**Why bother?**
- Single split can be lucky or unlucky (random seed effects)
- k-fold gives a more reliable estimate of generalization error
- Especially important in industrial QA where datasets are small (collecting labelled defect images is expensive)

---

### Key Terms Glossary

| Term | Definition |
|------|-----------|
| Feature | An input variable used by the model (e.g., temperature, pixel intensity) |
| Label / Target | The output we want to predict (e.g., 0 = normal, 1 = defective) |
| Hyperparameter | Setting chosen before training (e.g., learning rate, tree depth) — not learned |
| Epoch | One full pass through the training dataset |
| Overfitting | Model memorizes training data; fails on unseen data |
| Underfitting | Model is too simple to capture patterns even in training data |
| Regularization | Techniques to prevent overfitting (L1/L2 penalty, dropout) |
| Generalization | Model's ability to perform well on data it hasn't seen |
