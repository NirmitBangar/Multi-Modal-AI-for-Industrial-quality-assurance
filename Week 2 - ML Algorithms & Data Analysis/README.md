# Week 2 — ML Algorithms & Data Analysis

##  Goals for This Week

- Study the most important supervised ML algorithms from first principles
- Learn Pandas for data loading, cleaning, transformation, and grouping
- Practice EDA (Exploratory Data Analysis) — the step before any model training
- Connect all algorithms to the industrial QA context

---

##  Algorithms Studied

### 1. Linear & Logistic Regression

**Linear Regression**: Fits a hyperplane `y = Xw` to minimize MSE. Closed-form solution: `w = (X^T X)^{-1} X^T y`.

**Logistic Regression**: Binary classifier. Passes the linear output through a sigmoid to produce probabilities. Loss: binary cross-entropy.

In QA: Logistic Regression is the baseline. If a more complex model barely beats it, the added complexity isn't worth it.

**Decision boundary insight**: Logistic Regression draws a *linear* boundary. If defects cluster in a non-linear region of feature space, it will underfit → need tree-based or neural approaches.

---

### 2. Decision Trees

A Decision Tree recursively splits the feature space to minimize impurity (Gini index or entropy).

**Gini Impurity**: `G = 1 - Σ p_k²` — measures how "mixed" a node is. A pure node (all one class) has G = 0.

**Key hyperparameters**:
- `max_depth` — prevents overfitting (a fully grown tree memorizes training data)
- `min_samples_split` — minimum samples needed to split a node
- `min_samples_leaf` — minimum samples in a leaf node

**Decision Trees are:**
- ✅ Interpretable (can visualize the tree — useful for explaining QA decisions to plant managers)
- ✅ Handle non-linear boundaries
- ❌ High variance — small change in training data → completely different tree
- ❌ Prone to overfitting without regularization (max_depth)

This high variance is why we use **Ensemble Methods** (next section).

---

### 3. Ensemble Methods — Bagging & Random Forests

**Core idea**: One decision tree has high variance. Train *many* trees on different bootstrapped subsets of data, then average their predictions → variance reduces, bias stays the same.

**Bagging (Bootstrap Aggregating)**:
1. Sample N rows *with replacement* from training data (bootstrap)
2. Train a decision tree on each bootstrap sample
3. Aggregate: average (regression) or majority vote (classification)

**Random Forest** = Bagging + Feature Subsampling:
- At each node split, consider only a random subset of features (`sqrt(n_features)` for classification)
- This *decorrelates* the trees, further reducing variance

**Why this matters for QA**: Real factory data is messy and correlated. A single decision tree trained on noisy sensor data will overfit badly. A Random Forest of 100–500 trees is far more robust.

**Out-of-Bag (OOB) Error**:
Each tree is trained on ~63% of the data (due to bootstrapping). The remaining 37% ("out of bag") can be used as a free validation set — no need for a separate val split.

---

### 4. k-Nearest Neighbours (k-NN)

Non-parametric: no training phase. For a new point, find the k closest points in the training set (by Euclidean/Cosine distance), take majority vote.

**Key insight**: k-NN is the simplest intuition — "this new product looks like these k past products, which were all defective → probably defective." But it's O(n) at inference time and fails in high-dimensional spaces (curse of dimensionality).

In QA: Useful as a sanity-check baseline. If a complex CNN can't beat k-NN on image features, something is wrong.

---

##  EDA Concepts Practiced

### Why EDA Comes Before Modelling

Many students rush to fit models. In reality, 70% of a data scientist's time is EDA + preprocessing. Garbage in → garbage out.

EDA goals for a QA dataset:
1. **Distribution analysis** — Are defect counts balanced or heavily skewed? (Class imbalance → need SMOTE or class weights)
2. **Outlier detection** — Are there sensor readings that are physically impossible? (Negative temperature → data pipeline bug)
3. **Feature correlation** — Are temperature and pressure highly correlated? (Multicollinearity → affects Logistic Regression; fine for trees)
4. **Missing value patterns** — Are values missing at random or systematically? (A sensor that only reports when it trips → informative missingness)
5. **Temporal patterns** — Do defect rates spike at shift changes? (Time-of-day features might matter)

### Key Pandas Operations

| Operation | Code Pattern | Use Case |
|-----------|-------------|----------|
| Load CSV | `pd.read_csv('data.csv')` | Load sensor log |
| Inspect | `.head()`, `.info()`, `.describe()` | First look at data |
| Null check | `.isnull().sum()` | Find missing values |
| Filter | `df[df['temp'] > 80]` | Select anomalous rows |
| GroupBy | `df.groupby('shift')['defect'].mean()` | Defect rate by shift |
| Pivot | `pd.pivot_table(...)` | Cross-tabulation |
| Apply | `df['col'].apply(func)` | Custom transformation |
| Merge | `pd.merge(df1, df2, on='id')` | Join sensor + label tables |

---

##  Code Written This Week

### `ml_algorithms.py`
- Logistic Regression with gradient descent (from scratch + sklearn)
- Decision Tree implementation (sklearn) with visualization
- Random Forest comparison
- Feature importance plot

### `pandas_eda.py`
- Loading and cleaning a simulated industrial dataset
- Handling missing values (forward fill for time-series sensor data)
- Groupby analysis: defect rate by station, shift, product type
- Correlation matrix

### `visualizations.py`
- Distribution plots for each feature
- Defect rate heatmap (station × shift)
- Learning curves (train vs. val error vs. training set size)
- ROC curve for binary defect classifier

---

## 🔗 Resources Used

| Resource | Link |
|----------|------|
| ML Playlist (CodeWithHarry) | [YouTube](https://www.youtube.com/watch?v=UrsmFxEIp5k) — watched till Bagging |
| Pandas Tutorial | [YouTube](https://youtu.be/E9WGC0SLPVs?si=yrzPBON1ay1TQ3OP) |

---

##  Key Takeaway

> Random Forest outperformed Logistic Regression and single Decision Tree on the simulated QA sensor dataset. The key insight: in industrial settings, defect causation is rarely linear — it involves **interactions** between features (high temperature AND high vibration AND certain product type → defect). Tree-based ensembles capture these interactions naturally.
