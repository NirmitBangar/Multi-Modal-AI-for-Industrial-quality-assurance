# Week 1 — Python Foundations & Machine Learning Theory

##  Goals for This Week

By the end of this week, the aim was to:
- Set up a working Python environment (Google Colab)
- Develop comfort with core Python constructs used throughout the project
- Build an intuitive, first-principles understanding of what Machine Learning *actually is* before touching any library

---

##  Concepts Covered

### Python Fundamentals

| Concept | What I Learned |
|---------|---------------|
| Variables & Data Types | `int`, `float`, `str`, `bool`, `NoneType` — Python's dynamic typing vs static typing in C++ |
| Lists vs Tuples vs Dicts | Mutability, use cases — lists for sequences, dicts for mappings (key-value), tuples for fixed records |
| List Comprehensions | Pythonic way to build filtered/transformed lists in one line — heavily used in data pipelines |
| Functions & Scope | `def`, default args, `*args`, `**kwargs`; local vs global scope; pure functions vs side effects |
| File I/O | `open()`, context managers (`with` statement), reading CSVs manually before learning Pandas |
| Error Handling | `try/except/finally`, raising custom exceptions — critical for robust ML pipelines |

### NumPy Foundations

NumPy is the backbone of all numerical computing in Python. Key ideas:

- **ndarray vs Python list**: An ndarray stores homogeneous data in a contiguous memory block. Operations are vectorized — they run in C under the hood — making them orders of magnitude faster than Python loops.
- **Broadcasting**: How NumPy applies operations between arrays of different shapes. Example: adding a (3,) array to a (4,3) array broadcasts the smaller array across rows.
- **Vectorization vs. loops**: A Python loop over 1 million elements takes ~seconds; the equivalent NumPy operation takes milliseconds.

```python
import numpy as np

# Vectorized operation — no loop needed
a = np.array([1, 2, 3, 4, 5])
print(a ** 2)          # [1 4 9 16 25] — element-wise, in C speed
print(a.mean())        # 3.0
print(a.std())         # 1.414...
```

### Machine Learning Theory (First Principles)

Before writing a single line of ML code, it's important to understand what we're actually doing.

**What is Machine Learning?**

Traditional programming: you write rules → computer applies them to data → output.  
Machine Learning: you give data + outputs → algorithm finds the rules itself.

**Three Paradigms:**

| Type | Description | Industrial QA Example |
|------|-------------|----------------------|
| Supervised Learning | Learn from labelled data (input → known output) | "This image is [defective/normal]" |
| Unsupervised Learning | Find structure in unlabelled data | Cluster sensor readings to detect anomaly patterns |
| Reinforcement Learning | Agent learns by reward/penalty | Robotic arm learns to pick parts correctly |

**Bias-Variance Tradeoff:**

This is the most important concept in classical ML. Every model sits somewhere on this spectrum:

- **High Bias (Underfitting)**: Model is too simple → misses real patterns in training AND test data. Example: fitting a straight line to temperature-defect rate data that has a curve.
- **High Variance (Overfitting)**: Model is too complex → memorizes training data but fails on new data. Example: a decision tree that perfectly classifies training images but hallucinates on new ones.
- **Sweet spot**: Models like Random Forests and ensembles are specifically designed to reduce variance without increasing bias too much — directly relevant to building a reliable QA classifier.

**The Training-Validation-Test Split:**

Why do we split data into three parts?
- **Training set**: The model sees and learns from this.
- **Validation set**: We tune hyperparameters here. The model never explicitly trains on it, but we indirectly optimize on it → some leakage.
- **Test set**: Locked away until final evaluation. Gives honest estimate of real-world performance.

In an industrial QA setting, leaking test data would be catastrophic — you'd deploy a model that looks 99% accurate in the lab but fails in production.

---

##  Code Written This Week

### `python_foundations.py`
Covers: data types, loops, functions, list comprehensions, OOP basics (classes with `__init__`, methods), file I/O

### `numpy_intro.py`
Covers: array creation, slicing, broadcasting, vectorized math, linear algebra ops (`np.dot`, `np.linalg.inv`, `np.linalg.eig`)

### `ml_theory_notes.md`
Covers: Supervised vs unsupervised learning, hypothesis space, loss functions (MSE, cross-entropy), gradient descent intuition

---

##  Resources Used

| Resource | Link | Notes |
|----------|------|-------|
| Python Video Lecture (CodeWithHarry) | [YouTube](https://www.youtube.com/watch?v=UrsmFxEIp5k) | Chapters 1–8, skipped recursion |
| The Ultimate Python Handbook | [PDF](https://cwh-full-next-space.fra1.cdn.digitaloceanspaces.com/YouTube/The%20Ultimate%20Python%20Handbook.pdf) | Used for revision after video |
| Intro to ML | [YouTube](https://www.youtube.com/watch?v=ukzFI9rgwfU) | Conceptual overview, very clear |
| futurecoder.io | [Website](https://futurecoder.io/) | Interactive coding practice |

---

## Takeaway

> The difference between a programmer who knows Python and one who knows ML is not syntax — it's understanding *why* we split data, *what* a loss function measures, and *how* gradients tell a model which direction to improve. Week 1 built that intuition.
