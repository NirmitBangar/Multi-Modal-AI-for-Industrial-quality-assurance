"""
Week 1: NumPy for ML — The Foundation of Numerical Computing
=============================================================
NumPy (Numerical Python) is the backbone of the entire Python ML ecosystem.
PyTorch, TensorFlow, Pandas, Scikit-learn — they all sit on top of NumPy.

Understanding NumPy deeply means understanding:
  - WHY vectorization is fast (contiguous memory, BLAS)
  - HOW broadcasting works (rules for shape compatibility)
  - WHAT linear algebra ops look like before they're abstracted

This file explores NumPy through the lens of QA data:
  - Images as 3D arrays (H × W × C)
  - Sensor batches as 2D matrices (samples × features)
  - Statistical operations used in anomaly detection
"""

import numpy as np

np.random.seed(42)   # Reproducibility — always set this in ML experiments


# =============================================================================
# SECTION 1: Array Creation and Memory Layout
# =============================================================================

# The core of NumPy: the ndarray
# Key difference from Python list: ndarray stores ONE type, in a contiguous
# block of memory. This allows vectorized C operations → massive speedup.

# 1D array — e.g., a sequence of temperature readings
temps = np.array([72.1, 73.5, 74.0, 69.8, 80.3], dtype=np.float32)
print("Shape:", temps.shape)      # (5,)
print("DType:", temps.dtype)      # float32
print("ItemSize:", temps.itemsize, "bytes per element")  # 4 bytes for float32

# 2D array — e.g., sensor matrix: rows=samples, cols=features
# [temperature, vibration, pressure, speed] per sample
sensor_matrix = np.array([
    [72.1, 1.2, 8.5, 120],
    [73.5, 1.8, 9.0, 118],
    [80.3, 2.1, 8.2, 115],
    [68.9, 1.0, 7.8, 122],
    [95.4, 3.8, 12.1, 98],    # <- anomalous sample
], dtype=np.float64)

print("\nSensor Matrix shape:", sensor_matrix.shape)   # (5, 4) → 5 samples, 4 features
print("Total elements:", sensor_matrix.size)           # 20

# Simulating a grayscale image (e.g., from a line-scan camera)
# Shape: (height, width) — pixel intensity values 0–255
grayscale_img = np.random.randint(0, 256, size=(256, 256), dtype=np.uint8)

# RGB image: (height, width, channels)
rgb_img = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)

# Batch of images for CNN input: (batch_size, height, width, channels)
image_batch = np.random.randint(0, 256, size=(32, 256, 256, 3), dtype=np.uint8)
print("\nImage batch shape:", image_batch.shape)   # (32, 256, 256, 3)

# Memory footprint comparison
import sys
py_list = [float(i) for i in range(100_000)]
np_array = np.arange(100_000, dtype=np.float64)
print(f"\nPython list (100k floats): {sys.getsizeof(py_list):,} bytes")
print(f"NumPy array (100k float64): {np_array.nbytes:,} bytes")
# NumPy is ~8x more memory efficient here


# =============================================================================
# SECTION 2: Indexing, Slicing, Fancy Indexing
# =============================================================================

# 2D slicing: [row_slice, col_slice]
# First 3 samples, all features
subset = sensor_matrix[:3, :]
print("\nFirst 3 samples:\n", subset)

# Just the temperature column (column 0)
temperatures = sensor_matrix[:, 0]
print("Temperatures:", temperatures)

# Just the anomalous row
anomaly = sensor_matrix[4, :]
print("Anomalous sample:", anomaly)

# Boolean (mask) indexing — crucial for filtering in ML pipelines
# Find all samples where temperature > 75°C
high_temp_mask = sensor_matrix[:, 0] > 75.0
print("\nHigh temp mask:", high_temp_mask)            # [False False  True False  True]
print("High temp samples:\n", sensor_matrix[high_temp_mask])

# Fancy indexing: select rows by index list
selected = sensor_matrix[[0, 2, 4], :]   # Rows 0, 2, 4
print("\nSelected samples:\n", selected)


# =============================================================================
# SECTION 3: Vectorized Operations & Broadcasting
# =============================================================================

# --- Vectorized arithmetic (no loops!) ---
# This runs in C under the hood — crucial for performance with large datasets

raw_pixels = np.array([100, 150, 200, 255, 50], dtype=np.float32)

# Normalize pixels to [0, 1] — standard preprocessing before feeding to CNN
normalized_pixels = raw_pixels / 255.0
print("\nNormalized pixels:", normalized_pixels)

# Standardize sensor features: z = (x - mean) / std
# This is the most common normalization in ML (StandardScaler in sklearn does this)
means = sensor_matrix.mean(axis=0)   # Mean of each feature (column)
stds  = sensor_matrix.std(axis=0)    # Std of each feature (column)
standardized = (sensor_matrix - means) / stds  # Broadcasting! (5,4) - (4,) works

print("\nFeature means:", means.round(2))
print("Feature stds:", stds.round(2))
print("\nStandardized matrix:\n", standardized.round(3))
# Now all features have mean≈0, std≈1 — gradient descent trains much faster


# --- Broadcasting Rules ---
# NumPy can operate on arrays of different shapes if they are "broadcast-compatible":
# Starting from the trailing dimension, shapes are compatible if:
#   - They are equal, OR
#   - One of them is 1

# Example: Subtract per-channel mean from a batch of images
image_batch_f = image_batch.astype(np.float32)            # (32, 256, 256, 3)
channel_means = np.array([0.485, 0.456, 0.406]) * 255     # (3,) — ImageNet means
result = image_batch_f - channel_means                     # (32,256,256,3) - (3,) ← broadcast!
# NumPy broadcasts (3,) to (1,1,1,3) then to (32,256,256,3) automatically
print("\nBroadcasted image batch shape:", result.shape)    # (32, 256, 256, 3)


# =============================================================================
# SECTION 4: Statistical Operations — Used in EDA and Anomaly Detection
# =============================================================================

# Axis-wise operations: axis=0 → across rows (per column), axis=1 → across columns (per row)
print("\n--- Statistical Operations on Sensor Matrix ---")
print("Column means  (per feature):", sensor_matrix.mean(axis=0).round(2))
print("Row means     (per sample): ", sensor_matrix.mean(axis=1).round(2))
print("Global mean:                ", sensor_matrix.mean().round(2))
print("Column std:                 ", sensor_matrix.std(axis=0).round(2))

# Z-score anomaly detection (conceptual foundation of many anomaly methods)
z_scores = np.abs((sensor_matrix - means) / stds)
print("\nZ-scores:\n", z_scores.round(2))

# A sample is anomalous if ANY feature has z-score > 2.5
is_anomalous = np.any(z_scores > 2.5, axis=1)
print("\nAnomaly flags:", is_anomalous)   # [False False False False  True]
print("Anomalous sample indices:", np.where(is_anomalous)[0])   # [4]


# =============================================================================
# SECTION 5: Linear Algebra — The Math Behind ML
# =============================================================================

# --- Dot product and matrix multiplication ---
# The dot product is the most fundamental operation in ML:
#   - Linear regression: y = Xw (matrix-vector multiply)
#   - Neural network layer: output = activation(W @ input + b)
#   - Cosine similarity: cos(θ) = (a · b) / (|a| |b|)

# Simple linear regression: y = w0 + w1*x1 + w2*x2 + w3*x3 + w4*x4
# In matrix form: y = X @ w   where X is (n_samples, n_features), w is (n_features,)
weights = np.array([0.3, 0.5, -0.1, 0.2])   # Learned weights (would be from training)
predictions = sensor_matrix @ weights          # (5, 4) @ (4,) → (5,)
print("\n--- Linear Algebra ---")
print("Predictions (y = Xw):", predictions.round(3))

# --- Matrix inversion: used in closed-form OLS solution ---
# OLS solution: w = (X^T X)^{-1} X^T y
# This is what `LinearRegression` in sklearn computes under the hood
X = sensor_matrix
X_T_X = X.T @ X                       # (4, 4)
X_T_X_inv = np.linalg.inv(X_T_X)     # (4, 4)
print("X^T X shape:", X_T_X.shape)
print("Condition number:", round(np.linalg.cond(X_T_X), 2))
# High condition number → matrix nearly singular → use pseudo-inverse instead

# --- Eigendecomposition: basis of PCA ---
# PCA finds directions of maximum variance in data
# These directions are the eigenvectors of the covariance matrix
cov_matrix = np.cov(standardized.T)   # (4, 4) covariance matrix of features
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by descending eigenvalue (most variance first)
sort_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sort_idx].real
eigenvectors = eigenvectors[:, sort_idx].real

print("\nEigenvalues (explained variance):", eigenvalues.round(3))
print("Variance explained by PC1:", round(eigenvalues[0] / eigenvalues.sum() * 100, 1), "%")
# The first principal component captures the direction of greatest spread —
# in a QA dataset, this often corresponds to the "overall process health" axis


# =============================================================================
# SECTION 6: Performance Comparison — Loops vs NumPy
# =============================================================================

import time

n = 1_000_000
data = list(range(n))
np_data = np.arange(n, dtype=np.float64)

# Python loop
start = time.time()
result_loop = sum(x ** 2 for x in data)
loop_time = time.time() - start

# NumPy vectorized
start = time.time()
result_numpy = np.sum(np_data ** 2)
numpy_time = time.time() - start

print(f"\n--- Performance Comparison (n={n:,}) ---")
print(f"Python loop:  {loop_time:.4f}s → result={result_loop:,.0f}")
print(f"NumPy vector: {numpy_time:.4f}s → result={result_numpy:,.0f}")
print(f"Speedup: {loop_time / numpy_time:.1f}x faster")
# Typical result: NumPy is 20–100x faster for large arrays
