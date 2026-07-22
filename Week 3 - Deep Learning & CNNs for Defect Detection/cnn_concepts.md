# CNN Concepts — Detailed Notes
## Week 3: Deep Learning for Visual QA

---

## 1. The Convolution Operation — Mathematical Definition

For a 2D input `I` and kernel `K`, the discrete convolution at position (i, j):

```
(I * K)[i, j] = Σ_m Σ_n  I[i + m, j + n] · K[m, n]
```

The kernel slides across the image, computing a dot product at each position.

**Key parameters:**
- **Kernel size**: 3×3 (most common), 5×5, 1×1
- **Stride**: how many pixels the kernel moves per step (stride=1 → dense, stride=2 → downsample by 2)
- **Padding**: "same" → pad with zeros to preserve spatial dims; "valid" → no padding → shrinks output

**Output dimension formula:**
```
H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1
```
Example: 256×256 input, 3×3 kernel, stride=1, same padding → 256×256 output (unchanged)

---

## 2. Why 3×3 Kernels Dominate

The VGG paper (2014) showed that two stacked 3×3 conv layers have the same receptive field as one 5×5 layer, but with:
- **Fewer parameters**: 2×(3×3)=18 vs 5×5=25 weights per filter
- **More non-linearity**: Two ReLU activations instead of one → more expressive

This is why 3×3 is the universal default. 1×1 convolutions are used to change the number of channels without spatial operations (used in Inception, ResNet bottlenecks).

---

## 3. Feature Maps and Receptive Fields

Each conv layer produces **feature maps** — one per filter.

**Receptive field**: The region of the input that influences a particular output neuron.
- Layer 1 (3×3 kernel): receptive field = 3×3 pixels
- Layer 2 (3×3 kernel on layer 1 output): receptive field = 5×5 pixels
- Layer 3: receptive field = 7×7 pixels
- After 3 MaxPool layers (each halving dims): effective RF much larger

For defect detection, we need sufficient receptive field to "see" the full defect region. A scratch might be 40 pixels long → need enough layers that the RF covers at least that area.

---

## 4. MaxPooling vs AvgPooling vs Strided Convolution

| Method | Operation | Properties |
|--------|-----------|------------|
| MaxPooling | Take max value in window | Preserves most prominent feature, translation invariant |
| AvgPooling | Take average in window | Smoother, used in global pooling at end of network |
| Strided Conv | Skip pixels during convolution | Learnable downsampling, more flexibility |

MaxPooling (2×2, stride=2) is most common: halves spatial dimensions, doubles effective receptive field.

**Translation invariance intuition**: A 2×2 MaxPool means a feature detected at position (i, j) OR (i+1, j) OR (i, j+1) OR (i+1, j+1) produces the same output → the model doesn't care exactly *where* in the 2×2 region the feature is. Repeated pooling builds broad translation invariance.

---

## 5. Batch Normalization in CNNs

BatchNorm after a conv layer:
```
For each channel c:
  μ_c = mean of all activations in channel c across batch and spatial dims
  σ_c = std  of all activations in channel c
  x̂ = (x - μ_c) / (σ_c + ε)
  output = γ · x̂ + β   (learnable scale and shift per channel)
```

**Benefits:**
1. Reduces internal covariate shift (activations stay in a consistent range)
2. Allows higher learning rates (otherwise gradients become unstable)
3. Acts as mild regularization (each mini-batch introduces slight noise via μ, σ estimates)
4. Reduces dependence on careful weight initialization

**During inference**: Uses running mean/std accumulated during training (not mini-batch stats), so behavior is deterministic.

---

## 6. Skip Connections — ResNet Architecture

Standard CNN: `x → Conv → BN → ReLU → ... → output`

ResNet block: `x → Conv → BN → ReLU → Conv → BN → + x → ReLU → output`

The `+ x` is the identity skip connection ("shortcut").

**Why this matters — the vanishing gradient problem:**
In a 50-layer network, the gradient at layer 1 is:
```
∂L/∂W_1 = ∂L/∂output × ∂output/∂W_N × ... × ∂W_2/∂W_1
```
This is a product of ~50 Jacobian matrices. If each is < 1 → gradients shrink to 0 → early layers never update.

**Skip connections create an alternative path**:
```
∂L/∂x = ∂L/∂F(x) × ∂F(x)/∂x + ∂L/∂x   ← identity shortcut contributes directly
```
The identity term `+ ∂L/∂x` means gradients can flow directly back without passing through any weight matrices → gradient highways.

**Result**: ResNet-50 (50 layers) trains stably and outperforms VGG-16 (16 layers). ResNet-152 (152 layers) is feasible. Without skip connections, these depths are untrainable.

For industrial QA, ResNet-50 pre-trained on ImageNet is the standard backbone for defect detection — fine-tuned on MVTec AD patches.

---

## 7. Transfer Learning Strategy

**Why fine-tune instead of train from scratch?**

MVTec AD has ~5,000 training images. A ResNet-50 has 25M parameters.
Training 25M parameters on 5,000 images → massive overfitting.

**Transfer learning**:
1. Start with ResNet-50 pre-trained on ImageNet (1.28M images, 1,000 classes)
2. The pre-trained weights already encode: edges, textures, shapes, object parts
3. Freeze early layers (they already detect useful low-level features)
4. Replace and train the classification head only (or fine-tune all layers with very small LR)

**Fine-tuning schedule for MVTec AD:**
- **Epoch 1–5**: Freeze all conv layers, train only the FC head (LR=1e-3)
- **Epoch 6–15**: Unfreeze last 2 conv blocks, train with LR=1e-4
- **Epoch 16–30**: Unfreeze all layers, LR=1e-5 (very small — don't destroy ImageNet features)

This approach reaches 90%+ accuracy on MVTec AD categories with far less data and compute than training from scratch.

---

## 8. Evaluation Metrics for Visual QA

Standard accuracy is often misleading for QA tasks — defects are rare:
- If 95% of products are normal, a model that always predicts "normal" gets 95% accuracy
- But it catches 0% of defects → useless

**Better metrics:**

| Metric | Formula | Interpretation for QA |
|--------|---------|----------------------|
| Precision | TP / (TP + FP) | Of flagged defects, how many were real? (False alarm rate) |
| Recall | TP / (TP + FN) | Of real defects, how many were caught? (Miss rate) |
| F1 Score | 2 × (P × R) / (P + R) | Balance of precision and recall |
| ROC-AUC | Area under ROC curve | Overall discrimination ability, threshold-independent |

**Industrial QA trade-off:**
- High recall (few misses) is usually more important — missing a defect can cause product failure in the field
- High precision (few false alarms) prevents unnecessary line stoppages
- The trade-off is controlled by the **decision threshold** on the model's probability output
- Threshold optimization using precision-recall curve is a key engineering decision
