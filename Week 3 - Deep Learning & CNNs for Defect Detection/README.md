# Week 3 — Deep Learning & CNNs for Visual Defect Detection

##  Goals for This Week

- Understand neural networks from the mathematics of a single neuron upward
- Implement backpropagation from scratch — the algorithm that makes deep learning work
- Understand how Convolutional Neural Networks (CNNs) process image data
- Build a CNN in Keras for image-based defect classification
- Understand why CNNs outperform flat (fully connected) networks on images

---

##  Concepts Covered

### 1. The Artificial Neuron — Mathematical Foundation

A single neuron computes:

```
output = activation( Σ(w_i * x_i) + b )
       = activation( w^T x + b )
```

The activation function is what gives neural networks their power. Without it, stacking layers would just be a chain of matrix multiplications — which simplifies to a *single* matrix multiplication (a linear model). Non-linear activations allow networks to approximate *any* function.

| Activation | Formula | Used For | Properties |
|-----------|---------|----------|------------|
| Sigmoid | `1 / (1 + e^{-z})` | Output layer (binary) | Saturates → vanishing gradients |
| Tanh | `(e^z - e^{-z}) / (e^z + e^{-z})` | Hidden layers (older) | Zero-centered, still saturates |
| ReLU | `max(0, z)` | Hidden layers (default) | No saturation for z>0, fast |
| Leaky ReLU | `z if z>0 else 0.01z` | Hidden layers | Fixes "dying ReLU" problem |
| Softmax | `e^{z_i} / Σ e^{z_j}` | Multi-class output | Outputs sum to 1 → probabilities |

**The Dying ReLU Problem**: If a neuron's weights are initialized such that it always receives negative pre-activation input, its gradient is always zero → it never updates → it "dies." Leaky ReLU and proper initialization (He initialization) mitigate this.

---

### 2. Backpropagation — How Networks Learn

Backpropagation is **the chain rule of calculus applied recursively** through the network graph, computing `∂Loss / ∂w` for every weight.

Forward pass:
```
z^[l] = W^[l] a^[l-1] + b^[l]
a^[l] = g(z^[l])
```

Backward pass (chain rule):
```
δ^[L] = ∂Loss/∂a^[L] ⊙ g'(z^[L])          (output layer error)
δ^[l] = (W^[l+1]^T δ^[l+1]) ⊙ g'(z^[l])  (backpropagate error)
∂L/∂W^[l] = δ^[l] (a^[l-1])^T              (weight gradient)
∂L/∂b^[l] = δ^[l]                           (bias gradient)
```

The key insight: the gradient of a weight in an early layer depends on ALL the weights in later layers (via the chain rule). This is why deep networks were historically hard to train — gradients either explode or vanish as they backpropagate through many layers.

**Solutions to vanishing/exploding gradients**:
- Batch Normalization: normalizes activations, keeps them in the "linear regime" of activations
- Residual connections (ResNet): create "gradient highways" that allow gradients to flow directly
- Careful initialization: He initialization for ReLU (`std = sqrt(2/n_in)`)
- Gradient clipping: caps the norm of gradients to prevent explosion

---

### 3. Convolutional Neural Networks — Why Images Need Special Treatment

**The problem with fully connected networks on images:**

A 256×256 RGB image has `256 × 256 × 3 = 196,608` pixels. A fully connected layer with 1024 units would need `196,608 × 1024 ≈ 200 million weights` in the first layer alone. This is:
- Computationally expensive
- Prone to extreme overfitting (too many parameters relative to training samples)
- Ignores spatial structure (treats pixel at position (0,0) as completely unrelated to (0,1))

**CNNs solve this by exploiting two key properties of images:**
1. **Local connectivity**: A defect (scratch, dent) is spatially localized — nearby pixels are more related than distant ones. A `3×3` convolutional filter only looks at a 9-pixel neighbourhood.
2. **Translation equivariance**: A scratch in the top-left corner is the same type of defect as a scratch in the bottom-right. The *same filter weights* should detect it anywhere → weight sharing.

---

### 4. The Convolution Operation

A 2D convolution slides a kernel (filter) over the image and computes dot products:

```
Output[i,j] = Σ_{m,n} Input[i+m, j+n] * Kernel[m, n]
```

For a `3×3` kernel on a `256×256` image:
- Output size (same padding): `256×256`
- Output size (valid padding): `254×254`
- Parameters: `3×3 = 9` weights (shared across all positions!) vs `256×256 = 65,536` for a fully connected approach

A layer with 64 such filters learns 64 different visual patterns. Early layers learn edges and textures; deeper layers combine these into complex shapes and objects.

**Key CNN operations:**

| Operation | Purpose | Effect on spatial dims |
|-----------|---------|----------------------|
| Conv2D (same padding) | Feature extraction | Preserves H×W |
| Conv2D (valid padding) | Feature extraction | Reduces H×W |
| MaxPooling2D (2×2) | Spatial downsampling + translation invariance | Halves H and W |
| Dropout | Regularization — randomly zeros activations | No spatial change |
| Batch Normalization | Stabilizes training, allows higher LR | No spatial change |
| Flatten | Converts 3D feature maps to 1D vector for FC layers | H×W×C → HWC |

---

### 5. Training a CNN — The Full Pipeline

1. **Data preprocessing**: Resize → normalize → augment
2. **Architecture design**: Number of conv blocks, filter sizes, FC head
3. **Loss function**: Binary CE (defect/normal) or Categorical CE (defect type)
4. **Optimizer**: Adam (adaptive LR, momentum) — standard for most vision tasks
5. **Regularization**: Dropout, BatchNorm, data augmentation, weight decay
6. **Learning rate schedule**: Reduce on plateau or cosine annealing
7. **Early stopping**: Stop when validation loss stops improving

**Data Augmentation** is critical in industrial QA:
- We rarely have thousands of defect images (defects are rare events)
- Augmentation artificially multiplies the training set: horizontal flip, rotation, brightness jitter, adding Gaussian noise
- Teaches the model that defect identity doesn't change under these transforms

---

##  Code Written This Week

### `neural_network_scratch.py`
- Complete neural network class with forward pass and backpropagation
- Trained on the QA sensor dataset from Week 2 (tabular → MLP)
- Demonstrates gradient descent loop, loss tracking

### `cnn_keras.py`
- CNN built in Keras for image-based defect classification
- Trained on CIFAR-10 as a proxy (real MVTec AD dataset in final week)
- Includes data augmentation, Batch Normalization, Dropout
- Plots training curves, confusion matrix, and sample predictions

### `cnn_concepts.md`
- Detailed notes on convolution, receptive fields, feature maps
- Comparison: vanilla CNN vs ResNet-style skip connections

---

##  Resources Used

| Resource | Link |
|----------|------|
| Deep Learning & CNN Playlist (Dhaval Patel) | [YouTube Playlist](https://www.youtube.com/playlist?list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO) — Watched till video 32 |

---

##  Key Takeaway

> A CNN is not magic — it's a carefully designed inductive bias. By hardcoding the assumptions that visual patterns are local and translation-invariant, we go from 200M parameters to 200K. **This is the fundamental lesson of deep learning architecture design: good inductive biases are worth more than raw parameter count.**
