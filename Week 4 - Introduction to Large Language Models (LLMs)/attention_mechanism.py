"""
Week 4: Self-Attention & Multi-Head Attention from Scratch
===========================================================
"Attention Is All You Need" (Vaswani et al., 2017) is arguably the most
important paper in modern AI. This file implements the core mechanism
from scratch using NumPy, making every matrix operation explicit.

Goal: Build deep intuition for:
  1. Scaled dot-product attention
  2. Multi-head attention
  3. Positional encoding (how Transformers handle sequence order)
  4. The attention pattern visualization

After working through this file, the Transformer architecture in
frameworks like HuggingFace is no longer a black box.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)


# =============================================================================
# SECTION 1: TOKEN EMBEDDING (SIMPLIFIED)
# =============================================================================

def simple_tokenize(text: str) -> list[str]:
    """
    Simplified tokenizer: split by space.
    Real tokenizers (BPE, WordPiece) handle subwords to manage vocabulary size.
    """
    return text.lower().replace(',', '').replace('.', '').split()


def create_embedding_matrix(vocab: list[str], d_model: int = 8) -> dict[str, np.ndarray]:
    """
    Create random embeddings (in reality, learned during training).
    
    An embedding maps a discrete token (word) to a dense vector in R^d_model.
    d_model is typically 512–4096 in real LLMs.
    We use d_model=8 here for visualization clarity.
    """
    return {word: np.random.randn(d_model) for word in vocab}


# A sample sentence from the QA domain
sentence = "bearing temperature high vibration detected defect scratch"
tokens = simple_tokenize(sentence)
print(f"Tokens: {tokens}")

d_model = 16   # Embedding dimension (using small value for clarity)
seq_len = len(tokens)

# Create random "learned" embeddings
vocab = list(set(tokens))
embedding_matrix = create_embedding_matrix(vocab, d_model)

# Stack token embeddings into input matrix X: (seq_len, d_model)
X = np.array([embedding_matrix[t] for t in tokens])  # (7, 16)
print(f"\nInput embedding matrix shape: {X.shape}  (seq_len={seq_len}, d_model={d_model})")


# =============================================================================
# SECTION 2: POSITIONAL ENCODING
# =============================================================================

def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding (original "Attention Is All You Need" formula).

    The Transformer processes all tokens in PARALLEL — it has no inherent notion
    of order. Positional encodings inject position information by adding
    a unique signal to each position's embedding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Why sinusoidal?
    - PE(pos + k) can be expressed as a linear function of PE(pos) → the model
      can easily learn relative positions
    - Generalizes to sequences longer than seen in training (unlike learned PEs)
    - Different frequencies capture positional relationships at different scales

    Args:
        seq_len: Number of tokens in the sequence
        d_model: Embedding dimension

    Returns:
        PE: (seq_len, d_model) positional encoding matrix
    """
    PE = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len)[:, np.newaxis]         # (seq_len, 1)
    dim_pairs  = np.arange(0, d_model, 2)[np.newaxis, :]  # (1, d_model//2)

    # Denominator: 10000^(2i/d_model)
    div_term = np.power(10000, dim_pairs / d_model)

    PE[:, 0::2] = np.sin(positions / div_term)   # Even dimensions: sin
    PE[:, 1::2] = np.cos(positions / div_term)   # Odd dimensions: cos

    return PE

PE = positional_encoding(seq_len, d_model)
X_with_pos = X + PE    # Add positional encoding to token embeddings
print(f"\nAfter positional encoding: {X_with_pos.shape}")


# =============================================================================
# SECTION 3: SCALED DOT-PRODUCT ATTENTION
# =============================================================================

def scaled_dot_product_attention(Q: np.ndarray,
                                  K: np.ndarray,
                                  V: np.ndarray,
                                  mask: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Core attention mechanism: Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Step-by-step:
    1. Compute raw attention scores: QK^T  — shape (seq_len, seq_len)
       score[i,j] = dot product between query_i and key_j
       = "how much does position i want to attend to position j?"

    2. Scale by √d_k:
       Without scaling, for large d_k, the dot products grow large in magnitude.
       This pushes softmax into regions where gradients are tiny (saturation).
       Scaling keeps values in the "gentle slope" region.

    3. Apply mask (optional):
       In GPT-style decoders, mask future positions (set to -∞ before softmax)
       so the model can only attend to past and present tokens.
       In BERT/encoders, no mask → full attention to all positions.

    4. Softmax to get attention weights:
       Converts scores to probabilities summing to 1 along each row.

    5. Weighted sum of values:
       output = attention_weights × V
       Each output token is a weighted combination of all value vectors,
       weighted by how much that token attends to each position.

    Args:
        Q: Query matrix (seq_len, d_k)
        K: Key matrix (seq_len, d_k)
        V: Value matrix (seq_len, d_v)
        mask: Optional causal mask (seq_len, seq_len) — -inf for masked positions

    Returns:
        output: (seq_len, d_v) attended values
        attention_weights: (seq_len, seq_len) attention distribution (for visualization)
    """
    d_k = Q.shape[-1]

    # Step 1: Raw attention scores
    scores = Q @ K.T                    # (seq_len, seq_len)

    # Step 2: Scale
    scores = scores / np.sqrt(d_k)     # Prevents softmax saturation

    # Step 3: Apply causal mask (if provided)
    if mask is not None:
        scores = scores + mask          # Masked positions → -∞ → softmax → 0

    # Step 4: Softmax (stable implementation)
    scores_shifted = scores - scores.max(axis=-1, keepdims=True)  # Numerical stability
    exp_scores = np.exp(scores_shifted)
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Step 5: Weighted sum
    output = attention_weights @ V      # (seq_len, d_v)

    return output, attention_weights


# Create Q, K, V projection matrices (normally learned; here initialized randomly)
d_k = d_v = d_model // 2   # Projection dimension (typically d_model / n_heads)

W_Q = np.random.randn(d_model, d_k) * 0.1
W_K = np.random.randn(d_model, d_k) * 0.1
W_V = np.random.randn(d_model, d_v) * 0.1

# Project input to Q, K, V
Q = X_with_pos @ W_Q    # (seq_len, d_k)
K = X_with_pos @ W_K    # (seq_len, d_k)
V = X_with_pos @ W_V    # (seq_len, d_v)

# Compute attention (encoder-style: no causal mask)
attended_output, attention_weights = scaled_dot_product_attention(Q, K, V)

print(f"\n--- Single-Head Attention ---")
print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
print(f"Attention weights shape: {attention_weights.shape}")   # (7, 7)
print(f"Output shape: {attended_output.shape}")                # (7, 8)


# =============================================================================
# SECTION 4: MULTI-HEAD ATTENTION
# =============================================================================

class MultiHeadAttention:
    """
    Multi-head attention: Run h attention heads in parallel, each projecting
    to lower-dimensional subspaces, then concatenate and project back.

    Each head can specialize in different types of relationships:
    - One head might track syntactic structure
    - Another might track semantic similarity
    - Another might track positional proximity

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    """

    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Each head has its own Q, K, V projection matrices
        # In practice, implemented as one large matrix + splitting — more efficient
        self.W_Qs = [np.random.randn(d_model, self.d_head) * 0.1 for _ in range(n_heads)]
        self.W_Ks = [np.random.randn(d_model, self.d_head) * 0.1 for _ in range(n_heads)]
        self.W_Vs = [np.random.randn(d_model, self.d_head) * 0.1 for _ in range(n_heads)]
        self.W_O  = np.random.randn(d_model, d_model) * 0.1   # Output projection

        self.all_attention_weights = []   # Store for visualization

    def forward(self, X: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Args:
            X: Input embeddings (seq_len, d_model)
            mask: Optional causal mask

        Returns:
            Output (seq_len, d_model) — same shape as input (residual connection-ready)
        """
        self.all_attention_weights = []
        head_outputs = []

        for i in range(self.n_heads):
            Q_i = X @ self.W_Qs[i]   # (seq_len, d_head)
            K_i = X @ self.W_Ks[i]
            V_i = X @ self.W_Vs[i]

            out_i, attn_i = scaled_dot_product_attention(Q_i, K_i, V_i, mask)
            head_outputs.append(out_i)
            self.all_attention_weights.append(attn_i)

        # Concatenate all head outputs: (seq_len, n_heads * d_head) = (seq_len, d_model)
        concatenated = np.concatenate(head_outputs, axis=-1)

        # Final output projection: mixes information across heads
        output = concatenated @ self.W_O   # (seq_len, d_model)
        return output


# Multi-head attention with 4 heads
mha = MultiHeadAttention(d_model=d_model, n_heads=4)
mha_output = mha.forward(X_with_pos)

print(f"\n--- Multi-Head Attention (4 heads) ---")
print(f"Output shape: {mha_output.shape}")   # Should be (7, 16) = (seq_len, d_model)


# =============================================================================
# SECTION 5: CAUSAL MASK (GPT-style Decoder)
# =============================================================================

def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create a lower-triangular mask for autoregressive (decoder-only) models.

    GPT can only attend to PAST and CURRENT tokens, not FUTURE ones.
    This ensures the model never "sees the answer" during training.

    Mask:
        0   where attention is allowed (lower triangle)
        -inf where attention is blocked (upper triangle, not including diagonal)

    After softmax:
        softmax(-inf) ≈ 0 → blocked positions contribute nothing to output
    """
    mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
    # k=1: ones above the main diagonal (future positions only)
    return mask


causal_mask = create_causal_mask(seq_len)
print(f"\nCausal mask (lower-triangular):")
print(np.where(causal_mask == 0, 0, -np.inf))   # Display 0 and -∞

# Apply GPT-style masked attention
_, causal_attention_weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
print(f"\nCausal attention weights (note: upper triangle is 0):")
print(causal_attention_weights.round(3))


# =============================================================================
# SECTION 6: FEED-FORWARD NETWORK (FFN)
# =============================================================================

def feed_forward_network(X: np.ndarray, d_model: int, d_ff: int = None) -> np.ndarray:
    """
    The FFN applied to each position independently:
      FFN(x) = ReLU(xW_1 + b_1) W_2 + b_2

    d_ff is typically 4 * d_model (a design choice from the original paper).
    
    The FFN is where the model stores "knowledge" (factual associations).
    The attention layer routes information; the FFN processes it.
    
    Roughly: attention = "who to talk to", FFN = "what to say"
    """
    if d_ff is None:
        d_ff = 4 * d_model

    W1 = np.random.randn(d_model, d_ff) * 0.1
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * 0.1
    b2 = np.zeros(d_model)

    # First linear + ReLU
    hidden = np.maximum(0, X @ W1 + b1)   # (seq_len, d_ff)
    # Second linear
    output = hidden @ W2 + b2             # (seq_len, d_model)
    return output


# =============================================================================
# SECTION 7: ONE COMPLETE TRANSFORMER BLOCK
# =============================================================================

def layer_norm(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Layer Normalization: normalize across feature dimension.
    
    Unlike BatchNorm (normalizes across batch), LayerNorm normalizes across features.
    This makes it independent of batch size — critical for:
    - Variable-length sequences
    - Small batches
    - Inference with batch_size=1
    """
    mean = X.mean(axis=-1, keepdims=True)
    std  = X.std(axis=-1, keepdims=True)
    return (X - mean) / (std + eps)


def transformer_block(X: np.ndarray, n_heads: int = 4, d_ff: int = None) -> np.ndarray:
    """
    One complete Transformer encoder block:

    X → MHA → Residual + LayerNorm → FFN → Residual + LayerNorm → output

    Residual connections:
      output = LayerNorm(X + Sublayer(X))

    Why residual connections are critical:
    - They create gradient highways: ∂L/∂X can flow directly back without
      passing through attention/FFN layers
    - Without them, 12+ layer Transformers would have vanishing gradients
    - Same idea as ResNet: identity shortcut + learned residual
    """
    d_model = X.shape[-1]
    if d_ff is None:
        d_ff = 4 * d_model

    # Multi-Head Self-Attention with residual + LayerNorm
    mha_layer = MultiHeadAttention(d_model, n_heads)
    attn_output = mha_layer.forward(X)
    X = layer_norm(X + attn_output)          # Add & Norm

    # Feed-Forward Network with residual + LayerNorm
    ffn_output = feed_forward_network(X, d_model, d_ff)
    X = layer_norm(X + ffn_output)           # Add & Norm

    return X


# Run a complete transformer block
output = transformer_block(X_with_pos, n_heads=4)
print(f"\n--- Complete Transformer Block ---")
print(f"Input shape:  {X_with_pos.shape}")
print(f"Output shape: {output.shape}")   # Same as input — ready for next block


# =============================================================================
# SECTION 8: VISUALIZATION — ATTENTION HEATMAP
# =============================================================================

# Re-run single-head attention to get clean attention weights for visualization
Q_vis = X_with_pos @ W_Q
K_vis = X_with_pos @ W_K
V_vis = X_with_pos @ W_V
_, attn_vis = scaled_dot_product_attention(Q_vis, K_vis, V_vis)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Self-Attention in a QA Domain Sentence', fontsize=13, fontweight='bold')

# Attention heatmap
ax = axes[0]
im = ax.imshow(attn_vis, cmap='viridis', aspect='auto')
ax.set_xticks(range(seq_len))
ax.set_yticks(range(seq_len))
ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(tokens, fontsize=9)
ax.set_xlabel('Keys (Source positions)')
ax.set_ylabel('Queries (Target positions)')
ax.set_title('Attention Weights: Row i attends to column j')
plt.colorbar(im, ax=ax, label='Attention Weight')

# Annotate each cell
for i in range(seq_len):
    for j in range(seq_len):
        ax.text(j, i, f'{attn_vis[i,j]:.2f}', ha='center', va='center',
                fontsize=7, color='white' if attn_vis[i,j] < 0.2 else 'black')

# Causal mask visualization
ax2 = axes[1]
im2 = ax2.imshow(causal_attention_weights, cmap='Blues', aspect='auto')
ax2.set_xticks(range(seq_len))
ax2.set_yticks(range(seq_len))
ax2.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels(tokens, fontsize=9)
ax2.set_title('Causal (Masked) Attention — GPT-style Decoder\n(Upper triangle = 0: future tokens are invisible)')
plt.colorbar(im2, ax=ax2, label='Attention Weight')

plt.tight_layout()
plt.savefig('attention_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nAttention visualization saved to attention_heatmap.png")

# Print insight
print("\n--- Attention Analysis ---")
print("In the QA sentence:", ' '.join(tokens))
print("\nFor each token, highest-attention source token:")
for i, token in enumerate(tokens):
    top_source = tokens[np.argmax(attn_vis[i])]
    print(f"  '{token}' attends most to '{top_source}' (weight: {attn_vis[i].max():.3f})")

print("\n[Key Insight] The attention mechanism allows 'scratch' to directly")
print("attend to 'bearing' and 'defect', regardless of their distance in the")
print("sequence. In an RNN, this relationship would be diluted over 3+ steps.")
