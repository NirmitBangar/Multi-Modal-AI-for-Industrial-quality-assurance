# Transformer Architecture — Deep Dive Notes
## Week 4: LLM Fundamentals

---

## 1. The Big Picture: Why Transformers Replaced Everything

| Architecture | Sequential? | Long-range dependencies | Parallelizable | Year |
|---|---|---|---|---|
| RNN | Yes | Poor (vanishing gradients) | No | 1986 |
| LSTM | Yes | Better (gating) | No | 1997 |
| Transformer (Encoder-Decoder) | No | Perfect (direct attention) | Yes | 2017 |
| GPT (Decoder-only) | No | Perfect | Yes | 2018 |
| BERT (Encoder-only) | No | Perfect (bidirectional) | Yes | 2018 |

The 2017 paper "Attention Is All You Need" replaced both the encoder and decoder of the RNN-based seq2seq model with pure attention layers. The result trained faster, scaled better, and achieved superior performance across virtually every NLP benchmark.

---

## 2. Complete Transformer Architecture

### 2.1 Input Pipeline

```
Raw Text  →  Tokenizer  →  Token IDs  →  Embedding Lookup  →  + Positional Encoding  →  Transformer Blocks
```

**Tokenization** (Byte Pair Encoding — BPE):
- Vocabulary size is typically 30,000–100,000 tokens
- Subword units: "bearing" → ["bear", "ing"] or ["bearing"] depending on frequency
- Rare words are split into more frequent subwords → handles out-of-vocabulary words
- Numbers often split digit by digit: "1234" → ["1", "2", "3", "4"]

**Embedding Layer**:
- Maps each token ID to a dense vector of dimension `d_model`
- These embeddings are **learned** during pre-training
- Dimensionality: GPT-2 uses `d_model=768`; GPT-3 uses `d_model=12288`
- The embedding matrix has shape `(vocab_size, d_model)` — billions of operations affect it

**Positional Encoding** (PE):
- Transformers process all tokens in parallel → no inherent order
- PE injects positional information by adding a unique signal to each position
- Two main approaches:
  - Sinusoidal (original paper): fixed, mathematical formula
  - Learned PE (GPT-2+): trainable embedding per position
  - RoPE (LLaMA, GPT-NeoX): Rotary Position Embedding — encodes relative positions into the attention computation directly

---

### 2.2 The Transformer Block (Repeated N Times)

```
Input X (seq_len × d_model)
     │
     ├─────────────────────────┐
     │                         │ (Residual)
     ▼                         │
[Multi-Head Self-Attention]    │
     │                         │
     └────────────────────────►┤
                               │
                    [Layer Norm] ── X'
                               │
     ┌─────────────────────────┘
     │                         │ (Residual)
     ▼                         │
[Feed-Forward Network]         │
     │                         │
     └────────────────────────►┤
                               │
                    [Layer Norm] ── Output
```

**Hyperparameter choices in notable models:**

| Model | N (layers) | d_model | n_heads | d_ff | Total Params |
|-------|-----------|---------|---------|------|-------------|
| GPT-2 (small) | 12 | 768 | 12 | 3072 | 117M |
| GPT-2 (XL) | 48 | 1600 | 25 | 6400 | 1.5B |
| GPT-3 | 96 | 12288 | 96 | 49152 | 175B |
| LLaMA-2 7B | 32 | 4096 | 32 | 11008 | 7B |
| LLaMA-2 70B | 80 | 8192 | 64 | 28672 | 70B |

---

### 2.3 Feed-Forward Network — Where "Knowledge" Lives

The FFN has the structure:
```
FFN(x) = W_2 · ReLU(W_1 x + b_1) + b_2
          OR
FFN(x) = W_2 · GeLU(W_1 x + b_1) + b_2   (used in GPT-2+)
```

- `d_ff = 4 × d_model` in the original paper (a ratio that has held up empirically)
- Accounts for ~2/3 of all parameters in a Transformer
- Recent research (Anthropic, DeepMind) shows FFN layers store factual memories:
  "Paris is the capital of France" is encoded as a (key, value) pair in the FFN weights
- The attention mechanism *routes* — deciding which FFN "memories" to activate

**GeLU vs ReLU:**
- GeLU (Gaussian Error Linear Unit): `x · Φ(x)` where Φ is the Gaussian CDF
- Smoother than ReLU (no hard zero at x=0) → slightly better empirical performance
- Used in BERT, GPT-2/3/4, and most modern LLMs

---

### 2.4 Layer Normalization

Original Transformer used **Post-LN** (LayerNorm after residual):
```
x_out = LayerNorm(x + Sublayer(x))
```

Modern LLMs (GPT-2+) use **Pre-LN** (LayerNorm before sublayer):
```
x_out = x + Sublayer(LayerNorm(x))
```

Pre-LN trains more stably (gradients are better conditioned) and is now the default.

**RMSNorm** (LLaMA, T5): Simplified LayerNorm that only scales (no shift), using RMS instead of full standard deviation. ~10% faster with similar performance.

---

## 3. GPT vs BERT vs T5 — Architecture Comparison

| Aspect | GPT (Decoder) | BERT (Encoder) | T5 (Enc-Dec) |
|--------|--------------|----------------|--------------|
| Attention | Causal (masked) | Bidirectional | Both |
| Training objective | Next token prediction | Masked language model | Text-to-text |
| Context | Left-to-right only | Full bidirectional | Bidirectional encoder |
| Best for | Generation tasks | Classification, understanding | Any task (unified format) |
| Examples | GPT-4, LLaMA, Claude | BERT, RoBERTa, ALBERT | T5, FLAN-T5, mT5 |

**For our QA application:**
- **BERT-style**: Better for classifying defect type from description (understanding)
- **GPT-style**: Better for generating defect reports from structured input (generation)
- **Our approach**: Use a GPT-style API for flexible report generation

---

## 4. How LLMs Are Trained — The Three Phases

### Phase 1: Pre-training (Unsupervised, Very Expensive)

**Objective**: Predict the next token, given all previous tokens.
```
Loss = -Σ log P(token_t | token_1, ..., token_{t-1})
```

**Data**: CommonCrawl, Wikipedia, GitHub, books, arXiv (~trillions of tokens)
**Compute**: GPT-3 cost ~$4.6M to pre-train on 10,000 A100 GPUs for months
**What the model learns**: Grammar, facts, reasoning patterns, coding, math — all from the single objective of predicting the next word.

This is the **foundation model** — it knows a lot but doesn't know how to follow instructions yet.

### Phase 2: Supervised Fine-tuning (SFT)

Fine-tune on high-quality (instruction, response) pairs:
```
Input: "Classify this defect: scratch on bearing surface"
Output: "Defect Type: Surface scratch | Severity: High | Category: Mechanical wear"
```

Teaches the model to format outputs helpfully and follow instructions.
Much cheaper than pre-training: typically 1,000–100,000 examples.

### Phase 3: RLHF — Reinforcement Learning from Human Feedback

1. Generate multiple responses for each prompt
2. Human annotators rank responses from best to worst
3. Train a **Reward Model** to predict human preference scores
4. Fine-tune the LLM using PPO (Proximal Policy Optimization) to maximize reward while not diverging too far from SFT model (KL penalty)

This phase makes the model **helpful, harmless, and honest**.
Used by GPT-4, Claude, Gemini, LLaMA-2-Chat.

---

## 5. Tokenization — A Deeper Look

### Byte Pair Encoding (BPE)

Algorithm:
1. Start with character-level vocabulary
2. Count all adjacent pair frequencies in corpus
3. Merge the most frequent pair into a new token
4. Repeat N times (until vocabulary size is reached)

Example (simplified):
- Start: `['b','e','a','r','i','n','g']`
- After many merges: `['bearing']` (if frequent enough) or `['bear', 'ing']`

**Why this matters for QA text:**
- Domain-specific terms like "spindle", "deburring", "anodizing" may be split into unfamiliar subwords
- For production use, domain-adaptive tokenization or fine-tuning helps
- Rare technical terms cost more tokens → more expensive API calls

---

## 6. Attention Complexity and Context Length

Self-attention has **O(n²)** memory complexity in sequence length n:
- The attention matrix is (n × n)
- For n=4096 (GPT-3 context): 4096² = 16.8M elements per layer per batch
- For n=128K (GPT-4 Turbo): 128K² ≈ 16.4B elements — requires special sparse/linear attention tricks

**Approximate attention methods** (reduce O(n²) to O(n log n) or O(n)):
- FlashAttention: Reorders computation to reduce memory bandwidth (same result, faster)
- Sliding Window Attention (Mistral): Each token attends to only a local window
- Sparse Attention: Attend to selected positions only (local + global)

**For our QA use case**: Sensor readings + defect descriptions fit in <500 tokens easily. Context length is not a bottleneck — even the smallest API models are more than sufficient.

---

## 7. The Emerging Paradigm: Multi-Modal LLMs

Modern systems like GPT-4V, LLaVA, Gemini, and Claude integrate vision and language:

```
Image → Vision Encoder (ViT/CLIP) → Visual Tokens
                                          │
Text  → Text Tokenizer → Text Tokens ────┤
                                          ▼
                              LLM (processes combined token sequence)
                                          │
                                          ▼
                            "Defect: Surface scratch on bearing
                             Severity: High
                             Probable cause: Mechanical wear or contamination
                             Action: Remove part for detailed inspection"
```

**Vision Encoder Options:**
- **CLIP** (OpenAI): Trained on 400M image-text pairs using contrastive learning
- **ViT** (Google): Transformer applied directly to image patches (no convolutions!)
- **DINOv2** (Meta): Self-supervised ViT, excellent visual features without labels

**Projection Layer**: Maps visual token embeddings to the same dimensionality as text embeddings, so the LLM can process them in a shared token sequence.

This architecture is the **end state of our project's vision** — a system that takes a product image and sensor readings, processes both through a multi-modal LLM, and produces a structured defect report.

---

## 8. Key Takeaways for the Project

1. **Attention = routing mechanism**: decides which information to gather from context
2. **FFN = memory mechanism**: where factual knowledge and patterns are stored
3. **Scale is predictable**: Chinchilla scaling laws tell us how performance scales with compute and data — no mystery, just engineering
4. **Prompt engineering is powerful**: A well-crafted prompt can unlock sophisticated reasoning without any fine-tuning
5. **Multi-modal is the future**: The QA pipeline's ultimate form is a multi-modal LLM that handles vision and text jointly — and this technology exists today
