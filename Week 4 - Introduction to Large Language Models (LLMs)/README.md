# Week 4 — Large Language Models & Transformer Architecture

##  Goals for This Week

- Understand the Transformer architecture at the mathematical level
- Implement scaled dot-product self-attention from scratch (NumPy)
- Understand how LLMs are trained (pre-training, fine-tuning, RLHF)
- Learn prompt engineering techniques for industrial QA applications
- Understand how LLMs fit into the multi-modal QA pipeline

---

##  Why LLMs in an Industrial QA System?

A CNN can tell you *that* a defect exists. An LLM can tell you:

1. **What** the defect likely is (from visual + sensor context)
2. **Why** it probably occurred (root cause reasoning)
3. **What to do** (recommended corrective action)
4. **Who to notify** (severity-based escalation)

This is the difference between an alarm and an *insight*. The LLM reasoning layer transforms raw model outputs into actionable intelligence — the kind a **quality engineer** would provide.

---

##  Concepts Covered

### 1. The Limitation of RNNs — Why Transformers Emerged

Before Transformers (2017), sequence modelling used **Recurrent Neural Networks (RNNs)** and **LSTMs**.

**RNN problems:**
- **Sequential processing**: Each token processed one at a time → can't be parallelized → slow training on long sequences
- **Vanishing gradients**: Gradients shrink as they backpropagate through time steps → early tokens in long sequences are effectively "forgotten"
- **Long-range dependencies**: An RNN processing a 1,000-word document struggles to connect information from position 1 to position 999

**The Transformer's answer:**
- **Parallel processing**: Process all tokens simultaneously via matrix operations
- **Self-attention**: Every token directly attends to every other token, regardless of distance → long-range dependencies are free

---

### 2. The Attention Mechanism — Mathematical Foundation

Attention asks: "For each position, which other positions are most relevant?"

Given an input sequence, we compute three matrices from the input embeddings:
- **Q (Queries)**: What am I looking for?
- **K (Keys)**: What information do I have?
- **V (Values)**: What should I actually return?

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax( QK^T / √d_k ) V
```

Where:
- `QK^T` computes pairwise similarity between every query-key pair → (seq_len × seq_len) matrix
- `/ √d_k` scales down to prevent softmax saturation (if d_k is large, dot products grow large → softmax becomes peaked → gradients vanish)
- `softmax(...)` converts similarities to weights summing to 1 → attention distribution
- `× V` is a weighted sum of values, weighted by the attention scores

**Intuition (for QA application):**
When the model processes "The bearing shows a scratch — check the conveyor speed":
- The word "scratch" attends heavily to "bearing" and "conveyor" → the model understands context
- Without attention, distant words have no direct connection

---

### 3. Multi-Head Attention

Instead of a single attention computation, use `h` attention heads in parallel, each with its own Q, K, V projection matrices:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

**Why multiple heads?**
Each head can learn to attend to different types of relationships simultaneously:
- Head 1: Syntactic relationships (subject-verb agreement)
- Head 2: Semantic similarity (defect types that co-occur)
- Head 3: Positional relationships (what comes after what)

---

### 4. The Full Transformer Block

Each Transformer layer (repeated N times, typically 12–96 in modern LLMs):

```
x → [Multi-Head Self-Attention] → Add & LayerNorm → [FFN] → Add & LayerNorm → output

FFN: FC(d_model → 4*d_model, ReLU) → FC(4*d_model → d_model)
Add & LayerNorm: Residual connection + layer normalization
```

**Residual connections** (the "Add" in "Add & LayerNorm"):
```
output = LayerNorm(x + Sublayer(x))
```
- Same idea as ResNet: provides gradient highways → enables training of very deep models
- Without residual connections, a 100-layer Transformer would suffer severe vanishing gradients

**Layer Normalization**: Normalizes across the feature dimension (not the batch dimension like BatchNorm). Better suited for variable-length sequences.

---

### 5. From Transformer to GPT (Decoder-Only LLM)

GPT-style models use the **Transformer decoder block** only, with **causal masking** (each token can only attend to previous tokens, not future ones). This makes it an **autoregressive** model: it generates one token at a time, each time seeing all previously generated tokens.

**Pre-training**: Trained on internet-scale text (~billions of tokens) with the **next-token prediction** objective:
```
Maximize P(token_t | token_1, ..., token_{t-1})
```
This simple objective forces the model to learn grammar, facts, reasoning, and even coding from raw text.

**Sizes of Notable GPT-style Models:**

| Model | Parameters | Context Length | Key Innovation |
|-------|-----------|---------------|---------------|
| GPT-2 (2019) | 1.5B | 1,024 | Showed emergent in-context learning |
| GPT-3 (2020) | 175B | 4,096 | Few-shot prompting without fine-tuning |
| LLaMA-2 (2023) | 7B–70B | 4,096 | Open-source, competitive with GPT-3.5 |
| GPT-4 (2023) | ~1T (est.) | 128K | Multimodal, professional-level reasoning |

---

### 6. Prompt Engineering for Industrial QA

LLMs are used in the pipeline as a **reasoning layer** that interprets multi-modal model outputs and produces actionable natural-language reports.

**Core prompting techniques practiced:**

| Technique | What It Does | QA Application |
|-----------|-------------|----------------|
| Zero-shot prompting | No examples, direct instruction | "Classify this defect type: [description]" |
| Few-shot prompting | Provide examples in the prompt | Show 3 labelled defect descriptions before new query |
| Chain-of-Thought (CoT) | "Think step by step" → better reasoning | Root cause analysis |
| System prompt | Set role and constraints | "You are a quality control AI assistant..." |
| Structured output | Request JSON/table format | Parse output programmatically |

**The QA defect reasoning prompt template (developed this week):**
```
System: You are an expert industrial quality control AI. Your task is to
        analyze sensor readings and visual inspection results to provide
        a structured fault report.

User: A metal nut on production line ST-04 has been flagged.
      Visual model output: scratch detected (confidence 0.87)
      Sensor readings: temperature=91°C, vibration=3.2mm/s, pressure=12.1bar
      
      Provide: (1) Defect type confirmation, (2) Severity assessment,
               (3) Probable root cause, (4) Recommended action.
```

---

### 7. The Emerging Field: Multi-Modal LLMs

Recent models (GPT-4V, LLaVA, Gemini) can process *both* images and text:
- The image is encoded by a vision encoder (ViT/CLIP)
- The visual tokens are projected into the LLM's embedding space
- The LLM reasons over both visual tokens and text tokens jointly

**This is the ultimate form of our QA pipeline:**
- Feed the defective product image + sensor data directly to a multi-modal LLM
- Get a full diagnostic report in natural language
- No separate CNN module needed (the LLM handles vision internally)

---

##  Code Written This Week

### `attention_mechanism.py`
- Scaled dot-product attention from scratch (NumPy)
- Multi-head attention implementation
- Visualization of attention weights on sample QA text
- Positional encoding implementation

### `transformer_concepts.md`
- Complete Transformer architecture walkthrough
- Layer-by-layer mathematical derivation
- GPT vs BERT vs T5 architecture comparison
- Tokenization (BPE, WordPiece) explanation

### `llm_prompting.py`
- Prompt templates for QA defect analysis
- Zero-shot, few-shot, and chain-of-thought examples
- Structured output prompting (JSON format)
- Using the OpenAI API for QA defect reasoning

---

##  Resources Used

| Resource | Link |
|----------|------|
| Intro to LLM (Andrej Karpathy) | [YouTube](https://www.youtube.com/watch?v=zjkBMFhNj_g) |
| LLM Tutorial | [YouTube](https://www.youtube.com/watch?v=xZDB1naRUlk) |
| 3Blue1Brown — Transformers (Optional) | [YouTube Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — Chapters 5 & 6 (Attention) |

---

## Takeaway

> The Transformer's self-attention mechanism solves a fundamental problem: how to let every element in a sequence communicate with every other element in O(1) "communication steps" instead of O(n) (RNN). The price is O(n²) memory for the attention matrix — which is why context length is the key engineering constraint in modern LLMs. For our QA use case, contexts are short (sensor readings + defect description), so this is not a bottleneck.
