# building-tranformers-from-scratch
A minimal Transformer language model trained from scratch on a tiny dataset
# Tiny Transformer Language Model — From Scratch

This repository documents and implements a **minimal Transformer-based language model**, trained from scratch on a very small dataset.

The purpose is **understanding**, not scale or performance.

No pretrained weights.  
No prompt engineering.  
No frameworks hiding the math.

---

## 1. What Is a Language Model?

A language model assigns probabilities to token sequences:

\[
P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})
\]

Training means learning these conditional distributions from data.

Text generation is sampling from them.

---

## 2. Tokens and Vocabulary

Raw text is converted into **tokens**.

In this project we use **character-level tokenization**.

Given text:


Vocabulary: [' ', 'a', 'e', 'i', 'l', 'n', 'r', 's']

Mapping:

\[
\text{stoi}: \text{char} \rightarrow \mathbb{N}
\quad,\quad
\text{itos}: \mathbb{N} \rightarrow \text{char}
\]

Token sequence:

\[
x = [x_1, x_2, \ldots, x_T]
\quad,\quad
x_i \in \{0, \ldots, V-1\}
\]

where \( V \) is vocabulary size.

---

## 3. Embeddings

Tokens are discrete. Neural networks require vectors.

Each token index \( x_t \) is mapped to an embedding vector:

\[
\mathbf{e}_t = E[x_t]
\quad,\quad
E \in \mathbb{R}^{V \times d}
\]

where:
- \( d \) = embedding dimension

---

## 4. Positional Encoding

Transformers have **no recurrence** or **convolution**.  
Order must be injected.

We use **learned positional embeddings**:

\[
\mathbf{p}_t = P[t]
\quad,\quad
P \in \mathbb{R}^{T \times d}
\]

Final input representation:

\[
\mathbf{h}_t^{(0)} = \mathbf{e}_t + \mathbf{p}_t
\]

---

## 5. Self-Attention: Core Mechanism

For each position \( t \), compute:

\[
\mathbf{q}_t = W_Q \mathbf{h}_t
\quad
\mathbf{k}_t = W_K \mathbf{h}_t
\quad
\mathbf{v}_t = W_V \mathbf{h}_t
\]

Stacked as matrices:

\[
Q, K, V \in \mathbb{R}^{T \times d}
\]

---

### Attention Scores

\[
A = \frac{QK^T}{\sqrt{d}}
\]

This measures **token-to-token similarity**.

---

### Causal Masking

Language models **cannot see the future**.

We apply a lower-triangular mask \( M \):

\[
A_{ij} =
\begin{cases}
\frac{q_i \cdot k_j}{\sqrt{d}}, & j \le i \\
-\infty, & j > i
\end{cases}
\]

---

### Softmax

\[
\alpha_{ij} = \frac{e^{A_{ij}}}{\sum_k e^{A_{ik}}}
\]

Each row sums to 1.

---

### Weighted Sum

\[
\mathbf{z}_i = \sum_j \alpha_{ij} \mathbf{v}_j
\]

This is **contextualization**: token meanings depend on prior tokens.

---

## 6. Feed-Forward Network

Applied independently at each position:

\[
\text{FFN}(x) =
W_2 \sigma(W_1 x + b_1) + b_2
\]

Typically:

\[
d \rightarrow 4d \rightarrow d
\]

Adds **non-linearity and transformation power**.

---

## 7. Residual Connections & Layer Norm

Each sub-layer uses:

\[
x := x + \text{sublayer}(\text{LayerNorm}(x))
\]

This:
- stabilizes gradients
- allows deeper models
- improves optimization

---

## 8. Transformer Block

One block:

1. LayerNorm
2. Self-Attention
3. Residual Add
4. LayerNorm
5. Feed-Forward
6. Residual Add

Stack \( N \) blocks.

---

## 9. Output Projection

Final hidden states:

\[
H \in \mathbb{R}^{T \times d}
\]

Project to vocabulary:

\[
\text{logits} = HW_O
\quad,\quad
W_O \in \mathbb{R}^{d \times V}
\]

---

## 10. Loss Function

Training objective: **cross-entropy loss**

\[
\mathcal{L}
=
-\sum_{t=1}^{T}
\log P(x_{t+1} \mid x_{\le t})
\]

Equivalent to maximizing likelihood.

---

## 11. Training Process

1. Sample a sequence of length \( T \)
2. Predict next token at each position
3. Compute loss
4. Backpropagation
5. Gradient descent

---

## 12. Text Generation

Given a prompt \( x_{1:k} \):

1. Compute logits
2. Apply softmax
3. Sample next token
4. Append
5. Repeat

This is **ancestral sampling**.

---

## 13. Why This Works (Even With Tiny Data)

- Model learns **statistics**, not meaning
- With small data → memorization
- With large data → generalization

Architecture scales; dataset scales.

---

## 14. What This Is Not

- Not AGI
- Not reasoning
- Not understanding
- Not scalable

It is **a correct implementation of a Transformer language model**.

---

## 15. Why This Project Matters

This repository demonstrates understanding of:

- Tokenization
- Self-attention mathematics
- Causal modeling
- Training dynamics
- Transformer internals

No abstraction layers.
No pretrained shortcuts.

---

## 16. Further Extensions

- Multi-head attention
- Byte-pair encoding
- Larger datasets
- GPUs
- Scaling laws

---

## Final Note

Modern LLMs differ in **scale**, not **kind**.

This code is a small but faithful specimen of the same organism.

