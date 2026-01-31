# Chapter 11: The KV Cache — Why Inference Doesn't Crawl

*Part VI: Under the Hood*

---

Here's a puzzle.

When generating a response, the model processes tokens one at a time. For each new token, it seems like we need to run attention over the **entire sequence** — everything so far.

For a 1000-token conversation, generating the 1001st token requires attention over all 1000 previous tokens. The 1002nd token requires attention over 1001 tokens. And so on.

That sounds impossibly slow. So how does it actually work at reasonable speed?

The answer: **the KV cache**.

---

## The Problem: Redundant Computation

Let's trace what happens in attention without any optimization.

**Generating token 1001:**
```
Compute Q, K, V for tokens 1-1000 (already computed before!)
Compute Q, K, V for token 1001
Run attention: 1001 queries × 1001 keys
Get output
```

**Generating token 1002:**
```
Compute Q, K, V for tokens 1-1001 (computed AGAIN!)
Compute Q, K, V for token 1002
Run attention: 1002 queries × 1002 keys
Get output
```

See the problem? We recompute K and V for all previous tokens every single time.

Token 1's K and V are exactly the same whether we're generating token 100 or token 1000. We're doing the same work over and over.

---

## The Insight: K and V Don't Change

Here's the key observation:

For a token that's already been processed, its **Key (K)** and **Value (V)** vectors never change.

Why? Because they're computed from:
```
K = Embedding × W_k
V = Embedding × W_v
```

The embedding is fixed. The weights are fixed. So K and V are fixed.

Only the **Query (Q)** for the new token matters for the new computation.

---

## The Solution: Cache K and V

Instead of recomputing, we **store** the K and V vectors:

```
Step 1 (first token):
  Compute K₁, V₁
  Store in cache: [(K₁, V₁)]
  
Step 2:
  Retrieve: [(K₁, V₁)]
  Compute K₂, V₂
  Append to cache: [(K₁, V₁), (K₂, V₂)]
  
Step 3:
  Retrieve: [(K₁, V₁), (K₂, V₂)]
  Compute K₃, V₃
  Append to cache: [(K₁, V₁), (K₂, V₂), (K₃, V₃)]
  
...and so on
```

For each new token, we:
1. Retrieve cached K and V from all previous tokens
2. Compute K and V for the new token only
3. Run attention
4. Append the new K and V to the cache

---

## The Speedup

Without KV cache (naive approach):
```
Token 100:  Compute K, V for 100 tokens  → 100 units of work
Token 101:  Compute K, V for 101 tokens  → 101 units of work
Token 102:  Compute K, V for 102 tokens  → 102 units of work
```

With KV cache:
```
Token 100:  Compute K, V for 1 token + retrieve 99 from cache → 1 unit
Token 101:  Compute K, V for 1 token + retrieve 100 from cache → 1 unit  
Token 102:  Compute K, V for 1 token + retrieve 101 from cache → 1 unit
```

At token 100, the naive approach does 100× more work than necessary!

---

## Where the Cache Lives

The KV cache is stored in **GPU memory**.

For each layer, we store:
```
Layer 1: K₁, K₂, ..., Kₙ and V₁, V₂, ..., Vₙ
Layer 2: K₁, K₂, ..., Kₙ and V₁, V₂, ..., Vₙ
...
Layer 96: K₁, K₂, ..., Kₙ and V₁, V₂, ..., Vₙ
```

Note: Each layer has its own K and V cache. The vectors computed in layer 1 are different from those in layer 2 — they're progressively refined through the network.

---

## KV Cache Memory: The Math

The cache size formula per token:

```
Size = 2 (K and V) × num_layers × num_heads × head_dim × 2 bytes (fp16)
```

Or equivalently (since `num_heads × head_dim = embedding_dim`):

```
Size = 2 × num_layers × embedding_dim × 2 bytes
```

### Concrete Examples

| Model | Layers | Embedding Dim | Per Token | 1k Tokens | 100k Tokens |
|-------|--------|---------------|-----------|-----------|-------------|
| GPT-2 Small | 12 | 768 | 37 KB | 37 MB | 3.7 GB |
| GPT-2 XL | 48 | 1,600 | 307 KB | 307 MB | 30 GB |
| LLaMA 7B | 32 | 4,096 | 524 KB | 524 MB | 52 GB |
| GPT-3 175B | 96 | 12,288 | 4.7 MB | 4.7 GB | 470 GB |

GPT-3's 100k context would require nearly **half a terabyte** just for the KV cache — not counting the model weights!

This is why context length is limited — the KV cache eats GPU memory, and it scales linearly with sequence length and model depth.

---

## Key Properties of the KV Cache

| Property | Description |
|----------|-------------|
| **Per-request** | Each conversation has its own cache |
| **Temporary** | Discarded when the request ends |
| **Grows with length** | More tokens = bigger cache |
| **Lives in GPU RAM** | Fast access, limited space |
| **Per-layer** | Every transformer layer has its own cache |

---

## What the KV Cache Is NOT

Common misconceptions:

| Misconception | Reality |
|---------------|---------|
| "It's long-term memory" | No, it's per-request and temporary |
| "It stores knowledge" | No, it stores computed K/V vectors |
| "It persists between chats" | No, it's created fresh each time |
| "It's part of the model weights" | No, weights are separate |

The KV cache is an **optimization**, not a feature. It makes fast generation possible but doesn't add capabilities.

### Also Called "Attention State"

You may encounter the term **attention state** used interchangeably with KV cache. They refer to the same thing: the stored Key and Value vectors for processed tokens.

The name "attention state" emphasizes that this data represents the accumulated state needed for attention computation. The name "KV cache" emphasizes that it's cached (stored for reuse) Key and Value vectors.

### The Relationship to Context

This distinction is important:

- **KV cache** stores the K/V vectors that *enable* efficient context computation
- **Context** (contextual representation) is the *result* of computing attention over those K/V vectors

The KV cache is not context itself — it's the ingredients. Context is the dish you cook from those ingredients. Every time a new token is generated, the model uses the cached K/V vectors to compute a new contextual representation for that token.

---

## The Collaborative Story Analogy

Imagine you're writing a story collaboratively:

**Without KV cache**: Every time you add a sentence, you re-read the entire story from the beginning to understand the context.

**With KV cache**: You keep notes about what you've read. To add a new sentence, you just read the new sentence and check your notes.

The notes are the KV cache — a record of what you've processed, so you don't have to redo the work.

---

## Why Inference Slows With Length

Now you can answer this common question:

*"Why does inference get slower as conversations get longer?"*

Three reasons:

1. **Attention is O(n²)**: Each token attends to all previous tokens. More tokens = quadratically more attention computation.

2. **KV cache grows**: More tokens = more memory for the cache. Eventually you hit memory limits.

3. **Memory bandwidth**: Reading a larger KV cache takes more time, even if you have the memory.

This is why models have context limits — not just memory, but computational cost grows with length.

---

## Pause and Reflect

Consider: After generating a response, the KV cache is discarded.

When you send your next message:
- The entire conversation is re-tokenized
- New embeddings are computed
- A new KV cache is built from scratch

Nothing persists. Each request is independent.

This is why "the model remembers our conversation" is misleading — it re-reads everything each time.

---

## In Practice: KV Cache in OpenShift AI

The KV cache memory challenge is exactly what **vLLM's PagedAttention** solves (covered in Chapter 12). In **OpenShift AI**, the vLLM ServingRuntime for KServe provides this optimization automatically:

- **No memory fragmentation**: PagedAttention allocates KV cache in pages, not contiguous blocks
- **Higher throughput**: More concurrent requests fit in GPU memory
- **Automatic management**: The serving runtime handles cache allocation and eviction

When you deploy an LLM using the vLLM runtime in OpenShift AI, you get these KV cache optimizations without writing any code — the math you learned here runs efficiently under the hood.

---

## Chapter Takeaway

> **The KV cache stores Key and Value vectors for all processed tokens**, eliminating redundant computation. Each layer maintains its own cache. Memory grows linearly with sequence length: GPT-2 Small uses ~37 KB per token, while GPT-3 uses ~4.7 MB per token. A 100k context on GPT-3 would require ~470 GB — this is why context limits exist. The cache is per-request, temporary, and essential for practical autoregressive generation. **vLLM's PagedAttention, available in OpenShift AI, optimizes KV cache memory management.**

---

*Next: [Chapter 12: Serving Systems — vLLM and Friends](12-serving-systems.md)*
