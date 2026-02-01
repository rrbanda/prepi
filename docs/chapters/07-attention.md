# Chapter 7: Attention — The Core Mechanism

*Part IV: The Secret Sauce*

---

Read this sentence:

> "I went to the bank to deposit money."

Now this one:

> "I sat on the bank of the river."

Same word — "bank." Different meanings.

How did you know which was which?

You looked at the surrounding words. "Deposit money" signals finance. "River" signals waterways. You **paid attention** to the context.

This is exactly what attention does in a transformer — mathematically.

---

## The Core Idea

Every token needs to decide: **"What in my context should I focus on?"**

Not all surrounding words matter equally. In "The cat sat on the mat," the word "sat" cares most about:

- "cat" (who's sitting?)
- "mat" (sitting where?)

It cares less about "The" and "on."

Attention lets each token **selectively gather information** from the tokens that matter.

---

## Why the Model Must Decide (First Principles)

This is a fundamental question — why does the model need to decide which previous tokens matter?

The answer from first principles:

**Because the meaning of the current token depends on which parts of the prior context are relevant, and that relevance changes dynamically.**

If the model treated all previous tokens equally, it would fail at language.

### Language Is Contextual and Non-Local

Human language has properties that demand selective attention:

- Words refer to things far away in the sentence
- Meanings shift based on earlier clauses
- Pronouns depend on earlier nouns
- Negation flips meaning
- Numbers and units must align

Example:

> "The server failed because it ran out of memory."

To understand "it", the model must:

- Ignore irrelevant words ("because")
- Focus on "server"
- Not confuse it with "memory"

So the model must **selectively focus**.

### What Happens Without This Decision-Making

Imagine a model that treats all previous tokens equally — just averages everything.

Then:

- Long sentences become noise
- Important words get diluted
- Meaning collapses as context grows

This is exactly why older RNN-based models struggled with long context.

### Why the Decision Must Be Dynamic

Relevance changes token by token.

Example:

> "I put the glass on the table because it was unstable."

Now "it" refers to **table**, not **glass**.

The model cannot use a fixed rule like "pronouns always look at the last noun." It must compute relevance **dynamically** based on meaning.

That's what self-attention does.

### The Capability Self-Attention Provides

Self-attention allows the model to:

- **Select** which prior tokens matter
- **Weight** them differently
- **Ignore** irrelevant context
- **Combine** relevant information into a new representation

This happens numerically, not symbolically.

### Connection to the Forward Pass

During the forward pass on the GPU:

1. Current token creates a **Query**
2. Past tokens provide **Keys**
3. Similarity scores determine relevance
4. Values from relevant tokens are combined

The model literally computes: **"Who should I listen to right now?"**

### Why This Beats Older Approaches

Before transformers:

- RNNs / LSTMs processed tokens sequentially
- Information had to be carried forward step by step
- Long-range dependencies faded (vanishing gradients)

With self-attention:

- Any token can directly attend to any other token
- Long-range dependencies are easy
- Parallelizable on GPUs

That's why transformers won.

---

## The Library Analogy

Let's build intuition with a powerful analogy.

Imagine you walk into a **library** with a question.

1. **Your Question (Query)**: "What do I want to know?"
   
   You're looking for information about, say, "French history."

2. **Book Labels (Keys)**: "What does each book contain?"
   
   Each book has a label: "French Cuisine," "French History," "German Engineering," "French Art."

3. **Comparing**: You compare your question to each label.
   
   - "French History" — exact match! High relevance.
   - "French Cuisine" — partial match. Medium relevance.
   - "German Engineering" — no match. Low relevance.

4. **Reading (Values)**: You read the content from the matching books.
   
   You don't read every book equally. You focus on "French History," skim "French Cuisine," and ignore "German Engineering."

5. **Your Understanding**: A blend of the relevant books' content, weighted by relevance.

This is exactly how attention works:

- **Query (Q)**: What am I looking for?
- **Key (K)**: What do you offer?
- **Value (V)**: Here's my actual content.

---

## Q, K, V in Transformers

Every token generates three vectors:

```
For each token:
  Q = Query  = "What am I looking for?"
  K = Key    = "What do I offer?"
  V = Value  = "Here's my information"
```

These are computed from the token's embedding:

```
Q = Embedding × W_q  (weight matrix for queries)
K = Embedding × W_k  (weight matrix for keys)
V = Embedding × W_v  (weight matrix for values)
```

The matrices W_q, W_k, and W_v are learned during training. They transform embeddings into these three roles.

---

## A Worked Example with Numbers

Before diving into the general steps, let's trace through attention with actual numbers. We'll use tiny 4-dimensional vectors to keep the math tractable — real models use 64 or 128 dimensions per head, but the process is identical.

**Setup:** Three tokens: "The", "cat", "sat"

Let's say after the Q, K, V transformations, we have:

```
Token "The":  Q₁ = [1, 0, 1, 0]    K₁ = [1, 1, 0, 0]    V₁ = [1, 0, 0, 0]
Token "cat":  Q₂ = [0, 1, 1, 0]    K₂ = [0, 1, 1, 0]    V₂ = [0, 1, 0, 0]
Token "sat":  Q₃ = [1, 1, 0, 1]    K₃ = [1, 0, 0, 1]    V₃ = [0, 0, 1, 0]
```

**Goal:** Compute the new representation for "sat" (token 3).

### Step 1: Compute Similarity Scores

We compare Q₃ (what "sat" is looking for) against all Keys using dot products:

```
Q₃ · K₁ = [1,1,0,1] · [1,1,0,0] = (1×1) + (1×1) + (0×0) + (1×0) = 2
Q₃ · K₂ = [1,1,0,1] · [0,1,1,0] = (1×0) + (1×1) + (0×1) + (1×0) = 1
Q₃ · K₃ = [1,1,0,1] · [1,0,0,1] = (1×1) + (1×0) + (0×0) + (1×1) = 2

Raw scores: [2, 1, 2]
```

"Sat" finds "The" (score=2) and itself (score=2) most relevant, with "cat" (score=1) less so.

### Step 2: Scale and Softmax

We scale by √d (here √4 = 2) to prevent extreme values:

```
Scaled scores: [2/2, 1/2, 2/2] = [1.0, 0.5, 1.0]
```

Then apply softmax to convert to probabilities (values that sum to 1):

```
softmax([1.0, 0.5, 1.0]) ≈ [0.39, 0.22, 0.39]
```

These are the **attention weights** — how much "sat" should attend to each token.

### Step 3: Weighted Sum of Values

Now we blend the Value vectors using these weights:

```
Output for "sat" = 0.39 × V₁ + 0.22 × V₂ + 0.39 × V₃
                 = 0.39 × [1,0,0,0] + 0.22 × [0,1,0,0] + 0.39 × [0,0,1,0]
                 = [0.39, 0, 0, 0] + [0, 0.22, 0, 0] + [0, 0, 0.39, 0]
                 = [0.39, 0.22, 0.39, 0]
```

**The result:** "Sat" now has a new vector that blends information from all three tokens, weighted by relevance. It carries 39% information from "The", 22% from "cat", and 39% from itself.

This is one attention head. In practice, GPT-2 runs 12 of these in parallel (with different learned W_q, W_k, W_v matrices), then concatenates the results.

---

## The Attention Calculation

Now that you've seen the numbers, here are the general steps:

### Step 1: Compare Queries to Keys

Each token's Query is compared to every token's Key.

```
Token 1's Query vs Token 1's Key → score
Token 1's Query vs Token 2's Key → score
Token 1's Query vs Token 3's Key → score
...
```

High score = high relevance. The comparison is a dot product (multiply corresponding numbers and sum).

### Step 2: Convert to Probabilities

The scores are converted to percentages using **softmax**:

```
Raw scores: [2.0, 0.5, 0.1, -0.3]
After softmax: [0.71, 0.16, 0.11, 0.02]
```

Now they sum to 1. These are attention weights — how much to focus on each token.

### Step 3: Blend the Values

Finally, we take a weighted average of all the Value vectors:

```
Output = 0.71 × V₁ + 0.16 × V₂ + 0.11 × V₃ + 0.02 × V₄
```

The result is a **contextual representation** — this token's meaning, informed by its context.

---

## The Full Formula

For completeness, here's the attention equation:

```
Attention(Q, K, V) = softmax(Q × Kᵀ / √d) × V
```

Breaking it down:

- `Q × Kᵀ`: Compare each query to each key (dot products)
- `/ √d`: Scale by the square root of key dimension
- `softmax`: Convert to probabilities
- `× V`: Weighted blend of values

### Why Scale by √d?

This scaling factor is crucial for training stability.

When embedding dimensions are large (like 768 or 12,288), dot products between Q and K vectors can produce very large numbers. Large numbers pushed through softmax become extreme — one attention weight near 1, all others near 0.

The problem? **Extreme softmax outputs have tiny gradients.** The model can't learn because error signals barely flow backward.

Dividing by √d keeps the values in a moderate range where softmax produces reasonable probabilities and gradients flow properly. For GPT-2 with 64-dimensional keys per head, we divide by √64 = 8.

This is why it's called **scaled dot-product attention**.

---

**You might be wondering:** *"Why divide by √d specifically? Why not just divide by d, or use a different scaling factor?"*

The √d scaling comes from the variance of dot products. When Q and K have dimension d and their components are roughly independent with variance 1, the dot product has variance d (each of the d multiplications contributes variance 1, and they sum). Dividing by √d normalizes the variance to 1, keeping softmax inputs in a reasonable range. Dividing by d would over-scale and make attention weights too uniform. The square root is the mathematical result of variance normalization.

---

## A Concrete Example

Let's trace through our running example: **"The Eiffel Tower is located in"**

When processing "in" (the final token that will predict "Paris"):

1. **Generate Q, K, V** for each token:
   ```
   "The"      → Q₁, K₁, V₁
   " Eiff"    → Q₂, K₂, V₂
   "el"       → Q₃, K₃, V₃
   " Tower"   → Q₄, K₄, V₄
   " is"      → Q₅, K₅, V₅
   " located" → Q₆, K₆, V₆
   " in"      → Q₇, K₇, V₇
   ```

2. **Query from "in"** compares to all Keys:
   ```
   Q₇ vs K₁ ("The")      → low score
   Q₇ vs K₂ (" Eiff")    → HIGH score ← what landmark?
   Q₇ vs K₃ ("el")       → medium score
   Q₇ vs K₄ (" Tower")   → HIGH score ← what's being located?
   Q₇ vs K₅ (" is")      → low score
   Q₇ vs K₆ (" located") → medium score
   Q₇ vs K₇ (" in")      → medium score (self)
   ```

3. **Softmax** the scores:
   ```
   Attention weights: [0.03, 0.28, 0.08, 0.35, 0.04, 0.12, 0.10]
   ```

4. **Blend the Values**:
   ```
   New "in" = 0.03×V₁ + 0.28×V₂ + 0.08×V₃ + 0.35×V₄ + 0.04×V₅ + 0.12×V₆ + 0.10×V₇
   ```

The output for "in" now carries significant information from "Tower" (35%) and "Eiff" (28%). This contextual vector knows that we're asking *where the Eiffel Tower is located* — which is exactly what's needed to predict "Paris."

**The attention mechanism has extracted the relevant context for prediction.**

---

## Another Example: Disambiguation

Let's also look at: **"The bank by the river"**

When processing "bank":

```
Q₂ ("bank") vs K₅ ("river") → HIGH score ← relevant context!
```

Attention weights: `[0.05, 0.15, 0.05, 0.05, 0.70]`

The output for "bank" carries 70% information from "river."

If the sentence were "The bank holds money," "bank" would attend strongly to "money" instead, producing a completely different contextual vector.

**Same word, same starting embedding, different context, different output.**

---

---

**The Revelation:**

> The same word becomes different vectors depending on context.

"In" starts as the same embedding every time. But after attention, it becomes a different vector for "located in Paris" vs. "interested in cooking" vs. "in the morning." The contextual representation captures not just *what* the word is, but *what role it plays here*.

---

## Multi-Head Attention

In practice, transformers use **multi-head attention**.

Instead of one Q, K, V calculation, they run several in parallel. GPT-2 Small uses 12 heads; GPT-3 uses 96.

### How It Works

The embedding dimension is split among the heads:

```
GPT-2 Small: 768 dimensions ÷ 12 heads = 64 dims per head
GPT-3:       12,288 dimensions ÷ 96 heads = 128 dims per head
```

Each head runs a separate attention calculation on its slice of dimensions. Then all the outputs are concatenated back together and projected through a final weight matrix.

### Why Multiple Heads?

Different heads learn to focus on different things:

- Head 1: Syntactic relationships (subject-verb agreement)
- Head 2: Semantic relationships (bank-river)
- Head 3: Positional patterns (nearby words)
- Head 4: Long-range dependencies (pronouns to their referents)
- ...

It's like having multiple pairs of eyes reading the text, each watching for different patterns. The final output combines all perspectives.

---

**You might be wondering:** *"How do different heads 'learn' to focus on different patterns? Is this programmed, or does it emerge naturally?"*

This emerges during training — it's not programmed. Each head has its own W_q, W_k, W_v weight matrices, so they can learn different attention patterns independently. The model discovers that specialization helps overall prediction accuracy. There's no explicit instruction saying "head 3 should focus on syntax"; the loss function rewards patterns that improve predictions, and specialization is one way to achieve that. Researchers have found heads that specialize in syntax, semantics, coreference, and more — but these emerge, they're not designed.

---

### Dropout in Attention

During training, models apply **dropout** to attention weights — randomly zeroing out some weights and scaling up the rest. This prevents the model from relying too heavily on specific attention patterns and improves generalization. Typical dropout rates are 10-20%.

---

## Causal Masking (For Language Models)

In language models, there's a constraint: **you can't see the future.**

When predicting the next token after "The capital of France is," the model shouldn't peek at "Paris" if it's already in the sequence.

This is enforced with **causal masking**: each token can only attend to tokens before it (and itself).

```
Token 1 can see: [1]
Token 2 can see: [1, 2]
Token 3 can see: [1, 2, 3]
Token 4 can see: [1, 2, 3, 4]
...
```

Future tokens are masked out (their attention weights become zero).

---

**You might be wondering:** *"Can encoder models see future tokens? Is that why BERT is different from GPT?"*

Yes, exactly. Encoder models like BERT use bidirectional attention — every token can attend to every other token, including "future" ones (to the right). This is powerful for understanding tasks like classification or fill-in-the-blank. But it prevents autoregressive generation because you can't generate left-to-right if each position needs to see what comes after it. GPT-style decoder models use causal masking specifically because they generate text left-to-right, one token at a time.

---

## Why Architecture Matters for Attention

Not all transformer architectures use attention the same way. Understanding these differences explains why optimizations like PagedAttention are designed specifically for decoder-only models.

### Decoder-Only Models (GPT, LLaMA, Claude)

Decoder-only models generate tokens **one at a time, autoregressively**. Each new token needs access to the full context history — every previous token's Keys and Values.

```
Token 1: Generate → store K₁, V₁
Token 2: Attend to [K₁] → generate → store K₂, V₂  
Token 3: Attend to [K₁, K₂] → generate → store K₃, V₃
...
Token 1000: Attend to [K₁...K₉₉₉] → generate → store K₁₀₀₀, V₁₀₀₀
```

This creates:
- A **large and growing memory footprint** per request
- **Linear KV cache growth** with each new token
- **Variable-length sequences** across concurrent requests

This is exactly why PagedAttention (covered in Chapter 12) was designed — it handles the dynamic, growing KV cache efficiently through non-contiguous memory allocation.

### Encoder-Only Models (BERT)

Encoder models process the **entire input sequence in one forward pass**. There's no autoregressive generation — no token-by-token output.

```
Input: "The cat sat on the mat"
       ↓
All tokens processed simultaneously
       ↓
Output: Contextual embeddings for each token
```

Key characteristics:
- **No KV cache needed** — no incremental generation
- **Bidirectional attention** — each token sees all other tokens
- Memory use is **batch-wide and fixed**, not incremental

Since there's no growing KV cache to manage, PagedAttention provides no benefit for encoder-only models.

### Encoder-Decoder Models (T5, BART)

These models have two parts:
1. **Encoder**: Processes the input (like encoder-only)
2. **Decoder**: Generates output autoregressively (like decoder-only)

The decoder portion does require a KV cache, so PagedAttention can help. However, these architectures are less common for the use cases vLLM is optimized for — long chat-style inferencing and high-concurrency generation.

### Why This Matters for Production

| Architecture | KV Cache? | PagedAttention Benefit | Common Use |
|--------------|-----------|----------------------|------------|
| **Decoder-only** | Yes, grows linearly | **Maximum benefit** | Chat, generation, code |
| **Encoder-only** | No | None | Classification, embeddings |
| **Encoder-decoder** | Decoder only | Partial benefit | Translation, summarization |

When you're deploying LLMs in production (Chapter 12), understanding this helps you:
- Choose the right serving optimizations
- Predict memory requirements
- Understand why certain models scale differently

---

## Pause and Reflect

Try this exercise:

**"She gave her dog her old blanket"**

When processing the second "her" (before "old"):
- What tokens should it attend to strongly?
- What's it trying to figure out? (Whose blanket?)

When processing "blanket":
- What context matters?
- What's being predicted next?

The model learns these patterns from data — we don't program them.

---

## Why Attention Is Powerful

Attention provides:

1. **Selectivity**: Focus on what matters, ignore noise
2. **Flexibility**: Different patterns for different contexts
3. **Parallelism**: All computations happen simultaneously
4. **Interpretability**: Attention weights show what the model focused on

This is why "Attention Is All You Need" was a fitting title. Attention is the mechanism that enables everything else.

---

## In Practice: Attention Optimizations in OpenShift AI

The attention computation you've learned — Q × K^T × V — is the most expensive part of LLM inference. Serving LLMs at scale requires optimizing this bottleneck.

### The Problem

Standard attention has two challenges:

1. **O(n²) memory**: The Q × K^T matrix grows quadratically with sequence length
2. **Memory fragmentation**: Storing K and V for the KV cache wastes GPU memory

### PagedAttention (vLLM)

**PagedAttention** — the core innovation in vLLM — reimagines how K and V are stored. Instead of pre-allocating contiguous memory:

- K and V are stored in **pages** (like virtual memory)
- Pages can be anywhere in GPU memory
- No fragmentation, higher utilization

When you deploy models in **OpenShift AI** using the vLLM ServingRuntime, PagedAttention runs automatically. The `--gpu-memory-utilization` parameter controls how much GPU memory vLLM uses for the KV cache:

```yaml
args:
- --gpu-memory-utilization=0.95  # Use 95% of GPU memory
```

### FlashAttention

**FlashAttention** optimizes the attention computation itself:

- Fuses Q × K^T, softmax, and × V into one GPU kernel
- Reduces memory reads/writes (memory bandwidth is often the bottleneck)
- Enables longer context lengths without running out of memory

vLLM integrates FlashAttention when available. For NVIDIA GPUs (Ampere and newer), this happens automatically.

### Why This Matters

The attention math you learned in this chapter runs billions of times per second in production:

| Optimization | What It Solves | Available In |
|--------------|----------------|--------------|
| **PagedAttention** | KV cache fragmentation | vLLM (default in RHOAI) |
| **FlashAttention** | Attention memory bandwidth | vLLM on Ampere+ GPUs |
| **Multi-Query Attention** | Reduced KV cache size | Model architecture choice |

Understanding attention helps you debug issues like "model runs out of memory at long contexts" or "inference slows down dramatically" — these trace back to the O(n²) attention computation.

---

## Chapter Takeaway

> **Attention lets each token ask "What's relevant to me?"** and selectively blend information from matching tokens. Queries ask, Keys advertise, Values contain the content. Scaling by √d keeps gradients healthy during training. Multi-head attention (12 heads in GPT-2, 96 in GPT-3) provides multiple perspectives, with each head learning different patterns. The result is a contextual representation — the token's meaning informed by its surroundings. **In OpenShift AI, vLLM uses PagedAttention and FlashAttention to make this computation practical at scale.**

---

*Next: [Chapter 8: Context — The Most Misunderstood Word in AI](08-context.md)*
