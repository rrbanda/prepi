# Chapter 9: Inference — A Token's Journey

*Part V: Watching It Think*

---

Remember the promise from the Prologue?

> "By the end, you'll be able to explain — clearly, confidently, and correctly — how a large language model transforms your words into its response."

This is that chapter. You're about to watch it happen, step by step.

---

## The Setup

You type our running example:

> "The Eiffel Tower is located in"

And you wait for the model to complete it.

What happens inside the machine?

---

## Step 1: Tokenization

First, your text is broken into tokens — exactly as we learned in Chapter 3.

```
Input text: "The Eiffel Tower is located in"

Tokenization:
  "The"      → Token ID 464
  " Eiff"    → Token ID 36751
  "el"       → Token ID 417
  " Tower"   → Token ID 8765
  " is"      → Token ID 318
  " located" → Token ID 5140
  " in"      → Token ID 287

Result: [464, 36751, 417, 8765, 318, 5140, 287]
```

This happens on the CPU. The tokenizer is deterministic — same input always gives same output.

---

## Step 2: Embedding Lookup

Each token ID is converted to a vector by looking it up in the embedding matrix — exactly as we learned in Chapter 4.

```
Token ID 464   ("The")     → [0.12, -0.34, 0.56, 0.23, ...]  (768 dimensions)
Token ID 36751 (" Eiff")   → [-0.08, 0.91, -0.12, 0.45, ...]
Token ID 417   ("el")      → [0.33, -0.17, 0.29, -0.88, ...]
Token ID 8765  (" Tower")  → [0.67, 0.23, -0.45, 0.12, ...]
Token ID 318   (" is")     → [-0.22, 0.18, 0.89, -0.34, ...]
Token ID 5140  (" located")→ [0.45, -0.67, 0.11, 0.78, ...]
Token ID 287   (" in")     → [0.19, 0.42, -0.33, 0.56, ...]

Result: Matrix of 7 tokens × 768 dimensions
```

Now we have meaningful vectors, not just labels.

Positional encodings are also added here, so the model knows token order.

---

## Step 3: Transformer Layers

The embeddings pass through the transformer layers — exactly as we learned in Chapters 5-7. Each layer:

1. **Self-Attention**: Tokens gather information from relevant other tokens
2. **Feed-Forward Network**: Each token is processed independently
3. **Layer Normalization**: Values are kept in reasonable ranges

This repeats for every layer (e.g., 96 times for GPT-4 scale).

```
Embeddings (7 tokens × 768 dims)
      ↓
  Layer 1 → "in" starts gathering context from "Tower", "Eiffel"
      ↓
  Layer 2 → Context deepens
      ↓
    ...
      ↓
  Layer 96 → Final contextual vectors
      ↓
Output (7 tokens × 768 dims)
```

After all layers, each token's vector encodes its meaning, position, and relationship to all other tokens. The vector for "in" now carries rich information about *what* is being located *where*.

---

## Step 4: Output Projection

We want to predict the **next** token — the one after "in".

We take the **final token's vector** (the vector for "in") and project it to vocabulary space:

```
Final vector for "in": [0.89, -0.23, 0.45, ...]
                            ↓
              Output projection (matrix multiply)
                            ↓
Scores for EVERY token in vocabulary:

Token 0  ("!"): -3.2
Token 1  ("\""): -5.1
...
Token 6342 ("Paris"): 8.7  ← highest!
Token 6343 ("London"): 4.2
...
Token 50255 ("[END]"): -2.8
```

These scores are called **logits**. They're not probabilities yet — just raw scores.

---

**You might be wondering:** *"Why use only the final token's vector to predict the next token? Why not use all tokens or an average?"*

The final token's vector already includes context from all prior tokens via attention. Through the many transformer layers, information from "Eiffel," "Tower," and "located" has been gathered into the final position. Using it alone is sufficient and efficient. Averaging all tokens would add complexity without benefit, since attention already aggregated the needed information into that last position.

---

### Weight Tying (Efficiency Trick)

Interestingly, the original GPT-2 **shares weights** between the embedding matrix and the output projection matrix. They're the same matrix used in different directions:

- **Forward**: Token ID → Embedding (lookup row)
- **Backward**: Hidden state → Vocabulary scores (matrix multiply)

This saves memory — that's 50,257 × 768 = 38.6 million parameters that don't need to be stored twice.

Newer, larger models often don't tie weights, since the added capacity can help. But weight tying is worth understanding as it demonstrates the elegance of the architecture.

---

**You might be wondering:** *"How can the same matrix work in both directions (lookup vs matrix multiply)?"*

The embedding matrix is transposed for the output projection. Row lookup (token ID → embedding) uses the matrix as-is: given row index 318, retrieve the 768-dim vector at that row. Column projection (hidden state → vocabulary scores) uses the transpose: multiply a 768-dim hidden state by the 768×50257 transposed matrix to get 50257 scores. This works because the vocabulary size matches one dimension and embedding size matches the other. It's elegant parameter-sharing that reduces memory without changing expressiveness.

---

## Step 5: Softmax and Probabilities

The logits are converted to probabilities using **softmax**:

```
Logits:  [... -3.2, ..., 8.7, 4.2, ..., -2.8, ...]
             ↓ softmax
Probs:   [... 0.0001, ..., 0.73, 0.12, ..., 0.0002, ...]
```

Now we have a probability distribution over the entire vocabulary.

- "Paris" → 73% probability
- "London" → 12% probability
- "France" → 5% probability
- Everything else → tiny probabilities

---

## Step 6: Sampling

We need to pick one token. There are several strategies:

### Greedy Decoding
Always pick the highest probability:
```
Selected: "Paris" (73%)
```

Simple but can be repetitive and boring.

### Temperature Sampling
Flatten or sharpen the distribution, then sample randomly:

- Temperature = 0: Same as greedy (always pick highest)
- Temperature = 1: Sample according to original probabilities
- Temperature > 1: More random (flatter distribution)
- Temperature < 1: More deterministic (sharper distribution)

### Top-k Sampling
Only consider the **k most likely** tokens:

```
k = 50: Consider only the top 50 tokens
Sample from this reduced set
```

This prevents the model from ever selecting extremely unlikely tokens, no matter what.

### Top-p (Nucleus) Sampling
Only consider tokens that make up the top **p%** of total probability:

```
p = 0.9: Find the smallest set of tokens whose probabilities sum to 90%
Top 90%: ["Paris", "London", "France", "the", ...]
Sample from this reduced set
```

Top-p is more adaptive than top-k: if one token has 95% probability, top-p might only consider that one token, while top-k would still include 49 others.

### Beam Search
Instead of picking one token at a time, **beam search** tracks multiple candidate sequences (beams) in parallel:

```
Beam 1: "Paris" (73%)
Beam 2: "London" (12%)
Beam 3: "France" (5%)
```

Each beam is extended with its most likely next tokens, then the top beams are kept. This finds globally better sequences but is more expensive. It's often used for translation rather than open-ended generation.

Most production systems combine **temperature + top-p** (or top-k) for a balance of creativity and coherence.

---

**You might be wondering:** *"Which sampling strategy should I use in production?"*

It depends on your use case. For factual Q&A or code generation: low temperature (0.1-0.3) or greedy to maximize accuracy. For creative writing: higher temperature (0.7-1.0) with top-p (0.9) to allow diversity. For chat applications: moderate temperature (0.5-0.7) with top-p (0.9) balances coherence with variety. Start conservative and adjust based on output quality. There's no universal answer — experimentation is key.

---

### Sampling and Speculative Decoding

There's an important interaction between sampling strategies and **speculative decoding** — an optimization technique covered in Chapter 12.

Speculative decoding uses a small, fast "draft" model to predict several tokens ahead, then a larger model verifies or corrects those predictions. This speeds up generation significantly.

However, **sampling strategy affects speculative decoding effectiveness**:

| Sampling Strategy | Speculative Decoding Compatibility |
|-------------------|-----------------------------------|
| **Greedy (temperature=0)** | Excellent — highly predictable, high acceptance rate |
| **Low temperature (0.1-0.5)** | Good — draft model predictions are often accepted |
| **High temperature (>1.0)** | Poor — randomness leads to low acceptance rate, wasted computation |
| **Beam search** | Incompatible — fundamentally different decoding approach |

The key insight: **Speculative decoding relies on predictability**. If the target model's sampling is highly random, the draft model's guesses are often rejected, negating the speed benefit.

In production systems like vLLM:
- Requests with different sampling strategies may need separate batching
- Mixing speculative decoding with high-temperature requests adds scheduling complexity
- Teams must choose consistent sampling parameters for optimal throughput

This is why understanding sampling isn't just about output quality — it affects infrastructure efficiency too.

---

## Step 7: Repeat (Autoregressive Generation)

We've generated one token: "Paris".

Now we append it to the input and repeat:

```
Original: [464, 36751, 417, 8765, 318, 5140, 287]
                                              ↓ new token
Updated:  [464, 36751, 417, 8765, 318, 5140, 287, 6342]
```

The entire process runs again:
- Embeddings for all 8 tokens
- Transformer layers
- Output projection (now from the "Paris" position)
- Sampling → next token (maybe "," or ".")

And repeat. And repeat.

Until:
- A special [END] token is generated
- Maximum length is reached
- A stop sequence is encountered

---

## The Full Flow Diagram

```
┌───────────────────────────────────────────────┐
│ "The Eiffel Tower is located in"              │
└───────────────┬───────────────────────────────┘
                ↓
┌───────────────────────────────────────────────┐
│ TOKENIZER (CPU)                               │
│ → [464, 36751, 417, 8765, 318, 5140, 287]     │
└───────────────┬───────────────────────────────┘
                ↓
┌───────────────────────────────────────────────┐
│ EMBEDDING LOOKUP (GPU)                        │
│ → 7 vectors of 768 dimensions                 │
│ + positional encodings                        │
└───────────────┬───────────────────────────────┘
                ↓
┌───────────────────────────────────────────────┐
│ TRANSFORMER LAYERS (GPU)                      │
│ Layer 1: Attention + FFN                      │
│ Layer 2: Attention + FFN                      │
│ ...                                           │
│ Layer 96: Attention + FFN                     │
└───────────────┬───────────────────────────────┘
                ↓
┌───────────────────────────────────────────────┐
│ OUTPUT PROJECTION (GPU)                       │
│ Final token's vector → vocabulary scores      │
└───────────────┬───────────────────────────────┘
                ↓
┌───────────────────────────────────────────────┐
│ SOFTMAX + SAMPLING                            │
│ → "Paris" (Token ID 6342)                     │
└───────────────┬───────────────────────────────┘
                ↓
┌───────────────────────────────────────────────┐
│ APPEND + REPEAT                               │
│ → [464, 36751, 417, 8765, 318, 5140, 287,     │
│    6342, ...]                                 │
└───────────────────────────────────────────────┘
```

---

## The Key Insight: Weights Are Read-Only

During inference:

- The model's weights are **fixed**
- No learning happens
- The weights are **read** millions of times
- The weights are **never modified**

This is like a musician performing a song. The performance changes each time, but the musician's skill doesn't improve mid-song.

The model can't learn from your prompts during inference. That would require training.

---

## CPU vs GPU: Who Does What

The inference pipeline spans two processors with distinct responsibilities:

| Step | Runs On | Why |
|------|---------|-----|
| Tokenization | CPU | String processing and dictionary lookup |
| Embedding lookup | GPU | Accessing model weights in GPU memory |
| Transformer layers | GPU | Massive matrix multiplications |
| Output projection | GPU | Matrix multiplication to produce logits |
| Sampling | CPU (often) | Probability calculations and random selection |
| Detokenization | CPU | Token IDs back to text |

The model server orchestrates this entire flow. When you call an inference endpoint:

1. The **CPU** receives your text and runs the tokenizer
2. Token IDs are copied to **GPU** memory
3. The **GPU** runs the transformer forward pass (the expensive part)
4. Results come back to the **CPU** for sampling and text conversion
5. The **CPU** returns the response

This is why both CPU and GPU specifications matter for model serving, though GPU is typically the bottleneck for LLMs.

---

## Pause and Reflect

Notice: The model generated "Paris" not because it knows geography as facts, but because:

1. Training data had many sentences like "Eiffel Tower is in Paris"
2. Those examples shaped the weights
3. The weights now produce high probability for "Paris" given this context

Knowledge is probability patterns, not stored facts.

---

## In Practice: Model Inference with OpenShift AI

The inference pipeline you've learned — tokenize, embed, transform, project, sample — runs inside a **model server**. In **Red Hat OpenShift AI**, deployed models expose this as REST or gRPC APIs.

### Inference Endpoints

When you deploy a model using the vLLM ServingRuntime (covered in Chapter 12), OpenShift AI exposes OpenAI-compatible endpoints:

| Endpoint | Purpose |
|----------|---------|
| `/v1/completions` | Text completion (given a prompt, generate text) |
| `/v1/chat/completions` | Chat format (messages array with roles) |
| `/v1/models` | List available models |
| `/v1/embeddings` | Generate embeddings (embedding models only) |

### Calling an Inference Endpoint

Here's the inference process from Chapter 9, now as an API call:

```bash
# Request inference from a deployed LLM
curl -X POST https://llama-7b-myproject.apps.cluster.example.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-7b",
    "messages": [
      {"role": "user", "content": "The Eiffel Tower is in"}
    ],
    "max_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

The response streams back token by token:

```json
{
  "id": "cmpl-abc123",
  "object": "chat.completion",
  "model": "llama-7b",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Paris, France. The iconic iron lattice tower was constructed from 1887 to 1889..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 43,
    "total_tokens": 50
  }
}
```

### Sampling Parameters

The sampling strategies from this chapter map directly to API parameters:

| Parameter | What It Controls |
|-----------|------------------|
| `temperature` | Randomness (0 = greedy, >1 = more random) |
| `top_p` | Nucleus sampling threshold |
| `top_k` | Consider only top K tokens |
| `max_tokens` | Maximum tokens to generate |
| `stop` | Stop sequences to end generation |

### Token Authorization

For production deployments, secure your inference endpoint with token authentication:

```bash
curl -X POST https://llama-7b.apps.cluster.example.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

OpenShift AI integrates with Authorino for authentication and authorization.

### From Theory to Production

The complete inference flow in production:

```
User Request (JSON)
       ↓
   API Gateway (Route)
       ↓
   InferenceService
       ↓
   Model Server (vLLM)
       ↓
┌──────────────────────┐
│  Tokenize prompt     │ ← Same steps you learned
│  Embed tokens        │
│  Transform (N layers)│
│  Project to logits   │
│  Sample next token   │
│  Repeat until done   │
└──────────────────────┘
       ↓
   JSON Response (tokens → text)
```

Every concept in this chapter — tokenization, embeddings, attention, sampling — executes inside that model server for each request.

---

## Chapter Takeaway

> **Inference is the process of converting a prompt into a response, one token at a time.** Text is tokenized, embedded, passed through transformer layers, projected to vocabulary scores (sometimes using weight tying), sampled using strategies like temperature, top-k, or top-p, and repeated. Each token generation uses the entire sequence so far. Weights are read-only — no learning occurs during inference. **In OpenShift AI, this process runs inside a model server and is exposed via OpenAI-compatible REST APIs.**

---

*Next: [Chapter 10: Training — How Intelligence Emerges](10-training.md)*
