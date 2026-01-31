# Chapter 4: Giving Numbers Meaning

*Part II: The Translation Problem*

---

We've turned text into token IDs. But we said token IDs are meaningless. So how does the model understand anything?

This is where the magic really begins.

---

## The Key Insight: Meaning Is About Relationships

What does the word "cat" mean?

You might say: "A small, furry animal that meows."

But notice: You defined "cat" using its relationships to other concepts — "small," "furry," "animal," "meows."

Here's the profound idea behind embeddings:

**We can represent meaning by position in space.**

Imagine a map where every word has coordinates. Words that are similar sit close together:

- "Cat" and "dog" are nearby (both pets)
- "Cat" and "airplane" are far apart (unrelated concepts)
- "King" and "queen" are close (both royalty)

If we can give each token the right coordinates, then math on those coordinates corresponds to reasoning about concepts.

---

## What Is a Vector?

Before going further, let's demystify "vector."

A vector is just a list of numbers:

```
[0.25, -1.3, 0.08, 2.1, -0.5, ...]
```

That's it. Nothing scary.

You can think of it as:

- **Coordinates in space** — like GPS coordinates, but with hundreds of dimensions
- **A point** — each list identifies a specific location
- **A description** — each number captures one aspect of meaning

LLMs use vectors with hundreds or thousands of numbers. Different models use different sizes:

| Model | Embedding Dimensions |
|-------|---------------------|
| GPT-2 Small | 768 |
| GPT-2 Large | 1,280 |
| GPT-3 (175B) | 12,288 |

Each token gets its own vector of that size.

---

## The Embedding Matrix

Here's how it works in practice.

The model stores a giant table called the **embedding matrix**. It has one row for every token in the vocabulary:

```
Token ID 0    → [0.1, -0.3, 0.5, ...]  (768 numbers for GPT-2)
Token ID 1    → [0.2, 0.1, -0.8, ...]
Token ID 2    → [-0.4, 0.6, 0.2, ...]
...
Token ID 318  → [0.7, -0.2, 1.1, ...]  ← " is"
...
Token ID 50256 → [-0.1, 0.9, -0.3, ...] ← <|endoftext|>
```

The matrix size is **vocabulary size × embedding dimension**. For GPT-2 Small, that's 50,257 × 768 = **38.6 million numbers** just for embeddings!

When the model sees token ID 318, it looks up row 318 and retrieves the embedding:

```python
embedding = embedding_matrix[token_id]
```

This embedding is the meaning of that token — represented as coordinates in high-dimensional space.

---

## Why Similar Concepts End Up Nearby

The embeddings aren't assigned by hand. They're **learned during training** — optimized alongside all the other model weights.

This is different from older approaches like Word2Vec, where embeddings were pretrained separately and then frozen. In modern LLMs, the embedding matrix starts with random values and is refined during training to make the model's predictions better.

Here's the intuition: If two words appear in similar contexts, they get similar embeddings.

Consider:

- "The **cat** sat on the mat"
- "The **dog** sat on the mat"

"Cat" and "dog" appear in nearly identical contexts. During training, the model adjusts their embeddings to be similar, because similar embeddings lead to similar predictions.

Over billions of examples, meaningful structure emerges:

- Animals cluster together
- Countries cluster together
- Verbs cluster by tense and meaning

---

## The Famous Example: King - Man + Woman = Queen

You may have heard this result:

```
vector("King") - vector("Man") + vector("Woman") ≈ vector("Queen")
```

This isn't programmed. It emerges from the geometry of learned embeddings.

- Subtract "Man" from "King" → removes the "male" direction
- Add "Woman" → adds the "female" direction
- Result → lands near "Queen"

```
                    [Royal]
                       |
        King ●---------+--------- Queen ●
                       |
                       |
        Man ●----------+---------- Woman ●
                       |
                   [Gender]
```

The embedding space has learned abstract directions for concepts like "gender" and "royalty" — without ever being told these concepts exist.

---

## Embeddings ARE Weights

Here's an important technical point:

The embedding matrix is part of the model's **weights**. It's learned during training and saved with the model. It's not separate from the model — it IS the model (partially).

When you download GPT-2 or LLaMA, you're downloading the embedding matrix along with all the other weights. For GPT-2 Small, those 38.6 million embedding parameters represent about 31% of the model's total 124 million parameters.

The embeddings are optimized for the specific prediction task — unlike generic pretrained embeddings, they're tailored to what this particular model needs to predict well.

---

## Static vs. Contextual Embeddings

There's a subtle but important distinction:

**Static embeddings** (what we just described): Each token has ONE fixed embedding, regardless of context. "Bank" has the same embedding whether it's a riverbank or a financial bank.

**Contextual embeddings** (what transformers produce): After passing through transformer layers, each token's representation changes based on context. Now "bank" in "river bank" has a different vector than "bank" in "bank account."

The embedding matrix provides the starting point. The transformer layers create context-aware representations.

We'll see how in Part IV.

---

## The Critical Distinction (Expanded)

| Concept | What It Is | Has Meaning? | Example |
|---------|-----------|--------------|---------|
| Token ID | An integer | No — arbitrary label | 318 |
| Embedding | A vector | Yes — learned meaning | [0.7, -0.2, 1.1, ...] |

Token ID 318 is just an address. The embedding at row 318 is the content — the actual representation of what " is" means to this model.

### The Full Hierarchy: Token ID → Embedding → Vector → Tensor

These terms often cause confusion because they're related but distinct. Here's how they connect:

| Term | What It Is | Role in the Pipeline |
|------|-----------|---------------------|
| **Token ID** | Single integer | Index for lookup — has no meaning on its own |
| **Embedding** | A specific type of vector | Looked up using token ID — represents token meaning |
| **Vector** | Ordered list of numbers | Mathematical object that embeddings and hidden states are |
| **Tensor** | Multi-dimensional array | Container that holds vectors, matrices, and higher-dimensional data |

The key relationships:

- **Embeddings are vectors.** Every embedding is a vector — a list of numbers representing meaning.
- **Vectors are 1D tensors.** In deep learning frameworks, a vector is just a tensor with one dimension.
- **Tensors contain vectors.** The embedding matrix is a 2D tensor where each row is a vector (embedding).

When you see code or documentation referring to "tensors," it usually means the multi-dimensional arrays that contain model weights — including the embedding vectors.

### What Is a Dimension?

A dimension is simply one number in a vector — one axis in the space.

If a vector has 768 numbers, we say it has 768 dimensions. Each dimension captures some aspect of meaning, though no single dimension corresponds to a human-interpretable concept like "gender" or "size." Meaning is distributed across all dimensions.

Different models use different embedding dimensions:

- **768 dimensions**: GPT-2 Small, BERT-base
- **4,096 dimensions**: LLaMA-7B
- **12,288 dimensions**: GPT-3 (175B)

More dimensions generally mean more capacity to represent nuanced meaning — but also more memory and computation.

---

## Pause and Reflect

If "Paris" and "France" have similar embeddings (they appear in similar contexts), where do you think "Berlin" ends up?

And what's near Berlin?

You're starting to think like the model.

---

## In Practice: Embedding Models in OpenShift AI

The embeddings you've learned about aren't just internal to LLMs — you can use dedicated **embedding models** to convert text into vectors for search, retrieval, and RAG applications.

### Serving Embedding Models

**Red Hat OpenShift AI** can serve embedding models alongside LLMs. The Model Catalog includes embedding models like:

| Model | Dimensions | Use Case |
|-------|------------|----------|
| **nomic-embed-text-v1.5** | 768 | General-purpose text embeddings |
| **all-MiniLM-L6-v2** | 384 | Lightweight, fast embeddings |
| **BGE-large-en** | 1024 | High-quality English embeddings |

These models expose the `/v1/embeddings` endpoint:

```bash
curl -X POST https://nomic-embed.apps.cluster.example.com/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text-v1.5",
    "input": ["The Eiffel Tower is in Paris", "Berlin is the capital of Germany"]
  }'
```

Response:

```json
{
  "data": [
    {"embedding": [0.023, -0.041, 0.089, ...], "index": 0},
    {"embedding": [-0.012, 0.067, 0.034, ...], "index": 1}
  ]
}
```

### Why Separate Embedding Models?

LLMs generate embeddings internally, but dedicated embedding models are:
- **Smaller**: Faster inference, less GPU memory
- **Optimized for similarity**: Trained specifically for semantic search
- **Reusable**: Same embeddings work for search, clustering, classification

### Embeddings + RAG

Retrieval-Augmented Generation (RAG) uses embeddings to find relevant documents:

```
User Query: "What's the capital of France?"
        ↓
   Embed query → [0.23, -0.41, 0.89, ...]
        ↓
   Search vector database (similarity)
        ↓
   Retrieve: "Paris is the capital of France..."
        ↓
   Add to LLM prompt as context
        ↓
   LLM generates grounded response
```

OpenShift AI integrates with vector databases like Milvus (via Llama Stack) for this workflow. The embedding geometry you learned about directly enables semantic search — similar concepts have similar vectors, so searching by vector similarity finds relevant content.

---

## Chapter Takeaway

> **Embeddings are meaningful vectors.** Each token ID looks up an embedding from a learned matrix. Similar words get similar embeddings because they appear in similar contexts. This is where meaning enters the system — language becomes geometry, and geometry is math the model can compute. **In OpenShift AI, dedicated embedding models serve the `/v1/embeddings` endpoint for RAG and semantic search applications.**

---

## Part II Summary

You've learned how language becomes numbers:

1. **Tokenization** breaks text into pieces and assigns arbitrary integer IDs.
2. **Embeddings** convert those IDs into meaningful vectors — points in a high-dimensional space where similar concepts are nearby.

Now the model has something it can actually compute with: vectors of numbers that represent meaning.

But how does it process those vectors? What happens inside the transformer? That's next.

---

*Next: [Chapter 5: How Machines Learn Anything](05-neural-networks.md)*
