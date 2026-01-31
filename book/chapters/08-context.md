# Chapter 8: Context — The Most Misunderstood Word in AI

*Part IV: The Secret Sauce*

---

When people say "the model has a 128k context window," what do they actually mean?

And when we talk about "contextual embeddings," is that the same thing?

These are two completely different concepts, and confusing them leads to deep misunderstandings. Let's clear it up.

---

## Two Meanings of "Context"

### Meaning 1: Context Window (Input Limit)

The **context window** is simply: **How many tokens can the model see at once?**

- GPT-4 Turbo: 128,000 tokens
- Claude 3: 200,000 tokens
- Original GPT-3: 4,096 tokens

This is a limit, not a memory. It's like the size of your desk — it determines how many documents you can spread out at once.

If your conversation exceeds the context window, old messages get dropped. The model literally cannot see them anymore.

### Meaning 2: Contextual Representation (Computed Meaning)

A **contextual representation** is: **The vector computed for a token after attention**.

Before attention: Each token has a static embedding (from Chapter 4).

After attention: Each token has a contextual vector that blends information from other tokens.

"Bank" starts with the same embedding. After attention:
- In "river bank" → vector shifted toward nature/water
- In "bank account" → vector shifted toward finance

The contextual representation is the token's **computed meaning in this specific context**.

---

## The Critical Distinction

| Term | What It Is | Is It Stored? |
|------|-----------|---------------|
| Context Window | Maximum input length (a number like 128k) | It's a limit, not data |
| Contextual Representation | A vector computed by attention | Created during processing, then discarded |

These are not the same thing. One is a constraint. The other is a computation.

---

## The Desk Analogy

**Context window = desk size**

A bigger desk lets you spread out more papers. But it doesn't remember anything. Each time you sit down, you start fresh.

**Contextual representation = your understanding while reading**

While you're reading the papers on your desk, you form an understanding. That understanding exists only while you're working. When you leave, it's gone.

---

## A Common Misconception

Many people think:

> "The model remembers our conversation because of the context window."

This is subtly wrong. Here's what actually happens:

1. You send a message
2. The **entire conversation** (within the context window) is re-processed from scratch
3. The model generates a response
4. All internal states (including contextual representations) are discarded
5. Next message: goto step 1

Nothing is "remembered." The conversation history is re-read every single time.

The context window just determines how much history can be included.

---

**You might be wondering:** *"If the model re-reads everything each time, why does it seem to 'remember' earlier parts of the conversation? Why doesn't it contradict itself?"*

The model appears to remember because the full conversation history is included in the input each time. When you ask a follow-up, the model sees both your original question and the follow-up together, so it can respond consistently. It's not memory — it's re-reading. If you were to remove earlier messages from the input, the model wouldn't reference them because it literally can't see them. The apparent consistency comes from seeing the full history, not from any persistent internal state.

---

**The Revelation:**

> Nothing is remembered. Everything is re-read, every time.

When you ask a follow-up question, the model doesn't "recall" your previous question. It re-reads the entire conversation — including your previous question — as if for the first time. Each response is computed from scratch. There is no persistent memory between calls.

---

## Context Is Computed, Not Stored

Let's be precise:

**Before each response**, the model:
- Receives the full conversation as input
- Tokenizes everything
- Looks up embeddings
- Runs attention (creating contextual representations)
- Generates the response token by token

**After the response**, the model:
- Discards all internal states
- Has no memory of what just happened
- Will re-read everything next time

The contextual representations exist only during the computation. They're like thoughts you have while reading a book — real while you're reading, gone when you close the book.

---

## What "Understanding Context" Really Means

When we say a model "understands context," we mean:

**The attention mechanism successfully extracted relevant information from surrounding tokens.**

"The trophy didn't fit in the suitcase because it was too big."

For the model to resolve "it" correctly:
1. The token "it" generates a Query
2. "Trophy" and "suitcase" have Keys
3. Attention scores determine which is more relevant
4. If attention correctly focuses on "trophy," the model resolves the reference

The model doesn't "understand" in a human sense. It computes a weighted blend of vectors, and if the training was good, that blend captures the right relationships.

---

## Where Context Comes From

Context is created through **multiple attention layers**.

Layer 1: Local relationships
- Adjacent tokens mix

Layer 12: Medium-range patterns
- Phrases and clauses form

Layer 50: High-level understanding
- Concepts, themes, and discourse

Each layer refines the contextual representations. By the final layer, each token's vector encodes both its identity and its role in the broader text.

---

## Position Information

One subtlety: attention by itself doesn't know position. "Dog bites man" and "Man bites dog" would look the same!

Transformers solve this with **positional encodings** — extra information that encodes where each token appears.

### Absolute Position Embeddings (GPT-2 Style)

GPT-2 uses **learned absolute position embeddings**:

```
Position 0 → [0.1, -0.3, 0.5, ...]  (768 dimensions)
Position 1 → [0.2, 0.1, -0.4, ...]
Position 2 → [-0.1, 0.4, 0.2, ...]
...
Position 1023 → [0.3, -0.2, 0.8, ...]
```

These position embeddings are **added** to the token embeddings before entering the transformer:

```
Input = Token Embedding + Position Embedding
```

Both embeddings have the same dimension (768 for GPT-2 Small). The model learns position embeddings during training, just like it learns token embeddings.

The limitation: the model can't handle positions beyond what it was trained on (position 1024+ would have no embedding).

### Rotary Position Embeddings (RoPE)

Newer models like LLaMA use **Rotary Position Embeddings (RoPE)**:

Instead of adding position information to the embedding, RoPE encodes **relative** positions directly in the Q and K vectors during attention computation. It essentially rotates the vectors based on position.

The advantage: RoPE captures **relative distances** ("how far apart are these tokens?") rather than **absolute positions** ("is this token 5th or 500th?"). This generalizes better to longer sequences than the model was trained on.

Key distinction: GPT-2 uses learned absolute positions, while newer models often use RoPE for better length generalization.

---

**You might be wondering:** *"What does 'rotates the vectors based on position' actually mean? How is rotation different from addition?"*

RoPE applies a rotation matrix to Q and K vectors based on their positions. Mathematically, rotation preserves vector magnitude (length) while changing direction. Instead of adding a position vector (which shifts the point in space), rotation changes the angle of the vector. The key insight: when you compute Q × K for attention, the dot product of two rotated vectors depends on their *relative* rotation — which encodes relative distance. This means "token A is 5 positions before token B" is captured, regardless of absolute positions. That's why RoPE generalizes better to sequences longer than training.

---

## Pause and Reflect

Consider a 100-message conversation. The context window is 128k tokens.

- Does the model "remember" message 1? **No.** It re-reads it.
- Is message 1 always included? **Only if it fits.** If the conversation is too long, old messages are truncated.
- Does the model's "understanding" persist between messages? **No.** Everything is recomputed from scratch.

This is why:
- Adding context to prompts works (the model sees it fresh each time)
- The model can't learn from your corrections during a conversation
- Long conversations eventually "forget" the beginning (truncation)

---

## Why This Matters

Understanding this distinction prevents confusion:

| Misconception | Reality |
|--------------|---------|
| "The model remembers our chat" | It re-reads the chat each time |
| "It learned from my feedback" | It can't learn during inference |
| "Context window = memory" | Context window = input limit |
| "Context is stored somewhere" | Context is computed fresh, then discarded |

---

## Chapter Takeaway

> **"Context" has two meanings.** The context window is an input size limit. The contextual representation is a vector computed by attention. Neither is stored memory. The model re-reads and recomputes everything for each response. Nothing persists between calls.

---

## Part IV Summary

You've mastered the secret sauce:

1. **Attention** lets tokens selectively focus on relevant other tokens using Q, K, V
2. **Context** is computed dynamically by attention — not stored
3. **Multi-head attention** provides multiple perspectives
4. **Position encodings** add sequence information

---

## What You Can Now Explain

After Part IV, you can confidently explain:

- How attention works: Query, Key, Value, and weighted blending
- Why "in" in our running example focuses on "Tower" and "Eiffel" to predict "Paris"
- Why the same word becomes different vectors in different contexts
- The difference between "context window" (input limit) and "contextual representation" (computed meaning)
- Why models don't "remember" conversations — they re-read everything each time

You're about 60% of the way to understanding how LLMs work.

---

Now let's see the full flow: how "The Eiffel Tower is located in" becomes "Paris" — step by step.

*Next: [Chapter 9: Inference — A Token's Journey](09-inference.md)*
