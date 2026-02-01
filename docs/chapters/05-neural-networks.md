# Chapter 5: How Machines Learn Anything

*Part III: The Brain's Blueprint*

---

We've converted "The Eiffel Tower is located in" into 7 embedding vectors — 7 points in 768-dimensional space. Now what?

Those vectors need to be *processed* — transformed, analyzed, and eventually turned into a prediction of the next word. This is where neural networks come in.

Don't worry if you've never studied neural networks. We're starting from scratch.

---

## Forget What You Think You Know

The term "neural network" conjures images of glowing brain diagrams and sci-fi AI. Let's ground it in reality.

A neural network is:

- **Not a brain simulation** — it's inspired by brains, but loosely
- **Not mysterious** — it's just math (additions, multiplications, simple functions)
- **Not magic** — every step can be explained and understood

At its core, a neural network is a function that transforms input into output. The "neural" part just describes how we structured the function.

---

## What Is a Layer?

The building block of a neural network is a **layer**.

A layer is simply:

```
Input → Transformation → Output
```

That's it. Data goes in, something happens to it, data comes out.

For example:

```
Input:  [0.5, -0.2, 0.8]
        ↓
Layer:  Multiply by weights, add bias, apply function
        ↓
Output: [0.3, 0.7, -0.1, 0.4]
```

Notice the output can have a different size than the input. The layer reshapes and transforms the data.

---

## What Are Weights?

Inside each layer are **weights** — numbers that control how the transformation works.

Think of weights as **knobs**. Each knob adjusts some aspect of the transformation.

```
If weight W1 is high → the output emphasizes certain patterns
If weight W1 is low  → the output de-emphasizes those patterns
```

The model has billions of these knobs. The art of training is finding the right setting for each one.

Key insight:

> **Weights are learned, not programmed.** We don't set them by hand. The training process finds values that work.

---

**You might be wondering:** *"How can billions of weights be learned automatically? What does 'learned, not programmed' mean in practice?"*

During training, the model uses backpropagation: it makes a prediction, compares it to the correct answer, computes gradients (how much each weight contributed to the error), and adjusts weights to reduce that error. This happens across millions of examples. The weights start random and converge to useful values through repeated small updates. The architecture defines the structure; training finds the values.

---

## The Forward Pass

When data flows through a neural network, it's called a **forward pass**.

```
Input
  ↓
Layer 1: Transform
  ↓
Layer 2: Transform
  ↓
Layer 3: Transform
  ↓
...
  ↓
Layer N: Transform
  ↓
Output
```

Each layer takes the previous layer's output as its input. The data "flows forward" through the network.

For our running example:

```
"The Eiffel Tower is located in" (7 embedding vectors)
  ↓
Transformer Layer 1: vectors refined
  ↓
Transformer Layer 2: further refined
  ↓
...
  ↓
Transformer Layer 96 (GPT-3 175B has 96 layers): final representations
  ↓
Logits → "Paris" predicted as most likely next word
```

Each layer transforms the vectors, making them richer with context and meaning.

---

## Why "Deep" Matters

A "deep" neural network just means: lots of layers.

Why does depth help?

**Composition.** Each layer does something simple. But simple transformations, stacked many times, can represent incredibly complex functions.

Think of it like art:

- Layer 1: Recognize edges and basic shapes
- Layer 5: Recognize textures and patterns
- Layer 20: Recognize objects and concepts
- Layer 50: Recognize relationships and context

Each layer builds on the previous, creating increasingly abstract representations.

---

## The Factory Analogy

Imagine an assembly line:

```
Raw Materials (text)
    ↓
Station 1: Clean and prep
    ↓
Station 2: Shape basic forms
    ↓
Station 3: Add details
    ↓
Station 4: Refine and polish
    ↓
...
    ↓
Final Product (prediction)
```

Each station (layer) has workers with specific skills (weights). The raw materials get transformed at each step. What emerges is something much more refined than what went in.

The key: **The workers' skills are learned through practice (training), not programmed from day one.**

---

## Model = Architecture + Weights

Here's an important distinction:

**Architecture**: The blueprint. How many layers? What type? How are they connected?

**Weights**: The learned values inside the architecture.

Same architecture + different weights = completely different behavior.

When you hear "GPT-4" vs "GPT-3.5", both might use similar architectures, but they have different weights (and GPT-4 is larger — more layers, more weights).

---

## The Key Insight About Intelligence

Here's the most important point in this chapter:

> **Intelligence is not in the architecture. Intelligence is in the weights.**

Before training:

- Architecture exists ✓
- Weights are random ✗
- Model outputs gibberish ✗

After training:

- Architecture is the same
- Weights have been tuned
- Model outputs coherent text ✓

The architecture is like a blank brain. The weights are the knowledge, skills, and patterns it has learned.

---

**The Revelation:**

> Intelligence is in the weights, not the architecture.

You can download the GPT-2 architecture right now — it's open source. But without trained weights, it outputs random garbage. The architecture is just a container. The weights are everything.

This is why companies guard their weights carefully, and why model weights are often proprietary even when architectures are public.

---

## What Happens Inside Each Layer

While details vary, most layers in an LLM do something like:

1. **Linear transformation**: Multiply input by weight matrix
2. **Non-linearity**: Apply a function that introduces complexity (like ReLU or GELU)
3. **Normalization**: Keep values in reasonable ranges

The non-linearity is crucial. Without it, stacking layers would be pointless — multiple linear transformations collapse into one. The non-linearity lets each layer contribute something unique.

Let's explore three specific techniques that make deep networks trainable.

---

## Layer Normalization

Training deep networks is hard. As data passes through many layers, values can grow explosively large or shrink to near zero. This makes learning unstable.

**Layer Normalization** solves this by adjusting each layer's output to have:

- Mean = 0
- Variance = 1

```
Before normalization: [100.5, -50.2, 75.8, ...]  ← wild values
After normalization:  [0.8, -1.2, 0.4, ...]      ← tamed values
```

Think of it as a **thermostat for neural networks** — it keeps the "temperature" of activations in a comfortable range, preventing runaway heating or freezing.

GPT models use **Pre-LayerNorm**, meaning normalization happens *before* each attention and feed-forward block, not after. This ordering turns out to train more stably.

---

## Activation Functions: GELU vs ReLU

After a linear transformation, we need a non-linear function. Otherwise, stacking layers is pointless — multiple linear operations collapse into one.

The classic choice is **ReLU** (Rectified Linear Unit):

```
ReLU(x) = max(0, x)

If x > 0: output x
If x ≤ 0: output 0
```

ReLU is simple: positive values pass through, negative values become zero.

But GPT uses **GELU** (Gaussian Error Linear Unit):

```
GELU(x) ≈ x × probability that x is "large"
```

The key difference:

| Function | At x = -0.5 | At x = 0 | At x = 1 |
|----------|-------------|----------|----------|
| ReLU | 0 | 0 | 1 |
| GELU | -0.15 | 0 | 0.84 |

GELU is **smooth** — it has no sharp corner at zero. This smoothness helps optimization because gradients flow more consistently. GELU also allows small negative values through, unlike ReLU's hard cutoff.

You don't need to memorize the formula. Just know: **GPT uses GELU because it trains better than ReLU for language tasks.**

---

## Residual Connections (Shortcuts)

Here's a problem: In very deep networks (50+ layers), gradients during training can shrink to nearly zero. By the time error signals reach early layers, they're too small to cause meaningful updates. This is the **vanishing gradient problem**.

The solution is remarkably simple: **add shortcuts**.

```
Regular:    x → Layer → output

Shortcut:   x → Layer → output + x
```

Instead of just outputting the layer's result, we add the original input back:

```python
output = layer(x) + x  # This is a residual connection
```

Why does this help? 

1. **Guaranteed gradient path**: Even if the layer's gradients vanish, the shortcut provides a direct highway for gradients to flow backward.

2. **Learning becomes easier**: The layer only needs to learn the *difference* from input to output (the "residual"), not the entire transformation.

Think of it like this: Instead of learning to build a sculpture from nothing, you're learning what to add to an existing rough shape.

In GPT, every transformer block has two residual connections:
- After attention: `x = x + attention(x)`
- After feed-forward: `x = x + ffn(x)`

Without residual connections, training a 96-layer GPT-3 would be essentially impossible.

---

**You might be wondering:** *"Why doesn't adding the input back (x + layer(x)) mess up what the layer learned?"*

It actually helps, not hurts. The layer only needs to learn the *difference* — what to add or remove from the input. If the layer's contribution should be zero for some inputs, it can learn to output zeros, and the original input passes through unchanged. This makes learning easier: instead of learning the entire transformation from scratch, the layer learns incremental modifications. It's like giving an artist a sketch to refine rather than a blank canvas.

---

## Pause and Reflect

Consider this: The architecture is public. You can read the GPT-2 code. You can see exactly how the layers are structured.

What you CAN'T get (easily) is the weights from GPT-4. Those weights — billions of numbers — are where the "secret sauce" lives.

If you copied the architecture but trained with random weights on a small dataset, you'd get garbage. The value is in what was learned.

---

## Chapter Takeaway

> **Neural networks transform data through layers.** Each layer applies a learned transformation controlled by weights. Three techniques make deep networks trainable: **Layer Normalization** keeps values in reasonable ranges, **GELU activation** provides smooth non-linearity, and **Residual Connections** let gradients flow through shortcuts. The architecture defines the structure; the weights define the behavior. Before training, models know nothing. After training, intelligence emerges from the patterns encoded in the weights.

---

We have the machinery of neural networks — layers that transform data. But what makes transformers *special*? Before 2017, language models struggled with long text. Transformers changed everything.

*Next: [Chapter 6: The Transformer Revolution](06-transformers.md)*
