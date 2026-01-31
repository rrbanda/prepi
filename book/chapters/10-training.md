# Chapter 10: Training — How Intelligence Emerges

*Part V: Watching It Think*

---

We've seen how inference works — the model predicts tokens using its weights.

But how did those weights get their values in the first place?

How did billions of random numbers become a system that can write poetry and debug code?

---

## The Core Idea: Prediction and Correction

Training is surprisingly simple in concept:

1. Show the model some text
2. Ask it to predict the next token
3. Compare its prediction to the actual next token
4. Adjust the weights to make the prediction better
5. Repeat billions of times

That's it. Intelligence emerges from prediction and correction at massive scale.

---

## The Same Forward Pass

Training uses the **exact same forward pass** as inference:

```
Text → Tokenize → Embed → Transformer Layers → Logits
```

The difference is what happens after.

---

## The Training Example

Consider our running example as training text:

```
"The Eiffel Tower is located in Paris"
```

We create training examples by sliding a window:

| Input | Target |
|-------|--------|
| "The" | "Eiffel" |
| "The Eiffel" | "Tower" |
| "The Eiffel Tower" | "is" |
| "The Eiffel Tower is" | "located" |
| "The Eiffel Tower is located" | "in" |
| "The Eiffel Tower is located in" | "Paris" |

For each input, the model must predict the target. This is how the model learned that "Paris" follows our prompt.

---

## Step-by-Step Training

Let's trace through our running example:

**Input**: "The Eiffel Tower is located in"  
**Target**: "Paris"

### Step 1: Forward Pass

```
Input: "The Eiffel Tower is located in"
    ↓
Tokenize: [464, 36751, 417, 8765, 318, 5140, 287]
    ↓
Embed: 7 vectors
    ↓
Transformer layers: contextual vectors
    ↓
Output projection: logits for all 50,000 tokens
```

### Step 2: Check the Prediction

The model outputs a probability for every token:

```
P("Paris")  = 0.23   (23% — not confident)
P("London") = 0.18
P("the")    = 0.08
P("Rome")   = 0.07
... (50,000 probabilities)
```

The correct answer is "Paris," but the model only gave it 23% probability.

### Step 3: Compute the Loss

**Loss** measures how wrong the prediction was.

Common loss: **Cross-entropy loss**

```
Loss = -log(P(correct token))
     = -log(0.23)
     = 1.47
```

If the model had given "Paris" 90% probability:
```
Loss = -log(0.90) = 0.105  ← much lower loss
```

Lower loss = better prediction.

### Step 4: Backpropagation

Here's where the magic happens.

We need to answer: **"How should we adjust each weight to reduce the loss?"**

Backpropagation computes this. It works backward:

```
Loss
  ↓ how does this change?
Output projection weights
  ↓ how does this change?
Layer 96 attention weights
  ↓ how does this change?
Layer 96 FFN weights
  ↓
... all the way back ...
  ↓
Embedding weights
```

For each weight, we compute: "If I nudge this weight up/down, does the loss go up or down?"

This produces a **gradient** for each weight — a direction to move.

---

**You might be wondering:** *"How does backpropagation compute gradients? What does 'working backward' mean?"*

Backpropagation uses the chain rule of calculus. Starting from the loss, it computes how each weight affects the loss by propagating gradients backward through layers. For each layer, it multiplies gradients from later layers by local derivatives (how that layer's output changes with its weights). This yields a gradient per weight indicating which direction to adjust it to reduce loss. The "backward" flow mirrors the forward pass: if data flowed A → B → C → Loss, gradients flow Loss → C → B → A.

---

### Step 5: Update the Weights

Using the gradients, we adjust every weight:

```
new_weight = old_weight - learning_rate × gradient
```

If the gradient says "moving this weight up increases loss," we move it down.

The learning rate (a small number like 0.0001) controls how big the steps are.

### Step 6: Repeat

Now process another example. And another. And another.

Billions of examples. Trillions of tokens.

Each example nudges the weights slightly. Over time, they converge to values that predict well.

---

## The Dart-Throwing Analogy

Training is like learning to throw darts:

1. **Throw** (forward pass): Predict where the dart lands
2. **See the result** (loss): How far from bullseye?
3. **Adjust your form** (backpropagation): Figure out what to change
4. **Try again** (weight update): Incorporate the adjustment
5. **Repeat**: Thousands of throws

After enough practice, you hit the bullseye reliably.

---

**The Revelation:**

> Billions of tiny corrections. That's all learning is.

The model doesn't "know" that the Eiffel Tower is in Paris. It has adjusted billions of weights — each nudged by a tiny amount, millions of times — so that when the input pattern looks like "Eiffel Tower is located in," the output pattern activates "Paris" strongly. Knowledge is the accumulated result of countless small corrections.

---

## Why Training Data Matters

The weights learn from the training data. This has implications:

| If training data has... | The model learns... |
|-------------------------|---------------------|
| Mostly English | English well, other languages poorly |
| Outdated information | Outdated facts |
| Biased perspectives | Those biases |
| Code examples | To write code |
| Errors and misinformation | Errors and misinformation |

The model is a compressed reflection of its training data.

---

## Scale: The Numbers

Modern training operates at staggering scale:

| Metric | Typical Value |
|--------|---------------|
| Parameters (weights) | 7 billion to 1 trillion+ |
| Training tokens | 1 trillion to 15+ trillion |
| Training time | Weeks to months |
| GPUs | Thousands of A100s or H100s |
| Cost | Millions of dollars |

Each of those weights is nudged millions of times, gradually settling into values that make good predictions.

---

## Training Hyperparameters

The training process is controlled by several key settings, called **hyperparameters**:

### Optimizer: AdamW

Most LLMs use **AdamW** (Adam with Weight Decay). It's smarter than simple gradient descent:

- **Adam** adapts the learning rate for each weight based on its history
- **Weight decay** prevents weights from growing too large (regularization)

You don't need to know the math, but know that AdamW is the standard optimizer for transformers.

### Learning Rate Schedule

The learning rate isn't constant:

```
Phase 1 (Warmup):    0 → 3e-4   (gradually increase)
Phase 2 (Training):  3e-4 → 0   (gradually decrease)
```

- **Warmup**: Start small to avoid unstable early updates
- **Decay**: Lower the learning rate as training progresses for fine-tuning

Typical peak learning rates are around 1e-4 to 6e-4 (0.0001 to 0.0006).

### Batch Size

Instead of updating weights after each example, we average gradients over a **batch**:

- Typical batch sizes: 512 to 2,048 sequences
- Larger batches = more stable gradients, but need more memory

Gradient accumulation can simulate large batches on smaller hardware.

### Gradient Clipping

Sometimes gradients explode to huge values, destabilizing training. **Gradient clipping** caps the maximum gradient magnitude:

```
if gradient_norm > max_norm (e.g., 1.0):
    scale gradients down
```

This prevents catastrophic weight updates.

---

## Perplexity: An Intuitive Metric

Loss is useful but not intuitive. **Perplexity** makes it interpretable:

```
Perplexity = exp(loss)
```

| Loss | Perplexity | Interpretation |
|------|-----------|----------------|
| 4.0 | 55 | Model is ~55-way confused at each token |
| 3.0 | 20 | Model is ~20-way confused |
| 2.0 | 7.4 | Model is ~7-way confused |
| 1.5 | 4.5 | Model is ~4-way confused |

Think of perplexity as: **"On average, the model is choosing between this many equally likely options."**

Lower perplexity = more confident, better predictions. GPT-2 achieved perplexity around 15-20 on common benchmarks.

---

## Training Cost: Concrete Numbers

Training frontier models is extraordinarily expensive:

| Model | Estimated Training Cost |
|-------|------------------------|
| GPT-3 (175B) | ~$4.6 million |
| LLaMA 2 (70B) | ~$3 million |
| GPT-4 | Estimated $50-100+ million |

For context, training LLaMA 2 7B required **184,320 GPU hours** on A100 GPUs. At cloud prices, even "small" models cost tens of thousands of dollars to train.

This is why companies don't retrain from scratch — they fine-tune existing models or use techniques like LoRA that update only a small subset of weights.

---

**You might be wondering:** *"If training costs millions, how do smaller companies afford to use LLMs?"*

Most don't train from scratch. They use pre-trained models (LLaMA, Mistral, Qwen, etc.) and either: (1) use them as-is for general tasks, (2) fine-tune on domain-specific data (much cheaper — thousands, not millions), or (3) use API services from providers who absorb training costs. Training a frontier model requires massive resources; using or adapting one doesn't. Fine-tuning with LoRA can cost under $100 for small models on cloud GPUs.

---

## Training vs. Inference Summary

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Goal** | Learn patterns | Apply patterns |
| **Weights** | Updated constantly | Fixed (read-only) |
| **Knows the answer?** | Yes (supervised) | No (must predict) |
| **Hardware** | Massive GPU clusters | Single GPU or API |
| **Time** | Weeks to months | Milliseconds to seconds |
| **Happens** | Once (or fine-tuning) | Every time you use it |

---

## Where Knowledge Lives

Here's the deep question: "Where does knowledge live?"

Not in a database. Not in explicit rules. In **the relationships between weights**.

The model "knows" Paris is France's capital because thousands of weight values, arranged precisely, push "Paris" to the top when the input pattern matches.

It's not a fact stored somewhere. It's an emergent property of the weight geometry.

---

## Pause and Reflect

Think about this: The model has never been told that Paris is a city, that France is a country, or that countries have capitals.

It simply observed patterns like:
- "France... Paris" appears frequently
- "capital of France" often precedes "Paris"
- "The Eiffel Tower is in Paris" is a common sequence

From these statistics, the right prediction emerges.

Understanding is not required. Prediction is sufficient.

---

## Chapter Takeaway

> **Training is prediction + correction, repeated billions of times.** The model predicts the next token, compares to the actual answer, and adjusts weights using AdamW optimizer with learning rate scheduling. Perplexity measures how confused the model is — lower is better. Training GPT-3 cost ~$4.6 million; GPT-4 cost far more. Over massive scale, statistical patterns become encoded in weights. Knowledge emerges from geometry, not explicit storage.

---

## In Practice: Training Environments with OpenShift AI

The training process you've learned requires massive GPU clusters, specialized libraries, and careful environment management. **Red Hat OpenShift AI** provides an enterprise platform that makes this practical.

### Workbenches: Your Training Environment

In OpenShift AI, a **workbench** is a containerized environment designed for ML training. It runs as a Kubernetes pod with:

- **JupyterLab** for interactive development
- **Pre-installed ML libraries** (PyTorch, TensorFlow, scikit-learn)
- **GPU support** via NVIDIA operators and CUDA
- **Persistent storage** that survives restarts

Think of a workbench as your personal GPU-enabled development machine, but running in the cloud with enterprise security and scalability.

### GPU-Ready Workbench Images

OpenShift AI provides pre-configured images with CUDA drivers and ML frameworks:

| Image | Libraries Included | Use Case |
|-------|-------------------|----------|
| **PyTorch** | PyTorch, CUDA, cuDNN | Training transformer models |
| **TensorFlow** | TensorFlow, Keras, CUDA | Alternative framework |
| **Standard Data Science** | NumPy, Pandas, scikit-learn | Data preprocessing |
| **Minimal Python** | Base Python environment | Custom setups |

No more wrestling with CUDA driver compatibility — the images are tested and maintained.

### Data Connections: Where Models Live

Training produces model files that need to be stored. OpenShift AI uses **data connections** — configurations that connect workbenches to S3-compatible storage:

```yaml
# Data connection configuration (stored as a Secret)
AWS_ACCESS_KEY_ID: <access_key>
AWS_SECRET_ACCESS_KEY: <secret_key>
AWS_S3_ENDPOINT: https://s3.example.com
AWS_S3_BUCKET: models
```

Your training notebook can upload models directly to S3, making them available for serving:

```python
import boto3
import os

# Credentials injected as environment variables
s3 = boto3.client('s3',
    endpoint_url=os.getenv('AWS_S3_ENDPOINT'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

# Upload trained model
s3.upload_file('model.pt', os.getenv('AWS_S3_BUCKET'), 'models/llama-finetuned/model.pt')
```

### Model Registry: Version Control for Models

After training, you need to track model versions, metadata, and lineage. The **Model Registry** in OpenShift AI provides this:

| Capability | Description |
|------------|-------------|
| **Version tracking** | Register multiple versions of the same model |
| **Metadata storage** | Store hyperparameters, metrics, training data references |
| **Lineage tracking** | Know which dataset and code produced each model |
| **Deployment integration** | Deploy directly from registry to model serving |

The workflow:

```
Train Model → Upload to S3 → Register in Model Registry → Deploy to Serving
```

When you register a model, you capture:
- Model name and version
- Storage location (S3 URI)
- Description and labels
- Training metrics (accuracy, loss, etc.)

Later, when deploying, you can select any registered version — making rollbacks and A/B testing straightforward. The Model Registry API follows the ML Metadata (MLMD) standard, so it integrates with MLOps pipelines and tools.

### Distributed Training with KubeRay and Kueue

For training models that don't fit on a single GPU, OpenShift AI 3.x provides **distributed workloads** via the KubeRay Operator:

- **Ray**: Distributed computing framework for parallel training (managed by KubeRay)
- **Red Hat build of Kueue**: Job queuing and resource management across training, workbenches, and model serving
- **Kubeflow Trainer v2**: Simplified distributed training with TrainJob API (Technology Preview)
- **Multi-node training**: Spread training across multiple GPU nodes

KubeRay provides capabilities like mTLS, network isolation, and authentication for secure distributed training. This is how organizations train models at scale without managing infrastructure manually.

### The MLOps Workflow

OpenShift AI connects training to production through **Data Science Pipelines**:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Ingest     │ ──▶ │  Train      │ ──▶ │  Evaluate   │ ──▶ │  Deploy     │
│  Data       │     │  Model      │     │  Model      │     │  Model      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
     ▲                    │                   │                   │
     │                    ▼                   ▼                   ▼
   S3 Storage ◀────── Workbench ────▶ Metrics ────▶ Model Serving
```

Pipelines automate the repetitive cycle of training, evaluation, and deployment — turning the manual process you learned into reproducible, automated workflows.

---

## Part V Summary

You've seen the complete lifecycle:

1. **Inference**: Prompt → Tokenize → Embed → Transform → Predict → Sample → Repeat
2. **Training**: Same forward pass + loss + backpropagation + weight update

The model doesn't understand. It predicts. But it predicts so well that it appears to understand.

---

## What You Can Now Explain

After Part V, you can confidently explain:

- The complete journey of "The Eiffel Tower is located in" → "Paris"
- Every step: tokenization, embedding, transformer layers, output projection, softmax, sampling
- Why inference is one-token-at-a-time (autoregressive generation)
- How training works: forward pass, loss, backpropagation, weight update
- Why training costs millions of dollars (trillions of examples, billions of weight updates)
- What the Prologue promised — you can now explain how LLMs work

You're about 75% of the way to understanding how LLMs work. The remaining chapters cover the *engineering* that makes this practical at scale.

---

Now let's look under the hood at optimizations that make inference fast enough to be useful.

*Next: [Chapter 11: The KV Cache — Why Inference Doesn't Crawl](11-kv-cache.md)*
