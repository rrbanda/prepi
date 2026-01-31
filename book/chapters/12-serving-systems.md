# Chapter 12: Serving Systems — vLLM and Friends

*Part VI: Under the Hood*

---

You've learned how a single request works. But what about serving millions of users simultaneously?

That's where **serving systems** come in — software that optimizes LLM deployment for production workloads.

---

## The Production Challenge

A single user making one request is straightforward. But real deployments face:

- **Thousands of concurrent requests**
- **Limited GPU memory** (expensive!)
- **Variable request lengths** (some short, some long)
- **Latency requirements** (users expect fast responses)
- **Cost pressure** (GPU time is expensive)

Naive deployment wastes resources. Serving systems fix this.

---

## What Serving Systems Optimize

### 1. KV Cache Management

The KV cache (Chapter 11) can consume massive amounts of memory. Serving systems manage this carefully:

- **Efficient allocation**: Don't waste memory on padding
- **Memory sharing**: Reuse memory across requests when possible
- **Preemption**: Pause low-priority requests if memory is needed

### 2. Batching

Instead of processing one request at a time:

```
Request A → Process → Done
Request B → Process → Done
Request C → Process → Done
```

Process multiple requests together:

```
[Request A, B, C] → Process together → [Done, Done, Done]
```

GPUs excel at parallel computation. Batching uses this to serve more users with the same hardware.

### 3. Continuous Batching

The problem with simple batching: requests have different lengths.

```
Request A: 10 tokens (done quickly)
Request B: 100 tokens (still generating)
Request C: 5 tokens (done even quicker)
```

Continuous batching dynamically adds and removes requests from the batch as they complete:

```
Batch starts: [A, B, C]
A completes: [B, C, D]  ← D joins
C completes: [B, D, E]  ← E joins
B completes: [D, E, F]  ← F joins
```

No waiting. Maximum GPU utilization.

#### Continuous Batching Step-by-Step

Let's trace through exactly what happens at each decoding step:

| Decoding Step | What Happens | Active Requests |
|---------------|--------------|-----------------|
| Step 1 | Generate token 1 for requests A, B, C | A, B, C |
| Step 2 | Request D arrives → added to batch | A, B, C, D |
| Step 3 | Generate: token 2 for A,B,C + token 1 for D | A, B, C, D |
| Step 4 | A completes (hit stop token) → removed | B, C, D |
| Step 5 | Request E arrives → added | B, C, D, E |
| Step 6 | Generate: token 3 for B,C + token 2 for D + token 1 for E | B, C, D, E |

Key insight: **Every request is treated independently, even in the same batch.** Requests can have different prompt sizes and generate different numbers of tokens. The system handles this naturally.

#### Why Continuous Batching Matters

| Metric | Static Batching | Continuous Batching |
|--------|-----------------|---------------------|
| GPU utilization | Low (waiting for slowest request) | High (always processing) |
| Latency | High (blocked by batch formation) | Low (immediate processing) |
| Throughput | Limited | Maximized |
| Memory efficiency | Poor | Optimized |

This is the default behavior in vLLM — you get it automatically.

---

## The Scheduler: Heart of the Serving System

The **scheduler** is the central coordinator that makes continuous batching work. It's the brain of the inference engine.

### What the Scheduler Does

```
┌─────────────────────────────────────────────────────────┐
│                      SCHEDULER                          │
│                                                         │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐      │
│   │  WAITING  │───▶│  RUNNING  │───▶│ COMPLETED │      │
│   │   Queue   │    │   Queue   │    │           │      │
│   └───────────┘    └─────┬─────┘    └───────────┘      │
│         ▲                │                              │
│         │                ▼                              │
│         │          ┌───────────┐                        │
│         └──────────│  SWAPPED  │                        │
│                    │   Queue   │                        │
│                    └───────────┘                        │
└─────────────────────────────────────────────────────────┘
```

### Request Lifecycle States

| State | Description | When It Happens |
|-------|-------------|-----------------|
| **Waiting** | Request received, not yet processing | New requests enter here |
| **Running** | Actively generating tokens | Scheduler picked this request |
| **Swapped** | Paused, KV cache moved to CPU | Memory pressure, preemption |
| **Completed** | Finished generating | Hit stop token or max length |

### Core Scheduler Responsibilities

| Component | Description |
|-----------|-------------|
| `schedule()` | Entry point — decides what requests run next |
| Queue management | Tracks waiting, running, and swapped requests |
| Token allocation | Determines how many tokens each request gets per step |
| Block management | Coordinates with BlockSpaceManager for KV cache |
| Preemption logic | Decides when to pause requests to free memory |
| Speculative coordination | Manages multi-step decoding when enabled |

### Preemption: Handling Memory Pressure

When GPU memory fills up, the scheduler must make hard choices:

```
Scenario: GPU memory at 95% capacity
          New high-priority request arrives
          
Options:
1. SWAP: Move a running request's KV cache to CPU RAM
         - Slower but preserves progress
         - Request can resume later
         
2. RECOMPUTE: Evict KV cache entirely
              - Request restarts from prompt
              - Wastes prior computation
              - Frees more memory
```

The scheduler uses policies to decide:
- **FIFO**: First-in-first-out eviction
- **Priority-based**: Low-priority requests evicted first
- **Size-based**: Large KV caches evicted first

### Why Understanding the Scheduler Matters

When debugging production issues:

| Symptom | Likely Cause | Scheduler-Related Fix |
|---------|--------------|----------------------|
| High latency spikes | Queue buildup | Increase `max_num_seqs` |
| OOM errors | Too many running requests | Reduce `max_num_batched_tokens` |
| Inconsistent latency | Preemption happening | Increase GPU memory or reduce concurrency |
| Low throughput | Conservative scheduling | Tune `gpu-memory-utilization` higher |

---

## vLLM: PagedAttention

**vLLM** is a popular open-source serving system. Its key innovation is **PagedAttention**.

### The Problem: Memory Fragmentation

Traditional KV cache allocation:

```
Request A needs space for 1000 tokens → allocate contiguous 1000-token block
Request B needs space for 500 tokens  → allocate contiguous 500-token block
Request C needs space for 2000 tokens → allocate contiguous 2000-token block

What if Request B finishes first?
→ 500-token hole in memory
→ Request D needs 800 tokens → doesn't fit in the hole!
→ Memory fragmented, wasted space
```

### The Solution: Paged Memory

vLLM treats KV cache like operating system virtual memory:

```
Divide memory into small pages (e.g., 16 tokens each)

Request A: needs 1000 tokens → allocate 63 pages (non-contiguous OK!)
Request B: needs 500 tokens  → allocate 32 pages
Request C: needs 2000 tokens → allocate 125 pages

Request B finishes → 32 pages freed
Request D needs 800 tokens → gets 50 pages, using B's freed pages + new ones

No fragmentation!
```

Pages can be anywhere in memory. A request's pages don't need to be contiguous.

This simple insight dramatically increases how many concurrent requests can run.

---

---

## Speculative Decoding: Thinking Ahead

**Speculative decoding** is a powerful optimization technique that can dramatically reduce inference latency without changing the model's outputs.

### The Core Idea

Instead of generating one token at a time with your large model:

1. Use a **small, fast "draft" model** to predict several tokens ahead
2. Have the **large "target" model** verify those predictions in parallel
3. Accept correct predictions, reject wrong ones
4. Net result: fewer expensive forward passes through the large model

```
Traditional Decoding:
Token 1 → [Large Model] → Token 2 → [Large Model] → Token 3 → [Large Model] → ...
          (slow)                    (slow)                    (slow)

Speculative Decoding:
                    ┌──────────────────────────────────┐
[Draft Model] ────▶ │ Speculate: Token 2, 3, 4, 5     │
(fast)              └──────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
[Target Model] ───▶ │ Verify all 4 in ONE forward pass │
(expensive but      └──────────────────────────────────┘
 done once)                        │
                                   ▼
                    Accept: 2, 3, 4 ✓  Reject: 5 ✗
                    → Generated 3 tokens with 1 expensive pass!
```

### How It Works in Detail

**Step 1: Draft Generation**
```
Draft model generates K tokens speculatively (e.g., K=4)
Draft: "The" → "capital" → "of" → "France" → "is"
```

**Step 2: Parallel Verification**
```
Target model runs ONE forward pass on all tokens
Computes: P(capital|The), P(of|The capital), P(France|The capital of), P(is|...)
```

**Step 3: Accept/Reject**
```
Compare draft probabilities to target probabilities
Accept tokens where draft matches target's preference
Reject tokens where draft diverged
```

**Step 4: Continue**
```
If token 3 was rejected, regenerate from there
If all accepted, draft next batch
```

### Benefits

| Benefit | Explanation |
|---------|-------------|
| **Lower latency** | Fewer expensive forward passes through large model |
| **Same output quality** | Target model still makes final decisions |
| **No retraining needed** | Drop-in optimization |
| **Model-agnostic** | Works with any transformer decoder |

### Trade-offs and Challenges

| Challenge | Details |
|-----------|---------|
| **Extra GPU memory** | Both draft and target model must be loaded |
| **Acceptance rate varies** | Depends on draft model quality and sampling strategy |
| **Implementation complexity** | Two models must coordinate |
| **Not always faster** | Short sequences or high randomness may not benefit |

### When Speculative Decoding Helps

| Scenario | Benefit Level |
|----------|---------------|
| Long generations (100+ tokens) | **High** — amortizes setup cost |
| Greedy/low-temperature sampling | **High** — predictable, high acceptance |
| Code generation | **High** — structured, predictable patterns |
| Short responses (<20 tokens) | **Low** — overhead exceeds benefit |
| High temperature/creative writing | **Low** — randomness kills acceptance rate |
| Beam search | **Not compatible** |

### Choosing a Draft Model

The draft model should be:

1. **Same tokenizer** as target (required — must share vocabulary)
2. **Similar generation style** (high acceptance rate)
3. **Much smaller** (otherwise no speed benefit)
4. **Fast** (the whole point is speed)

Common pairings:
- Mistral 7B (target) + Mistral 1.3B (draft)
- LLaMA 70B (target) + LLaMA 7B (draft)
- Custom distilled models

### vLLM Speculative Decoding

vLLM supports speculative decoding out of the box:

```bash
# Enable speculative decoding with a draft model
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-70B \
    --speculative-model meta-llama/Llama-3.2-7B \
    --num-speculative-tokens 4
```

Key parameters:
- `--speculative-model`: Path to the draft model
- `--num-speculative-tokens`: How many tokens to speculate (typically 3-5)
- `--speculative-draft-tensor-parallel-size`: Tensor parallelism for draft model

### Measuring Speculative Decoding Effectiveness

Track the **acceptance rate** — what percentage of speculated tokens are accepted:

| Acceptance Rate | Interpretation |
|-----------------|----------------|
| >80% | Excellent — significant speedup |
| 60-80% | Good — worthwhile optimization |
| 40-60% | Marginal — consider tuning |
| <40% | Poor — may not be worth the overhead |

If acceptance rate is low, check:
1. Is sampling too random? Lower temperature.
2. Is draft model too different from target? Try a better-matched draft.
3. Is the task unpredictable? Some content is inherently hard to speculate.

---

## Other Serving Systems

vLLM isn't the only option:

| System | Key Feature |
|--------|-------------|
| **vLLM** | PagedAttention, continuous batching |
| **TensorRT-LLM** | NVIDIA optimization, custom kernels |
| **Text Generation Inference (TGI)** | HuggingFace, easy to use |
| **Triton Inference Server** | NVIDIA, production-focused |
| **DeepSpeed-Inference** | Microsoft, large model support |

Different trade-offs: ease of use vs. raw performance, open-source vs. vendor-supported.

### Competitive Landscape: Understanding the Alternatives

When evaluating serving systems, understanding the key differences helps you make informed decisions:

#### NVIDIA TensorRT-LLM (TRT-LLM)

TensorRT-LLM is NVIDIA's optimized inference engine for LLMs.

| Aspect | Details |
|--------|---------|
| **Hardware** | NVIDIA GPUs only — no AMD, Intel, or CPU fallback |
| **Optimization approach** | Low-level graph optimizations, kernel fusion, quantization |
| **Strengths** | Maximum performance on NVIDIA hardware, tight CUDA integration |
| **Weaknesses** | Vendor lock-in, less flexible, steeper learning curve |
| **Best for** | Production workloads fully committed to NVIDIA |

Key difference from vLLM: TRT-LLM optimizes the *computation* (matrix operations), while vLLM optimizes *memory management* (PagedAttention). They solve different bottlenecks.

#### NVIDIA NIM (NVIDIA Inference Microservices)

NIM is NVIDIA's containerized model-serving solution.

| Aspect | Details |
|--------|---------|
| **What it is** | Pre-packaged containers for specific models (LLaMA, Mistral, etc.) |
| **Backend** | Uses TRT-LLM *or* vLLM depending on model and hardware |
| **Strengths** | Simple to deploy, OpenAI-compatible API, optimized out of box |
| **Weaknesses** | NVIDIA-only, less customization, closed source |
| **Best for** | Teams wanting fastest path to production on NVIDIA |

#### SGLang

SGLang is a fast, open-source serving framework with a programming focus.

| Aspect | Details |
|--------|---------|
| **Focus** | Developer-friendly programming interface for LLM workflows |
| **Key innovation** | RadixAttention for prefix caching, structured output support |
| **Strengths** | Excellent for agents, function-calling, complex pipelines |
| **Weaknesses** | Smaller community than vLLM, less enterprise focus |
| **Best for** | Building agentic applications, RAG pipelines |

#### Comparison Summary

| Feature | vLLM | TRT-LLM | NIM | SGLang |
|---------|------|---------|-----|--------|
| **Open source** | Yes | Yes | No | Yes |
| **Hardware flexibility** | NVIDIA, AMD, Intel | NVIDIA only | NVIDIA only | NVIDIA, AMD |
| **PagedAttention** | Yes | Different approach | Via vLLM/TRT | RadixAttention |
| **Ease of use** | High | Medium | Very high | High |
| **Customization** | High | High | Limited | High |
| **Enterprise support** | Via Red Hat | NVIDIA | NVIDIA | Community |

---

## Quantization: Making Models Smaller

**Quantization** reduces the precision of model weights, trading some accuracy for dramatic improvements in memory usage and inference speed.

### Why Quantization Matters

A typical LLM weight is stored in FP16 (16-bit floating point). Quantization converts these to lower precision:

```
FP16 (16-bit): 1.234567890123
INT8 (8-bit):  1.23
INT4 (4-bit):  1.2
```

| Precision | Bits per Weight | Memory Reduction | Speed Impact |
|-----------|-----------------|------------------|--------------|
| FP32 | 32 bits | Baseline | Slow |
| FP16/BF16 | 16 bits | 2× smaller | Standard |
| INT8 | 8 bits | 4× smaller | Faster |
| INT4 | 4 bits | 8× smaller | Much faster |

### Quantization Formats for vLLM

Different quantization formats optimize for different scenarios:

| Format | Description | Best Use Case |
|--------|-------------|---------------|
| **W4A16** | 4-bit weights, FP16 activations | Memory-constrained inference, edge deployment, containerized apps |
| **W8A8-INT8** | 8-bit weights, INT8 activations (per-token) | High-throughput serving, general purpose, works on any GPU |
| **W8A8-FP8** | 8-bit weights, FP8 activations | Accuracy-sensitive + memory constraints, Hopper GPUs (H100) |
| **2:4 Sparsity + FP8** | Structured sparsity with FP8 | Maximum speed on H100/Blackwell, production APIs |

### How to Choose

```
Decision Tree:

1. Is accuracy critical (legal, medical, financial)?
   → Use W8A8-FP8 or no quantization
   
2. Memory constrained (small GPU, edge, many models)?
   → Use W4A16
   
3. High throughput priority (API serving)?
   → Use W8A8-INT8
   
4. Have H100/Blackwell GPUs?
   → Consider 2:4 Sparsity + FP8 for max performance
```

### Quantization Tools

#### llm-compressor (Red Hat/Neural Magic)

The recommended tool for quantizing models for vLLM deployment:

```python
from llmcompressor import compress
from llmcompressor.modifiers import SmoothQuantModifier, GPTQModifier

# Quantize a model to W4A16
recipe = [
    SmoothQuantModifier(smoothing_strength=0.5),
    GPTQModifier(scheme="W4A16", ignore=["lm_head"]),
]

compressed_model = compress(
    model="meta-llama/Llama-3.2-7B",
    recipe=recipe,
    calibration_data="calibration_dataset"
)
```

llm-compressor supports:
- Post-training quantization (PTQ)
- Multiple quantization algorithms (GPTQ, SmoothQuant, AWQ)
- Calibration with your data

#### Other Ecosystem Tools

| Tool | Description |
|------|-------------|
| **AutoAWQ** | Activation-aware Weight Quantization |
| **bitsandbytes** | 8-bit and 4-bit quantization for training and inference |
| **GGUF** | Single-file quantized format (vLLM supports single-file only) |

### Quantization Trade-offs

| Aspect | Lower Precision | Higher Precision |
|--------|-----------------|------------------|
| Memory usage | Less | More |
| Inference speed | Faster | Slower |
| Accuracy | Lower (usually small) | Higher |
| Hardware support | May need specific GPU | Universal |

### Accuracy Impact

Well-quantized models typically recover **99%+ of baseline accuracy**:

| Model | Baseline (FP16) | W8A8-INT8 | W4A16 |
|-------|-----------------|-----------|-------|
| LLaMA 7B | 100% | 99.5% | 98.8% |
| Mistral 7B | 100% | 99.6% | 99.1% |
| LLaMA 70B | 100% | 99.7% | 99.3% |

*Accuracy measured on standard benchmarks (MMLU, HellaSwag, etc.)*

The accuracy drop is often imperceptible in practice, while the memory and speed gains are substantial.

---

## What They DON'T Change

Here's the important point: **Serving systems optimize how models run, not what models do.**

They do NOT change:
- The model architecture
- The model weights
- The attention math
- What the model outputs

They DO change:
- How fast you get the output
- How many users you can serve
- How much GPU memory you need
- How much it costs

Think of serving systems as **traffic management for GPUs**. They don't make the cars faster — they make the highway more efficient.

---

## When to Care About This

| Situation | Do You Need This Knowledge? |
|-----------|----------------------------|
| Using ChatGPT API | No — handled for you |
| Building a chatbot product | Maybe — for cost optimization |
| Deploying your own models | Yes — critical for performance |
| Working on ML/AI infrastructure | Yes — shows systems understanding |
| ML research | Maybe — depends on focus |

If you're just using LLMs through APIs, you can treat this as a black box. If you're deploying or optimizing, this is essential.

---

## Production Metrics Deep Dive

Understanding metrics is essential for operating LLMs in production. Different use cases prioritize different metrics.

### Latency Metrics

Latency isn't a single number — it's a collection of measurements across the inference pipeline.

#### Time to First Token (TTFT)

**Definition**: Time from request arrival to first token returned.

```
User sends request → [Prompt processing] → First token appears
                     |←──── TTFT ────────→|
```

**Why it matters**:
- Most user-visible latency metric
- Critical for chatbots, copilots, interactive tools
- High TTFT makes applications feel sluggish

**What affects TTFT**:
| Factor | Impact |
|--------|--------|
| Prompt length | Longer prompts = higher TTFT (more prefill computation) |
| Queue depth | More waiting requests = higher TTFT |
| Model size | Larger models = higher TTFT |
| Prefix caching | Repeated prompts = lower TTFT |

**vLLM tuning for TTFT**:
- Enable prefix caching: `--enable-prefix-caching`
- Use chunked prefill for long prompts
- Tune max batch size to prevent queue buildup

#### Time Per Output Token (TPOT)

**Definition**: Average time between consecutive generated tokens.

```
Token 1 → [TPOT] → Token 2 → [TPOT] → Token 3 → ...
```

**Why it matters**:
- Determines streaming "smoothness"
- Critical for code assistants, real-time summarization
- Users perceive smooth streaming as faster

**What affects TPOT**:
| Factor | Impact |
|--------|--------|
| Batch size | More concurrent requests = slightly higher TPOT |
| Quantization | Lower precision = lower TPOT |
| Speculative decoding | Reduces effective TPOT |

#### Inter-Token Latency (ITL)

**Definition**: Time between each subsequent token during streaming.

ITL and TPOT are related but distinct:
- **TPOT** = average across all tokens
- **ITL** = per-token measurement (can vary)

High ITL variance causes "stuttering" in streamed responses.

#### End-to-End Request Latency

**Definition**: Total time from request to complete response.

```
Request → [TTFT] → Token 1 → ... → Token N → Response complete
|←──────────────── End-to-End ────────────────────────→|
```

This includes:
- Queue waiting time
- Prompt processing (prefill)
- All token generation
- Network round-trip

### Throughput Metrics

#### Tokens Per Second (TPS)

**Definition**: Total output tokens generated per second across all requests.

```
System generates 1000 tokens/second across 50 concurrent requests
TPS = 1000
```

**Why it matters**:
- Raw capacity of the system
- Higher TPS = more users served
- Lower infrastructure cost per token

#### Requests Per Second (RPS)

**Definition**: Number of complete requests processed per second.

RPS depends on average output length:
```
TPS = 1000, average output = 100 tokens
RPS = 1000 / 100 = 10 requests/second
```

#### Goodput

**Definition**: Requests per second that meet Service Level Objectives (SLOs).

```
Goodput = Requests meeting SLAs / Total time
```

Example SLOs:
- TTFT < 500ms at P90
- TPOT < 50ms at P90
- End-to-end < 10s at P99

**Why it matters**:
- Aligns infrastructure performance with user-facing requirements
- Throughput alone doesn't guarantee good user experience
- Helps evaluate if a system is *usable*, not just *fast*

#### P95/P99 Latency

**Definition**: Latency at the 95th/99th percentile.

```
P99 = 500ms means 99% of requests complete within 500ms
```

**Why it matters**:
- Enterprise SLAs often defined at P95/P99
- Mean latency hides outliers
- Users remember the worst experiences

| Percentile | Typical Use |
|------------|-------------|
| P50 (median) | Typical user experience |
| P90 | Most users' worst experience |
| P95 | SLA threshold for many enterprises |
| P99 | Tail latency for critical applications |

### System Metrics

#### GPU Memory Utilization

**Definition**: Percentage of GPU VRAM in use.

```yaml
# vLLM setting
--gpu-memory-utilization=0.95  # Use 95% of GPU memory
```

| Utilization | Interpretation |
|-------------|----------------|
| <50% | Underutilized — can serve more requests or use larger batch |
| 70-90% | Healthy range |
| >95% | Risk of OOM, may need to reduce concurrency |

#### KV Cache Hit Rate

**Definition**: How often previously computed tokens are reused from cache.

High hit rates occur with:
- Repeated system prompts
- Similar conversations
- Prefix caching enabled

High hit rates reduce:
- TTFT (don't recompute prefill)
- GPU compute (reuse rather than recalculate)

#### Concurrency / Max Request Capacity

**Definition**: How many simultaneous requests the system handles.

Limited by:
- GPU memory (KV cache per request)
- Scheduler configuration (`max_num_seqs`)
- Compute bandwidth

### Cost Metrics

#### Cost Per Token

**Definition**: Infrastructure cost divided by tokens generated.

```
Monthly GPU cost: $10,000
Monthly tokens: 100M
Cost per token: $0.0001
```

Ways to reduce cost per token:
- Quantization (more tokens per GB VRAM)
- Higher batching efficiency
- Better GPU utilization
- Speculative decoding (more tokens per forward pass)

### Metric Optimization by Use Case

Different applications prioritize different metrics:

| Use Case | Primary Metrics | Secondary Metrics |
|----------|-----------------|-------------------|
| **Chatbots** | TTFT, ITL | End-to-end latency |
| **Code assistants** | TTFT, TPOT | Accuracy |
| **Document processing** | Throughput (TPS) | Cost per token |
| **RAG applications** | TTFT, concurrency | Memory utilization |
| **LLM-as-a-Service** | P95 latency, throughput | Cost, availability |
| **Interactive agents** | TTFT, ITL | Consistency |

---

## Evaluation and Benchmarking Tools

Measuring LLM performance requires specialized tools. Here are the most important ones:

### GuideLLM

**GuideLLM** is a benchmarking framework designed for evaluating LLM inference performance against realistic workloads.

#### What It Measures

| Metric Category | Specific Metrics |
|-----------------|------------------|
| **Latency** | TTFT (mean, median, P99), TPOT, ITL, request latency |
| **Throughput** | Requests/second, output tokens/second, total tokens/second |
| **Concurrency** | Average concurrent requests |

#### Running a Benchmark

```bash
# Install
pip install guidellm

# Run benchmark against any OpenAI-compatible server
guidellm \
    --target "http://localhost:8000/v1" \
    --model "meta-llama/Llama-3.2-8B" \
    --data-type emulated \
    --data "prompt_tokens=512,generated_tokens=128"
```

#### Understanding Results

GuideLLM runs a **sweep** across multiple load levels:

```
Benchmark      | Requests/sec | Output TPS | TTFT P99 | TPOT P99
---------------|--------------|------------|----------|----------
synchronous    | 1.2          | 153        | 45ms     | 12ms
constant@6     | 6.2          | 791        | 81ms     | 14ms
constant@28    | 27.2         | 3,400      | 112ms    | 15ms
throughput     | 44.4         | 5,600      | 363ms    | 18ms
```

**Interpretation**:
- As load increases, throughput increases but latency degrades
- Find the "sweet spot" where throughput is high but P99 latency is acceptable
- Use this to set production concurrency limits

#### Calculating Goodput

1. Define your SLOs (e.g., TTFT < 200ms, TPOT < 20ms at P90)
2. Run GuideLLM with various load levels
3. Count requests meeting SLOs
4. Divide by test duration = Goodput

### lm-eval-harness

**lm-eval-harness** is the standard framework for evaluating model accuracy on benchmarks.

#### What It Measures

| Category | Benchmarks |
|----------|------------|
| **General knowledge** | MMLU, HellaSwag, ARC, TruthfulQA |
| **Reasoning** | GSM8K, MATH, BIG-Bench |
| **Code** | HumanEval, MBPP |
| **Language understanding** | LAMBADA, WinoGrande |

#### Running Evaluations

```bash
# Install
pip install lm-eval

# Evaluate a model on MMLU
lm_eval --model vllm \
    --model_args pretrained=meta-llama/Llama-3.2-8B \
    --tasks mmlu \
    --batch_size auto
```

#### When to Use

- **Comparing quantized vs. full-precision models**: Does quantization hurt accuracy?
- **Evaluating fine-tuned models**: Did fine-tuning improve task performance?
- **Model selection**: Which model performs best on your target tasks?

### Ragas

**Ragas** (Retrieval Augmented Generation Assessment) evaluates RAG pipeline quality.

#### What It Measures

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Is the answer grounded in retrieved context? |
| **Answer relevancy** | Does the answer address the question? |
| **Context precision** | Are retrieved documents relevant? |
| **Context recall** | Did retrieval find all relevant information? |

#### When to Use

- Evaluating RAG applications
- Comparing retrieval strategies
- Measuring hallucination rates in grounded generation

### Choosing the Right Tool

| Goal | Tool |
|------|------|
| Benchmark inference speed | GuideLLM |
| Measure model accuracy | lm-eval-harness |
| Evaluate RAG quality | Ragas |
| Production monitoring | Prometheus + Grafana with vLLM metrics |

---

## Key Metrics Summary

Production deployments care about:

| Metric | Definition |
|--------|------------|
| **TTFT** | Time To First Token — how fast the response starts |
| **TPOT** | Time Per Output Token — streaming speed |
| **TPS** | Tokens Per Second — system throughput |
| **Goodput** | Requests/sec meeting SLOs |
| **P95/P99** | Tail latency for SLA compliance |
| **Memory utilization** | How efficiently GPU memory is used |
| **Cost per token** | Infrastructure cost per generated token |

Serving systems optimize these metrics simultaneously.

---

## Pause and Reflect

Here's a mental model:

- **The model** is like a factory's machinery — the actual production capability
- **Serving systems** are like factory logistics — scheduling, resource allocation, throughput optimization

You can have great machinery but terrible logistics (inefficient). Or great logistics with limited machinery (optimized but capped).

Modern LLM deployment needs both: capable models AND efficient serving.

---

## In Practice: Serving LLMs with OpenShift AI

The serving systems you've learned about — vLLM, TGI, OpenVINO — are all available in **Red Hat OpenShift AI** as managed runtimes. The platform handles the infrastructure complexity so you can focus on deploying models.

### The Single-Model Serving Platform

OpenShift AI provides a **single-model serving platform** based on KServe. Each LLM gets its own dedicated model server, enabling:

- Independent scaling per model
- GPU resource isolation
- Model-specific configurations
- REST and gRPC API endpoints

### Key Components

The serving stack has three layers:

| Component | Role |
|-----------|------|
| **ServingRuntime** | Defines the container image and configuration (e.g., vLLM) |
| **InferenceService** | Deploys a specific model using a runtime |
| **KServe** | Orchestrates lifecycle, routing, and scaling |

### Understanding the Three Layers

These three components solve different problems and operate at different levels. Understanding their boundaries prevents confusion.

#### KServe: The Control Plane

KServe is the orchestration layer. Think of it as the manager that ensures models are running, healthy, and accessible.

**What KServe owns:**
- Deploying model server pods
- Autoscaling (adding/removing replicas)
- Health checks and restarts
- Traffic routing between versions
- Exposing external endpoints

**What KServe does NOT do:**
- Load model weights
- Run GPU computations
- Manage the KV cache
- Generate tokens
- Touch the inference math

KServe manages model servers the way Kubernetes manages pods. It doesn't replace the model server — it operates it.

#### Model Server: The Execution Engine

The model server (vLLM, Triton, TGI) is where inference actually happens. This is the component that uses your GPU.

**What the model server owns:**
- Loading model weights into GPU memory
- Running the tokenizer
- Executing transformer forward passes
- Managing the KV cache
- Applying PagedAttention optimizations
- Generating tokens one by one

**The key insight:** When you deploy a model through KServe, KServe launches a model server pod. The model server does all the work described in Chapters 9-11. KServe just makes sure that pod is running and accessible.

#### InferenceService: The Abstraction

An InferenceService is the Kubernetes resource that ties everything together. It specifies:
- Which model to serve
- Which runtime to use (vLLM, OpenVINO, etc.)
- Resource requirements (GPUs, memory)
- Scaling parameters

When you create an InferenceService, KServe reads it and creates the necessary pods, services, and routes.

### InferenceService vs OpenShift Route

A common point of confusion: how is an InferenceService different from an OpenShift Route?

| Aspect | OpenShift Route | InferenceService |
|--------|-----------------|------------------|
| **Layer** | Networking | Application/ML serving |
| **Purpose** | Expose HTTP(S) traffic | Serve models with lifecycle management |
| **ML awareness** | None | Full (knows about models, runtimes, GPUs) |
| **Autoscaling** | No | Yes |
| **GPU awareness** | No | Yes |
| **Health checks** | Basic HTTP | Model-specific readiness |

An InferenceService *uses* Routes (indirectly) to expose its endpoint. But they solve different problems:
- A Route is networking plumbing — it moves HTTP traffic.
- An InferenceService defines and operates the inference workload that traffic reaches.

You could deploy a model using just a Route and a plain Deployment, but you'd lose autoscaling, model lifecycle management, and the serving abstractions that make production deployment practical.

### Where Service Mesh Fits

If you're using Istio or another service mesh, you might wonder: does the mesh replace KServe?

No — they complement each other.

| Technology | Layer | Provides |
|------------|-------|----------|
| **Istio/Envoy** | Network/service mesh | mTLS, authentication, traffic policies, observability |
| **KServe** | Model serving control plane | Model lifecycle, inference abstraction, ML-aware scaling |

Istio can sit in front of KServe, providing security and observability at the network layer. KServe handles the model-serving concerns that a generic service mesh doesn't understand.

### vLLM in OpenShift AI

Remember PagedAttention from earlier? It's available via the **vLLM ServingRuntime for KServe**:

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ServingRuntime
metadata:
  name: vllm-cuda-runtime
  annotations:
    opendatahub.io/apiProtocol: REST
    opendatahub.io/recommended-accelerators: '["nvidia.com/gpu"]'
    openshift.io/display-name: vLLM NVIDIA GPU ServingRuntime for KServe
spec:
  containers:
  - name: kserve-container
    image: registry.redhat.io/rhaiis/vllm-cuda-rhel9:latest
    command: ["python", "-m", "vllm.entrypoints.openai.api_server"]
    args:
    - --port=8080
    - --model=/mnt/models
    - --served-model-name={{.Name}}
    ports:
    - containerPort: 8080
      protocol: TCP
  supportedModelFormats:
  - name: vLLM
    autoSelect: true
```

This ServingRuntime tells OpenShift AI how to run vLLM containers. The image uses the official Red Hat AI Inference Server.

### Deploying a Model

To deploy an LLM, create an **InferenceService** that references your model. OpenShift AI 3.x supports OCI-based model URIs via the Model Catalog:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama-32-3b-instruct
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
    opendatahub.io/hardware-profile-name: gpu-profile
    opendatahub.io/model-type: generative
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 1
    model:
      modelFormat:
        name: vLLM
      runtime: vllm-cuda-runtime
      storageUri: oci://quay.io/redhat-ai-services/modelcar-catalog:llama-3.2-3b-instruct
      args:
      - --dtype=half
      - --max-model-len=20000
      - --gpu-memory-utilization=0.95
      resources:
        requests:
          nvidia.com/gpu: "1"
          memory: 6Gi
        limits:
          nvidia.com/gpu: "1"
          memory: 16Gi
```

This creates:
- A pod running vLLM with your model loaded via OCI registry
- A Kubernetes Service for internal access
- An external Route for API access
- RawDeployment mode (the recommended mode in RHOAI 3.x)

### Available Model-Serving Runtimes

OpenShift AI 3.x includes these pre-installed runtimes:

| Runtime | Best For | Protocol |
|---------|----------|----------|
| **vLLM ServingRuntime** | LLMs with PagedAttention (recommended for LLMs) | REST (OpenAI-compatible) |
| **OpenVINO Model Server** | Intel-optimized inference | REST |
| **Distributed Inference with llm-d** | High-throughput distributed LLM serving | REST |

You can also add custom runtimes for specific model frameworks.

### Model Catalog vs. Model Registry

OpenShift AI 3.x provides two ways to discover and deploy models:

| Source | What It Contains | Use Case |
|--------|------------------|----------|
| **Model Catalog** | Pre-built Red Hat models (Llama, Granite, Mistral, embedding models) | Quick deployment of curated, tested models |
| **Model Registry** | Your organization's trained/fine-tuned models | Deploy models you've built or customized |

**Model Catalog** → Deploy with one click from the dashboard. Models are pulled from OCI registries.

**Model Registry** → Register models after training (Chapter 10), then deploy any version. Supports rollbacks, A/B testing, and audit trails.

Both integrate with the deployment wizard in Gen AI Studio, so the serving experience is consistent regardless of source.

### Calling Your Deployed Model

Once deployed, the InferenceService exposes OpenAI-compatible endpoints:

```bash
# Chat completions endpoint
curl -X POST https://llama-7b-myproject.apps.cluster.example.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-7b",
    "messages": [
      {"role": "user", "content": "Explain transformers in one paragraph."}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

The response streams back just like calling OpenAI's API — but running on your own infrastructure.

### GPU Acceleration Options

OpenShift AI 3.x supports multiple accelerators via **Hardware Profiles**:

| Accelerator | Support Level | Examples |
|-------------|---------------|----------|
| **NVIDIA GPU** | Fully Supported | T4, A10, A100, L4, L40S, H100, H200, B200 |
| **AMD GPU** | Fully Supported | MI210, MI300X |
| **Intel Gaudi** | Fully Supported | Gaudi 2, Gaudi 3 |
| **Google TPU** | Technology Preview | v4, v5e, v6e |
| **IBM Spyre** | Supported | Power, Z |

The platform handles driver installation and device allocation automatically through the respective GPU Operators (NVIDIA GPU Operator, AMD GPU Operator, etc.).

### Autoscaling

With KServe RawDeployment mode (the default in OpenShift AI 3.x), models scale using Kubernetes HPA (Horizontal Pod Autoscaler):

- **Scale up**: Traffic spike? Add replicas automatically based on metrics.
- **Scale down**: Low traffic? Reduce replicas to save resources.
- **Scale limits**: Cap maximum replicas to control spend.

```yaml
spec:
  predictor:
    minReplicas: 1    # Minimum replicas
    maxReplicas: 4    # Maximum replicas
    scaleTarget: 1    # Concurrent requests per replica for scaling
```

Note: Scale-to-zero is available via the Serverless deployment mode, but RawDeployment is recommended for production LLM workloads as it provides more predictable performance.

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    OpenShift AI Platform                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                  Data Science Project                    │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │ │
│  │  │  Workbench  │  │    Data     │  │     Model       │  │ │
│  │  │ (Training)  │──│  Connection │──│    Server       │  │ │
│  │  │             │  │   (S3)      │  │   (vLLM)        │  │ │
│  │  └─────────────┘  └─────────────┘  └────────┬────────┘  │ │
│  └─────────────────────────────────────────────│───────────┘ │
│                                                │              │
│  ┌─────────────────────────────────────────────│───────────┐ │
│  │                    KServe                   │            │ │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───▼─────────┐  │ │
│  │  │ServingRuntime │  │InferenceService│  │   Route    │  │ │
│  │  │    (vLLM)     │──│  (llama-7b)   │──│  (HTTPS)   │  │ │
│  │  └───────────────┘  └───────────────┘  └───────────┘   │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                              ▲
                              │ REST/gRPC
                    ┌─────────┴─────────┐
                    │  User/Application │
                    └───────────────────┘
```

### Distributed Inference with llm-d

For high-throughput scenarios or large models that span multiple GPUs, OpenShift AI 3.x offers **llm-d** (distributed inference). This uses a separate CRD:

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceService
metadata:
  name: qwen3-sample
  labels:
    kueue.x-k8s.io/queue-name: default
    opendatahub.io/genai-asset: "true"
spec:
  replicas: 1
  model:
    uri: oci://registry.redhat.io/rhelai1/modelcar-qwen3-8b-fp8-dynamic:latest
    name: RedHatAI/Qwen3-8B-FP8-dynamic
  router:
    route: {}
    gateway: {}
  scheduler: {}
  template:
    containers:
    - name: main
      resources:
        limits:
          nvidia.com/gpu: "1"
          memory: 32Gi
```

llm-d provides:
- **Intelligent routing** across model replicas
- **Multi-node/multi-GPU** support for Mixture of Experts (MoE) models
- **Integration with Kueue** for workload scheduling
- **Gateway API** for ingress with authentication

### Why This Matters

Without a platform like OpenShift AI, deploying vLLM means:
- Managing Kubernetes YAML manually
- Configuring GPU node taints/tolerations
- Setting up load balancers and TLS
- Building monitoring and logging pipelines
- Handling authentication and authorization

OpenShift AI abstracts this complexity. You get vLLM's PagedAttention benefits with enterprise-grade operations.

### GitOps for AI Infrastructure

Enterprise deployments take this further with **GitOps** — managing the entire AI platform through version-controlled YAML files. This approach provides:

- **Infrastructure Reproducibility**: Dev and prod clusters are identical
- **Audit Trail**: Complete history of all configuration changes
- **Automated Rollbacks**: Revert to previous configurations instantly
- **Declarative Model Serving**: Models deployed as code

A GitOps-managed AI platform might organize model deployments like this:

```
components/configs/model-serving/
├── granite-3.3-8b-instruct.yaml    # Granite LLM deployment
├── llama-3.2-3b.yaml               # Llama deployment
├── mistral-small-24b.yaml          # Mistral deployment
├── nomic-embed-text-v1-5.yaml      # Embedding model
└── serving-runtimes/
    ├── vllm-runtime.yaml           # vLLM ServingRuntime
    └── openvino-runtime.yaml       # OpenVINO ServingRuntime
```

Each model is a declarative InferenceService:

```yaml
# granite-3.3-8b-instruct.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: granite-3-3-8b-instruct
  annotations:
    serving.kserve.io/autoscalerClass: hpa
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 4
    model:
      modelFormat:
        name: vLLM
      runtime: vllm-runtime
      storageUri: s3://models/granite-3.3-8b-instruct/
      resources:
        limits:
          nvidia.com/gpu: "1"
          memory: 48Gi
```

Git becomes the single source of truth. To deploy a new model:
1. Add a YAML file to the repository
2. Create a pull request for review
3. Merge to main — ArgoCD automatically deploys
4. Model is live with full audit trail

This pattern scales to dozens of models across multiple environments, with GPU autoscaling policies and API gateway configurations all managed declaratively.

---

## Chapter Takeaway

> **Serving systems optimize LLM deployment** through KV cache management, batching, and memory efficiency. vLLM's PagedAttention treats the KV cache like virtual memory, eliminating fragmentation. These systems don't change the model — they make running it practical at scale. **OpenShift AI provides vLLM and other runtimes as managed ServingRuntimes**, with KServe orchestrating deployment, scaling, and API exposure.

---

## Part VI Summary

You've seen the engineering that makes LLMs practical:

1. **KV cache** eliminates redundant computation by storing K/V vectors
2. **Serving systems** optimize deployment through batching and memory management

These are systems concerns, separate from the ML itself. But essential for real-world use.

Now we bring it all together.

---

*Next: [Chapter 13: The Complete Vocabulary](13-vocabulary.md)*
