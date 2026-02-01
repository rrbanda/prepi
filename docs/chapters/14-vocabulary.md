# Chapter 14: The Complete Vocabulary

*Part VII: Mastery*

---

You've covered a lot of ground. This chapter consolidates every term into a single reference.

Use this as a quick lookup when reviewing concepts or discussing LLMs with colleagues.

---

## Core Concepts

| Term | Plain English Definition | Covered In |
|------|-------------------------|------------|
| **LLM** | Large Language Model — a model that predicts the next token | Chapter 1 |
| **Token** | A piece of text (word, subword, or character) | Chapter 3 |
| **Token ID** | An integer label for a token (no meaning) | Chapter 3 |
| **Vocabulary** | The fixed set of all possible tokens | Chapter 3 |
| **Tokenizer** | The tool that breaks text into token IDs | Chapter 3 |
| **BPE** | Byte Pair Encoding — common tokenization algorithm | Chapter 3 |

---

## Representations

| Term | Plain English Definition | Covered In |
|------|-------------------------|------------|
| **Vector** | A list of numbers; a point in space | Chapter 4 |
| **Embedding** | A meaningful vector for a token | Chapter 4 |
| **Embedding Matrix** | Lookup table: Token ID → Embedding | Chapter 4 |
| **Tensor** | A container for numbers (1D=vector, 2D=matrix, nD=tensor) | Chapter 4 |
| **Contextual Representation** | A vector that includes information from context | Chapter 8 |

---

## Model Structure

| Term | Plain English Definition | Covered In |
|------|-------------------------|------------|
| **Neural Network** | A function that transforms inputs through layers | Chapter 5 |
| **Layer** | A step of transformation (input → output) | Chapter 5 |
| **Weights** | Learned numbers that control transformations | Chapter 5 |
| **Architecture** | The blueprint of the model (layers, sizes, connections) | Chapter 5 |
| **Layer Normalization** | Normalizes activations to mean=0, variance=1 for stable training | Chapter 5 |
| **Pre-LayerNorm** | GPT's approach: normalize *before* attention/FFN, not after | Chapter 5 |
| **GELU** | Gaussian Error Linear Unit — GPT's smooth activation function | Chapter 5 |
| **ReLU** | Rectified Linear Unit — simpler activation (max(0, x)) | Chapter 5 |
| **Residual Connection** | Adding input to output (x + layer(x)) to help gradient flow | Chapter 5 |
| **Transformer** | The architecture that uses self-attention | Chapter 6 |
| **Transformer Block** | One round of attention + FFN with residuals and normalization | Chapter 6 |
| **Feed-Forward Network (FFN)** | Two-layer MLP in each block (expands 4×, then contracts) | Chapter 6 |
| **Decoder-only** | Transformer that generates left-to-right | Chapter 6 |
| **Mixture of Experts (MoE)** | Architecture with multiple FFNs where only some activate per token | Chapter 6 |
| **Router** | Small network that selects which experts to use in MoE | Chapter 6 |
| **Expert** | One of several parallel FFN variants in MoE architecture | Chapter 6 |
| **Sparse Activation** | Only a subset of parameters used per token (MoE) | Chapter 6 |

---

## Attention Mechanism

| Term | Plain English Definition | Covered In |
|------|-------------------------|------------|
| **Self-Attention** | Mechanism for tokens to "see" each other | Chapter 7 |
| **Query (Q)** | "What am I looking for?" | Chapter 7 |
| **Key (K)** | "What do I offer/contain?" | Chapter 7 |
| **Value (V)** | "Here's my actual information" | Chapter 7 |
| **Attention Score** | How relevant one token is to another | Chapter 7 |
| **Attention Weights** | Probabilities derived from attention scores | Chapter 7 |
| **Multi-Head Attention** | Multiple parallel attention computations | Chapter 7 |
| **Causal Masking** | Preventing tokens from seeing future tokens | Chapter 7 |

---

## Context

| Term | Plain English Definition | Covered In |
|------|-------------------------|------------|
| **Context Window** | Maximum number of tokens the model can see | Chapter 8 |
| **Contextual Representation** | A token's vector after attention (context-aware) | Chapter 8 |
| **Position Encoding** | Information added to embeddings about token position | Chapter 8 |
| **RoPE** | Rotary Position Embeddings — a position encoding method | Chapter 8 |

---

## Inference

| Term | Plain English Definition | Covered In |
|------|-------------------------|------------|
| **Inference** | Running the model to generate output | Chapter 9 |
| **Forward Pass** | Data flowing through the model from input to output | Chapter 9 |
| **Logits** | Raw scores for each vocabulary token before softmax | Chapter 9 |
| **Softmax** | Function that converts logits to probabilities | Chapter 9 |
| **Weight Tying** | Sharing embedding matrix with output projection (saves params) | Chapter 9 |
| **Sampling** | Choosing the next token based on probabilities | Chapter 9 |
| **Autoregressive** | Generating one token at a time, each based on previous | Chapter 9 |
| **Temperature** | Parameter controlling randomness in sampling | Chapter 9 |
| **Top-k Sampling** | Sample only from the k most likely tokens | Chapter 9 |
| **Top-p / Nucleus** | Sample only from tokens comprising top p% probability | Chapter 9 |
| **Beam Search** | Keep multiple candidate sequences, pick best at end | Chapter 9 |
| **Greedy Decoding** | Always choosing the highest probability token | Chapter 9 |

---

## Training

| Term | Plain English Definition | Covered In |
|------|-------------------------|------------|
| **Training** | The process of learning weights from data | Chapter 10 |
| **Self-Supervised Learning** | Training where labels come from the data itself (next word = label) | Chapter 10 |
| **Pretraining** | Initial training on large unlabeled text datasets | Chapter 10 |
| **Loss** | A number measuring how wrong the prediction was | Chapter 10 |
| **Cross-Entropy Loss** | Common loss function: -log(probability of correct token) | Chapter 10 |
| **Perplexity** | exp(loss) — effective vocabulary size the model is uncertain about | Chapter 10 |
| **Backpropagation** | Computing how each weight affects the loss | Chapter 10 |
| **Gradient** | Direction to adjust a weight to reduce loss | Chapter 10 |
| **Gradient Clipping** | Capping gradient magnitude to prevent exploding updates | Chapter 10 |
| **Learning Rate** | How big the weight adjustments are | Chapter 10 |
| **AdamW** | Standard optimizer for LLMs (Adam with weight decay) | Chapter 10 |
| **Learning Rate Schedule** | Warmup then decay of learning rate during training | Chapter 10 |
| **Training Data** | The text used to train the model | Chapter 10 |

---

## Fine-Tuning and Alignment

| Term | Plain English Definition | Covered In |
|------|-------------------------|------------|
| **Fine-Tuning** | Adapting a pre-trained model to a specific task or domain | Chapter 10 |
| **Full Fine-Tuning** | Updating all model weights during fine-tuning | Chapter 10 |
| **LoRA** | Low-Rank Adaptation — fine-tuning with small adapter matrices | Chapter 10 |
| **QLoRA** | LoRA applied to a quantized model for even lower memory | Chapter 10 |
| **PEFT** | Parameter-Efficient Fine-Tuning — umbrella term for LoRA-like methods | Chapter 10 |
| **SFT** | Supervised Fine-Tuning — training on (prompt, response) pairs | Chapter 10 |
| **RLHF** | Reinforcement Learning from Human Feedback — learning from human preferences | Chapter 10 |
| **DPO** | Direct Preference Optimization — simpler alternative to RLHF | Chapter 10 |
| **Alignment** | Training models to be helpful, harmless, and honest | Chapter 10 |
| **Reward Model** | Model trained to predict human preferences (used in RLHF) | Chapter 10 |

---

## Prompting and Reasoning

| Term | Plain English Definition | Covered In |
|------|-------------------------|------------|
| **Chain-of-Thought (CoT)** | Prompting the model to show reasoning steps | Chapter 15 |
| **Few-Shot Prompting** | Including examples in the prompt to guide behavior | Appendix |
| **Zero-Shot** | Asking the model to perform a task without examples | Appendix |
| **System Prompt** | Instructions that set the model's behavior/persona | Appendix |
| **RAG** | Retrieval-Augmented Generation — adding retrieved documents to prompts | Chapter 15 |
| **Tool Use** | Model generates function calls to external tools | Appendix |

---

## Concepts and Philosophy

| Term | Plain English Definition | Covered In |
|------|-------------------------|------------|
| **Emergent Behavior** | Capabilities that appear at scale without explicit training | Chapter 15 |
| **Scaling Laws** | Relationships between model size, data, compute, and performance | Chapter 15 |
| **Hallucination** | Model generating plausible but false information | Chapter 15 |
| **Foundation Model** | A large pretrained model adapted for various downstream tasks | Chapter 10 |
| **Base Model** | A pretrained model before fine-tuning (same as foundation model) | Chapter 10 |

---

## Optimization

| Term | Plain English Definition | Covered In |
|------|-------------------------|------------|
| **KV Cache** | Stored Key/Value vectors to avoid recomputation | Chapter 11 |
| **Attention State** | Synonym for KV cache — emphasizes it's the state needed for attention | Chapter 11 |
| **BlockSpaceManager** | vLLM's memory manager that implements PagedAttention | Chapter 11 |
| **Batching** | Processing multiple requests together | Chapter 12 |
| **Continuous Batching** | Dynamically adding/removing requests from batch | Chapter 12 |
| **PagedAttention** | vLLM's method for non-contiguous KV cache storage | Chapter 12 |
| **Scheduler** | Central coordinator managing request queues and batch execution | Chapter 12 |
| **Preemption** | Pausing or evicting requests to free memory | Chapter 12 |
| **Throughput** | Tokens generated per second (system-wide) | Chapter 12 |
| **Latency** | Time to generate response | Chapter 12 |
| **TTFT** | Time To First Token — latency until first output token | Chapter 12 |
| **TPOT** | Time Per Output Token — average time between tokens | Chapter 12 |
| **ITL** | Inter-Token Latency — time between consecutive tokens | Chapter 12 |
| **Goodput** | Requests/second meeting SLO constraints | Chapter 12 |
| **P95/P99** | 95th/99th percentile latency | Chapter 12 |

---

## Speculative Decoding

| Term | Plain English Definition | Covered In |
|------|-------------------------|------------|
| **Speculative Decoding** | Optimization using a draft model to predict ahead | Chapter 12 |
| **Draft Model** | Small, fast model that generates speculative tokens | Chapter 12 |
| **Target Model** | Large, accurate model that verifies speculative tokens | Chapter 12 |
| **Acceptance Rate** | Percentage of speculated tokens accepted by target | Chapter 12 |

---

## Quantization

| Term | Plain English Definition | Covered In |
|------|-------------------------|------------|
| **Quantization** | Reducing weight precision to save memory and speed | Chapter 12 |
| **W4A16** | 4-bit weights, 16-bit activations | Chapter 12 |
| **W8A8-INT8** | 8-bit weights and activations (integer) | Chapter 12 |
| **W8A8-FP8** | 8-bit weights and activations (floating point) | Chapter 12 |
| **llm-compressor** | Tool for quantizing models for vLLM | Chapter 12 |
| **GPTQ** | Post-training quantization algorithm | Chapter 12 |
| **AWQ** | Activation-aware Weight Quantization | Chapter 12 |

---

## Evaluation

| Term | Plain English Definition | Covered In |
|------|-------------------------|------------|
| **GuideLLM** | Benchmarking tool for LLM inference performance | Chapter 12 |
| **lm-eval-harness** | Framework for evaluating model accuracy on benchmarks | Chapter 12 |
| **Ragas** | Evaluation framework for RAG applications | Chapter 12 |
| **MMLU** | Multi-task benchmark for language understanding | Chapter 12 |
| **Benchmark Sweep** | Testing across multiple load levels | Chapter 12 |

---

## Serving Systems

| Term | What It Is |
|------|------------|
| **vLLM** | Open-source serving system with PagedAttention |
| **TensorRT-LLM** | NVIDIA's optimized serving library (NVIDIA-only) |
| **NIM** | NVIDIA Inference Microservices — containerized model serving |
| **SGLang** | Fast serving framework with RadixAttention, agentic focus |
| **TGI** | HuggingFace's Text Generation Inference |
| **Triton** | NVIDIA's inference server |
| **KServe** | Control plane for model serving — orchestrates lifecycle, scaling, routing |
| **InferenceService** | Kubernetes resource that defines a deployed model |
| **ServingRuntime** | Template defining how to run a model server (container, config) |
| **Model Server** | The execution engine (vLLM, Triton) that runs inference on GPU |
| **llm-d** | Distributed inference system for high-throughput LLM serving |

---

## OpenShift AI Platform

| Term | What It Is |
|------|------------|
| **OpenShift AI Operator** | Meta-operator that deploys and manages all OpenShift AI components |
| **Model Catalog** | Curated library of GenAI models from Red Hat, IBM, Meta, NVIDIA |
| **Workbench** | Isolated development environment (Jupyter, code-server, RStudio) |
| **AI Pipelines** | ML workflow automation based on Kubeflow Pipelines 2.0 |
| **Kueue** | Kubernetes-native job queuing and quota management |
| **Ray/KubeRay** | Distributed compute orchestration for training |
| **Training Hub** | Framework for fine-tuning foundation models |
| **Docling** | Python library for converting unstructured documents to structured formats |
| **Model Registry** | Central repository for versioning and tracking model lifecycle |
| **TrustyAI** | Responsible AI toolkit for bias detection and drift monitoring |
| **Feature Store** | Centralized storage for ML features (based on Feast) |
| **Serverless Mode** | KServe deployment with scale-to-zero via Knative |
| **RawDeployment Mode** | KServe deployment using standard Kubernetes resources |
| **Llama Stack** | Unified AI runtime for GenAI workloads (RAG, agents) |
| **Ragas** | Evaluation framework for RAG pipelines |

---

## Quick Reference Card

Memorize these key relationships:

```
Text
  ↓ Tokenizer
Token IDs (integers, no meaning)
  ↓ Embedding Matrix
Embeddings (vectors, meaning)
  ↓ + Position Encodings
Input to Transformer
  ↓ Self-Attention (Q × K → weights → blend V)
  ↓ Feed-Forward Network
  ↓ × N layers
Contextual Vectors
  ↓ Output Projection
Logits (scores for all vocab)
  ↓ Softmax
Probabilities
  ↓ Sampling
Next Token
  ↓ Repeat
Full Response
```

---

## Chapter Takeaway

> This vocabulary reference covers all major terms from the book. Use it for quick lookups during review. Understanding the relationships between terms is as important as knowing definitions.

---

*Next: [Chapter 15: The Deep Insight — Why It Works](15-why-it-works.md)*
