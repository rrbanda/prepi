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

## Key Metrics

Production deployments care about:

| Metric | Definition |
|--------|------------|
| **Throughput** | Tokens generated per second (across all requests) |
| **Latency (TTFT)** | Time To First Token — how fast the response starts |
| **Latency (TPS)** | Tokens Per Second per request — how fast it streams |
| **Memory utilization** | How efficiently GPU memory is used |
| **Cost per token** | How much each generated token costs |

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
