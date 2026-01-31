# The LLM Handbook

**A Complete Guide to Understanding Large Language Models — From Concepts to Production**

---

## About This Book

This handbook takes you from zero knowledge to complete understanding of how Large Language Models work. Whether you're building AI products, deploying models to production, or simply curious about the technology behind ChatGPT, this guide will give you the mental models you need.

**Practical OpenShift AI Content** — Key chapters include "In Practice" sections showing how LLM concepts apply in **Red Hat OpenShift AI**, with hands-on examples for training workbenches, model serving with vLLM/KServe, and production deployment using GitOps.

No prior machine learning knowledge required. Just curiosity.

---

## Who This Book Is For

- **Engineers** who use LLMs but want to understand them deeply
- **MLOps practitioners** deploying models to production
- **Product managers** working with AI teams
- **Curious minds** who want to know how the magic works

---

## Table of Contents

### Prologue
- [Prologue: The Promise](chapters/00-prologue.md)

### Part I: The Magic Trick
*Understanding what LLMs do and the fundamental challenge they solve*

- [Chapter 1: What Just Happened?](chapters/01-what-just-happened.md)
- [Chapter 2: The Language Barrier](chapters/02-the-language-barrier.md)

### Part II: The Translation Problem  
*How language becomes numbers that preserve meaning*

- [Chapter 3: Breaking Language into Pieces](chapters/03-tokenization.md)
- [Chapter 4: Giving Numbers Meaning](chapters/04-embeddings.md)

### Part III: The Brain's Blueprint
*Neural networks and what makes transformers special*

- [Chapter 5: How Machines Learn Anything](chapters/05-neural-networks.md)
- [Chapter 6: The Transformer Revolution](chapters/06-transformers.md)

### Part IV: The Secret Sauce
*Self-attention — the core mechanism that enables understanding*

- [Chapter 7: Attention — The Core Mechanism](chapters/07-attention.md)
- [Chapter 8: Context — The Most Misunderstood Word in AI](chapters/08-context.md)

### Part V: Watching It Think
*Step-by-step walkthrough of inference and training*

- [Chapter 9: Inference — A Token's Journey](chapters/09-inference.md)
- [Chapter 10: Training — How Intelligence Emerges](chapters/10-training.md)

### Part VI: Under the Hood
*Optimization and systems that make LLMs practical*

- [Chapter 11: The KV Cache — Why Inference Doesn't Crawl](chapters/11-kv-cache.md)
- [Chapter 12: Serving Systems — vLLM and Friends](chapters/12-serving-systems.md)
- [Chapter 13: The OpenShift AI Platform](chapters/13-openshift-ai-platform.md)

### Part VII: Mastery
*Synthesis and deep insights*

- [Chapter 14: The Complete Vocabulary](chapters/14-vocabulary.md)
- [Chapter 15: The Deep Insight — Why It Works](chapters/15-why-it-works.md)

### Appendix
- [Appendix: Where to Go Next](chapters/16-appendix.md)

---

## How to Read This Book

This book is designed to be read sequentially. Each chapter builds on the previous ones. However, if you're already familiar with certain concepts, here's a guide:

| If you know... | Skip to... |
|----------------|------------|
| What LLMs do | Chapter 3 (Tokenization) |
| Tokenization & embeddings | Chapter 5 (Neural Networks) |
| Neural network basics | Chapter 7 (Attention) |
| How transformers work | Chapter 9 (Inference) |
| LLM fundamentals, want production | Chapter 12 (Serving Systems) |

---

## Conventions Used

Throughout this book, you'll see:

- **The Running Example** — We trace a single prompt through every concept: *"The Eiffel Tower is located in"* → *"Paris"*. By the end, you'll understand exactly what happens to these words at every step.
- **Pause and Reflect** — Moments to think before continuing
- **Chapter Takeaway** — The key insight from each chapter, boxed for easy review
- **What You Can Now Explain** — Checkpoints after each Part showing your growing capabilities
- **In Practice with OpenShift AI** — How concepts apply in production using Red Hat OpenShift AI
- **Interactive Moments** — Questions to test your understanding
- Code blocks with concrete examples
- YAML examples for Kubernetes/OpenShift deployments
- Analogies to ground abstract concepts

---

*Let's begin.*
