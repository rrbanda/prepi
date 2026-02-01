---
hide:
  - navigation
  - toc
---

# The LLM Handbook

<p class="hero-subtitle">A Complete Guide to Understanding Large Language Models — From Concepts to Production</p>

---

## About This Book

This handbook takes you from zero knowledge to complete understanding of how Large Language Models work. Whether you're building AI products, deploying models to production, or simply curious about the technology behind ChatGPT, this guide will give you the mental models you need.

**No prior machine learning knowledge required. Just curiosity.**

[:material-book-open-variant: Start Reading](chapters/00-prologue.md){ .md-button .md-button--primary }
[:material-format-list-bulleted: View All Chapters](#table-of-contents){ .md-button }

---

## Who This Book Is For

| Reader | What You'll Get |
|--------|-----------------|
| **Software Engineers** | Deep understanding of LLMs to use them more effectively |
| **MLOps Practitioners** | Production deployment knowledge with vLLM, KServe, OpenShift AI |
| **Product Managers** | Mental models to evaluate AI capabilities and limitations |
| **Technical Leaders** | Framework for making informed AI investment decisions |
| **Curious Minds** | Clear, demystified explanations of how the magic works |

---

## The Journey

We'll trace a single prompt through the entire system:

> **"The Eiffel Tower is located in"** → **"Paris"**

By the final chapter, you'll know exactly what happens to these words — how they become numbers, how those numbers interact, and how "Paris" emerges as the response.

---

## Table of Contents

### Prologue
- [**The Promise**](chapters/00-prologue.md) — What this book will teach you

### Part I: The Magic Trick
*Understanding what LLMs do and the fundamental challenge they solve*

- [**Chapter 1: What Just Happened?**](chapters/01-what-just-happened.md)
- [**Chapter 2: The Language Barrier**](chapters/02-the-language-barrier.md)

### Part II: The Translation Problem
*How language becomes numbers that preserve meaning*

- [**Chapter 3: Breaking Language into Pieces**](chapters/03-tokenization.md)
- [**Chapter 4: Giving Numbers Meaning**](chapters/04-embeddings.md)

### Part III: The Brain's Blueprint
*Neural networks and what makes transformers special*

- [**Chapter 5: How Machines Learn Anything**](chapters/05-neural-networks.md)
- [**Chapter 6: The Transformer Revolution**](chapters/06-transformers.md)

### Part IV: The Secret Sauce
*Self-attention — the core mechanism that enables understanding*

- [**Chapter 7: Attention — The Core Mechanism**](chapters/07-attention.md)
- [**Chapter 8: Context — The Most Misunderstood Word in AI**](chapters/08-context.md)

### Part V: Watching It Think
*Step-by-step walkthrough of inference and training*

- [**Chapter 9: Inference — A Token's Journey**](chapters/09-inference.md)
- [**Chapter 10: Training — How Intelligence Emerges**](chapters/10-training.md)

### Part VI: Under the Hood
*Optimization and systems that make LLMs practical*

- [**Chapter 11: The KV Cache**](chapters/11-kv-cache.md)
- [**Chapter 12: Serving Systems — vLLM and Friends**](chapters/12-serving-systems.md)
- [**Chapter 13: The OpenShift AI Platform**](chapters/13-openshift-ai-platform.md)

### Part VII: Mastery
*Synthesis and deep insights*

- [**Chapter 14: The Complete Vocabulary**](chapters/14-vocabulary.md)
- [**Chapter 15: The Deep Insight — Why It Works**](chapters/15-why-it-works.md)

### Appendix
- [**Where to Go Next**](chapters/16-appendix.md)

---

## How to Use This Book

### :material-book-arrow-right: Read Sequentially (Recommended)
Each chapter builds on the previous ones. Concepts introduced early become building blocks for later explanations.

**Estimated reading time**: 4-6 hours for the complete book.

### :material-run-fast: The Fast Path

| If You Already Understand... | Start At |
|------------------------------|----------|
| What LLMs do (prediction, not retrieval) | [Chapter 3: Tokenization](chapters/03-tokenization.md) |
| Tokenization and embeddings | [Chapter 5: Neural Networks](chapters/05-neural-networks.md) |
| Neural network basics | [Chapter 7: Attention](chapters/07-attention.md) |
| Transformer architecture | [Chapter 9: Inference](chapters/09-inference.md) |
| LLM fundamentals, want production | [Chapter 12: Serving Systems](chapters/12-serving-systems.md) |

### :material-magnify: Reference Mode

- **[Chapter 14 (Vocabulary)](chapters/14-vocabulary.md)** — Quick lookup for any term
- **Chapter Takeaways** — Boxed summaries at the end of each chapter
- **"What You Can Now Explain"** — Checkpoints after each Part

---

## Conventions Used

| Section | Purpose |
|---------|---------|
| **"You might be wondering..."** | Anticipates and answers common questions |
| **"The Revelation"** | Key insights worth remembering |
| **"Chapter Takeaway"** | The key insight from each chapter |
| **"Try It Yourself"** | Hands-on exercises with code |
| **"In Practice with OpenShift AI"** | Production deployment examples |

---

<div style="text-align: center; margin-top: 3rem;">

[:material-book-open-variant: Begin with the Prologue](chapters/00-prologue.md){ .md-button .md-button--primary .md-button--lg }

</div>
