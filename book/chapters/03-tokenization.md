# Chapter 3: Breaking Language into Pieces

*Part II: The Translation Problem*

---

Here's a question that seems simple but isn't:

*What's the basic unit of language?*

Is it words? Characters? Sentences?

For LLMs, the answer is none of the above. It's **tokens** — and understanding tokens is essential.

---

## Why Not Just Use Words?

Your first instinct might be: "Just give each word a number."

```
"The" = 1  
"cat" = 2  
"sat" = 3  
...and so on.
```

This approach has problems.

### Problem 1: Too Many Words

English has hundreds of thousands of words. Add technical jargon, names, misspellings, and new slang, and you're looking at millions of possibilities. That's too many for a model to handle efficiently.

### Problem 2: New Words Break Everything

What happens when someone types "ChatGPT" or "cryptocurrency" or "unfriend"? If the word isn't in your dictionary, you're stuck. The model literally cannot process it.

### Problem 3: Wasted Similarity

"Running," "runner," and "runs" are clearly related. But if they're three separate entries with unrelated numbers, the model has to learn their relationship from scratch. That's inefficient.

---

## Why Not Just Use Characters?

Okay, what about individual letters?

```
"T" = 1, "h" = 2, "e" = 3...
```

### Problem: Too Granular

Now every word is a long sequence. "Understanding" becomes 13 separate items. The model has to work much harder to learn that these letters form a coherent concept.

Characters also lose meaning. "T" by itself tells you almost nothing.

---

## The Goldilocks Solution: Subword Tokens

The answer is a clever middle ground: **subword tokenization**.

Instead of words or characters, we break text into *pieces* that balance frequency and meaning:

- Common words stay whole: "the," "is," "and"
- Rare words get split: "tokenization" → "token" + "ization"
- Very rare words split further: "Nguyen" → "Ng" + "uy" + "en"

This approach solves all three problems:

1. **Limited vocabulary**: GPT-2 uses exactly **50,257 tokens** — enough to cover virtually everything
2. **No unknown words**: Any text can be broken into known pieces
3. **Shared meaning**: "Running" and "runner" share the "run" piece

---

## The LEGO Analogy

Think of tokenization like LEGO bricks.

Instead of carving each toy from scratch (whole words), you build everything from a set of reusable bricks (tokens). 

"Playing" uses the same "-ing" brick as "running," "thinking," and "jumping." The model learns what "-ing" means once and applies it everywhere.

---

## Watching Tokenization Happen

Let's see it in action with our running example:

```
Input: "The Eiffel Tower is located in"

Tokens: ["The", " Eiff", "el", " Tower", " is", " located", " in"]

Token IDs: [464, 36751, 417, 8765, 318, 5140, 287]
```

Notice what happened:

- "The" is common — stays whole
- "Eiffel" splits into "Eiff" + "el" — it's a proper noun, less common
- "Tower", "is", "located", "in" — all common enough to be single tokens
- Spaces are attached to the following word (" Eiff", " Tower")

Here's another example:

```
Input: "ChatGPT is amazing!"

Tokens: ["Chat", "G", "PT", " is", " amazing", "!"]

Token IDs: [16047, 38, 2898, 318, 4998, 0]
```

"ChatGPT" splits into three pieces because it's a newer word that wasn't common when the vocabulary was built.

Each token then gets an **ID** — just an integer that identifies it.

---

## The Vocabulary: A Fixed Dictionary

Before training begins, the vocabulary is decided. It's a fixed list, something like:

| Token ID | Token |
|----------|-------|
| 0 | "!" |
| 1 | "\"" |
| ... | ... |
| 318 | " is" |
| ... | ... |
| 16047 | "Chat" |
| ... | ... |
| 50256 | `<|endoftext|>` |

The final token, `<|endoftext|>`, is a special marker used to separate unrelated documents during training. When the model sees this token, it knows one piece of text has ended and another begins.

This vocabulary never changes. The tokenizer (the tool that splits text) uses these same rules forever.

Key properties:

- **Created before training** — the vocabulary is built once and never changes
- **Model-specific** — GPT-2 has 50,257 tokens; GPT-4 and LLaMA have different vocabularies
- **Size varies** — typically 32,000 to 100,000+ tokens depending on the model
- **Deterministic** — the same text always produces the same tokens

---

## How Tokenization Actually Works: BPE

The most common algorithm is **Byte Pair Encoding (BPE)**. Here's the intuition:

1. Start with all individual characters as tokens
2. Scan a large text corpus and find the most common pair of adjacent tokens
3. Merge that pair into a new token
4. Repeat until you reach the desired vocabulary size (e.g., 50,257 for GPT-2)

Example progression:

```
Start:    ["l", "o", "w", "e", "r"]
Step 1:   ["lo", "w", "e", "r"]     (merged "l" + "o" — common pair)
Step 2:   ["low", "e", "r"]         (merged "lo" + "w")
Step 3:   ["lowe", "r"]             (merged "low" + "e")
Step 4:   ["lower"]                 (merged "lowe" + "r")
```

The key insight: **merges are determined by frequency in the training corpus**. Common words like "the" become single tokens early. Rare words like "cryptocurrency" stay as multiple pieces because their subparts weren't merged.

This is why BPE handles unknown words gracefully — any word can be built from the character-level tokens that always exist in the vocabulary.

---

**The Revelation:**

> Any word — even one never seen before — can be built from pieces.

This is profound. The tokenizer will never fail. "Cryptocurrency," "ChatGPT," a typo like "teh," a name like "Nguyen" — everything breaks down into known pieces. The model can process *any* text, in *any* language, including text that didn't exist when it was trained.

---

## The Critical Distinction (Remember This)

**Token IDs are just labels. They have no meaning.**

Token ID 16047 means nothing mathematically. It's like a coat check number — it tells you which coat, but says nothing about the coat itself.

You can't compare token IDs:

- Is token 16047 similar to token 16048? No idea. The numbers are arbitrary.
- Is token 318 bigger than token 100? Meaningless question.

Token IDs are addresses, not meanings. The meaning comes next.

---

## What "Encoding" Really Means

You'll often hear tokenization described as "encoding" text. This terminology can be confusing, so let's be precise:

**Encoding means:** mapping text to pre-existing token IDs using the tokenizer's rules.

**Encoding does NOT mean:**
- Creating new token IDs
- Modifying the vocabulary
- Any kind of learning or intelligence

The tokenizer performs a deterministic lookup. Given the same text, it always produces the same token IDs. The vocabulary and rules are fixed before training and never change.

### What Happens When a Word Isn't in the Vocabulary?

The tokenizer never fails. If a whole word doesn't exist as a token, the tokenizer splits it into smaller pieces that do exist:

```
"cryptocurrency"
→ ["crypt", "oc", "urrency"]  (example split)
→ [token IDs exist for each piece]
```

It keeps splitting until every piece matches something in the vocabulary. Worst case, it falls back to individual characters or byte-level tokens.

This is why:
- LLMs can handle any text, including typos, new words, and multiple languages
- Rare or complex words cost more tokens (affecting cost and context usage)
- The model never encounters an "unknown word" error

### Where Tokenization Runs

Tokenization happens on the **CPU**, not the GPU.

```
CPU: Text → Tokenizer → Token IDs
         ↓
GPU: Token IDs → Embedding lookup → Model computation
```

The tokenizer is string processing and dictionary lookup — operations CPUs handle efficiently. The GPU takes over once we have token IDs and need to do the heavy matrix math of the transformer.

In practice, the **model server** (like vLLM) handles this automatically. It runs the tokenizer, sends token IDs to the GPU, and converts output token IDs back to text.

---

## Pause and Reflect

Try to tokenize this sentence in your head:

*"I love transformers!"*

Would "transformers" be one token or multiple? What about "love"?

For most models: "I" = 1 token, " love" = 1 token, " transformers" = 1 token, "!" = 1 token. Common words stay whole.

---

## In Practice: Tokenization in OpenShift AI

When deploying LLMs in **Red Hat OpenShift AI**, tokenization happens automatically inside the model server. But understanding tokenization helps you configure models correctly.

### Chat Templates

Different models expect conversations in different formats. A **chat template** defines how messages are tokenized:

```
User: Hello
Assistant: Hi there!
```

Gets transformed into something like:

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Hi there!<|eot_id|>
```

In vLLM deployments, you can specify custom chat templates:

```yaml
args:
- --chat-template=/opt/app-root/template/tool_chat_template_llama3.2_json.jinja
- --enable-auto-tool-choice
- --tool-call-parser=llama3_json
```

This is crucial for **tool calling** — the template tells the model how to format function calls and parse responses.

### Why This Matters

Incorrect tokenization causes subtle failures:
- Wrong chat template → model produces garbage or ignores instructions
- Mismatched tokenizer → model hallucinates or stops mid-sentence
- Missing special tokens → model doesn't know when to stop

When using the Model Catalog in OpenShift AI, these configurations are pre-set correctly for each model. But if you're deploying custom models, understanding tokenization helps you debug issues.

### Token Limits and Context

The `--max-model-len` parameter in vLLM sets the maximum context length in tokens:

```yaml
args:
- --max-model-len=20000
```

This directly relates to Chapter 3's concepts — the model can process up to 20,000 tokens of input + output combined. Longer conversations require truncation or a model with larger context support.

---

## Chapter Takeaway

> **Tokenization breaks text into learnable pieces** — not too big, not too small. Each piece gets a Token ID, which is just an arbitrary integer label. GPT-2's vocabulary contains exactly 50,257 tokens, including special markers like `<|endoftext|>` for document boundaries. The tokenizer and vocabulary are fixed before training and never change. Token IDs have no meaning on their own — they're just addresses. **In OpenShift AI, chat templates control how conversations are tokenized for each model.**

---

We've turned "The Eiffel Tower is located in" into `[464, 36751, 417, 8765, 318, 5140, 287]`. But these are just arbitrary labels — coat check numbers. How do they become *meaning*?

That's where embeddings come in.

---

*Next: [Chapter 4: Giving Numbers Meaning](04-embeddings.md)*
