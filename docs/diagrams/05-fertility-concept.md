# 🌱 Fertility: Why Fewer Tokens Per Word Wins

> **Fertility** = average tokens per whitespace word.  
> Lower is better. A tokenizer that splits `acetylcholinesterase` into 2 pieces knows medicine.
> One that splits it into 8 is wasting context window and gradient signal.

This page explains the main metric used in the lab.

If you want the simplest version, fertility answers one question:

**How many token pieces does the tokenizer need, on average, to represent the same text?**

Lower fertility means denser tokenization. Higher fertility means more fragmentation.

## What happens to one clinical term

Start with the single-word example below.

It shows the same medical term going through two tokenizers. One tokenizer breaks it into many small pieces. The other keeps it in fewer, larger pieces.

That difference is exactly what fertility is trying to measure at scale.

```mermaid
flowchart LR
    WORD["🔬 Input word\n'acetylcholinesterase'\n(Alzheimer's enzyme)"]

    subgraph cl100k_path ["🤖 cl100k  (GPT-4, 100k vocab)"]
        C1["ac"] --> C2["et"] --> C3["yl"] --> C4["cho"] --> C5["lin"] --> C6["ester"] --> C7["ase"]
        CSCORE["📊 7 tokens\nFertility = 7.0 for this word"]
    end

    subgraph custom_path ["🏥 custom-med  (16k, PubMed-trained)"]
        M1["acetylcholin"] --> M2["ester"] --> M3["ase"]
        MSCORE["📊 3 tokens\nFertility = 3.0 for this word"]
    end

    WORD --> cl100k_path
    WORD --> custom_path

    style MSCORE fill:#e8f5e9,stroke:#2e7d32
    style CSCORE fill:#ffebee,stroke:#c62828
```

In plain language: fewer pieces means the tokenizer has already done more of the grouping work before the model sees the word.

## Why low fertility matters

The next diagram answers the obvious follow-up question: why should anyone care about a difference like 3 pieces versus 7 pieces?

The answer is that token count affects both efficiency and representation quality.

```mermaid
flowchart TD
    LF["🎯 Low Fertility\n(fewer tokens per word)"]

    B1["📐 Context window efficiency\nGPT-4 context = 128k tokens\nA 200-word abstract:\n  cl100k → ~294 tokens\n  custom-med → ~276 tokens\n  = 18 extra abstracts fit per batch"]

    B2["🎓 Better gradient signal\nEach token gets its own embedding vector\nFewer, longer tokens → richer representation\nof domain concepts per embedding slot"]

    B3["💰 Lower compute cost\nFewer tokens = fewer attention ops (O(n²))\n5 % fewer tokens → ~10 % cheaper attention"]

    B4["🧩 Semantic wholeness\n'acetylcholinesterase' in 3 pieces\n→ model can learn the whole concept\nin 3 vs 7 embedding updates"]

    LF --> B1
    LF --> B2
    LF --> B3
    LF --> B4

    style LF fill:#fff3e0,stroke:#e65100
```

So fertility is not just a cosmetic metric. It affects how much text fits in context, how much compute the model spends, and how fragmented important medical concepts become.

## The actual numbers from this lab

The table below moves from one word to the full held-out dataset.

This is important because the lab's claim is not based on a few hand-picked examples. It is based on average behavior over many unseen medical abstracts.

**Medical held-out set** (`pubmed_heldout.jsonl` — 5 000 abstracts, lower = better)

| Rank | Tokenizer | Fertility | Notes |
|------|-----------|-----------|-------|
| 🥇 1 | 🏥 **custom-med** (16k) | **1.381** | Domain training wins |
| 🥈 2 | 🤖 o200k (200k) | 1.439 | 12× bigger vocab, still loses |
| 🥉 3 | 🤖 cl100k (100k) | 1.470 | 6× bigger vocab, still loses |
| 4 | 📰 general-bpe (16k) | 1.759 | Same size as custom-med — domain is the secret |

> 🏆 `custom-med` wins with fertility **1.381**  
> Even `cl100k` (6× bigger vocab) loses — at **1.470**  
> `general-bpe` (same 16k size) is worst at **1.759** — vocab size isn't the secret, **domain is**

That is the key reading rule: lower fertility on held-out medical text means the tokenizer is representing medical language more efficiently.

## The flip side — general text

This second table is just as important as the first one.

If `custom-med` also won on general English, it would be harder to argue that it truly specialized for medicine. Losing on general text is the honest sign that its vocabulary budget was spent on medical patterns.

**General held-out set** (`general_heldout.jsonl` — 5 000 wikitext paragraphs, lower = better)

| Rank | Tokenizer | Fertility | Notes |
|------|-----------|-----------|-------|
| 🥇 1 | 🤖 o200k (200k) | **1.401** | Web-scale vocab wins on web text |
| 🥈 2 | 🤖 cl100k (100k) | 1.434 | Strong general baseline |
| 🥉 3 | 📰 general-bpe (16k) | 1.503 | Small but general-trained |
| 4 | 🏥 **custom-med** (16k) | 1.812 | Specialised vocab — expected to lose here |

> 🔄 `custom-med` **loses** on general text (1.812 — worst!)  
> This is expected and healthy — it specialised its 16k budget on medical morphemes  
> A student who notices this flip has understood domain specialisation

In plain language: a domain tokenizer is not supposed to win everywhere. It is supposed to win where its domain gives it an advantage.

## Fertility over a full abstract

This last diagram turns the metric into something more concrete.

Instead of looking at one word, it asks: what happens over an entire abstract?

Small differences in fertility per word become much larger differences when you multiply them across hundreds or thousands of documents.

```mermaid
flowchart LR
    ABS["📄 One PubMed abstract\n~200 whitespace words"]

    subgraph tokens_cl ["🤖 cl100k encoding"]
        TC["~294 tokens\n(fertility 1.470 × 200)"]
    end

    subgraph tokens_cm ["🏥 custom-med encoding"]
        TM["~276 tokens\n(fertility 1.381 × 200)"]
    end

    ABS --> tokens_cl
    ABS --> tokens_cm

    DIFF["💡 18 fewer tokens per abstract\n× 1 000 abstracts in a batch\n= 18 000 fewer tokens\n≈ 141 extra abstracts fit in same\n128k context window"]

    tokens_cl --> DIFF
    tokens_cm --> DIFF

    style TM fill:#e8f5e9,stroke:#2e7d32
    style DIFF fill:#fff3e0,stroke:#e65100
```

That is why fertility matters in practice. A small improvement per word can turn into meaningful savings in context usage and compute over a real dataset.

---

## Glossary at a glance

| Term | Formula | What it tells you |
|------|---------|------------------|
| **Fertility** | tokens ÷ whitespace words | Avg pieces per word (lower = more efficient) |
| **Chars/token** | characters ÷ tokens | Avg characters per token (higher = denser) |
| **Single-token rate** | fraction of terms = 1 token | Vocab coverage of domain jargon |

If you only remember one thing from this page, remember this: **lower fertility means the tokenizer needs fewer pieces to express the same text.**

---
*Back to theory: [01 why tokenization matters](../theory/01-why-tokenization-matters.md) · [04 metrics](../theory/04-metrics.md)*
