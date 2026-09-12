# ⚠️ The Pretrained-Model Trap

> **"I'll just swap in my custom tokenizer on Qwen/LLaMA and do LoRA."**
> This sounds reasonable. It is catastrophically wrong.

This page is the visual version of the most important warning in the lab.

The short version is simple: a pretrained tokenizer and a pretrained model were learned together. If you replace one but keep the other, the mapping between token IDs and embedding rows breaks.

## The core mismatch

Read this first diagram as a mismatch between two tables:

- the tokenizer decides which integer ID each string gets
- the model's embedding matrix decides what each integer ID means

If those two tables were built together, everything is aligned.
If you swap only the tokenizer, the same integer IDs now point to the wrong meanings.

```mermaid
flowchart TD
    TJ["🏥 custom-med tokenizer.json\n16 000 token IDs\nlearned from PubMed"]
    PW["🧠 Qwen-2.5 weights\n152 000 embedding rows\none row per Qwen token ID"]

    TJ -- "❌ ID 4521 in custom-med\n= 'gluc'" --> MISMATCH
    PW -- "❌ Row 4521 in Qwen\n= learned repr of 'the'" --> MISMATCH

    MISMATCH["🔥 Embedding mismatch!\nRow 4521 was trained to mean 'the'\nnow receives signal for 'gluc'\nAll 16 000 IDs point to wrong rows"]

    MISMATCH --> SCRAMBLE["💥 Scrambled model\nLoRA only updates a few adapters\nUnderlying embedding confusion\npersists throughout fine-tuning\nModel outputs nonsense or random text"]

    style MISMATCH fill:#ffebee,stroke:#c62828
    style SCRAMBLE fill:#ffebee,stroke:#c62828
```

In plain language: the model is not confused because the new tokenizer is medical. It is confused because token ID 4521 no longer means the same thing to the tokenizer and the model.

## Two valid paths, one fatal path

This second diagram is about choices.

It shows one unsafe path and two safe ones:

- unsafe: replace the tokenizer of a pretrained model
- safe: keep the pretrained tokenizer and extend it carefully
- safe: train a new model from scratch with the custom tokenizer

```mermaid
flowchart LR
    START["🎯 Goal: Medical LLM"]

    subgraph wrong ["❌ WRONG PATH — The Trap"]
        W1["Take Qwen-2.5 weights\n(trained with Qwen tokenizer)"]
        W2["Swap tokenizer.json\nfor custom-med (16k vocab)"]
        W3["LoRA fine-tune on\nmedical text"]
        W4["🔥 FAIL\nIDs don't match embedding rows\nGradients flow into wrong rows\nModel never converges properly"]
        W1 --> W2 --> W3 --> W4
    end

    subgraph right_a ["✅ RIGHT PATH A — Keep the tokenizer"]
        A1["Take Qwen-2.5 weights\n(trained with Qwen tokenizer)"]
        A2["Keep Qwen tokenizer\n(IDs still match rows!)"]
        A3["add_tokens() for\nnew medical terms"]
        A4["resize_token_embeddings()\n→ new rows initialised randomly"]
        A5["LoRA fine-tune\nNew rows learn from scratch\nOld rows adapt via LoRA"]
        A6["✅ Works\nVocab and embeddings stay aligned"]
        A1 --> A2 --> A3 --> A4 --> A5 --> A6
    end

    subgraph right_b ["✅ RIGHT PATH B — Train from scratch"]
        B1["custom-med tokenizer\n(16k, PubMed-trained)"]
        B2["Random weight init\n(new 16k embedding table)"]
        B3["Pre-train LM from scratch\non medical corpus"]
        B4["LoRA or full fine-tune\nfor downstream tasks"]
        B5["✅ Works\nTokenizer and embeddings\nborn together — always aligned"]
        B1 --> B2 --> B3 --> B4 --> B5
    end

    START --> wrong
    START --> right_a
    START --> right_b

    style wrong fill:#ffebee,stroke:#c62828
    style right_a fill:#e8f5e9,stroke:#2e7d32
    style right_b fill:#e8f5e9,stroke:#2e7d32
```

That is the decision rule this page wants to teach: if the model already exists, keep its tokenizer unless you are doing a controlled vocabulary extension.

## Why the alignment matters: a concrete example

This example makes the mismatch less abstract.

Before the swap, token ID 7821 and embedding row 7821 refer to the same concept.

After the swap, token ID 7821 may now mean `empagliflozin`, but embedding row 7821 still contains the meaning learned for some older Qwen token.

So the model starts with the wrong input representation before attention even begins.

```mermaid
flowchart LR
    subgraph qwen_world ["🧠 Qwen's world (trained)"]
        QT["Token ID 7821\n= Ġmedical\n(Qwen vocab)"]
        QE["Embedding row 7821\n[0.34, -0.12, 0.78, ...]\nEncodes meaning of 'medical'"]
        QT <-->|"✅ match"| QE
    end

    subgraph swapped ["⚠️ After custom-med swap"]
        ST["Token ID 7821\n= 'empagliflozin'\n(custom-med vocab)"]
        SE["Embedding row 7821\n[0.34, -0.12, 0.78, ...]\nStill encodes 'medical' meaning!"]
        ST <-->|"❌ mismatch"| SE
    end

    qwen_world -- "Swap tokenizer\nonly" --> swapped

    style swapped fill:#fff3e0,stroke:#e65100
```

That is why LoRA is not enough. LoRA can adapt behavior, but it does not automatically rebuild the full vocabulary-to-embedding alignment from scratch.

## The golden rule

This final diagram gives the memory rule for the whole topic.

If you remember only one sentence from this page, remember this one: tokenizer IDs and embedding rows must stay aligned.

```mermaid
flowchart TD
    RULE["📏 Golden Rule\nA tokenizer and its model weights are\n**inseparable**. They were trained together.\nYou cannot split them without breaking both."]

    C1["🔑 Keep tokenizer + weights together\n→ extend vocabulary safely with add_tokens"]
    C2["🏗️ Train together from scratch\n→ build your own LM on custom-med tokens"]

    RULE --> C1
    RULE --> C2

    style RULE fill:#fff3e0,stroke:#e65100
```

> **This lab** focuses on tokenizers, not LM training. But understanding this trap helps you see
> why the tokenizer step is not just a preprocessing detail — it **determines what model you can use**.

In plain language: better tokenization alone does not give you permission to plug that tokenizer into any pretrained model.

---
*Back to theory: [06 pretrained model trap](../theory/06-pretrained-model-trap.md)*
