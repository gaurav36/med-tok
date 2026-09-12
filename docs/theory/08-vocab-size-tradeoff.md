# Vocab size is a trade-off, not a trophy

> 📉 **Visual:** [Diminishing returns + embedding cost → docs/diagrams/06-vocab-size-knee.md](../diagrams/06-vocab-size-knee.md)

There is **no single best vocabulary size** for every tokenizer.

Making the vocabulary larger usually gives shorter tokenized sequences. But it also makes the model carry a larger embedding table. So the question is not "how large can I make the vocab?" The real question is "when do extra vocabulary entries stop helping enough to justify their cost?"

The earlier lab proved that **domain** beats a 100k general vocab. This experiment
asks the next question: *given the same PubMed train set, which budget?*

That is why this page matters. Once you decide to build a medical tokenizer, you still need to choose how many merges or vocabulary entries to allow.

## Practical experiment

1. Train multiple tokenizers on **the same corpus**: 16k, 32k, 50k, 64k, 100k.
2. Measure **average tokens per document** (or per MB). Smaller is generally better —
   the model then processes fewer tokens.
3. Compare **diminishing returns**. The last jump often saves ~1% of tokens.
4. Consider **embedding cost**. Larger vocab grows both the input embedding and the
   output layer.

Same algorithm. Same `data/pubmed_train.jsonl`. Only `vocab_size` changes.
Eval is on held-out abstracts (`data/pubmed_heldout.jsonl`) — never the train set.

This is important because it isolates one variable at a time. If the corpus, algorithm, and evaluation set all stay fixed, then any difference you see comes from vocabulary size alone.

CLI (run during setup):

```bash
uv run python scripts/sweep_vocab_size.py \
    --corpus data/pubmed_train.jsonl \
    --heldout data/pubmed_heldout.jsonl \
    --no-qwen
```

## Worked example (large corpus — not this lab)

These numbers are a **lesson table** from a large pretraining corpus. They teach
you how to read flattening. They are **not** the PubMed answer.

| Vocabulary size | Avg. tokens/doc |
|-----------------|-----------------|
| 16k | 1,250 |
| 32k | 1,050 |
| 50k | 980 |
| 64k | 960 |
| 100k | 950 |

64k → 100k saves about **1%** of tokens. Pick **64k**. Extra 36k embedding rows
are not worth 10 tokens per document.

This is what people mean by the **knee** of the curve. Early increases in vocabulary size help a lot. Later increases help only a little. The best choice is often near the point where the curve starts to flatten.

Your PubMed run will look different. 45k abstracts are tens of millions of tokens,
not billions. The knee should sit **left** of this example — usually 16k or 32k.

In plain language: smaller corpora usually cannot justify extremely large vocabularies, because they do not contain enough repeated patterns to make all those extra entries useful.

## Embedding cost

```text
embedding params ≈ vocab_size × d_model
```

With `d_model = 1024` (a Qwen-scale stand-in):

| vocab | params | vs 16k |
|------:|-------:|-------:|
| 16k | 16,384,000 | 1.00× |
| 32k | 32,768,000 | 2.00× |
| 50k | 51,200,000 | 3.12× |
| 64k | 65,536,000 | 4.00× |
| 100k | 102,400,000 | **6.25×** |

Those rows exist twice: input embedding **and** output softmax. After the knee,
you buy a 6× table for almost no compression.

That is the trade-off the page wants students to notice:

- larger vocab can reduce token count
- larger vocab also increases model size
- after a certain point, the token savings become too small to be worth the extra parameters

## Corpus-size rule

| Train-set scale | Recommended vocab |
|-----------------|-------------------|
| Small (<1B tokens) | **16k–32k** |
| Medium (1–50B) | **32k–64k** |
| Very large (100B+) | **50k–100k** |

Modern LLM pretraining often uses **32k–64k**, with **50k** as a common sweet
spot. That default assumes a **large** corpus. This lab's PubMed sample is small.
Do not copy 50k because "that's what Qwen uses."

This is a common beginner mistake. Students often see a number used by a large production model and assume it must also be best for a much smaller custom corpus. Usually it is not.

BPE `vocab_size` is a **maximum**. If `actual < requested`, the corpus ran out of
merges with `min_frequency=2`. That is a finding, not a crash.

In simpler terms: asking for 100k does not guarantee you will get 100k useful merges. If the corpus is too small or too repetitive, the tokenizer may stop earlier.

## Rule of thumb

Choose the **smallest** vocabulary that achieves **near-minimum** tokenization
length. Once the token count stops improving much, stop. Extra vocab is extra
parameters.

So the goal is not to win a "largest vocab" contest. The goal is to find the smallest size that already gets almost all of the compression benefit.

The notebook helper `recommend_vocab(rows, threshold=0.02)` picks the smallest
size within 2% of the best (lowest) avg tokens/doc.

This pick is for **training a tokenizer** or a **from-scratch** model. It does
**not** license swapping that tokenizer onto Qwen. Read [06](06-pretrained-model-trap.md).
