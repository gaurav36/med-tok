# General-purpose tokenizers

A general-purpose tokenizer is trained on a **broad mix of text**: web pages, books, code, Wikipedia, and other everyday language sources.

That makes it useful across many domains. But it also means it does not spend much of its vocabulary budget on rare medical strings.

So these tokenizers are not bad. They are just optimized for a different world. They are built to be good on average across many kinds of text, not especially good on clinical or biomedical language.

## The main idea

When a tokenizer is trained on general text, it learns merges for patterns that appear often in general text.

That usually means:

- common English words
- common suffixes and prefixes
- punctuation and spacing patterns seen on the web
- frequent code and markup fragments

Medical terms do appear on the web, but usually not often enough to earn many dedicated merges. So long drug names, ICD strings, and lab units often get chopped into many smaller pieces.

## The three general baselines in this lab

**tiktoken `cl100k_base`.** Byte-level BPE used by the GPT-4 family. About 100k vocabulary items. It comes with fixed merge tables. It does **not** learn anything from this lab's medical corpus.

**tiktoken `o200k_base`.** Byte-level BPE used by GPT-4o. About 200k vocabulary items. It is a stronger general baseline because a larger vocabulary can compress more text, including some medical text, but it is still trained for general use.

**`general-bpe` (fairness control).** A 16k byte-level BPE trained in this lab on `data/general_train.jsonl` (wikitext-103). It uses the same algorithm and the same vocabulary size as `custom-med`.

This is the most important control in the lab. If `custom-med` beats `general-bpe` on medical text, then the difference comes from **training domain**, not from vocabulary size or algorithm choice.

```python
import tiktoken
enc_cl = tiktoken.get_encoding("cl100k_base")
enc_o  = tiktoken.get_encoding("o200k_base")

enc_cl.encode("empagliflozin 10 mg daily")
# → many ids; decode each to see the fragments

enc_o.encode("empagliflozin 10 mg daily")
# → slightly fewer ids (200k vocab > 100k), still fragments
```

If network allows, Qwen (`Qwen/Qwen3-0.6B`) is loadable via `AutoTokenizer` and behaves like
another large general tokenizer. It is **optional** — tiktoken + `custom-med` + `general-bpe`
is enough to show the full pattern.

So students should not read `cl100k` and `o200k` as "bad tokenizers." They are strong general tokenizers. The point of the lab is that a tokenizer can still lose badly when the evaluation domain is much narrower than its training mix.

## What general tokenizers are good at

Everyday English, code, mixed web text. `data/medical_control.txt` and `data/general_heldout.jsonl`
are that world. A medical BPE is **not** required to beat them there. Domain fit, not universal
superiority.

This is an important mindset: a domain tokenizer is not supposed to win everywhere. It is supposed to win where its training corpus gives it an advantage.

Expected result on `general_heldout.jsonl`:

| tokenizer | fertility |
|-----------|----------|
| `o200k` | lowest |
| `cl100k` | slightly higher |
| `general-bpe` (16k) | slightly higher — small vocab costs something on general text |
| `custom-med` (16k) | highest — paid for medical specialization |

That last row is not a failure. It is evidence that the tokenizer really specialized.

## What they are bad at (healthcare)

- Brand and generic drug names (`empagliflozin`, `oseltamivir`, `acetylcholinesterase`)
- Gene symbols glued to clinical words (`BRCA1 pathogenic variant`)
- Billing codes (`ICD-10-CM E11.65`)
- Lab units (`mg/dL`, `μg/mL`, `pg/mL`)
- Greek / mixed Unicode (`β-lactam`, `Naïve CD4+`)

Single-token rate on 20 medical terms: **cl100k 0/20, o200k 0/20** — not a single medical term
exists as one token. `custom-med` gets 2/20. That gap is the vocabulary story.

In simpler language: general tokenizers usually know enough to process these terms, but not enough to represent them compactly. They can still read the text, but they read it in many fragments.

Count pieces. That is the whole demo.

That is why this page matters before the notebook. Once students see the split counts side by side, they can connect the outcome to the training domain of each tokenizer rather than assuming it is only about vocabulary size.
