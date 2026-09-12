# Metrics

> ⚖️ **Fairness design diagram:** [4-tokenizer 2×2 design → docs/diagrams/02-fair-comparison.md](../diagrams/02-fair-comparison.md)  
> 🌱 **Fertility concept diagram:** [What fertility means → docs/diagrams/05-fertility-concept.md](../diagrams/05-fertility-concept.md)

Do not just look at token chips and guess. Use a small set of metrics and read them carefully.

This page explains what the lab measures and why those measurements matter.

## The big picture

In this lab, a tokenizer is better when it represents the same text in fewer, denser, more meaningful pieces without breaking round-trip decoding.

That is why we measure three things:

- **fertility**: how many token pieces the tokenizer uses on average
- **single-token rate**: how often important medical terms appear as one token
- **round-trip correctness**: whether text survives encode then decode unchanged

## Fertility

```text
fertility = n_tokens / n_whitespace_words
```

Lower is better. It means the tokenizer uses fewer pieces for the same text.

You can think of fertility as "average pieces per word." If a tokenizer turns one medical sentence into fewer pieces than another tokenizer, it is usually representing that sentence more compactly.

Report fertility on:
- **held-out domain text** (`pubmed_heldout.jsonl`) — the fair in-domain eval; custom-med should win here
- **held-out general text** (`general_heldout.jsonl`) — cross-domain; custom-med is expected to lose
- `medical_probes.txt` — 20 curated illustration phrases (not the pass bar; covered in `05`)

Also useful: **tokens per 100 characters** — insensitive to how you define a "word."

Why this matters:

- fewer tokens means less context-window usage
- fewer tokens usually means lower inference cost
- fewer tokens means the model sees long medical terms in fewer fragments

But fertility is not everything. A tokenizer can be compact and still be a bad fit for a task if it breaks decoding or handles important terms poorly. That is why the other metrics matter too.

## Single-token rate

For a list of target domain terms (`empagliflozin`, `creatinine`, `myocardial`, ...):

```text
single_token_rate = count(terms encoded as 1 piece) / count(terms)
```

Higher is better for a domain term list.

This metric answers a different question from fertility. Instead of asking "how compact is the text overall?" it asks "does the tokenizer have direct vocabulary support for the important words in this domain?"

Custom-med should be higher on the medical term list. General tokenizers often do much worse because they were not trained to spend vocabulary budget on medical jargon.

If a term appears as one token, that usually means the tokenizer has learned it as a stable unit. If it appears as many pieces, the tokenizer can still process it, but less directly.

## Round-trip

```text
decode(encode(x)) == x
```

Required for production. Byte-level BPE should round-trip all Unicode (`μg`, `β-lactam`, `Naïve`).
If it fails, the pretokenizer/decoder pair is wrong.

This metric is easy to overlook, but it is essential. A tokenizer is not useful if it changes the text when you encode and decode it.

So round-trip is the basic safety check: before asking whether the tokenizer is compact, first check that it is correct.

## The fair baseline: same-size general BPE

Comparing custom-med (16k vocab, domain) against `cl100k` (100k vocab, general) conflates two
variables: **domain** and **vocab size**. The honest control is a 16k BPE trained on general
text (wikitext-103) — same algorithm, same vocab budget, different training domain.

| tokenizer | vocab size | domain |
|-----------|-----------|--------|
| `cl100k` | ~100k | general |
| `o200k` | ~200k | general |
| `general-bpe` | 16k | general (wikitext) |
| **`custom-med`** | 16k | medical (PubMed) |

If `custom-med` beats `general-bpe` on medical held-out text, the gap is caused by **domain**,
not vocab size. If they tie, domain didn't matter — training corpus quality did.

This is the most important comparison in the lab. Without `general-bpe`, it would be easy to say that the result came from vocabulary size alone. With `general-bpe`, that excuse disappears.

## Control set

Custom **may lose** on `general_heldout.jsonl`. That is a pass, not a fail. You trained on
abstracts, not novels. If custom also crushes general English, the corpus leaked too much
generic prose — inspect `medical_corpus.txt` or `pubmed_train.jsonl`.

This is why the cross-domain evaluation is useful. It checks whether the tokenizer truly specialized or whether it somehow stayed broadly general.

## Winner column

On each probe, the tokenizer with the fewest pieces wins that probe. Ties are ties.

But the real pass condition for the lab is broader than a few cherry-picked examples. **The golden assert in the tests is this: custom-med fertility must be lower than cl100k fertility on held-out PubMed abstracts.**

That is the key point: use the probe examples to build intuition, but use held-out aggregate metrics to make the real claim.
