# Why custom wins in healthcare

> ⚖️ **Fairness design diagram:** [4-tokenizer 2×2 design → docs/diagrams/02-fair-comparison.md](../diagrams/02-fair-comparison.md)

General BPE was not paid to memorize drug names. Your BPE was — and the data proves it.

## Predict first

Suppose you train two tokenizers with the same byte-level BPE algorithm and the same 16k vocabulary size:

- one on PubMed abstracts
- one on general web or encyclopedia text

Which one should tokenize `empagliflozin`, `acetylcholinesterase`, or `vancomycin trough` into fewer pieces?

The PubMed tokenizer should win, because those strings appear often enough in medical text to earn merge budget. The general tokenizer spends that same budget on common general-English pairs instead.

## BPE is frequency-driven

Byte-pair encoding is a greedy merge algorithm. At each step it finds the most frequent pair
of adjacent tokens in the training corpus and merges them into one symbol. After enough merges:

- A word seen thousands of times (`empagliflozin` in PubMed abstracts) accumulates enough
  adjacent-pair frequency to survive as a **single merge** — one token.
- The same word is **vanishingly rare** on the general web. `cl100k` or `o200k` has never seen
  enough `empagliflozin` occurrences to merge it, so it falls back to byte-level fragments:
  `Ġemp`, `ag`, `lif`, `lo`, `zin` — five unrelated-looking pieces until the model learns to
  glue them.

The key idea is simple: BPE does not know medicine. It only knows frequency. If medical strings are frequent in the training corpus, BPE will compress them. If they are rare, BPE will not spend merges on them.

In other words, the tokenizer is a mirror of the corpus it was trained on. Train on PubMed, and the merge budget gets spent on drug names, laboratory terms, diagnosis phrases, abbreviations, and clinical units. Train on general text, and that same budget gets spent on everyday English.

## One concrete example

Take the string `empagliflozin 10 mg daily`.

- In a medical corpus, `empagliflozin` appears often enough that several of its character pairs become worth merging.
- In a general corpus, that same word is so rare that the tokenizer keeps spending its merges on words like `the`, `patient`, `because`, `tion`, or `ing` instead.

So even when both tokenizers have the same vocabulary size, they learn different priorities:

- the medical tokenizer compresses drug names, lab terms, and diagnosis phrases
- the general tokenizer compresses common English patterns

That is why the custom tokenizer wins on medical text without needing a larger vocabulary.

This is the central idea of the lab: vocabulary size matters, but **where the merge budget is spent** matters more. A small tokenizer trained on the right domain can beat a much larger tokenizer trained on the wrong one.

## What the numbers mean

| Metric | custom-med | cl100k | o200k |
|--------|-----------|--------|-------|
| `empagliflozin 10 mg daily` | ~6–7 pieces | 10 pieces | 9 pieces |
| `acetylcholinesterase inhibitor` | 3–4 pieces | 7 pieces | 6 pieces |
| Held-out PubMed fertility | **lower** | higher | higher |
| Held-out general English | higher | **lower** | **lower** |

The last two rows are the teaching payoff. Custom-med wins in-domain and loses out-of-domain —
exactly what specialization should do.

This "flip" is important. If custom-med also won on general English, you would have less evidence that it truly specialized. Losing out-of-domain is the honest sign that its merge budget was spent on medical language.

## The fairness control: general-bpe (16k on wikitext)

The naive objection: "maybe custom wins because it has a smaller vocabulary, not because it
knows medicine." The `general-bpe` control (same 16k vocab, same byte-level BPE algorithm,
trained on wikitext-103) destroys that argument:

- `general-bpe` and `custom-med` have **identical algorithm and vocab size**.
- `general-bpe` beats `cl100k`/`o200k` on general English — it spends its 16k merges where
  general text is dense.
- `custom-med` beats `general-bpe` on medical text — it spends its 16k merges where medical
  text is dense (`empagliflozin`, `acetylcholinesterase`, `creatinine`, ...).

The only variable left is **training domain**. That is the proof.

This is the most important comparison in the lab. Comparing `custom-med` only against `cl100k` or `o200k` is not enough, because those tokenizers also differ in vocabulary size and training mixture. Comparing `custom-med` against `general-bpe` isolates the real cause.

That is what makes `general-bpe` a control rather than just another baseline. It answers a specific question: if two tokenizers get the same number of merges, does the domain of the training corpus change the result? In this lab, the answer is yes.

## Worked probes

| Probe | Why the general tokenizer over-splits |
|-------|--------------------------------------|
| `empagliflozin 10 mg daily` | Long generic drug name; rare on the open web |
| `acetylcholinesterase inhibitor` | Compound chemistry + clinical noun |
| `BRCA1 pathogenic variant` | Gene symbol + standardized phrase |
| `ICD-10-CM E11.65` | Dots, hyphens, digits, billing syntax |
| `serum creatinine 1.4 mg/dL` | Lab name + unit |
| `β-lactam allergy` | Greek letter + hyphenated drug class |
| `levothyroxine 75 μg daily` | Microgram sign; uncommon outside clinical notes |
| `vancomycin trough 18 μg/mL` | Drug + PK jargon + unit |
| `ST-elevation myocardial infarction` | Hyphenated syndrome name |

Prediction exercise: before running any cell, guess the piece counts. General tokenizers
usually shatter the drug names. Custom should not.

## What "win" means in this lab

Fewer tokens for the same clinical string → shorter sequences → more of a clinical note fits
in the model's context window → one (or two) embedding rows per concept instead of a jigsaw
of fragments.

`empagliflozin` as one token ID is a drug. As five fragment IDs it is noise until the model
accumulates many training examples to learn the composition.

That does not mean one token is magically smarter than five tokens. It means the model has to do less reconstruction work. The concept arrives in larger, more coherent pieces.

For students, the easiest way to read the result is: fewer pieces means the tokenizer has already done more of the grouping work before the model sees the text.

## Honest limit

This lab measures tokenization quality, not full model task accuracy.

What it does prove is simpler: on medical text, a medical tokenizer can represent words in fewer, more meaningful pieces.

Real task improvements, such as better classification or NER, happen when a model is actually trained or pretrained with that tokenizer.

See [06 — the pretrained-model trap](06-pretrained-model-trap.md) for exactly why you cannot
drop `artifacts/medical-bpe-hf/` onto Qwen and LoRA it.

## Not a win on every probe

Some probes are already common enough in general corpora that `cl100k` handles them well
(`pneumonia`, `COVID-19`, `chronic kidney disease`). The teaching examples are the ones that
**explode** in general tokenizers. Those are the ones to show first.
