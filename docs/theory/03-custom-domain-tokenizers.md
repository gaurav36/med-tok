# Custom domain tokenizers

> 🔬 **Visual algorithm diagram:** [BPE training flowchart → docs/diagrams/01-bpe-algorithm.md](../diagrams/01-bpe-algorithm.md)

BPE learns merges from **your** corpus.

That means the tokenizer reflects the text you train it on. Frequent character patterns in PubMed abstracts become larger tokens. Rare patterns do not.

This is the whole reason to train a domain tokenizer: you want the vocabulary to spend its budget on the language that matters for your task.

## Main idea

A custom domain tokenizer is not trying to be good at everything.

It is trying to be especially good at one kind of text. In this lab, that text is biomedical and clinical language from PubMed abstracts.

So instead of spending merges on general web patterns, the tokenizer spends them on things like:

- drug names
- diagnosis phrases
- lab terms and units
- biomedical prefixes and suffixes

## How this lab trains

HuggingFace `tokenizers` (Rust, Python bindings):

1. Byte-level pre-tokenizer (every UTF-8 byte is representable → no `[UNK]`).
2. `BpeTrainer` with `vocab_size=16000` (class-fast, not Hub-scale 50k).
3. `train_from_iterator` over `data/pubmed_train.jsonl` (45k PubMed abstracts, disjoint
   from the 5k held-out eval set).
4. Wrap with `PreTrainedTokenizerFast` so files look like a Hub tokenizer.

In simple terms: we feed PubMed text into a byte-level BPE trainer and let it learn which character patterns are worth turning into reusable tokens.

Scripts: `scripts/train_medical_tokenizer.py`, `scripts/wrap_medical_tokenizer.py`.

Fallback if you have no network: `models/pretrained/medical_bpe_tiny/` (trained on the small
authored `data/medical_corpus.txt`). The tiny fallback shows the same pattern but with weaker
coverage because the corpus is only ~840 lines.

That fallback is useful for teaching because it preserves the same idea on a much smaller scale: better in-domain tokenization comes from in-domain training data.

## The fairness control: `general-bpe`

To isolate domain from vocab size, we also train:

```bash
uv run python scripts/train_medical_tokenizer.py \
    --corpus data/general_train.jsonl \
    --out artifacts/general-bpe/tokenizer.json \
    --vocab-size 16000
```

**Same script. Same algorithm. Same vocab size. Different corpus (wikitext-103).**

This is what makes the comparison fair. If the medical tokenizer wins, it cannot claim credit from using a different algorithm or a larger vocabulary. The only meaningful difference is the training corpus.

Expected result on medical held-out (`pubmed_heldout.jsonl`):

| tokenizer | fertility | interpretation |
|-----------|----------|---------------|
| `custom-med` 16k | **lowest** | domain merges earn their place |
| `cl100k` ~100k | middle | large vocab but wrong domain |
| `o200k` ~200k | slightly better than cl100k | very large vocab, still wrong domain |
| `general-bpe` 16k | **highest** | same size as custom-med, wrong domain — the proof |

If `custom-med` beats `general-bpe` on medical text, the cause is domain training, not vocab
budget. `general-bpe` spent its 16k merges on wikitext frequency. `custom-med` spent them on
PubMed frequency.

That is the key lesson: the merge budget is limited, so where you spend it matters.

## PubMedBERT vs BioBERT (the paper story)

**BioBERT** kept BERT's Wikipedia WordPiece. Continued pretrain on PubMed. Same alphabet, more
medical text.

**PubMedBERT** built a biomedical vocab from PubMed **and** pretrained from scratch. In-domain
whole words got their own embedding rows. It won on most biomedical benchmarks.

Lesson: a custom vocab helps **when the model is trained with that vocab**. The vocab alone does
not upgrade a frozen general LLM. See [06](06-pretrained-model-trap.md).

This is an important guardrail. It is easy to hear "custom tokenizer" and assume it is a drop-in upgrade. It is not. The tokenizer helps most when the model is trained to use it.

## Byte-level, not WordPiece

Decoder LLMs (GPT, Llama, Qwen) use byte-level BPE. This lab matches that. WordPiece (`##`) is
BERT-era. SentencePiece Unigram is T5-era. Do not mix those marks into this comparison.

So if you see `##` pieces in older BERT examples, that is a different tokenizer family. This lab stays close to decoder-style tokenization.

## Honest limits of a 16k vocab

- Some medical terms that appear in PubMed still don't make the cut if higher-frequency words
  consume the merge budget. `empagliflozin` tied with `cl100k` at 10 pieces — the 16k vocab was
  exhausted on more frequent terms (`hemoglobin`, `lymphocyte`, `myocardial`).
- The `μg` Unicode character (U+03BC) encodes as 2 UTF-8 bytes (`0xCE 0xBC`). PubMed often
  writes `microg/mL` instead — so the byte pair for the μ character was never frequent enough
  to merge. That is why `vancomycin trough 18 μg/mL` fragments in `custom-med`. Not a bug; a
  training-data normalisation issue.
- A larger vocab closes some of these gaps, but bigger is not automatically better.
  Run the [vocab-size experiment](08-vocab-size-tradeoff.md): train 16k / 32k / 50k / 64k / 100k
  on the **same** PubMed corpus, then pick the smallest size near minimum avg tokens/doc.
  This lab's corpus is small (<1B tokens) → the recommended band is **16k–32k**.

The important takeaway is that a custom tokenizer can still be imperfect. It is better aligned with the domain, but it is still constrained by corpus size and vocabulary budget.
