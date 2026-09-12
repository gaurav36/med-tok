# 🗺️ Full Lab Pipeline

> **Two tracks, one destination:** The data and models are built once in the setup phase.
> Then the notebook is opened and everything prepared is explored.

This page shows the whole lab from raw data to final comparison.

If you want the short version, read the diagram in three parts:

1. **setup** builds the datasets and tokenizers
2. **shared artifacts** store the files produced by setup
3. **lab usage** explores those artifacts in the notebook, CLI, and tests

## Complete pipeline flowchart

```mermaid
flowchart LR
    subgraph instructor ["⚙️ Setup Phase (run once, needs internet)"]
        direction TB
        I1["🌐 download_pubmed_sample.py\n→ data/pubmed_sample.jsonl\n(50k abstracts, ~180 MB)"]
        I2["✂️ split_corpus.py\n→ data/pubmed_train.jsonl  45k\n→ data/pubmed_heldout.jsonl  5k"]
        I3["🏋️ train_medical_tokenizer.py\n--input pubmed_train.jsonl\n--output artifacts/medical-bpe-pubmed/\n→ tokenizer.json (16k vocab)"]
        I4["🌍 download_general_sample.py\n→ data/general_train.jsonl  45k\n→ data/general_heldout.jsonl  5k\n(wikitext-103)"]
        I5["🏋️ train_medical_tokenizer.py\n--input general_train.jsonl\n--output artifacts/general-bpe/\n→ tokenizer.json (16k vocab)"]
        I6["🎁 wrap_medical_tokenizer.py\n→ models/pretrained/medical_bpe_tiny/\n   tokenizer.json\n   tokenizer_config.json"]
        I7["📉 sweep_vocab_size.py\n16k 32k 50k 64k 100k\navg tokens per held-out doc"]

        I1 --> I2
        I2 --> I3
        I4 --> I5
        I3 --> I6
        I3 --> I7
    end

    subgraph student ["🎓 Student Experience (offline, repo cloned)"]
        direction TB
        S1["📦 uv sync\n(install all deps)"]
        S2["📓 Open notebook\nnotebooks/04_custom_vs_general.ipynb"]
        S3["▶️ Run all cells\n4-way compare then vocab-size experiment\n(predict, table, embedding cost, pick)"]
        S4["🧪 pytest tests/\ncompare + vocab-sweep tests"]
        S5["📚 Read theory docs\ndocs/theory/\ndocs/diagrams/"]

        S1 --> S2
        S2 --> S3
        S3 --> S4
        S3 --> S5
    end

    subgraph outputs ["📊 Shared Artefacts"]
        O1["🏥 artifacts/medical-bpe-pubmed/tokenizer.json\n16k vocab, PubMed-trained"]
        O2["📰 artifacts/general-bpe/tokenizer.json\n16k vocab, wikitext-trained"]
        O5["📉 artifacts/medical-bpe-pubmed-16k to 100k\nvocab-size sweep"]
        O3["🔬 data/pubmed_heldout.jsonl\n5k unseen medical abstracts"]
        O4["📰 data/general_heldout.jsonl\n5k unseen general paragraphs"]
    end

    instructor --> outputs
    outputs --> student

    subgraph cli ["🖥️ CLI (anytime)"]
        C1["python scripts/compare_tokenizers.py\n  --heldout-medical data/pubmed_heldout.jsonl\n  --heldout-general data/general_heldout.jsonl"]
        C2["python scripts/sweep_vocab_size.py\n  --corpus data/pubmed_train.jsonl\n  --heldout data/pubmed_heldout.jsonl"]
    end

    outputs --> cli

    style instructor fill:#fff8e1,stroke:#f9a825
    style student fill:#e8f5e9,stroke:#2e7d32
    style outputs fill:#e3f2fd,stroke:#1565c0
    style cli fill:#f3e5f5,stroke:#6a1b9a
```

## How to read this pipeline

The left side is the preparation phase. This is where corpora are downloaded, split, and used to train tokenizers.

The center is the shared output of that preparation:

- held-out datasets for fair evaluation
- trained tokenizers for medical and general text
- optional vocab-size sweep artifacts

The right side is how those outputs get used:

- the notebook for the main learning experience
- CLI scripts for quick comparisons
- tests for numerical verification

So the notebook is not doing everything from scratch. It is consuming artifacts that were prepared earlier.

## Why the split into phases matters

The setup phase needs more time and may need internet access. The notebook phase is the teaching and exploration phase.

That separation is useful because it lets the lab work in two modes:

- **full pipeline mode**: download data, train tokenizers, run the whole experiment
- **offline learning mode**: reuse prepared artifacts and focus on understanding the results

This is why the repo includes fallback files such as `models/pretrained/medical_bpe_tiny/`.

## Script quick reference

| Script | When | Input | Output |
|--------|------|-------|--------|
| `download_pubmed_sample.py` | Setup | HuggingFace Hub | `data/pubmed_sample.jsonl` |
| `split_corpus.py` | Setup | pubmed_sample.jsonl | train + heldout splits |
| `download_general_sample.py` | Setup | HuggingFace Hub | general train + heldout |
| `train_medical_tokenizer.py` | Setup (×2) | any `.jsonl` corpus | `tokenizer.json` (BPE) |
| `wrap_medical_tokenizer.py` | Setup | tokenizer.json | HuggingFace-compatible model |
| `compare_tokenizers.py` | Lab | heldout files | Fertility table (terminal) |
| `sweep_vocab_size.py` | Lab | pubmed train + heldout | 16k–100k avg tokens/doc table |
| `pytest tests/` | Lab | held-out files | compare + vocab-sweep asserts |

## Main takeaway

The pipeline is designed so that each stage answers one question:

- data preparation makes the train and held-out split fair
- tokenizer training asks what the corpus teaches the vocabulary
- comparison asks which tokenizer is best on held-out text
- vocab sweep asks how large the medical vocabulary should be
- tests check that the main claims are reproducible

In plain language: this is not just a notebook with nice plots. It is a small end-to-end experiment.

---
*Back to theory: [00 overview](../theory/00-overview.md)*
