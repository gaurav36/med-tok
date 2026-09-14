# 🧑‍💻 Building Guide — Build & Compare a Domain Tokenizer

**Session duration:** 4 hours  
**What you'll build:** A medical BPE tokenizer trained on PubMed abstracts, then compare it fairly against 3 other tokenizers to prove domain training — not vocab size — drives the performance gap.  
**What you already know:** BPE algorithm, merge rules, vocab size, byte-level encoding.

---

## ⚡ Quick orientation

We compare **4 tokenizers** today:

| Name | Vocab | Trained on | Role |
|------|-------|-----------|------|
| `cl100k` | ~100k | General web (GPT-4) | Real-world baseline |
| `o200k` | ~200k | General web (GPT-4o) | Stronger baseline |
| `custom-med` | 16k | PubMed abstracts | **You train this today** |
| `general-bpe` | 16k | wikitext-103 | Fairness control — same size as custom-med |

> The `general-bpe` row is the key. If `custom-med` beats it on medical text, the win comes from **domain**, not vocab size.

---

## 📂 Phase 0 — Setup (5 min)

Clone the repo and install dependencies:

```bash
git clone git@github.com:gaurav36/med-tok.git
cd med-tok
uv sync
```

Confirm everything installed:

```bash
uv run python -c "import tokenizers, tiktoken, transformers, datasets; print('all good')"
# Expected: all good
```

---

## 📥 Phase 1 — Download the corpora (10–15 min, needs internet)

### 1a · Medical corpus — 50k PubMed abstracts

```bash
uv run python scripts/download_pubmed_sample.py --max-docs 50000
# Streams from HuggingFace Hub (Parquet) → no dataset-script issues
# → data/pubmed_sample.jsonl  (~50k rows, ~180 MB)
# Takes ~5–10 min depending on connection
```

### 1b · Split into train and held-out sets

```bash
uv run python scripts/split_corpus.py
# → data/pubmed_train.jsonl    (45 000 abstracts — used for training)
# → data/pubmed_heldout.jsonl  ( 5 000 abstracts — NEVER used for training)
```

> ❓ **Why held-out?** We measure tokenizer quality on text it has never seen. If we tested on training data, results would be inflated. The held-out set is the fair judge.

### 1c · General corpus — 50k wikitext-103 paragraphs

```bash
uv run python scripts/download_general_sample.py
# Downloads wikitext-103 from Salesforce/wikitext (HuggingFace Hub)
# Automatically splits into train + held-out
# → data/general_train.jsonl    (45 000 paragraphs)
# → data/general_heldout.jsonl  ( 5 000 paragraphs)
# Takes ~3–5 min
```

#### If you want a 100k-document split instead

```bash
uv run python scripts/download_general_sample.py  --train-n 95000 --heldout-n 5000
```

### 1d · Verify all 4 files

```bash
wc -l data/pubmed_train.jsonl data/pubmed_heldout.jsonl data/general_train.jsonl data/general_heldout.jsonl
```

On Windows / VS Code 

```bash
Get-Item data/pubmed_train.jsonl, data/pubmed_heldout.jsonl, data/general_train.jsonl, data/general_heldout.jsonl | ForEach-Object { [pscustomobject]@{ File = $_.Name; Lines = (Get-Content $_.FullName | Measure-Object -Line).Lines } } | Format-Table -AutoSize
```

Expected output (when `--max-docs 50000` was used):
```
 45000 data/pubmed_train.jsonl
  5000 data/pubmed_heldout.jsonl
 45000 data/general_train.jsonl
  5000 data/general_heldout.jsonl
```

Expected output (when `--train-n 95000 --heldout-n 5000` was used)
```
File                  Lines
----                  -----
pubmed_train.jsonl    95000
pubmed_heldout.jsonl   5000
general_train.jsonl   95000
general_heldout.jsonl  5000
```

---

## 🔍 Phase 2 — Inspect the data (5 min)

Look at one PubMed abstract:

```bash
head -1 data/pubmed_train.jsonl | python -m json.tool
```

On Windows / VS Code:

```powershell
Get-Content data/pubmed_train.jsonl -TotalCount 1 | uv run python -m json.tool
```

Look at one wikitext paragraph:

```bash
head -1 data/general_train.jsonl | python -m json.tool
```

On Windows / VS Code:

```powershell
Get-Content data/general_train.jsonl -TotalCount 1 | uv run python -m json.tool
```

> ❓ **Think:** What types of words will appear most often in PubMed that never appear in wikitext?  
> Examples: `empagliflozin`, `acetylcholinesterase`, `SGLT2`, `myocardial`  
> BPE will find these pairs and merge them. wikitext BPE never will.

---

## 🏋️ Phase 3 — Train the tokenizers (10–15 min)

### 3a · Train `custom-med` on PubMed

#### 16k tokenizer
```bash
uv run python scripts/train_medical_tokenizer.py \
  --input data/pubmed_train.jsonl \
  --vocab-size 16000 \
  --output artifacts/medical-bpe-pubmed/
# → artifacts/medical-bpe-pubmed/tokenizer.json

```
#### On Windows / VS Code
```
uv run python scripts/train_medical_tokenizer.py --corpus data/pubmed_train.jsonl --vocab-size 16000 --out artifacts/medical-bpe-pubmed/tokenizer.json
```

This command trains the medical tokenizer on PubMed abstracts and writes the result to `artifacts/medical-bpe-pubmed/tokenizer.json`.

While it runs, watch the progress bar — each step = one more merge rule added to the vocab.

#### 32k tokenizer

```bash
uv run python scripts/train_medical_tokenizer.py --corpus data/pubmed_train.jsonl --vocab-size 32000 --out artifacts/medical-bpe-pubmed/tokenizer.json
# → artifacts/medical-bpe-pubmed/tokenizer.json
```

After it finishes, peek inside:

```bash
python -c "
import json
t = json.load(open('artifacts/medical-bpe-pubmed/tokenizer.json'))
print('First 20 merges:')
for m in t['model']['merges'][:20]:
    print(' ', m)
"
```

On Window/VS Code

```bash
uv run python -c "import json; t=json.load(open(r'artifacts/medical-bpe-pubmed/tokenizer.json', encoding='utf-8')); print('First 20 merges:'); [print(' ', m) for m in t['model']['merges'][:20]]"
```

> ❓ **Notice:** Early merges are high-frequency medical character pairs (`ch`, `ol`, `in`, `ester`). These co-occur thousands of times per abstract.

### 3b · Train `general-bpe` on wikitext

```bash
uv run python scripts/train_medical_tokenizer.py \
  --input data/general_train.jsonl \
  --vocab-size 16000 \
  --output artifacts/general-bpe/
```

#### Working command for 16k vocab
```bash
uv run python scripts/train_medical_tokenizer.py --corpus data/general_train.jsonl --vocab-size 16000 --out artifacts/general-bpe/tokenizer.json

# → artifacts/general-bpe/tokenizer.json
```

#### Working command for 32k vocab
```bash
uv run python scripts/train_medical_tokenizer.py --corpus data/general_train.jsonl --vocab-size 32000 --out artifacts/general-bpe/tokenizer.json

# → artifacts/general-bpe/tokenizer.json
```

Peek at its early merges:

```bash
python -c "
import json
t = json.load(open('artifacts/general-bpe/tokenizer.json'))
print('First 20 merges:')
for m in t['model']['merges'][:20]:
    print(' ', m)
"
```


#### For uv based env, use this command 

```bash
uv run python -c "import json; t=json.load(open(r'artifacts/general-bpe/tokenizer.json', encoding='utf-8')); print('First 20 merges:'); [print(' ', m) for m in t['model']['merges'][:20]]"
```

> ❓ **Compare:** Early merges are `th`, `in`, `er`, `on` — common English, zero medical morphemes. Same algorithm, completely different vocabulary.

### 3c · Wrap `custom-med` for HuggingFace

```bash
uv run python scripts/wrap_medical_tokenizer.py --tokenizer-json artifacts/medical-bpe-pubmed/tokenizer.json --out artifacts/medical-bpe-pubmed-hf
# → artifacts/medical-bpe-pubmed-hf/tokenizer.json
# → artifacts/medical-bpe-pubmed-hf/tokenizer_config.json
```

This wraps the trained `custom-med` tokenizer into HuggingFace format so it can be loaded with `PreTrainedTokenizerFast`.

Use the explicit `--tokenizer-json` and `--out` arguments so the script reads the trained medical tokenizer rather than relying on older defaults.

---

## 🖥️ Phase 4 — Terminal comparison (5 min)

Quick look at all 4 tokenizers side by side:

```bash
uv run python scripts/compare_tokenizers.py --heldout-medical data/pubmed_heldout.jsonl --heldout-general data/general_heldout.jsonl --general-bpe artifacts/general-bpe/tokenizer.json
```

You'll see a fertility table. **Don't interpret the numbers yet** — do the notebook first, then come back to explain what you see.

---

Find the knee of the fertility curve: the smallest vocabulary size that gets close to the best compression result. For this PubMed corpus (<1B tokens), the recommended band is usually 16k-32k rather than the 50k "modern LLM default".

What this command does in simple terms:

- It trains or loads several versions of the medical tokenizer at different vocab sizes such as 16k, 32k, 50k, 64k, and 100k.
- It then tests each one on held-out medical text and measures how many tokens are needed on average.
- Fewer tokens means better compression.
- The goal is to find the smallest vocab size that gets close to the best result, instead of blindly choosing a huge vocabulary.

```bash
uv run python .\scripts\sweep_vocab_size.py --corpus .\data\pubmed_heldout.jsonl --heldout .\data\pubmed_heldout.jsonl --no-qwen
```
## 📓 Phase 5 — Notebook lab (45 min)

```bash
jupyter lab notebooks/04_custom_vs_general.ipynb
```

Run every cell in order. For each cell, answer the question below before moving on.

| Cell | Question to answer in your own words |
|------|--------------------------------------|
| 1 — Imports | Did everything load? If not, which library is missing? |
| 2 — Load 4 tokenizers | Which has the largest vocabulary? Which is smallest? |
| 3 — Single sentence comparison | Count the pieces for `acetylcholinesterase` in each tokenizer. Write down the 4 numbers. |
| 4 — Vocab membership | Which medical terms appear as a **single token** only in `custom-med`? |
| 5 — Fertility on held-out | Which tokenizer wins on medical text? Does the result surprise you? |
| 6 — 2×2 chart | `custom-med` **loses** on general text. Is this a bug or a feature? Explain. |
| 7 — Exercises | Write full answers (see below) |

### ✍️ Written exercises (answer in notebook markdown cells)

**Exercise 1:**  
`custom-med` beats `cl100k` on medical text. `cl100k` has 6× the vocabulary size. Explain in one sentence why `custom-med` still wins.

**Exercise 2:**  
`general-bpe` has the exact same vocab size (16k) and the exact same algorithm as `custom-med`, but scores worse on medical text. What does this prove about the cause of `custom-med`'s win?

**Exercise 3:**  
Would training `custom-med` on 90k abstracts instead of 45k make fertility better? Why/why not?

**Exercise 4 (stretch):**  
What would happen to the fertility numbers if we used `--vocab-size 32000` for `custom-med`? Would it beat `cl100k` by a larger margin? Where does this improvement come from — and where does it stop?

---

## 📊 Phase 6 — The fairness numbers (class discussion)

After the notebook, the instructor will draw this on the board. Make sure you understand each row:

```
Tokenizer       vocab   medical fertility   general fertility
─────────────────────────────────────────────────────────────
custom-med       16k       1.381 ✅ wins      1.812 ❌ worst
general-bpe      16k       1.759 ❌           1.503 ✅
cl100k          ~100k      1.470             1.434
o200k           ~200k      1.439             1.401
```

Key question: **Why does `custom-med` lose badly on general text?**  
Answer: It spent all 16k merge budget on medical morphemes. There's no room left for common English pairs. This *proves* it genuinely specialised — a tokenizer that didn't specialise would show no flip.

---

## ⚠️ Phase 7 — The pretrained-model trap (15 min)

Open the diagram:

```bash
# In VS Code or any markdown viewer:
# docs/diagrams/04-pretrained-trap.md
```

Or read the theory doc:

```bash
cat docs/theory/06-pretrained-model-trap.md
```

On Windows / VS Code:

```powershell
Get-Content docs/theory/06-pretrained-model-trap.md
```

**The question:** You want to LoRA fine-tune LLaMA-3 on medical text. You swap in `custom-med` as the tokenizer. Will it work?

> **Answer:** No. Token IDs in `custom-med` no longer match the embedding rows LLaMA was trained with. Row 4521 in LLaMA's embedding table was trained to represent a specific token — now your new tokenizer sends `empagliflozin` there. LoRA cannot fix this mismatch. The model will produce garbage or never converge.

**The two valid paths:**
1. Keep LLaMA's tokenizer → use `add_tokens()` for new medical terms → `resize_token_embeddings()` → LoRA
2. Train a new LM from scratch using `custom-med` as the tokenizer from day 1

---

## 🚀 Phase 8 — Publish your tokenizer to HuggingFace Hub (10 min)

### 8a · Create your HuggingFace token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → choose **Write** scope → copy the token (starts with `hf_`)

### 8b · Set up `.env`

```bash
cp .env.example .env
# Open .env and replace hf_xxx... with your real token
```

On Windows / VS Code:

```powershell
Copy-Item .env.example .env
```

> ⚠️ Never commit `.env` — it's in `.gitignore` already.

### 8c · Wrap your trained tokenizer for HuggingFace

```bash
uv run python scripts/wrap_medical_tokenizer.py --tokenizer-json artifacts/medical-bpe-pubmed/tokenizer.json --out artifacts/medical-bpe-pubmed-hf
# → artifacts/medical-bpe-pubmed-hf/tokenizer.json
# → artifacts/medical-bpe-pubmed-hf/tokenizer_config.json
```

### 8d · Push to the Hub

```bash
uv run python scripts/push_to_hub.py --tokenizer-dir artifacts/medical-bpe-pubmed-hf --repo-id YOUR_HF_USERNAME/medical-bpe-16k
# Example: --repo-id johndoe/medical-bpe-16k
```

##### Command for 32k tokenizer
```bash
uv run python scripts/push_to_hub.py --tokenizer-dir artifacts/medical-bpe-pubmed-hf --repo-id YOUR_HF_USERNAME/medical-bpe-32k
# Example: --repo-id johndoe/medical-bpe-32k
```

Expected output:
```
creating/verifying repo  johndoe/medical-bpe-16k  (public) ...
uploading 2 file(s) from artifacts/medical-bpe-pubmed-hf ...
✅  done — https://huggingface.co/johndoe/medical-bpe-16k
```

### 8e · Verify it works

```python
from transformers import PreTrainedTokenizerFast

tok = PreTrainedTokenizerFast.from_pretrained("YOUR_HF_USERNAME/medical-bpe-16k")
print(tok.tokenize("acetylcholinesterase inhibitor"))
```

---

## ✅ Phase 9 — Run the test suite (5 min)

```bash
uv run python -m pytest tests/test_medical_compare.py -v
```

This form works cleanly across platforms and avoids the Windows `uv trampoline failed to canonicalize script path` issue.

#### On Windows / VS Code

```bash
uv run python -m pytest tests/test_medical_compare.py -v
```

Each green test = one of the lab's claims is numerically verified on your machine.  
All 12 should pass if all 4 data files exist. You'll see which assertions correspond to which claims.

---

## 📚 Reading map

Work through these in order — each takes 5–10 min:

| When | File | What it covers |
|------|------|---------------|
| Before Phase 3 | `docs/theory/03-custom-domain-tokenizers.md` | How BPE training works |
| Before Phase 5 | `docs/theory/04-metrics.md` | Fertility, single-token rate, fairness design |
| After Phase 6 | `docs/theory/05-why-custom-wins-in-healthcare.md` | Why domain frequency is the mechanism |
| Phase 7 | `docs/theory/06-pretrained-model-trap.md` | Embedding mismatch explained |
| Self-study | `docs/theory/01-why-tokenization-matters.md` | The bigger picture |
| Self-study | `docs/theory/02-general-purpose-tokenizers.md` | How cl100k and o200k work |
| Optional | `docs/theory/07-optional-lora-sft.md` | LoRA + SFT if you want to go further |

---

## 🗂️ Visual diagrams (open in any markdown viewer)

| Diagram | What it shows |
|---------|--------------|
| `docs/diagrams/01-bpe-algorithm.md` | BPE training loop step by step |
| `docs/diagrams/02-fair-comparison.md` | Why we need 4 tokenizers, the isolation argument |
| `docs/diagrams/03-lab-pipeline.md` | Every script and what it produces |
| `docs/diagrams/04-pretrained-trap.md` | Wrong path vs right path when building a medical LLM |
| `docs/diagrams/05-fertility-concept.md` | What fertility means with real numbers |

---

## 🔑 Three things to remember when you leave

1. **BPE is frequency-driven** — train on the wrong corpus, the wrong pairs get merged
2. **Domain beats vocab size** — `general-bpe` (16k) proves it: same size, bigger fertility gap
3. **Tokenizer and model weights are inseparable** — born together, cannot be swapped without retraining
