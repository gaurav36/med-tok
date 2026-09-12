# Optional extension — LoRA SFT (stock tokenizer)

Not part of notebook 04. Separate session. GPU optional.

This page is about a different goal from the tokenizer lab.

In the main lab, you study how tokenizers split medical text. Here, the goal is to fine-tune a small pretrained language model on a medical question-answering dataset using LoRA.

**Use the base model’s tokenizer.** For this step, you are adapting the model's behavior, not replacing its vocabulary. Do not mix in the custom medical BPE.

## What this page is for

This optional extension shows the next practical step after the tokenizer lab:

- keep a pretrained model such as Qwen
- keep its original tokenizer
- train a lightweight LoRA adapter on MedMCQA

This lets students see the difference between two separate ideas:

- **tokenizer training**: teaching text to be split better
- **SFT with LoRA**: teaching a pretrained model to answer a task better

They are related, but they are not the same operation.

## Install

```bash
uv sync --extra sft
```

Pulls `torch`, `accelerate>=1.14`, `peft>=0.20`, `trl>=1.12`. Default `uv sync` for class laptops stays CPU-only.

You only need these packages if you actually want to run the fine-tuning step. The main tokenizer lab does not require them.

## Shape of the run

At a high level, the run looks like this:

1. load a pretrained base model
2. keep its original tokenizer
3. format MedMCQA into prompt/answer examples
4. train LoRA adapter weights instead of updating the full model
5. save the adapter for later inference

LoRA is useful because it changes only a small number of parameters, so the experiment is much cheaper than full fine-tuning.

```python
from datasets import load_dataset
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# Stock tokenizer. Do not load artifacts/medical-bpe-hf.
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
args = SFTConfig(
    output_dir="artifacts/medmcqa-lora",
    learning_rate=1e-4,
    max_length=1024,
    completion_only_loss=True,
    bf16=True,
)
# Map openlifescienceai/medmcqa to prompt/completion.
# SFTTrainer(..., peft_config=peft_config, processing_class=base_tokenizer)
```

Start with `Qwen/Qwen3-0.6B` on a 16GB GPU. 8B wants QLoRA (`BitsAndBytesConfig` on `SFTTrainer`).

If you are new to SFT, the important idea is simple: the model already knows language. You are only nudging it toward a specific task format, such as multiple-choice medical questions.

## Why stock tokenizer here

SFT teaches the model to answer exam-style items. It does not need a new alphabet.

That is why this page keeps the original tokenizer:

- the pretrained model already matches that tokenizer
- LoRA assumes the model-tokenizer interface is already valid
- changing the tokenizer here would create the mismatch explained in doc `06`

If you later decide that a few medical strings should become single tokens, the safe path is still to extend the base tokenizer with `add_tokens()`, not replace it.

So the order of ideas is:

1. first learn tokenizer behavior in this lab
2. then learn LoRA/SFT with the stock tokenizer
3. only after that consider careful vocabulary extension

## Safety

MedMCQA LoRA is an exam-taking adapter. It is not medical advice and not a clinical system.

Even if the model answers benchmark-style questions well, that does not make it reliable for diagnosis or treatment.
