# The pretrained-model trap

> ⚠️ **Visual trap diagram:** [Wrong vs right paths → docs/diagrams/04-pretrained-trap.md](../diagrams/04-pretrained-trap.md)

**Board sentence.** Tokenizer IDs must match embedding rows. Custom BPE changes the alphabet. A pretrained model’s embeddings still speak the old alphabet.

## Predict, then reveal

Prompt: *Your custom medical tokenizer causes fertility to drop by 20% on discharge summaries compared with the base Llama tokenizer. You now want to LoRA fine-tune Llama-8B on medical text. Should you swap in the custom tokenizer?*

The tempting answer is **yes**:

- better tokenizer on medical text
- better tokenization should help the model
- LoRA will adapt the rest

That answer is **wrong**.

**Reveal:** do **not** replace the tokenizer of a pretrained model.

Why not? Because replacing `tokenizer.json` changes the meaning of the token IDs, but it does **not** change what the pretrained embedding rows already mean.

Example:

- your new tokenizer may output ID 128 for a medical merge
- Llama's embedding row 128 still means whatever Llama was originally trained to store at ID 128

So the model receives the wrong representation from the very first lookup. LoRA on `q_proj` / `v_proj` cannot cheaply repair a broken vocabulary-to-embedding mapping. Training loss may move. Model quality will not improve in a reliable way.

That is **not** the same problem as “the tokenizer makes slightly awkward splits.” Models can often tolerate imperfect splits **if those same splits existed during pretraining**. A full ID remapping is a much more serious failure.

## Worked example

Imagine the base model was pretrained with this meaning table:

- token ID 1287 = ` patient`
- token ID 4521 = ` the`
- token ID 9102 = ` diagnosis`

During pretraining, embedding row 4521 learned the statistical meaning of ` the`. Every later layer expects row 4521 to carry that meaning.

Now you swap in your custom medical tokenizer. In that tokenizer, ID 4521 might now mean `empagliflozin` or `gluc` or some other PubMed merge.

So at inference time, the pipeline becomes:

1. Your new tokenizer sees `empagliflozin` and emits ID 4521.
2. The pretrained model looks up embedding row 4521.
3. But row 4521 still contains the representation for ` the`, because that is what the model was trained on.
4. The rest of the network receives the wrong meaning from the very first lookup.

This is why the problem is deeper than tokenization quality. The model is not merely getting a less efficient split. It is getting the **wrong symbol** for that split.

## Why LoRA does not rescue it

LoRA updates a small subset of weights. It is useful when the model's existing token IDs already line up with its embedding table and you only need to adapt behavior.

Here, the failure happens before those adapter layers can help:

1. The tokenizer emits an ID with a new meaning.
2. The embedding lookup maps that ID to an old meaning.
3. Every attention and MLP block downstream starts from corrupted inputs.

LoRA can adjust how the network reacts to inputs it understands. It cannot cheaply relearn an entire vocabulary-to-embedding alignment that was broken at the input layer.

## Safe mental model

Treat a pretrained tokenizer and a pretrained embedding matrix as two halves of the same table:

- the tokenizer decides which integer represents each string
- the embedding table decides what each integer means

If you replace one half and keep the other, the table is broken.

## Safe paths (not this lab)

1. **Train or continue-pretrain a model with the custom tokenizer from the start.**

	This is the cleanest path. Your medical tokenizer defines the IDs, and the model learns embeddings for those IDs during pretraining.

	Example:
	- build `custom-med` on PubMed
	- initialize a fresh LM with vocab size 16k
	- pretrain the LM using that tokenizer
	- optionally do SFT or LoRA later

	Why this works: the tokenizer vocabulary and embedding table are created together, so every ID means the same thing to both.

2. **Keep the pretrained model's tokenizer and extend it with a small number of new domain terms.**

	This is the normal production path when you want to adapt Llama or Qwen without retraining the whole model.

	Example:

	```python
	tok = AutoTokenizer.from_pretrained("meta-llama/..." )
	model = AutoModelForCausalLM.from_pretrained("meta-llama/..." )

	new_terms = ["empagliflozin", "acetylcholinesterase", "myocardial"]
	tok.add_tokens(new_terms)
	model.resize_token_embeddings(len(tok))
	```

	Then you fine-tune. Old token IDs still point to the same old rows. Only the newly added rows must learn from scratch.

	Why this works: you did **not** replace the tokenizer's existing alphabet. You only appended a few new entries.

3. **Use a custom tokenizer only as a discovery tool, then add selected strings to the base tokenizer.**

	This is a more advanced version of move 2.

	Example:
	- train a medical tokenizer on PubMed
	- find terms that it encodes as 1 token but Llama splits into 4 or 5 pieces
	- add only those high-value strings to the original Llama tokenizer
	- resize embeddings and fine-tune

	Why this works: the pretrained model still keeps its original ID-to-row mapping. You are using the custom tokenizer for analysis, not for direct replacement.

## Unsafe path (this lab’s anti-pattern)

```python
# WRONG — do not do this
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tok = AutoTokenizer.from_pretrained("artifacts/medical-bpe-hf")
# then train / LoRA as if this were compatible
```

This looks attractive because the tokenizer is better on medical text by itself. But it breaks the pretrained model immediately.

Why it is unsafe:

1. `Qwen/Qwen3-0.6B` was trained with Qwen's own tokenizer.
2. Its embedding row 0, row 1, row 2, ... already have fixed meanings from that tokenizer.
3. Your medical tokenizer assigns different meanings to those same integer IDs.
4. The model now reads the wrong embedding for almost every token.

So the failure is not subtle. It is not "slightly worse tokenization." It is a broken interface between tokenizer and model weights.

### Quick rule to remember

- **Safe:** keep the old IDs and optionally add new ones
- **Unsafe:** replace the old IDs with a completely new vocabulary

## What this lab grades

You can explain the chips and the fertility table. You can say the board sentence. You do **not** ship a Qwen adapter with a replaced tokenizer.
