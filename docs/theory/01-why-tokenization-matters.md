# Why tokenization matters

A language model does not read raw characters or whole words directly. It reads **token IDs**.

Each token ID points to one row of the model's embedding table. That row is the model's starting representation for that piece of text.

So before a model can reason about a sentence, the tokenizer has already decided how that sentence will be broken into pieces. That is why tokenization matters so much.

## Start with one simple idea

If a useful medical term is split into many small fragments, the model has to do more work to recover the full meaning.

If that same term is split into one or two larger pieces, the model starts from a cleaner representation.

This does not mean a model can never learn from fragmented pieces. It can. But it usually costs more context, more computation, and more learning effort.

## Split word, split meaning

`empagliflozin` as **one** token → one vector for the drug.

`empagliflozin` as **six** pieces → six vectors. Attention must reassemble “this is one drug” every time the word appears. The model can learn that trick if those splits existed during pretraining. It is wasteful. On a new tokenizer with new IDs, there is nothing to reassemble yet.

In plain language: the more pieces you create, the more reconstruction work you push onto the model.

That is especially important in medicine, where many terms are long, rare, and morphologically complex:

- drug names like `empagliflozin`
- enzyme names like `acetylcholinesterase`
- diagnosis phrases like `ST-elevation myocardial infarction`

General tokenizers often split these into many parts because they were trained on broader text, not dense medical language.

## The window is tokens, not words

Context length is counted in tokens. An 800-word discharge summary might be ~1,400 tokens with a general vocab and ~1,000 with a medical vocab. The extra 400 tokens are **fertility tax**: less room for history, labs, and the actual question.

Inference cost and latency scale with token count. Over-segmentation is slower and more expensive even when quality is unchanged.

This is one of the easiest mistakes for beginners to make: they think in words, but the model pays in tokens.

So when a tokenizer wastes tokens on over-splitting, you lose space in the context window and you also increase compute.

That is why fewer tokens for the same clinical note can matter even before you talk about model accuracy.

## Training a tokenizer is not training a model

Tokenizer training is a **deterministic** statistical pass: count pairs, merge, write a vocab. Model training is gradient descent on embeddings and weights.

HuggingFace’s course: you train a new tokenizer on domain data, then you train (or continue-pretrain) a **model** that uses that tokenizer. This lab stops at the tokenizer and the comparison. Doc `06` covers what happens if you skip the model step.

That distinction matters:

- **tokenizer training** decides how text is split
- **model training** teaches weights what those splits mean

This lab focuses on the first part. It shows that medical text can be segmented more efficiently with a medical tokenizer.

It does **not** mean you can take any pretrained model, swap in the tokenizer, and expect everything to work. That is exactly the trap explained later in doc `06`.
