#!/usr/bin/env python3
"""Train a class-scale byte-level BPE on the authored medical corpus.

Run: uv run python scripts/train_medical_tokenizer.py
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "medical_corpus.txt"
DEFAULT_OUT = ROOT / "artifacts" / "medical-bpe" / "tokenizer.json"
SPECIAL_TOKENS = ["<|endoftext|>", "<pad>"]


"""
Return the repository root used by this training script.

This helper keeps the rest of the file independent from the shell's current working
directory. Other paths in the script are built from this root so the command works the
same way whether it is launched from the repository root or another folder.
"""
def find_root() -> Path:
    #Repo root (directory that contains data/medical_corpus.txt).
    return ROOT


"""
Yield batches of non-empty lines for train_from_iterator.

Plain text: one document per line. JSONL: uses the ``text`` field when present.

Read the corpus file in small batches for tokenizer training.

The trainer accepts an iterator, so this function streams text instead of loading the
full corpus into memory. It skips blank lines, supports plain-text input, and also
accepts JSONL rows when they contain a top-level `text` field.
"""
def iter_corpus_batches(path: Path, batch_size: int = 64) -> Iterator[list[str]]:
    # Collect training examples until we have a full batch ready to yield.
    batch: list[str] = []
    # Open the corpus as UTF-8 text so we can iterate over it one line at a time.
    with path.open(encoding="utf-8") as handle:
        # Read each source line as one potential training example.
        for line in handle:
            # Remove surrounding whitespace and trailing newlines before any further checks.
            raw = line.strip()
            # Skip empty lines because they do not provide any useful training signal.
            if not raw:
                continue
            # Default to treating the cleaned line as plain text.
            text = raw
            # If the line looks like JSON, try to extract its text payload.
            if raw.startswith("{"):
                try:
                    # Import locally because JSON parsing is only needed for JSONL inputs.
                    import json

                    # Parse the JSON object from the current line.
                    obj = json.loads(raw)
                    # Use the top-level `text` field when it exists and is non-empty.
                    if isinstance(obj, dict) and obj.get("text"):
                        # Normalize the extracted value back into a stripped Python string.
                        text = str(obj["text"]).strip()
                except json.JSONDecodeError:
                    # Fall back to the raw text if the line is not valid JSON after all.
                    text = raw
            # Skip lines that become empty after text extraction and normalization.
            if not text:
                continue
            # Add the cleaned example to the current batch.
            batch.append(text)
            # Yield once the requested batch size has been reached.
            if len(batch) >= batch_size:
                # Hand the batch to the trainer and start a fresh list for later lines.
                yield batch
                batch = []
    # Emit any remaining partial batch after the file loop finishes.
    if batch:
        yield batch


def count_lines(path: Path) -> int:
    """Count non-empty lines (progress length for the trainer)."""
    n = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


"""
Build the base byte-level BPE tokenizer configuration.

This function wires together the tokenizer model, byte-level pre-tokenizer, matching
decoder, and post-processor. The offset trimming step matters because byte-level GPT
style markers should not leak into downstream span handling.
"""
def build_byte_level_bpe() -> Tokenizer:
    """Empty byte-level BPE with GPT-style pretokenizer/decoder."""
    # Create a tokenizer object that will learn BPE merges from the training corpus.
    tokenizer = Tokenizer(models.BPE())
    # Split input into byte-level units so any UTF-8 text can be represented safely.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # Reconstruct normal text from byte-level tokens during decoding.
    tokenizer.decoder = decoders.ByteLevel()
    # Example: "blood pressure"
    #
    # One possible tokenization is:
    # token piece   | token id
    # "blood"      | 4123
    # " pressure"  | 9821
    #
    # Notice that the second token piece still stores the leading space.
    # The vocabulary is allowed to store pieces like " pressure", not only plain
    # words like "pressure".
    #
    # What do we actually send to the model?
    # We send only the token IDs: [4123, 9821]
    #
    # When the model output is decoded:
    # - 4123 maps back to "blood"
    # - 9821 maps back to " pressure"
    # - joining those pieces gives "blood pressure"
    #
    # Easy way to think about this: there are TWO different things here.
    # 1. Token value stored in the vocabulary, like " pressure"
    # 2. Offset metadata, like (6, 14)
    #
    # Here "downstream code" means any later code that uses tokenizer spans, for example:
    # - highlighting a word in the original sentence
    # - extracting an entity span from text
    # - mapping model output back to the user-visible text
    #
    # With `trim_offsets=True`:
    # - what we SEND to the model is still the same token IDs, for example [4123, 9821]
    # - the token piece is still " pressure"
    # - only the reported character positions are cleaned up to match visible text
    # - so the offset for " pressure" is reported like (6, 14), which points to "pressure"
    #
    # With `trim_offsets=False`:
    # - what we SEND to the model is still the same token IDs, for example [4123, 9821]
    # - the token piece is still " pressure"
    # - but the reported character positions may include the extra byte-level whitespace area
    # - that can make span-based code point to a less clean range than the user expects
    #
    # So `trim_offsets` changes offset metadata only.
    # It does NOT change the token text, token bytes, token IDs, or decoded final text.
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    # Return the configured tokenizer so the training function can fit it on the corpus.
    return tokenizer


"""
Train a byte-level BPE tokenizer and save the resulting tokenizer artifact.

The `min_frequency` setting controls how rare a pair can be before it is ignored during
merge learning. The initial alphabet includes all byte-level symbols up front so the
tokenizer can safely represent arbitrary UTF-8 input even when some bytes are uncommon
in the training corpus.
"""
def train_byte_level_bpe(
    corpus: Path,
    output: Path,
    vocab_size: int = 16000,
    # minimum frequency for a token to be included in the vocabulary. frequency here means the token must appear at least this many times in the corpus.
    min_frequency: int = 2, # minimum frequency for a token to be included in the vocabulary. frequency here means the token must appear at least this many times in the corpus.
) -> Tokenizer:
    """Train BPE and save tokenizer.json. Returns the trained Tokenizer."""
    # Stop immediately if the caller points to a missing corpus file.
    if not corpus.is_file():
        # Raise a clear error with the bad path so the caller knows what to fix.
        raise FileNotFoundError(f"corpus not found: {corpus}")

    # Start from the shared byte-level tokenizer configuration defined above.
    tokenizer = build_byte_level_bpe()

    # Configure the Rust BPE trainer with the requested vocabulary budget and merge rules.
    trainer = trainers.BpeTrainer(
        # Cap how many learned tokens and merges the trainer is allowed to keep.
        vocab_size=vocab_size,
        # Ignore pairs that are too rare to be useful across the corpus.
        min_frequency=min_frequency,
        # Reserve these tokens explicitly so they are always available after training.
        special_tokens=SPECIAL_TOKENS,
        # include all byte-level characters in the initial alphabet
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        # Let the underlying library show training progress in the terminal.
        show_progress=True,
    )

    # Count documents first so the progress bar has a stable total length.
    length = count_lines(corpus)

    # Feed the trainer batches from the streaming iterator instead of loading the full corpus.
    tokenizer.train_from_iterator(
        # Stream pre-cleaned text batches from disk.
        iter_corpus_batches(corpus),
        # Apply the BPE trainer configuration created above.
        trainer=trainer,
        # Give the trainer the total number of examples for progress reporting.
        length=length,
    )

    # Create the destination folder if it does not already exist.
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save the trained tokenizer as a single tokenizer.json artifact.
    tokenizer.save(str(output))
    # Return the trained tokenizer so callers can inspect or reuse it immediately.
    return tokenizer


"""
Parse command-line arguments and run tokenizer training.

This entry point converts the user-provided paths into absolute paths before training.
That normalization allows `~` in paths and avoids ambiguity about where the tokenizer
artifact should be written.
"""
def main() -> None:
    # Create the CLI parser using the module docstring as the top-level command description.
    parser = argparse.ArgumentParser(description=__doc__)
    # Register the input corpus path argument for the training data file.
    parser.add_argument(
        "--corpus",
        # Parse the provided value directly as a pathlib Path object.
        type=Path,
        # Fall back to the repository's default medical corpus when the user omits this flag.
        default=DEFAULT_CORPUS,
        # Show a short help message in `--help` output.
        help="UTF-8 text, one document per line",
    )
    # Register the output tokenizer path argument.
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    # Register the target vocabulary size for the BPE trainer.
    parser.add_argument("--vocab-size", type=int, default=16000)
    # Register the minimum pair frequency threshold used during merge learning.
    parser.add_argument("--min-frequency", type=int, default=2)
    # Parse the actual command-line values supplied by the user.
    args = parser.parse_args()

    # Run training with normalized absolute paths so downstream code sees consistent locations.
    tokenizer = train_byte_level_bpe(
        # Expand `~` and convert the corpus path into an absolute path.
        args.corpus.expanduser().resolve(),
        # expanduser used for handling '~' in paths. and resolve converts it to an absolute path.
        args.out.expanduser().resolve(),
        # Forward the requested vocabulary size into the training function.
        vocab_size=args.vocab_size,
        # Forward the requested merge-frequency threshold into the training function.
        min_frequency=args.min_frequency,
    )
    # Print the saved path and learned vocabulary size so the caller can confirm the result.
    print(f"saved {args.out}  vocab={tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
