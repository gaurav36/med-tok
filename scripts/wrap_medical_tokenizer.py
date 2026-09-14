#!/usr/bin/env python3
"""Wrap tokenizer.json as a HuggingFace PreTrainedTokenizerFast directory.

Run: uv run python scripts/wrap_medical_tokenizer.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "artifacts" / "medical-bpe" / "tokenizer.json"
DEFAULT_OUT = ROOT / "artifacts" / "medical-bpe-hf"


"""
Wrap a raw `tokenizer.json` file into a tokenizer directory.

The input `tokenizer.json` already contains the learned tokenizer model. This helper
loads that file into `PreTrainedTokenizerFast`, attaches Hugging Face-style special
token settings, and then saves the result in the standard directory format.
"""
def wrap_tokenizer(
    tokenizer_json: Path,
    output_dir: Path,
    eos_token: str = "<|endoftext|>",
    pad_token: str = "<pad>",
) -> PreTrainedTokenizerFast:
    """Load tokenizer.json and save_pretrained to output_dir."""
    # Fail early if the input tokenizer artifact is missing.
    if not tokenizer_json.is_file():
        raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_json}")

    # Load the raw tokenizer.json into Hugging Face's fast-tokenizer wrapper.
    # `tokenizer_file` points at the tokenizers library artifact we trained earlier.
    hf_tok = PreTrainedTokenizerFast(
        # Read the existing tokenizer definition from disk.
        tokenizer_file=str(tokenizer_json),
        # We are not defining a beginning-of-sequence token for this tokenizer.
        bos_token=None,
        # Register the end-of-sequence token so Hugging Face knows which string marks sequence end.
        eos_token=eos_token,
        # Register the padding token so batched inputs can be padded consistently.
        pad_token=pad_token,
        # We are not defining a separate unknown token here.
        unk_token=None,
    )

    # Ensure the target folder exists before writing tokenizer files into it.
    output_dir.mkdir(parents=True, exist_ok=True)

    # `save_pretrained` writes Hugging Face tokenizer files into `output_dir`.
    # In this project that includes:
    # - tokenizer.json: the serialized tokenizer model and processing pipeline
    # - tokenizer_config.json: Hugging Face metadata such as special-token settings
    # The method may also write extra files in other setups, but these two are the key
    # files you see in this repository.
    hf_tok.save_pretrained(output_dir)

    # Return the wrapped tokenizer so callers can inspect it or use it immediately.
    return hf_tok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    hf_tok = wrap_tokenizer(
        args.tokenizer_json.expanduser().resolve(),
        args.out.expanduser().resolve(),
    )
    print(f"saved {args.out}  size={len(hf_tok)}")


if __name__ == "__main__":
    main()
