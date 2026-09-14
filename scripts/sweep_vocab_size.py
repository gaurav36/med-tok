#!/usr/bin/env python3
"""Vocab-size sweep on one medical corpus.

Train (or load) byte-level BPE at 16k / 32k / 50k / 64k / 100k. Measure average
tokens per held-out document. Pick the smallest vocab near the minimum.

Instructor:

  uv run python scripts/sweep_vocab_size.py \\
      --corpus data/pubmed_train.jsonl \\
      --heldout data/pubmed_heldout.jsonl \\
      --no-qwen
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

# Resolve project paths once so the CLI works from any current working directory.
ROOT = Path(__file__).resolve().parents[1]
# Keep the scripts directory importable because this file reuses helpers from sibling scripts.
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compare_tokenizers import (  # noqa: E402
    DOMAIN_TERMS,
    EncodeFn,
    fertility,
    iter_jsonl_texts,
    load_custom_encode,
    load_tiktoken_encode,
    read_lines,
    single_token_rate,
    tokens_per_100_chars,
    try_load_qwen,
)
from train_medical_tokenizer import train_byte_level_bpe  # noqa: E402

console = Console()

# Default vocabulary sizes used to search for the compression "knee".
DEFAULT_SIZES: tuple[int, ...] = (16_000, 32_000, 50_000, 64_000, 100_000)
# Simple embedding-width assumption for estimating vocabulary cost.
D_MODEL = 1024
# Accept vocab sizes that are within 2% of the best compression as "near the knee".
KNEE_THRESHOLD = 0.02

# Lesson table (large-corpus example). Used when sweep artifacts are missing.
WORKED_EXAMPLE: tuple[tuple[int, float], ...] = (
    (16_000, 1250.0),
    (32_000, 1050.0),
    (50_000, 980.0),
    (64_000, 960.0),
    (100_000, 950.0),
)

BAND_SMALL = "16k-32k"
BAND_MEDIUM = "32k-64k"
BAND_LARGE = "50k-100k"


@dataclass
class SweepRow:
    """One vocab size evaluated on a held-out set."""

    # Requested/actual let us show when the trainer saturated before reaching the target size.
    requested: int
    actual: int
    mean_tokens_per_doc: float
    fertility: float = 0.0
    tok_per_100ch: float = 0.0
    single_tok_rate: float = 0.0
    vs_previous_pct: float | None = None
    vs_base_pct: float | None = None
    name: str = ""
    path: Path | None = None
    saturated: bool = False

    def __post_init__(self) -> None:
        # Fill display-friendly defaults after the dataclass is created.
        if not self.name:
            self.name = size_label(self.requested)
        self.saturated = self.actual < self.requested


# ── Pure helpers (notebook + tests import these) ─────────────────────────────


def size_label(vocab_size: int) -> str:
    """16000 → '16k'."""
    # Short labels keep tables compact and match the vocabulary names used in the guide.
    if vocab_size % 1000 == 0:
        return f"{vocab_size // 1000}k"
    return str(vocab_size)


def delta_pct(current: float, previous: float) -> float | None:
    """Percent change from previous to current. Negative = fewer tokens."""
    # Guard against division by zero if the previous metric is missing or degenerate.
    if previous == 0:
        return None
    # Example:
    # previous = 1000, current = 950 -> -5.0%
    # Negative is good here because fewer tokens/doc means better compression.
    return 100.0 * (current - previous) / previous


def recommend_vocab(
    rows: Sequence[tuple[int, float]],
    threshold: float = KNEE_THRESHOLD,
) -> int:
    """Smallest vocab whose avg tokens/doc is within threshold of the best (lowest)."""
    if not rows:
        raise ValueError("rows must be non-empty")
    # Lower avg tokens/doc means the tokenizer compressed the held-out corpus better.
    best = min(avg for _, avg in rows)
    # Keep every vocab size that is "close enough" to the best score.
    near = [(size, avg) for size, avg in rows if avg <= best * (1.0 + threshold)]
    # Pick the smallest near-best vocab to avoid paying for unnecessary embedding rows.
    return min(size for size, _ in near)


def corpus_band(n_tokens: int) -> str:
    """Recommended vocab band from estimated train-set token count."""
    # This is a teaching heuristic, not a law of tokenizer training.
    # Example:
    # ~400M train tokens -> 16k-32k
    # ~5B train tokens   -> 32k-64k
    if n_tokens < 1_000_000_000:
        return BAND_SMALL
    if n_tokens < 50_000_000_000:
        return BAND_MEDIUM
    return BAND_LARGE


def embedding_params(vocab_size: int, d_model: int = D_MODEL) -> int:
    """Input-embedding (or output softmax) parameter count: vocab × d_model."""
    # Example: 16k vocab × 1024 dims = 16,384,000 embedding parameters.
    return vocab_size * d_model


def sweep_json_path(root: Path, vocab_size: int) -> Path:
    """artifacts/medical-bpe-pubmed-{16k,...}/tokenizer.json"""
    # Example: 32000 -> artifacts/medical-bpe-pubmed-32k/tokenizer.json
    return root / "artifacts" / f"medical-bpe-pubmed-{size_label(vocab_size)}" / "tokenizer.json"


def alias_16k_path(root: Path) -> Path:
    """Existing lab artifact used as the 16k sweep alias."""
    # The original 16k lab artifact predates the sweep naming pattern.
    return root / "artifacts" / "medical-bpe-pubmed" / "tokenizer.json"


def resolve_existing_json(root: Path, vocab_size: int) -> Path | None:
    """Prefer labeled sweep dir; 16k may fall back to the lab alias."""
    # The lab keeps the main 16k tokenizer in a legacy location, so accept both names.
    labeled = sweep_json_path(root, vocab_size)
    # First prefer the size-labeled directory created by this sweep script.
    if labeled.is_file():
        return labeled
    if vocab_size == 16_000:
        alias = alias_16k_path(root)
        # Then fall back to the older 16k lab artifact if it exists.
        if alias.is_file():
            return alias
    return None


def train_or_load(
    corpus: Path,
    output: Path,
    vocab_size: int,
    skip_existing: bool = True,
    min_frequency: int = 2,
) -> Path:
    """Train BPE unless skip_existing and tokenizer.json already exists."""
    output = Path(output)
    # Reuse a previously trained artifact when the caller allows it.
    if skip_existing and output.is_file():
        return output
    # Otherwise train a fresh tokenizer into the requested output path.
    # Example:
    # vocab_size=32000 -> train a 32k tokenizer and save it at the chosen output path.
    train_byte_level_bpe(
        corpus, output, vocab_size=vocab_size, min_frequency=min_frequency
    )
    return output


def evaluate_tokenizer(
    tokenizer_json: Path,
    texts: Sequence[str],
    requested: int,
    terms: Sequence[str] = DOMAIN_TERMS,
) -> SweepRow:
    """Mean tokens/doc, fertility, tok/100ch, single-token rate, actual vocab."""
    from tokenizers import Tokenizer

    # Load the tokenizer twice for two different jobs:
    # - Tokenizer.from_file(...) gives us the actual learned vocab size
    # - load_custom_encode(...) gives us a simple encode function for metrics
    # Example:
    # a requested 32k tokenizer may actually contain fewer tokens if training saturates early.
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    encode = load_custom_encode(tokenizer_json)
    return evaluate_encode(
        name=size_label(requested),
        encode=encode,
        texts=texts,
        requested=requested,
        actual=tokenizer.get_vocab_size(),
        terms=terms,
        path=tokenizer_json,
    )


def evaluate_encode(
    name: str,
    encode: EncodeFn,
    texts: Sequence[str],
    requested: int,
    actual: int,
    terms: Sequence[str] = DOMAIN_TERMS,
    path: Path | None = None,
) -> SweepRow:
    """Evaluate any encode fn (custom BPE or production baseline)."""
    # Materialize token pieces once so all metrics below use the same tokenization result.
    # Example:
    # "blood pressure is high" -> ["blood", " pressure", " is", " high"]
    pieces = [encode(t) for t in texts]
    n_docs = max(1, len(pieces))
    # Mean tokens/doc is the main compression metric used for the sweep table.
    # Example:
    # 5000 documents producing 6,900 total tokens/doc on average -> 1.38 fertility-like compression signal.
    mean_n = sum(len(p) for p in pieces) / n_docs
    return SweepRow(
        requested=requested,
        actual=actual,
        mean_tokens_per_doc=mean_n,
        fertility=fertility(pieces, texts) if texts else 0.0,
        tok_per_100ch=tokens_per_100_chars(pieces, texts) if texts else 0.0,
        single_tok_rate=single_token_rate(encode, terms),
        name=name,
        path=path,
    )


def attach_deltas(rows: Sequence[SweepRow]) -> list[SweepRow]:
    """Fill vs-previous and vs-first-row percent columns."""
    out = list(rows)
    if not out:
        return out
    base = out[0].mean_tokens_per_doc
    prev: float | None = None
    for i, row in enumerate(out):
        # Compare each row both to the immediately previous size and to the 16k baseline.
        # Example:
        # 16k -> 1000 tokens/doc, 32k -> 950 tokens/doc
        # vs_previous = -5.0%, vs_16k = -5.0%
        row.vs_previous_pct = None if prev is None else delta_pct(row.mean_tokens_per_doc, prev)
        row.vs_base_pct = None if i == 0 else delta_pct(row.mean_tokens_per_doc, base)
        prev = row.mean_tokens_per_doc
    return out


def estimate_corpus_tokens(path: Path, max_docs: int = 50_000) -> int:
    """Whitespace-word proxy for train-set size (not model tokens)."""
    if not path.is_file():
        return 0
    # JSONL corpora store text inside records; plain-text corpora use one example per line.
    if path.suffix == ".jsonl":
        texts = iter_jsonl_texts(path, max_docs=max_docs)
    else:
        texts = read_lines(path)
    # This is only a rough corpus-size estimate for the rule-of-thumb footer.
    # Example:
    # "high blood pressure" counts as 3 whitespace words here.
    # This is simpler than true tokenizer counting, but good enough for the footer heuristic.
    return sum(max(1, len(t.split())) for t in texts)


def load_heldout_texts(path: Path, max_docs: int) -> list[str]:
    """JSONL abstracts or plain-text lines."""
    # Keep this helper separate so the rest of the script does not care about file format.
    if path.suffix == ".jsonl":
        return iter_jsonl_texts(path, max_docs=max_docs)
    return read_lines(path)


def worked_example_rows() -> list[SweepRow]:
    """Lesson numbers so class can discuss flattening without trained artifacts."""
    # These are teaching numbers used only when real trained artifacts are unavailable.
    rows = [
        SweepRow(
            requested=size,
            actual=size,
            mean_tokens_per_doc=avg,
            name=size_label(size),
        )
        for size, avg in WORKED_EXAMPLE
    ]
    return attach_deltas(rows)


def load_available_sweep_rows(
    root: Path,
    texts: Sequence[str],
    sizes: Sequence[int] = DEFAULT_SIZES,
) -> tuple[list[SweepRow], bool]:
    """Load trained artifacts. If none exist, return the worked example.

    Returns (rows, used_example).
    """
    if not texts:
        return worked_example_rows(), True
    rows: list[SweepRow] = []
    for size in sizes:
        # Skip missing artifacts instead of failing so the notebook can still show partial results.
        # Example:
        # if only 16k and 32k tokenizers exist, the table will still render those two rows.
        json_path = resolve_existing_json(root, size)
        if json_path is None:
            continue
        rows.append(evaluate_tokenizer(json_path, texts, requested=size))
    if not rows:
        return worked_example_rows(), True
    return attach_deltas(rows), False


# ── Display ──────────────────────────────────────────────────────────────────


def _fmt_pct(value: float | None) -> str:
    # Use an em dash when no comparison exists yet, such as the first row in a table.
    if value is None:
        return "—"
    return f"{value:+.1f}%"


def print_sweep_table(rows: Sequence[SweepRow], title: str) -> None:
    # Rich tables make the compression tradeoff easy to read in a terminal session.
    # Example row meaning:
    # 32k | 32,000 | 31,842 | 945.2 | -5.4% | -5.4% | ...
    # This means the 32k tokenizer actually learned 31,842 tokens and improved compression by 5.4% vs 16k.
    table = Table(title=title, box=box.SIMPLE_HEAD)
    table.add_column("vocab")
    table.add_column("requested", justify="right")
    table.add_column("actual", justify="right")
    table.add_column("avg tokens/doc", justify="right")
    table.add_column("vs previous", justify="right")
    table.add_column("vs 16k", justify="right")
    table.add_column("fertility", justify="right")
    table.add_column("tok/100ch", justify="right")
    table.add_column("single-tok", justify="right")
    for row in rows:
        actual = str(row.actual)
        # Mark saturated rows so students notice when requested vocab was larger than usable merges.
        if row.saturated:
            actual = f"{row.actual}*"
        table.add_row(
            row.name,
            f"{row.requested:,}",
            actual,
            f"{row.mean_tokens_per_doc:,.1f}",
            _fmt_pct(row.vs_previous_pct),
            _fmt_pct(row.vs_base_pct),
            f"{row.fertility:.3f}",
            f"{row.tok_per_100ch:.2f}",
            f"{row.single_tok_rate:.0%}",
        )
    console.print(table)


def print_embedding_table(sizes: Sequence[int], d_model: int) -> None:
    # Show the parameter cost of larger vocabularies, not just their compression benefit.
    # Example:
    # going from 16k to 100k with d_model=1024 multiplies embedding rows by about 6.25x.
    base = embedding_params(sizes[0], d_model) if sizes else 1
    table = Table(
        title=f"Embedding cost  (params ≈ vocab × d_model={d_model})",
        box=box.SIMPLE_HEAD,
    )
    table.add_column("vocab")
    table.add_column("embedding params", justify="right")
    table.add_column("vs smallest", justify="right")
    for size in sizes:
        n = embedding_params(size, d_model)
        table.add_row(size_label(size), f"{n:,}", f"{n / base:.2f}×")
    console.print(table)


def print_footer(
    rows: Sequence[SweepRow],
    n_train_tokens: int,
    d_model: int,
) -> None:
    # Summarize the best practical choice after the detailed metric tables above.
    # Example:
    # if 32k is within 2% of the best score and 64k is only slightly better,
    # the footer recommends 32k as the cheaper practical choice.
    pairs = [(r.requested, r.mean_tokens_per_doc) for r in rows]
    knee = recommend_vocab(pairs)
    band = corpus_band(n_train_tokens) if n_train_tokens else BAND_SMALL
    console.print(
        f"\n[bold]Knee / rule of thumb:[/] smallest vocab within "
        f"{KNEE_THRESHOLD:.0%} of best avg tokens/doc → [green]{size_label(knee)}[/]"
    )
    if n_train_tokens:
        console.print(
            f"[bold]Corpus-size rule:[/] ~{n_train_tokens:,} whitespace-word tokens "
            f"→ band [cyan]{band}[/]"
        )
        if band == BAND_SMALL and knee > 32_000:
            console.print(
                "[yellow]Mismatch:[/] small corpus (<1B) usually wants 16k–32k. "
                "A bigger knee is extra embedding rows for little compression."
            )
    saturated = [r for r in rows if r.saturated]
    if saturated:
        labels = ", ".join(r.name for r in saturated)
        console.print(
            f"[yellow]Saturation:[/] actual vocab < requested at {labels}. "
            "BPE vocab_size is a max; min_frequency=2 starved the rest of the budget."
        )
    print_embedding_table([r.requested for r in rows], d_model)
    console.print(
        "[dim]Rule of thumb: choose the smallest vocabulary that achieves "
        "near-minimum tokenization length. Extra vocab = extra parameters "
        "in both the input embedding and the output softmax.[/]\n"
    )


def print_baseline_table(rows: Sequence[SweepRow]) -> None:
    if not rows:
        return
    # This second table answers: how close is our custom medical tokenizer to real production tokenizers?
    table = Table(
        title="Production baselines (not trained on this corpus)",
        box=box.SIMPLE_HEAD,
    )
    table.add_column("tokenizer")
    table.add_column("vocab", justify="right")
    table.add_column("avg tokens/doc", justify="right")
    table.add_column("fertility", justify="right")
    for row in rows:
        table.add_row(
            row.name,
            f"{row.actual:,}",
            f"{row.mean_tokens_per_doc:,.1f}",
            f"{row.fertility:.3f}",
        )
    console.print(table)


def tiktoken_vocab_size(encoding: str) -> int:
    import tiktoken

    # Ask tiktoken for the actual size instead of relying on a rough label like "~100k".
    return tiktoken.get_encoding(encoding).n_vocab


def collect_baselines(
    texts: Sequence[str],
    include_qwen: bool,
) -> list[SweepRow]:
    rows: list[SweepRow] = []
    # Compare the custom sweeps to strong production tokenizers trained elsewhere.
    # Example:
    # cl100k_base is a general-purpose tokenizer, so it may be larger but still worse on PubMed terms.
    for name, enc_name, approx in (
        ("cl100k", "cl100k_base", 100_256),
        ("o200k", "o200k_base", 200_019),
    ):
        encode = load_tiktoken_encode(enc_name)
        actual = tiktoken_vocab_size(enc_name)
        rows.append(
            evaluate_encode(name, encode, texts, requested=approx, actual=actual)
        )
    if include_qwen:
        qwen = try_load_qwen()
        if qwen is not None:
            from transformers import AutoTokenizer

            # Load the Hugging Face tokenizer only to report its actual vocabulary size.
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", use_fast=True)
            rows.append(
                evaluate_encode(
                    "qwen", qwen, texts, requested=len(tok), actual=len(tok)
                )
            )
    return rows


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_sizes(raw: str) -> list[int]:
    # Allow --sizes 16000,32000,... on the CLI instead of repeating the flag many times.
    # Example:
    # --sizes 16000,32000,64000 -> [16000, 32000, 64000]
    sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("need at least one vocab size")
    return sizes


def main(argv: Sequence[str] | None = None) -> None:
    # The CLI lets the same script work for the class defaults and for custom experiments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "data" / "pubmed_train.jsonl",
    )
    parser.add_argument(
        "--heldout",
        type=Path,
        default=ROOT / "data" / "pubmed_heldout.jsonl",
    )
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default=list(DEFAULT_SIZES),
        help="comma-separated vocab sizes (default: 16000,32000,50000,64000,100000)",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse tokenizer.json if present (default: true)",
    )
    parser.add_argument("--no-qwen", action="store_true")
    parser.add_argument("--d-model", type=int, default=D_MODEL)
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args(argv)

    # Normalize all incoming paths once so the rest of the script can use absolute paths.
    # Example:
    # --corpus .\data\pubmed_train.jsonl -> C:\...\data\pubmed_train.jsonl
    root = args.root.expanduser().resolve()
    corpus = args.corpus.expanduser().resolve()
    heldout = args.heldout.expanduser().resolve()
    sizes: list[int] = list(args.sizes)

    # Build or reuse one tokenizer artifact per requested vocabulary size.
    json_paths: dict[int, Path] = {}
    for size in sizes:
        labeled = sweep_json_path(root, size)
        existing = resolve_existing_json(root, size) if args.skip_existing else None
        if existing is not None:
            console.print(f"[dim]reuse {size_label(size)}: {existing}[/]")
            json_paths[size] = existing
            continue
        if not corpus.is_file():
            raise FileNotFoundError(
                f"corpus not found: {corpus}. Train with "
                "scripts/train_medical_tokenizer.py or pass --corpus."
            )
        console.print(f"[cyan]train {size_label(size)} → {labeled}[/]")
        # Example:
        # size=50000 -> train a 50k tokenizer and save it under artifacts/medical-bpe-pubmed-50k/
        json_paths[size] = train_or_load(
            corpus=corpus,
            output=labeled,
            vocab_size=size,
            skip_existing=False,
            min_frequency=args.min_frequency,
        )

    # Fall back to the hard-coded lesson table if the held-out set is missing.
    if not heldout.is_file():
        console.print(f"[yellow]held-out not found: {heldout} — using worked example[/]")
        rows = worked_example_rows()
        print_sweep_table(
            rows,
            "Worked example (large-corpus lesson numbers — not this PubMed run)",
        )
        print_footer(rows, n_train_tokens=0, d_model=args.d_model)
        return

    texts = load_heldout_texts(heldout, max_docs=args.max_docs)
    console.print(f"[dim]held-out: {len(texts)} docs from {heldout}[/]")

    # Evaluate every requested tokenizer on the same held-out text slice.
    # This keeps the comparison fair because every tokenizer sees the exact same evaluation set.
    rows = attach_deltas(
        [
            evaluate_tokenizer(json_paths[size], texts, requested=size)
            for size in sizes
        ]
    )
    print_sweep_table(rows, "Vocab-size sweep — avg tokens / held-out document")
    print_baseline_table(collect_baselines(texts, include_qwen=not args.no_qwen))

    # Estimate corpus size only for the rule-of-thumb footer; this is not used in scoring.
    n_train = estimate_corpus_tokens(corpus) if corpus.is_file() else 0
    print_footer(rows, n_train_tokens=n_train, d_model=args.d_model)


if __name__ == "__main__":
    main()
