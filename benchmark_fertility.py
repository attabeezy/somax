#!/usr/bin/env python3
"""Benchmark token fertility for WAXAL tokenizers.

Measures F = Tokens / Words across tokenizer configurations. No model
loading required — runs fast on any hardware including Colab CPU.

Compares the unified WAXAL tokenizer against a baseline (e.g. Llama-3.2-1B)
to quantify the reduction in the Tokenization Tax.

Usage:
    # Baseline (Llama tokenizer)
    python benchmark_fertility.py --tokenizer meta-llama/Llama-3.2-1B --test-file data/akan/twi_tts_test.jsonl

    # WAXAL unified tokenizer
    python benchmark_fertility.py --tokenizer models/tokenizers/akan/unified_tokenizer.json --waxal --test-file data/akan/twi_tts_test.jsonl

    # Compare both
    python benchmark_fertility.py --tokenizer meta-llama/Llama-3.2-1B --waxal-tokenizer models/tokenizers/akan/unified_tokenizer.json --test-file data/akan/twi_tts_test.jsonl --compare
"""

import argparse
import json
import statistics
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class FertilityResult:
    """Token fertility benchmark results."""

    tokenizer_id: str
    fertility: float
    total_tokens: int
    total_words: int
    num_samples: int
    per_sample_fertilities: list[float]

    @property
    def fertility_std(self) -> float:
        return statistics.stdev(self.per_sample_fertilities) if len(self.per_sample_fertilities) > 1 else 0.0

    def to_dict(self) -> dict:
        return {
            "tokenizer_id": self.tokenizer_id,
            "fertility_mean": self.fertility,
            "fertility_std": self.fertility_std,
            "total_tokens": self.total_tokens,
            "total_words": self.total_words,
            "num_samples": self.num_samples,
        }


def load_test_texts(test_file: Path, max_samples: int = 500) -> list[str]:
    """Load texts from a JSONL test file."""
    texts = []
    with open(test_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                item = json.loads(line.strip())
                text = item.get("transcription") or item.get("text", "")
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue
    return texts


def measure_fertility_hf(tokenizer_id: str, texts: list[str]) -> FertilityResult:
    """Measure fertility using a HuggingFace tokenizer."""
    from transformers import AutoTokenizer

    print(f"Loading HuggingFace tokenizer: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    total_tokens, total_words = 0, 0
    per_sample = []

    for text in texts:
        words = len(text.split())
        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        total_tokens += token_count
        total_words += words
        if words > 0:
            per_sample.append(token_count / words)

    return FertilityResult(
        tokenizer_id=tokenizer_id,
        fertility=total_tokens / total_words if total_words > 0 else 0.0,
        total_tokens=total_tokens,
        total_words=total_words,
        num_samples=len(texts),
        per_sample_fertilities=per_sample,
    )


def measure_fertility_waxal(tokenizer_path: str, texts: list[str]) -> FertilityResult:
    """Measure fertility using the unified WAXAL tokenizer."""
    from transformers import PreTrainedTokenizerFast

    print(f"Loading WAXAL tokenizer: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    total_tokens, total_words = 0, 0
    per_sample = []

    for text in texts:
        words = len(text.split())
        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        total_tokens += token_count
        total_words += words
        if words > 0:
            per_sample.append(token_count / words)

    return FertilityResult(
        tokenizer_id=tokenizer_path,
        fertility=total_tokens / total_words if total_words > 0 else 0.0,
        total_tokens=total_tokens,
        total_words=total_words,
        num_samples=len(texts),
        per_sample_fertilities=per_sample,
    )


def print_result(result: FertilityResult, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}Tokenizer: {result.tokenizer_id}")
    print(f"  Token Fertility (F): {result.fertility:.3f} ± {result.fertility_std:.3f} tokens/word")
    print(f"  Total tokens: {result.total_tokens}  |  Total words: {result.total_words}")
    print(f"  Samples: {result.num_samples}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Token fertility benchmark")
    parser.add_argument("--tokenizer", type=str, help="HuggingFace tokenizer ID (baseline)")
    parser.add_argument("--waxal-tokenizer", type=str, help="Path to unified WAXAL tokenizer JSON")
    parser.add_argument("--waxal", action="store_true", help="Treat --tokenizer as a WAXAL tokenizer path")
    parser.add_argument("--compare", action="store_true", help="Run both and show reduction")
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output", type=str, help="Save results to JSON")
    args = parser.parse_args()

    test_file = Path(args.test_file)
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    texts = load_test_texts(test_file, args.max_samples)
    if not texts:
        raise ValueError(f"No texts loaded from {test_file}")

    print(f"Loaded {len(texts)} samples from {test_file}")

    results = {}

    if args.compare:
        if not args.tokenizer or not args.waxal_tokenizer:
            parser.error("--compare requires both --tokenizer and --waxal-tokenizer")
        baseline = measure_fertility_hf(args.tokenizer, texts)
        waxal = measure_fertility_waxal(args.waxal_tokenizer, texts)
        print_result(baseline, "Baseline")
        print_result(waxal, "WAXAL")
        if baseline.fertility == 0.0:
            print("\n=== ERROR: Baseline fertility is 0 — no words found in test file ===")
            return
        reduction = (baseline.fertility - waxal.fertility) / baseline.fertility * 100
        target_met = reduction >= 30.0
        print(f"\n=== Fertility Reduction: {reduction:.1f}% {'✓ TARGET MET' if target_met else '✗ target: ≥30%'} ===")
        results = {"baseline": baseline.to_dict(), "waxal": waxal.to_dict(), "reduction_pct": reduction}
    elif args.waxal or (args.tokenizer and Path(args.tokenizer).exists()):
        path = args.waxal_tokenizer or args.tokenizer
        result = measure_fertility_waxal(path, texts)
        print_result(result, "WAXAL")
        results = result.to_dict()
    elif args.tokenizer:
        result = measure_fertility_hf(args.tokenizer, texts)
        print_result(result, "Baseline")
        results = result.to_dict()
    else:
        parser.error("Provide --tokenizer or --waxal-tokenizer")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
