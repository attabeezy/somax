#!/usr/bin/env python3
"""Benchmark inference performance on edge hardware.

Measures full model inference metrics on the Dell Latitude 7400 (8GB RAM):
- Tokens per second (TPS)
- Inference latency (seconds)
- Memory usage (MB)

Run this locally after exporting to GGUF. Not intended for Colab — use
scripts/benchmark_fertility.py for tokenizer-only evaluation on cloud hardware.

Usage:
    # GGUF model (edge deployment)
    python scripts/benchmark_inference.py --model models/gguf/model-Q4_K_M.gguf --test-file data/akan/twi_tts_test.jsonl

    # HuggingFace model (pre-export validation)
    python scripts/benchmark_inference.py --model checkpoints/variant_D/final/ --huggingface --test-file data/akan/twi_tts_test.jsonl
"""

import argparse
import json
import time
import statistics
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceResult:
    """Results from an inference benchmark run."""

    model_id: str
    tokens_per_second: float
    latency_mean: float
    latency_std: float
    total_tokens: int
    num_samples: int
    memory_mb: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "tokens_per_second": self.tokens_per_second,
            "latency_mean": self.latency_mean,
            "latency_std": self.latency_std,
            "total_tokens": self.total_tokens,
            "num_samples": self.num_samples,
            "memory_mb": self.memory_mb,
        }


def load_test_texts(test_file: Path, max_samples: int = 50) -> list[str]:
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


def get_memory_mb() -> Optional[float]:
    """Get current process RSS memory in MB."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def benchmark_gguf(model_path: Path, texts: list[str]) -> InferenceResult:
    """Benchmark a GGUF model with llama-cpp-python."""
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError("llama-cpp-python required. Install with: pip install llama-cpp-python")

    print(f"Loading GGUF model: {model_path}")
    llm = Llama(model_path=str(model_path), n_ctx=2048, n_threads=4, verbose=False)

    total_tokens = 0
    latencies = []

    print(f"Running inference benchmark on {len(texts)} samples...")
    for text in texts:
        token_count = len(llm.tokenize(text.encode("utf-8")))
        total_tokens += token_count

        start = time.perf_counter()
        llm(text, max_tokens=10, temperature=0.0)
        latencies.append(time.perf_counter() - start)

    return InferenceResult(
        model_id=str(model_path),
        tokens_per_second=total_tokens / sum(latencies) if latencies else 0.0,
        latency_mean=statistics.mean(latencies),
        latency_std=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        total_tokens=total_tokens,
        num_samples=len(texts),
        memory_mb=get_memory_mb(),
    )


def benchmark_huggingface(model_id: str, texts: list[str]) -> InferenceResult:
    """Benchmark a HuggingFace model."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers and torch required. Install with: pip install transformers torch"
        )

    print(f"Loading HuggingFace model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    total_tokens = 0
    latencies = []

    print(f"Running inference benchmark on {len(texts)} samples...")
    for text in texts:
        encodings = tokenizer(text, return_tensors="pt")
        total_tokens += encodings["input_ids"].shape[1]

        start = time.perf_counter()
        with torch.no_grad():
            model.generate(encodings["input_ids"], max_new_tokens=10, do_sample=False)
        latencies.append(time.perf_counter() - start)

    return InferenceResult(
        model_id=model_id,
        tokens_per_second=total_tokens / sum(latencies) if latencies else 0.0,
        latency_mean=statistics.mean(latencies),
        latency_std=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        total_tokens=total_tokens,
        num_samples=len(texts),
        memory_mb=get_memory_mb(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge inference benchmark")
    parser.add_argument(
        "--model", type=str, required=True, help="GGUF path or HuggingFace model ID"
    )
    parser.add_argument(
        "--huggingface", action="store_true", help="Force HuggingFace model loading"
    )
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument(
        "--max-samples", type=int, default=50, help="Max samples (keep low for edge devices)"
    )
    parser.add_argument("--output", type=str, help="Save results to JSON")
    args = parser.parse_args()

    test_file = Path(args.test_file)
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    texts = load_test_texts(test_file, args.max_samples)
    if not texts:
        raise ValueError(f"No texts loaded from {test_file}")

    print(f"Loaded {len(texts)} samples")

    model_path = Path(args.model)
    if not args.huggingface and model_path.exists() and model_path.suffix == ".gguf":
        result = benchmark_gguf(model_path, texts)
    else:
        result = benchmark_huggingface(args.model, texts)

    print(f"\n=== Inference Benchmark Results ===")
    print(f"Model:            {result.model_id}")
    print(f"Tokens/second:    {result.tokens_per_second:.1f}")
    print(f"Latency (mean):   {result.latency_mean:.4f}s ± {result.latency_std:.4f}s")
    print(f"Total tokens:     {result.total_tokens}")
    print(f"Samples:          {result.num_samples}")
    if result.memory_mb:
        print(f"Memory:           {result.memory_mb:.0f} MB")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
