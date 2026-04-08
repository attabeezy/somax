#!/usr/bin/env python3
"""Train a unified BPE vocabulary for dual-stream tokenization.

Trains a single BPE tokenizer on combined ASR+TTS data so both streams
share a compatible embedding space when fine-tuning the same Llama base
model. Stream-specific token statistics are saved as metadata so the
DualCoreTokenizer knows which tokens are ASR- or TTS-dominant.

Usage:
    python scripts/train_bpe.py --input data/akan/ --output models/tokenizers/ --vocab-size 8000
"""

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Iterator

from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer

# Mirrors download.py DATASET_CONFIGS — must stay in sync.
LANG_FILE_PREFIXES: dict[str, dict[str, str | None]] = {
    "akan":    {"asr": "aka_asr", "tts": "twi_tts"},
    "yoruba":  {"asr": None,      "tts": "yor_tts"},
    "swahili": {"asr": None,      "tts": "swa_tts"},
}


def read_jsonl_texts(jsonl_path: Path) -> Iterator[str]:
    """Yield transcription/text strings from a JSONL file."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                text = item.get("transcription") or item.get("text", "")
                if text:
                    yield text
            except json.JSONDecodeError:
                continue


def collect_token_counts(texts: list[str], tokenizer: Tokenizer) -> Counter:
    """Count token frequencies across a list of texts."""
    counts: Counter = Counter()
    for text in texts:
        encoding = tokenizer.encode(text)
        counts.update(encoding.tokens)
    return counts


def train_unified_tokenizer(
    asr_texts: list[str],
    tts_texts: list[str],
    vocab_size: int,
    output_dir: Path,
    language: str,
) -> dict:
    """Train one BPE tokenizer on combined ASR+TTS data.

    Saves the tokenizer JSON and a stream_token_stats.json file that
    records which tokens are dominant in each stream.

    Args:
        asr_texts: Texts from the ASR (spontaneous) split.
        tts_texts: Texts from the TTS (formal) split.
        vocab_size: Target vocabulary size.
        output_dir: Directory to save outputs.
        language: Language code for metadata.

    Returns:
        Training statistics dict.
    """
    all_texts = asr_texts + tts_texts

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "[PAD]",  # 0 — keep existing order for backward compat
            "[UNK]",  # 1
            "[CLS]",  # 2
            "[SEP]",  # 3
            "[MASK]", # 4
            "<s>",    # 5 — bos_token for causal LM
            "</s>",   # 6 — eos_token for causal LM
            "<pad>",  # 7 — pad_token alias for causal LM
        ],
        show_progress=True,
    )

    print(f"Training unified BPE on {len(all_texts)} samples ({len(asr_texts)} ASR + {len(tts_texts)} TTS)...")
    tokenizer.train_from_iterator(all_texts, trainer=trainer)

    tokenizer_path = output_dir / "unified_tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Unified tokenizer saved to: {tokenizer_path}")

    tokenizer_config = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "[UNK]",
        "model_max_length": 8192,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    config_path = output_dir / "tokenizer_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"Tokenizer config saved to: {config_path}")

    print("Computing per-stream token statistics...")
    asr_counts = collect_token_counts(asr_texts, tokenizer)
    tts_counts = collect_token_counts(tts_texts, tokenizer)

    vocab = tokenizer.get_vocab()
    stream_stats = {}
    for token in vocab:
        asr_freq = asr_counts.get(token, 0)
        tts_freq = tts_counts.get(token, 0)
        total = asr_freq + tts_freq
        if total == 0:
            dominant = "shared"
        elif asr_freq / total > 0.7:
            dominant = "asr"
        elif tts_freq / total > 0.7:
            dominant = "tts"
        else:
            dominant = "shared"
        stream_stats[token] = {
            "asr_freq": asr_freq,
            "tts_freq": tts_freq,
            "dominant": dominant,
        }

    stats_path = output_dir / "stream_token_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stream_stats, f, ensure_ascii=False, indent=2)
    print(f"Stream token stats saved to: {stats_path}")

    dominant_counts = Counter(v["dominant"] for v in stream_stats.values())
    return {
        "vocab_size": len(vocab),
        "total_samples": len(all_texts),
        "asr_samples": len(asr_texts),
        "tts_samples": len(tts_texts),
        "language": language,
        "tokenizer_path": str(tokenizer_path),
        "stream_token_stats_path": str(stats_path),
        "token_dominance": dict(dominant_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train unified BPE vocabulary")
    parser.add_argument("--input", type=str, default="data/akan/")
    parser.add_argument("--output", type=str, default="models/tokenizers/")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--language", type=str, default="akan")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) / args.language
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    prefixes = LANG_FILE_PREFIXES[args.language]
    asr_file = input_dir / f"{prefixes['asr']}_train.jsonl" if prefixes["asr"] else None
    tts_file = input_dir / f"{prefixes['tts']}_train.jsonl"

    asr_texts, tts_texts = [], []

    if asr_file and asr_file.exists():
        asr_texts = list(read_jsonl_texts(asr_file))
        print(f"ASR: {len(asr_texts)} samples from {asr_file}")
    elif asr_file:
        print(f"WARNING: ASR file not found: {asr_file}")
    else:
        print(f"No ASR config for {args.language}, skipping.")

    if tts_file.exists():
        tts_texts = list(read_jsonl_texts(tts_file))
        print(f"TTS: {len(tts_texts)} samples from {tts_file}")
    else:
        print(f"WARNING: TTS file not found: {tts_file}")

    if not asr_texts and not tts_texts:
        raise ValueError("No training data found. Run scripts/download.py first.")

    stats = train_unified_tokenizer(asr_texts, tts_texts, args.vocab_size, output_dir, args.language)

    summary_path = output_dir / "training_stats.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nTraining stats saved to: {summary_path}")
    print(f"Vocab size: {stats['vocab_size']}")
    print(f"Token dominance: {stats['token_dominance']}")
    print("Done.")


if __name__ == "__main__":
    main()
