#!/usr/bin/env python3
"""Staged LoRA training for dual-stream tokenization.

Implements all 5 experimental groups defined in configs/variants.yaml:
- Control: Standard Llama-3.2-1B (no fine-tuning)
- Variant A: ASR only
- Variant B: TTS only
- Variant C: ASR + TTS (mixed)
- Variant D: TTS -> ASR -> TTS (primary hypothesis)
- Variant E: ASR -> TTS

Usage:
    python scripts/train_lora.py --group D --data data/akan/ --output checkpoints/
    python scripts/train_lora.py --group D --config configs/variants.yaml --data data/akan/

Designed for Colab T4 GPU execution.
"""

import argparse
import json
from pathlib import Path
from typing import Literal
from dataclasses import dataclass

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType


TrainingGroup = Literal["control", "A", "B", "C", "D", "E"]


@dataclass
class TrainingConfig:
    """Configuration for a single training stage."""

    name: str
    data_split: Literal["asr", "tts", "mixed"]
    learning_rate: float
    epochs: int
    batch_size: int = 4
    max_length: int = 512


def load_variant_configs(config_path: Path, group: str) -> tuple[list[TrainingConfig], dict]:
    """Load variant stage configs and model settings from YAML.

    Args:
        config_path: Path to variants.yaml.
        group: Variant key (e.g. 'D').

    Returns:
        Tuple of (stage configs list, lora settings dict).
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    variant = raw["variants"].get(group)
    if variant is None:
        raise ValueError(f"Unknown variant '{group}'. Available: {list(raw['variants'].keys())}")

    stages = [
        TrainingConfig(
            name=s["name"],
            data_split=s["data_split"],
            learning_rate=float(s["learning_rate"]),
            epochs=int(s["epochs"]),
            batch_size=int(s.get("batch_size", 4)),
            max_length=int(s.get("max_length", 512)),
        )
        for s in variant.get("stages", [])
    ]

    lora_cfg = raw.get("model", {}).get("lora", {})
    return stages, lora_cfg


def load_jsonl_texts(jsonl_path: Path) -> list[str]:
    """Load texts from JSONL file."""
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                text = item.get("transcription") or item.get("text", "")
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue
    return texts


def prepare_dataset(texts: list[str], tokenizer, max_length: int = 512) -> Dataset:
    """Prepare HuggingFace Dataset from a list of texts."""

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    dataset = Dataset.from_dict({"text": texts})
    return dataset.map(tokenize_function, batched=True, remove_columns=["text"])


def train_stage(
    model, tokenizer, texts: list[str], config: TrainingConfig, output_dir: Path, stage_num: int
) -> None:
    """Execute a single training stage."""
    print(f"\n=== Stage {stage_num}: {config.name} ===")
    print(f"  Data split: {config.data_split}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Samples: {len(texts)}")

    dataset = prepare_dataset(texts, tokenizer, config.max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir / f"stage_{stage_num}_{config.name}"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    print(f"Stage {stage_num} complete.")


def initialize_embeddings_from_llama(
    model,
    waxal_tokenizer: PreTrainedTokenizerFast,
    llama_tokenizer,
    llama_embed_snapshot: torch.Tensor,
) -> None:
    """Warm-initialize the resized embedding table using Llama token averages.

    For each WAXAL token string, tokenizes it with the Llama tokenizer, retrieves
    the corresponding rows from the pre-resize Llama embedding snapshot, and averages
    them. Tokens that produce empty Llama encodings fall back to the snapshot mean.
    Both embed_tokens and lm_head are initialized identically.

    Args:
        model: The model after resize_token_embeddings() has been called.
        waxal_tokenizer: The loaded WAXAL PreTrainedTokenizerFast (8k vocab).
        llama_tokenizer: The original Llama AutoTokenizer used for lookup only.
        llama_embed_snapshot: Full Llama embedding matrix snapshotted BEFORE resize,
                              shape [128256, hidden_size], on CPU.
    """
    print("Warm-initializing embedding table from Llama token averages...")
    mean_embed = llama_embed_snapshot.mean(dim=0)
    waxal_vocab = waxal_tokenizer.get_vocab()
    fallback_count = 0

    embed_layer = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    dtype = embed_layer.weight.dtype

    with torch.no_grad():
        new_embed = embed_layer.weight.clone()
        new_lm_head = lm_head.weight.clone()

        for token_str, waxal_id in waxal_vocab.items():
            llama_ids = llama_tokenizer.encode(token_str, add_special_tokens=False)
            valid_ids = [i for i in llama_ids if i < llama_embed_snapshot.shape[0]]
            if valid_ids:
                init_vec = llama_embed_snapshot[valid_ids].mean(dim=0)
            else:
                init_vec = mean_embed
                fallback_count += 1
            new_embed[waxal_id] = init_vec.to(dtype)
            new_lm_head[waxal_id] = init_vec.to(dtype)

        embed_layer.weight.copy_(new_embed)
        lm_head.weight.copy_(new_lm_head)

    print(f"  Initialized {len(waxal_vocab)} rows. Fallback to mean: {fallback_count} tokens.")


def train_variant(
    model_id: str,
    data_dir: Path,
    output_dir: Path,
    configs: list[TrainingConfig],
    lora_cfg: dict,
    language: str = "akan",
    tokenizer_path: Path | None = None,
) -> None:
    """Train a variant through one or more staged training runs.

    Handles both single-stage variants (A, B, C) and multi-stage variants
    (D: TTS->ASR->TTS, E: ASR->TTS) with the same code path.

    When tokenizer_path is provided, the custom WAXAL BPE tokenizer replaces
    the Llama tokenizer. Model embeddings are resized and warm-initialized from
    Llama token averages so training converges within the T4 time budget.
    """
    print(f"\nLoading model: {model_id}")

    if tokenizer_path is not None:
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"WAXAL tokenizer not found: {tokenizer_path}")
        print(f"Loading WAXAL tokenizer: {tokenizer_path}")
        llama_tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_path),
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="[UNK]",
        )
        use_custom_tokenizer = True
        print(f"WAXAL vocab size: {len(tokenizer)}")
    else:
        llama_tokenizer = AutoTokenizer.from_pretrained(model_id)
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
        tokenizer = llama_tokenizer
        use_custom_tokenizer = False

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if use_custom_tokenizer:
        # Snapshot FULL Llama embeddings BEFORE resize — resize truncates to 8k rows,
        # making Llama IDs > 7999 unreachable from the weight matrix.
        with torch.no_grad():
            llama_embed_snapshot = model.get_input_embeddings().weight.detach().clone().cpu()

        print(f"Resizing embeddings: {llama_tokenizer.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        initialize_embeddings_from_llama(model, tokenizer, llama_tokenizer, llama_embed_snapshot)

        del llama_embed_snapshot, llama_tokenizer  # free memory before LoRA wrapping

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.1),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        modules_to_save=["embed_tokens", "lm_head"] if use_custom_tokenizer else None,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    _LANG_FILE_PREFIXES = {
        "akan":    {"asr": "aka_asr", "tts": "twi_tts"},
        "yoruba":  {"asr": None,      "tts": "yor_tts"},
        "swahili": {"asr": None,      "tts": "swa_tts"},
    }
    prefixes = _LANG_FILE_PREFIXES.get(language, {"asr": language, "tts": language})
    asr_file = data_dir / f"{prefixes['asr']}_train.jsonl" if prefixes["asr"] else None
    tts_file = data_dir / f"{prefixes['tts']}_train.jsonl"

    asr_texts = load_jsonl_texts(asr_file) if (asr_file and asr_file.exists()) else []
    tts_texts = load_jsonl_texts(tts_file) if tts_file.exists() else []

    if asr_texts:
        print(f"Loaded {len(asr_texts)} ASR samples")
    if tts_texts:
        print(f"Loaded {len(tts_texts)} TTS samples")

    for stage_num, config in enumerate(configs, 1):
        if config.data_split == "asr":
            texts = asr_texts
        elif config.data_split == "tts":
            texts = tts_texts
        else:
            texts = asr_texts + tts_texts

        if not texts:
            raise ValueError(f"No training data found for split '{config.data_split}'")

        train_stage(model, tokenizer, texts, config, output_dir, stage_num)

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nFinal model saved to: {final_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA variants")
    parser.add_argument(
        "--group",
        type=str,
        default="D",
        choices=["control", "A", "B", "C", "D", "E"],
        help="Training variant to execute",
    )
    parser.add_argument("--model", type=str, default=None, help="Override base model ID from config")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to WAXAL unified_tokenizer.json. Resizes model embeddings with warm init from Llama.",
    )
    parser.add_argument("--data", type=str, default="data/akan/")
    parser.add_argument("--language", type=str, default="akan", choices=["akan", "yoruba", "swahili"])
    parser.add_argument("--output", type=str, default="checkpoints/")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/variants.yaml",
        help="Path to variants YAML config",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    output_dir = Path(args.output) / f"variant_{args.group}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    configs, lora_cfg = load_variant_configs(config_path, args.group)

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    model_id = args.model or raw_config.get("model", {}).get("base_id", "meta-llama/Llama-3.2-1B")

    print(f"Variant:    {args.group}")
    print(f"Base model: {model_id}")
    print(f"Data:       {args.data}")
    print(f"Output:     {output_dir}")
    print(f"Stages:     {len(configs) if configs else 'None (control)'}")

    if args.group == "control":
        print("Control group: No training required. Use base model directly.")
        return

    train_variant(
        model_id,
        data_dir,
        output_dir,
        configs,
        lora_cfg,
        language=args.language,
        tokenizer_path=Path(args.tokenizer_path) if args.tokenizer_path else None,
    )
    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()
