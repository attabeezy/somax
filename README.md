# project-soma

**Eliminating the Tokenization Tax for Twi via Dual-Stream Tokenization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Notebooks

| Notebook | Colab | Kaggle |
|----------|-------|--------|
| `notebooks/train_eval.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/attabeezy/somax/blob/main/notebooks/train_eval.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/attabeezy/somax/blob/main/notebooks/train_eval.ipynb) |

## Overview

Modern LLM tokenizers are optimized for English, resulting in a **"Tokenization Tax"** for African languages like Twi (Akan). This project implements a research-to-production framework that:

- Reduces token fertility (F = tokens/words) by ≥30%
- Uses dual-stream processing for spontaneous speech (ASR) and formal text
- Deploys efficiently on edge devices (8GB RAM)

### Data Sources

| Stream | Source | Size | Notes |
|--------|--------|------|-------|
| **ASR (spontaneous)** | `google/WaxalNLP` — `aka_asr` | ~100k | Noisy Twi/Akan speech transcriptions |
| **Formal text** | `ghananlpcommunity/pristine-twi` | ~999k | Clean, structured Twi text |

The Ghana NLP Pristine-Twi dataset eliminates the data imbalance of WAXAL-only training (WAXAL TTS has only ~900 samples vs ~100k ASR).

## Key Components

### 1. **WAXALRouter** (`somax/router.py`)
Lightweight stream classifier that routes input to the appropriate linguistic regime:
- **Robust stream** (ASR-optimized): Handles conversational text, fillers, code-switching
- **Logic stream** (formal-text-optimized): Handles formal, semantic-rich text

Uses a trained TF-IDF + logistic regression classifier when available; falls back to a
regex heuristic for zero-dependency environments.

### 2. **DualCoreTokenizer** (`somax/tokenizer.py`)
Manages stream-aware tokenization using a **Unified 8k BPE Vocabulary**. Both streams share
a single vocabulary trained on combined ASR + formal data, ensuring compatible token IDs
and embedding alignment across all model variants.

### 3. **Training Pipeline** (`scripts/`)
- `download.py` - Download WAXAL ASR + Ghana NLP pristine-twi from HuggingFace
- `train_bpe.py` - Train unified 8k BPE vocabulary with causal LM special tokens
- `train_router.py` - Train the TF-IDF + logistic regression router classifier
- `train_lora.py` - Train LoRA variants (A–E) with optional custom tokenizer and embedding warm-init
- `export_gguf.py` - Merge LoRA adapters and export to GGUF via llama.cpp

## Quick Start

### Prerequisites

```bash
# Install all dependencies
pip install -e ".[dev,train]"

# Or install manually:
# Core
pip install transformers tokenizers llama-cpp-python psutil
# Training (cloud)
pip install peft bitsandbytes datasets accelerate torch sentencepiece
```

A HuggingFace account with access approved for `meta-llama/Llama-3.2-1B` is required.

### Option 1: Run End-to-End in Google Colab

1. Open `notebooks/train_eval.ipynb` in VS Code or upload to Colab
2. Run cell 1 — it will prompt for your HuggingFace token via a secure widget
3. Run all remaining cells in order

### Option 2: Run Locally

```bash
# 1. Download datasets (WAXAL ASR + Ghana NLP pristine-twi)
python scripts/download.py --output data/

# 2. Train unified BPE vocabulary (8k tokens, adds causal LM special tokens)
python scripts/train_bpe.py --input data/twi/ --output models/tokenizers/

# 3. Train router classifier
python scripts/train_router.py --data data/twi/ --output models/router/

# 4. Train LoRA variant D with custom tokenizer (recommended)
#    --tokenizer-path connects the custom vocabulary to the model embeddings
python scripts/train_lora.py \
    --group D \
    --data data/twi/ \
    --output checkpoints/ \
    --tokenizer-path models/tokenizers/twi/unified_tokenizer.json

# 5. Benchmark fertility reduction
python scripts/benchmark_fertility.py \
    --tokenizer meta-llama/Llama-3.2-1B \
    --waxal-tokenizer models/tokenizers/twi/unified_tokenizer.json \
    --test-file data/twi/pristine_twi_test.jsonl \
    --compare

# 6. Export to GGUF (requires llama.cpp built from source)
python scripts/export_gguf.py \
    --checkpoint checkpoints/variant_D/final/ \
    --output models/gguf/ \
    --quantization Q4_K_M

# 7. Benchmark edge inference (run on Dell Latitude 7400)
python scripts/benchmark_inference.py \
    --model models/gguf/model-Q4_K_M.gguf \
    --test-file data/twi/pristine_twi_test.jsonl
```

## Training Variants

| Group | Training Sequence | Rationale |
|-------|-------------------|-----------|
| **Control** | Standard Llama-3.2-1B | Baseline "Taxed" performance |
| **Variant A** | ASR Only | Pure robustness to conversational noise |
| **Variant B** | Formal Only | Maximum semantic density and logic |
| **Variant C** | ASR + Formal (Mixed) | Standard joint-distribution training |
| **Variant D** | Formal → ASR → Formal | **Primary Hypothesis**: Anchor logic, adapt to noise, refine logic |
| **Variant E** | ASR → Formal | Test if phonetic grounding aids later reasoning |

## Metrics

### Primary: Token Fertility
```
F = Total Tokens / Total Words
```
**Target**: Reduce F by ≥30% vs baseline

### Secondary
- Tokens per second (TPS)
- Inference latency (seconds)
- Memory usage (MB)

## Hardware Requirements

### Training (Cloud)
- **Platform**: Google Colab
- **GPU**: T4 (16GB VRAM)
- **Runtime**: ~2-4 hours per variant

### Deployment (Edge)
- **Device**: Dell Latitude 7400
- **RAM**: 8GB
- **Format**: 4-bit GGUF quantization

## Project Structure

```
SOMAX/
├── data/                  # Twi datasets (gitignored)
├── models/                # Trained models (gitignored)
├── scripts/               # Training pipeline
│   ├── download.py
│   ├── train_bpe.py
│   ├── train_router.py
│   ├── train_lora.py
│   ├── export_gguf.py
│   ├── benchmark_fertility.py  # Fertility metrics
│   └── benchmark_inference.py  # Edge performance metrics
├── requirements.txt
├── pyproject.toml
├── Makefile
└── README.md
```

## Usage Example

```python
from somax import DualCoreTokenizer

# Initialize with unified tokenizer
tokenizer = DualCoreTokenizer(
    tokenizer_path="models/tokenizers/twi/unified_tokenizer.json",
    language="twi",
)

# Automatic routing based on input
conversational = "uhm chale me dwo o"  # Routed to ASR stream
formal = "The president delivered a formal address"  # Routed to formal stream

# Check routing
print(tokenizer.classify(conversational))  # "robust"
print(tokenizer.classify(formal))           # "logic"

# Encode (returns token IDs from the unified 8k vocabulary)
tokens = tokenizer.encode(conversational)

# Encode and get stream classification in one call
ids, stream = tokenizer.encode_with_stream(conversational)
print(stream)  # "robust" — use this to select the correct LoRA adapter at inference
```

## Results

After training, compare with baseline:

```bash
# Baseline
python scripts/benchmark_fertility.py --tokenizer meta-llama/Llama-3.2-1B --test-file data/twi/pristine_twi_test.jsonl

# Trained
python scripts/benchmark_fertility.py --tokenizer models/tokenizers/twi/unified_tokenizer.json --waxal --test-file data/twi/pristine_twi_test.jsonl
```

Expected improvement:
- **Baseline fertility**: ~4.0 tokens/word
- **Target fertility**: ~2.8 tokens/word (30% reduction)

## Roadmap

- [x] Project skeleton
- [x] Dataset integration (WAXAL ASR + Ghana NLP pristine-twi)
- [x] Training scripts (LoRA variants A–E)
- [x] Edge library (router + tokenizer)
- [x] Colab notebook
- [x] Tokenizer-embedding alignment fix (custom 8k BPE connected to model training via warm-init)
- [ ] Benchmark on edge hardware (Dell Latitude 7400)
- [ ] Open-source model release

## Citation

If you use this work, please cite the WAXAL dataset and the Ghana NLP Pristine-Twi dataset:

```bibtex
@article{waxal2026,
  title={WAXAL: A Large-Scale Multilingual African Language Speech Corpus},
  author={Anonymous},
  journal={arXiv preprint arXiv:2602.02734},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Research for the WAXAL dataset
- Ghana NLP Community for the Pristine-Twi dataset
- HuggingFace for model distribution
- The African NLP community
