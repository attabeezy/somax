# somax

**Eliminating the Tokenization Tax for African Languages via Dual-Stream Tokenization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

Modern LLM tokenizers are optimized for English, resulting in a **"Tokenization Tax"** for African languages like Akan, Yoruba, and Swahili. This project implements a research-to-production framework that:

- Reduces token fertility (F = tokens/words) by ≥30%
- Uses dual-stream processing for spontaneous speech (ASR) and formal text (TTS)
- Deploys efficiently on edge devices (8GB RAM)

## Key Components

### 1. **WAXALRouter** (`somax/router.py`)
Lightweight stream classifier that routes input to the appropriate linguistic regime:
- **Robust stream** (ASR-optimized): Handles conversational text, fillers, code-switching
- **Logic stream** (TTS-optimized): Handles formal, semantic-rich text

Uses a trained TF-IDF + logistic regression classifier when available; falls back to a
regex heuristic for zero-dependency environments.

### 2. **DualCoreTokenizer** (`somax/tokenizer.py`)
Manages stream-aware tokenization using a **Unified 8k BPE Vocabulary**. Both streams share
a single vocabulary trained on combined WAXAL ASR+TTS data, ensuring compatible token IDs
and embedding alignment across all model variants.

### 3. **Training Pipeline** (`scripts/`)
- `download.py` - Download WAXAL dataset from HuggingFace (`google/WaxalNLP`)
- `train_bpe.py` - Train unified 8k BPE vocabulary with causal LM special tokens
- `train_router.py` - Train the TF-IDF + logistic regression router classifier
- `train_lora.py` - Train LoRA variants (A–E) with optional WAXAL tokenizer and embedding warm-init
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

1. Open `notebooks/pipeline.ipynb` in VS Code or upload to Colab
2. Run cell 1 — it will prompt for your HuggingFace token via a secure widget
3. Run all remaining cells in order

### Option 2: Run Locally

```bash
# 1. Download WAXAL dataset (Akan)
python scripts/download.py --lang akan --output data/

# 2. Train unified BPE vocabulary (8k tokens, adds causal LM special tokens)
python scripts/train_bpe.py --input data/akan/ --output models/tokenizers/

# 3. Train router classifier
python scripts/train_router.py --data data/akan/ --output models/router/

# 4. Train LoRA variant D with WAXAL tokenizer (recommended)
#    --tokenizer-path connects the custom vocabulary to the model embeddings
python scripts/train_lora.py \
    --group D \
    --data data/akan/ \
    --output checkpoints/ \
    --tokenizer-path models/tokenizers/akan/unified_tokenizer.json

# 5. Benchmark fertility reduction
python benchmark_fertility.py \
    --tokenizer meta-llama/Llama-3.2-1B \
    --waxal-tokenizer models/tokenizers/akan/unified_tokenizer.json \
    --test-file data/akan/twi_tts_test.jsonl \
    --compare

# 6. Export to GGUF (requires llama.cpp built from source)
python scripts/export_gguf.py \
    --checkpoint checkpoints/variant_D/final/ \
    --output models/gguf/ \
    --quantization Q4_K_M

# 7. Benchmark edge inference (run on Dell Latitude 7400)
python benchmark_inference.py \
    --model models/gguf/model-Q4_K_M.gguf \
    --test-file data/akan/twi_tts_test.jsonl
```

## Dataset

**WAXAL Dataset** (Google Research, Feb 2026)
- **Source**: `google/WaxalNLP` on HuggingFace
- **Languages**: Akan, Yoruba, Swahili (others available)
- **Splits**: ASR (spontaneous) and TTS (formal)
- **License**: CC-BY-4.0

### Language Configuration

| Language | ASR Config | TTS Config | Notes |
|----------|------------|-------------|-------|
| Akan | `aka_asr` | `twi_tts` | Proof-of-concept |
| Yoruba | *(none)* | `yor_tts` | TTS only |
| Swahili | *(none)* | `swa_tts` | TTS only |

## Training Variants

| Group | Training Sequence | Rationale |
|-------|-------------------|-----------|
| **Control** | Standard Llama-3.2-1B | Baseline "Taxed" performance |
| **Variant A** | ASR Only | Pure robustness to conversational noise |
| **Variant B** | TTS Only | Maximum semantic density and logic |
| **Variant C** | ASR + TTS (Mixed) | Standard joint-distribution training |
| **Variant D** | TTS → ASR → TTS | **Primary Hypothesis**: Anchor logic, adapt to noise, refine logic |
| **Variant E** | ASR → TTS | Test if phonetic grounding aids later reasoning |

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
├── data/                  # WAXAL subsets (gitignored)
├── models/                # Trained models (gitignored)
├── scripts/               # Training pipeline
│   ├── download.py
│   ├── train_bpe.py
│   ├── train_router.py
│   ├── train_lora.py
│   └── export_gguf.py
├── somax/         # Edge Python library
│   ├── __init__.py
│   ├── router.py          # Stream classifier
│   └── tokenizer.py       # Dual-core tokenizer (Unified)
├── notebooks/             # Colab notebooks
│   └── pipeline.ipynb
├── benchmark_fertility.py  # Fertility metrics
├── benchmark_inference.py  # Edge performance metrics
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
    tokenizer_path="models/tokenizers/akan/unified_tokenizer.json",
    language="akan",
)

# Automatic routing based on input
conversational = "uhm chale me dwo o"  # Routed to ASR stream
formal = "The president delivered a formal address"  # Routed to TTS stream

# Check routing
print(tokenizer.classify(conversational))  # "robust"
print(tokenizer.classify(formal))           # "logic"

# Encode (returns token IDs from the unified 8k WAXAL vocabulary)
tokens = tokenizer.encode(conversational)

# Encode and get stream classification in one call
ids, stream = tokenizer.encode_with_stream(conversational)
print(stream)  # "robust" — use this to select the correct LoRA adapter at inference
```

## Results

After training, compare with baseline:

```bash
# Baseline
python benchmark_fertility.py --tokenizer meta-llama/Llama-3.2-1B --test-file data/akan/twi_tts_test.jsonl

# Trained
python benchmark_fertility.py --tokenizer models/tokenizers/akan/unified_tokenizer.json --waxal --test-file data/akan/twi_tts_test.jsonl
```

Expected improvement:
- **Baseline fertility**: ~4.0 tokens/word
- **Target fertility**: ~2.8 tokens/word (30% reduction)

## Roadmap

- [x] Project skeleton
- [x] Dataset integration (WaxalNLP)
- [x] Training scripts (LoRA variants A–E)
- [x] Edge library (router + tokenizer)
- [x] Colab notebook
- [x] Tokenizer-embedding alignment fix (custom 8k BPE connected to model training via warm-init)
- [ ] Benchmark on edge hardware (Dell Latitude 7400)
- [ ] Open-source model release

## Citation

If you use this work, please cite the WAXAL dataset:

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
- HuggingFace for model distribution
- The African NLP community
