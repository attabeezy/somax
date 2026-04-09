.PHONY: setup download train_bpe train_router train_lora export_gguf benchmark benchmark_edge clean lint test

LANG ?= akan
GROUP ?= D
TOKENIZER_PATH ?= models/tokenizers/$(LANG)/unified_tokenizer.json

setup:
	pip install -e ".[dev,train]"

download:
	python scripts/download.py --lang $(LANG) --output data/

train_bpe:
	python scripts/train_bpe.py --input data/$(LANG)/ --output models/tokenizers/ --language $(LANG)

train_router:
	python scripts/train_router.py --data data/$(LANG)/ --output models/router/ --language $(LANG)

train_lora:
	python scripts/train_lora.py \
		--group $(GROUP) \
		--data data/$(LANG)/ \
		--output checkpoints/ \
		--language $(LANG) \
		--tokenizer-path $(TOKENIZER_PATH)

export_gguf:
	python scripts/export_gguf.py \
		--checkpoint checkpoints/variant_$(GROUP)/final/ \
		--output models/gguf/ \
		--quantization Q4_K_M

benchmark:
	python scripts/benchmark_fertility.py \
		--tokenizer meta-llama/Llama-3.2-1B \
		--waxal-tokenizer $(TOKENIZER_PATH) \
		--test-file data/$(LANG)/twi_tts_test.jsonl \
		--compare

benchmark_edge:
	python scripts/benchmark_inference.py \
		--model models/gguf/model-Q4_K_M.gguf \
		--test-file data/$(LANG)/twi_tts_test.jsonl

lint:
	ruff check .
	black --check .
	mypy somax/

test:
	@if [ -d "tests" ]; then pytest tests/ -v; else echo "No tests/ directory found."; fi

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf data/*.pt data/*.bin data/*.gguf
	find . -type d -name "__pycache__" -exec rm -rf {} +
