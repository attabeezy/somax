#!/usr/bin/env python3
"""Export trained model to GGUF format for edge deployment.

Converts LoRA-adapted model to merged model, then quantizes to 4-bit GGUF
for deployment on Dell Latitude 7400 (8GB RAM).

Prerequisites:
    - llama.cpp compiled with Python bindings
    - Or use: pip install llama-cpp-python

Usage:
    # After training
    python scripts/export_gguf.py --checkpoint checkpoints/variant_D/final/ --output models/gguf/

    # Convert base tokenizer
    python scripts/export_gguf.py --checkpoint meta-llama/Llama-3.2-1B --output models/gguf/ --base-only
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_llama_cpp() -> bool:
    """Check if llama.cpp tools are available.

    Returns:
        True if llama.cpp is installed.
    """
    try:
        result = subprocess.run(
            ["python", "-c", "import llama_cpp; print(llama_cpp.__version__)"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def merge_lora_model(checkpoint_dir: Path, output_dir: Path) -> Path:
    """Merge LoRA adapters with base model.

    Args:
        checkpoint_dir: Directory containing LoRA adapter.
        output_dir: Directory to save merged model.

    Returns:
        Path to merged model directory.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "transformers and peft required. Install with: pip install transformers peft"
        )

    print(f"Loading base model...")

    adapter_config = checkpoint_dir / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(f"LoRA adapter not found at {checkpoint_dir}")

    with open(adapter_config, "r") as f:
        import json

        config = json.load(f)

    base_model_id = config.get("base_model_name_or_path")
    if not base_model_id:
        raise ValueError("base_model_name_or_path not found in adapter_config.json")

    print(f"Base model: {base_model_id}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print(f"Loading LoRA adapter from {checkpoint_dir}...")
    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))

    print("Merging adapter into base model...")
    merged_model = model.merge_and_unload()

    merged_dir = output_dir / "merged_model"
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {merged_dir}...")
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    print("Merge complete!")
    return merged_dir


def find_llama_cpp() -> tuple[Path, Path] | None:
    """Locate llama.cpp conversion and quantization tools.

    Searches for a sibling 'llama.cpp' directory relative to this script,
    then falls back to PATH.

    Returns:
        Tuple of (convert_script, quantize_bin) paths, or None if not found.
    """
    # Check sibling llama.cpp directory first
    candidates = [
        Path(__file__).parent.parent / "llama.cpp",
        Path.cwd() / "llama.cpp",
    ]
    for base in candidates:
        convert = base / "convert-hf-to-gguf.py"
        quantize = base / "llama-quantize"
        if not quantize.exists():
            quantize = base / "llama-quantize.exe"
        if convert.exists() and quantize.exists():
            return convert, quantize

    # Fall back to PATH
    import shutil
    quantize_bin = shutil.which("llama-quantize")
    convert_script = shutil.which("convert-hf-to-gguf.py")
    if quantize_bin and convert_script:
        return Path(convert_script), Path(quantize_bin)

    return None


def convert_to_gguf(model_dir: Path, output_dir: Path, quantization: str = "Q4_K_M") -> Path:
    """Convert merged model to GGUF format using llama.cpp.

    Args:
        model_dir: Directory containing merged HuggingFace model.
        output_dir: Directory to save GGUF file.
        quantization: Quantization method (Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q8_0).

    Returns:
        Path to quantized GGUF file.

    Raises:
        FileNotFoundError: If llama.cpp tools are not found.
        subprocess.CalledProcessError: If conversion or quantization fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tools = find_llama_cpp()
    if tools is None:
        raise FileNotFoundError(
            "llama.cpp tools not found. Either:\n"
            "  1. Clone llama.cpp into a sibling directory:  git clone https://github.com/ggerganov/llama.cpp\n"
            "     then build it: cd llama.cpp && make\n"
            "  2. Add llama-quantize and convert-hf-to-gguf.py to PATH"
        )

    convert_script, quantize_bin = tools
    fp16_path = output_dir / "model-f16.gguf"
    gguf_path = output_dir / f"model-{quantization}.gguf"

    print(f"Converting to GGUF (FP16)...")
    subprocess.run(
        [sys.executable, str(convert_script), str(model_dir), "--outfile", str(fp16_path), "--outtype", "f16"],
        check=True,
    )
    print(f"FP16 GGUF saved to: {fp16_path}")

    print(f"Quantizing to {quantization}...")
    try:
        subprocess.run(
            [str(quantize_bin), str(fp16_path), str(gguf_path), quantization],
            check=True,
        )
    finally:
        if fp16_path.exists():
            fp16_path.unlink()
    print(f"Quantized GGUF saved to: {gguf_path}")

    return gguf_path



def main() -> None:
    parser = argparse.ArgumentParser(description="Export to GGUF format")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to LoRA checkpoint or base model ID"
    )
    parser.add_argument("--output", type=str, default="models/gguf/")
    parser.add_argument(
        "--quantization",
        type=str,
        default="Q4_K_M",
        choices=["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0"],
    )
    parser.add_argument(
        "--base-only", action="store_true", help="Skip LoRA merge, just prepare base model"
    )
    parser.add_argument(
        "--no-convert", action="store_true", help="Skip GGUF conversion, just merge LoRA"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    is_base_model = not checkpoint_path.exists()

    if args.base_only or is_base_model:
        print(f"Using base model directly: {args.checkpoint}")
        print("No LoRA merge required.")
        convert_to_gguf(Path(args.checkpoint), output_dir, args.quantization)
        return

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Exporting model...")
    print(f"  Input: {checkpoint_path}")
    print(f"  Output: {output_dir}")
    print(f"  Quantization: {args.quantization}")
    print()

    merged_dir = merge_lora_model(checkpoint_path, output_dir)

    if args.no_convert:
        print("\nSkipping GGUF conversion (--no-convert)")
        print(f"Merged model saved to: {merged_dir}")
        return

    gguf_path = convert_to_gguf(merged_dir, output_dir, args.quantization)
    print(f"\nDone. GGUF model: {gguf_path}")


if __name__ == "__main__":
    main()
