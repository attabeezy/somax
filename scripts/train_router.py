#!/usr/bin/env python3
"""Train a stream classifier for WAXALRouter.

Trains a logistic regression classifier (TF-IDF features) on the WAXAL
ASR/TTS split to distinguish conversational from formal text. The resulting
model replaces the regex heuristic in WAXALRouter for data-driven routing.

Usage:
    python scripts/train_router.py --data data/akan/ --output models/router/ --language akan
"""

import argparse
import json
import pickle
from pathlib import Path

# Mirrors download.py DATASET_CONFIGS — must stay in sync.
LANG_FILE_PREFIXES: dict[str, dict[str, str | None]] = {
    "akan":    {"asr": "aka_asr", "tts": "twi_tts"},
    "yoruba":  {"asr": None,      "tts": "yor_tts"},
    "swahili": {"asr": None,      "tts": "swa_tts"},
}


def load_labeled_texts(data_dir: Path, language: str) -> tuple[list[str], list[int]]:
    """Load ASR and TTS texts with stream labels.

    ASR texts are labeled 0 (robust stream), TTS texts are labeled 1 (logic stream).

    Args:
        data_dir: Directory containing JSONL files.
        language: Language code (e.g. 'akan').

    Returns:
        Tuple of (texts, labels).
    """
    texts, labels = [], []

    prefixes = LANG_FILE_PREFIXES[language]
    asr_file = data_dir / f"{prefixes['asr']}_train.jsonl" if prefixes["asr"] else None
    tts_file = data_dir / f"{prefixes['tts']}_train.jsonl"

    for path, label in [(asr_file, 0), (tts_file, 1)]:
        if path is None:
            print(f"No {'ASR' if label == 0 else 'TTS'} config for {language}, skipping.")
            continue
        if not path.exists():
            print(f"WARNING: {path} not found, skipping.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    text = item.get("transcription") or item.get("text", "")
                    if text:
                        texts.append(text)
                        labels.append(label)
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {sum(1 for l in labels if l == label)} {'ASR' if label == 0 else 'TTS'} samples")

    return texts, labels


def train_classifier(texts: list[str], labels: list[int]):
    """Train a TF-IDF + logistic regression classifier.

    Args:
        texts: Training text samples.
        labels: Stream labels (0=robust/ASR, 1=logic/TTS).

    Returns:
        Trained sklearn Pipeline.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    import numpy as np

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=20000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])

    scores = cross_val_score(pipeline, texts, labels, cv=5, scoring="f1_macro")
    print(f"Cross-val F1 (5-fold): {scores.mean():.3f} ± {scores.std():.3f}")

    pipeline.fit(texts, labels)
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WAXALRouter stream classifier")
    parser.add_argument("--data", type=str, default="data/akan/")
    parser.add_argument("--output", type=str, default="models/router/")
    parser.add_argument("--language", type=str, default="akan")
    args = parser.parse_args()

    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Training router classifier for language: {args.language}")
    print(f"Data: {data_dir}")

    texts, labels = load_labeled_texts(data_dir, args.language)

    if not texts:
        raise ValueError("No training data found. Run scripts/download.py first.")

    print(f"\nTotal samples: {len(texts)} ({labels.count(0)} ASR, {labels.count(1)} TTS)")
    print("Training classifier...")

    classifier = train_classifier(texts, labels)

    output_path = output_dir / f"{args.language}_router.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(classifier, f)

    print(f"\nClassifier saved to: {output_path}")

    sample_asr = "uhm chale me dwo o"
    sample_tts = "The president delivered a formal address to the assembly"
    pred_asr = classifier.predict([sample_asr])[0]
    pred_tts = classifier.predict([sample_tts])[0]
    print(f"\nSanity check:")
    print(f"  '{sample_asr}' -> {'robust' if pred_asr == 0 else 'logic'}")
    print(f"  '{sample_tts}' -> {'robust' if pred_tts == 0 else 'logic'}")


if __name__ == "__main__":
    main()
