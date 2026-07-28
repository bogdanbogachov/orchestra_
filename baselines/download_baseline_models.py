#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModel, AutoTokenizer


MODELS = [
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "output": "all-MiniLM-L6-v2",
        "kind": "encoder",
    },
    {
        "name": "distilbert-base-uncased",
        "output": "distilbert-base-uncased",
        "kind": "encoder",
    },
]


def download_model(model_name: str, output_dir: Path, kind: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_name} -> {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"Saved {model_name} to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download baseline Hugging Face models for offline cluster runs.")
    parser.add_argument("--output-root", default="downloaded_models")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    for item in MODELS:
        download_model(
            model_name=item["name"],
            output_dir=output_root / item["output"],
            kind=item["kind"],
        )

    print("Done.")


if __name__ == "__main__":
    main()
