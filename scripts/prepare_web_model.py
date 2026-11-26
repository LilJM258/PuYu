"""Download only the configured public base model or tokenizer for local use."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.context import load_model_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-only", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    config = load_model_config()
    patterns = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "config.json", "generation_config.json", "LICENSE", "chat_template.jinja", "*.safetensors.index.json"]
    if not args.tokenizer_only:
        patterns.append("*.safetensors")
    path = snapshot_download(config["model_id"], revision=config.get("revision", "main"),
                             cache_dir=config["cache_dir"], allow_patterns=patterns)
    print(json.dumps({"model_id": config["model_id"], "snapshot": path,
                      "tokenizer_only": args.tokenizer_only}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
