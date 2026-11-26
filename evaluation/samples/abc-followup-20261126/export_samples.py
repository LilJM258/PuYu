"""Export the frozen ABC follow-up inputs and references from the repo's data."""
import argparse
import hashlib
import json
from pathlib import Path


def jsonl(records):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode("utf-8")


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=folder.parents[2] / "training_data_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((folder / "抽样记录.json").read_bytes())

    pools = {}
    prompts = []
    files = {}
    for task, spec in manifest["tasks"].items():
        source = spec["source_file"]
        if source not in pools:
            with (args.data_dir / source).open(encoding="utf-8-sig") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            pools[source] = {row["id"]: row for row in records}
        references = []
        for ident in spec["ids"]:
            row = pools[source].get(ident)
            if row is None or (row["task"], row["split"]) != (task, spec["split"]):
                parser.error(f"Missing or mismatched sample {ident} in {source}")
            references.append(row)
            prompts.append({
                "id": ident,
                "task": task,
                "split": spec["split"],
                "prompt": row["messages"][0]["content"],
            })
        files[spec["reference_file"]] = (jsonl(references), spec["reference_sha256"])

    files["模型输入.jsonl"] = (jsonl(prompts), manifest["prompts_sha256"])
    for name, (content, expected) in files.items():
        if hashlib.sha256(content).hexdigest() != expected:
            parser.error(f"{name} differs from the frozen sample package. Check the source data version.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Choose an empty output directory.")
    for name, (content, _) in files.items():
        path = args.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    print(f"Exported {len(prompts)} inputs and three reference files to {args.output_dir}")


if __name__ == "__main__":
    main()
