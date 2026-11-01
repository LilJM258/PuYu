"""Download the member-2 handoff at a fixed public GitHub commit and verify Git blob hashes."""

import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "3cb5481323b3ec9481ebd96097841e0e61fe3b42"
FILES = {
    "train-00001.jsonl": "c690dc25786afac0af390b3482aeabfb24ca0d5d",
    "train-00002.jsonl": "79094d5b5ce3b1096be98156b01da13ffec35bcf",
    "validation.jsonl": "24778b2ab863c9f4a1c1940b69a6357feea52b9f",
    "test.jsonl": "af65bbd30bbe132a8ae7c23943a785301b25defc",
    "训练数据使用说明.md": "582b257737b21fca11664275c4ccb3a898c0219b",
    "数据许可说明.md": "a4385500b238a5021378f5063541fa34530d97c1",
}


def blob_hash(path):
    raw = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def main():
    destination = ROOT / "data/raw" / ("puyu_" + COMMIT[:7])
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": "LilJM258/PuYu", "commit": COMMIT, "files": {}}
    for name, expected in FILES.items():
        target = destination / name
        if not target.exists():
            url = "https://raw.githubusercontent.com/LilJM258/PuYu/" + COMMIT + "/training_data_v1/" + urllib.parse.quote(name)
            partial = target.with_suffix(target.suffix + ".download")
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if blob_hash(partial) != expected:
                raise ValueError("Downloaded file checksum mismatch: " + name)
            partial.replace(target)
        if blob_hash(target) != expected:
            raise ValueError("Existing snapshot checksum mismatch: " + name)
        manifest["files"][name] = {"git_blob_sha1": expected, "bytes": target.stat().st_size,
                                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        print("Verified:", name, target.stat().st_size, "bytes", flush=True)
    (destination / "download_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Snapshot saved:", destination)


if __name__ == "__main__":
    main()
