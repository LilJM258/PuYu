"""Run deterministic regression checks without using the user's live B profile."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    temp_root = ROOT/'results/test_temp'
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='member6-', dir=temp_root) as folder:
        env = dict(os.environ, PUYU_MODEL_CONFIG=str(ROOT/'configs/web_model.json'),
                   TEMP=folder, TMP=folder)
        return subprocess.call([sys.executable, '-X', 'utf8', '-m', 'unittest',
                                'discover', '-s', 'tests', '-v'], cwd=ROOT, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
