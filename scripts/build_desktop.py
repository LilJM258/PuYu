"""Build the Windows desktop distribution from the current source tree."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def collect_backend_sources(app):
    for name in ('app_paths.py', 'inference/__init__.py', 'inference/context.py',
                 'inference/schema.py', 'inference/model_pipeline.py', 'scripts/model_worker.py'):
        destination = app / 'backend' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)


def collect_inspection_sources(app):
    # Gradio reads its component source files to build runtime event interfaces.
    from importlib.util import find_spec
    for name in ('gradio', 'gradio_client'):
        source = Path(find_spec(name).origin).parent
        target = app / '_internal' / name
        for pattern in ('*.py', '*.pyi'):
            for path in source.rglob(pattern):
                destination = target / path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    from PyInstaller.utils.hooks import collect_data_files
    for package in ('groovy',):
        for source, relative in collect_data_files(package):
            destination = app / '_internal' / relative / Path(source).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "build/desktop")
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm",
                    "--distpath", str(args.dist_dir.resolve()), "--workpath", str(args.work_dir.resolve()),
                    str(ROOT / "PuYu-desktop.spec")], cwd=ROOT, check=True)
    app = args.dist_dir.resolve() / "PuYu"
    collect_inspection_sources(app)
    collect_backend_sources(app)
    if args.model_dir:
        target = app / "models/Qwen3-1.7B"
        target.mkdir(parents=True, exist_ok=True)
        index = args.model_dir / 'model.safetensors.index.json'
        weights = set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values()) if index.is_file() else {'model.safetensors'}
        required = {'config.json', 'tokenizer.json', 'tokenizer_config.json'} | weights
        for name in required:
            source = args.model_dir / name
            if not source.is_file():
                raise SystemExit(f"Model directory is incomplete: {source}")
        optional = {'generation_config.json', 'vocab.json', 'merges.txt', 'LICENSE', 'chat_template.jinja', 'model.safetensors.index.json'}
        names = required | {name for name in optional if (args.model_dir / name).is_file()}
        for name in names:
            shutil.copy2(args.model_dir / name, target / name)
    (app / "使用说明.txt").write_text(
        "英语作文批改工作台桌面版\n\n双击 PuYu.exe 打开独立窗口。首次启动请等待工作台加载。\n"
        "整个 PuYu 文件夹需要一起保留或发送给其他同学。\n"
        "选择团队模拟接口可验证流程，选择展示样例可查看颜色标注。\n"
        "本地批改使用 Qwen3-1.7B。模型 B 需配置基础权重、Adapter 和独立推理环境。\n"
        "首次使用本地模型需要加载权重。失败回答、原始输出与耗时均可在报告中查看。\n"
        "报告、运行日志和窗口配置保存在 %LOCALAPPDATA%\\PuYu。\n"
        "下载报告时会打开另存为窗口，可选择保存位置。\n"
        "运行环境为 Windows 10/11 64 位，需要 Microsoft Edge WebView2 Runtime。\n",
        encoding="utf-8-sig")
    print(json.dumps({"application": str(app / "PuYu.exe"), "includes_base_model": bool(args.model_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
