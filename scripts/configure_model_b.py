"""Write ignored local source/desktop configuration for a delivered Qwen3 B adapter."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure(base, adapter, runtime, desktop=None):
    base, adapter, runtime = [Path(value).resolve() for value in (base, adapter, runtime)]
    metadata = json.loads((adapter/'adapter_config.json').read_text(encoding='utf-8'))
    if metadata.get('base_model_name_or_path') != 'Qwen/Qwen3-1.7B':
        raise ValueError('Adapter 基础模型与官方 Qwen3-1.7B 不一致。')
    if json.loads((base/'config.json').read_text(encoding='utf-8')).get('model_type') != 'qwen3':
        raise ValueError('基础权重配置不是 Qwen3。')
    if not runtime.is_file() or not (adapter/'adapter_model.safetensors').is_file():
        raise ValueError('缺少 Python 推理环境或 Adapter 权重。')
    index = base/'model.safetensors.index.json'
    if index.is_file():
        weights = set(json.loads(index.read_text())['weight_map'].values())
    else:
        weights = {'model.safetensors'}
    if not (base/'config.json').is_file() or not all((base/name).is_file() for name in weights):
        raise ValueError('基础模型权重目录不完整。')
    config = json.loads((ROOT/'configs/web_model.json').read_text(encoding='utf-8'))
    config.update(model_group='B', model_path=str(base), adapter_path=str(adapter),
                  device='cuda', context_tokens=512, max_essay_words=150,
                  max_new_tokens={'scoring': 192, 'feedback': 256, 'correction': 256},
                  runtime_python=str(runtime), worker_path=str(ROOT/'scripts/model_worker.py'))
    output = ROOT/'configs/web_model.local.json'
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    outputs = [str(output)]
    if desktop:
        desktop = Path(desktop).resolve()
        worker = desktop/'backend/scripts/model_worker.py'
        if not worker.is_file():
            raise ValueError('桌面版未包含后台源码，请先重新打包。')
        desktop_config = dict(config, worker_path=str(worker))
        target = desktop/'configs/web_model.local.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(desktop_config, ensure_ascii=False, indent=2), encoding='utf-8')
        outputs.append(str(target))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--python-path', required=True)
    parser.add_argument('--desktop-dir')
    args = parser.parse_args()
    print(json.dumps(configure(args.base_model, args.adapter, args.python_path, args.desktop_dir),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
