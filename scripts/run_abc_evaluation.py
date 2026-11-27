"""Freeze inputs, reuse inference.py and evaluation/, and retain every prediction.

prepare uses local paths; run is resumable and runs validation before full test.
No training, data resampling, JSON repair, or paid services are performed here.
"""
import argparse
from collections import Counter
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from time import perf_counter

REPO = Path(__file__).resolve().parents[1]
TASKS = {'essay_scoring': 'scoring', 'essay_feedback': 'feedback', 'grammar_correction': 'correction'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n' for value in values)
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise ValueError(f'Refusing to replace changed records: {path}')
    path.write_text(text, encoding='utf-8')


def checked_rows(path, split):
    values = rows(path)
    ids = [row['id'] for row in values]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate gold IDs')
    for row in values:
        if row['split'] != split or row['task'] not in TASKS:
            raise ValueError('Unexpected split or task')
        messages = row['messages']
        if len(messages) != 2 or messages[0]['role'] != 'user' or messages[1]['role'] != 'assistant':
            raise ValueError('Expected a separate user prompt and reference answer')
    return values


def select_test(values, scoring=-1, feedback=-1, seed=42):
    selected = []
    for task, count in (('essay_scoring', scoring), ('essay_feedback', feedback)):
        candidates = sorted((row for row in values if row['task'] == task), key=lambda row: row['id'])
        if count == -1:
            selected.extend(candidates)
        elif 1 <= count <= len(candidates):
            selected.extend(random.Random(seed).sample(candidates, count))
        else:
            raise ValueError(f'Invalid requested sample count for {task}: {count}')
    random.Random(seed).shuffle(selected)
    return selected


def prepare(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'code').mkdir()
    for source in (Path(__file__), REPO / 'inference.py'):
        shutil.copy2(source, out / 'code' / source.name)
    validation = checked_rows(REPO / 'training_data_v1/validation.jsonl', 'validation')
    test_pool = checked_rows(REPO / 'training_data_v1/test.jsonl', 'test')
    test = select_test(test_pool, args.test_scoring, args.test_feedback)
    # The feedback pilot was selected before C outputs, using seed 42.
    pilot_feedback = rows(args.existing_pilot)
    val_by_id = {row['id']: row for row in validation}
    if any(val_by_id.get(row['id']) != row for row in pilot_feedback):
        raise ValueError('Old frozen pilot differs from current validation')
    pilot = list(pilot_feedback)
    for task in ('essay_scoring', 'grammar_correction'):
        candidates = sorted((row for row in validation if row['task'] == task), key=lambda row: row['id'])
        pilot.extend(random.Random(42).sample(candidates, 5))
    random.Random(42).shuffle(pilot)
    paths = {}
    for split, values in (('validation', pilot), ('test', test)):
        save_rows(out / f'gold/{split}/all.jsonl', values)
        paths[split] = {'count': len(values), 'by_task': dict(Counter(row['task'] for row in values)),
                        'sha256': sha(out / f'gold/{split}/all.jsonl')}
        for task in TASKS:
            subset = [row for row in values if row['task'] == task]
            if subset:
                save_rows(out / f'gold/{split}/{task}.jsonl', subset)
    # Freeze all source files used for metric computation as well.
    shutil.copytree(REPO / 'evaluation', out / 'code/evaluation',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'human'))
    provenance = {
        'base_model': {'path': str(args.base.resolve()), 'id': 'Qwen/Qwen3-1.7B',
            'weights': load(args.base / 'verified-weights.json')},
        'adapters': {group: {'path': str(path.resolve()), 'sha256': sha(path / 'adapter_model.safetensors')}
                     for group, path in (('B', args.adapter_b), ('C', args.adapter_c))},
    }
    for name, expected in provenance['base_model']['weights'].items():
        if sha(args.base / name) != expected:
            raise ValueError(f'Base model weight hash mismatch: {name}')
    protocol = {'version': 'abc_frozen_test_v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'run_dir': str(out), 'repo': str(REPO), 'eval_python': str(args.eval_python.resolve()),
        'inference_python': sys.executable, 'model': provenance, 'datasets': paths,
        'seed': 42, 'generation': {'max_new_tokens': 512, 'max_time': 180,
            'context_tokens': 4096, 'do_sample': False, 'repetition_penalty': 1.05,
            'enable_thinking': False, 'eos_token_id': 151645, 'dtype': 'bfloat16',
            'device': 'cuda', 'batch_size': 1, 'json_repair': False, 'input_truncation': False},
        'scope': {'validation': 'same 5 feedback + seeded 5 scoring + seeded 5 correction',
            'test': f'{len(test)} of {len(test_pool)} available test records, identical IDs for A/B/C',
            'sampling': {'seed': 42, 'scoring_requested': args.test_scoring, 'feedback_requested': args.test_feedback,
                'selection': 'uniform sample without replacement within task, IDs sorted first; frozen before test outputs',
                'full_dataset': len(test) == len(test_pool), 'pool_sha256': sha(REPO / 'training_data_v1/test.jsonl')},
            'correction_test': 'not available; validation results only',
            'prior_test_use': 'Historical test records were used for debugging; this is a frozen rerun, not a pristine unseen holdout.',
            'B_training_data': 'Actual B JSONL/resampling history has not been supplied; pure causal teacher-data benefit is not established.',
            'context': 'Training max_length=512 is not the native inference window. Evaluation uses 4096 for all models; desktop limits stay unchanged.',
            'optional_metrics': 'BERTScore not requested; no judge API or invented human ratings.'},
        'source_hashes': {str(path.relative_to(out / 'code')): sha(path)
            for path in sorted((out / 'code').rglob('*.py'))}}
    save(out / 'protocol.json', protocol)
    print(json.dumps({'protocol': str(out / 'protocol.json'), 'datasets': paths}, ensure_ascii=False), flush=True)


def verify_protocol(protocol):
    out = Path(protocol['run_dir'])
    for relative, expected in protocol['source_hashes'].items():
        if sha(out / 'code' / relative) != expected:
            raise ValueError(f'Frozen code changed: {relative}')
    for split, description in protocol['datasets'].items():
        if sha(out / f'gold/{split}/all.jsonl') != description['sha256']:
            raise ValueError('Frozen input changed')
    return out


def predict(protocol, group, split):
    out = verify_protocol(protocol)
    import importlib.metadata
    import torch
    import transformers
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.set_num_threads(4)
    transformers.set_seed(protocol['seed'])
    spec = importlib.util.spec_from_file_location('puyu_original_inference', out / 'code/inference.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    adapter = None if group == 'A' else protocol['model']['adapters'][group]['path']
    if adapter and sha(Path(adapter) / 'adapter_model.safetensors') != protocol['model']['adapters'][group]['sha256']:
        raise ValueError('Adapter changed')
    values = checked_rows(out / f'gold/{split}/all.jsonl', split)
    directory = out / f'predictions/{split}/{group}'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'all.jsonl'
    existing = rows(target) if target.exists() else []
    done = {row['id'] for row in existing}
    expected = {row['id'] for row in values}
    if len(done) != len(existing) or not done <= expected:
        raise ValueError('Prediction IDs conflict with frozen input')
    fingerprint = hashlib.sha256(json.dumps({'protocol': sha(out / 'protocol.json'), 'group': group,
                                            'split': split}, sort_keys=True).encode()).hexdigest()
    if any(row.get('run_fingerprint') != fingerprint for row in existing):
        raise ValueError('Cannot resume predictions from another model/protocol')
    started = perf_counter()
    if len(done) != len(values):
        print(f'Loading {group}: {adapter or "official original"}', flush=True)
        model, tokenizer = original.load_local_model(protocol['model']['base_model']['path'], adapter, {'': 'cuda'})
        warmup_started = perf_counter()
        original.generate_raw(model, tokenizer, 'Return a JSON object.', max_new_tokens=1,
                              context_tokens=protocol['generation']['context_tokens'])
        torch.cuda.synchronize()
        metadata = {'group': group, 'split': split, 'run_fingerprint': fingerprint,
            'model': protocol['model'], 'generation': protocol['generation'],
            'synthetic_one_token_warmup_seconds': perf_counter() - warmup_started,
            'gpu': torch.cuda.get_device_name(0),
            'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'accelerate')}}
        save(directory / 'metadata.json', metadata)
        for row in values:
            if row['id'] in done:
                continue
            torch.cuda.synchronize()
            tick = perf_counter()
            item = {'id': row['id'], 'task': row['task'], 'split': split, 'model_group': group,
                'run_fingerprint': fingerprint, 'raw_output': '', 'status': 'success'}
            try:
                raw, metrics = original.generate_raw(model, tokenizer, row['messages'][0]['content'],
                    **{key: protocol['generation'][key] for key in ('max_new_tokens', 'max_time', 'context_tokens')})
                torch.cuda.synchronize()
                item.update(raw_output=raw, generation=metrics)
            except Exception as error:
                item.update(status='error', error=f'{type(error).__name__}: {error}')
                if isinstance(error, RuntimeError):
                    # Device faults should not silently turn the rest of the dataset into failed outputs.
                    with target.open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                        stream.flush()
                    raise
            item['generation_seconds'] = perf_counter() - tick
            with target.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            done.add(row['id'])
            save(out / 'progress.json', {'stage': 'generation', 'split': split, 'model': group,
                'completed': len(done), 'total': len(values), 'last_id': row['id'],
                'last_generation_seconds': item['generation_seconds'], 'process_elapsed_seconds': perf_counter() - started,
                'updated_at': datetime.now(timezone.utc).isoformat()})
            print(f'{split} {group} {len(done)}/{len(values)} {item["status"]} {item["generation_seconds"]:.2f}s', flush=True)
    complete = rows(target)
    for task in TASKS:
        subset = [row for row in complete if row['task'] == task]
        if subset:
            save_rows(directory / f'{task}.jsonl', subset)
    save(directory / 'completion.json', {'complete': len(done) == len(values), 'rows': len(complete),
        'runtime_errors': sum(row['status'] == 'error' for row in complete),
        'predictions_sha256': sha(target), 'run_fingerprint': fingerprint})


def evaluate(protocol, group, split):
    out = verify_protocol(protocol)
    for task, module in TASKS.items():
        gold = out / f'gold/{split}/{task}.jsonl'
        if not gold.exists():
            continue
        prediction = out / f'predictions/{split}/{group}/{task}.jsonl'
        report_dir = out / f'reports/{split}/{group}/{task}'
        report = report_dir / 'report.json'
        if report.exists():
            inputs = load(report)['inputs']
            if inputs['gold']['sha256'] != sha(gold) or inputs['predictions']['sha256'] != sha(prediction):
                raise ValueError('Existing metric report does not match frozen predictions')
            continue
        command = [protocol['eval_python'], '-X', 'utf8', '-m', f'evaluation.{module}']
        if module == 'correction':
            command.append('parallel')
        command += ['--gold', str(gold), '--predictions', str(prediction), '--model', f'{group}_qwen3_1.7B',
                    '--split', split, '--output-dir', str(report_dir)]
        subprocess.run(command, cwd=out / 'code', check=True)


def comparison(protocol, split):
    out = Path(protocol['run_dir'])
    report = {'split': split, 'scope': protocol['scope'], 'models': {},
        'interpretation': 'Report coverage and failures alongside valid-only metrics. Similarity is not feedback correctness. No causal benefit or D filtering claim.'}
    for group in ('A', 'B', 'C'):
        report['models'][group] = {}
        for task in TASKS:
            path = out / f'reports/{split}/{group}/{task}/report.json'
            if path.exists():
                report['models'][group][task] = load(path)
    save(out / f'comparison-{split}.json', report)


def finalize(protocol):
    out = Path(protocol['run_dir'])
    lines = ['A/B/C 同条件自动评测交接', '',
        '数据范围：' + protocol['scope']['test'],
        '评分保留六维原始小数；反馈指标为参考相似度，不能代替内容正确性。',
        '原始失败保留，未修补 JSON，未补造人工分数。',
        '纠错仅有验证集结果；没有纠错正式测试参考。',
        '历史 test 曾用于调试，本轮为冻结方案后的重新评测，不称为从未接触的测试集。',
        'B 的实际训练 JSONL/重采样历史未取得，不能把 B/C 差异完全归因于教师数据。', '']
    def fmt(value):
        return '未定义' if value is None else f'{value:.6f}'
    for split in ('validation', 'test'):
        comparison_report = load(out / f'comparison-{split}.json')
        lines.append('划分：' + split)
        for group, tasks in comparison_report['models'].items():
            scoring = tasks.get('essay_scoring')
            if scoring:
                macro = scoring['macro']
                lines.append(f"{group} 评分：{scoring['fully_valid_samples']}/{scoring['samples']} 篇六维完整；"
                    f"MAE={fmt(macro['mae'])}，RMSE={fmt(macro['rmse'])}，QWK={fmt(macro['qwk'])}，"
                    f"失败惩罚MAE={fmt(macro['failure_penalized_mae'])}")
            feedback = tasks.get('essay_feedback')
            if feedback:
                metrics = feedback['metrics']
                rouge = metrics['rouge']
                bleu = metrics['bleu']
                lines.append(f"{group} 反馈：{feedback['valid_samples']}/{feedback['total_samples']} 条合法；"
                    f"ROUGE-L F1={fmt(rouge['rougeL']['f1'] if rouge else None)}，"
                    f"BLEU(0–100)={fmt(bleu['score_0_100'] if bleu else None)}")
            correction = tasks.get('grammar_correction')
            if correction:
                lines.append(f"{group} 纠错：详见 reports/{split}/{group}/grammar_correction/report.json；"
                             '采用全文单参考 ERRANT，不是官方 BEA 基准。')
        lines.append('')
    lines.extend(['文件位置：', 'predictions/：逐条原始输出、耗时、失败、模型版本及完成清单。',
                  'reports/：复用 5 号程序生成的指标、逐样本记录和失败清单。',
                  'protocol.json：固定的数据指纹、模型、种子和生成参数。',
                  'comparison-validation.json / comparison-test.json：三模型完整汇总。',
                  'human-review/：两位评价者的空白匿名评分材料；private_manifest.json 仅由协调者保管。'])
    (out / '交接说明与指标.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    review = out / 'human-review'
    if not (review / 'private_manifest.json').exists():
        command = [protocol['eval_python'], '-X', 'utf8', '-m', 'evaluation.feedback_review', 'prepare',
                   '--gold', str(out / 'gold/test/essay_feedback.jsonl'), '--split', 'test', '--output-dir', str(review)]
        for group in ('A', 'B', 'C'):
            command += ['--prediction', f'{group}={out / f"predictions/test/{group}/essay_feedback.jsonl"}']
        subprocess.run(command, cwd=out / 'code', check=True)
    packets = rows(review / 'review_packets.jsonl')
    for number in (1, 2):
        directory = review / f'reviewer-{number:02d}'
        directory.mkdir(exist_ok=True)
        shutil.copy2(review / 'review_packets.jsonl', directory / 'review_packets.jsonl')
        if not (directory / 'ratings.jsonl').exists():
            shutil.copy2(review / 'ratings_template.jsonl', directory / 'ratings.jsonl')
        template = ['评价者编号：', '评价日期：',
                    '日期请含时区，例如 2026-11-26T15:00:00+08:00。独立填写；无效输出不打内容分。', '']
        for packet in packets:
            feedback = packet['feedback'] if packet['feedback'] is not None else '\n'.join(packet['raw_outputs'])
            template += [f"===== {packet['output_id']} =====", f"样本编号：{packet['sample_id']}",
                '【作文】', packet['essay'], '【反馈】', feedback, '【仅填写以下栏目】',
                '问题识别准确性：', '建议具体性：', '修改有效性：', '忠实于原文：',
                '证据与理由：', '不适用或分歧说明：', '逐项判断JSON：[]', '']
        form = directory / '人工评分表.txt'
        if not form.exists():
            form.write_text('\n'.join(template), encoding='utf-8')
        rubric = Path(protocol['repo']) / 'evaluation/human/rubric.md'
        shutil.copy2(rubric, directory / '评价标准.md')
        (directory / '填写说明.txt').write_text(
            '四个维度按评价标准独立填写1–5整数；修改有效性可填NA并写明原因。\n'
            '失败输出所有分数留空，并在证据栏说明。不要修改作文、反馈或编号。\n'
            '只接收本目录；不要查看协调者的模型映射。所有初始分数均为空。\n', encoding='utf-8')
        archive = review / f'reviewer-{number:02d}-独立评分材料.zip'
        if not archive.exists():
            shutil.make_archive(str(archive.with_suffix('')), 'zip', directory)
    # Keep a portable internal handoff separate from the GitHub source changes.
    bundle = out / 'team-handoff'
    bundle.mkdir(exist_ok=True)
    for name in ('predictions', 'reports', 'gold', 'code', 'human-review'):
        shutil.copytree(out / name, bundle / name, dirs_exist_ok=True)
    for name in ('protocol.json', 'comparison-validation.json', 'comparison-test.json', '交接说明与指标.txt'):
        shutil.copy2(out / name, bundle / name)
    for group in ('B', 'C'):
        source = Path(protocol['model']['adapters'][group]['path'])
        destination = bundle / 'models' / group
        destination.mkdir(parents=True, exist_ok=True)
        for pattern in ('adapter*.json', 'adapter*.safetensors', 'tokenizer*.json', 'special_tokens_map.json',
                        'vocab.json', 'merges.txt', 'chat_template.jinja', 'generation_config.json', 'README.md'):
            for file in source.glob(pattern):
                shutil.copy2(file, destination / file.name)
    c_train = Path(protocol['model']['adapters']['C']['path']).parent
    train_destination = bundle / 'C-training'
    train_destination.mkdir(exist_ok=True)
    for name in ('run-config.json', 'resolved-training-args.json', 'training-result.json', 'completion.json',
                 'batch-probe.json', 'steps.jsonl', 'metrics.jsonl', 'wrapper-used.py'):
        shutil.copy2(c_train / name, train_destination / name)
    shutil.copy2(c_train.parent / 'train-console.log', train_destination / 'train-console.log')
    shutil.copy2(c_train.parent / 'code/train_qlora.py', train_destination / 'train_qlora.py')
    training_input = Path(load(c_train / 'run-config.json')['source_data'])
    shutil.copy2(training_input.parent / 'c-data-audit.json', train_destination / 'c-data-audit.json')
    shutil.copy2(training_input.parent / 'input-manifest.json', train_destination / 'input-manifest.json')
    save(bundle / 'files-sha256.json', {str(path.relative_to(bundle)): sha(path)
        for path in sorted(bundle.rglob('*')) if path.is_file() and path.name != 'files-sha256.json'})
    archive = out / '5号接手-ABC模型预测与评测.zip'
    if not archive.exists():
        shutil.make_archive(str(archive.with_suffix('')), 'zip', bundle)
    save(out / 'delivery.json', {'internal_bundle': str(bundle), 'zip': str(archive), 'zip_sha256': sha(archive),
        'weights_included': ['B adapter', 'C adapter'], 'official_base_weights_included': False,
        'raw_essays_and_predictions_for_internal_team_only': True, 'human_numeric_ratings_filled': 0})


def run(protocol):
    out = verify_protocol(protocol)
    if (out / 'completion.json').exists():
        print('ABC_EVALUATION_ALREADY_COMPLETE', flush=True)
        return
    for split in ('validation', 'test'):
        for group in ('A', 'B', 'C'):
            subprocess.run([protocol['inference_python'], '-u', '-X', 'utf8',
                str(out / 'code/run_abc_evaluation.py'), 'predict', '--protocol', str(out / 'protocol.json'),
                '--group', group, '--split', split], check=True)
            evaluate(protocol, group, split)
        comparison(protocol, split)
        if split == 'validation':
            # This confirms loading, execution, IDs and file alignment, not model quality.
            completions = [load(out / f'predictions/validation/{group}/completion.json') for group in ('A', 'B', 'C')]
            if any(not item['complete'] or item['runtime_errors'] for item in completions):
                raise RuntimeError('Validation execution failed; formal test has not started')
            save(out / 'validation-gate.json', {'passed': True, 'json_failures_retained': True,
                'scope': 'execution and alignment only; no tuning on test results'})
    finalize(protocol)
    save(out / 'completion.json', {'complete': True, 'completed_at': datetime.now(timezone.utc).isoformat(),
        'expected_test_predictions': sum(protocol['datasets']['test']['count'] for _ in range(3)),
        'protocol_sha256': sha(out / 'protocol.json'), 'human_ratings_complete': False,
        'B_training_data_verified': False, 'D_scheduled': False})
    save(out / 'progress.json', {'stage': 'complete', 'completed_at': datetime.now(timezone.utc).isoformat()})
    print('ABC_EVALUATION_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    for name in ('output', 'base', 'adapter-b', 'adapter-c', 'eval-python', 'existing-pilot'):
        prep.add_argument('--' + name, required=True, type=Path)
    prep.add_argument('--test-scoring', type=int, default=-1, help='-1 means all scoring records')
    prep.add_argument('--test-feedback', type=int, default=-1, help='-1 means all feedback records')
    for name in ('run', 'predict'):
        sub = commands.add_parser(name)
        sub.add_argument('--protocol', required=True, type=Path)
        if name == 'predict':
            sub.add_argument('--group', required=True, choices=('A', 'B', 'C'))
            sub.add_argument('--split', required=True, choices=('validation', 'test'))
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      HF_HUB_DISABLE_TELEMETRY='1', PYTHONUTF8='1')
    awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001) if os.name == 'nt' else False
    try:
        if args.command == 'prepare':
            prepare(args)
        elif args.command == 'predict':
            predict(load(args.protocol), args.group, args.split)
        else:
            run(load(args.protocol))
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
