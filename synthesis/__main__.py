"""Run from the project root: python -m synthesis --help."""

import argparse
import json
import sys
from pathlib import Path

from .common import ROOT, read_json, write_jsonl
from .pipeline import create_audit, export_matched, filter_run, generate, report_audit


def main():
    parser = argparse.ArgumentParser(description="3号：教师反馈合成与筛选（mock 可离线运行）")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="离线验证；不调用教师，不产生真实质量指标")
    demo.add_argument("--run-dir", default="results/runs/demo")
    gen = commands.add_parser("generate", help="生成并保存原始响应；同一配置可断点续跑")
    gen.add_argument("--input", required=True)
    gen.add_argument("--config", required=True)
    gen.add_argument("--run-dir", required=True)
    gen.add_argument("--reserved", help="2号的 validation/test JSONL；仅在本地比对，不发给教师")
    gen.add_argument("--max-new", type=int, help="本次最多处理的新作文数量；重试仍计费")
    filt = commands.add_parser("filter", help="生成 C/D 候选池及质量统计")
    filt.add_argument("--run-dir", required=True)
    audit = commands.add_parser("audit", help="生成人工抽检 JSONL，不覆盖已存在文件")
    audit.add_argument("--run-dir", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--count", type=int, default=20)
    audit.add_argument("--seed", type=int, default=42)
    report = commands.add_parser("audit-report", help="汇总已完成的人工抽检；待审核项不计入通过率")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--audit", required=True)
    report.add_argument("--output", required=True)
    export = commands.add_parser("export", help="按固定随机种子导出等量 C/D")
    export.add_argument("--run-dir", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--count", type=int)
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--audit")
    export.add_argument("--require-human", action="store_true")
    export.add_argument("--allow-debug", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "demo":
            run_dir = Path(args.run_dir)
            fixtures = read_json(ROOT / "examples" / "demo_fixtures.json")
            input_path = run_dir / "demo_input.jsonl"
            write_jsonl(input_path, [f["input"] for f in fixtures])
            generated = generate(input_path, ROOT / "configs" / "mock.json", run_dir)
            filtered = filter_run(run_dir)
            audit_path = run_dir / "human_audit.jsonl"
            if not audit_path.exists():
                create_audit(run_dir, audit_path, count=len(fixtures))
            result = {"generation": generated, "filter": filtered, "audit": str(audit_path)}
        elif args.command == "generate":
            result = generate(args.input, args.config, args.run_dir, args.reserved, args.max_new)
        elif args.command == "filter":
            result = filter_run(args.run_dir)
        elif args.command == "audit":
            result = create_audit(args.run_dir, args.output, args.count, args.seed)
        elif args.command == "audit-report":
            result = report_audit(args.run_dir, args.audit, args.output)
        else:
            result = export_matched(args.run_dir, args.output_dir, args.count, args.seed,
                                    args.audit, args.require_human, args.allow_debug)
    except (ValueError, OSError, KeyError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
