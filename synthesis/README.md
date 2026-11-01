# 成员 3：教师反馈合成与质量检查

本模块用于英语作文反馈试生成：调用教师模型，保存原始响应，检查 JSON/Schema/引用，产生质量与 token/估算费用记录。Python 3.9+；离线验证不需要 GPU 或 API Key。

## 快速离线验证

在仓库根目录执行：

```powershell
python -m pip install -r requirements-synthesis.txt
python -X utf8 -m synthesis demo
python -X utf8 -m unittest discover -s tests -q
```

demo 使用 examples/demo_fixtures.json 中自写作文和预设响应，不联网。预设 10 条中，8 条格式合格、6 条规则通过，仅用于程序验证，不是教师质量指标。重复执行跳过已保存的响应；修改代码、Prompt 或配置后使用新的运行目录。

## 目录及接口

| 路径 | 用途 |
|---|---|
| synthesis/ | 生成、校验、筛选、抽检和等量 C/D 候选导出 |
| schemas/ | 作文输入和反馈 JSON Schema |
| configs/prompts/ | feedback_v1～v4 提示词，保留版本 |
| configs/teacher.example.json | 通用 Chat Completions 配置模板 |
| configs/deepseek*.json | 已验证的教师设置及报价快照，不含密钥 |
| scripts/ | 下载固定版本、数据检查、试生成、定位辅助与本地汇总 |
| tests/ | 使用本地样例和模拟服务的回归测试 |
| docs/member3/ | 数据接口、人工审核和第一周统计说明 |

此模块的输出 Schema 用于 essay_feedback 训练任务，包含 summary、feedback、suggestions；与仓库现有整份网页报告的 scores/errors 接口用途不同。评分标签沿用源数据；接口集成由 1、2、5 号确认。

## 教师配置

真实调用前，复制模板为 configs/teacher.local.json，填写账号对应的接口地址和模型名称。当前公共 DeepSeek 设置以服务端模型名 deepseek-flash 调用，报价文件是历史快照，后续批次应核对服务商文档及账户账单。

密钥通过配置的环境变量 TEACHER_API_KEY 读取；可在用户自己的 PowerShell 中输入：

```powershell
$env:TEACHER_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'Teacher API Key' -AsSecureString)).Password
```

密钥只用于当前终端及其启动的进程。配置模板、日志和训练标签不保存密钥。程序不会自动读取 .env；若小组使用 .env，请先在本机加载到环境变量，再运行脚本。

## 通用训练作文试生成

输入格式见 [数据接口](../docs/member3/data-contract.md)，正式输入必须为 train，保留 source/id/dataset_version。示例路径由成员 2 交接后准备：

```powershell
python -X utf8 -m synthesis generate --input data/processed/pilot_train.jsonl --config configs/teacher.local.json --reserved data/processed/reserved.jsonl --run-dir results/runs/pilot_v1 --max-new 5
python -X utf8 -m synthesis filter --run-dir results/runs/pilot_v1
python -X utf8 -m synthesis audit --run-dir results/runs/pilot_v1 --output results/runs/pilot_v1/audit.jsonl --count 20
```

这些命令会调用配置的教师服务并产生用量。--max-new 限制本次处理的新作文数，重试上限由配置决定。--reserved 只用于本地交叉检查，不把验证/测试文本发给教师；近重复与官方划分来源仍由成员 2 审计。

原始回答先保留，再做筛选；不自动修复坏 JSON、重复键、超限条目或虚构引用来伪造通过结果。自动规则不判断所有语法/语义错误，人工审核状态初始为 pending。

## 本项目第一周的可复现路径

成员 2 的数据位于独立 PR/交付包。下载器按固定提交取得文件并校验 Git blob 哈希，数据仅保存在本机：

```powershell
python -X utf8 scripts/fetch_dataset_snapshot.py
python -X utf8 scripts/audit_handoff.py
python -X utf8 scripts/prepare_ellipse_pilot.py
```

真实调用的历史批次脚本包括 test_deepseek.py、run_first5.py、run_remaining45.py、run_guided5.py、run_new50_v4.py；它们固定了样本数、版本与输入指纹。已有运行目录只能按同版本续跑，不要删除日志或改变 Prompt 后混跑。

第四版在输入端附上程序提供的唯一原文片段，每篇最多选择 6 个较完整片段；输出 Schema 仍允许最多 20 条，原有筛选规则不变。脚本会另存实际逐篇 Prompt 和候选复制诊断，不把定位辅助完整执行当作默认事实。

同一批次中出现服务错误时先核查；质量失败原样保存，不重生成到通过为止。运行前应确认样本、费用预算和教师输出的训练使用条件。

## 结果与人工审核

results/runs/ 保存原始响应、输入快照、配置/代码/Prompt 指纹、质量和成本记录。data/ 与 results/runs/ 中的实际作文/日志未随模块上传，应按小组约定单独共享。

阅读 [人工审核说明](../docs/member3/review-guide.md)。可以用 Markdown 阅读版填写判断，但真人未确认的结论和评分必须保持待审核。费用估算不等于实际扣费；正式使用前要核对来源、版本和审核策略。

第一周试生成统计见 [报告](../docs/member3/week1-report.md)。当前 41 项离线测试通过，测试不访问真实 API。
