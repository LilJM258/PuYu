# 教师反馈合成：LEAF 格式

2026-11-03 起，新批次使用 `{"feedback":"一段英文反馈"}`，与人工 LEAF 样本的目标结构及训练任务指令一致。不要求独立的总体评价、原文片段、改法或解释字段。

## 离线验证

```sh
python -m pip install -r requirements-synthesis.txt
python -m synthesis demo
python -m unittest discover -s tests -q
```

默认 demo 使用 configs/mock_leaf.json 和 examples/demo_leaf_fixtures.json，不调用 API，不产生真实质量结论；4 个预设样例中 2 个格式合格，D 内容审核仍为 pending。

## 新批次配置与生成

复制 configs/teacher.leaf.example.json 为 configs/teacher.local.json，填写 HTTPS 接口和教师模型，保持 feedback_format=leaf_v1 和 feedback_language=English。DeepSeek 新格式配置为 configs/deepseek_leaf.json；服务可用性和价格仍需在真实调用前确认。密钥通过 TEACHER_API_KEY 环境变量传入，程序不自动读取 .env。

```sh
python -m synthesis generate --input data/processed/pilot_train.jsonl --reserved data/processed/reserved.jsonl --config configs/teacher.local.json --run-dir results/runs/leaf_pilot_v1 --max-new 5
python -m synthesis filter --run-dir results/runs/leaf_pilot_v1
python -m synthesis audit --run-dir results/runs/leaf_pilot_v1 --output results/runs/leaf_pilot_v1/audit.jsonl --count 20
```

真实 generate 会产生 API 用量。示例输入需由训练作文准备，不能直接复制路径后假定数据存在。原始响应和版本指纹保留，格式失败不修复为合格标签，不重生成到通过。

## 筛选和训练导出

输入与输出规范见 docs/member3/data-contract.md 及 schemas/feedback_leaf.schema.json。

C 是格式合格池；整段反馈不能继续使用旧的引用匹配、位置唯一性和局部改法规则。程序对新格式提示需要人工内容审核，quote_match_rate 不适用。filtered/D_pool.jsonl 只表示待审核候选。

人工填写 audit.jsonl：检查判断是否正确、建议是否具体、是否忠实于原文及是否虚构问题。新格式 D 导出强制要求人工 accept，不接受 pending。评分未填写时保持 null，不由助手代替真人判断。

```sh
python -m synthesis audit-report --run-dir results/runs/leaf_pilot_v1 --audit results/runs/leaf_pilot_v1/audit.jsonl --output results/runs/leaf_pilot_v1/audit_report.json
python -m synthesis export --run-dir results/runs/leaf_pilot_v1 --audit results/runs/leaf_pilot_v1/audit.jsonl --output-dir data/synthetic/leaf_matched_v1 --count 10
```

导出数不能超过合格 C 与人工接受 D 的可用数量。mock 导出需额外 --allow-debug，输出保持 debug，不能用于正式训练。messages 使用与 LEAF 相同的 user 指令和 feedback 字符串答案；保留原文及来源信息，避免训练/验证/测试泄漏。

## 历史版本与恢复

configs/teacher.example.json、deepseek.json、deepseek_first5*.json、deepseek_guided5_v4.json、mock.json、feedback_v1～v4、schemas/feedback.schema.json 和旧定位辅助/周报脚本属于 structured_v1 历史方案，仅供旧版本复现。未标 feedback_format 的旧配置按历史格式解析；新配置必须明确 leaf_v1。

第一周旧结果见 docs/member3/week1-report.md，不改写其数字或输出。旧数据不自动拼接成新标签，真实 LEAF 教师反馈需新批次生成。修改 Prompt、格式或代码后使用新的运行目录，禁止混跑。

运行锁 .running.lock 表示同一目录有任务进行中；仅在确认对应进程已停止后手工清除失效锁。续跑使用同配置、输入和代码，变更指纹会拒绝继续。
