# A/B/C 同条件推理与自动评测

本入口复用根目录 `inference.py` 的 Qwen3 加载、Adapter 合并及生成逻辑，并调用既有 `evaluation.scoring`、`evaluation.feedback`、`evaluation.correction`。不重新训练、不修补输出，也不替代独立人工评价。

C 的训练依赖版本另存于 `requirements-training-qlora.txt`。本机实际 PyTorch 为 2.9.1+cu128；在独立训练环境安装适合设备的 CUDA 构建，避免把 CPU 构建当成 GPU 环境。训练复现使用原有 `train_qlora.py --data_path <交付的新JSONL> --output_dir <新模型目录> --max_steps 1500 --max_length 512`，其余配置沿用代码及保存的 `resolved-training-args.json`。真实批次的 labels/EOS 检查结果另见训练交付记录。

`c-data-audit.json` 是训练前的数据检查快照，其中 `training_performed=false` 只表示审计程序本身没有训练。正式训练是否完成，以 `C-training/completion.json`、步数和权重校验记录为准。

## 本轮范围

- A：官方 Qwen3-1.7B；B：4 号交付的 Adapter；C：从同一官方模型独立训练 1500 步的 Adapter。
- C 使用新交付的 `train_C_fixed(1).jsonl`，沿用原程序 512 token 超长整条过滤，不重采样。实际保留原始 2857 条、教师 134 条。
- 验证：既有 5 篇反馈作文，加 seed=42 选定的评分、纠错各 5 条。三模型使用同一批 15 条记录。
- 本轮测试：评分 50 条、反馈 20 条，按任务均匀无放回抽样，seed=42，先冻结 ID，再生成三模型输出。属于抽样结果，不代表已完成 3070 条全量测试。
- 测试包没有带参考答案的纠错样本，纠错只报告验证集全文单参考 ERRANT，不称为官方 BEA 成绩。
- 历史 test 曾用于调试，本轮为固定方案后的重新评测，不称为从未接触的测试集。
- B 实际使用的训练 JSONL、是否重采样仍需原训练者核对，因此 B/C 差异不能直接解释为教师数据的纯因果效果。D 不纳入本轮。

## 固定参数

三模型均使用 BF16、CUDA、batch size=1、seed=42、greedy decoding、repetition_penalty=1.05、enable_thinking=False、eos_token_id=151645。每条最多生成 512 token，max_time=180 秒。输入及预留输出合计上限 4096 token，超限明确记录失败，不截断。

训练 `max_length=512` 是训练预处理限制；本地官方配置的原生位置上限为 40960。4096 是本轮统一的推理预算，不改变桌面工作台现有输入限制。每次加载后先用固定合成提示生成 1 token 预热，预热不计入评测样本。

## 运行

训练数据核查可使用`python training/audit_c_data.py --data-path <C训练JSONL> --tokenizer <本地官方分词器目录> --output-dir results/runs/c-audit-v1`。程序复用原有标签与正文检查，并直接执行当前train_qlora.py的预处理函数，报告原始/教师数量、长度过滤及与验证/测试集的ID和规范化正文重合，不重采样或训练。输出目录必须不存在，默认max-length为512；训练入口仍使用原train_qlora.py。

推理使用已有的独立 GPU 环境。自动指标使用独立评测环境，按 `requirements-evaluation.txt` 安装，并安装官方 spaCy `en_core_web_sm`。基础 ROUGE/BLEU、ERRANT 不依赖 BERTScore；本轮未启用 BERTScore 或外部裁判 API。

先准备本地模型：基础模型目录须包含完整权重、分词器和 `verified-weights.json`（权重文件名到 SHA-256 的映射）；B/C 目录须包含 Adapter 和配置。基础权重校验文件应来自可信下载校验结果，不能用未核实的本地哈希代替来源核对。

以下路径为示例，按机器实际位置填写。`existing-pilot.jsonl` 是先前冻结的 5 条反馈金标准；内部交付包的 `gold/validation/essay_feedback.jsonl` 可用于复现。

```powershell
python scripts/run_abc_evaluation.py prepare --output results/runs/abc_v1 --base models/Qwen3-1.7B --adapter-b models/B --adapter-c models/C --eval-python .venv-evaluation/Scripts/python.exe --existing-pilot existing-pilot.jsonl --test-scoring 50 --test-feedback 20
python scripts/run_abc_evaluation.py run --protocol results/runs/abc_v1/protocol.json
```

若明确运行全量测试，将两个抽样参数都设为 `-1`，并使用新的输出目录。已有抽样结果不冒充全量结果。

`prepare` 冻结金标准、代码快照、模型及数据指纹。`run` 依次完成三模型验证、自动指标和执行检查，再进入测试；验证检查保证运行和对齐，不以 JSON 成功率筛选模型或调参。所有正式结果生成后，自动汇总指标、准备两份空白匿名评分材料并生成内部交付压缩包。

断点续跑重复执行同一条 `run` 命令。已保存的失败不会被自动重试以提高通过率；模型、数据或代码指纹变化会拒绝混用。文件损坏时应先保留原件并检查，不能静默删掉半行或失败记录。已完成的指标目录不会覆盖。

## 输出和交付

- `progress.json`、`run-console.log`：当前模型、划分、完成数、逐条耗时和运行日志。
- `predictions/<split>/<group>/`：原始输出、执行错误、结束状态、版本、逐任务 JSONL 和完成清单。
- `reports/<split>/<group>/<task>/`：直接由 5 号既有程序生成的指标、逐样本结果、失败及来源指纹。
- `comparison-validation.json`、`comparison-test.json`：三模型完整报告汇总；合法样本上的指标必须结合覆盖率解释，各模型有效子集可能不同。
- `交接说明与指标.txt`：可读的指标表和局限说明。
- `human-review/reviewer-01/`、`reviewer-02/`：独立评分表、匿名原文/反馈、量表；分数留空。`private_manifest.json` 只由协调者保管，不放进评价者压缩包。
- `team-handoff/`、`5号接手-ABC模型预测与评测.zip`：内部交付的 B/C Adapter、C 训练记录、推理快照、真实预测及评测报告；不重复打包官方基础权重。
- `completion.json`：仅在全部运行、指标和交付包生成后出现。其 `complete=true` 表示本轮预声明范围完成，不代表全量测试、真人评分或 B 训练数据核验完成。

真实作文、原始预测和模型权重沿用项目本地保存约定。GitHub 提交可复用源码、测试和说明；内部交付包另行共享，不上传整个项目目录。

评分保留六维小数，报告 MAE/RMSE/QWK 与补充失败惩罚 MAE。反馈报告 ROUGE/BLEU 和失败率，相似度不能代替内容正确性；不预设 C 一定提升，也不讨论 D 的筛选效果。

## 检查

```powershell
python -m unittest discover -s tests -p "test_*evaluation.py" -v
```

新增检查覆盖固定抽样、全量选取不漏样本、重复 ID、错误划分、原文/参考消息分离、冻结数据及代码变化、原始输出覆盖保护。真实模型加载和生成另由验证流程记录。
