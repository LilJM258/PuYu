# 5 号：三任务评测工具

更新：2026-11-20（程序始于2026-11-09）。评分、作文反馈、语法纠错分别评测，不要求一个模型回答同时包含三种结果。模型原始输出、ID、失败记录和输入版本保留；不在评测前修复、补造或静默丢弃失败样本。

| 任务 | 入口 | 已实现 | 详细说明 |
|---|---|---|---|
| 六维评分 | `python -m evaluation.scoring` | MAE、RMSE、实际分数距离 QWK、逐维失败与 4 分误差惩罚 | [评分](scoring-readme.md) |
| 作文反馈 | `python -m evaluation.feedback` | ROUGE-1/2/L、BLEU、可选 BERTScore，覆盖率与原始失败记录 | [反馈](feedback-readme.md) |
| 反馈内容质量 | `python -m evaluation.feedback_review` | 匿名评审材料、6 号量表 v0.2 评分汇总、评审一致性与分歧；可导入有来源记录的外部裁判分数 | [反馈](feedback-readme.md) |
| 语法纠错 | `python -m evaluation.correction` | ERRANT Precision/Recall/F0.5；全文单参考、已对齐官方 M2 两种入口 | [纠错](correction-readme.md) |

反馈相似度不能证明建议正确；人评汇总不会自动替人判断。没有实现付费 API 裁判调用或原版 FeedEval 训练，外部 LLM 评分与人评严格分开。GEC 全文自动提取编辑结果不冒充官方 BEA 逐句基准成绩。

## 环境与安装

评分与人评材料/汇总只依赖 Python 标准库（Python 3.9+）。自动反馈/GEC 建议使用独立 Python 3.12 虚拟环境，避免修改训练或网页环境：

```sh
python -m venv .venv-evaluation
# macOS/Linux
source .venv-evaluation/bin/activate
# Windows PowerShell 使用 .venv-evaluation\Scripts\Activate.ps1
python -m pip install -r requirements-evaluation.txt
python -m spacy download en_core_web_sm
```

BERTScore 单独可选安装：

```sh
python -m pip install -r requirements-evaluation-bert.txt
```

反馈基础依赖不下载模型；GEC 英文模型约 13 MB。BERTScore 需显式指定模型，可能下载较大的公开权重；也可以使用本地模型目录。程序不调用付费 API。

**与 Web 环境分开：** 本次验证的 Transformers 4.57.6 要求 huggingface-hub<1，而 main 的 Gradio 6.31.0 要求 huggingface-hub>=1.16，不能同时满足。不要把 BERTScore 依赖装入现有网页环境，也不为此修改 requirements-web.txt。不同环境可以读取同一份预测文件并输出评测报告。

## 数据与复现

- 原始训练包的 messages 格式可直接读取；也支持带 id/task/source/split/input/output 的准备格式，具体见各任务说明。
- 通过 ID 匹配，不按行序猜测。原始输出优先于解析结果，重复 ID 失败，额外 ID 单独记录。
- 开发用 debug 合成样例和 validation。固定模型/规则后才做最终 test 评测，不为缺失测试数据而改用训练集。
- 当前包有 400 条 LEAF 验证参考和 220 条 WI_LOCNESS 验证记录；test 有 LEAF 参考，但没有带答案的 GEC 样本。纠错正式测试仍需 2 号提供；若已有官方对齐 M2，可直接用 M2 入口。
- 输出目录必须不存在，避免覆盖先前报告。报告记录输入 SHA-256、算法与依赖版本；不同样本/版本的分数不可无说明直接比较。
- 多模型共同有效子集自动比较尚未实现；各模型的成功样本指标必须结合覆盖率解释。反馈/GEC 不沿用“评分失败加 4 分”的单位特定惩罚，不伪造低分或空改法。

## 测试

```sh
python -m unittest discover -s tests -p 'test_scoring_evaluation.py' -v
python -m unittest discover -s tests -p 'test_feedback_evaluation.py' -v
python -m unittest discover -s tests -p 'test_correction_evaluation.py' -v
python -m unittest discover -s tests -p 'test_bertscore_integration.py' -v
```

BERTScore 接口测试使用本地构造的微型随机模型，仅验证真实库调用、文件读取和不静默截断，不代表语义质量。缺少相应可选依赖/英文模型时测试会明确跳过；本次已在独立环境实际运行所需依赖验证。Web 回归在独立 Web 环境运行，不把依赖冲突误报为模型错误。

## 当前模型与小批次联调

项目已统一官方Qwen3-1.7B，不能把旧Qwen2.5结果当新版对照。2026-11-20的val_tuned为10条评分＋10条反馈：评分全部有效，MAE=0.4500、RMSE=0.6050、QWK=0.0840；反馈9条有效。这里只说明程序已接通，不能作为正式模型效果结论，参见 [持续摘要](../docs/member5/evaluation-status.md)。

混合预测先按task拆分，对交付ID抽取同划分参考子集；直接以全部validation为10条小批次的应评范围会错误计入未交付预测。保留原始输出和失败，不修补JSON。根 inference.py 当前默认test且limit=-1会漏每池最后一条，不直接复制其默认命令做验证集或完整测试；见 [入口说明](../docs/model-migration.md)。训练Qwen3的依赖版本须另核对，不能将可选BERTScore的最低版本当成训练环境要求。

2026-11-20 同步后回归：共120项，119项通过，1项BERTScore集成因未安装可选依赖明确跳过；ERRANT及英文模型已实际验证。历史微型BERT接口通过记录不替代本次跳过状态；本次未训练学生模型、未进行GPU验证。
