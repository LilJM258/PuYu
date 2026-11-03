# 成员 5：ELLIPSE 评分评测初版

入口：`python -m evaluation.scoring`，只用 Python 标准库，无需 GPU、模型下载或教师 API。

## 已确定的规则

- 六维为 cohesion、syntax、vocabulary、phraseology、grammar、conventions。
- 人工与预测分数均保留原始小数，合法范围 1～5；不取整、不吸附到半分档、不截断、不补造评分。
- 有效性按维度判断：五维合法、一维非法时，五维正常计分，一维失败；另统计六维全部有效的作文比例。
- 有效预测逐维计算 MAE、RMSE、QWK，报告每项分母与覆盖率。
- “计入失败惩罚的平均绝对误差”是项目自定义补充指标：有效维度取实际绝对误差，失败维度取 4，再除以全部应评维度数。它不是常规 MAE，也不是将失败填成预测 1 分。
- 宏平均为有定义维度的算术平均，同时输出参与平均的维度数，不能把少数维度的平均伪装为完整六维结果。

## 不取整的 QWK 口径

使用二次分歧权重 `w(a,b)=((a-b)/4)^2`，按原始分数的实际距离计算。观测分歧取配对样本平均，期望分歧取真实/预测各自经验边际分布独立配对的平均：

```text
QWK = 1 - observed_squared_disagreement / expected_squared_disagreement
```

分母中的共同尺度 4² 可约去，代码直接计算平方距离。对于完整等距评分网格（如 1、1.5、…、5），与在该网格上使用等级索引二次权重的 QWK 等价。3.7 等其他合法小数也按实际距离计算，不视为非法、不四舍五入。运行记录明确标为 `quadratic_actual_score_distance`。

这是保留数值间距的口径，不能与“先取整再计算”的结果无说明混比，也不能把观测到的不同小数压缩成等距离类别序号。传统分类实现使用类别索引权重，参考 [scikit-learn 官方实现](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_classification.py)；本项目明确使用实际分数距离，推广到网格外小数时应披露此差异。

没有有效配对，或双方在全部样本上都给同一个相同分数时，QWK 无定义，输出 null 与原因，不强行写成 1。MAE/RMSE 仍按自身定义计算。

## 输入

### 人工答案

直接读取 training_data_v1/validation.jsonl 或 test.jsonl 中的 ELLIPSE、essay_scoring 样本，从 assistant 消息读取 scores。通过原始 ID 匹配，不依赖行序。非评分任务跳过。

人工答案缺失、重复 ID、错误 split 或非法分数会停止评测，不把坏人工标签算成模型失败。开发使用验证集或合成小样例；测试集仅在评测方案固定后用于正式评测。

### 模型预测

每个 UTF-8 JSONL 文件仅对应一个模型和一个划分。推荐保存原始输出：

```json
{"id":"ellipse:example","raw_output":"{\"scores\":{\"cohesion\":3.5,\"syntax\":3.7,\"vocabulary\":4.0,\"phraseology\":3.5,\"grammar\":3.0,\"conventions\":4.0}}"}
```

也支持上游仅做忠实 JSON 解析、未修复的 scores：

```json
{"id":"ellipse:example","scores":{"cohesion":3.5,"syntax":3.7,"vocabulary":4.0,"phraseology":3.5,"grammar":3.0,"conventions":4.0}}
```

若 raw_output 与 scores 同时存在，以 raw_output 为准，不能用修复后的 scores 覆盖坏输出。生成异常可以记录 `{"id":"...","status":"error","raw_output":"原始返回或错误记录"}`。文件中未提供某个应评 ID，也计为该篇六维失败。

## 异常处理

| 情况 | 处理 |
|---|---|
| 一维缺失、非数字、布尔值、越界 | 仅该维失败，其余合法维度保留 |
| 缺少整个 scores、原始回答不是合法 JSON、生成报错 | 该篇六维失败 |
| 同一个应评 ID 多次出现 | 六维失败，不任取一个有利结果 |
| JSON 重复键、非有限数字、指数溢出 | 不修复；记录对应非法原始行/回答 |
| JSONL 行不可解析或没有 ID | 保存 unassigned；无法归属的应评 ID 仍计为缺失预测，不凭行序猜测 |
| 不属于本次人工答案的 ID | 保存 unknown_ids，不扩大评测分母 |
| 额外评分键 | 单独记录，六个已知维度仍分别判断；本程序不是整份报告 Schema 的严格通过率检测 |

## 运行与输出

先跑合成算例（明确不是模型效果）：

```sh
python -m evaluation.scoring --gold evaluation/examples/gold.jsonl --predictions evaluation/examples/predictions.jsonl --model synthetic_example_only --split debug --output-dir results/scoring_example_v1
```

该算例有 3 篇、18 个应评维度，其中 1 个维度故意越界。合法小数 3.7 保持原值。预期：17/18 维度有效，2/3 篇全部有效，宏平均 MAE=1/18≈0.05556，失败惩罚宏平均=5/18≈0.27778。

模型正式预测文件准备好后，例如：

```sh
python -m evaluation.scoring --gold training_data_v1/validation.jsonl --predictions results/predictions/B_validation.jsonl --model B_checkpoint_name --split validation --output-dir results/evaluation/B_validation_v1
```

输出目录必须尚不存在，避免覆盖历史结果。输出包括：

- report.json：逐维指标、宏平均、覆盖率、失败原因计数、方法版本、输入文件路径和 SHA-256。
- samples.jsonl：每篇六维的真实值、预测值、合法状态和原始预测记录。
- failures.jsonl：按失败维度记录原因及原始预测记录。
- unassigned.jsonl：无法归属到 ID 的输入行。
- unknown_ids.jsonl：本次答案集中不存在的预测 ID。

## 验证与边界

```sh
python -m unittest discover -s tests -p 'test_scoring_evaluation.py' -v
```

测试使用合成小样例验证手算误差、实际距离 QWK、半分网格等价性、保留小数、部分失败、全失败、重复 ID、坏 JSON、原始输出优先与命令行输出；不使用测试集选择规则。

当前一次评测一个模型。多模型共同有效子集的辅助对照尚未实现；主结果仍应报告各自覆盖率与全量失败惩罚，不能只挑共同成功样本。运行耗时、推理显存、纠错与人工反馈质量也不属于此程序的评分指标。
