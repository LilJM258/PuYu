# 语法纠错：ERRANT Precision / Recall / F0.5

两种入口有不同评测单位，报告中明确标识；不通过全文字符串完全相同来代替纠错质量。保留原文、预测、参考、错误状态及版本。

## 安装

在独立评测环境执行：

```sh
python -m pip install -r requirements-evaluation-gec.txt
python -m spacy download en_core_web_sm
```

M2 比较不加载英文模型，全文自动提取编辑需要英文模型。程序不在运行时暗中下载。

## 一、全文参考入口：可先用于现有验证数据

读取 WI_LOCNESS/grammar_correction 的 messages（精确移除已知任务指令），或使用带 input/output 的准备格式。预测每行采用原始样本 ID：

```json
{"id":"wi_locness:example:a0","raw_output":"{\"corrected_text\":\"The complete corrected essay.\"}"}
```

也支持未经修复的 corrected_text 字符串字段。空文本、坏 JSON、生成异常、缺失/重复 ID 均失败；未修改的原文可以是合法预测，是否应当修改由参考编辑决定。

```sh
python -m evaluation.correction parallel --gold evaluation/examples/correction_gold.jsonl --predictions evaluation/examples/correction_predictions.jsonl --model synthetic_only --split debug --output-dir results/gec_example_v1
```

验证集示例：

```sh
python -m evaluation.correction parallel --gold training_data_v1/validation.jsonl --predictions results/predictions/B_correction_validation.jsonl --model B_checkpoint --split validation --output-dir results/evaluation/B_gec_validation_v1
```

实现步骤：

1. 通过 ID 对齐完整原文、人工纠正全文和模型纠正全文，保留未经修改的原始记录。
2. 仅为 token 对齐折叠空白（含换行），使用 ERRANT + spaCy 分词和编辑提取，分别生成参考与预测 M2。每个完整记录作为一个对齐单位，不擅自按标点切句后强行对齐。
3. 使用官方 errant_compare 的 span-based correction 模式选择/比较编辑；记录 TP、FP、FN，然后按原始计数重新计算 Precision、Recall、F0.5，避免仅保留 CLI 显示时的四位小数。
4. 保存 M2、原始 ERRANT 输出、依赖/英文模型版本、输入指纹、有效数量、失败记录和诊断性全文匹配率。

Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F0.5=1.25PR/(0.25P+R)。遵循 ERRANT 空分母约定：没有预测编辑时 precision=1，没有参考编辑时 recall=1；两边均无编辑时 F0.5=1，须结合编辑计数解释。这些约定不是对损坏/缺失生成打满分。

**限制：** 这是“全文单参考自动提取编辑”协议，不是官方 BEA 逐句 M2 成绩；参考可能遗漏其他合理改法。若同一原始 ID 有多个 :a0/:a1 标注记录，会拒绝按独立作文重复计数，要求先准备多参考 M2。当前验证集可用于开发，test 包没有带答案纠错样本；不能自动改用训练集宣称测试成绩。

合法预测上的 ERRANT 分数必须与覆盖率一起看。非法生成不被伪造成“未修改原文”，也不混入有效编辑指标；本版不为其发明固定 F0.5 惩罚，失败单独报告。模型间正式比较应冻结同一数据与失败规则。

## 二、已对齐 M2 入口：保留官方参考标注与多标注者

```sh
python -m evaluation.correction m2 --reference-m2 evaluation/examples/gec_reference.m2 --hypothesis-m2 evaluation/examples/gec_prediction.m2 --model synthetic_only --split debug --output-dir results/gec_m2_example_v1
```

要求每个 block 一条 S 原句和合法 A 编辑/noop；预测与参考的 S 行数量、顺序、分词必须完全一致。保留多标注者编号，由 ERRANT 执行参考选择，不自行挑选最有利的文本。输出计数、P/R/F0.5、输入版本和原始比较输出。

M2 本身不提供生成失败信息，因此不凭 M2 推断“100% 生成成功”；需要另行保存完整生成清单和覆盖率。使用官方数据、同样分词及同版工具时才讨论与该基准的可比性。ERRANT 文档提示新版结果可能与 BEA-2019 版本不同，本程序不自动声称官方成绩。

参考：[ERRANT 作者文档](https://github.com/chrisjbryant/errant)、[官方比较代码](https://github.com/chrisjbryant/errant/blob/main/errant/commands/compare_m2.py)。本程序不修改官方参考，也不从 LEAF 反馈拼接 corrected_text。
