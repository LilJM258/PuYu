# PuYu 作文批改 Web 模块

基于 Gradio 的作文批改演示界面，独立展示 ELLIPSE 六维评分、LEAF 英文反馈和 GEC 完整纠正文本。当前支持固定模拟样例与 JSON 联调，尚未连接真实模型。

## 功能

- 六维评分保留 1–5 原始小数，不取整、不补分、不生成综合分。
- 三个任务独立接收与校验；缺失或失败的任务不覆盖其他有效结果。
- 原文红底表示明确错误、黄底表示可选改进、绿底表示亮点；标注来自模拟样例元数据，悬停可查看原因。
- 纠正文本用蓝底突出有变化或新增的整句；删除整句时单独提示。页面包含颜色图例。
- 提供人工反馈评价量表、空白记录和校准示例。

## 安装与运行

Python 3.11–3.13；验证环境为 Python 3.13。以下命令从仓库根目录执行。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe run_web.py
```

macOS / Linux：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-web.txt
.venv/bin/python run_web.py
```

访问 http://127.0.0.1:7860；使用 `--port 7861` 可指定其他端口。Windows 也可使用 `run_web.bat`。默认仅监听本机，不需要模型权重、GPU 或 API 密钥。

## 使用

选择样例后点击“载入样例”和“生成批改报告”。JSON 联调模式下，评分、反馈和纠错分别粘贴到对应输入框，允许只提供一项。

原文三色标注只适用于精确匹配的模拟样例，编辑原文后不沿用原有标注。蓝色标记根据句子差异生成，不表示修改必然正确；按句末标点和换行近似分句，缩写可能影响边界。模拟结果不计入模型实验。

## 验证与资源重建

Windows 示例；其他系统使用相应虚拟环境 Python 路径：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_web_*.py" -v
.\.venv\Scripts\python.exe scripts\build_web_fixtures.py
```

浏览器检查为可选开发步骤。安装 `requirements-web-dev.txt`，保持 Web 服务运行后执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-web-dev.txt
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe scripts\check_web.py
```

已有本机 Edge 时可改用 `--browser msedge`，无需安装 Chromium。`--url` 可指定服务地址；检查结果默认写入 `results/web_smoke/`，`--output-dir` 可指定其他目录。

## 目录

| 位置 | 内容 |
|---|---|
| `run_web.py`、`web/` | 启动入口与界面 |
| `inference/schema.py`、`inference/backend.py` | 校验与固定模拟数据入口 |
| `configs/web_report.schema.json` | 展示层任务类型 Schema |
| `data/mock/examples.json` | 原创模拟作文、输出与人工颜色标注 |
| `evaluation/human/` | 人工反馈评价材料 |
| `tests/test_web_*.py` | Web 模块逻辑测试 |
| `scripts/build_web_fixtures.py`、`scripts/check_web.py` | 样例重建与浏览器检查 |

接口细节见 [Web 接口](docs/web-interface.md)，合入现有仓库的说明见 [集成说明](docs/web-integration.md)。
