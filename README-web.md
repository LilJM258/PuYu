# PuYu 作文批改 Web 模块

第三、四周已加入 Qwen3-1.7B A/B 本地接入口、桌面独立窗口、生成耗时与预算校验、原始报告导出及人工复核表。完整运行与交付说明见 [6号第三第四周系统交付](docs/member6-week34.md)。桌面源码入口为 `python run_desktop.py`，本机模型配置由 `scripts/configure_model_b.py` 生成；模型权重与 EXE 不放入 Git。

配置模型 B 后页面默认使用真实本地模型；未配置时保留团队共享模拟接口。`process_essay(essay_text)` 的旧回调继续调用 `inference.mock_api.mock_generate_review(essay_text)`，保持已有集成契约。

## 安装与运行

Python 3.11–3.13，验证环境为 Python 3.13。在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe run_web.py
```

访问 http://127.0.0.1:7860。`--port 7861` 可更换端口，Windows 可使用 `run_web.bat`。原有 `python web/app.py` 入口及 `process_essay(essay_text)` 回调保留。其他系统使用 `.venv/bin/python`。

桌面窗口入口为 `python run_desktop.py`，UI 环境另安装 requirements-desktop.txt；真实 B 使用独立推理环境和本机配置，见 [第三第四周说明](docs/member6-week34.md)。已配置的本机直接双击发行版 PuYu.exe，不需要手动启动网页服务。

## 输出来源

- **本地模型（Qwen）**：实际调用官方 Qwen3-1.7B，B 加载已交付 LoRA；独立生成评分、反馈和完整纠错，保留原始输出、失败、时间与模型指纹。
- **团队模拟接口（1号）**：实际调用共享 `mock_generate_review()`。直接显示其 scores、overall_feedback、errors，保留完整原始响应。没有 corrected_text 时不补造全文。
- **展示样例（非接口）**：三个原创固定案例，仅验证颜色、布局与全文差异。载入此类样例会明确切换来源，不作为团队接口的失败备用数据。
- **接口 JSON 联调**：分别验证评分、LEAF 字符串反馈及全文纠错。可只提供一项，不改变团队报告协议或教师训练标签。

原文红底表示明确错误、黄底表示可选改进、绿底表示亮点，均来自展示样例的人工标注。蓝底只表示纠正全文中的变化或新增句子，不能证明修改正确。按句末标点和换行近似分句，缩写可能影响边界。

共享模拟接口是固定响应，返回的引用可能与当前输入不同，页面会明确提示。字段合法、引用存在和语义正确是不同层次的检查；模拟数据不计入真实模型实验。人工反馈评价页可保存真实报告的署名复核；正式双人盲评使用独立匿名材料。

## 验证

```powershell
.\.venv\Scripts\python.exe scripts\check_member6.py
.\.venv\Scripts\python.exe scripts\build_web_fixtures.py
```

可选浏览器验证：先安装 requirements-web-dev.txt，保持 Web 服务运行，再执行：

```powershell
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe scripts\check_web.py
```

本机已有 Edge 时使用 `--browser msedge`。`--url` 指定服务地址，`--output-dir` 指定结果目录，默认保存到 `results/web_smoke/`。

## 文件

- `web/`：界面、安全文本渲染和颜色展示。
- `inference/backend.py`：共享接口适配与独立展示数据读取，不实现新的默认假接口。
- `inference/schema.py`：展示层校验，保留原始小数，不取整、不截分、不另造总分。
- `data/mock/examples.json`：独立展示资料，与接口响应分开。
- `evaluation/human/`：评价标准、空白记录和校准示例。
- `tests/test_web_*.py`：页面逻辑、共享接口调用与异常路径验证。

详见 [接口适配](docs/web-interface.md) 与 [集成说明](docs/web-integration.md)。
