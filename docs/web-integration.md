# Web 模块集成

本目录按 PuYu 仓库相对路径组织，是 Web 模块源码增量，不是完整训练仓库。将文件按目录合入成员功能分支，再提交 PR 审查。

## 文件合并

- `README-web.md` 为独立模块说明；保留仓库根 README，可在其中添加到该文档的链接。
- `requirements-web.txt` 与 `requirements-web-dev.txt` 为独立依赖文件，不覆盖训练、合成模块的依赖。
- `web/app.py` 为本次需要替换的界面入口；同时合入 `web/presentation.py`、`web/style.css` 等配套文件。
- `inference/schema.py` 和 `inference/backend.py` 仅负责 Web 校验与模拟，不替换 `model_pipeline.py`、`validator.py` 或 `mock_api.py`。
- `evaluation/human/` 与自动评测模块并存，不覆盖 `evaluation/README.md` 或评分脚本。
- `__init__.py` 用于独立运行；目标仓库已有非空同名文件时，保留其内容并合并所需声明。

仓库忽略规则应包含 `.venv/`、`__pycache__/`、`*.py[cod]`、`.env` 和 `results/web_smoke/`；保留现有 `.gitignore` 规则后补充缺项。

## 集成验证

1. 安装 `requirements-web.txt`。
2. 运行 `python -m unittest discover -s tests -p "test_web_*.py" -v`。
3. 运行 `python run_web.py`，验证三个模拟案例、JSON 单任务与失败提示、颜色图例和输入清空。
4. 按 `README-web.md` 执行可选浏览器检查。

当前代码不加载真实模型。模型接入应遵循 `docs/web-interface.md`，并独立保存原始成功及失败输出。人工评价表为规则与记录模板，校准示例不代表真实模型或独立评价实验结果。
