# Web 集成与提交

按负责人最新要求，在 feat/member6-web 分支开发，先同步远程分支并合入最新 main，保留其他成员更新，仅整合本模块必要的文件。

## 共享文件

- web/app.py：保留并适配主分支的 mock_generate_review 调用、六维评分和 process_essay 回调；扩展 Gradio 展示。
- inference/mock_api.py：使用主分支原文件，不替换、不重新定义函数。
- inference/model_pipeline.py、validator.py：本模块不修改，也不在启动时导入加载模型。
- 根 README、训练/合成代码和数据：保持主分支内容。
- 人工评价材料位于 evaluation/human，与自动评测模块并存。

## 验证

安装 requirements-web.txt 后运行 `python -m unittest discover -s tests -p "test_web_*.py" -v`。验证包括共享函数确实被调用、原始响应保留、错误状态、异常、未提供纠正全文、小数与越界结果，以及独立样例不冒充共享接口。

启动 `python run_web.py` 或 `python web/app.py`，按 README-web.md 执行浏览器检查。默认验证团队接口，另外检查展示样例的颜色和独立 JSON 输入。

## 提交

在验证和逐文件检查后执行 git add .、git commit、git push origin feat/member6-web。不得直接推 main。网络故障可重试，必要时仅补传实际修改的具体文件，不上传整个目录，不强制覆盖远端更新。

继续使用现有 PR #3，由负责人在网页或本地合入 main。保留已有上传和合并历史，不为处理本次集成而删除其他成员文件或重写仓库历史。
