import gradio as gr
import json
import sys
import os

# 将项目根目录加入路径，方便导入 inference 中的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 暂时导入 mock 假接口进行测试
from inference.mock_api import mock_generate_review

def process_essay(essay_text):
    if not essay_text.strip():
        return "请输入作文内容。", None
    
    # 调用接口（目前是 mock 假数据）
    result = mock_generate_review(essay_text)
    
    if result.get("status") == "error":
        return f"批改失败: {result.get('message')}", None
    
    # 提取分数和反馈用于展示
    scores = result.get("scores", {})
    feedback = result.get("overall_feedback", "")
    errors = result.get("errors", [])
    
    # 组装一个好看的 Markdown 文本用于展示
    md_text = f"### 评分结果\n"
    md_text += f"- **语法 (Grammar)**: {scores.get('grammar', 'N/A')}\n"
    md_text += f"- **词汇 (Vocabulary)**: {scores.get('vocabulary', 'N/A')}\n"
    md_text += f"- **结构 (Structure)**: {scores.get('structure', 'N/A')}\n"
    md_text += f"- **连贯性 (Coherence)**: {scores.get('coherence', 'N/A')}\n"
    md_text += f"- **内容 (Content)**: {scores.get('content', 'N/A')}\n"
    md_text += f"- **综合 (Overall)**: {scores.get('overall', 'N/A')}\n\n"
    md_text += f"### 总体评价\n{feedback}\n\n"
    md_text += f"### 错误明细\n"
    
    for i, err in enumerate(errors):
        md_text += f"{i+1}. **[{err.get('type', '错误')}]** `{err.get('original', '')}`  ->  `{err.get('corrected', '')}`\n"
        md_text += f"   *解释*: {err.get('explanation', '')}\n"
        
    return md_text, json.dumps(result, indent=2, ensure_ascii=False)

# 搭建 Gradio 界面
with gr.Blocks(title="英语作文智能批改系统") as demo:
    gr.Markdown("# 📝 英语作文智能批改系统")
    gr.Markdown("基于合成数据增强与知识蒸馏的科研项目")
    
    with gr.Row():
        with gr.Column():
            essay_input = gr.Textbox(lines=10, label="请输入英语作文", placeholder="在此粘贴你的作文...")
            submit_btn = gr.Button("开始批改", variant="primary")
            
        with gr.Column():
            md_output = gr.Markdown(label="批改报告")
            json_output = gr.JSON(label="原始 JSON 数据")
            
    submit_btn.click(fn=process_essay, inputs=essay_input, outputs=[md_output, json_output])

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)