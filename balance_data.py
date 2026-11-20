import json
import random

# 2号的两个原始数据文件
input_files = [
    "training_data_v1/train-00001.jsonl",
    "training_data_v1/train-00002.jsonl"
]
# 输出的新文件名
output_file = "training_data_v1/train_balanced.jsonl"

# 设置随机种子，保证每次处理的结果一样
random.seed(42)

keep_data = []      # 保留的样本（正常样本）
undersample = []    # 需要被减少的样本（集中在3.0的）
oversample = []     # 需要被增加的样本（偏离3.0的）

print("开始读取并合并两个文件...")
for file in input_files:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # 只有评分任务才需要平衡分数，其他任务（反馈、纠错）直接保留
            if data.get("task") != "essay_scoring":
                keep_data.append(data)
                continue
            
            try:
                # 提取模型回答的 JSON 并计算平均分
                scores_str = data["messages"][1]["content"]
                scores = json.loads(scores_str)["scores"]
                avg_score = sum(scores.values()) / len(scores)
                
                # 核心逻辑：如果平均分在 2.8 到 3.2 之间（最偷懒的分数）
                if 2.8 <= avg_score <= 3.2:
                    # 丢弃 60% 的这种数据
                    if random.random() > 0.6:
                        undersample.append(data)
                else:
                    # 其他分数段（比如2.0, 4.0）保留，并额外复制一份（过采样）
                    oversample.append(data)
                    oversample.append(data) 
            except Exception as e:
                # 如果解析失败，当普通数据保留
                keep_data.append(data)

# 合并：正常数据 + 减少后的中庸数据 + 复制过的高质量/低质量数据
final_dataset = keep_data + undersample + oversample
random.shuffle(final_dataset)  # 打乱顺序

# 保存新文件
with open(output_file, "w", encoding="utf-8") as f:
    for item in final_dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("处理完成！")
print(f"原始总数据量: {len(keep_data) + len(undersample) + int(len(oversample)/2)}")
print(f"处理后总数据量: {len(final_dataset)}")
print(f"已保存到: {output_file}")