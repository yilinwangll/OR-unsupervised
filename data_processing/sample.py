import json
import random
import os

# 原始文件路径
input_path = "/home/work/mllm_datas/yilin/code/OR-R1/datasets/OR-Instruct-Data-3K/OR-Instruct-Data-3K_final.json"

# 输出文件路径（保存在同一目录）
output_path = os.path.join(os.path.dirname(input_path), "sample_100.json")

# 读取原始 JSON 数据（假设是列表格式）
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 检查数据是否为列表
if not isinstance(data, list):
    raise ValueError("JSON 文件根元素必须是一个列表（包含多条样本）")

# 如果总数不足100条，就全部保留
sample_size = min(1000, len(data))
sampled_data = random.sample(data, sample_size)

# 保存采样结果
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)

print(f"成功采样 {len(sampled_data)} 条数据，已保存至: {output_path}")
