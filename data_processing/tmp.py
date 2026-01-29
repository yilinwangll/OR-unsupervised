import json
import random
import os
import math

def split_json_randomly(input_path, num_splits=4):
    # 1. 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ 错误: 找不到文件 {input_path}")
        return

    # 2. 加载数据
    print(f"正在加载数据: {input_path} ...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取 JSON 失败: {e}")
        return

    total_len = len(data)
    print(f"✅ 数据加载完成，共 {total_len} 条记录。")

    # 3. 随机打乱数据 (Random Shuffle)
    print("正在随机打乱数据...")
    random.shuffle(data)

    # 4. 计算切分大小并保存
    # 使用 math.ceil 确保如果除不尽，前面的包稍微多一点，覆盖所有数据
    chunk_size = math.ceil(total_len / num_splits)
    
    base_dir = os.path.dirname(input_path)
    file_name = os.path.basename(input_path).replace('.json', '')

    print(f"开始拆分为 {num_splits} 份...")

    for i in range(num_splits):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        
        # 获取当前切片
        subset = data[start_idx:end_idx]
        
        # 如果切片为空（数据量极少的情况），停止循环
        if not subset:
            break

        # 构造输出文件名: merged_all_datasets_part_1.json, etc.
        output_filename = f"{file_name}_part_{i+1}.json"
        output_path = os.path.join(base_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 保证中文正常显示，indent=2 保证格式美观
            json.dump(subset, f, ensure_ascii=False, indent=2)
        
        print(f"  -> 已保存: {output_filename} (包含 {len(subset)} 条数据)")

    print("\n🎉 所有拆分任务已完成！")

# --- 配置路径并运行 ---
if __name__ == "__main__":
    target_file = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/trainset/processed/merged_all_datasets.json"
    
    split_json_randomly(target_file, num_splits=4)
