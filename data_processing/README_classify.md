# 数据集分类工具使用说明

## 功能说明

`classify_datasets.py` 用于对处理后的数据集进行分类，分类维度包括：
1. **问题类型**：线性规划、整数规划、非线性规划、网络流、调度等
2. **问题领域**：运输、生产、投资、医疗、教育等
3. **答案类型**：代码答案、数字答案、文本答案
4. **优化类型**：最小化、最大化

## 使用方法

### 1. 使用默认路径（推荐）

脚本会自动基于脚本位置计算默认路径：
- 输入目录：`../datasets/trainset/processed/`
- 输出目录：`../datasets/trainset/classified/`

```bash
# 从任意位置运行
python /home/work/mllm_datas/yilin/code/OR-SR1/knowledge/classify_datasets.py

# 或者在脚本目录下运行
cd /home/work/mllm_datas/yilin/code/OR-SR1/knowledge
python classify_datasets.py
```

### 2. 指定输入和输出目录

```bash
python classify_datasets.py \
    --input-dir /path/to/processed \
    --output-dir /path/to/classified
```

### 3. 处理单个文件

```bash
python classify_datasets.py \
    --input-file /path/to/input.json \
    --output-file /path/to/output.json
```

### 4. 指定要处理的文件列表

```bash
python classify_datasets.py \
    --input-dir /path/to/processed \
    --output-dir /path/to/classified \
    --files file1.json file2.json file3.json
```

## 输出格式

分类后的JSON文件格式：

```json
{
  "en_question": "问题内容...",
  "answer": "答案内容...",
  "classification": {
    "problem_type": "integer_programming",
    "domain": "transportation",
    "answer_type": "code",
    "optimization_type": "minimize"
  }
}
```

## 分类结果说明

### 问题类型 (problem_type)
- `linear_programming`: 线性规划
- `integer_programming`: 整数规划
- `binary_programming`: 二进制规划
- `nonlinear_programming`: 非线性规划
- `network_flow`: 网络流问题
- `scheduling`: 调度问题
- `knapsack`: 背包问题
- `assignment`: 分配问题
- `other`: 其他类型

### 问题领域 (domain)
- `transportation`: 运输
- `production`: 生产
- `investment`: 投资
- `healthcare`: 医疗
- `education`: 教育
- `energy`: 能源
- `agriculture`: 农业
- `retail`: 零售
- `construction`: 建筑
- `packaging`: 包装
- `other`: 其他领域

### 答案类型 (answer_type)
- `code`: 代码答案
- `numeric`: 数字答案
- `text`: 文本答案
- `text_with_model`: 包含数学模型的文本答案

### 优化类型 (optimization_type)
- `minimize`: 最小化
- `maximize`: 最大化
- `unknown`: 未知

## 查看统计信息

运行脚本后会自动显示分类统计信息。也可以使用 `view_classification_stats.py` 查看详细统计：

```bash
python view_classification_stats.py
```
