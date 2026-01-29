import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# --- 配置信息 ---
API_KEY = "sk-c9db0320fc0a4f65b78c4bae56a03c82"
API_URL = "https://api.deepseek.com/v1/chat/completions"
FILE_PATH = "/home/work/mllm_datas/yilin/code/OR-dataset/grpo_data/data_selected_new.json"
CONCURRENCY = 240

def question2cloze_logic(question_text: str) -> str:
    """构建完整的 Prompt 模板"""
    return f"""Transform the following optimization problem into a challenging fill-in-the-blank (cloze) exercise. Your goal is to remove key information that a solver would need to formulate the mathematical model.

        **Replace each of the following with "____":**

        1. **Optimization objective word**: "minimize", "maximize", "largest", "smallest", "most", "least" (when indicating optimization goal)

        2. **All numeric parameters**, including:
        - Resource limits (e.g., "100 acres" → "____ acres")
        - Minimum/maximum requirements (e.g., "at least 7 acres" → "at least ____ acres")
        - Profit/cost coefficients (e.g., "$2.5" → "$____")
        - Ratios and multipliers (e.g., "three times" → "____ times")

        3. **Constraint direction phrases** — replace the ENTIRE phrase with a single "____":
        - "at least", "at most", "no more than", "no less than" → "____"
        - "a minimum of", "a maximum of" → "____"
        - "cannot exceed", "must exceed" → "____"
        - "up to", "more than", "less than" → "____"

        4. **Bound-indicating words** when they modify a constraint (replace as single unit with number):
        - "a minimum of 7" → "____ ____"
        - "at most three times" → "____ ____ times"
        - "maximum capacity of 500" → "____ ____ ____"

        **Rules:**
        - Each distinct replaceable element becomes exactly one "____"
        - Preserve all original punctuation, sentence structure, and formatting
        - Do NOT add explanations, labels, or commentary
        - Output ONLY the transformed text

        **Example:**
        Input: "He must plant at least 10 acres of wheat and can use a maximum of 50 workers."
        Output: "He must plant ____ ____ acres of wheat and can use ____ ____ ____ workers."

        Input: "The company wants to maximize profit, earning $5 per unit."
        Output: "The company wants to ____ profit, earning $____ per unit."

        ---

        Original problem:
        {question_text.strip()}

        Cloze version:"""

def fetch_cloze(item):
    """单条数据请求函数"""
    if 'en_question' not in item or not item['en_question']:
        return item
    if 'cloze' in item:
        return item
    prompt = question2cloze_logic(item['en_question'])
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        # 设置超时防止死锁
        response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            res_data = response.json()
            item['cloze'] = res_data['choices'][0]['message']['content'].strip()
        else:
            item['cloze'] = f"Error: {response.status_code}"
    except Exception as e:
        item['cloze'] = f"Exception: {str(e)}"
    
    return item

def main():
    # 1. 加载原始数据
    print(f"正在读取文件: {FILE_PATH}")
    if not os.path.exists(FILE_PATH):
        print("错误：找不到指定的 JSON 文件。")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 多线程并发处理
    print(f"开始生成 Cloze，并发数: {CONCURRENCY}")
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        # 使用 list(tqdm(...)) 来阻塞等待结果并显示进度条
        updated_data = list(tqdm(executor.map(fetch_cloze, data), total=len(data)))

    # 3. 覆盖写入原文件
    print(f"正在覆盖写入文件...")
    with open(FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=4)
    
    print("处理完成！数据已成功更新并保存。")

if __name__ == "__main__":
    main()
