import re
from openai import OpenAI

MODEL_NAME = "Qwen3-8B"
OPENAI_API_KEY = 'FAKE_API_KEY'
API_ENDPOINT = "http://127.0.0.1:8001/v1/"

client = OpenAI(base_url=API_ENDPOINT, api_key=OPENAI_API_KEY)


def call_api(prompt, temperature=0.7, max_tokens=8192):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    return response.choices[0].message.content.strip()


def _extract_optimal_value(text: str, max_tries: int = 3) -> str | None:
    """通过 LLM 从文本中提取最优值"""
    if not text:
        return None
    prompt = f"""
    You are an optimization expert. Your task is to extract EXACTLY ONE LINE from the text below that states the FINAL optimal value (e.g., "Maximum profit: $100", "Optimal cost: 50 units").
    
    STRICT RULES:
    1. Return ONLY the raw line from the text - character-for-character identical, no modifications.
    2. If multiple lines mention values, choose the LAST occurrence that represents the FINAL result.
    3. If NO line clearly states the final optimal value, return EMPTY STRING (no characters).
    4. NEVER add:
        - Prefixes/suffixes (e.g., "Answer:", quotes, line numbers)
        - Explanations or comments
        - Newline characters before/after the line
        - Any content not present in the original text
    
    Text to analyze (between ---):
    ---
    {text}
    ---
    
    Response format example when found:
    Optimal profit: $150
    
    Response format example when NOT found:
    [EMPTY - no characters at all]
    """
    for attempt in range(max_tries):
        try:
            result = call_api(prompt, temperature=0.0, max_tokens=100)
            # 额外后处理：确保只返回单行且无额外空格
            if result and "\n" not in result.strip():
                return result.strip()
            # 显式处理空响应
            if not result or result.strip() == "":
                return None
        except Exception as e:
            print(f"尝试 {attempt + 1} 失败: {e}")
    return None


def test_extract_optimal_value():
    """测试 LLM 提取最优值"""
    
    test_cases = [
        # 测试用例1: 标准优化输出
        """
        Solver Status: Optimal
        Iterations: 156
        Variables: 20
        Constraints: 15
        Optimal objective value: 15632.50
        Solution time: 0.5s
        """,
        
        # 测试用例2: 利润最大化
        """
        === Optimization Results ===
        Total revenue: $50,000
        Total cost: $35,000
        Maximum profit: $15,000
        Constraints satisfied: Yes
        """,
        
        # 测试用例3: 成本最小化
        """
        Linear Programming Solution
        Status: OPTIMAL
        Minimum cost: 1234.56
        Number of iterations: 42
        """,
        
        # 测试用例4: 中文输出
        """
        求解完成
        决策变量数: 10
        约束条件数: 5
        最优目标值: 9876.54
        求解时间: 1.2秒
        """,
        
        # 测试用例5: 复杂输出
        """
        CPLEX> optimize
        Tried aggregator 1 time.
        LP Presolve eliminated 3 rows and 2 columns.
        All rows and columns eliminated.
        Presolve time = 0.00 sec.
        
        Objective value = 2500.00
        """,
        """
        expect output line 2)
        """
    ]
    
    print("=" * 60)
    print("测试 LLM 提取最优值")
    print("=" * 60)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试用例 {i}:")
        print("-" * 40)
        print(f"输入文本:\n{text}")
        print("-" * 40)
        
        result = _extract_optimal_value(text)
        
        print(f"提取结果: {result}")
        print("=" * 60)


if __name__ == "__main__":
    test_extract_optimal_value()