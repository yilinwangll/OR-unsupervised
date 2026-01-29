import json
import requests
from openai import OpenAI
import re


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

data = read_jsonl('/home/work/mllm_datas/yilin/code/OR-R1/datasets/trainset/train_all.jsonl')
MODEL_NAME = "checkpoint-200"

OPENAI_API_KEY = 'FAKE_API_KEY'
api_endpoint="http://0.0.0.0:8000/v1/"

client = OpenAI(base_url=api_endpoint, api_key=OPENAI_API_KEY) 

def call_api(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.0,
        max_tokens=8192
    )
    extracted_text = response.choices[0].message.content.strip()
    return extracted_text

# def question2prompt(question_text: str) -> str:
    
#     template = r"""
#         Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.
#         {Question}
#         """
    
#     prompt = template.replace("{Question}", question_text.strip()).strip()
    
#     return prompt

def question2prompt(question_text: str) -> str:
    template = r"""
        You are tasked with solving an operations research problem through a structured three-step process:  
        1. Reasoning: Understand and analyze the problem.  
        2. Modeling: Formulate a precise mathematical model.  
        3. Implementation: Translate the model into executable Python code using `coptpy`.

        Please respond in the following exact format:

        <think>
        Your step-by-step reasoning and interpretation of the problem here.
        </think>
        <model>
        Your precise mathematical model formulation here, including decision variables, objective function, and constraints.
        </model>
        <code>
        Your complete and executable Python code using `coptpy` that implements the above model.
        </code>

Question: {Question}
"""
    
    prompt = template.replace("{Question}", question_text.strip()).strip()
    
    return prompt
# def question2prompt(question_text: str) -> str:
#     template = r"""
#         You are tasked with solving an operations research problem through a structured three-step process:  
#         1. Reasoning: Understand and analyze the problem.  
#         2. Modeling: Formulate a precise mathematical model.  
#         3. Implementation: Translate the model into executable Python code using `coptpy`.

#         Please respond in the following exact format:

#         <think>
#         Your step-by-step reasoning and interpretation of the problem here.
#         </think>
#         <code>
#         Your complete and executable Python code using `coptpy` that implements the above model.
#         </code>

# Question: {Question}
# """
    
#     prompt = template.replace("{Question}", question_text.strip()).strip()
    
#     return prompt

def cloze2answer_prompt(cloze_question: str, model_description: str) -> str:
    template = r"""
        You are given:
        1. An incomplete operations research problem with blanks marked as "____".
        2. A precise mathematical model that correctly represents the intended problem.

        Your task is to fill in all "____" in the problem statement **strictly according to the provided mathematical model**, ensuring:
        - Numerical values match exactly (e.g., if the model says "≤ 1000", the blank for the number must be "1000").
        - Constraint direction words (e.g., "at least", "cannot exceed", "up to") are chosen to accurately reflect the inequality direction in the model (e.g., "≤" → "cannot exceed" or "up to"; "≥" → "at least").

        Guidelines:
        - Use natural language phrases for constraint directions (e.g., "cannot exceed", "must be at least") — do not use symbols like "≤" in the filled text.
        - All numbers must be written as plain numerals (e.g., "1000", not "one thousand").
        - Do not alter any part of the sentence structure except replacing "____" with the correct word or number.
        - If multiple blanks appear in one constraint, fill them in the order: [relationship phrase] [number].

        Return ONLY the completed problem text with all blanks filled. No explanations, no markdown.

        ---
        Mathematical Model (for reference):
        {Model}

        ---
        Incomplete Problem:
        {ClozeQuestion}
        """
    prompt = template.replace("{ClozeQuestion}", cloze_question.strip()) \
                     .replace("{Model}", model_description.strip())
    return prompt.strip()

def question2cloze(question_text: str) -> str:
    prompt = f"""Transform the following optimization problem into a challenging fill-in-the-blank (cloze) exercise. Your goal is to remove key information that a solver would need to formulate the mathematical model.

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
    return prompt

i = 240
def run_code(code, url="http://47.98.187.25:8000/api/execute"):

    return requests.post(url, json={"msg": code}, timeout=300).json()["response"]
    
while True:
    # import pdb; pdb.set_trace()
    question = data[i]['question']
    prompt = question2prompt(data[i]['question'])
    # cloze = call_api(question2cloze(data[i]['question']))

    response = call_api(prompt)
    import pdb; pdb.set_trace()

    # code = re.search(r'<code>\s*```python\s*(.*?)\s*```', response, re.DOTALL).group(1)
    # import re; code = re.search(r'```python\n(.*?)\n```', response, re.DOTALL).group(1)
    model = (match.group(1) if (match := re.search(r'<model>(.*?)</model>', response, re.DOTALL)) else None)
    code = (match.group(1) if (match := re.search(r'<code>(.*?)</code>', response, re.DOTALL)) else None)
    
    
    i += 1
        
    # question_cloze= call_api(cloze2answer_prompt(cloze, model))

        
    





