import json

input_file = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/error_data_classified/classified_all_retrieval_results.json"
output_file = "/home/work/mllm_datas/yilin/code/OR-SR1/datasets/error_data_classified/retrieved_items.json"

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

retrieved_items = []
seen = set()

for entry in data:
    for ret in entry['retrieved']:
        idx = ret['corpus_index']
        if idx not in seen:
            seen.add(idx)
            retrieved_items.append(ret['item'])

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(retrieved_items, f, ensure_ascii=False, indent=2)

print(f"Total unique retrieved items: {len(retrieved_items)}")
print(f"Saved to {output_file}")
