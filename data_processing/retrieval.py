import os
os.environ["NCCL_NVLS_ENABLE"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"

import json
import argparse
import numpy as np
from collections import defaultdict
from vllm import LLM
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_file", type=str, default="/home/work/mllm_datas/yilin/code/OR-SR1/datasets/error_data_classified/classified_all.json")
    parser.add_argument("--corpus_file", type=str, default="/home/work/mllm_datas/yilin/code/OR-SR1/datasets/error_data_classified/classified_merged_all_datasets.json")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="/home/work/models/Qwen3-Embedding-8B")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--bt_weight", type=float, default=0.3)
    parser.add_argument("--q_weight", type=float, default=0.7)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--batch_size", type=int, default=1024)
    return parser.parse_args()

def encode_batched(model, texts, batch_size, pbar, desc=""):
    embeddings_list = []
    total = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        pbar.set_postfix_str(f"{desc} {i//batch_size+1}/{total}")
        batch = texts[i:i+batch_size]
        outputs = model.embed(batch, use_tqdm=False)
        embeddings_list.append(np.array([o.outputs.embedding for o in outputs]))
        pbar.update(1)
    embeddings = np.vstack(embeddings_list)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

def main():
    args = parse_args()
    
    with open(args.query_file, 'r', encoding='utf-8') as f:
        query_data = json.load(f)
    with open(args.corpus_file, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
    
    print(f"Query: {len(query_data)}, Corpus: {len(corpus_data)}")
    
    model = LLM(
        model=args.model_path,
        task="embed",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="half",
        max_model_len=18192,
        disable_log_stats=True
    )
    
    corpus_bt = [item.get('business_type', '') for item in corpus_data]
    corpus_q = [item.get('en_question', '') for item in corpus_data]
    query_bt = [item.get('business_type', '') for item in query_data]
    query_q = [item.get('en_question', '') for item in query_data]
    
    def calc_batches(n): return (n + args.batch_size - 1) // args.batch_size
    total_steps = calc_batches(len(corpus_bt)) + calc_batches(len(corpus_q)) + calc_batches(len(query_bt)) + calc_batches(len(query_q)) + len(query_data)
    
    pbar = tqdm(total=total_steps, desc="Progress")
    
    corpus_bt_embs = encode_batched(model, corpus_bt, args.batch_size, pbar, "Corpus BT")
    corpus_q_embs = encode_batched(model, corpus_q, args.batch_size, pbar, "Corpus Q")
    query_bt_embs = encode_batched(model, query_bt, args.batch_size, pbar, "Query BT")
    query_q_embs = encode_batched(model, query_q, args.batch_size, pbar, "Query Q")
    
    corpus_by_class = defaultdict(list)
    for idx, item in enumerate(corpus_data):
        corpus_by_class[item.get('math_class', 'Unknown')].append((idx, item))
    
    corpus_cache = {}
    for mc, items in corpus_by_class.items():
        indices = np.array([i for i, _ in items])
        corpus_cache[mc] = {
            'indices': indices,
            'items': [item for _, item in items],
            'bt_embs': corpus_bt_embs[indices],
            'q_embs': corpus_q_embs[indices]
        }
    
    results = []
    for q_idx in range(len(query_data)):
        pbar.set_postfix_str(f"Retrieve {q_idx+1}/{len(query_data)}")
        q_item = query_data[q_idx]
        q_mc = q_item.get('math_class', 'Unknown')
        q_bt_emb = query_bt_embs[q_idx]
        q_q_emb = query_q_embs[q_idx]
        
        candidates = []
        
        if q_mc in corpus_cache:
            cache = corpus_cache[q_mc]
            bt_sim = cache['bt_embs'] @ q_bt_emb
            q_sim = cache['q_embs'] @ q_q_emb
            combined = args.bt_weight * bt_sim + args.q_weight * q_sim
            top_ids = np.argsort(combined)[::-1][:args.top_k]
            for rank, idx in enumerate(top_ids):
                candidates.append({
                    'rank': rank + 1, 'corpus_index': int(cache['indices'][idx]),
                    'similarity': float(combined[idx]), 'math_class_match': True, 'item': cache['items'][idx]
                })
        
        if len(candidates) < args.top_k:
            remaining = args.top_k - len(candidates)
            other_cands = []
            for mc, cache in corpus_cache.items():
                if mc == q_mc:
                    continue
                bt_sim = cache['bt_embs'] @ q_bt_emb
                q_sim = cache['q_embs'] @ q_q_emb
                combined = args.bt_weight * bt_sim + args.q_weight * q_sim
                top_ids = np.argsort(combined)[::-1][:remaining]
                for idx in top_ids:
                    other_cands.append({
                        'corpus_index': int(cache['indices'][idx]), 'similarity': float(combined[idx]),
                        'math_class': mc, 'item': cache['items'][idx]
                    })
            other_cands.sort(key=lambda x: x['similarity'], reverse=True)
            for c in other_cands[:remaining]:
                candidates.append({
                    'rank': len(candidates) + 1, 'corpus_index': c['corpus_index'],
                    'similarity': c['similarity'], 'math_class_match': False, 'item': c['item']
                })
        
        results.append({'query_index': q_idx, 'query': q_item, 'retrieved': candidates})
        pbar.update(1)
    
    pbar.close()
    
    output_file = args.output_file or args.query_file.replace('.json', '_retrieval_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()

    #  /home/work/mllm_datas/yilin/code/OR-SR1/datasets/error_data_classified/classified_all_retrieval_results.json