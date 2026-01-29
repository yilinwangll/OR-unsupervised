#!/usr/bin/env python3
"""
查看分类统计信息的工具脚本
"""

import json
from pathlib import Path
from collections import defaultdict


def analyze_classification(file_path):
    """分析分类文件并返回统计信息"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = {
        'total': len(data),
        'problem_types': defaultdict(int),
        'domains': defaultdict(int),
        'answer_types': defaultdict(int),
        'optimization_types': defaultdict(int),
        'combinations': defaultdict(int)
    }
    
    for item in data:
        if 'classification' not in item:
            continue
        
        cls = item['classification']
        
        stats['problem_types'][cls.get('problem_type', 'unknown')] += 1
        stats['domains'][cls.get('domain', 'unknown')] += 1
        stats['answer_types'][cls.get('answer_type', 'unknown')] += 1
        stats['optimization_types'][cls.get('optimization_type', 'unknown')] += 1
        
        # 组合统计
        combo = f"{cls.get('problem_type', 'unknown')}_{cls.get('domain', 'unknown')}"
        stats['combinations'][combo] += 1
    
    return stats


def print_stats(stats, title):
    """打印统计信息"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Total items: {stats['total']}")
    
    print(f"\nProblem Types:")
    for ptype, count in sorted(stats['problem_types'].items(), key=lambda x: -x[1]):
        percentage = (count / stats['total']) * 100
        print(f"  {ptype:25s}: {count:6d} ({percentage:5.2f}%)")
    
    print(f"\nDomains:")
    for domain, count in sorted(stats['domains'].items(), key=lambda x: -x[1]):
        percentage = (count / stats['total']) * 100
        print(f"  {domain:25s}: {count:6d} ({percentage:5.2f}%)")
    
    print(f"\nAnswer Types:")
    for atype, count in sorted(stats['answer_types'].items(), key=lambda x: -x[1]):
        percentage = (count / stats['total']) * 100
        print(f"  {atype:25s}: {count:6d} ({percentage:5.2f}%)")
    
    print(f"\nOptimization Types:")
    for otype, count in sorted(stats['optimization_types'].items(), key=lambda x: -x[1]):
        percentage = (count / stats['total']) * 100
        print(f"  {otype:25s}: {count:6d} ({percentage:5.2f}%)")
    
    print(f"\nTop 10 Problem-Domain Combinations:")
    for combo, count in sorted(stats['combinations'].items(), key=lambda x: -x[1])[:10]:
        percentage = (count / stats['total']) * 100
        print(f"  {combo:50s}: {count:6d} ({percentage:5.2f}%)")


def main():
    classified_dir = Path('/home/work/mllm_datas/yilin/code/OR-SR1/datasets/trainset/classified')
    
    files = [
        classified_dir / 'Evo_Step_dataset_classified.json',
        classified_dir / 'resocratic-29k_classified.json',
        classified_dir / 'train_all_classified.json'
    ]
    
    all_stats = {
        'total': 0,
        'problem_types': defaultdict(int),
        'domains': defaultdict(int),
        'answer_types': defaultdict(int),
        'optimization_types': defaultdict(int),
        'combinations': defaultdict(int)
    }
    
    for file_path in files:
        if file_path.exists():
            stats = analyze_classification(file_path)
            print_stats(stats, f"Statistics for {file_path.name}")
            
            # 合并到总体统计
            all_stats['total'] += stats['total']
            for key in ['problem_types', 'domains', 'answer_types', 'optimization_types', 'combinations']:
                for k, v in stats[key].items():
                    all_stats[key][k] += v
    
    # 打印总体统计
    print_stats(all_stats, "Overall Statistics (All Files Combined)")


if __name__ == '__main__':
    main()
