#!/usr/bin/env python3
"""
对处理后的数据集进行分类
分类维度：
1. 问题类型（线性规划、整数规划、非线性规划、网络流、调度等）
2. 问题领域（运输、生产、投资、医疗、教育等）
3. 答案类型（代码答案、数字答案、文本答案）
4. 优化类型（最小化、最大化）
"""

import json
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict


class DatasetClassifier:
    def __init__(self):
        # 问题类型关键词
        self.problem_type_keywords = {
            'linear_programming': [
                'linear programming', 'linear program', 'lp', 'linear optimization',
                'linear constraint', 'linear objective'
            ],
            'integer_programming': [
                'integer programming', 'integer program', 'ip', 'integer variable',
                'integer constraint', 'whole number', 'whole numbers', 'integers',
                'vtype=COPT.INTEGER', 'vtype="INTEGER"'
            ],
            'binary_programming': [
                'binary variable', 'binary programming', '0 or 1', 'binary constraint',
                'vtype=COPT.BINARY', 'vtype="BINARY"', 'either selected or not'
            ],
            'nonlinear_programming': [
                'nonlinear', 'non-linear', 'quadratic', 'polynomial',
                's^2', 'x**2', 'x^2', 'nonlinear objective', 'nonlinear constraint',
                'pyscipopt'  # pyscipopt通常用于非线性问题
            ],
            'network_flow': [
                'network', 'flow', 'shortest path', 'maximum flow', 'minimum cost flow',
                'transportation', 'assignment', 'routing', 'distribution center',
                'supply chain', 'logistics network'
            ],
            'scheduling': [
                'schedule', 'scheduling', 'makespan', 'job', 'machine', 'operation',
                'landing time', 'aircraft', 'production planning', 'time period'
            ],
            'knapsack': [
                'knapsack', 'backpack', 'capacity constraint', 'budget constraint',
                'select items', 'choose items'
            ],
            'assignment': [
                'assignment', 'assign', 'allocate', 'allocation', 'matching'
            ]
        }
        
        # 问题领域关键词
        self.domain_keywords = {
            'transportation': [
                'transportation', 'vehicle', 'route', 'shipping', 'delivery',
                'fleet', 'truck', 'van', 'logistics', 'distribution'
            ],
            'production': [
                'production', 'manufacturing', 'factory', 'machine', 'assembly',
                'product', 'component', 'inventory', 'supply chain'
            ],
            'investment': [
                'investment', 'portfolio', 'budget', 'cost', 'profit', 'revenue',
                'financial', 'asset', 'stock', 'fund'
            ],
            'healthcare': [
                'hospital', 'medical', 'patient', 'healthcare', 'diagnostic',
                'treatment', 'radiation', 'dose', 'medicine'
            ],
            'education': [
                'university', 'school', 'student', 'classroom', 'teacher',
                'education', 'textbook', 'cafeteria', 'menu'
            ],
            'energy': [
                'power', 'energy', 'electricity', 'solar', 'wind', 'renewable',
                'power plant', 'kilowatt'
            ],
            'agriculture': [
                'farm', 'crop', 'acre', 'agriculture', 'pumpkin', 'carrot',
                'growing', 'harvest'
            ],
            'retail': [
                'store', 'retail', 'warehouse', 'inventory', 'demand', 'supply',
                'customer', 'order'
            ],
            'construction': [
                'construction', 'building', 'project', 'material', 'cost'
            ],
            'packaging': [
                'packaging', 'container', 'box', 'dimension', 'surface area',
                'volume'
            ]
        }
        
        # 优化类型关键词
        self.optimization_keywords = {
            'minimize': [
                'minimize', 'minimum', 'minimizing', 'least', 'lowest',
                'reduce', 'minimization'
            ],
            'maximize': [
                'maximize', 'maximum', 'maximizing', 'most', 'highest',
                'increase', 'maximization'
            ]
        }
    
    def classify_problem_type(self, question, answer):
        """分类问题类型"""
        text = (question + ' ' + answer).lower()
        scores = {}
        
        for ptype, keywords in self.problem_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text)
            if score > 0:
                scores[ptype] = score
        
        if not scores:
            return 'other'
        
        # 返回得分最高的类型
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def classify_domain(self, question):
        """分类问题领域"""
        text = question.lower()
        scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text)
            if score > 0:
                scores[domain] = score
        
        if not scores:
            return 'other'
        
        # 返回得分最高的领域
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def classify_answer_type(self, answer):
        """分类答案类型"""
        answer_lower = answer.lower().strip()
        
        # 检查是否是代码
        if 'import ' in answer or 'def ' in answer or 'class ' in answer:
            return 'code'
        
        # 检查是否是纯数字
        try:
            float(answer_lower)
            return 'numeric'
        except ValueError:
            pass
        
        # 检查是否包含代码块
        if '```' in answer or '```python' in answer or '```py' in answer:
            return 'code'
        
        # 检查是否包含数学公式或代码片段
        if 'model.addVar' in answer or 'model.setObjective' in answer or 'model.addConstr' in answer:
            return 'code'
        
        # 检查是否包含数学建模描述
        if 'mathematical model' in answer_lower or 'objective function' in answer_lower:
            return 'text_with_model'
        
        # 默认返回文本
        return 'text'
    
    def classify_optimization_type(self, question):
        """分类优化类型"""
        text = question.lower()
        
        minimize_score = sum(1 for keyword in self.optimization_keywords['minimize'] 
                           if keyword in text)
        maximize_score = sum(1 for keyword in self.optimization_keywords['maximize'] 
                           if keyword in text)
        
        if minimize_score > maximize_score:
            return 'minimize'
        elif maximize_score > minimize_score:
            return 'maximize'
        else:
            return 'unknown'
    
    def classify_item(self, item):
        """对单个数据项进行分类"""
        question = item.get('en_question', '')
        answer = item.get('answer', '')
        
        classification = {
            'problem_type': self.classify_problem_type(question, answer),
            'domain': self.classify_domain(question),
            'answer_type': self.classify_answer_type(answer),
            'optimization_type': self.classify_optimization_type(question)
        }
        
        return classification
    
    def classify_dataset(self, input_file, output_file=None):
        """对整个数据集进行分类"""
        print(f"Classifying {input_file}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        classified_data = []
        stats = defaultdict(int)
        
        for item in data:
            classification = self.classify_item(item)
            
            # 添加分类信息到数据项
            classified_item = item.copy()
            classified_item['classification'] = classification
            
            classified_data.append(classified_item)
            
            # 统计
            stats[f"problem_type_{classification['problem_type']}"] += 1
            stats[f"domain_{classification['domain']}"] += 1
            stats[f"answer_type_{classification['answer_type']}"] += 1
            stats[f"optimization_type_{classification['optimization_type']}"] += 1
        
        # 保存分类后的数据
        if output_file is None:
            output_file = str(input_file).replace('.json', '_classified.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(classified_data, f, ensure_ascii=False, indent=2)
        
        print(f"  Processed {len(classified_data)} items")
        print(f"  Saved to {output_file}")
        
        # 打印统计信息
        print("\n  Classification Statistics:")
        print("  Problem Types:")
        for key, count in sorted(stats.items()):
            if key.startswith('problem_type_'):
                print(f"    {key.replace('problem_type_', '')}: {count}")
        
        print("  Domains:")
        for key, count in sorted(stats.items()):
            if key.startswith('domain_'):
                print(f"    {key.replace('domain_', '')}: {count}")
        
        print("  Answer Types:")
        for key, count in sorted(stats.items()):
            if key.startswith('answer_type_'):
                print(f"    {key.replace('answer_type_', '')}: {count}")
        
        print("  Optimization Types:")
        for key, count in sorted(stats.items()):
            if key.startswith('optimization_type_'):
                print(f"    {key.replace('optimization_type_', '')}: {count}")
        
        print()
        
        return classified_data, stats


def get_script_dir():
    """获取脚本所在目录"""
    return Path(__file__).parent.absolute()


def main():
    parser = argparse.ArgumentParser(
        description='对处理后的数据集进行分类',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认路径（基于脚本位置）
  python classify_datasets.py
  
  # 指定输入和输出目录
  python classify_datasets.py --input-dir /path/to/processed --output-dir /path/to/classified
  
  # 只处理特定文件
  python classify_datasets.py --input-file /path/to/file.json --output-file /path/to/output.json
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        help='输入目录路径（包含处理后的JSON文件）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='输出目录路径（保存分类后的JSON文件）'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        help='单个输入文件路径（如果指定，只处理这个文件）'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='单个输出文件路径（与--input-file一起使用）'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        help='要处理的文件名列表（相对于input-dir）'
    )
    
    args = parser.parse_args()
    
    classifier = DatasetClassifier()
    
    # 获取脚本所在目录
    script_dir = get_script_dir()
    
    # 确定输入和输出目录
    if args.input_file:
        # 处理单个文件
        input_file = Path(args.input_file)
        if not input_file.is_absolute():
            input_file = script_dir / input_file
        
        if not input_file.exists():
            print(f"✗ Error: Input file not found: {input_file}")
            sys.exit(1)
        
        if args.output_file:
            output_file = Path(args.output_file)
            if not output_file.is_absolute():
                output_file = script_dir / output_file
        else:
            output_file = input_file.parent / input_file.name.replace('.json', '_classified.json')
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing single file: {input_file}")
        _, stats = classifier.classify_dataset(input_file, output_file)
        
        return
    
    # 处理多个文件
    if args.input_dir:
        base_dir = Path(args.input_dir)
        if not base_dir.is_absolute():
            base_dir = script_dir / base_dir
    else:
        # 默认路径：基于脚本位置计算
        base_dir = script_dir.parent / 'datasets' / 'trainset' / 'processed'
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = script_dir / output_dir
    else:
        # 默认路径：基于脚本位置计算
        output_dir = script_dir.parent / 'datasets' / 'trainset' / 'classified'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定要处理的文件列表
    if args.files:
        files_to_classify = [base_dir / f for f in args.files]
    else:
        # 默认文件列表
        files_to_classify = [
            base_dir / 'Evo_Step_dataset_processed.json',
            base_dir / 'resocratic-29k_processed.json',
            base_dir / 'train_all_processed.json'
        ]
    
    print(f"Input directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files to process: {len(files_to_classify)}\n")
    
    all_stats = defaultdict(int)
    processed_count = 0
    
    for input_file in files_to_classify:
        if input_file.exists():
            output_file = output_dir / input_file.name.replace('_processed.json', '_classified.json')
            _, stats = classifier.classify_dataset(input_file, output_file)
            
            # 合并统计信息
            for key, count in stats.items():
                all_stats[key] += count
            processed_count += 1
        else:
            print(f"✗ File not found: {input_file}\n")
    
    if processed_count == 0:
        print("✗ No files were processed. Please check the input directory and file names.")
        sys.exit(1)
    
    # 打印总体统计
    print("=" * 60)
    print("Overall Classification Statistics:")
    print("=" * 60)
    
    print("\nProblem Types:")
    problem_types = {k.replace('problem_type_', ''): v 
                     for k, v in all_stats.items() if k.startswith('problem_type_')}
    for ptype, count in sorted(problem_types.items(), key=lambda x: -x[1]):
        print(f"  {ptype}: {count}")
    
    print("\nDomains:")
    domains = {k.replace('domain_', ''): v 
               for k, v in all_stats.items() if k.startswith('domain_')}
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        print(f"  {domain}: {count}")
    
    print("\nAnswer Types:")
    answer_types = {k.replace('answer_type_', ''): v 
                    for k, v in all_stats.items() if k.startswith('answer_type_')}
    for atype, count in sorted(answer_types.items(), key=lambda x: -x[1]):
        print(f"  {atype}: {count}")
    
    print("\nOptimization Types:")
    opt_types = {k.replace('optimization_type_', ''): v 
                 for k, v in all_stats.items() if k.startswith('optimization_type_')}
    for otype, count in sorted(opt_types.items(), key=lambda x: -x[1]):
        print(f"  {otype}: {count}")


if __name__ == '__main__':
    main()
