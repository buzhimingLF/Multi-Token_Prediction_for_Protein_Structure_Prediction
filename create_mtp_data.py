"""
MTP 训练数据格式转换脚本
将蛋白质序列转换为 Multi-Token Prediction 训练格式
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple


def create_mtp_training_data(
    sequences_file: str,
    output_file: str,
    n_future_tokens: int = 4,
    max_seq_length: int = 512,
    min_seq_length: int = 50,
    train_ratio: float = 0.9
) -> Tuple[int, int]:
    """
    创建 MTP 训练数据
    
    MTP 核心思想：预测未来 n 个 token，而不是只预测下一个 token
    
    Args:
        sequences_file: 蛋白质序列 JSON 文件
        output_file: 输出文件路径
        n_future_tokens: 预测的未来 token 数量（论文建议 4）
        max_seq_length: 最大序列长度
        min_seq_length: 最小序列长度
        train_ratio: 训练集比例
        
    Returns:
        (训练样本数, 验证样本数)
    """
    print(f"加载序列数据: {sequences_file}")
    with open(sequences_file, 'r') as f:
        data = json.load(f)
    
    print(f"总共 {len(data)} 个序列")
    
    # 过滤序列长度
    filtered_data = [
        d for d in data 
        if min_seq_length <= d['sequence_length'] <= max_seq_length
    ]
    print(f"过滤后 {len(filtered_data)} 个序列 (长度 {min_seq_length}-{max_seq_length})")
    
    # 打乱数据
    random.seed(42)
    random.shuffle(filtered_data)
    
    # 划分训练集和验证集
    split_idx = int(len(filtered_data) * train_ratio)
    train_data = filtered_data[:split_idx]
    val_data = filtered_data[split_idx:]
    
    # 创建训练样本
    train_samples = []
    val_samples = []
    
    for sample in train_data:
        train_samples.append({
            'input': sample['full_sequence'],
            'pdb_code': sample['pdb_code'],
            'binding_data': sample['binding_data'],
            'n_future_tokens': n_future_tokens
        })
    
    for sample in val_data:
        val_samples.append({
            'input': sample['full_sequence'],
            'pdb_code': sample['pdb_code'],
            'binding_data': sample['binding_data'],
            'n_future_tokens': n_future_tokens
        })
    
    # 保存数据
    output_path = Path(output_file)
    
    train_file = output_path.parent / f"{output_path.stem}_train.json"
    val_file = output_path.parent / f"{output_path.stem}_val.json"
    
    with open(train_file, 'w') as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    print(f"训练集保存到: {train_file} ({len(train_samples)} 样本)")
    
    with open(val_file, 'w') as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)
    print(f"验证集保存到: {val_file} ({len(val_samples)} 样本)")
    
    # 统计信息
    print("\n" + "=" * 50)
    print("数据集统计")
    print("=" * 50)
    print(f"训练集: {len(train_samples)} 样本")
    print(f"验证集: {len(val_samples)} 样本")
    print(f"MTP 预测头数量: {n_future_tokens}")
    
    # 序列长度分布
    lengths = [len(s['input']) for s in train_samples + val_samples]
    print(f"\n序列长度分布:")
    print(f"  最短: {min(lengths)}")
    print(f"  最长: {max(lengths)}")
    print(f"  平均: {sum(lengths) / len(lengths):.1f}")
    
    # 长度区间统计
    bins = [(50, 100), (100, 200), (200, 300), (300, 400), (400, 512)]
    print(f"\n长度区间分布:")
    for low, high in bins:
        count = sum(1 for l in lengths if low <= l < high)
        print(f"  {low}-{high}: {count} ({count/len(lengths)*100:.1f}%)")
    
    return len(train_samples), len(val_samples)


def create_instruction_format(
    sequences_file: str,
    output_file: str,
    max_seq_length: int = 512,
    min_seq_length: int = 50
):
    """
    创建指令微调格式的数据
    
    格式：
    {
        "instruction": "预测蛋白质序列的下一个氨基酸",
        "input": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
        "output": "GSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
    }
    """
    print(f"加载序列数据: {sequences_file}")
    with open(sequences_file, 'r') as f:
        data = json.load(f)
    
    # 过滤序列长度
    filtered_data = [
        d for d in data 
        if min_seq_length <= d['sequence_length'] <= max_seq_length
    ]
    
    # 创建指令格式数据
    instruction_data = []
    
    for sample in filtered_data:
        seq = sample['full_sequence']
        # 使用前半部分作为输入，后半部分作为输出
        split_point = len(seq) // 2
        
        instruction_data.append({
            'instruction': '根据给定的蛋白质序列片段，预测后续的氨基酸序列。',
            'input': seq[:split_point],
            'output': seq[split_point:],
            'pdb_code': sample['pdb_code'],
            'binding_data': sample['binding_data']
        })
    
    # 保存
    with open(output_file, 'w') as f:
        json.dump(instruction_data, f, indent=2, ensure_ascii=False)
    
    print(f"指令格式数据保存到: {output_file}")
    print(f"总共 {len(instruction_data)} 个样本")
    
    return len(instruction_data)


def main():
    """主函数"""
    base_dir = Path(__file__).parent
    sequences_file = base_dir / "protein_sequences.json"
    
    if not sequences_file.exists():
        print("错误: 请先运行 data_preprocessing.py 生成序列数据")
        return
    
    print("=" * 60)
    print("创建 MTP 训练数据")
    print("=" * 60)
    
    # 创建 MTP 格式数据
    create_mtp_training_data(
        str(sequences_file),
        str(base_dir / "mtp_data.json"),
        n_future_tokens=4,  # 论文建议值
        max_seq_length=512,
        min_seq_length=50,
        train_ratio=0.9
    )
    
    print("\n" + "=" * 60)
    print("创建指令微调格式数据")
    print("=" * 60)
    
    # 创建指令格式数据
    create_instruction_format(
        str(sequences_file),
        str(base_dir / "instruction_data.json"),
        max_seq_length=512,
        min_seq_length=50
    )


if __name__ == "__main__":
    main()
