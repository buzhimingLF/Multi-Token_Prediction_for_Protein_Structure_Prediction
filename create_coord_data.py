"""
创建坐标预测训练数据
从 protein_coords_data.json 生成适合MTP训练的格式

任务说明：
- 输入：蛋白质序列
- 输出：每个氨基酸的Cα原子坐标 (x,y,z)
- 使用MTP将输出长度限制为k=序列长度
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import argparse


def normalize_coords(coords: np.ndarray, method: str = 'center_scale') -> tuple:
    """
    归一化坐标

    Args:
        coords: (seq_len, 3) 坐标数组
        method: 归一化方法
            - 'center': 只零均值化
            - 'center_scale': 零均值化 + 标准化 (推荐)

    Returns:
        (normalized_coords, stats) 元组
        - normalized_coords: 归一化后的坐标
        - stats: 统计信息字典 {'mean': [...], 'std': [...]}
    """
    # 计算质心
    centroid = coords.mean(axis=0)
    # 零均值化
    centered = coords - centroid

    if method == 'center':
        return centered, {'mean': centroid.tolist(), 'std': None}
    elif method == 'center_scale':
        # 计算标准差
        std = centered.std()
        # 标准化
        normalized = centered / (std + 1e-8)  # 避免除0
        return normalized, {'mean': centroid.tolist(), 'std': float(std)}
    else:
        raise ValueError(f"未知的归一化方法: {method}")


def create_training_data(input_file: str, output_train: str, output_val: str,
                        max_seq_len: int = 512, train_ratio: float = 0.9):
    """
    创建训练和验证数据集

    Args:
        input_file: 输入文件（protein_coords_data.json）
        output_train: 训练集输出文件
        output_val: 验证集输出文件
        max_seq_len: 最大序列长度（超过的截断）
        train_ratio: 训练集比例
    """
    print(f"加载数据: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"原始数据量: {len(data)}")

    # 过滤：只保留序列长度 <= max_seq_len 的样本
    filtered_data = []
    for item in data:
        seq_len = item['seq_len']
        if seq_len <= max_seq_len:
            filtered_data.append(item)

    print(f"过滤后数据量 (seq_len <= {max_seq_len}): {len(filtered_data)}")

    # 归一化坐标
    print("归一化坐标 (center_scale方法)...")
    for item in filtered_data:
        coords = np.array(item['coords'])  # (seq_len, 3)
        normalized_coords, stats = normalize_coords(coords, method='center_scale')
        item['coords'] = normalized_coords.tolist()
        item['norm_stats'] = stats  # 保存归一化统计信息，推理时需要反归一化

    # 划分训练集和验证集
    np.random.seed(42)
    np.random.shuffle(filtered_data)

    split_idx = int(len(filtered_data) * train_ratio)
    train_data = filtered_data[:split_idx]
    val_data = filtered_data[split_idx:]

    print(f"\n数据划分:")
    print(f"  训练集: {len(train_data)} 个样本")
    print(f"  验证集: {len(val_data)} 个样本")

    # 保存
    print(f"\n保存训练集: {output_train}")
    with open(output_train, 'w') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    print(f"保存验证集: {output_val}")
    with open(output_val, 'w') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    # 统计信息
    train_lengths = [item['seq_len'] for item in train_data]
    val_lengths = [item['seq_len'] for item in val_data]

    print(f"\n训练集序列长度统计:")
    print(f"  最短: {min(train_lengths)}")
    print(f"  最长: {max(train_lengths)}")
    print(f"  平均: {np.mean(train_lengths):.1f}")

    print(f"\n验证集序列长度统计:")
    print(f"  最短: {min(val_lengths)}")
    print(f"  最长: {max(val_lengths)}")
    print(f"  平均: {np.mean(val_lengths):.1f}")

    # 显示示例
    print(f"\n训练集示例:")
    sample = train_data[0]
    print(f"  PDB Code: {sample['pdb_code']}")
    print(f"  序列长度: {sample['seq_len']}")
    print(f"  序列: {sample['sequence'][:50]}...")
    print(f"  前3个Cα坐标:")
    for i, coord in enumerate(sample['coords'][:3]):
        print(f"    {i+1}: [{coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f}]")


def main():
    parser = argparse.ArgumentParser(description='创建坐标预测训练数据')
    parser.add_argument('--input', type=str, default='protein_coords_data.json',
                        help='输入文件（protein_coords_data.json）')
    parser.add_argument('--output_train', type=str, default='coord_train.json',
                        help='训练集输出文件')
    parser.add_argument('--output_val', type=str, default='coord_val.json',
                        help='验证集输出文件')
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='训练集比例')

    args = parser.parse_args()

    # 设置路径
    base_dir = Path(__file__).parent
    input_file = base_dir / args.input
    output_train = base_dir / args.output_train
    output_val = base_dir / args.output_val

    # 检查输入文件是否存在
    if not input_file.exists():
        print(f"错误: 输入文件不存在: {input_file}")
        print("请先运行: python data_preprocessing.py --extract_coords")
        return

    # 创建训练数据
    print("=" * 60)
    print("创建坐标预测训练数据")
    print("=" * 60)

    create_training_data(
        str(input_file),
        str(output_train),
        str(output_val),
        max_seq_len=args.max_seq_len,
        train_ratio=args.train_ratio
    )

    print("\n完成！")


if __name__ == "__main__":
    main()
