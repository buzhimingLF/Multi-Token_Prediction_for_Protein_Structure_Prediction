"""
创建距离矩阵预测训练数据
从 protein_coords_data.json 生成距离矩阵格式的训练数据

v2方案：预测原子间距离矩阵（旋转/平移不变）
- 输入：蛋白质序列
- 输出：Cα原子间的距离矩阵 (N, N)
- 距离是旋转/平移不变的，模型更容易学习
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from scipy.spatial.distance import pdist, squareform


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    计算距离矩阵

    Args:
        coords: (N, 3) 坐标数组

    Returns:
        (N, N) 距离矩阵，dist[i,j] = ||coord_i - coord_j||
    """
    # 使用scipy计算成对距离
    # pdist返回压缩形式，squareform转为完整矩阵
    dist_matrix = squareform(pdist(coords, metric='euclidean'))
    return dist_matrix


def normalize_distance_matrix(dist_matrix: np.ndarray, method: str = 'log_scale') -> Tuple[np.ndarray, dict]:
    """
    归一化距离矩阵

    Args:
        dist_matrix: (N, N) 距离矩阵
        method: 归一化方法
            - 'scale': 除以最大距离，归一化到[0,1]
            - 'log_scale': log(1+d)然后归一化，压缩大距离的影响
            - 'zscore': Z-score标准化

    Returns:
        (normalized_matrix, stats) 元组
    """
    if method == 'scale':
        max_dist = dist_matrix.max()
        normalized = dist_matrix / (max_dist + 1e-8)
        stats = {'method': 'scale', 'max_dist': float(max_dist)}

    elif method == 'log_scale':
        # log(1+d) 压缩大距离
        log_dist = np.log1p(dist_matrix)
        max_log = log_dist.max()
        normalized = log_dist / (max_log + 1e-8)
        stats = {'method': 'log_scale', 'max_log_dist': float(max_log)}

    elif method == 'zscore':
        # 只对上三角（不含对角线）计算统计量
        upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        mean_dist = upper_tri.mean()
        std_dist = upper_tri.std()
        normalized = (dist_matrix - mean_dist) / (std_dist + 1e-8)
        # 对角线保持为0
        np.fill_diagonal(normalized, 0)
        stats = {'method': 'zscore', 'mean': float(mean_dist), 'std': float(std_dist)}

    else:
        raise ValueError(f"未知的归一化方法: {method}")

    return normalized, stats


def create_training_data(input_file: str, output_train: str, output_val: str,
                        max_seq_len: int = 256, train_ratio: float = 0.9,
                        norm_method: str = 'log_scale'):
    """
    创建距离矩阵训练数据

    Args:
        input_file: 输入文件（protein_coords_data.json）
        output_train: 训练集输出文件
        output_val: 验证集输出文件
        max_seq_len: 最大序列长度（距离矩阵为N×N，内存限制）
        train_ratio: 训练集比例
        norm_method: 距离矩阵归一化方法
    """
    print(f"加载数据: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"原始数据量: {len(data)}")

    # 过滤：只保留序列长度 <= max_seq_len 的样本
    # 注意：距离矩阵大小为 N×N，需要更严格的长度限制
    filtered_data = []
    for item in data:
        seq_len = item['seq_len']
        if seq_len <= max_seq_len:
            filtered_data.append(item)

    print(f"过滤后数据量 (seq_len <= {max_seq_len}): {len(filtered_data)}")

    # 计算距离矩阵
    print(f"\n计算距离矩阵 (归一化方法: {norm_method})...")
    global_distances = []  # 收集所有距离用于全局统计

    for i, item in enumerate(filtered_data):
        coords = np.array(item['coords'])  # (seq_len, 3)

        # 计算距离矩阵
        dist_matrix = compute_distance_matrix(coords)

        # 收集距离统计
        upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        global_distances.extend(upper_tri.tolist())

        # 归一化
        norm_dist_matrix, stats = normalize_distance_matrix(dist_matrix, method=norm_method)

        # 只保存上三角（节省空间，因为距离矩阵是对称的）
        # 但为了简化，我们保存完整矩阵
        item['distance_matrix'] = norm_dist_matrix.tolist()
        item['dist_norm_stats'] = stats

        # 保留原始坐标用于评估（可选）
        # item['coords'] 已经存在

        if (i + 1) % 500 == 0:
            print(f"  处理进度: {i+1}/{len(filtered_data)}")

    # 计算全局距离统计
    global_distances = np.array(global_distances)
    global_stats = {
        'mean_distance': float(global_distances.mean()),
        'std_distance': float(global_distances.std()),
        'min_distance': float(global_distances.min()),
        'max_distance': float(global_distances.max()),
        'median_distance': float(np.median(global_distances))
    }
    print(f"\n全局距离统计:")
    print(f"  平均距离: {global_stats['mean_distance']:.2f} Å")
    print(f"  标准差: {global_stats['std_distance']:.2f} Å")
    print(f"  最小距离: {global_stats['min_distance']:.2f} Å")
    print(f"  最大距离: {global_stats['max_distance']:.2f} Å")
    print(f"  中位数: {global_stats['median_distance']:.2f} Å")

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
        json.dump({
            'samples': train_data,
            'global_stats': global_stats,
            'config': {
                'max_seq_len': max_seq_len,
                'norm_method': norm_method
            }
        }, f, ensure_ascii=False)

    print(f"保存验证集: {output_val}")
    with open(output_val, 'w') as f:
        json.dump({
            'samples': val_data,
            'global_stats': global_stats,
            'config': {
                'max_seq_len': max_seq_len,
                'norm_method': norm_method
            }
        }, f, ensure_ascii=False)

    # 统计信息
    train_lengths = [item['seq_len'] for item in train_data]
    val_lengths = [item['seq_len'] for item in val_data]

    print(f"\n训练集序列长度统计:")
    print(f"  最短: {min(train_lengths)}")
    print(f"  最长: {max(train_lengths)}")
    print(f"  平均: {np.mean(train_lengths):.1f}")

    # 距离矩阵大小估算
    avg_len = np.mean(train_lengths)
    print(f"\n距离矩阵大小估算:")
    print(f"  平均矩阵大小: {avg_len:.0f} × {avg_len:.0f} = {avg_len**2:.0f} 个元素")
    print(f"  最大矩阵大小: {max(train_lengths)} × {max(train_lengths)} = {max(train_lengths)**2} 个元素")

    # 显示示例
    print(f"\n训练集示例:")
    sample = train_data[0]
    print(f"  PDB Code: {sample['pdb_code']}")
    print(f"  序列长度: {sample['seq_len']}")
    print(f"  序列: {sample['sequence'][:50]}...")
    print(f"  距离矩阵形状: {len(sample['distance_matrix'])} × {len(sample['distance_matrix'][0])}")
    dist_sample = np.array(sample['distance_matrix'])
    print(f"  距离矩阵统计 (归一化后):")
    print(f"    最小值: {dist_sample.min():.4f}")
    print(f"    最大值: {dist_sample.max():.4f}")
    print(f"    对角线: {dist_sample[0,0]:.4f} (应为0)")


def main():
    parser = argparse.ArgumentParser(description='创建距离矩阵预测训练数据')
    parser.add_argument('--input', type=str, default='protein_coords_data.json',
                        help='输入文件（protein_coords_data.json）')
    parser.add_argument('--output_train', type=str, default='distmat_train.json',
                        help='训练集输出文件')
    parser.add_argument('--output_val', type=str, default='distmat_val.json',
                        help='验证集输出文件')
    parser.add_argument('--max_seq_len', type=int, default=256,
                        help='最大序列长度（默认256，距离矩阵较大）')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='训练集比例')
    parser.add_argument('--norm_method', type=str, default='log_scale',
                        choices=['scale', 'log_scale', 'zscore'],
                        help='距离矩阵归一化方法')

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
    print("创建距离矩阵预测训练数据 (v2方案)")
    print("=" * 60)
    print(f"优势: 距离是旋转/平移不变的")
    print("=" * 60)

    create_training_data(
        str(input_file),
        str(output_train),
        str(output_val),
        max_seq_len=args.max_seq_len,
        train_ratio=args.train_ratio,
        norm_method=args.norm_method
    )

    print("\n完成！")


if __name__ == "__main__":
    main()
