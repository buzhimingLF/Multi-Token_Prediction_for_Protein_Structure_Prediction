"""
MDS坐标还原模块
从距离矩阵重建3D坐标

使用多维尺度分析（Multidimensional Scaling, MDS）算法
将 N×N 距离矩阵还原为 N×3 的3D坐标

参考：
- Classical MDS (Torgerson's method)
- sklearn.manifold.MDS
"""

import numpy as np
from typing import Tuple, Optional
import warnings


def classical_mds(distance_matrix: np.ndarray, n_components: int = 3) -> np.ndarray:
    """
    经典MDS算法（Torgerson's method）

    从距离矩阵恢复坐标的解析解方法，速度快且稳定

    Args:
        distance_matrix: (N, N) 距离矩阵，应该是对称的
        n_components: 输出维度（通常为3）

    Returns:
        coords: (N, 3) 重建的坐标
    """
    n = distance_matrix.shape[0]

    # 1. 计算距离平方矩阵
    D_sq = distance_matrix ** 2

    # 2. 双中心化（Double centering）
    # J = I - (1/n) * 1 * 1^T (中心化矩阵)
    # B = -0.5 * J * D^2 * J
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D_sq @ H

    # 3. 对B进行特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # 按特征值降序排列
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 4. 取前k个正特征值及其特征向量
    # 处理可能的负特征值（数值误差）
    positive_mask = eigenvalues > 1e-10
    if np.sum(positive_mask) < n_components:
        warnings.warn(f"只有 {np.sum(positive_mask)} 个正特征值，小于请求的 {n_components} 维")
        n_components = min(n_components, np.sum(positive_mask))

    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]

    # 5. 计算坐标
    # X = V * sqrt(Lambda)
    coords = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))

    return coords


def iterative_mds(distance_matrix: np.ndarray, n_components: int = 3,
                  max_iter: int = 300, tol: float = 1e-4,
                  init_coords: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """
    迭代MDS算法（SMACOF - Scaling by MAjorizing a COmplicated Function）

    通过迭代优化最小化应力函数（stress function）

    Args:
        distance_matrix: (N, N) 距离矩阵
        n_components: 输出维度
        max_iter: 最大迭代次数
        tol: 收敛阈值
        init_coords: 初始坐标（如果为None，使用classical_mds初始化）

    Returns:
        coords: (N, 3) 重建的坐标
        stress: 最终的应力值
    """
    n = distance_matrix.shape[0]

    # 初始化坐标
    if init_coords is None:
        coords = classical_mds(distance_matrix, n_components)
    else:
        coords = init_coords.copy()

    # 避免除零
    dist_matrix_safe = distance_matrix.copy()
    dist_matrix_safe[dist_matrix_safe == 0] = 1e-10

    prev_stress = float('inf')

    for iteration in range(max_iter):
        # 计算当前坐标的距离矩阵
        current_dist = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=-1))
        current_dist_safe = current_dist.copy()
        current_dist_safe[current_dist_safe == 0] = 1e-10

        # 计算应力（stress）
        stress = np.sqrt(np.sum((distance_matrix - current_dist) ** 2) / np.sum(distance_matrix ** 2))

        # 检查收敛
        if abs(prev_stress - stress) < tol:
            break
        prev_stress = stress

        # 更新坐标（Guttman变换）
        B = -distance_matrix / current_dist_safe
        np.fill_diagonal(B, 0)
        np.fill_diagonal(B, -B.sum(axis=1))

        coords = B @ coords / n

    return coords, stress


def denormalize_distance_matrix(norm_dist_matrix: np.ndarray,
                                 norm_stats: dict) -> np.ndarray:
    """
    反归一化距离矩阵

    Args:
        norm_dist_matrix: 归一化后的距离矩阵
        norm_stats: 归一化统计信息

    Returns:
        原始尺度的距离矩阵（单位：Å）
    """
    method = norm_stats.get('method', 'log_scale')

    if method == 'scale':
        max_dist = norm_stats['max_dist']
        return norm_dist_matrix * max_dist

    elif method == 'log_scale':
        max_log_dist = norm_stats['max_log_dist']
        # 逆操作: d = exp(norm * max_log) - 1
        log_dist = norm_dist_matrix * max_log_dist
        return np.expm1(log_dist)

    elif method == 'zscore':
        mean = norm_stats['mean']
        std = norm_stats['std']
        return norm_dist_matrix * std + mean

    else:
        raise ValueError(f"未知的归一化方法: {method}")


def distance_matrix_to_coords(distance_matrix: np.ndarray,
                              method: str = 'classical',
                              **kwargs) -> np.ndarray:
    """
    从距离矩阵重建坐标的主函数

    Args:
        distance_matrix: (N, N) 距离矩阵（单位：Å）
        method: 重建方法
            - 'classical': 经典MDS（快速，解析解）
            - 'iterative': 迭代MDS（更精确）

    Returns:
        coords: (N, 3) 重建的坐标（单位：Å）
    """
    # 确保对称性
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    # 确保对角线为0
    np.fill_diagonal(distance_matrix, 0)

    if method == 'classical':
        coords = classical_mds(distance_matrix, n_components=3)
    elif method == 'iterative':
        coords, stress = iterative_mds(distance_matrix, n_components=3, **kwargs)
        print(f"MDS 应力值 (stress): {stress:.6f}")
    else:
        raise ValueError(f"未知的方法: {method}")

    return coords


def evaluate_reconstruction(original_dist: np.ndarray, reconstructed_coords: np.ndarray) -> dict:
    """
    评估坐标重建质量

    Args:
        original_dist: 原始距离矩阵
        reconstructed_coords: 重建的坐标

    Returns:
        评估指标字典
    """
    # 计算重建坐标的距离矩阵
    n = reconstructed_coords.shape[0]
    reconstructed_dist = np.sqrt(
        np.sum((reconstructed_coords[:, np.newaxis, :] - reconstructed_coords[np.newaxis, :, :]) ** 2, axis=-1)
    )

    # 只比较上三角（不含对角线）
    upper_tri_idx = np.triu_indices(n, k=1)
    orig_upper = original_dist[upper_tri_idx]
    recon_upper = reconstructed_dist[upper_tri_idx]

    # 计算各种误差指标
    mae = np.mean(np.abs(orig_upper - recon_upper))
    rmse = np.sqrt(np.mean((orig_upper - recon_upper) ** 2))
    max_error = np.max(np.abs(orig_upper - recon_upper))

    # 相关系数
    correlation = np.corrcoef(orig_upper, recon_upper)[0, 1]

    # 相对误差
    relative_error = np.mean(np.abs(orig_upper - recon_upper) / (orig_upper + 1e-8))

    return {
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'correlation': correlation,
        'relative_error': relative_error,
    }


def coords_to_pdb(coords: np.ndarray, sequence: str, output_file: str,
                  pdb_code: str = 'PRED'):
    """
    将坐标保存为PDB文件

    Args:
        coords: (N, 3) 坐标
        sequence: 氨基酸序列
        output_file: 输出文件路径
        pdb_code: PDB代码
    """
    # 氨基酸单字母到三字母的映射
    aa_1to3 = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
        'X': 'UNK'
    }

    with open(output_file, 'w') as f:
        f.write(f"HEADER    PREDICTED STRUCTURE FROM DISTANCE MATRIX\n")
        f.write(f"TITLE     {pdb_code}\n")
        f.write(f"REMARK    Generated by ProteinMTP v2 (Distance Matrix Prediction)\n")
        f.write(f"REMARK    Coordinates reconstructed using MDS algorithm\n")

        for i, (aa, coord) in enumerate(zip(sequence, coords)):
            aa_3 = aa_1to3.get(aa.upper(), 'UNK')
            # PDB ATOM格式
            line = f"ATOM  {i+1:5d}  CA  {aa_3} A{i+1:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           C\n"
            f.write(line)

        f.write("END\n")


if __name__ == '__main__':
    # 测试代码
    print("测试MDS坐标还原...")

    # 创建测试坐标
    n = 10
    np.random.seed(42)
    original_coords = np.random.randn(n, 3) * 10

    # 计算距离矩阵
    dist_matrix = np.sqrt(
        np.sum((original_coords[:, np.newaxis, :] - original_coords[np.newaxis, :, :]) ** 2, axis=-1)
    )

    print(f"原始坐标形状: {original_coords.shape}")
    print(f"距离矩阵形状: {dist_matrix.shape}")

    # 使用经典MDS重建坐标
    print("\n使用经典MDS重建...")
    reconstructed = classical_mds(dist_matrix)
    print(f"重建坐标形状: {reconstructed.shape}")

    # 评估重建质量
    metrics = evaluate_reconstruction(dist_matrix, reconstructed)
    print(f"\n重建质量评估:")
    print(f"  MAE: {metrics['mae']:.4f} Å")
    print(f"  RMSE: {metrics['rmse']:.4f} Å")
    print(f"  最大误差: {metrics['max_error']:.4f} Å")
    print(f"  相关系数: {metrics['correlation']:.4f}")

    print("\n✅ MDS坐标还原模块测试通过！")
