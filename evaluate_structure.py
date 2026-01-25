"""
评估蛋白质结构预测结果
计算RMSD、TM-score等评估指标

用法:
    python evaluate_structure.py \
        --pred predicted_structure.pdb \
        --true true_structure.pdb
"""

import argparse
import numpy as np


def parse_pdb_coords(pdb_file: str):
    """从PDB文件提取Cα坐标"""
    coords = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and 'CA' in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords)


def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray, aligned: bool = False) -> float:
    """
    计算RMSD (Root Mean Square Deviation)

    Args:
        coords1: 第一组坐标 (N, 3)
        coords2: 第二组坐标 (N, 3)
        aligned: 是否已对齐(如果False,会先进行Kabsch对齐)

    Returns:
        RMSD值(单位: Å)
    """
    assert coords1.shape == coords2.shape, "坐标维度必须相同"

    if not aligned:
        # Kabsch对齐
        coords1, coords2 = kabsch_align(coords1, coords2)

    # 计算RMSD
    diff = coords1 - coords2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return rmsd


def kabsch_align(coords1: np.ndarray, coords2: np.ndarray):
    """
    Kabsch算法: 刚体对齐两组坐标

    Args:
        coords1: 第一组坐标 (N, 3)
        coords2: 第二组坐标 (N, 3)

    Returns:
        (aligned_coords1, coords2) 对齐后的坐标
    """
    # 中心化
    centroid1 = coords1.mean(axis=0)
    centroid2 = coords2.mean(axis=0)

    centered1 = coords1 - centroid1
    centered2 = coords2 - centroid2

    # 计算旋转矩阵
    H = centered1.T @ centered2
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 处理镜像
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 应用旋转和平移
    aligned1 = (R @ centered1.T).T + centroid2

    return aligned1, coords2


def compute_tm_score(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    计算TM-score (Template Modeling score)
    TM-score范围[0,1],越接近1表示结构越相似

    Args:
        coords1: 预测坐标 (N, 3)
        coords2: 真实坐标 (N, 3)

    Returns:
        TM-score值
    """
    N = len(coords1)

    # TM-score的归一化长度
    d0 = 1.24 * (N - 15) ** (1/3) - 1.8 if N > 15 else 0.5

    # Kabsch对齐
    aligned1, coords2 = kabsch_align(coords1, coords2)

    # 计算每个原子的距离
    distances = np.sqrt(np.sum((aligned1 - coords2)**2, axis=1))

    # 计算TM-score
    tm_score = np.sum(1 / (1 + (distances / d0)**2)) / N

    return tm_score


def compute_gdt_ts(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    计算GDT-TS (Global Distance Test - Total Score)
    GDT-TS范围[0,100],越高表示结构越相似

    Args:
        coords1: 预测坐标 (N, 3)
        coords2: 真实坐标 (N, 3)

    Returns:
        GDT-TS值
    """
    # Kabsch对齐
    aligned1, coords2 = kabsch_align(coords1, coords2)

    # 计算每个原子的距离
    distances = np.sqrt(np.sum((aligned1 - coords2)**2, axis=1))

    # 计算在不同阈值下的覆盖率
    thresholds = [1.0, 2.0, 4.0, 8.0]  # Å
    coverages = []

    for threshold in thresholds:
        coverage = np.sum(distances < threshold) / len(distances)
        coverages.append(coverage)

    # GDT-TS是四个阈值下覆盖率的平均值
    gdt_ts = np.mean(coverages) * 100

    return gdt_ts


def compute_contact_map_accuracy(coords1: np.ndarray, coords2: np.ndarray, threshold: float = 8.0):
    """
    计算接触图准确率
    接触定义为Cα距离小于threshold的氨基酸对

    Args:
        coords1: 预测坐标 (N, 3)
        coords2: 真实坐标 (N, 3)
        threshold: 接触距离阈值(Å)

    Returns:
        (precision, recall, f1)
    """
    N = len(coords1)

    # 计算距离矩阵
    def dist_matrix(coords):
        diff = coords[:, None, :] - coords[None, :, :]
        return np.sqrt(np.sum(diff**2, axis=2))

    dist1 = dist_matrix(coords1)
    dist2 = dist_matrix(coords2)

    # 接触图(只考虑序列距离>5的氨基酸对)
    mask = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) > 5

    contacts1 = (dist1 < threshold) & mask
    contacts2 = (dist2 < threshold) & mask

    # 计算TP, FP, FN
    tp = np.sum(contacts1 & contacts2)
    fp = np.sum(contacts1 & ~contacts2)
    fn = np.sum(~contacts1 & contacts2)

    # 计算precision, recall, f1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description='评估蛋白质结构预测')

    parser.add_argument('--pred', type=str, required=True,
                        help='预测的PDB文件')
    parser.add_argument('--true', type=str, required=True,
                        help='真实的PDB文件')
    parser.add_argument('--all_metrics', action='store_true',
                        help='计算所有评估指标(包括TM-score, GDT-TS等)')

    args = parser.parse_args()

    print("=" * 60)
    print("蛋白质结构预测评估")
    print("=" * 60)

    # 读取坐标
    print(f"读取预测结构: {args.pred}")
    pred_coords = parse_pdb_coords(args.pred)
    print(f"  预测坐标数量: {len(pred_coords)}")

    print(f"读取真实结构: {args.true}")
    true_coords = parse_pdb_coords(args.true)
    print(f"  真实坐标数量: {len(true_coords)}")

    # 检查长度是否匹配
    if len(pred_coords) != len(true_coords):
        print(f"\n警告: 坐标数量不匹配!")
        print(f"  预测: {len(pred_coords)}, 真实: {len(true_coords)}")
        # 取较短的长度
        min_len = min(len(pred_coords), len(true_coords))
        pred_coords = pred_coords[:min_len]
        true_coords = true_coords[:min_len]
        print(f"  截取到前 {min_len} 个氨基酸")

    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)

    # 1. RMSD(未对齐)
    rmsd_unaligned = compute_rmsd(pred_coords, true_coords, aligned=False)
    print(f"\nRMSD(Kabsch对齐): {rmsd_unaligned:.3f} Å")

    # 2. RMSD(对齐后)
    aligned_pred, _ = kabsch_align(pred_coords, true_coords)
    rmsd_aligned = compute_rmsd(aligned_pred, true_coords, aligned=True)
    print(f"RMSD(对齐后): {rmsd_aligned:.3f} Å")

    if args.all_metrics:
        # 3. TM-score
        print("\n计算TM-score...")
        tm_score = compute_tm_score(pred_coords, true_coords)
        print(f"TM-score: {tm_score:.4f}")

        # 4. GDT-TS
        print("\n计算GDT-TS...")
        gdt_ts = compute_gdt_ts(pred_coords, true_coords)
        print(f"GDT-TS: {gdt_ts:.2f}")

        # 5. 接触图准确率
        print("\n计算接触图准确率(8Å阈值)...")
        precision, recall, f1 = compute_contact_map_accuracy(pred_coords, true_coords, threshold=8.0)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

    # 解释结果
    print("\n" + "=" * 60)
    print("评估标准参考")
    print("=" * 60)
    print("RMSD:")
    print("  < 2.0 Å  : 高质量预测")
    print("  2.0-5.0 Å: 中等质量")
    print("  > 5.0 Å  : 低质量")

    if args.all_metrics:
        print("\nTM-score:")
        print("  > 0.5: 相同折叠")
        print("  0.4-0.5: 相似折叠")
        print("  < 0.4: 不同折叠")

        print("\nGDT-TS:")
        print("  > 50: 高质量")
        print("  30-50: 中等质量")
        print("  < 30: 低质量")


if __name__ == '__main__':
    main()
