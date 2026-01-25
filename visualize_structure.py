"""
可视化蛋白质结构
使用matplotlib简单可视化Cα骨架

用法:
    python visualize_structure.py --pdb predicted_structure.pdb --output structure.png
"""

import argparse
import numpy as np
from pathlib import Path


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


def visualize_3d(coords: np.ndarray, output_file: str = None, title: str = "Protein Structure"):
    """
    3D可视化蛋白质结构

    Args:
        coords: (N, 3) 坐标数组
        output_file: 输出图片路径
        title: 图片标题
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("错误: 需要安装matplotlib")
        print("运行: pip install matplotlib")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制骨架
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
            'b-', linewidth=1.5, alpha=0.6, label='Backbone')

    # 绘制Cα原子
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=np.arange(len(coords)), cmap='viridis',
               s=50, alpha=0.8, label='Cα atoms')

    # 设置标签
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_zlabel('Z (Å)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()

    # 调整视角
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {output_file}")
    else:
        plt.show()

    plt.close()


def visualize_2d_projections(coords: np.ndarray, output_file: str = None):
    """
    2D投影可视化(XY, XZ, YZ三个平面)

    Args:
        coords: (N, 3) 坐标数组
        output_file: 输出图片路径
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("错误: 需要安装matplotlib")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # XY平面
    axes[0].plot(coords[:, 0], coords[:, 1], 'b-', linewidth=1, alpha=0.6)
    axes[0].scatter(coords[:, 0], coords[:, 1], c=np.arange(len(coords)),
                    cmap='viridis', s=30, alpha=0.8)
    axes[0].set_xlabel('X (Å)')
    axes[0].set_ylabel('Y (Å)')
    axes[0].set_title('XY Projection')
    axes[0].grid(True, alpha=0.3)

    # XZ平面
    axes[1].plot(coords[:, 0], coords[:, 2], 'b-', linewidth=1, alpha=0.6)
    axes[1].scatter(coords[:, 0], coords[:, 2], c=np.arange(len(coords)),
                    cmap='viridis', s=30, alpha=0.8)
    axes[1].set_xlabel('X (Å)')
    axes[1].set_ylabel('Z (Å)')
    axes[1].set_title('XZ Projection')
    axes[1].grid(True, alpha=0.3)

    # YZ平面
    axes[2].plot(coords[:, 1], coords[:, 2], 'b-', linewidth=1, alpha=0.6)
    axes[2].scatter(coords[:, 1], coords[:, 2], c=np.arange(len(coords)),
                    cmap='viridis', s=30, alpha=0.8)
    axes[2].set_xlabel('Y (Å)')
    axes[2].set_ylabel('Z (Å)')
    axes[2].set_title('YZ Projection')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"2D投影图已保存到: {output_file}")
    else:
        plt.show()

    plt.close()


def compute_statistics(coords: np.ndarray):
    """计算结构统计信息"""
    # 计算相邻Cα距离
    distances = []
    for i in range(len(coords) - 1):
        dist = np.linalg.norm(coords[i+1] - coords[i])
        distances.append(dist)

    distances = np.array(distances)

    # 计算回转半径(radius of gyration)
    centroid = coords.mean(axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - centroid)**2, axis=1)))

    # 计算最大距离
    max_dist = 0
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist > max_dist:
                max_dist = dist

    print("\n结构统计信息:")
    print(f"  氨基酸数量: {len(coords)}")
    print(f"  相邻Cα平均距离: {distances.mean():.2f} Å")
    print(f"  相邻Cα距离标准差: {distances.std():.2f} Å")
    print(f"  相邻Cα距离范围: [{distances.min():.2f}, {distances.max():.2f}] Å")
    print(f"  回转半径: {rg:.2f} Å")
    print(f"  最大原子间距离: {max_dist:.2f} Å")
    print(f"  质心坐标: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")


def main():
    parser = argparse.ArgumentParser(description='可视化蛋白质结构')

    parser.add_argument('--pdb', type=str, required=True,
                        help='PDB文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出图片路径')
    parser.add_argument('--mode', type=str, default='3d',
                        choices=['3d', '2d', 'both'],
                        help='可视化模式: 3d/2d/both')
    parser.add_argument('--stats', action='store_true',
                        help='显示统计信息')

    args = parser.parse_args()

    print("=" * 60)
    print("蛋白质结构可视化")
    print("=" * 60)

    # 读取PDB
    print(f"读取PDB文件: {args.pdb}")
    coords = parse_pdb_coords(args.pdb)
    print(f"读取了 {len(coords)} 个Cα原子")

    # 统计信息
    if args.stats or args.output is None:
        compute_statistics(coords)

    # 可视化
    if args.mode in ['3d', 'both']:
        output_3d = args.output if args.output else None
        if args.mode == 'both' and args.output:
            output_3d = str(Path(args.output).with_suffix('')) + '_3d.png'

        print("\n生成3D可视化...")
        visualize_3d(coords, output_3d, title=f"Protein Structure ({len(coords)} residues)")

    if args.mode in ['2d', 'both']:
        output_2d = args.output if args.output else None
        if args.mode == 'both' and args.output:
            output_2d = str(Path(args.output).with_suffix('')) + '_2d.png'

        print("\n生成2D投影...")
        visualize_2d_projections(coords, output_2d)

    print("\n完成!")


if __name__ == '__main__':
    main()
