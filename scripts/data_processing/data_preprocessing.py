"""
数据预处理脚本：从 PDBbind 数据集提取蛋白质序列和坐标
用于 ProteinMTP 项目的数据准备

修改说明（2026-01-25）：
- 原版本只提取序列信息
- 新版本同时提取序列和Cα原子坐标
- 为蛋白质结构预测任务准备训练数据
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

# 氨基酸三字母到单字母的映射
AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    # 非标准氨基酸
    'MSE': 'M',  # 硒代甲硫氨酸
    'HSD': 'H', 'HSE': 'H', 'HSP': 'H',  # 组氨酸变体
    'CYX': 'C',  # 二硫键半胱氨酸
}


def extract_sequence_from_pdb(pdb_file: str) -> Dict[str, str]:
    """
    从 PDB 文件中提取蛋白质序列（仅序列，不含坐标）

    Args:
        pdb_file: PDB 文件路径

    Returns:
        字典，键为链ID，值为氨基酸序列（单字母）
    """
    sequences = {}

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('SEQRES'):
                # SEQRES 格式：
                # SEQRES   1 Z  229  VAL ASP ASP ASP ASP LYS ILE VAL GLY GLY TYR THR CYS
                parts = line.split()
                chain_id = parts[2]
                residues = parts[4:]  # 氨基酸列表

                if chain_id not in sequences:
                    sequences[chain_id] = []

                for res in residues:
                    if res in AA_3TO1:
                        sequences[chain_id].append(AA_3TO1[res])
                    else:
                        sequences[chain_id].append('X')  # 未知氨基酸

    # 将列表转换为字符串
    return {chain: ''.join(seq) for chain, seq in sequences.items()}


def extract_sequence_and_coords_from_pdb(pdb_file: str, use_atom_only: bool = True) -> Tuple[Dict[str, str], Dict[str, List]]:
    """
    从 PDB 文件中提取蛋白质序列和Cα原子坐标

    Args:
        pdb_file: PDB 文件路径
        use_atom_only: 是否只从ATOM记录提取序列(确保序列和坐标一一对应)

    Returns:
        (sequences, coords) 元组
        - sequences: 字典，键为链ID，值为氨基酸序列（单字母）
        - coords: 字典，键为链ID，值为Cα坐标列表 [(x1,y1,z1), (x2,y2,z2), ...]
    """
    if use_atom_only:
        # 方案1: 从ATOM记录提取序列和坐标(推荐,确保一一对应)
        sequences = {}
        coords = {}

        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    chain_id = line[21]
                    residue = line[17:20].strip()

                    if chain_id not in sequences:
                        sequences[chain_id] = []
                        coords[chain_id] = []

                    # 提取氨基酸
                    aa = AA_3TO1.get(residue, 'X')
                    sequences[chain_id].append(aa)

                    # 提取坐标
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords[chain_id].append([x, y, z])

        # 转换序列为字符串
        sequences = {chain: ''.join(seq) for chain, seq in sequences.items()}

    else:
        # 方案2: 从SEQRES和ATOM分别提取(可能不匹配)
        sequences = {}
        coords = {}

        with open(pdb_file, 'r') as f:
            for line in f:
                # 提取序列（从SEQRES记录）
                if line.startswith('SEQRES'):
                    parts = line.split()
                    chain_id = parts[2]
                    residues = parts[4:]

                    if chain_id not in sequences:
                        sequences[chain_id] = []

                    for res in residues:
                        if res in AA_3TO1:
                            sequences[chain_id].append(AA_3TO1[res])
                        else:
                            sequences[chain_id].append('X')

                # 提取Cα原子坐标（从ATOM记录）
                elif line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    chain_id = line[21]
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    if chain_id not in coords:
                        coords[chain_id] = []
                    coords[chain_id].append([x, y, z])

        # 转换序列为字符串
        sequences = {chain: ''.join(seq) for chain, seq in sequences.items()}

    return sequences, coords


def scan_data_directory(data_dir: str) -> List[Dict]:
    """
    直接扫描数据目录,获取所有 PDB 文件
    用于没有索引文件的情况

    Args:
        data_dir: 数据目录路径

    Returns:
        复合物信息列表
    """
    complexes = []

    for subdir in ['1981-2000', '2001-2010', '2011-2019']:
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.exists(subdir_path):
            continue

        # 扫描该年份范围下的所有 PDB 目录
        for pdb_code in os.listdir(subdir_path):
            pdb_dir = os.path.join(subdir_path, pdb_code)
            if not os.path.isdir(pdb_dir):
                continue

            # 检查是否存在 protein.pdb 文件
            protein_pdb = os.path.join(pdb_dir, f"{pdb_code}_protein.pdb")
            if os.path.exists(protein_pdb):
                complexes.append({
                    'pdb_code': pdb_code,
                    'binding_data': 'N/A',
                    'resolution': 'N/A',
                    'year': subdir.split('-')[0]
                })

    return complexes


def parse_index_file(index_file: str) -> List[Dict]:
    """
    解析索引文件，提取复合物信息

    Args:
        index_file: 索引文件路径

    Returns:
        复合物信息列表
    """
    complexes = []
    
    with open(index_file, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if not line or line.startswith('#') or line.startswith('='):
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                pdb_code = parts[0]
                resolution = parts[1]
                year = parts[2]
                binding_data = parts[3]
                
                complexes.append({
                    'pdb_code': pdb_code,
                    'resolution': resolution,
                    'year': year,
                    'binding_data': binding_data
                })
    
    return complexes


def process_dataset(data_dir: str, index_file: str = None, output_file: str = None, max_samples: int = None, extract_coords: bool = False):
    """
    处理整个数据集，提取蛋白质序列（可选：同时提取坐标）

    Args:
        data_dir: P-L 数据目录
        index_file: 索引文件路径(如果为None或不存在,则直接扫描目录)
        output_file: 输出 JSON 文件路径
        max_samples: 最大样本数（用于测试）
        extract_coords: 是否提取坐标（用于结构预测任务）
    """
    import numpy as np

    # 解析索引文件或直接扫描目录
    if index_file and os.path.exists(index_file):
        print(f"解析索引文件: {index_file}")
        complexes = parse_index_file(index_file)
    else:
        print(f"索引文件不存在,直接扫描数据目录: {data_dir}")
        complexes = scan_data_directory(data_dir)

    print(f"找到 {len(complexes)} 个复合物")

    # 处理每个复合物
    results = []
    processed = 0
    errors = 0

    for complex_info in complexes:
        if max_samples and processed >= max_samples:
            break

        pdb_code = complex_info['pdb_code']

        # 查找 PDB 文件
        pdb_file = None
        for subdir in ['1981-2000', '2001-2010', '2011-2019']:
            potential_path = os.path.join(data_dir, subdir, pdb_code, f"{pdb_code}_protein.pdb")
            if os.path.exists(potential_path):
                pdb_file = potential_path
                break

        if not pdb_file:
            errors += 1
            continue

        try:
            if extract_coords:
                # 提取序列和坐标
                sequences, coords = extract_sequence_and_coords_from_pdb(pdb_file)

                if sequences and coords:
                    # 合并所有链的序列和坐标
                    full_sequence = ''.join(sequences.values())
                    full_coords = []
                    for chain in sequences.keys():
                        if chain in coords:
                            full_coords.extend(coords[chain])

                    # 检查序列长度和坐标长度是否匹配
                    if len(full_sequence) != len(full_coords):
                        print(f"警告: {pdb_code} 序列长度({len(full_sequence)})与坐标数量({len(full_coords)})不匹配，跳过")
                        errors += 1
                        continue

                    results.append({
                        'pdb_code': pdb_code,
                        'sequence': full_sequence,
                        'coords': full_coords,  # [[x,y,z], ...]
                        'seq_len': len(full_sequence),
                        'binding_data': complex_info['binding_data'],
                        'resolution': complex_info['resolution'],
                        'year': complex_info['year']
                    })
                    processed += 1

            else:
                # 只提取序列
                sequences = extract_sequence_from_pdb(pdb_file)

                if sequences:
                    full_sequence = ''.join(sequences.values())

                    results.append({
                        'pdb_code': pdb_code,
                        'sequences': sequences,
                        'full_sequence': full_sequence,
                        'sequence_length': len(full_sequence),
                        'binding_data': complex_info['binding_data'],
                        'resolution': complex_info['resolution'],
                        'year': complex_info['year']
                    })
                    processed += 1

            if processed % 1000 == 0:
                print(f"已处理 {processed} 个复合物...")

        except Exception as e:
            errors += 1
            print(f"处理 {pdb_code} 时出错: {e}")

    # 保存结果
    print(f"\n处理完成！")
    print(f"成功: {processed}, 错误: {errors}")
    print(f"保存到: {output_file}")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印统计信息
    if results:
        if extract_coords:
            lengths = [r['seq_len'] for r in results]
            print(f"\n序列长度统计:")
            print(f"  最短: {min(lengths)}")
            print(f"  最长: {max(lengths)}")
            print(f"  平均: {sum(lengths) / len(lengths):.1f}")
        else:
            lengths = [r['sequence_length'] for r in results]
            print(f"\n序列长度统计:")
            print(f"  最短: {min(lengths)}")
            print(f"  最长: {max(lengths)}")
            print(f"  平均: {sum(lengths) / len(lengths):.1f}")

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='PDBbind数据预处理')
    parser.add_argument('--extract_coords', action='store_true',
                        help='是否提取坐标（用于结构预测任务）')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大样本数（用于测试）')
    args = parser.parse_args()

    # 设置路径
    base_dir = Path(__file__).parent
    data_dir = base_dir / "P-L"
    index_file = base_dir / "index" / "INDEX_general_PL.2020R1.lst"

    if args.extract_coords:
        output_file = base_dir / "protein_coords_data.json"
    else:
        output_file = base_dir / "protein_sequences.json"

    # 检查数据是否存在
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        print("请先下载 PDBbind 数据集")
        return

    if not index_file.exists():
        print(f"警告: 索引文件不存在: {index_file}")
        print(f"将直接扫描数据目录: {data_dir}")

    # 处理数据集
    print("=" * 50)
    print(f"开始处理 PDBbind 数据集 (extract_coords={args.extract_coords})")
    print("=" * 50)

    results = process_dataset(
        str(data_dir),
        str(index_file),
        str(output_file),
        max_samples=args.max_samples,
        extract_coords=args.extract_coords
    )

    # 显示示例
    if results:
        print("\n示例数据:")
        sample = results[0]
        print(f"  PDB Code: {sample['pdb_code']}")

        if args.extract_coords:
            print(f"  序列长度: {sample['seq_len']}")
            print(f"  坐标数量: {len(sample['coords'])}")
            print(f"  序列前50个氨基酸: {sample['sequence'][:50]}...")
            print(f"  前3个Cα坐标: {sample['coords'][:3]}")
        else:
            print(f"  序列长度: {sample['sequence_length']}")
            print(f"  结合数据: {sample['binding_data']}")
            print(f"  序列前50个氨基酸: {sample['full_sequence'][:50]}...")


if __name__ == "__main__":
    main()
