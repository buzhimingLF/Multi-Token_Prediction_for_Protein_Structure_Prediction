"""
数据预处理脚本：从 PDBbind 数据集提取蛋白质序列
用于 ProteinMTP 项目的数据准备
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
    从 PDB 文件中提取蛋白质序列
    
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


def process_dataset(data_dir: str, index_file: str, output_file: str, max_samples: int = None):
    """
    处理整个数据集，提取蛋白质序列
    
    Args:
        data_dir: P-L 数据目录
        index_file: 索引文件路径
        output_file: 输出 JSON 文件路径
        max_samples: 最大样本数（用于测试）
    """
    # 解析索引文件
    print(f"解析索引文件: {index_file}")
    complexes = parse_index_file(index_file)
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
            # 提取序列
            sequences = extract_sequence_from_pdb(pdb_file)
            
            if sequences:
                # 合并所有链的序列
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
        lengths = [r['sequence_length'] for r in results]
        print(f"\n序列长度统计:")
        print(f"  最短: {min(lengths)}")
        print(f"  最长: {max(lengths)}")
        print(f"  平均: {sum(lengths) / len(lengths):.1f}")
    
    return results


def main():
    """主函数"""
    # 设置路径
    base_dir = Path(__file__).parent
    data_dir = base_dir / "P-L"
    index_file = base_dir / "index" / "INDEX_general_PL.2020R1.lst"
    output_file = base_dir / "protein_sequences.json"
    
    # 检查数据是否存在
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        print("请先下载 PDBbind 数据集")
        return
    
    if not index_file.exists():
        print(f"错误: 索引文件不存在: {index_file}")
        return
    
    # 处理数据集（先处理100个样本测试）
    print("=" * 50)
    print("开始处理 PDBbind 数据集")
    print("=" * 50)
    
    # 处理全部数据
    results = process_dataset(
        str(data_dir),
        str(index_file),
        str(output_file),
        max_samples=None  # 处理全部数据
    )
    
    # 显示示例
    if results:
        print("\n示例数据:")
        sample = results[0]
        print(f"  PDB Code: {sample['pdb_code']}")
        print(f"  序列长度: {sample['sequence_length']}")
        print(f"  结合数据: {sample['binding_data']}")
        print(f"  序列前50个氨基酸: {sample['full_sequence'][:50]}...")


if __name__ == "__main__":
    main()
