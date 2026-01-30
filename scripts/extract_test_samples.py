"""
从 protein_sequences.json 提取测试样本
按序列长度分层选择代表性样本
"""

import json

def main():
    print("加载数据集...")
    with open('protein_sequences.json', 'r') as f:
        data = json.load(f)

    # 按长度分类
    short = [d for d in data if d['sequence_length'] < 100]
    medium = [d for d in data if 100 <= d['sequence_length'] < 300]
    long = [d for d in data if 300 <= d['sequence_length'] <= 512]
    very_long = [d for d in data if d['sequence_length'] > 512]

    print(f"\n数据集统计:")
    print(f"  短序列 (<100 aa): {len(short)}")
    print(f"  中等序列 (100-300 aa): {len(medium)}")
    print(f"  长序列 (300-512 aa): {len(long)}")
    print(f"  超长序列 (>512 aa): {len(very_long)} (超出max_seq_len，不可用)")

    # 选择代表性样本
    test_samples = []

    if short:
        sample = short[len(short)//2]  # 中位数位置
        test_samples.append({
            'pdb_code': sample['pdb_code'],
            'sequence': sample['full_sequence'],
            'length': sample['sequence_length'],
            'category': 'short',
            'binding_data': sample.get('binding_data', 'N/A'),
            'resolution': sample.get('resolution', 'N/A')
        })
        print(f"\n✓ 短序列样本: {sample['pdb_code']} ({sample['sequence_length']} aa)")

    if medium:
        sample = medium[0]  # 第一个
        test_samples.append({
            'pdb_code': sample['pdb_code'],
            'sequence': sample['full_sequence'],
            'length': sample['sequence_length'],
            'category': 'medium',
            'binding_data': sample.get('binding_data', 'N/A'),
            'resolution': sample.get('resolution', 'N/A')
        })
        print(f"✓ 中等序列样本: {sample['pdb_code']} ({sample['sequence_length']} aa)")

    if long:
        sample = long[min(10, len(long)-1)]  # 选择第10个或最后一个
        test_samples.append({
            'pdb_code': sample['pdb_code'],
            'sequence': sample['full_sequence'],
            'length': sample['sequence_length'],
            'category': 'long',
            'binding_data': sample.get('binding_data', 'N/A'),
            'resolution': sample.get('resolution', 'N/A')
        })
        print(f"✓ 长序列样本: {sample['pdb_code']} ({sample['sequence_length']} aa)")

    # 保存测试样本
    output_file = 'test_samples.json'
    with open(output_file, 'w') as f:
        json.dump(test_samples, f, indent=2)

    print(f"\n测试样本已保存到: {output_file}")
    print(f"共 {len(test_samples)} 个样本")

    # 输出详细信息
    print("\n" + "="*60)
    print("测试样本详情:")
    print("="*60)
    for i, sample in enumerate(test_samples, 1):
        print(f"\n{i}. {sample['pdb_code']} ({sample['category']})")
        print(f"   长度: {sample['length']} aa")
        print(f"   结合数据: {sample['binding_data']}")
        print(f"   分辨率: {sample['resolution']} Å")
        print(f"   序列前50个: {sample['sequence'][:50]}...")

if __name__ == '__main__':
    main()
