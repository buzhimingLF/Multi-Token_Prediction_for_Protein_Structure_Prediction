"""
测试8B模型推理功能
从 protein_sequences.json 中选择几个代表性样本进行推理测试
"""

import json
import os
import subproces
import sys

def select_test_samples(data_file='protein_sequences.json', n_samples=5):
    """选择测试样本"""
    with open(data_file, 'r') as f:
        data = json.load(f)

    # 按序列长度分层采样
    short = [d for d in data if d['sequence_length'] < 100]
    medium = [d for d in data if 100 <= d['sequence_length'] < 300]
    long = [d for d in data if 300 <= d['sequence_length'] <= 512]

    print(f"数据集统计:")
    print(f"  短序列 (<100): {len(short)}")
    print(f"  中等序列 (100-300): {len(medium)}")
    print(f"  长序列 (300-512): {len(long)}")
    print(f"  总计: {len(data)}")

    # 选择样本
    samples = []
    if short:
        samples.append(short[0])
    if medium:
        samples.append(medium[0])
        if len(medium) > 1:
            samples.append(medium[len(medium)//2])
    if long:
        samples.append(long[0])

    return samples[:n_samples]

def run_inference(model_path, sequence, pdb_code, output_dir='inference_results'):
    """运行推理"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{pdb_code}_predicted.pdb')

    cmd = [
        'python3', 'infer_protein_structure.py',
        '--model_path', model_path,
        '--sequence', sequence,
        '--output', output_file,
        '--use_4bit'  # 使用4bit量化减少内存
    ]

    print(f"\n{'='*60}")
    print(f"推理样本: {pdb_code}")
    print(f"序列长度: {len(sequence)}")
    print(f"序列: {sequence[:50]}..." if len(sequence) > 50 else f"序列: {sequence}")
    print(f"输出文件: {output_file}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            print("✅ 推理成功!")
            if os.path.exists(output_file):
                size = os.path.getsize(output_file)
                print(f"输出文件大小: {size} bytes")
            return True, output_file
        else:
            print("❌ 推理失败!")
            print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return False, None

    except subprocess.TimeoutExpired:
        print("❌ 推理超时 (>10分钟)")
        return False, None
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        return False, None

def main():
    model_path = 'output_structure_8b'

    if not os.path.exists(model_path):
        print(f"错误: 模型目录不存在: {model_path}")
        sys.exit(1)

    print("="*60)
    print("ProteinMTP 8B模型推理测试")
    print("="*60)

    # 选择测试样本
    print("\n1. 选择测试样本...")
    samples = select_test_samples(n_samples=3)  # 先测试3个样本

    print(f"\n选择的测试样本:")
    for i, s in enumerate(samples, 1):
        print(f"  {i}. {s['pdb_code']} - 长度: {s['sequence_length']}")

    # 运行推理
    print("\n2. 运行推理...")
    results = []
    for sample in samples:
        success, output_file = run_inference(
            model_path,
            sample['full_sequence'],
            sample['pdb_code']
        )
        results.append({
            'pdb_code': sample['pdb_code'],
            'sequence_length': sample['sequence_length'],
            'success': success,
            'output_file': output_file
        })

    # 汇总结果
    print("\n" + "="*60)
    print("推理结果汇总")
    print("="*60)
    successful = sum(1 for r in results if r['success'])
    print(f"成功: {successful}/{len(results)}")

    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"{status} {r['pdb_code']} (长度: {r['sequence_length']})")
        if r['success'] and r['output_file']:
            print(f"   输出: {r['output_file']}")

    # 如果有成功的，尝试可视化
    if successful > 0:
        print("\n3. 可视化预测结果...")
        for r in results:
            if r['success'] and r['output_file']:
                viz_output = r['output_file'].replace('.pdb', '_viz.png')
                cmd = [
                    'python3', 'visualize_structure.py',
                    '--pdb', r['output_file'],
                    '--output', viz_output,
                    '--mode', 'both',
                    '--stats'
                ]
                try:
                    subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    if os.path.exists(viz_output):
                        print(f"✅ 可视化完成: {viz_output}")
                except:
                    print(f"⚠️  可视化失败: {r['pdb_code']}")

if __name__ == '__main__':
    main()
