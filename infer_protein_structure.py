"""
ProteinMTP 推理脚本 - 坐标预测
使用训练好的模型预测蛋白质结构

用法:
    python infer_protein_structure.py \
        --model_path ./output_structure \
        --sequence "MKTAYIAKQRQISFVK" \
        --output predicted_structure.pdb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from train_protein_structure import LabelWiseAttention, ProteinStructureMTP


def load_model(model_path: str, device: str = 'cuda'):
    """
    加载训练好的模型

    Args:
        model_path: 模型路径
        device: 设备 ('cuda' or 'cpu')

    Returns:
        (model, tokenizer, config) 元组
    """
    print(f"加载模型: {model_path}")

    # 加载配置
    config_path = os.path.join(model_path, 'training_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # 默认配置
        config = {
            'max_seq_len': 512,
            'model_name': 'Qwen/Qwen2.5-0.5B'
        }
        print("警告: 未找到训练配置,使用默认配置")

    # 加载tokenizer（从原始模型或保存的tokenizer）
    if os.path.exists(os.path.join(model_path, 'tokenizer_config.json')):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 检查是否使用新的保存格式（model.pt）
    model_pt_path = os.path.join(model_path, 'model.pt')

    if os.path.exists(model_pt_path):
        # 新格式：直接加载model.pt
        print("使用新格式加载模型（model.pt）")

        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            torch_dtype=torch.float32,
            trust_remote_code=True
        )

        # 应用LoRA配置（必须与训练时相同）
        from peft import LoraConfig, TaskType, get_peft_model
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=True,
            r=config.get('lora_rank', 8),
            lora_alpha=config.get('lora_alpha', 16),
            lora_dropout=0.0,
        )
        base_model = get_peft_model(base_model, lora_config)

        # 创建MTP模型
        model = ProteinStructureMTP(base_model, max_seq_len=config['max_seq_len'])

        # 加载训练好的权重
        state_dict = torch.load(model_pt_path, map_location=device)
        model.load_state_dict(state_dict)

    else:
        # 旧格式：使用PeftModel加载
        print("使用旧格式加载模型（PEFT）")
        base_model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            torch_dtype=torch.float32,
            trust_remote_code=True
        )

        # 加载LoRA权重
        from peft import PeftModel
        base_model = PeftModel.from_pretrained(base_model, model_path)
        base_model = base_model.merge_and_unload()  # 合并LoRA权重

        # 创建MTP模型
        model = ProteinStructureMTP(base_model, max_seq_len=config['max_seq_len'])

    model = model.to(device)
    model.eval()

    return model, tokenizer, config


def predict_structure(model, tokenizer, sequence: str, device: str = 'cuda'):
    """
    预测蛋白质结构

    Args:
        model: 训练好的模型
        tokenizer: tokenizer
        sequence: 氨基酸序列
        device: 设备

    Returns:
        预测的坐标 (seq_len, 3)
    """
    seq_len = len(sequence)

    # Tokenize
    encoding = tokenizer(
        sequence,
        return_tensors='pt',
        truncation=True,
        max_length=512
    )

    input_ids = encoding['input_ids'][0]
    attention_mask = encoding['attention_mask'][0]

    # 添加placeholder tokens
    unk_token_id = tokenizer.convert_tokens_to_ids('<unk>') if '<unk>' in tokenizer.get_vocab() else tokenizer.unk_token_id
    added_pl_tokens = [unk_token_id] * seq_len
    input_ids = torch.cat([input_ids, torch.tensor(added_pl_tokens, dtype=input_ids.dtype)])[None]
    attention_mask = torch.cat([attention_mask, torch.tensor([1] * seq_len, dtype=attention_mask.dtype)])[None]

    # 移到设备
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # 推理
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_of_pl_tokens=seq_len
        )

    pred_coords = outputs['pred_coords'][0].cpu().numpy()  # (seq_len, 3)
    return pred_coords


def denormalize_coords(coords: np.ndarray, norm_stats: dict) -> np.ndarray:
    """
    反归一化坐标

    Args:
        coords: 归一化后的坐标
        norm_stats: 归一化统计信息 {'mean': [...], 'std': ...}

    Returns:
        反归一化后的坐标
    """
    if norm_stats['std'] is not None:
        # 反标准化
        coords = coords * norm_stats['std']

    # 反零均值化
    coords = coords + np.array(norm_stats['mean'])

    return coords


def save_as_pdb(sequence: str, coords: np.ndarray, output_file: str):
    """
    保存为PDB格式

    Args:
        sequence: 氨基酸序列
        coords: 坐标数组 (seq_len, 3)
        output_file: 输出文件路径
    """
    # 单字母到三字母映射
    AA_1TO3 = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
        'X': 'UNK'
    }

    with open(output_file, 'w') as f:
        # 写入头部
        f.write("HEADER    PREDICTED PROTEIN STRUCTURE\n")
        f.write("REMARK   Generated by ProteinMTP\n")

        # 写入ATOM记录
        for i, (aa, coord) in enumerate(zip(sequence, coords), start=1):
            residue = AA_1TO3.get(aa, 'UNK')
            x, y, z = coord

            # PDB ATOM格式
            line = f"ATOM  {i:5d}  CA  {residue} A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            f.write(line)

        f.write("END\n")

    print(f"结构已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='ProteinMTP 推理脚本')

    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--sequence', type=str, required=True,
                        help='输入的氨基酸序列')
    parser.add_argument('--output', type=str, default='predicted_structure.pdb',
                        help='输出PDB文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')

    # 归一化参数(如果训练时使用了归一化)
    parser.add_argument('--norm_mean', type=float, nargs=3, default=None,
                        help='归一化均值 (x y z)')
    parser.add_argument('--norm_std', type=float, default=None,
                        help='归一化标准差')

    args = parser.parse_args()

    print("=" * 60)
    print("ProteinMTP 推理 - 蛋白质结构预测")
    print("=" * 60)
    print(f"序列: {args.sequence}")
    print(f"序列长度: {len(args.sequence)}")
    print()

    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用,使用CPU")
        args.device = 'cpu'

    # 加载模型
    model, tokenizer, config = load_model(args.model_path, args.device)

    # 预测结构
    print("预测结构...")
    pred_coords = predict_structure(model, tokenizer, args.sequence, args.device)

    print(f"预测完成! 坐标形状: {pred_coords.shape}")

    # 反归一化(如果需要)
    if args.norm_mean is not None and args.norm_std is not None:
        print("反归一化坐标...")
        norm_stats = {
            'mean': args.norm_mean,
            'std': args.norm_std
        }
        pred_coords = denormalize_coords(pred_coords, norm_stats)

    # 保存为PDB
    save_as_pdb(args.sequence, pred_coords, args.output)

    # 显示前3个坐标
    print("\n前3个Cα坐标:")
    for i, coord in enumerate(pred_coords[:3], start=1):
        print(f"  {i}: [{coord[0]:8.3f}, {coord[1]:8.3f}, {coord[2]:8.3f}]")


if __name__ == '__main__':
    main()
