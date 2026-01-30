"""
ProteinMTP 推理脚本 - 距离矩阵预测版本 (v2)

预测蛋白质序列的距离矩阵，然后使用MDS还原3D坐标

流程：
1. 输入蛋白质序列
2. 模型预测距离矩阵 (N×N)
3. 反归一化距离矩阵
4. 使用MDS还原3D坐标
5. 输出PDB文件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model

from mds_reconstruct import (
    distance_matrix_to_coords,
    denormalize_distance_matrix,
    coords_to_pdb,
    evaluate_reconstruction
)


class LabelWiseAttention(nn.Module):
    """参考师兄代码的 LabelWiseAttention"""
    def __init__(self, inSize, classNum):
        super(LabelWiseAttention, self).__init__()
        self.U = nn.Linear(inSize, classNum)

    def forward(self, X):
        alpha = F.softmax(self.U(X), dim=1)
        X = torch.matmul(X.transpose(1, 2), alpha)
        return X.transpose(1, 2)


class DistanceMatrixMTP(nn.Module):
    """距离矩阵预测 MTP 模型"""

    def __init__(self, base_model, max_seq_len: int):
        super().__init__()
        self.base_model = base_model
        self.max_seq_len = max_seq_len

        config = base_model.config
        hidden_size = config.hidden_size

        self.dist_proj = LabelWiseAttention(hidden_size, max_seq_len)
        self.dist_head = nn.Linear(hidden_size, max_seq_len)

    def forward(self, input_ids, attention_mask, num_of_pl_tokens, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        tmp = outputs.hidden_states[-1]
        lm_part = tmp[..., :-num_of_pl_tokens, :]
        pl_part = tmp[..., -num_of_pl_tokens:, :]
        lm_part_pooled = lm_part.max(dim=1, keepdims=True)[0]

        dist_features = self.dist_proj(
            torch.cat([lm_part_pooled, pl_part], dim=1)
        )

        pred_dist = self.dist_head(dist_features)
        pred_dist = pred_dist[:, :num_of_pl_tokens, :num_of_pl_tokens]

        # 对称化
        pred_dist = (pred_dist + pred_dist.transpose(-1, -2)) / 2

        # 对角线设为0
        diag_mask = torch.eye(num_of_pl_tokens, device=pred_dist.device, dtype=pred_dist.dtype)
        pred_dist = pred_dist * (1 - diag_mask)

        return pred_dist


def load_model(model_path: str, device: str = 'cuda', use_4bit: bool = False):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")

    # 加载配置
    config_path = os.path.join(model_path, 'training_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"模型类型: {config.get('model_type', 'unknown')}")
    print(f"基础模型: {config['model_name']}")
    print(f"最大序列长度: {config['max_seq_len']}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 确定设备和精度
    if device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        if use_4bit:
            model_dtype = torch.float16
        elif torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float16
    else:
        device = 'cpu'
        model_dtype = torch.float32

    print(f"设备: {device}, 精度: {model_dtype}")

    # 加载模型权重
    model_pt_path = os.path.join(model_path, 'model.pt')

    if use_4bit and device == 'cuda':
        print("使用4bit量化加载...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_dtype,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            quantization_config=bnb_config,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            device_map='auto'
        )
    else:
        print(f"加载基础模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            torch_dtype=model_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

    # 应用LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=True,
        r=config.get('lora_rank', 16),
        lora_alpha=config.get('lora_alpha', 32),
        lora_dropout=0.0,
    )
    base_model = get_peft_model(base_model, lora_config)

    # 创建MTP模型
    model = DistanceMatrixMTP(base_model, max_seq_len=config['max_seq_len'])

    # 加载权重
    print("加载训练权重...")
    state_dict = torch.load(model_pt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    del state_dict

    if not use_4bit:
        model = model.to(device)

    model.eval()
    return model, tokenizer, config


def predict_distance_matrix(model, tokenizer, sequence: str, device: str = 'cuda'):
    """
    预测蛋白质序列的距离矩阵

    Args:
        model: 训练好的模型
        tokenizer: tokenizer
        sequence: 蛋白质序列
        device: 设备

    Returns:
        pred_dist: (N, N) 预测的距离矩阵（归一化后）
    """
    seq_len = len(sequence)

    # Tokenize
    inputs = tokenizer(sequence, return_tensors='pt')
    input_ids = inputs['input_ids'][0]
    attention_mask = inputs['attention_mask'][0]

    # 添加placeholder tokens
    unk_token_id = tokenizer.convert_tokens_to_ids('<unk>') if '<unk>' in tokenizer.get_vocab() else tokenizer.unk_token_id
    added_pl_tokens = [unk_token_id] * seq_len
    input_ids = torch.cat([input_ids, torch.tensor(added_pl_tokens, dtype=input_ids.dtype)])[None]
    attention_mask = torch.cat([attention_mask, torch.tensor([1] * seq_len, dtype=attention_mask.dtype)])[None]

    # 移动到设备
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # 推理
    with torch.no_grad():
        pred_dist = model(input_ids, attention_mask, num_of_pl_tokens=seq_len)

    return pred_dist[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='ProteinMTP 推理 - 距离矩阵预测版本')

    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--sequence', type=str, required=True,
                        help='蛋白质序列')
    parser.add_argument('--output', type=str, default='predicted_structure.pdb',
                        help='输出PDB文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--use_4bit', action='store_true',
                        help='使用4bit量化')
    parser.add_argument('--mds_method', type=str, default='classical',
                        choices=['classical', 'iterative'],
                        help='MDS坐标还原方法')
    parser.add_argument('--save_distmat', action='store_true',
                        help='同时保存距离矩阵')

    args = parser.parse_args()

    print("=" * 60)
    print("ProteinMTP 推理 - 距离矩阵预测版本 (v2)")
    print("=" * 60)
    print(f"序列: {args.sequence[:50]}..." if len(args.sequence) > 50 else f"序列: {args.sequence}")
    print(f"序列长度: {len(args.sequence)}")
    print()

    # 加载模型
    model, tokenizer, config = load_model(args.model_path, args.device, args.use_4bit)

    # 检查序列长度
    if len(args.sequence) > config['max_seq_len']:
        print(f"警告: 序列长度 {len(args.sequence)} 超过最大长度 {config['max_seq_len']}")
        print(f"将截断到 {config['max_seq_len']}")
        sequence = args.sequence[:config['max_seq_len']]
    else:
        sequence = args.sequence

    # 预测距离矩阵
    print("\n预测距离矩阵...")
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    pred_dist_norm = predict_distance_matrix(model, tokenizer, sequence, device)
    print(f"预测距离矩阵形状: {pred_dist_norm.shape}")

    # 反归一化距离矩阵
    print("\n反归一化距离矩阵...")
    norm_config = config.get('norm_config', {'norm_method': 'log_scale'})
    global_stats = config.get('global_stats', {})

    # 使用训练时保存的归一化参数
    # 这里我们假设使用 log_scale 方法
    # 实际项目中应该从训练配置中读取具体参数
    if norm_config.get('norm_method') == 'log_scale':
        # 估计 max_log_dist（如果没有保存，使用典型值）
        max_log_dist = 4.0  # log1p(50) ≈ 4.0，50Å是蛋白质的典型最大距离
        norm_stats = {'method': 'log_scale', 'max_log_dist': max_log_dist}
    else:
        norm_stats = {'method': 'scale', 'max_dist': 50.0}

    pred_dist = denormalize_distance_matrix(pred_dist_norm, norm_stats)

    print(f"距离矩阵统计（反归一化后）:")
    print(f"  最小距离: {pred_dist.min():.2f} Å")
    print(f"  最大距离: {pred_dist.max():.2f} Å")
    print(f"  平均距离: {pred_dist.mean():.2f} Å")

    # 保存距离矩阵（可选）
    if args.save_distmat:
        distmat_file = args.output.replace('.pdb', '_distmat.npy')
        np.save(distmat_file, pred_dist)
        print(f"\n距离矩阵已保存到: {distmat_file}")

    # 使用MDS还原坐标
    print(f"\n使用 {args.mds_method} MDS 还原3D坐标...")
    coords = distance_matrix_to_coords(pred_dist, method=args.mds_method)
    print(f"重建坐标形状: {coords.shape}")

    # 评估重建质量
    metrics = evaluate_reconstruction(pred_dist, coords)
    print(f"\nMDS重建质量:")
    print(f"  MAE: {metrics['mae']:.4f} Å")
    print(f"  RMSE: {metrics['rmse']:.4f} Å")
    print(f"  相关系数: {metrics['correlation']:.4f}")

    # 保存PDB文件
    print(f"\n保存PDB文件: {args.output}")
    coords_to_pdb(coords, sequence, args.output)

    print("\n推理完成！")
    print(f"输出文件: {args.output}")


if __name__ == '__main__':
    main()
