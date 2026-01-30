"""
ProteinMTP 训练脚本 - 距离矩阵预测版本 (v2)
基于 Multi-Token Prediction 的蛋白质结构预测模型

v2方案：预测原子间距离矩阵（旋转/平移不变）
- 输入：蛋白质序列
- 输出：Cα原子间的距离矩阵 (N, N)
- 优势：距离是旋转/平移不变的，模型更容易学习

参考：
- 师兄的 SFT_main.py (MTP核心逻辑)
- AlphaFold 的距离矩阵预测思路
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
import os
import warnings
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from torch.utils.data import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm

warnings.filterwarnings('ignore')


def get_device_info():
    """获取设备信息"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU 检测: {gpu_name} ({gpu_memory:.1f} GB)")
        return device, gpu_memory
    else:
        print("GPU 不可用，使用 CPU")
        return torch.device('cpu'), 0


class DistMatDataset(Dataset):
    """距离矩阵数据集"""

    def __init__(self, json_path: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"加载数据: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.samples = data['samples']
        self.global_stats = data.get('global_stats', {})
        self.config = data.get('config', {})

        print(f"加载了 {len(self.samples)} 个样本")
        print(f"全局距离统计: 平均={self.global_stats.get('mean_distance', 'N/A'):.2f}Å")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            'sequence': item['sequence'],
            'distance_matrix': item['distance_matrix'],  # (N, N) 归一化后的距离矩阵
            'seq_len': item['seq_len'],
            'pdb_code': item.get('pdb_code', ''),
            'dist_norm_stats': item.get('dist_norm_stats', {}),
        }


class DistMatCollator:
    """MTP 数据整理器 - 距离矩阵版本"""

    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        """
        关键：在输入末尾添加 num_of_pl_tokens 个 placeholder tokens
        对于距离矩阵预测：num_of_pl_tokens = seq_len (输出 N×N 矩阵)
        """
        sequences = [item['sequence'] for item in batch]
        dist_matrices = [torch.tensor(item['distance_matrix'], dtype=torch.float32) for item in batch]
        seq_lens = [item['seq_len'] for item in batch]

        # Tokenize 序列
        encodings = self.tokenizer(
            sequences,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 目前只支持 batch_size=1
        assert len(batch) == 1, "目前只支持 batch_size=1"

        input_ids = encodings['input_ids'][0]
        attention_mask = encodings['attention_mask'][0]

        # 添加 placeholder tokens
        # 对于距离矩阵：我们需要输出 N×N 个值
        # 但为了效率，我们只用 N 个 placeholder，每个预测 N 个距离值
        num_of_pl_tokens = seq_lens[0]
        unk_token_id = self.tokenizer.convert_tokens_to_ids('<unk>') if '<unk>' in self.tokenizer.get_vocab() else self.tokenizer.unk_token_id

        added_pl_tokens = [unk_token_id] * num_of_pl_tokens
        input_ids = torch.cat([input_ids, torch.tensor(added_pl_tokens, dtype=input_ids.dtype)])[None]
        attention_mask = torch.cat([attention_mask, torch.tensor([1] * num_of_pl_tokens, dtype=attention_mask.dtype)])[None]

        # 距离矩阵标签
        dist_matrix = dist_matrices[0]  # (seq_len, seq_len)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'distance_matrix': dist_matrix[None],  # (1, seq_len, seq_len)
            'num_of_pl_tokens': num_of_pl_tokens,
        }


class LabelWiseAttention(nn.Module):
    """
    参考师兄代码的 LabelWiseAttention
    将 (B, seq_len, hidden_size) 映射到 (B, class_num, hidden_size)
    """
    def __init__(self, inSize, classNum):
        super(LabelWiseAttention, self).__init__()
        self.U = nn.Linear(inSize, classNum)

    def forward(self, X):
        alpha = F.softmax(self.U(X), dim=1)
        X = torch.matmul(X.transpose(1, 2), alpha)
        return X.transpose(1, 2)


class DistanceMatrixMTP(nn.Module):
    """
    距离矩阵预测 MTP 模型

    输出: (B, N, N) 距离矩阵
    - 每个 placeholder token 预测该残基到所有其他残基的距离
    - 距离矩阵是对称的，但我们预测完整矩阵，损失函数会处理对称性
    """

    def __init__(self, base_model, max_seq_len: int):
        super().__init__()
        self.base_model = base_model
        self.max_seq_len = max_seq_len

        config = base_model.config
        hidden_size = config.hidden_size

        # LabelWiseAttention 映射 placeholder tokens
        self.dist_proj = LabelWiseAttention(hidden_size, max_seq_len)

        # 距离预测头：每个残基预测到所有残基的距离
        # 输出维度: hidden_size -> max_seq_len
        self.dist_head = nn.Linear(hidden_size, max_seq_len)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()

    def forward(self, input_ids, attention_mask, num_of_pl_tokens, distance_matrix=None, **kwargs):
        """
        Args:
            input_ids: (B, L+k) - L是输入长度，k是placeholder数量
            attention_mask: (B, L+k)
            num_of_pl_tokens: k - placeholder tokens数量（=序列长度N）
            distance_matrix: (B, N, N) - 真实距离矩阵（训练时提供）
        """
        # 获取 hidden states
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        tmp = outputs.hidden_states[-1]  # (B, L+k, hidden_size)

        # 分离输入部分和placeholder部分
        lm_part = tmp[..., :-num_of_pl_tokens, :]  # (B, L, hidden_size)
        pl_part = tmp[..., -num_of_pl_tokens:, :]   # (B, k, hidden_size)

        # 池化输入特征
        lm_part_pooled = lm_part.max(dim=1, keepdims=True)[0]  # (B, 1, hidden_size)

        # 通过 LabelWiseAttention 映射
        dist_features = self.dist_proj(
            torch.cat([lm_part_pooled, pl_part], dim=1)
        )  # (B, max_seq_len, hidden_size)

        # 预测距离矩阵
        # 每个位置预测到所有位置的距离
        pred_dist = self.dist_head(dist_features)  # (B, max_seq_len, max_seq_len)

        # 只取前 num_of_pl_tokens 个（实际序列长度）
        pred_dist = pred_dist[:, :num_of_pl_tokens, :num_of_pl_tokens]  # (B, N, N)

        # 对称化处理（距离矩阵应该是对称的）
        pred_dist = (pred_dist + pred_dist.transpose(-1, -2)) / 2

        # 对角线设为0（自身到自身的距离为0）
        # 创建对角线mask
        diag_mask = torch.eye(num_of_pl_tokens, device=pred_dist.device, dtype=pred_dist.dtype)
        pred_dist = pred_dist * (1 - diag_mask)

        # 计算损失
        loss = None
        metrics = {}
        if distance_matrix is not None:
            # 确保目标也是对称的且对角线为0
            target = distance_matrix[:, :num_of_pl_tokens, :num_of_pl_tokens]

            # 处理数据类型
            if pred_dist.dtype == torch.bfloat16:
                loss = F.mse_loss(pred_dist.float(), target.float())
            else:
                loss = F.mse_loss(pred_dist, target)

            # 计算额外指标
            with torch.no_grad():
                # MAE (Mean Absolute Error)
                if pred_dist.dtype == torch.bfloat16:
                    mae = F.l1_loss(pred_dist.float(), target.float())
                else:
                    mae = F.l1_loss(pred_dist, target)
                metrics['mae'] = mae.item()

                # 只计算上三角的误差（因为矩阵对称）
                upper_tri_mask = torch.triu(torch.ones_like(pred_dist), diagonal=1).bool()
                if pred_dist.dtype == torch.bfloat16:
                    pred_upper = pred_dist.float()[upper_tri_mask]
                    target_upper = target.float()[upper_tri_mask]
                else:
                    pred_upper = pred_dist[upper_tri_mask]
                    target_upper = target[upper_tri_mask]
                upper_mae = F.l1_loss(pred_upper, target_upper)
                metrics['upper_tri_mae'] = upper_mae.item()

        return {
            'loss': loss,
            'pred_distance_matrix': pred_dist,
            'metrics': metrics,
        }


class DistMatTrainer(Trainer):
    """距离矩阵预测训练器"""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        distance_matrix = inputs.pop("distance_matrix")
        num_of_pl_tokens = inputs.pop("num_of_pl_tokens")

        # 确保类型匹配
        model_dtype = next(model.parameters()).dtype
        if distance_matrix.dtype != model_dtype:
            distance_matrix = distance_matrix.to(dtype=model_dtype)

        outputs = model(**inputs, num_of_pl_tokens=num_of_pl_tokens, distance_matrix=distance_matrix)
        loss = outputs['loss']

        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description='ProteinMTP 训练脚本 - 距离矩阵预测版本 (v2)')

    # 数据参数
    parser.add_argument('--train_data', type=str, default='distmat_train.json')
    parser.add_argument('--val_data', type=str, default='distmat_val.json')

    # 模型参数
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--max_seq_len', type=int, default=256,
                        help='最大序列长度（距离矩阵为N×N，建议不超过256）')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='./output_distmat')

    # 大模型优化参数
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='启用梯度检查点以节省显存')
    parser.add_argument('--use_4bit', action='store_true',
                        help='使用4bit量化加载模型')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)

    args = parser.parse_args()

    print("=" * 60)
    print("ProteinMTP 训练 - 距离矩阵预测版本 (v2)")
    print("=" * 60)
    print("优势: 距离是旋转/平移不变的，模型更容易学习")
    print("=" * 60)
    print(f"模型: {args.model_name}")
    print(f"最大序列长度: {args.max_seq_len}")
    print(f"距离矩阵大小: {args.max_seq_len} × {args.max_seq_len}")
    print(f"学习率: {args.learning_rate}")
    print(f"梯度累积步数: {args.gradient_accumulation_steps}")
    print()

    # 检测设备
    device, gpu_memory = get_device_info()

    # 自动调整大模型配置
    model_size = args.model_name.lower()
    if '7b' in model_size or '8b' in model_size:
        print("\n检测到大模型(7B+)，自动启用优化...")
        if not args.gradient_checkpointing:
            args.gradient_checkpointing = True
            print("  - 自动启用梯度检查点")
        if args.gradient_accumulation_steps < 8:
            args.gradient_accumulation_steps = 8
            print(f"  - 自动增加梯度累积步数到 {args.gradient_accumulation_steps}")
        if args.lora_rank > 8:
            args.lora_rank = 8
            args.lora_alpha = 16
            print(f"  - 自动调整LoRA rank到 {args.lora_rank}")

    # 加载 tokenizer 和模型
    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {'trust_remote_code': True}

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            model_kwargs['torch_dtype'] = torch.bfloat16
            print("使用 BFloat16 精度")
        else:
            model_kwargs['torch_dtype'] = torch.float16
            print("使用 Float16 精度")
    else:
        model_kwargs['torch_dtype'] = torch.float32
        print("使用 Float32 精度 (CPU)")

    if args.use_4bit and torch.cuda.is_available():
        print("启用 4bit 量化...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs['quantization_config'] = bnb_config

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs
    )

    if args.gradient_checkpointing:
        print("启用梯度检查点...")
        base_model.gradient_checkpointing_enable()
        base_model.config.use_cache = False

    # 应用 LoRA
    print("应用 LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    # 创建 MTP 模型
    model = DistanceMatrixMTP(base_model, max_seq_len=args.max_seq_len)

    # 确保自定义层与base_model使用相同的dtype
    model_dtype = next(base_model.parameters()).dtype
    model.dist_proj = model.dist_proj.to(dtype=model_dtype)
    model.dist_head = model.dist_head.to(dtype=model_dtype)

    # 加载数据集
    print("\n加载数据集...")
    train_dataset = DistMatDataset(args.train_data, tokenizer, args.max_seq_len)
    val_dataset = DistMatDataset(args.val_data, tokenizer, args.max_seq_len) if os.path.exists(args.val_data) else None

    # 数据整理器
    collator = DistMatCollator(tokenizer, args.max_seq_len)

    # 训练参数
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=500 if val_dataset else None,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        remove_unused_columns=False,
        report_to="none",
        save_safetensors=False,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch" if not args.use_4bit else "paged_adamw_8bit",
        dataloader_pin_memory=True if torch.cuda.is_available() else False,
    )

    # 创建训练器
    trainer = DistMatTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # 开始训练
    print("\n开始训练...")
    trainer.train()

    # 保存模型
    print(f"\n保存模型到 {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt'))
    tokenizer.save_pretrained(args.output_dir)

    # 保存训练配置
    config_to_save = {
        'model_name': args.model_name,
        'max_seq_len': args.max_seq_len,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'model_type': 'distance_matrix',  # 标记为距离矩阵模型
        'global_stats': train_dataset.global_stats,
        'norm_config': train_dataset.config,
    }
    with open(os.path.join(args.output_dir, 'training_config.json'), 'w') as f:
        json.dump(config_to_save, f, indent=2)

    print(f"模型保存完成！")
    print(f"  - 模型权重: {os.path.join(args.output_dir, 'model.pt')}")
    print(f"  - Tokenizer: {args.output_dir}")
    print(f"  - 训练配置: {os.path.join(args.output_dir, 'training_config.json')}")

    print("\n训练完成！")


if __name__ == '__main__':
    main()
