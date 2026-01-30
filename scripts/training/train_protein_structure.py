"""
ProteinMTP 训练脚本 - 坐标回归版本
基于 Multi-Token Prediction 的蛋白质结构预测模型

参考师兄的 SFT_main.py，改造为坐标回归任务：
- 使用 placeholder tokens 限定输出长度为 k=序列长度
- 输出从分类logits改为坐标 (x,y,z)
- 损失函数从分类损失改为MSE回归损失

支持的模型规模：
- Qwen2.5-0.5B: 适合快速验证，CPU/GPU均可
- Qwen2.5-1.5B: 中等规模，需要GPU
- Qwen2.5-7B: 大规模，需要GPU + 梯度检查点 + 混合精度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
import os
import warnings
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

# 检测GPU可用性
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


class ProteinCoordDataset(Dataset):
    """蛋白质坐标数据集"""

    def __init__(self, json_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"加载数据: {json_path}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        print(f"加载了 {len(self.data)} 个样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'sequence': item['sequence'],
            'coords': item['coords'],  # [[x,y,z], ...]
            'seq_len': item['seq_len'],
            'pdb_code': item.get('pdb_code', ''),
        }


class ProteinCoordCollator:
    """MTP 数据整理器 - 坐标回归版本"""

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        """
        参考师兄代码的 Qwen3_VL_Lora_Collator
        关键：在输入末尾添加 num_of_pl_tokens 个 placeholder tokens
        """
        sequences = [item['sequence'] for item in batch]
        coords_list = [torch.tensor(item['coords'], dtype=torch.float32) for item in batch]
        seq_lens = [item['seq_len'] for item in batch]

        # Tokenize 序列
        encodings = self.tokenizer(
            sequences,
            padding=False,  # 手动padding
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 处理batch（目前只支持batch_size=1，与师兄代码一致）
        assert len(batch) == 1, "目前只支持 batch_size=1"

        input_ids = encodings['input_ids'][0]
        attention_mask = encodings['attention_mask'][0]

        # 添加 placeholder tokens（参考师兄代码）
        # num_of_pl_tokens = 序列长度
        num_of_pl_tokens = seq_lens[0]
        unk_token_id = self.tokenizer.convert_tokens_to_ids('<unk>') if '<unk>' in self.tokenizer.get_vocab() else self.tokenizer.unk_token_id

        added_pl_tokens = [unk_token_id] * num_of_pl_tokens
        input_ids = torch.cat([input_ids, torch.tensor(added_pl_tokens, dtype=input_ids.dtype)])[None]
        attention_mask = torch.cat([attention_mask, torch.tensor([1] * num_of_pl_tokens, dtype=attention_mask.dtype)])[None]

        # 坐标标签
        coords = coords_list[0]  # (seq_len, 3)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'coords': coords[None],  # (1, seq_len, 3)
            'num_of_pl_tokens': num_of_pl_tokens,
        }


class LabelWiseAttention(nn.Module):
    """
    参考师兄代码的 LabelWiseAttention
    将 (B, seq_len, hidden_size) 映射到 (B, class_num, hidden_size)
    在我们的任务中：class_num = 序列长度
    """
    def __init__(self, inSize, classNum):
        super(LabelWiseAttention, self).__init__()
        self.U = nn.Linear(inSize, classNum)

    def forward(self, X):
        # X: batchSize × seqLen × inSize
        alpha = F.softmax(self.U(X), dim=1)  # => batchSize × seqLen × classNum
        X = torch.matmul(X.transpose(1, 2), alpha)  # => batchSize × inSize × classNum
        return X.transpose(1, 2)


class ProteinStructureMTP(nn.Module):
    """
    蛋白质结构预测 MTP 模型
    参考师兄的 HuggingfaceModelWithMTP，改造为坐标回归
    """

    def __init__(self, base_model, max_seq_len: int):
        super().__init__()
        self.base_model = base_model
        self.max_seq_len = max_seq_len

        # 获取隐藏层大小
        config = base_model.config
        hidden_size = config.hidden_size

        # 参考师兄代码的 LabelWiseAttention
        # 将 placeholder tokens 的 hidden states 映射到每个氨基酸
        self.coord_proj = LabelWiseAttention(hidden_size, max_seq_len)

        # 坐标预测头：将特征映射到 3D 坐标
        self.coord_head = nn.Linear(hidden_size, 3)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """启用梯度检查点（委托给base_model）"""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """禁用梯度检查点（委托给base_model）"""
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()

    def compute_rmsd(self, pred_coords, true_coords):
        """
        计算RMSD (Root Mean Square Deviation)
        这是蛋白质结构预测的标准评估指标

        Args:
            pred_coords: (B, seq_len, 3) 预测坐标
            true_coords: (B, seq_len, 3) 真实坐标

        Returns:
            rmsd: 标量RMSD值
        """
        # 计算每个原子的距离平方
        dist_sq = torch.sum((pred_coords - true_coords) ** 2, dim=-1)  # (B, seq_len)
        # 计算均方根
        rmsd = torch.sqrt(torch.mean(dist_sq))
        return rmsd

    def forward(self, input_ids, attention_mask, num_of_pl_tokens, coords=None, **kwargs):
        """
        参考师兄代码的 forward 逻辑

        Args:
            input_ids: (B, L+k) - L是输入长度，k是placeholder数量
            attention_mask: (B, L+k)
            num_of_pl_tokens: k - placeholder tokens数量（=序列长度）
            coords: (B, seq_len, 3) - 真实坐标（训练时提供）
        """
        # 获取 hidden states（参考师兄代码）
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        tmp = outputs.hidden_states[-1]  # (B, L+k, hidden_size)

        # 分离输入部分和placeholder部分（参考师兄代码）
        lm_part = tmp[..., :-num_of_pl_tokens, :]  # (B, L, hidden_size)
        pl_part = tmp[..., -num_of_pl_tokens:, :]   # (B, k, hidden_size)

        # 池化输入特征（参考师兄代码）
        lm_part_pooled = lm_part.max(dim=1, keepdims=True)[0]  # (B, 1, hidden_size)

        # 通过 LabelWiseAttention 映射（参考师兄代码）
        coord_features = self.coord_proj(
            torch.cat([lm_part_pooled, pl_part], dim=1)
        )  # (B, max_seq_len, hidden_size)

        # 预测坐标
        pred_coords = self.coord_head(coord_features)  # (B, max_seq_len, 3)

        # 只取前 num_of_pl_tokens 个（实际序列长度）
        pred_coords = pred_coords[:, :num_of_pl_tokens, :]  # (B, seq_len, 3)

        # 计算损失和指标
        loss = None
        metrics = {}
        if coords is not None:
            # CPU不支持BFloat16的MSE，需要转换为float32
            if pred_coords.dtype == torch.bfloat16:
                loss = F.mse_loss(pred_coords.float(), coords.float())
            else:
                loss = F.mse_loss(pred_coords, coords)

            # 计算RMSD作为额外的评估指标
            with torch.no_grad():
                if pred_coords.dtype == torch.bfloat16:
                    rmsd = self.compute_rmsd(pred_coords.float(), coords.float())
                else:
                    rmsd = self.compute_rmsd(pred_coords, coords)
                metrics['rmsd'] = rmsd.item()

        return {
            'loss': loss,
            'pred_coords': pred_coords,
            'metrics': metrics,
        }


class CoordMTPTrainer(Trainer):
    """坐标回归 MTP 训练器"""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        coords = inputs.pop("coords")
        num_of_pl_tokens = inputs.pop("num_of_pl_tokens")

        # 确保coords与模型使用相同的dtype
        model_dtype = next(model.parameters()).dtype
        if coords.dtype != model_dtype:
            coords = coords.to(dtype=model_dtype)

        outputs = model(**inputs, num_of_pl_tokens=num_of_pl_tokens, coords=coords)
        loss = outputs['loss']

        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description='ProteinMTP 训练脚本 - 坐标回归版本')

    # 数据参数
    parser.add_argument('--train_data', type=str, default='coord_train.json')
    parser.add_argument('--val_data', type=str, default='coord_val.json')

    # 模型参数
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--max_seq_len', type=int, default=512)

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1)  # 目前只支持1
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='./output_structure')

    # 大模型优化参数
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='启用梯度检查点以节省显存（推荐7B模型使用）')
    parser.add_argument('--use_4bit', action='store_true',
                        help='使用4bit量化加载模型（极端显存优化）')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='梯度累积步数（大模型建议增加）')

    args = parser.parse_args()

    print("=" * 60)
    print("ProteinMTP 训练 - 坐标回归版本")
    print("=" * 60)
    print(f"模型: {args.model_name}")
    print(f"最大序列长度: {args.max_seq_len}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"梯度累积步数: {args.gradient_accumulation_steps}")
    print(f"梯度检查点: {'启用' if args.gradient_checkpointing else '禁用'}")
    print(f"4bit量化: {'启用' if args.use_4bit else '禁用'}")
    print()

    # 检测设备
    device, gpu_memory = get_device_info()

    # 根据模型大小自动调整配置
    model_size = args.model_name.lower()
    if '7b' in model_size or '8b' in model_size:
        print("\n检测到大模型(7B+)，自动启用优化...")
        if not args.gradient_checkpointing:
            args.gradient_checkpointing = True
            print("  - 自动启用梯度检查点")
        if args.gradient_accumulation_steps < 8:
            args.gradient_accumulation_steps = 8
            print(f"  - 自动增加梯度累积步数到 {args.gradient_accumulation_steps}")
        # 7B模型建议使用更小的LoRA rank以节省显存
        if args.lora_rank > 8 and gpu_memory < 24:
            args.lora_rank = 8
            args.lora_alpha = 16
            print(f"  - 自动调整LoRA rank到 {args.lora_rank}")

    # 加载 tokenizer 和模型
    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 配置模型加载参数
    model_kwargs = {
        'trust_remote_code': True,
    }

    # 设置数据类型
    if torch.cuda.is_available():
        # GPU: 使用bf16或fp16
        if torch.cuda.is_bf16_supported():
            model_kwargs['torch_dtype'] = torch.bfloat16
            print("使用 BFloat16 精度")
        else:
            model_kwargs['torch_dtype'] = torch.float16
            print("使用 Float16 精度")
    else:
        # CPU: 使用float32
        model_kwargs['torch_dtype'] = torch.float32
        print("使用 Float32 精度 (CPU)")

    # 4bit量化配置（极端显存优化）
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

    # 启用梯度检查点（大模型必需）
    if args.gradient_checkpointing:
        print("启用梯度检查点...")
        base_model.gradient_checkpointing_enable()
        # 需要设置use_cache=False
        base_model.config.use_cache = False

    # 应用 LoRA（参考师兄代码）
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
    model = ProteinStructureMTP(base_model, max_seq_len=args.max_seq_len)

    # 确保自定义层与base_model使用相同的dtype
    model_dtype = next(base_model.parameters()).dtype
    model.coord_proj = model.coord_proj.to(dtype=model_dtype)
    model.coord_head = model.coord_head.to(dtype=model_dtype)

    # 加载数据集
    print("\n加载数据集...")
    train_dataset = ProteinCoordDataset(args.train_data, tokenizer, args.max_seq_len)
    val_dataset = ProteinCoordDataset(args.val_data, tokenizer, args.max_seq_len) if os.path.exists(args.val_data) else None

    # 数据整理器
    collator = ProteinCoordCollator(tokenizer, args.max_seq_len)

    # 训练参数
    # 自动检测是否支持 bf16，否则使用 fp16
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
        save_safetensors=False,  # 避免共享权重导致的保存错误
        gradient_checkpointing=args.gradient_checkpointing,
        # 优化器配置
        optim="adamw_torch" if not args.use_4bit else "paged_adamw_8bit",
        # 显存优化
        dataloader_pin_memory=True if torch.cuda.is_available() else False,
    )

    # 创建训练器
    trainer = CoordMTPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # 开始训练
    print("\n开始训练...")
    trainer.train()

    # 保存模型（手动保存以避免共享权重问题）
    print(f"\n保存模型到 {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存模型状态（使用torch.save）
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt'))

    # 保存tokenizer
    tokenizer.save_pretrained(args.output_dir)

    # 计算全局归一化统计量（用于推理时的反归一化）
    print("\n计算全局归一化统计量...")
    global_means = []
    global_stds = []
    for item in train_dataset.data:
        if 'norm_stats' in item:
            global_means.append(item['norm_stats']['mean'])
            global_stds.append(item['norm_stats']['std'])

    import numpy as np
    global_mean_centroid = np.mean(global_means, axis=0).tolist() if global_means else [0.0, 0.0, 0.0]
    global_mean_std = float(np.mean(global_stds)) if global_stds else 1.0

    print(f"全局平均质心: [{global_mean_centroid[0]:.2f}, {global_mean_centroid[1]:.2f}, {global_mean_centroid[2]:.2f}] Å")
    print(f"全局平均标准差: {global_mean_std:.2f} Å")

    # 保存训练配置
    config_to_save = {
        'model_name': args.model_name,
        'max_seq_len': args.max_seq_len,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'global_norm_mean': global_mean_centroid,
        'global_norm_std': global_mean_std,
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
