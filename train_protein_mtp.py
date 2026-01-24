"""
ProteinMTP 训练脚本
基于 Multi-Token Prediction 的蛋白质序列生成模型

改造自师兄的 VL 模型代码，适配纯文本蛋白质序列
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
    TrainingArguments
)
from torch.utils.data import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm

warnings.filterwarnings('ignore')


class ProteinSequenceDataset(Dataset):
    """蛋白质序列数据集"""
    
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
        sequence = item['input']
        return {
            'sequence': sequence,
            'pdb_code': item.get('pdb_code', ''),
        }


class ProteinMTPCollator:
    """MTP 数据整理器"""
    
    def __init__(self, tokenizer, max_length: int = 512, n_future_tokens: int = 4):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_future_tokens = n_future_tokens
        
    def __call__(self, batch):
        sequences = [item['sequence'] for item in batch]
        
        # Tokenize 序列
        encodings = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 创建 MTP 标签：每个位置预测未来 n 个 token
        input_ids = encodings['input_ids']
        batch_size, seq_len = input_ids.shape
        
        # 标签：对于每个位置 i，标签是 [i+1, i+2, ..., i+n]
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': encodings['attention_mask'],
            'labels': labels,
        }


class MultiTokenPredictionHead(nn.Module):
    """Multi-Token Prediction 头"""
    
    def __init__(self, hidden_size: int, vocab_size: int, n_future_tokens: int = 4):
        super().__init__()
        self.n_future_tokens = n_future_tokens
        
        # 每个未来 token 一个独立的预测头
        self.prediction_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size)
            for _ in range(n_future_tokens)
        ])
        
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        Returns:
            logits: list of (batch_size, seq_len, vocab_size) for each future token
        """
        return [head(hidden_states) for head in self.prediction_heads]


class ProteinMTPModel(nn.Module):
    """蛋白质 MTP 模型"""
    
    def __init__(self, base_model, n_future_tokens: int = 4):
        super().__init__()
        self.base_model = base_model
        self.n_future_tokens = n_future_tokens
        
        # 获取隐藏层大小和词表大小
        config = base_model.config
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        
        # MTP 预测头
        self.mtp_heads = MultiTokenPredictionHead(
            hidden_size, vocab_size, n_future_tokens
        )
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # 获取基础模型的隐藏状态
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        hidden_states = outputs.hidden_states[-1]  # 最后一层隐藏状态
        
        # MTP 预测
        mtp_logits = self.mtp_heads(hidden_states)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss = self._compute_mtp_loss(mtp_logits, labels)
        
        return {
            'loss': loss,
            'logits': outputs.logits,  # 标准 next-token 预测
            'mtp_logits': mtp_logits,  # MTP 预测
        }
    
    def _compute_mtp_loss(self, mtp_logits, labels):
        """计算 MTP 损失"""
        total_loss = 0
        batch_size, seq_len = labels.shape
        
        for i, logits in enumerate(mtp_logits):
            # 对于第 i 个预测头，预测位置 j 的 token 应该是 labels[j+i+1]
            shift = i + 1
            if shift >= seq_len:
                continue
                
            # 移位对齐
            shifted_logits = logits[:, :-shift, :].contiguous()
            shifted_labels = labels[:, shift:].contiguous()
            
            # 计算交叉熵损失
            loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
                ignore_index=self.base_model.config.pad_token_id or -100
            )
            total_loss += loss
        
        return total_loss / len(mtp_logits)


class MTPTrainer(Trainer):
    """MTP 训练器"""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description='ProteinMTP 训练脚本')
    
    # 数据参数
    parser.add_argument('--train_data', type=str, default='mtp_data_train.json')
    parser.add_argument('--val_data', type=str, default='mtp_data_val.json')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--n_future_tokens', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=512)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='./output')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ProteinMTP 训练")
    print("=" * 60)
    print(f"模型: {args.model_name}")
    print(f"MTP 预测头数量: {args.n_future_tokens}")
    print(f"最大序列长度: {args.max_length}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print()
    
    # 加载 tokenizer 和模型
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
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
    model = ProteinMTPModel(base_model, n_future_tokens=args.n_future_tokens)
    
    # 加载数据集
    print("\n加载数据集...")
    train_dataset = ProteinSequenceDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = ProteinSequenceDataset(args.val_data, tokenizer, args.max_length) if args.val_data else None
    
    # 数据整理器
    collator = ProteinMTPCollator(tokenizer, args.max_length, args.n_future_tokens)
    
    # 训练参数
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
        bf16=True,
        gradient_accumulation_steps=4,
        remove_unused_columns=False,
        report_to="none",
    )
    
    # 创建训练器
    trainer = MTPTrainer(
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
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("\n训练完成！")


if __name__ == '__main__':
    main()
