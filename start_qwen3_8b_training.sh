#!/bin/bash
# 启动Qwen3-8B距离矩阵模型训练

echo "=== Qwen3-8B 距离矩阵模型训练 ==="
echo "⚠️  使用 Qwen3-8B (NOT Qwen2.5-8B)"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

source venv_distmat/bin/activate

python scripts/training/train_distmat.py \
    --train_data distmat_train.json \
    --val_data distmat_val.json \
    --model_name Qwen/Qwen3-8B \
    --max_seq_len 256 \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 16 \
    --output_dir ./output_distmat_qwen3_8b

echo ""
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
