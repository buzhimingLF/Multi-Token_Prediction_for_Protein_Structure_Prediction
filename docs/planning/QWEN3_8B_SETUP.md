# Qwen3-8B 模型设置指南

## 当前状态

### 已完成 ✅
- [x] 项目结构整理
- [x] CLAUDE.md更新（明确使用Qwen3-8B）
- [x] 训练脚本准备

### 待完成 ❌
- [ ] 下载Qwen3-8B模型
- [ ] 运行8B模型训练

---

## 问题

### 网络问题
- 当前环境无法访问 huggingface.co
- 错误: `Network is unreachable`
- 无法自动下载模型

### 已有的模型
只有小模型:
- ✅ Qwen2.5-0.5B (已下载，可用于测试)
- ❌ Qwen3-8B (未下载)

---

## 解决方案

### 方案1: 手动下载模型

在有网络的环境中下载模型，然后传输到服务器：

```bash
# 方法1: 使用huggingface-cli下载
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-8B --local-dir ./qwen3_8b_model

# 方法2: 使用git clone
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-8B ./qwen3_8b_model

# 传输到服务器
scp -r ./qwen3_8b_model user@server:~/.cache/huggingface/hub/models--Qwen--Qwen3-8B
```

### 方案2: 使用镜像站

```bash
# 如果有国内镜像
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-8B
```

### 方案3: 使用实验室GPU服务器

如果实验室有GPU服务器且能访问外网：
1. 在实验室服务器下载模型
2. 传输到当前服务器
3. 设置 `HF_HUB_OFFLINE=1` 使用本地模型

---

## 模型下载后

### 1. 放置位置
模型应该放在:
```
~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/
```

### 2. 验证下载
```bash
ls -la ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/
# 应该包含:
# - config.json
# - pytorch_model-00001-of-00002.bin
# - tokenizer.json
# ...等文件
```

### 3. 运行训练
```bash
# 激活环境
source venv_distmat/bin/activate

# 离线模式（使用本地模型）
export HF_HUB_OFFLINE=1

# 启动训练
./start_qwen3_8b_training.sh
```

---

## Qwen3-8B 训练配置

### 推荐配置（RTX 4090 24GB）
```bash
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
```

### 内存优化
如果遇到OOM:
```bash
# 添加4bit量化
--use_4bit

# 或增加梯度累积
--gradient_accumulation_steps 32
```

### 预期资源使用
- 显存: ~20-22GB (使用梯度检查点)
- 内存: ~30GB
- 训练时间: ~2-4小时 (3 epochs)

---

## 模型大小

### Qwen3-8B
- 模型权重: ~16GB (FP16)
- LoRA适配器: ~100MB
- 总大小: ~17GB

### 与0.5B对比
| 模型 | 参数量 | 显存需求 | 预期质量 |
|------|--------|----------|----------|
| Qwen2.5-0.5B | 0.5B | ~2GB | 基准 |
| Qwen3-8B | 8B | ~20GB | 预期显著提升 |

---

## 后续步骤

1. **下载模型**: 使用方案1/2/3之一
2. **验证下载**: 检查模型文件完整性
3. **运行训练**: `./start_qwen3_8b_training.sh`
4. **监控进度**: 使用 `tail -f` 查看日志

---

## 参考链接

- 模型: https://huggingface.co/Qwen/Qwen3-8B
- 文档: https://huggingface.co/docs/huggingface_hub
