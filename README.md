# ProteinMTP

**基于多标记预测的蛋白质结构预测研究**

*Multi-Token Prediction for Protein Structure Prediction*

---

## 项目简介

本项目是一个毕业论文研究项目，旨在将 Multi-Token Prediction (MTP) 技术应用于**蛋白质结构预测**任务。

**核心思路**：利用MTP将大语言模型(LLM)的输出转换为定长格式，使得输出长度k与蛋白质序列长度一致，每个输出位置唯一对应序列上的某个氨基酸，从而实现氨基酸/原子三维坐标的回归预测。

## 研究目标

- 探索MTP在蛋白质结构预测任务（坐标预测）上的应用
- 通过MTP实现LLM的定长输出，将输出长度限制为k=序列长度
- 验证MTP方法相比直接生成坐标字符串的优势（梯度平滑性、任务上限）

## 为什么用MTP？

### MTP的作用
MTP通过placeholder tokens实现定长输出：
- **限定输出长度**：k = 蛋白质序列长度
- **位置对齐**：每个placeholder token对应一个氨基酸
- **支持回归**：placeholder的hidden states可以直接映射到连续的坐标值(x,y,z)

### 为什么不直接用LLM生成坐标字符串？

传统方法可以将坐标预测转化为字符串生成任务（如生成"1.23,4.56,7.89"），让LLM直接生成坐标文本。但这种方式存在致命问题：

| 问题 | 说明 |
|-----|------|
| **梯度不平滑** | 字符串生成是离散的token预测，每个数字字符的梯度无法直接反映坐标的连续性 |
| **任务上限低** | 模型学的是"数字字符→坐标"的间接映射，而不是直接优化坐标精度 |
| **损失函数局限** | 只能用交叉熵损失优化token预测，无法用MSE等回归损失 |

### MTP的优势

| 优势 | 说明 |
|-----|------|
| **直接输出坐标** | 每个位置直接输出(x,y,z)连续值 |
| **回归损失** | 可以用MSE Loss直接优化坐标精度 |
| **梯度平滑** | 坐标是连续的，梯度可以顺畅地反向传播 |
| **定长输出** | 输出长度固定，便于batch训练 |

## 参考资料

- **核心论文**: [arxiv:2504.03159](https://arxiv.org/abs/2504.03159)
- **数据集**: [PDBbind-Plus](https://www.pdbbind-plus.org.cn/)

## 技术栈

- Python 3.8+
- PyTorch (CUDA 11.8+)
- Transformers (Hugging Face)
- PEFT (LoRA微调)
- bitsandbytes (4bit量化，可选)

## 项目结构

```
ProteinMTP/
├── README.md                      # 项目说明
├── DISCUSSION.md                  # 进度与讨论
├── start.md                       # 需求文档
│
├── SFT_main.py                    # 参考代码：师兄的MTP分类任务实现
├── SFT_infer.py                   # 参考代码：推理脚本
│
├── data_preprocessing.py          # 数据预处理：提取序列+坐标
├── create_coord_data.py           # 数据准备：归一化、划分数据集
│
├── train_protein_structure.py    # 训练脚本：坐标回归
├── infer_protein_structure.py    # 推理脚本：结构预测
├── evaluate_structure.py          # 评估脚本：RMSD/TM-score等
├── visualize_structure.py         # 可视化脚本：3D/2D结构图
│
├── output_structure/              # 训练输出目录
│   ├── model.pt                   # 模型权重
│   ├── training_config.json       # 训练配置
│   └── ...
└── ...
```

## 快速开始

### 环境配置

```bash
git clone git@github.com:buzhimingLF/ProteinMTP---Multi-Token-Prediction-for-Protein-Sequence-Generation.git
cd ProteinMTP

# 安装依赖
pip install -r requirements.txt

# GPU用户：确保安装CUDA兼容的PyTorch
# 查看CUDA版本: nvidia-smi
# 安装对应版本的PyTorch: https://pytorch.org/get-started/locally/
```

### 数据准备

```bash
# 1. 从 PDBbind-Plus 下载数据集到 P-L/ 目录
# https://www.pdbbind-plus.org.cn/download
# Demo: PDBbind v2020.R1 -> Index files 和 Protein-ligand complex structures

# 2. 提取序列和坐标
python3 data_preprocessing.py --extract_coords --max_samples 5000

# 3. 创建训练数据集
python3 create_coord_data.py --max_seq_len 512
```

---

## 正式训练指南

### 模型选择

| 模型 | 参数量 | 显存需求 | 推荐场景 |
|-----|--------|---------|---------|
| Qwen2.5-0.5B | 0.5B | ~4GB | 快速验证、CPU可用 |
| Qwen3-4B | 4B | ~10GB | 中等规模实验 |
| **Qwen3-8B** | 8B | ~18-22GB | **正式训练（推荐）** |

### 训练命令

#### 小模型快速验证 (0.5B)
```bash
python3 train_protein_structure.py \
    --train_data coord_train.json \
    --val_data coord_val.json \
    --model_name Qwen/Qwen2.5-0.5B \
    --max_seq_len 512 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --output_dir ./output_structure_0.5b
```

#### 正式训练 (8B) - 推荐
```bash
# RTX 3090 (24GB) 推荐配置
python3 train_protein_structure.py \
    --train_data coord_train.json \
    --val_data coord_val.json \
    --model_name Qwen/Qwen3-8B \
    --max_seq_len 512 \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 8 \
    --output_dir ./output_structure_8b
```

#### 极端显存优化 (4bit量化)
```bash
# 显存不足时使用4bit量化
python3 train_protein_structure.py \
    --train_data coord_train.json \
    --val_data coord_val.json \
    --model_name Qwen/Qwen3-8B \
    --max_seq_len 512 \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --gradient_checkpointing \
    --use_4bit \
    --gradient_accumulation_steps 16 \
    --output_dir ./output_structure_8b_4bit
```

### 训练参数说明

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--model_name` | 基础模型名称 | Qwen/Qwen2.5-0.5B |
| `--max_seq_len` | 最大序列长度 | 512 |
| `--num_epochs` | 训练轮数 | 3 |
| `--learning_rate` | 学习率 | 1e-4 |
| `--lora_rank` | LoRA秩 | 16 |
| `--lora_alpha` | LoRA alpha | 32 |
| `--gradient_checkpointing` | 启用梯度检查点 | False |
| `--use_4bit` | 使用4bit量化 | False |
| `--gradient_accumulation_steps` | 梯度累积步数 | 4 |

---

## 推理预测

```bash
# 预测单个序列的结构（自动反归一化）
python3 infer_protein_structure.py \
    --model_path ./output_structure_8b \
    --sequence "MKTAYIAKQRQISFVKTIGDEVQREAPGDSRLAGHFELSC" \
    --output predicted_structure.pdb
```

---

## 评估与可视化

```bash
# 评估预测结果
python3 evaluate_structure.py \
    --pred predicted_structure.pdb \
    --true true_structure.pdb \
    --all_metrics

# 可视化结构
python3 visualize_structure.py \
    --pdb predicted_structure.pdb \
    --output structure_viz.png \
    --mode both \
    --stats
```

---

## 模型上传与分享

### 方法1: Git LFS (推荐小于2GB的模型)

```bash
# 1. 安装Git LFS
git lfs install

# 2. 追踪大文件
git lfs track "*.pt"
git lfs track "*.bin"
git lfs track "*.safetensors"

# 3. 添加并提交
git add .gitattributes
git add output_structure_8b/
git commit -m "添加8B模型权重"
git push origin main
```

### 方法2: Hugging Face Hub (推荐大模型)

```bash
# 1. 安装huggingface_hub
pip install huggingface-hub

# 2. 登录
huggingface-cli login

# 3. 上传模型
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='./output_structure_8b',
    repo_id='your-username/ProteinMTP-8B',
    repo_type='model'
)
"
```

### 方法3: 云盘分享

将 `output_structure_8b/` 文件夹打包上传到：
- 百度网盘
- 阿里云盘
- Google Drive

---

## Windows本地使用

### 1. 环境配置

```powershell
# 创建虚拟环境
python -m venv venv
.\venv\Scripts\activate

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft accelerate numpy matplotlib
```

### 2. 下载模型

**方法A: 从Git仓库克隆**
```powershell
git lfs install
git clone https://github.com/buzhimingLF/ProteinMTP---Multi-Token-Prediction-for-Protein-Sequence-Generation.git
cd ProteinMTP---Multi-Token-Prediction-for-Protein-Sequence-Generation
```

**方法B: 从Hugging Face下载**
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="your-username/ProteinMTP-8B",
    local_dir="./output_structure_8b"
)
```

**方法C: 从云盘下载**
下载后解压到项目的 `output_structure_8b/` 目录

### 3. 运行推理

```powershell
python infer_protein_structure.py ^
    --model_path ./output_structure_8b ^
    --sequence "MKTAYIAKQRQISFVKTIGDEVQREAPGDSRLAGHFELSC" ^
    --output predicted.pdb
```

### 4. 可视化结果

```powershell
python visualize_structure.py ^
    --pdb predicted.pdb ^
    --output structure.png ^
    --mode both ^
    --stats
```

---

## 评估指标参考

### RMSD (Root Mean Square Deviation)
- < 2.0 Å: 高质量预测
- 2.0-5.0 Å: 中等质量
- > 5.0 Å: 低质量

### TM-score
- > 0.5: 相同折叠
- 0.4-0.5: 相似折叠
- < 0.4: 不同折叠

### GDT-TS
- > 50: 高质量
- 30-50: 中等质量
- < 30: 低质量

---

## 实验进度

- [x] 确定研究方向
- [x] 获取参考资料和代码
- [x] 理解MTP核心逻辑
- [x] 代码改造（分类 -> 坐标回归）
- [x] 数据预处理（提取原子坐标）
- [x] 推理和评估脚本
- [x] Phase 1 验证实验 (0.5B)
- [x] Phase 2 正式训练 (Qwen3-8B) ✅ 完成
- [ ] **方法改进：距离矩阵预测**（进行中）
- [ ] 模型评估与优化
- [ ] 论文撰写

## ⚠️ 重要：方法改进说明

### 问题：绝对坐标回归的局限性

当前版本（v1）直接预测原子的绝对坐标 (x, y, z)，存在根本性问题：
- 绝对坐标受蛋白质整体**平移**和**旋转**影响
- 同一结构在不同位置/朝向会产生完全不同的坐标值
- 模型难以学习"结构本质"，而是在学习"空间位置"

### 解决方案：距离矩阵预测

改为预测**原子间距离矩阵**（N×N），距离是旋转/平移不变的：
- 蛋白质无论放在哪里、朝向如何，原子间距离不变
- 模型更容易学习到真正的结构信息
- 后续通过MDS算法从距离矩阵还原坐标

详见 [DISCUSSION.md](./DISCUSSION.md) 中的"2026-01-30 师兄反馈"记录。

## 常见问题

### Q: GPU显存不足怎么办？
A: 尝试以下方法：
1. 使用 `--gradient_checkpointing` 启用梯度检查点
2. 使用 `--use_4bit` 启用4bit量化
3. 增加 `--gradient_accumulation_steps` 减少每步显存占用
4. 减小 `--max_seq_len` 限制序列长度

### Q: CUDA版本不兼容怎么办？
A: 安装与驱动兼容的PyTorch版本：
```bash
# 查看CUDA版本
nvidia-smi

# 安装对应版本
# CUDA 11.8: pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1: pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### Q: 训练时出现NaN loss怎么办？
A: 可能的解决方案：
1. 降低学习率（如从1e-4降到5e-5）
2. 检查数据是否有异常值
3. 使用gradient clipping

### Q: 磁盘空间不足怎么办？
A: 8B模型需要约17GB空间下载，可以将HuggingFace缓存移到大磁盘：
```bash
# 1. 创建大磁盘上的缓存目录
mkdir -p /mnt/data/huggingface_cache

# 2. 移动现有缓存
mv ~/.cache/huggingface/* /mnt/data/huggingface_cache/

# 3. 创建软链接
rm -rf ~/.cache/huggingface
ln -s /mnt/data/huggingface_cache ~/.cache/huggingface

# 4. 验证
ls -la ~/.cache/huggingface
```

同样可以将训练输出目录指向大磁盘：
```bash
python3 train_protein_structure.py \
    --output_dir /mnt/data/protein_mtp_output/output_structure_8b \
    ...
```

---

## 文档

- [需求文档](./start.md)
- [讨论文档](./DISCUSSION.md)
- [项目总结](./PROJECT_SUMMARY.md)
- [Claude Code指南](./CLAUDE.md)

## License

MIT License

---

*本项目为毕业论文研究项目*
*最后更新: 2026-01-29*
