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
- PyTorch
- Transformers (Hugging Face)
- PEFT (LoRA微调)
- DeepSpeed (可选)

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
├── train_protein_structure.py    # ✅ 训练脚本：坐标回归
├── infer_protein_structure.py    # ✅ 推理脚本：结构预测
├── evaluate_structure.py          # ✅ 评估脚本：RMSD/TM-score等
├── visualize_structure.py         # ✅ 可视化脚本：3D/2D结构图
│
├── train_protein_mtp.py           # ⚠️ 旧版本(标准MTP),不推荐使用
└── ...
```

**核心文件说明**：

| 文件 | 作用 | 状态 |
|-----|------|------|
| `SFT_main.py` | 师兄的参考代码(分类任务) | ✅ 参考 |
| `data_preprocessing.py` | 从PDB提取序列和坐标 | ✅ 完成 |
| `create_coord_data.py` | 创建训练数据集(归一化、划分) | ✅ 完成 |
| `train_protein_structure.py` | 训练坐标回归模型 | ✅ 推荐 |
| `infer_protein_structure.py` | 预测蛋白质结构(生成PDB) | ✅ 新增 |
| `evaluate_structure.py` | 评估预测结果(RMSD/TM-score) | ✅ 新增 |
| `visualize_structure.py` | 可视化结构(3D/2D投影) | ✅ 新增 |

**关键组件**：
- `LabelWiseAttention`: 将placeholder tokens映射到输出位置
- `ProteinStructureMTP`: MTP模型,输出(B, seq_len, 3)坐标
- RMSD评估指标: 蛋白质结构预测的标准指标

## 快速开始

### 环境配置

```bash
git clone git@github.com:buzhimingLF/ProteinMTP---Multi-Token-Prediction-for-Protein-Sequence-Generation.git
cd ProteinMTP

# 安装依赖
pip install -r requirements.txt

# 或手动安装核心依赖
pip install torch transformers peft accelerate numpy matplotlib
```

### 数据准备

```bash
# 1. 从 PDBbind-Plus 下载数据集到 P-L/ 目录
https://www.pdbbind-plus.org.cn/download
Demo: PDBbind v2020.R1 ->Index files 和 Protein-ligand complex structures
# 2. 提取序列和坐标
python3 data_preprocessing.py --extract_coords --max_samples 1000

# 3. 创建训练数据集
python3 create_coord_data.py --max_seq_len 512
```

### 训练模型

```bash
# 使用Qwen2.5-0.5B训练
python3 train_protein_structure.py \
    --train_data coord_train.json \
    --val_data coord_val.json \
    --model_name Qwen/Qwen2.5-0.5B \
    --max_seq_len 512 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --output_dir ./output_structure
```

### 推理预测

```bash
# 预测单个序列的结构（自动反归一化）
python3 infer_protein_structure.py \
    --model_path ./output_structure \
    --sequence "MKTAYIAKQRQISFVKTIGDEVQREAPGDSRLAGHFELSC" \
    --output predicted_structure.pdb

# 注：模型会自动使用训练时的全局归一化统计量进行反归一化
# 如需自定义归一化参数，可使用：
# python3 infer_protein_structure.py \
#     --model_path ./output_structure \
#     --sequence "..." \
#     --norm_mean 16.75 19.87 24.65 \
#     --norm_std 10.53 \
#     --output predicted_structure.pdb
```

### 评估与可视化

```bash
# 评估预测结果(与真实结构对比)
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

## 完整工作流程

### 1. 数据准备流程
```bash
# Step 1: 提取坐标数据(需要先下载PDBbind数据集)
python data_preprocessing.py --extract_coords --max_samples 1000

# Step 2: 创建训练数据集
python create_coord_data.py --max_seq_len 512 --train_ratio 0.9
```

### 2. 模型训练流程
```bash
# 训练模型
python train_protein_structure.py \
    --train_data coord_train.json \
    --val_data coord_val.json \
    --model_name Qwen/Qwen2.5-0.5B \
    --max_seq_len 512 \
    --num_epochs 3 \
    --output_dir ./output_structure
```

### 3. 推理与评估流程
```bash
# Step 1: 预测结构
python infer_protein_structure.py \
    --model_path ./output_structure \
    --sequence "MKTAYIAKQRQISFVKTIGDEVQREAPGDSRLAGHFELSC" \
    --output predicted.pdb

# Step 2: 可视化
python visualize_structure.py \
    --pdb predicted.pdb \
    --output structure.png \
    --mode both \
    --stats

# Step 3: 评估(如果有真实结构)
python evaluate_structure.py \
    --pred predicted.pdb \
    --true true_structure.pdb \
    --all_metrics
```

## 实验进度

- [x] 确定研究方向
- [x] 获取参考资料和代码
- [x] 理解MTP核心逻辑
- [x] 代码改造（分类 -> 坐标回归）
- [x] 数据预处理（提取原子坐标）
- [x] 推理和评估脚本
- [ ] 基线实验
- [ ] 正式实验
- [ ] 论文撰写

## 文档

- [需求文档](./start.md)
- [讨论文档](./DISCUSSION.md)

## License

MIT License

---

*本项目为毕业论文研究项目*
