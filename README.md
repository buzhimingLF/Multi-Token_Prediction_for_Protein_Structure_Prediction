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
├── README.md                    # 项目说明
├── DISCUSSION.md                # 进度与讨论
├── start.md                     # 需求文档
├── SFT_main.py                  # 参考代码：师兄的MTP分类任务实现
├── SFT_infer.py                 # 参考代码：推理脚本
├── train_protein_mtp.py         # 蛋白质结构预测训练代码（需改造）
├── data_preprocessing.py        # 数据预处理：提取序列
├── create_coord_data.py         # 数据准备：提取序列+坐标（待完成）
└── ...
```

**文件说明**：
- `SFT_main.py`：师兄提供的参考代码，展示了如何用MTP做分类任务
  - 核心：通过placeholder tokens实现定长输出（k=类别数）
  - 关键组件：`LabelWiseAttention`、`HuggingfaceModelWithMTP`
- `train_protein_mtp.py`：需要改造的训练代码
  - 当前状态：实现了标准的MTP（预测未来k个token）
  - 改造目标：参考`SFT_main.py`，实现坐标回归（k=序列长度）

## 快速开始

### 环境配置

```bash
git clone git@github.com:buzhimingLF/ProteinMTP---Multi-Token-Prediction-for-Protein-Sequence-Generation.git
cd ProteinMTP

pip install torch transformers peft deepspeed
```

### 数据准备

1. 从 [PDBbind-Plus](https://www.pdbbind-plus.org.cn/) 下载数据集
2. 运行数据预处理脚本提取原子坐标

### 训练模型

```bash
# 待完善
python train_protein_structure.py --data_path <data_path> --model_name Qwen/Qwen2.5-0.5B
```

## 实验进度

- [x] 确定研究方向
- [x] 获取参考资料和代码
- [x] 理解MTP核心逻辑
- [ ] 代码改造（分类 -> 坐标回归）
- [ ] 数据预处理（提取原子坐标）
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
