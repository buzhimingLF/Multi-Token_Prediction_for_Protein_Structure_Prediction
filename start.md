# 毕业论文项目需求文档

## 项目概述

**项目名称**：
- 中文名：基于多标记预测的蛋白质结构预测研究
- 英文名：ProteinMTP - Multi-Token Prediction for Protein Structure Prediction

**研究方向**：将 Multi-Token Prediction (MTP) 技术应用于蛋白质结构预测任务（预测氨基酸/原子的三维坐标）

---

## 研究背景与动机

### 原始想法
最初计划做自回归优化，匹配蛋白质序列的k值，一次预测一个蛋白质。

### 师兄建议
经过讨论，师兄建议直接往 Multi-Token Prediction 方向做：
- 两边的工程量其实差不多
- MTP 在蛋白质结构预测上有更好的研究价值和创新点

### 任务方向修正（2026-01-25）
明确了任务的核心方向：
- **目标是蛋白质结构预测（坐标预测），不是序列生成**
- MTP的作用是将LLM输出转为定长格式，k值与序列长度一致
- 每个输出位置唯一对应序列上的某个氨基酸
- 参考师兄代码中MTP用于分类的思路，迁移到坐标回归

### 为什么不直接用LLM生成坐标字符串？
虽然可以将坐标预测转化为字符串生成任务（直接生成"x:1.23,y:4.56,z:7.89"这样的文本），但这种方式存在致命问题：
- **梯度不平滑**：字符串生成是离散的token预测，每个数字字符的预测梯度无法直接反映坐标的连续性
- **任务上限低**：模型需要学习"数字→坐标"的映射关系，而不是直接优化坐标精度
- **损失函数难以设计**：无法直接使用MSE等回归损失，只能用交叉熵损失优化token预测

通过MTP实现定长输出，可以：
- 每个氨基酸对应一个固定的输出位置
- 直接输出连续的坐标值(x,y,z)
- 使用MSE等回归损失直接优化坐标精度
- 梯度更平滑，训练更稳定

---

## 核心参考资料

### 参考论文
- **论文链接**：https://arxiv.org/abs/2504.03159
- **论文主题**：Multi-Token Prediction 相关研究
- **需要重点关注**：
  - [x] MTP 的核心思想和实现方法
  - [ ] 与传统自回归方法的对比
  - [ ] 在序列生成任务上的应用

### 数据集
- **数据集名称**：PDBbind-Plus
- **数据集链接**：https://www.pdbbind-plus.org.cn/
- **数据集特点**：
  - 蛋白质-配体结合数据
  - 包含蛋白质序列信息
  - 包含原子三维坐标信息（ATOM记录）

### 参考代码
- **来源**：师兄提供的基于 Qwen 实现的 MTP 代码
- **原始用途**：多标签分类任务（带视频的VL模型）
- **改造方向**：改成坐标回归任务，适配蛋白质结构预测

---

## MTP核心原理（基于师兄代码）

### 1. MTP的作用：限定LLM输出为定长
师兄代码中，MTP通过placeholder tokens实现定长输出：
```python
# 在输入序列末尾添加固定数量的placeholder tokens
added_pl_tokens = [processor.tokenizer.get_vocab()['<unk>']] * num_of_pl_tokens
inputs['input_ids'] = torch.cat([inputs['input_ids'][0], torch.tensor(added_pl_tokens)])
```

**关键点**：
- `num_of_pl_tokens` 决定输出长度
- 在师兄的分类任务中，`num_of_pl_tokens` = 类别数量
- **在我们的任务中，`num_of_pl_tokens` = 蛋白质序列长度**

### 2. 从hidden states到预测结果
```python
# 获取LLM的hidden states
tmp = self.model.model.model(**inputs).last_hidden_state

# 分离原始输入部分和placeholder部分
lm_part = tmp[..., :-num_of_pl_tokens, :]  # 输入序列的hidden states
pl_part = tmp[..., -num_of_pl_tokens:, :]   # placeholder的hidden states

# 通过LabelWiseAttention映射到固定数量的输出
# 在分类任务中：每个输出对应一个类别
# 在结构预测中：每个输出对应一个氨基酸
lm_part_pooled = lm_part.max(dim=1, keepdims=True)[0]  # 池化输入特征
output_features = self.proj(torch.cat([lm_part_pooled, pl_part], dim=1))  # (B, num_of_pl_tokens, D)
```

### 3. 迁移到蛋白质结构预测的关键修改

| 维度 | 师兄的分类任务 | 我们的结构预测任务 |
|------|--------------|-----------------|
| 输入 | 视频+文本描述 | 蛋白质序列（氨基酸字符串） |
| `num_of_pl_tokens` | 类别数量（固定） | 序列长度（变长，需padding） |
| 输出维度 | (B, 类别数, 1) → logits | (B, 序列长度, 3) → (x,y,z)坐标 |
| 损失函数 | MultiLabelSoftMarginLoss | MSE Loss（回归损失） |
| 输出含义 | 每个位置表示一个类别的概率 | 每个位置表示一个氨基酸的坐标 |

**核心思路一致**：都是通过MTP将LLM的输出转换为定长格式，只是输出的语义不同。

---

## 技术栈与环境

### 硬件环境
| 配置项 | 规格 |
|--------|------|
| GPU | NVIDIA RTX 3090 |
| 显存 | 24GB |

### 软件依赖
- PyTorch
- Transformers (Hugging Face)
- PEFT (LoRA 微调)
- DeepSpeed (可选)

### 基础模型选择
- 优先使用最小模型跑通，再逐步加大
- 候选：Qwen-0.5B, Qwen-1.8B, ESM-2

---

## 待完成任务清单

### 第一阶段：环境准备与代码理解
- [x] 阅读并理解参考论文
- [x] 下载并探索 PDBbind-Plus 数据集
- [x] 理解师兄的 MTP 代码实现
- [x] 明确任务方向：坐标预测而非序列生成

### 第二阶段：代码改造
- [ ] 数据预处理：从PDB文件提取原子坐标
- [ ] 模型改造：分类任务 -> 坐标回归任务
- [ ] 损失函数：分类损失 -> MSE回归损失
- [ ] 评估指标：RMSD, GDT-TS, TM-score

### 第三阶段：实验与验证
- [ ] 使用最小模型跑通整个流程
- [ ] 验证 MTP 在蛋白质结构预测上的效果
- [ ] 逐步扩大模型规模和数据量

### 第四阶段：论文撰写
- [ ] 整理实验结果
- [ ] 撰写毕业论文

---

## 代码改造要点

### 1. 数据处理改造
**目标**：提取PDB文件中的氨基酸序列和对应的原子坐标

```python
# 从PDB文件提取序列和Cα原子坐标
def parse_pdb_sequence_and_coords(pdb_file):
    """
    返回：
    - sequence: 氨基酸序列字符串，如 "MKTAYIAKQRQISFVK..."
    - coords: Cα原子坐标列表，shape=(seq_len, 3)，每行是(x,y,z)
    """
    sequence = []
    coords = []
    for line in open(pdb_file):
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':  # Cα原子
            # 提取氨基酸残基名
            residue = line[17:20].strip()
            sequence.append(AA_3TO1[residue])

            # 提取坐标
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])

    return ''.join(sequence), np.array(coords)
```

### 2. 模型改造（参考师兄代码的HuggingfaceModelWithMTP）

**核心修改**：
1. `num_of_pl_tokens` = 序列长度（需要动态设置，或使用max_seq_len+padding）
2. 输出从分类logits改为坐标回归
3. 使用LabelWiseAttention将placeholder的hidden states映射到每个氨基酸

```python
class ProteinStructureMTP(nn.Module):
    def __init__(self, base_model, max_seq_len):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size

        # 参考师兄代码的LabelWiseAttention
        self.coord_proj = LabelWiseAttention(hidden_size, max_seq_len)

        # 坐标预测头：将每个氨基酸的特征映射到3D坐标
        self.coord_head = nn.Linear(hidden_size, 3)  # 输出(x,y,z)

    def forward(self, inputs, num_of_pl_tokens):
        # 获取hidden states
        hidden = self.base_model(**inputs, output_hidden_states=True).last_hidden_state

        # 分离输入部分和placeholder部分（参考师兄代码）
        lm_part = hidden[..., :-num_of_pl_tokens, :]
        pl_part = hidden[..., -num_of_pl_tokens:, :]

        # 池化输入特征
        lm_part_pooled = lm_part.max(dim=1, keepdims=True)[0]

        # 通过LabelWiseAttention映射到每个氨基酸
        coord_features = self.coord_proj(torch.cat([lm_part_pooled, pl_part], dim=1))

        # 预测坐标
        coords = self.coord_head(coord_features)  # (B, seq_len, 3)
        return coords
```

### 3. 损失函数改造

**从分类损失改为回归损失**：

```python
# 师兄代码：MultiLabelSoftMarginLoss（分类）
loss = nn.MultiLabelSoftMarginLoss()(logits, labels)

# 我们的代码：MSE Loss（回归）
loss = F.mse_loss(pred_coords, true_coords)

# 可选：添加mask处理padding部分
mask = (labels != -100)  # padding位置的label设为-100
loss = F.mse_loss(pred_coords[mask], true_coords[mask])
```

### 4. 数据准备脚本（create_coord_data.py）

需要生成训练数据格式：
```json
[
  {
    "pdb_code": "1a00",
    "sequence": "MKTAYIAKQRQISFVK...",
    "coords": [[x1,y1,z1], [x2,y2,z2], ...],
    "seq_len": 128
  },
  ...
]
```

---

## 待确认问题

1. **坐标预测的具体形式**
   - 预测每个氨基酸的Cα原子坐标？还是所有骨架原子？
   - 坐标是绝对坐标还是相对坐标？

2. **输入设计**
   - 输入是蛋白质序列（氨基酸字符串）？
   - 是否需要加入其他信息（如配体信息）？

3. **评估指标**
   - RMSD（均方根偏差）
   - GDT-TS
   - TM-score

---

## 时间规划

| 阶段 | 时间 | 任务 |
|------|------|------|
| Week 1-2 | 论文阅读 | 深入理解 MTP 论文和相关工作 |
| Week 3-4 | 数据准备 | 下载、探索、预处理数据集（提取坐标） |
| Week 5-6 | 代码改造 | 完成代码从分类到坐标回归的改造 |
| Week 7-8 | 初步实验 | 小模型跑通，验证可行性 |
| Week 9-12 | 正式实验 | 扩大规模，完善实验 |
| Week 13-16 | 论文撰写 | 整理结果，撰写论文 |

---

## 沟通记录

### 2026-01-24 与师兄讨论
- 确定方向为 Multi-Token Prediction
- 师兄提供了论文链接、数据集链接、参考代码

### 2026-01-25 任务方向修正
- 明确目标是蛋白质结构预测（坐标预测），不是序列生成
- MTP的作用是将LLM输出转为定长格式
- 参考师兄代码中MTP用于分类的思路，迁移到坐标回归

---

## 数据集说明

### 确定的下载方案：PDBbind v.2020.R1

**下载文件**：
| 序号 | 文件 | 大小 | 说明 |
|------|------|------|------|
| 1 | Index files | 497KB | 索引文件 |
| 2 | Protein-ligand complex structures | 3.1GB | 19,037个复合物结构 |

**选择理由**：
1. 免费 - 注册用户即可下载
2. 质量较高 - 使用v.2024的新工作流程重新处理
3. 数据量适中 - 19,037个复合物足够训练

**PDB文件中的关键信息**：
- SEQRES记录：氨基酸序列
- ATOM记录：原子坐标（这是我们需要的）

---

*文档最后更新：2026-01-25*
