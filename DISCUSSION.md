# ProteinMTP 项目进度与讨论

**日期**：2026-01-29
**作者**：黄海军
**GitHub**：git@github.com:buzhimingLF/ProteinMTP---Multi-Token-Prediction-for-Protein-Sequence-Generation.git

---

## 项目背景

本项目旨在将 Multi-Token Prediction (MTP) 技术应用于**蛋白质结构预测**任务，具体目标是**预测蛋白质中氨基酸/原子的三维坐标信息**。

**核心思路**：利用MTP将LLM的输出转换为定长格式，使得输出长度与蛋白质序列长度一致（k=序列长度），每个输出位置唯一对应序列上的某个氨基酸，从而实现坐标回归预测。

**MTP在本项目中的作用**：
1. **限定输出长度**：通过placeholder tokens，将LLM的输出限制为k=序列长度
2. **位置对齐**：每个placeholder token对应一个氨基酸的输出位置
3. **支持回归**：placeholder的hidden states可以直接映射到连续的坐标值

**为什么不直接用LLM生成坐标字符串？**
虽然可以将坐标预测转化为字符串生成任务（如生成"1.23,4.56,7.89"），但这种方式存在致命问题：
- **梯度不平滑**：字符串生成是离散token预测，无法直接优化坐标精度
- **任务上限低**：模型学的是"数字字符→坐标"的间接映射
- **损失函数局限**：只能用交叉熵，无法用MSE等回归损失

通过MTP实现定长输出，可以直接进行回归预测，梯度更平滑，训练更稳定。

## 目前进展

**已完成**
- [x] 确定研究方向：MTP用于蛋白质结构预测
- [x] 获取参考论文：[arxiv:2504.03159](https://arxiv.org/abs/2504.03159)
- [x] 确定数据集：[PDBbind-Plus](https://www.pdbbind-plus.org.cn/)
- [x] 获取师兄的参考代码（基于Qwen的MTP实现，用于分类任务）
- [x] 理解MTP核心逻辑：通过placeholder tokens实现定长输出
- [x] 代码改造（分类任务 -> 坐标回归任务）
- [x] 数据预处理流程设计（提取原子坐标）
- [x] Phase 1可行性验证：小规模训练测试通过
- [x] Phase 2正式训练：Qwen3-8B + GPU训练完成

**进行中**
- [ ] 模型方法改进（距离矩阵预测）
- [ ] 模型评估与优化

**待开始**
- [ ] Phase 3优化与扩展
- [ ] 与其他方法对比（AlphaFold/ESMFold）

### Phase 1 验证结果 (2026-01-28)

**数据**：
- 提取了1000个蛋白质样本
- 训练集657个，验证集73个（长度≤512）
- 坐标已归一化（零均值+标准化）

**训练**：
- 模型：Qwen/Qwen2.5-0.5B + LoRA
- 训练时间：~25分钟/epoch (CPU)
- Loss: 2.62 → 1.01（持续下降）

**推理**：
- ✅ 模型保存/加载正常
- ✅ 坐标自动反归一化（恢复到Å单位）
- ✅ PDB文件输出正常
- ✅ 可视化功能正常

### Phase 2 正式训练结果 (2026-01-29)

**训练配置**：
- 模型：Qwen3-8B（从 Qwen2.5-7B 升级）
- GPU：RTX 3090 (24GB)
- 数据：2876 训练样本，320 验证样本
- 训练轮数：3 epochs (1077 steps)
- 优化：梯度检查点 + LoRA (r=8, alpha=16) + BFloat16
- 训练时间：约 1 小时 28 分钟

**训练结果**：
- ✅ 训练完成，Loss 正常下降
- ✅ 模型保存成功 (16GB)
- ✅ 全局归一化统计已保存
- 保存路径：`/mnt/data/protein_mtp_output/output_structure_8b/`

**推理内存需求**：
- 8B 模型需要约 32GB 内存（CPU+GPU 总计）
- 当前服务器（31GB RAM + 24GB VRAM）勉强够用
- 建议使用 4bit 量化或更大内存机器进行推理

**代码更新**：
- ✅ 升级 transformers 到 4.51.0（支持 Qwen3）
- ✅ 升级 PyTorch 到 2.2.0（支持新 transformers）
- ✅ 修复 ProteinStructureMTP 类的 gradient_checkpointing 方法
- ✅ 更新 CLAUDE.md 和 README.md 文档

---

### Phase 2 正式训练计划 (2026-01-28)

**目标**：使用7B大模型进行正式训练

**配置**：
- 模型：Qwen/Qwen2.5-7B
- GPU：RTX 3090 (24GB)
- 数据：5000个蛋白质样本
- 训练轮数：3 epochs
- 优化：梯度检查点 + LoRA + 混合精度

**训练命令**：
```bash
python3 train_protein_structure.py \
    --train_data coord_train.json \
    --val_data coord_val.json \
    --model_name Qwen/Qwen2.5-7B \
    --max_seq_len 512 \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 8 \
    --output_dir ./output_structure_7b
```

**代码更新**：
- ✅ 添加 `--gradient_checkpointing` 参数
- ✅ 添加 `--use_4bit` 4bit量化支持
- ✅ 添加 `--gradient_accumulation_steps` 参数
- ✅ 自动检测GPU并选择最佳dtype
- ✅ 大模型自动优化配置

---

## MTP核心原理（基于师兄代码分析）

### 1. 定长输出的实现方式

师兄代码中MTP的核心思路：**通过placeholder tokens限定输出长度**

```python
# 在输入序列末尾添加固定数量的placeholder tokens
added_pl_tokens = [processor.tokenizer.get_vocab()['<unk>']] * num_of_pl_tokens
inputs['input_ids'] = torch.cat([inputs['input_ids'][0], torch.tensor(added_pl_tokens)])
```

**这样做的效果**：
- LLM处理完原始输入后，会继续处理这些placeholder tokens
- 这些placeholder tokens的hidden states包含了"输出位置"的信息
- **输出长度固定为`num_of_pl_tokens`**
- 在师兄的分类任务中：`num_of_pl_tokens` = 类别数量
- **在我们的任务中：`num_of_pl_tokens` = 蛋白质序列长度**

### 2. 从hidden states到预测结果

```python
class HuggingfaceModelWithMTP(nn.Module):
    def forward(self, inputs):
        # 获取LLM的hidden states
        tmp = self.model.model.model(**inputs).last_hidden_state

        # 分离原始输入部分和placeholder部分
        lm_part = tmp[..., :-num_of_pl_tokens, :]    # 原始输入的hidden states
        pl_part = tmp[..., -num_of_pl_tokens:, :]     # placeholder的hidden states

        # 池化输入特征
        lm_part_pooled = lm_part.max(dim=1, keepdims=True)[0]

        # 通过LabelWiseAttention将hidden states映射到固定数量的输出
        # 这里的关键是：pl_part包含了num_of_pl_tokens个位置的特征
        lab_part = self.label_proj(torch.cat([lm_part_pooled, pl_part], dim=1))
        # lab_part shape: (B, num_of_pl_tokens, hidden_size)

        # 在分类任务中：映射到1维（每个位置是一个类别的logit）
        return {'y_logit': self.label_linear(lab_part)[..., 0]}  # (B, num_classes)
```

### 3. 迁移到蛋白质结构预测的关键修改

| 维度 | 师兄的分类任务 | 我们的结构预测任务 |
|------|--------------|-----------------|
| **输入** | 视频+文本描述 | 蛋白质序列（氨基酸字符串） |
| **`num_of_pl_tokens`** | 类别数量（固定，如100） | 序列长度（变长，需padding） |
| **placeholder的语义** | 每个placeholder代表一个类别 | 每个placeholder代表一个氨基酸 |
| **输出形状** | (B, 类别数, 1) → (B, 类别数) | (B, 序列长度, 3) |
| **输出含义** | 每个位置是一个类别的logit | 每个位置是一个氨基酸的(x,y,z)坐标 |
| **损失函数** | MultiLabelSoftMarginLoss | MSE Loss（回归损失） |

**核心思路一致**：都是通过MTP将LLM的输出转换为定长格式（k个输出位置），只是输出的语义和任务类型不同。

---

## 待讨论问题

### 1. 坐标预测的具体形式 ✅ 已解决
- ~~预测每个氨基酸的Cα原子坐标？还是所有骨架原子？~~
- ~~坐标是绝对坐标还是相对坐标？~~
- ~~是否需要预测侧链原子？~~

**结论（2026-01-30）**：
- 当前预测Cα原子坐标
- **重要修正**：不再预测绝对坐标，改为预测**距离矩阵**（旋转/平移不变）
- 侧链原子暂不考虑，后续优化

### 2. 输入设计 ✅ 已确定
- 输入是蛋白质序列（氨基酸字符串）
- 暂不加入配体信息，保持任务简单

### 3. 模型选择 ✅ 已确定
- 正式训练使用 Qwen3-8B
- 硬件：RTX 3090 (24GB) + 梯度检查点 + LoRA

### 4. 评估指标
- RMSD（均方根偏差）- 需要从距离矩阵还原坐标后计算
- GDT-TS
- TM-score
- **新增：距离矩阵误差（MAE/MSE）**

### 5. 距离矩阵方案的技术细节（待确定）
- 输出完整的 N×N 矩阵还是只输出上三角（N*(N-1)/2 个值）？
- 如何处理变长序列？padding部分的距离如何mask？
- 从距离矩阵还原坐标用什么算法？（MDS、梯度下降等）

---

## 代码改造计划

### 1. 数据处理改造

**目标**：从PDB文件提取序列和坐标，准备训练数据

```python
def parse_pdb_sequence_and_coords(pdb_file):
    """
    从PDB文件提取序列和Cα原子坐标

    返回：
    - sequence: 氨基酸序列字符串，如"MKTAYIAK..."
    - coords: Cα坐标数组，shape=(seq_len, 3)
    """
    sequence, coords = [], []
    for line in open(pdb_file):
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            residue = line[17:20].strip()
            sequence.append(AA_3TO1[residue])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            coords.append([x, y, z])
    return ''.join(sequence), np.array(coords)
```

**数据格式设计**：
```json
{
  "pdb_code": "1a00",
  "sequence": "MKTAYIAKQRQISFVK...",
  "coords": [[x1,y1,z1], [x2,y2,z2], ...],
  "seq_len": 128
}
```

**关键点**：
- 需要处理变长序列（padding/截断）
- 坐标需要归一化（零均值化或归一化到[-1,1]）
- padding部分的坐标label设为-100（在loss计算时mask掉）

### 2. 模型改造（参考师兄的HuggingfaceModelWithMTP）

```python
class ProteinStructureMTP(nn.Module):
    def __init__(self, base_model, max_seq_len):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size

        # 参考师兄代码的LabelWiseAttention
        # 将 num_of_pl_tokens 个位置的hidden states映射到输出
        self.coord_proj = LabelWiseAttention(hidden_size, max_seq_len)

        # 坐标预测头：将特征映射到3D坐标
        self.coord_head = nn.Linear(hidden_size, 3)

    def forward(self, inputs, num_of_pl_tokens):
        # 获取hidden states
        hidden = self.base_model(**inputs, output_hidden_states=True).last_hidden_state

        # 参考师兄代码：分离输入和placeholder
        lm_part = hidden[..., :-num_of_pl_tokens, :]
        pl_part = hidden[..., -num_of_pl_tokens:, :]

        # 池化输入特征
        lm_part_pooled = lm_part.max(dim=1, keepdims=True)[0]

        # LabelWiseAttention映射
        coord_features = self.coord_proj(
            torch.cat([lm_part_pooled, pl_part], dim=1)
        )  # (B, max_seq_len, hidden_size)

        # 预测坐标
        coords = self.coord_head(coord_features)  # (B, max_seq_len, 3)
        return coords
```

**关键修改点**：
1. 输出维度：从 `(B, num_classes, 1)` 改为 `(B, seq_len, 3)`
2. 预测头：从 `nn.Linear(hidden_size, 1)` 改为 `nn.Linear(hidden_size, 3)`
3. `num_of_pl_tokens` 从固定值改为动态（等于序列长度）

### 3. 损失函数改造

```python
# 师兄代码：分类损失
loss = nn.MultiLabelSoftMarginLoss()(logits, labels)

# 我们的代码：回归损失（带mask）
def compute_coord_loss(pred_coords, true_coords, mask):
    """
    pred_coords: (B, seq_len, 3)
    true_coords: (B, seq_len, 3)
    mask: (B, seq_len) - padding位置为0，有效位置为1
    """
    # 只计算有效位置的损失
    loss = F.mse_loss(pred_coords * mask.unsqueeze(-1),
                       true_coords * mask.unsqueeze(-1))
    return loss
```

---

## 实验计划

**Phase 1: 可行性验证（Week 1-2）**
- 用最小配置跑通流程
- 模型：Qwen-0.5B
- 数据：PDBbind子集（100-500条）
- 任务：预测Cα原子坐标

**Phase 2: 基线实验（Week 3-4）**
- 建立基线，评估MTP方法的效果
- 对比不同的坐标表示方式

**Phase 3: 优化与扩展（Week 5-8）**
- 优化模型结构
- 扩大数据集规模
- 尝试更大的模型

---

## 会议记录

### 2026-01-30 师兄反馈：绝对坐标回归问题

**反馈来源**：师兄代码审查

**核心问题**：
当前方案直接回归"绝对坐标"（蛋白质原子在三维空间的实际 x/y/z 值）存在根本性缺陷——绝对坐标受蛋白质整体的**平移**和**旋转**影响大，模型很难学准。

**问题分析**：
1. 同一个蛋白质结构，放在不同的空间位置（平移）会产生完全不同的坐标值
2. 同一个蛋白质结构，不同的朝向（旋转）也会产生完全不同的坐标值
3. 但这些不同的坐标实际上代表的是**相同的结构**
4. 模型需要学习"忽略"这些平移和旋转带来的差异，这大大增加了学习难度

**师兄给出的3个解决方案**：

| 方案 | 描述 | 优点 | 缺点 | 实现难度 |
|-----|------|------|------|---------|
| **方案1: FAPE Loss** | 使用AlphaFold的Frame-Aligned Point Error损失函数，对齐蛋白质的局部参考帧（骨架结构）计算误差 | 专门为蛋白质结构设计，生物学意义强 | 实现复杂，需要理解AlphaFold架构 | 高 |
| **方案2: 距离矩阵+MSE** | 将原子坐标转换为pair-wise距离矩阵（N×N），预测原子间的欧氏距离 | **旋转/平移不变**，改动代码少，落地快 | 输出维度增大（N² vs N×3），需要从距离还原坐标 | **低（推荐）** |
| **方案3: 等变网络** | 使用SE(3)等变神经网络，输入的几何变换对应输出的同类型变换 | 理论上最优雅 | 需要重新设计网络架构，实现复杂度高 | 高 |

**落地决策**：
采用**方案2：距离矩阵+MSE**，原因：
1. 改动成本最低，只需修改输出层和损失函数
2. 距离是旋转/平移不变的，模型更容易学习
3. 后续可以通过多维尺度分析（MDS）从距离矩阵还原坐标
4. FAPE loss和等变网络的难度较高，作为后续优化方向

**技术细节**：
```
原方案：
  输入: 蛋白质序列 "MKTAYIAK..."
  输出: (N, 3) 绝对坐标矩阵
  损失: MSE(pred_coords, true_coords)
  问题: 平移/旋转敏感

新方案：
  输入: 蛋白质序列 "MKTAYIAK..."
  输出: (N, N) 距离矩阵（或上三角 N*(N-1)/2 个距离）
  损失: MSE(pred_dist_matrix, true_dist_matrix)
  优点: 平移/旋转不变
```

**代码修改计划**：
1. 修改数据预处理：从坐标计算距离矩阵
2. 修改模型输出层：从 (N, 3) 改为 (N, N) 或 (N*(N-1)/2,)
3. 修改损失函数：计算距离矩阵的MSE
4. 添加坐标还原模块：从预测的距离矩阵重建坐标（用于可视化和评估）

**后续优化方向**：
- 尝试FAPE loss（需要深入研究AlphaFold代码）
- 探索等变网络（如SE(3)-Transformer, E(n) Equivariant GNN）

---

### 2026-01-24 初次讨论
和师兄讨论，确定方向为MTP。师兄给了资料和代码，建议先用小模型跑通。

### 2026-01-25 任务方向修正
明确了任务的核心方向：
- 目标是蛋白质结构预测（坐标预测），不是序列生成
- MTP的作用是将LLM输出转为定长格式，k值与序列长度一致
- 参考师兄代码中MTP用于分类的思路，迁移到坐标回归

---

## 数据集探索记录 (2026-01-24)

**PDBbind+ 数据集统计**：
- Protein-ligand Complex: 27,385
- Protein-Protein Complex: 4,594
- Protein-nucleic acid Complex: 1,400+

**下载方案**：
选了 PDBbind v.2020.R1（免费版），包含 19,037 个蛋白质-配体复合物结构。

**数据集结构**：
- `index/` 目录下是索引文件
- `P-L/` 目录下是复合物结构（PDB, Mol2, SDF）
- PDB文件包含：
  - SEQRES记录（氨基酸序列）
  - ATOM记录（原子坐标）<- 这是我们需要的

**数据预处理需要做的事**：
1. 从PDB文件提取ATOM记录
2. 解析每个原子的坐标 (x, y, z)
3. 按氨基酸组织坐标数据
4. 设计训练数据格式

---

## 参考资源

- **论文**: [Multi-Token Prediction](https://arxiv.org/abs/2504.03159)
- **数据集**: [PDBbind-Plus](https://www.pdbbind-plus.org.cn/)
- **代码**: 师兄的MTP实现（分类任务）, Hugging Face Transformers, PEFT
- **相关工作**: AlphaFold, ESMFold, RoseTTAFold
