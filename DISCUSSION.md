# ProteinMTP 项目进度与讨论文档

> 📅 **创建日期**：2026-01-24 
> 👤 **作者**：黄海军  
> 🔗 **GitHub 仓库**：git@github.com:buzhimingLF/ProteinMTP---Multi-Token-Prediction-for-Protein-Sequence-Generation.git

---

## 📌 项目简介

**项目名称**：ProteinMTP - 基于多标记预测的蛋白质序列生成研究

**核心目标**：将 Multi-Token Prediction (MTP) 技术应用于蛋白质序列预测任务，探索 MTP 在生物序列生成领域的潜力。

---

## 🚀 当前进度

### ✅ 已完成
- [x] 确定研究方向：Multi-Token Prediction for Protein Sequences
- [x] 获取参考论文：[arxiv:2504.03159](https://arxiv.org/abs/2504.03159)
- [x] 确定数据集：[PDBbind-Plus](https://www.pdbbind-plus.org.cn/)
- [x] 获取师兄的参考代码（基于 Qwen 的 MTP 实现）

### 🔄 进行中
- [ ] 阅读理解参考论文
- [ ] 探索 PDBbind-Plus 数据集结构
- [ ] 分析师兄代码的核心逻辑

### ⏳ 待开始
- [ ] 代码改造（VL → 纯文本）
- [ ] 数据预处理流程设计
- [ ] 实验设计与执行

---

## ❓ 待讨论问题

### 🔴 高优先级

#### 1. 数据集选择与处理
**问题**：PDBbind-Plus 数据集主要是蛋白质-配体结合数据，如何从中提取适合 MTP 训练的蛋白质序列数据？

**我的想法**：
- 方案A：直接使用蛋白质序列作为训练数据
- 方案B：结合配体信息，做条件生成任务

**需要师兄确认**：
- 数据集的具体使用方式？
- 是否有推荐的数据预处理流程？

---

#### 2. 任务定义
**问题**：具体的预测任务是什么？

**可能的任务类型**：
| 任务类型 | 描述 | 难度 |
|---------|------|------|
| 序列补全 | 给定部分序列，预测剩余部分 | ⭐⭐ |
| 序列生成 | 从头生成完整蛋白质序列 | ⭐⭐⭐ |
| 条件生成 | 根据功能描述生成序列 | ⭐⭐⭐⭐ |
| 序列优化 | 优化现有序列的某些属性 | ⭐⭐⭐ |

**需要师兄确认**：推荐哪种任务类型？

---

#### 3. 模型选择
**问题**：最小模型具体选择哪个？

**候选模型**：
- Qwen-0.5B（最小，适合快速验证）
- Qwen-1.8B（平衡性能和资源）
- ESM-2（专门针对蛋白质的预训练模型）

**硬件限制**：RTX 3090 (24GB)

**需要师兄建议**：
- 是否继续使用 Qwen 系列？
- 还是考虑使用蛋白质专用模型如 ESM？

---

### 🟡 中优先级

#### 4. MTP 预测头数量
**问题**：Multi-Token Prediction 中，一次预测多少个 token 比较合适？

**参考**：
- 论文中的设置是？
- 蛋白质序列的特点（20种氨基酸）是否需要特殊考虑？

---

#### 5. 评估指标
**问题**：如何评估 MTP 在蛋白质序列上的效果？

**可能的指标**：
- Perplexity
- 序列准确率
- BLEU Score
- 生物学有效性（需要额外验证）

---

### 🟢 低优先级

#### 6. 代码改造细节
**问题**：师兄代码中的一些实现细节

**观察到的问题**：
- `HuggingfaceModelWithMTP` 类中使用了 `LabelWiseAttention`，这个在纯文本任务中如何调整？
- `num_of_pl_tokens` 参数的作用和设置？

---

## 💡 我的想法与思考

### 关于 MTP 的理解
Multi-Token Prediction 的核心思想是一次预测多个 token，而不是传统自回归的一次一个。这在蛋白质序列上可能有以下优势：

1. **效率提升**：蛋白质序列通常很长（几百到几千个氨基酸），MTP 可以显著加速生成
2. **全局依赖**：蛋白质的功能依赖于整体结构，MTP 可能更好地捕捉长程依赖
3. **并行化**：更适合 GPU 并行计算

### 潜在挑战
1. **序列特殊性**：蛋白质序列只有20种氨基酸，词表很小，是否需要特殊处理？
2. **生物学约束**：生成的序列需要满足一定的生物学规则
3. **评估困难**：如何验证生成序列的生物学有效性？

---

## 📊 实验计划

### Phase 1: 可行性验证（Week 1-2）
```
目标：用最小配置跑通整个流程
- 模型：Qwen-0.5B 或更小
- 数据：PDBbind-Plus 子集（1000条）
- 任务：简单的序列补全
```

### Phase 2: 基线实验（Week 3-4）
```
目标：建立基线，对比 MTP vs 传统自回归
- 实现标准自回归作为 baseline
- 实现 MTP 版本
- 对比两者的性能和效率
```

### Phase 3: 优化与扩展（Week 5-8）
```
目标：优化模型，扩大实验规模
- 调整 MTP 参数（预测头数量等）
- 扩大数据集规模
- 尝试更大的模型
```

---

## 📝 会议记录

### 2026-01-24 初次讨论
**参与者**：我、师兄

**讨论内容**：
- 确定研究方向为 MTP
- 师兄提供了参考资料和代码
- 建议先用小模型跑通

**Action Items**：
- [x] 我：阅读论文，理解 MTP 核心思想
- [x] 我：探索数据集，了解数据格式
- [ ] 我：分析师兄代码，理解实现细节

---

### 2026-01-24 数据集探索记录

**PDBbind+ 数据集统计**：
| 类型 | 数量 |
|------|------|
| Protein-ligand Complex | 27,385 |
| Protein-Protein Complex | 4,594 |
| Protein-nucleic acid Complex | 1,400+ |

---

### 📥 数据集下载方案（确定）

**首选：PDBbind v.2020.R1（免费版本）**

| 序号 | 文件 | 大小 | 说明 |
|------|------|------|------|
| 1 | **Index files** | 497KB | 索引文件（PDB代码、分辨率、结合数据等） |
| 2 | **Protein-ligand complex structures** | 3.1GB | 19,037个蛋白质-配体复合物结构 |

**选择理由**：
1. ✅ **免费** - 注册用户即可下载
2. ✅ **质量更高** - 使用v.2024的新工作流程重新处理
3. ✅ **数据量适中** - 19,037个复合物足够训练
4. ✅ **适合初期实验** - 先用这个跑通流程

**数据特点**：
- 包含 PDBbind v.2020 中仍保留在 v.2024 的复合物
- 结构文件已用 v.2024 新工作流程重新处理
- 部分复合物的结合数据已更新

**下载步骤**：
1. 访问 https://www.pdbbind-plus.org.cn/
2. 注册账号并登录
3. 下载 PDBbind v.2020.R1 的 Index files 和 Protein-ligand complex structures
4. ❗ **注意**：下载链接有时效性，点击后需立即下载

**如果链接过期（Request has expired）**：
- 重新登录网站
- 重新点击下载按钮获取新链接
- 立即下载（不要等待）
- 如持续失败，联系技术支持：support@pdbbind-plus.org.cn

**下一步行动**：
1. ✅ 注册 PDBbind 账号下载数据集
2. ✅ 分析数据格式
3. 设计预处理流程
4. 开始代码改造工作

---

### 2026-01-24 数据集结构分析

**数据集已下载并解压到以下目录**：
```
毕业论文/
├── index/                    # 索引文件
│   ├── INDEX_general_PL.2020R1.lst   # 蛋白质-配体 (19,037条)
│   ├── INDEX_general_PP.2020R1.lst   # 蛋白质-蛋白质 (2,798条)
│   ├── INDEX_general_PN.2020R1.lst   # 蛋白质-核酸 (1,032条)
│   ├── INDEX_general_NL.2020R1.lst   # 核酸-配体 (143条)
│   └── README
└── P-L/                      # 蛋白质-配体复合物结构
    ├── 1981-2000/            # 1,111个复合物
    ├── 2001-2010/            # 5,691个复合物
    └── 2011-2019/            # 12,235个复合物
```

**索引文件格式** (INDEX_general_PL.2020R1.lst)：
```
# PDB code, resolution, release year, binding data, reference, ligand name
2tpi  2.10  1982  Kd=49uM       // 2tpi.pdf (2-mer)
5tln  2.30  1982  Ki=0.43uM     // 5tln.pdf (BAN)
```

**复合物目录结构** (每个 PDB code 对应一个目录)：
```
2tpi/
├── 2tpi_protein.pdb    # 蛋白质结构 (PDB格式)
├── 2tpi_pocket.pdb     # 结合位点结构
├── 2tpi_ligand.mol2    # 配体结构 (Mol2格式)
└── 2tpi_ligand.sdf     # 配体结构 (SDF格式)
```

**PDB文件中的蛋白质序列** (SEQRES记录)：
```
SEQRES   1 Z  229  VAL ASP ASP ASP ASP LYS ILE VAL GLY GLY TYR THR CYS
SEQRES   2 Z  229  GLY ALA ASN THR VAL PRO TYR GLN VAL SER LEU ASN SER
...
```
- 使用三字母缩写表示氨基酸
- 需要转换为单字母序列用于模型训练

**氨基酸编码表**：
| 三字母 | 单字母 | 名称 |
|--------|--------|------|
| ALA | A | 丙氨酸 |
| CYS | C | 半胱氨酸 |
| ASP | D | 天冒氨酸 |
| GLU | E | 谷氨酸 |
| PHE | F | 苯丙氨酸 |
| GLY | G | 甘氨酸 |
| HIS | H | 组氨酸 |
| ILE | I | 异亮氨酸 |
| LYS | K | 赖氨酸 |
| LEU | L | 亮氨酸 |
| MET | M | 甲硫氨酸 |
| ASN | N | 天冒酰胺 |
| PRO | P | 脂氨酸 |
| GLN | Q | 谷氨酰胺 |
| ARG | R | 精氨酸 |
| SER | S | 丝氨酸 |
| THR | T | 苏氨酸 |
| VAL | V | 缮氨酸 |
| TRP | W | 色氨酸 |
| TYR | Y | 酪氨酸 |

**数据预处理流程**：
1. ✅ 从 PDB 文件提取 SEQRES 记录
2. ✅ 将三字母氨基酸转换为单字母序列
3. ✅ 结合索引文件中的结合数据
4. 设计训练数据格式

---

### 2026-01-24 数据预处理脚本完成

**创建了 `data_preprocessing.py`**，功能：
- 解析索引文件，提取复合物信息
- 从 PDB 文件提取 SEQRES 记录
- 将三字母氨基酸转换为单字母序列
- 输出 JSON 格式的数据文件

**测试结果**（100个样本）：
```
成功: 100, 错误: 0
序列长度统计:
  最短: 104
  最长: 2856
  平均: 412.6
```

**示例数据**：
```json
{
  "pdb_code": "2tpi",
  "sequence_length": 287,
  "binding_data": "Kd=49uM",
  "full_sequence": "VDDDDKIVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKS..."
}
```

**下一步**：
1. ✅ 处理全部 19,037 个复合物
2. ✅ 设计 MTP 训练数据格式
3. 开始代码改造

---

### 2026-01-24 MTP 训练数据准备完成

**全量数据处理结果**：
```
原始数据: 19,037 个复合物
成功率: 100%
序列长度: 24-5400, 平均 576.3
```

**MTP 训练数据集**（过滤后序列长度 50-512）：
| 数据集 | 样本数 |
|--------|--------|
| 训练集 | 10,350 |
| 验证集 | 1,150 |
| **总计** | **11,500** |

**序列长度分布**：
| 长度区间 | 数量 | 占比 |
|----------|------|------|
| 50-100 | 305 | 2.7% |
| 100-200 | 2,185 | 19.0% |
| 200-300 | 3,628 | 31.5% |
| 300-400 | 3,295 | 28.7% |
| 400-512 | 2,046 | 17.8% |

**生成的文件**：
```
mtp_data_train.json      # MTP 训练集 (10,350 样本)
mtp_data_val.json        # MTP 验证集 (1,150 样本)
instruction_data.json    # 指令微调格式 (11,500 样本)
protein_sequences.json   # 原始序列数据 (19,037 样本)
```

**MTP 配置**：
- 预测头数量: 4（论文建议值）
- 最大序列长度: 512

**下一步**：✅ 开始代码改造（VL → 纯文本）

---

### 2026-01-24 MTP 训练代码完成

**创建了 `train_protein_mtp.py`**，主要改造：

1. **模型架构**：
   - 基础模型：Qwen2.5-0.5B（纯文本，替代 VL 模型）
   - MTP 预测头：4 个独立的 Linear 层
   - LoRA 微调：减少显存占用

2. **数据处理**：
   - `ProteinSequenceDataset`: 加载蛋白质序列
   - `ProteinMTPCollator`: MTP 标签生成

3. **损失函数**：
   - 多头交叉熵损失
   - 每个头预测不同偏移的未来 token

**运行方式**：
```bash
python train_protein_mtp.py \
    --train_data mtp_data_train.json \
    --val_data mtp_data_val.json \
    --model_name Qwen/Qwen2.5-0.5B \
    --n_future_tokens 4 \
    --batch_size 4 \
    --num_epochs 3
```

**下一步**：
1. 在 3090 GPU 上测试训练
2. 调整超参数
3. 评估模型效果

---

## 🔗 参考资源

### 论文
- [Multi-Token Prediction](https://arxiv.org/abs/2504.03159) - 核心参考论文

### 数据集
- [PDBbind-Plus](https://www.pdbbind-plus.org.cn/) - 蛋白质-配体结合数据库

### 代码
- 师兄的 MTP 实现（本地）
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft) - LoRA 微调

### 相关工作
- ESM (Evolutionary Scale Modeling) - Meta 的蛋白质语言模型
- ProtTrans - 蛋白质 Transformer 模型
- AlphaFold - 蛋白质结构预测

---

## 📬 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues
- 微信/邮件（私下沟通）

---

*最后更新：2026-01-24*
