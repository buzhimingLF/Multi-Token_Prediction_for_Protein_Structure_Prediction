# ProteinMTP 项目完整总结

## 项目完成状态 ✅

所有核心功能已完成并测试通过!

---

## 📋 已完成的工作

### 1. 任务方向修正 ✅
- ✅ 明确任务目标: 蛋白质结构预测(坐标预测),而非序列生成
- ✅ 理解MTP作用: 限定LLM输出长度k=序列长度
- ✅ 明确技术路线: 参考师兄分类任务,改造为坐标回归
- ✅ 解释为何不用字符串生成(梯度不平滑、任务上限低)

### 2. 文档完善 ✅
- ✅ [start.md](start.md): 详细说明任务方向、MTP原理、代码改造要点
- ✅ [DISCUSSION.md](DISCUSSION.md): 更新项目背景、MTP分析、改造计划
- ✅ [README.md](README.md): 完整使用说明、工作流程、项目结构

### 3. 数据处理 ✅
- ✅ [data_preprocessing.py](data_preprocessing.py):
  - 从PDB文件提取序列和Cα坐标
  - 支持ATOM记录直接提取(确保序列坐标一一对应)
  - 检查序列长度与坐标数量匹配
  - 支持命令行参数控制

- ✅ [create_coord_data.py](create_coord_data.py):
  - 坐标归一化(零均值化+标准化)
  - 保存归一化统计信息(推理时反归一化)
  - 序列长度过滤
  - 自动划分训练集/验证集
  - 详细的数据统计输出

### 4. 模型训练 ✅
- ✅ [train_protein_structure.py](train_protein_structure.py):
  - 参考师兄代码实现MTP坐标回归
  - 使用LabelWiseAttention映射placeholder tokens
  - 输出维度: (B, seq_len, 3)
  - MSE Loss + RMSD评估指标
  - 支持LoRA微调(显存友好)
  - 保存训练配置到JSON

**核心组件**:
```python
class LabelWiseAttention(nn.Module):
    """将placeholder tokens映射到输出位置"""

class ProteinStructureMTP(nn.Module):
    """MTP坐标回归模型"""
    - coord_proj: LabelWiseAttention
    - coord_head: Linear(hidden_size, 3)
    - compute_rmsd: RMSD评估函数
```

### 5. 推理预测 ✅
- ✅ [infer_protein_structure.py](infer_protein_structure.py):
  - 加载训练好的模型(包括LoRA权重)
  - 预测单个序列的结构
  - 支持坐标反归一化
  - 输出标准PDB格式文件
  - 命令行友好

**示例**:
```bash
python infer_protein_structure.py \
    --model_path ./output_structure \
    --sequence "MKTAYIAK..." \
    --output predicted.pdb
```

### 6. 评估脚本 ✅
- ✅ [evaluate_structure.py](evaluate_structure.py):
  - **RMSD**: Root Mean Square Deviation
    - 实现Kabsch算法刚体对齐
    - 处理镜像情况
  - **TM-score**: Template Modeling score (范围[0,1])
  - **GDT-TS**: Global Distance Test (范围[0,100])
  - **接触图准确率**: Precision/Recall/F1
  - 提供评估标准参考

**示例**:
```bash
python evaluate_structure.py \
    --pred predicted.pdb \
    --true true_structure.pdb \
    --all_metrics
```

### 7. 可视化脚本 ✅
- ✅ [visualize_structure.py](visualize_structure.py):
  - 3D可视化蛋白质骨架
  - 2D投影(XY/XZ/YZ平面)
  - 结构统计信息:
    - 相邻Cα距离
    - 回转半径
    - 最大原子间距离
  - 输出高质量PNG图片

**示例**:
```bash
python visualize_structure.py \
    --pdb structure.pdb \
    --output viz.png \
    --mode both \
    --stats
```

### 8. 项目配置 ✅
- ✅ [requirements.txt](requirements.txt): 统一依赖管理
- ✅ 完整的README使用说明
- ✅ Git提交历史清晰

---

## 🎯 核心技术要点

### MTP的作用
```
输入: 蛋白质序列 "MKTAYIAK..."
      ↓ Tokenize
      [token1, token2, ..., tokenN]
      ↓ 添加placeholder tokens
      [token1, ..., tokenN, <unk>, <unk>, ..., <unk>]
                              └─────k个placeholder─────┘
                                     k=序列长度
      ↓ 通过LLM
      hidden_states: (B, N+k, hidden_size)
      ↓ 分离 + LabelWiseAttention
      输出: (B, k, 3) 坐标
```

### 为什么不用字符串生成?

| 方法 | 字符串生成 | MTP坐标回归 |
|-----|----------|------------|
| 输出形式 | "1.23,4.56,7.89" | [1.23, 4.56, 7.89] |
| 预测方式 | 逐token预测 | 直接回归 |
| 损失函数 | 交叉熵 | MSE |
| 梯度 | 离散,不平滑 | 连续,平滑 |
| 任务上限 | 低(间接映射) | 高(直接优化) |

### 参考师兄代码的关键点
1. **placeholder tokens**: 限定输出长度
2. **LabelWiseAttention**: 映射到输出位置
3. **hidden states分离**: lm_part + pl_part
4. **池化输入特征**: max pooling

---

## 📊 完整工作流程

### 流程图
```
PDBbind数据集
    ↓ data_preprocessing.py --extract_coords
protein_coords_data.json
    ↓ create_coord_data.py
coord_train.json + coord_val.json
    ↓ train_protein_structure.py
训练好的模型(./output_structure)
    ↓ infer_protein_structure.py
predicted_structure.pdb
    ↓ evaluate_structure.py / visualize_structure.py
评估报告 + 可视化图片
```

### 命令示例
```bash
# 1. 数据准备
python data_preprocessing.py --extract_coords --max_samples 1000
python create_coord_data.py --max_seq_len 512

# 2. 训练
python train_protein_structure.py \
    --train_data coord_train.json \
    --val_data coord_val.json \
    --model_name Qwen/Qwen2.5-0.5B \
    --max_seq_len 512 \
    --num_epochs 3 \
    --output_dir ./output_structure

# 3. 推理
python infer_protein_structure.py \
    --model_path ./output_structure \
    --sequence "MKTAYIAKQRQISFVK..." \
    --output predicted.pdb

# 4. 评估
python evaluate_structure.py \
    --pred predicted.pdb \
    --true true_structure.pdb \
    --all_metrics

# 5. 可视化
python visualize_structure.py \
    --pdb predicted.pdb \
    --output structure.png \
    --mode both --stats
```

---

## 🔬 技术创新点

### 1. MTP用于结构预测
- 首次将MTP从分类任务迁移到坐标回归
- k值从固定(类别数)改为动态(序列长度)
- 输出从logits改为连续坐标

### 2. 归一化策略
- 零均值化: 消除平移
- 标准化: 消除尺度差异
- 保存统计信息: 支持反归一化

### 3. 评估体系
- RMSD: 基础指标
- TM-score: 拓扑相似度
- GDT-TS: 多阈值覆盖率
- 接触图: 长程相互作用

### 4. 完整工具链
- 数据 → 训练 → 推理 → 评估 → 可视化
- 端到端流程
- 命令行友好

---

## 📈 实验指标参考

### RMSD标准
- < 2.0 Å: 高质量预测
- 2.0-5.0 Å: 中等质量
- > 5.0 Å: 低质量

### TM-score标准
- > 0.5: 相同折叠
- 0.4-0.5: 相似折叠
- < 0.4: 不同折叠

### GDT-TS标准
- > 50: 高质量
- 30-50: 中等质量
- < 30: 低质量

---

## 🎓 后续研究方向

### 1. 模型优化
- [ ] 尝试更大的基础模型(Qwen-1.8B, Qwen-7B)
- [ ] 使用ESM-2预训练模型(专门针对蛋白质)
- [ ] 实现dynamic padding支持更大batch_size
- [ ] 添加auxiliary loss(如距离约束、角度约束)

### 2. 数据增强
- [ ] 使用更多PDB数据
- [ ] 数据增强策略(旋转、平移)
- [ ] 多任务学习(结构+功能预测)

### 3. 评估完善
- [ ] 实现更多评估指标(如lDDT)
- [ ] 与AlphaFold/ESMFold对比
- [ ] 长序列蛋白质测试

### 4. 应用扩展
- [ ] 蛋白质-配体对接
- [ ] 突变效应预测
- [ ] 蛋白质设计

---

## 🙏 致谢

- **师兄**: 提供MTP分类任务参考代码
- **PDBbind-Plus**: 提供高质量蛋白质数据集
- **Qwen团队**: 提供开源LLM基础模型
- **Hugging Face**: 提供transformers和PEFT库

---

## 📝 Git提交记录

```
4fd4366 - 优化: 实现推理时坐标自动反归一化功能
510c9f2 - 修复: 解决模型保存失败问题
70e480b - 修复: 解决数据处理和训练脚本的多个关键问题
62f1755 - 文档: 添加项目完整总结文档
c611590 - 优化: 完善项目代码,添加评估、推理、可视化功能
c8beb39 - 修正: 任务方向从序列生成改为结构预测(坐标回归)
1e7e8ea - Add: MTP 训练脚本 train_protein_mtp.py
a26d03d - Add: MTP 训练数据准备完成
943dc1c - Add: 数据预处理脚本 data_preprocessing.py
```

### 最新修复 (2026-01-28)

**问题1: PDB文件权限错误**
- 原因: PDB文件被设置为可执行权限
- 解决: `chmod 644` 批量修复

**问题2: 索引文件缺失**
- 原因: INDEX_general_PL.2020R1.lst 不存在
- 解决: 添加 `scan_data_directory()` 自动扫描目录

**问题3: NumPy/scikit-learn兼容性**
- 原因: numpy.dtype size changed
- 解决: 重新安装 scikit-learn

**问题4: BFloat16 CPU不兼容**
- 原因: CPU不支持BFloat16的MSE运算
- 解决: 改用Float32

**问题5: 模型保存失败**
- 原因: Qwen的lm_head和embed_tokens共享权重，safetensors无法处理
- 解决: 使用torch.save保存model.pt

---

## ✅ 项目完成检查清单

- [x] 任务方向理解正确
- [x] 文档完整清晰
- [x] 数据处理脚本完善
- [x] 训练代码实现正确
- [x] 推理脚本可用
- [x] 评估脚本完整
- [x] 可视化功能完善
- [x] README使用说明详细
- [x] requirements.txt配置
- [x] Git提交历史清晰
- [x] 代码注释充分
- [x] 所有功能已推送到GitHub

---

**项目状态**: ✅ **Phase 1完成，准备开始正式训练!**

## 🚀 下一步计划：Phase 2 正式训练

### 推荐配置
```bash
# 使用更多数据和更长训练
python3 data_preprocessing.py --extract_coords --max_samples 5000
python3 create_coord_data.py --max_seq_len 512

# 正式训练（3个epochs）
python3 train_protein_structure.py \
    --train_data coord_train.json \
    --val_data coord_val.json \
    --model_name Qwen/Qwen2.5-0.5B \
    --max_seq_len 512 \
    --num_epochs 3 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --output_dir ./output_structure
```

### 评估计划
1. 使用验证集中的样本提取真实结构
2. 运行RMSD/TM-score评估
3. 与随机预测基线对比

*最后更新: 2026-01-28*
