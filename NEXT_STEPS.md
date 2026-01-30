# ProteinMTP 后续步骤规划

**更新时间**: 2026-01-30 16:15
**当前状态**: 距离矩阵模型训练中 (64%, 467/725步)

---

## 当前训练状态

### 训练进度
- **模型**: Qwen2.5-0.5B
- **方法**: 距离矩阵预测 (v2方案)
- **进度**: 64% (467/725步)
- **速度**: ~1.6秒/步
- **预计完成**: 约7分钟
- **GPU状态**: RTX 4090, 利用率0%, 显存2GB/24GB, 温度41°C
- **输出目录**: `./output_distmat`

### 训练监控
```bash
# 实时查看训练进度
tail -f /tmp/claude-11685/-home-haijun-huang-Multi-Token-Prediction-for-Protein-Structure-Prediction/tasks/b005c53.output

# 使用监控脚本
./monitor_training.sh

# 检查GPU状态
nvidia-smi
```

---

## 短期任务 (训练完成后立即执行)

### 1. 验证训练结果 ✓ 优先级：高
**目标**: 确认模型训练成功，验证输出文件完整性

**步骤**:
```bash
# 检查模型文件
ls -lh output_distmat/
cat output_distmat/training_config.json

# 验证必要文件存在
# - model.pt (模型权重)
# - training_config.json (训练配置)
# - tokenizer相关文件
```

**预期结果**:
- 所有必要的模型文件完整
- training_config.json包含全局统计信息
- 模型大小约几百MB

---

### 2. 模型推理测试 ✓ 优先级：高
**目标**: 验证训练好的模型能正常推理

**步骤**:
```bash
# 激活虚拟环境
source venv_distmat/bin/activate

# 使用测试序列进行推理
python infer_distmat.py \
    --model_path ./output_distmat \
    --sequence "MKTAYIAKQRQISFVKTIGDEVQREAPGDSRLAGHFELSC" \
    --output test_pred_distmat.pdb \
    --mds_method classical \
    --save_distmat

# 检查输出
ls -lh test_pred_distmat.pdb
ls -lh test_pred_distmat_distmat.npy
```

**预期结果**:
- 成功预测距离矩阵
- MDS重建3D坐标
- 输出PDB文件格式正确
- 坐标范围合理（10-50 Å）

**如果推理失败**:
- 检查模型加载是否正常
- 验证归一化参数是否正确保存
- 查看错误日志定位问题

---

### 3. 评估预测质量 ✓ 优先级：高
**目标**: 定量评估模型性能

**步骤**:
```bash
# 在验证集上评估
python evaluate_distmat_model.py \
    --model_path ./output_distmat \
    --test_data distmat_val.json \
    --num_samples 10 \
    --output_dir ./eval_results

# 计算指标:
# - 距离矩阵MAE/RMSE
# - MDS重建质量
# - 与真实结构的RMSD (如果有ground truth)
```

**关键指标**:
- **距离MAE**: 期望 < 5 Å
- **MDS相关系数**: 期望 > 0.7
- **RMSD** (如果有GT): 期望 < 10 Å

---

## 中期任务 (1-3天)

### 4. 结果分析与可视化 ✓ 优先级：中
**目标**: 深入分析模型预测结果

**任务**:
- [ ] 可视化预测的距离矩阵 vs 真实距离矩阵
- [ ] 分析预测误差分布
- [ ] 检查长程vs短程距离预测准确度
- [ ] 对比不同序列长度的预测效果
- [ ] 3D结构可视化（PyMOL或py3Dmol）

**工具**:
```python
# 创建可视化脚本
import matplotlib.pyplot as plt
import numpy as np

# 绘制距离矩阵热图
# 绘制误差分布直方图
# 绘制长程vs短程距离scatter plot
```

---

### 5. 与v1方案对比 ✓ 优先级：中
**目标**: 验证v2(距离矩阵)相比v1(绝对坐标)的优势

**对比维度**:
| 维度 | v1 (绝对坐标) | v2 (距离矩阵) |
|------|--------------|--------------|
| **训练Loss收敛** | ? | 待测 |
| **预测稳定性** | 平移/旋转敏感 | 不变 |
| **RMSD** | ? | 待测 |
| **长程距离准确度** | ? | 待测 |

**步骤**:
- 如果有v1训练的模型，使用相同测试集对比
- 分析哪种方案更适合蛋白质结构预测

---

### 6. 扩大训练规模 ✓ 优先级：中
**目标**: 使用更大的模型和更多数据

**选项A: 更大的模型**
```bash
# Qwen2.5-1.5B 或 Qwen3-4B
python train_distmat.py \
    --model_name Qwen/Qwen2.5-1.5B \
    --max_seq_len 256 \
    --num_epochs 5 \
    --gradient_checkpointing \
    --use_4bit \
    --output_dir ./output_distmat_1.5b
```

**选项B: 更多数据**
```bash
# 使用全部PDBbind数据（增加到5000+样本）
python create_distmat_data.py \
    --input protein_coords_data.json \
    --output distmat_full_train.json \
    --max_samples 5000 \
    --max_seq_len 256
```

**选项C: 更长序列**
```bash
# 支持512长度序列（距离矩阵512×512，需要更多显存）
python train_distmat.py \
    --max_seq_len 512 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 16
```

---

## 长期优化方向 (1-2周)

### 7. 模型架构改进
- [ ] **FAPE Loss**: 实现AlphaFold的Frame-Aligned Point Error
- [ ] **注意力机制**: 在距离矩阵预测中引入2D attention
- [ ] **多尺度预测**: 同时预测局部和全局距离
- [ ] **置信度估计**: 为每个预测距离输出置信分数

### 8. 数据增强
- [ ] **旋转增强**: 虽然距离不变，但可以增强输入序列表示
- [ ] **掩码训练**: 随机mask部分残基，提高泛化能力
- [ ] **噪声注入**: 在训练时给距离矩阵加噪声

### 9. 多任务学习
- [ ] 同时预测距离矩阵和接触图（Contact Map）
- [ ] 同时预测二级结构（Alpha helix, Beta sheet）
- [ ] 同时预测溶剂可及性（Solvent Accessibility）

### 10. 基准对比
- [ ] 与AlphaFold2/3对比
- [ ] 与ESMFold对比
- [ ] 在CASP数据集上评估

---

## 实验记录与文档

### 需要更新的文档
- [ ] **DISCUSSION.md**: 记录v2模型训练结果
- [ ] **README.md**: 更新距离矩阵方案的使用说明
- [ ] **CLAUDE.md**: 添加虚拟环境使用说明

### 实验日志
建议创建 `experiments/` 目录记录每次实验:
```
experiments/
├── 2026-01-30_distmat_0.5b/
│   ├── config.json          # 训练配置
│   ├── training_log.txt     # 训练日志
│   ├── eval_results.json    # 评估结果
│   └── visualizations/      # 可视化图表
```

---

## 潜在问题与解决方案

### 问题1: MDS重建质量不佳
**现象**: 距离矩阵预测准确，但MDS还原的3D坐标RMSD很高

**可能原因**:
- 距离矩阵噪声累积
- MDS算法局限性（只能恢复到旋转/平移/镜像）

**解决方案**:
1. 尝试Iterative MDS代替Classical MDS
2. 使用基于梯度下降的坐标优化
3. 结合距离矩阵和角度信息

### 问题2: 长程距离预测不准
**现象**: 残基间距离<10Å预测准确，>20Å误差大

**可能原因**:
- 长程相互作用信息在LLM hidden states中衰减
- 训练数据中长程距离样本不足

**解决方案**:
1. 引入2D卷积或attention在距离矩阵维度
2. 分层预测：先预测粗粒度距离，再细化
3. 增加长程距离的loss权重

### 问题3: 不同长度序列泛化性差
**现象**: 训练长度256，推理长度100效果差

**可能原因**:
- Placeholder数量变化导致特征分布偏移
- 位置编码不适配变长输入

**解决方案**:
1. 训练时混合不同长度序列
2. 使用相对位置编码
3. 动态调整placeholder数量

---

## 检查清单

训练完成后，按顺序检查：

- [ ] 1. 训练日志无异常（loss正常下降）
- [ ] 2. 模型文件完整（model.pt, config.json, tokenizer）
- [ ] 3. 推理脚本能正常运行
- [ ] 4. 预测的PDB文件格式正确
- [ ] 5. 坐标数值范围合理（10-50 Å）
- [ ] 6. 距离矩阵MAE < 5 Å
- [ ] 7. MDS重建相关系数 > 0.7
- [ ] 8. 代码已提交到GitHub
- [ ] 9. 实验结果记录到DISCUSSION.md
- [ ] 10. 清理临时文件和大文件

---

## 资源管理

### 磁盘空间
当前使用情况:
- 虚拟环境: 5.4GB
- 训练数据: ~1GB (distmat_train.json + distmat_val.json)
- 模型输出: 待确认

**清理建议**:
```bash
# 训练完成后可删除
rm -rf venv_distmat/  # 可重建
rm -f distmat_train.json distmat_val.json  # 可重新生成

# 保留重要文件
# - output_distmat/  (训练好的模型)
# - protein_coords_data.json  (原始数据)
# - 所有Python脚本
```

### GPU资源
- 当前训练: RTX 4090 (24GB)
- 显存使用: ~2GB (小模型)
- 建议保留用于后续推理和更大模型训练

---

**下一步行动**: 等待训练完成 → 验证结果 → 推理测试 → 性能评估
