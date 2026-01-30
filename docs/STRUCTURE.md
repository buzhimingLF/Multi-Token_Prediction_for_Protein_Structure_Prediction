# 项目目录结构说明

## 根目录文件

### 核心文档
- `README.md` - 项目主文档
- `CLAUDE.md` - Claude Code使用指南
- `DISCUSSION.md` - 技术讨论和实验记录
- `PROJECT_SUMMARY.md` - 项目总结
- `start.md` - 项目需求文档

### 参考代码（不要修改）
- `SFT_main.py` - 师兄的MTP参考实现
- `SFT_infer.py` - 参考推理脚本

## 目录结构

```
.
├── docs/                       # 📚 文档目录
│   ├── reports/               # 实验报告
│   │   ├── TRAINING_STATUS.md
│   │   ├── TRAINING_COMPLETE.md
│   │   ├── INFERENCE_TEST_REPORT.md
│   │   └── SESSION_SUMMARY.md
│   ├── planning/              # 规划文档
│   │   └── NEXT_STEPS.md
│   └── STRUCTURE.md           # 本文件
│
├── scripts/                    # 🔧 脚本目录
│   ├── training/              # 训练脚本
│   │   ├── train_distmat.py           # v2: 距离矩阵训练
│   │   ├── train_protein_structure.py # v1: 坐标回归训练
│   │   └── train_protein_mtp.py       # 早期MTP版本
│   │
│   ├── inference/             # 推理脚本
│   │   ├── infer_distmat.py           # v2: 距离矩阵推理
│   │   └── infer_protein_structure.py # v1: 坐标推理
│   │
│   ├── data_processing/       # 数据处理
│   │   ├── data_preprocessing.py      # PDB数据提取
│   │   ├── create_distmat_data.py     # 距离矩阵数据生成
│   │   ├── create_coord_data.py       # 坐标数据生成
│   │   └── create_mtp_data.py         # MTP数据格式
│   │
│   ├── evaluation/            # 评估脚本
│   │   ├── evaluate_structure.py      # 结构评估
│   │   └── mds_reconstruct.py         # MDS坐标重建
│   │
│   ├── extract_test_samples.py  # 测试样本提取
│   ├── test_inference.py        # 推理测试
│   └── monitor_training.sh      # 训练监控
│
├── output_distmat/             # 🤖 v2模型输出（距离矩阵）
│   ├── model.pt
│   ├── training_config.json
│   └── checkpoint-*/
│
├── venv_distmat/               # 🐍 Python虚拟环境
│
├── P-L/                        # 📊 PDBbind数据集
├── index/                      # 📊 数据集索引
│
└── 数据文件（已gitignore）
    ├── distmat_train.json      # 距离矩阵训练数据
    ├── distmat_val.json        # 距离矩阵验证数据
    ├── protein_coords_data.json # 原始坐标数据
    └── test_samples.json       # 测试样本
```

## 使用说明

### 训练模型
```bash
# v2 距离矩阵方案（推荐）
python scripts/training/train_distmat.py \
    --train_data distmat_train.json \
    --val_data distmat_val.json \
    --model_name Qwen/Qwen2.5-0.5B \
    --output_dir ./output_distmat
```

### 推理预测
```bash
# v2 距离矩阵推理
python scripts/inference/infer_distmat.py \
    --model_path ./output_distmat \
    --sequence "YOUR_SEQUENCE" \
    --output output.pdb
```

### 数据准备
```bash
# 1. 提取PDB坐标
python scripts/data_processing/data_preprocessing.py --extract_coords

# 2. 生成距离矩阵训练数据
python scripts/data_processing/create_distmat_data.py \
    --input protein_coords_data.json \
    --output distmat_train.json
```

### 评估模型
```bash
# 结构评估
python scripts/evaluation/evaluate_structure.py \
    --pred predicted.pdb \
    --true ground_truth.pdb
```

## 文件命名规范

### 训练输出
- `output_distmat/` - 距离矩阵模型（v2）
- `output_structure/` - 坐标回归模型（v1）
- `output_*/checkpoint-N/` - 训练检查点

### 文档
- `*_REPORT.md` - 实验报告
- `*_STATUS.md` - 状态记录
- `*_SUMMARY.md` - 总结文档

### 脚本
- `train_*.py` - 训练脚本
- `infer_*.py` - 推理脚本
- `create_*.py` - 数据生成
- `evaluate_*.py` - 评估脚本

## 更新日志

### 2026-01-30
- ✅ 创建目录结构
- ✅ 分类整理脚本文件
- ✅ 整理文档报告
- ✅ 添加本说明文档
