# ProteinMTP

**基于多标记预测的蛋白质序列生成研究**

*Multi-Token Prediction for Protein Sequence Generation*

---

## 📖 项目简介

本项目是一个毕业论文研究项目，旨在将 **Multi-Token Prediction (MTP)** 技术应用于蛋白质序列预测任务。与传统的自回归模型（一次预测一个 token）不同，MTP 可以一次预测多个 token，从而提高生成效率并更好地捕捉序列的全局依赖关系。

## 🎯 研究目标

- 探索 MTP 在蛋白质序列生成任务上的应用
- 对比 MTP 与传统自回归方法的性能差异
- 验证 MTP 在生物序列领域的有效性

## 📚 参考资料

- **核心论文**: [arxiv:2504.03159](https://arxiv.org/abs/2504.03159)
- **数据集**: [PDBbind-Plus](https://www.pdbbind-plus.org.cn/)

## 🛠️ 技术栈

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- PEFT (LoRA 微调)
- DeepSpeed (可选)

## 📁 项目结构

```
ProteinMTP/
├── README.md              # 项目说明文档
├── start.md               # 需求文档（持续更新）
├── DISCUSSION.md          # 进度与讨论文档
├── SFT_main.py           # 训练主程序（参考代码）
├── SFT_infer.py          # 推理程序（参考代码）
└── ...
```

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone git@github.com:buzhimingLF/ProteinMTP---Multi-Token-Prediction-for-Protein-Sequence-Generation.git
cd ProteinMTP

# 安装依赖
pip install torch transformers peft deepspeed
```

### 数据准备

1. 从 [PDBbind-Plus](https://www.pdbbind-plus.org.cn/) 下载数据集
2. 按照数据预处理流程准备训练数据

### 训练模型

```bash
# 基础训练命令（待完善）
python SFT_main.py --json_path <data_path> --resource_path <resource_path>
```

## 📊 实验进度

- [x] 确定研究方向
- [x] 获取参考资料和代码
- [ ] 阅读理解参考论文
- [ ] 数据集探索与预处理
- [ ] 代码改造（VL → 纯文本）
- [ ] 基线实验
- [ ] 正式实验
- [ ] 论文撰写

## 📝 文档

- [需求文档](./start.md) - 项目需求和技术细节
- [讨论文档](./DISCUSSION.md) - 进度跟踪和问题讨论

## 🤝 贡献

欢迎提出问题和建议！请通过 GitHub Issues 进行交流。

## 📄 License

MIT License

---

*本项目为毕业论文研究项目*
