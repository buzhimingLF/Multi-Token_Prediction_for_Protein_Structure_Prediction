#!/bin/bash
# 监控训练进度脚本

OUTPUT_FILE="/tmp/claude-11685/-home-haijun-huang-Multi-Token_Prediction_for_Protein_Structure_Prediction/tasks/b005c53.output"

echo "=========================================="
echo "ProteinMTP 训练监控"
echo "=========================================="
echo "训练输出文件: $OUTPUT_FILE"
echo ""

# 显示最新的训练进度
echo "=== 最新训练进度 ==="
tail -50 "$OUTPUT_FILE" | grep -E "(it/s|epoch|loss|步|完成)"

echo ""
echo "=== 进程状态 ==="
ps aux | grep "train_distmat.py" | grep -v grep | awk '{printf "PID: %s, CPU: %s%%, MEM: %s%%\n", $2, $3, $4}'

echo ""
echo "=== GPU状态 ==="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "=========================================="
echo "使用 'tail -f $OUTPUT_FILE' 查看实时输出"
echo "=========================================="
