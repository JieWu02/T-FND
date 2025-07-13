#!/bin/bash

# 两阶段训练脚本
# --first_stage_epochs: 第一阶段固定训练的epoch数
# --second_stage_epochs: 第二阶段最大epoch数（可早停）

# 清理可能残留的进程
echo "清理可能残留的Python进程..."
pkill -f "python.*main.py" 2>/dev/null || true
sleep 2

python3 main.py \
    --data weibo \
    --batch 32 \
    --gpus 4,5,6,7 \
    --extra ddp_test \
    --port 12345 \
    --first_stage_epochs 5 \
    --second_stage_epochs 50 