#!/bin/bash

# 设置参数
DEVICE="cuda"  # 使用CUDA进行训练
BATCH_SIZE=16
EPOCHS=200
LEARNING_RATE=1e-3
PATIENCE=15

# 训练数据和配置文件路径
JSON_DIR="./data/json_files"
CSV_FILE="./data/effective_conductivity_results.csv"
CONFIG_FILE="./configs/training.yaml"

# 运行训练脚本
python -m src.train \
    --json_dir $JSON_DIR \
    --csv_file $CSV_FILE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --patience $PATIENCE \
    --device $DEVICE \
    --config $CONFIG_FILE