# Deep Sets 热导率预测模型

## 项目简介

本项目实现了一个基于Deep Sets架构的神经网络模型,用于预测矩形区域内掺杂椭圆粒子的复合材料的等效热导率矩阵。

### 模型特点

- **置换不变性**: Deep Sets架构天然适合处理集合数据(椭圆粒子),对粒子顺序不敏感
- **支持变长输入**: 可以处理不同数量的椭圆粒子
- **物理约束**: 损失函数包含对称性约束,确保热导率矩阵的物理合理性
- **两种架构**: 提供基础版本和带注意力机制的增强版本

### 输入输出

**输入**:
- 椭圆粒子集合: 每个椭圆由5个参数描述 (x, y, a, b, theta_deg)
- 全局参数: 体积分数(phi)、矩形尺寸(Lx, Ly)、基质热导率(km)、夹杂物热导率(ki)

**输出**:
- 等效热导率矩阵的4个分量: [k_xx, k_xy, k_yx, k_yy]

## 文件结构

```
EMT_NN/
├── deep_sets_model.py          # Deep Sets模型定义
├── data_loader.py              # 数据加载和预处理
├── train.py                    # 训练脚本
├── predict.py                  # 预测脚本
├── configs_example.md          # 配置文件示例
├── requirements.txt            # 依赖包列表
├── generated_samples_bat/      # 样本数据目录
│   └── json_files/            # JSON格式的样本文件
└── effective_conductivity_results.csv  # 热导率真实值
```

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖:
- torch >= 1.10.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- tqdm >= 4.62.0
- tensorboard >= 2.7.0

## 快速开始

### 1. 训练模型

**使用默认配置训练基础模型**:
```bash
python train.py
```

**使用自定义配置文件**:
```bash
python train.py --config config_basic.json
```

**训练注意力模型**:
```bash
python train.py --config config_attention.json
```

### 2. 查看训练进度

使用TensorBoard查看训练曲线:
```bash
tensorboard --logdir outputs/deep_sets_basic/tensorboard
```

### 3. 预测

**批量预测所有样本**:
```bash
python predict.py --checkpoint outputs/deep_sets_basic/checkpoints/best_model.pt --mode batch
```

**预测单个样本**:
```bash
python predict.py --checkpoint outputs/deep_sets_basic/checkpoints/best_model.pt --mode single --json_file generated_samples_bat/json_files/sample_0aecc93d.json
```

## 模型架构

### 基础Deep Sets模型

```
输入椭圆集合 (N, 5)
    ↓
编码器 (独立处理每个椭圆)
    [Linear(5, 64) → ReLU → BatchNorm]
    [Linear(64, 128) → ReLU → BatchNorm]
    [Linear(128, 256) → ReLU → BatchNorm]
    ↓
编码特征 (N, 256)
    ↓
聚合器 (置换不变)
    Mean + Max pooling
    ↓
聚合特征 (512,)
    ↓
拼接全局特征 (5,)
    ↓
解码器
    [Linear(517, 256) → ReLU → Dropout]
    [Linear(256, 128) → ReLU → Dropout]
    [Linear(128, 64) → ReLU → Dropout]
    [Linear(64, 4)]
    ↓
输出: [k_xx, k_xy, k_yx, k_yy]
```

### 注意力增强模型

在基础模型的基础上,在聚合步骤前添加了多头自注意力机制:

```
编码特征 (N, 256)
    ↓
多头自注意力 (4 heads)
    ↓
注意力加权特征 (N, 256)
    ↓
Mean pooling
    ↓
后续与基础模型相同
```

## 配置参数说明

### 数据相关
- `json_dir`: JSON文件目录
- `csv_file`: CSV文件路径
- `batch_size`: 批次大小
- `train_ratio`, `val_ratio`, `test_ratio`: 数据集划分比例

### 模型相关
- `model_type`: 模型类型 ("basic" 或 "attention")
- `encoder_hidden_dims`: 编码器隐藏层维度列表
- `decoder_hidden_dims`: 解码器隐藏层维度列表
- `aggregation`: 聚合方式 ("mean", "sum", "max", "mean_max")
- `include_global_features`: 是否包含全局特征
- `attention_heads`: 注意力头数(仅用于attention模型)

### 训练相关
- `num_epochs`: 训练轮数
- `learning_rate`: 学习率
- `weight_decay`: L2正则化系数
- `mse_weight`: MSE损失权重
- `symmetry_weight`: 对称性约束权重
- `positive_weight`: 正定性约束权重
- `early_stopping_patience`: 早停耐心值
- `save_interval`: 保存间隔

## 损失函数

模型使用复合损失函数:

```python
Loss = mse_weight × MSE(pred, target) 
     + symmetry_weight × (k_xy - k_yx)²
     + positive_weight × ReLU(-k_xx) + ReLU(-k_yy)
```

- **MSE损失**: 基础预测误差
- **对称性约束**: 强制k_xy ≈ k_yx
- **正定性约束**: 确保对角元素为正

## 性能评估指标

- **MAE** (Mean Absolute Error): 平均绝对误差
- **RMSE** (Root Mean Square Error): 均方根误差
- **MAPE** (Mean Absolute Percentage Error): 平均绝对百分比误差
- **R²** (R-squared): 决定系数

## 实验建议

### 超参数调优

1. **学习率**: 从0.001开始,根据训练曲线调整
2. **批次大小**: 根据GPU内存调整(8, 16, 32)
3. **模型深度**: 可以尝试更深或更浅的网络
4. **聚合方式**: 比较mean, max, mean_max的效果
5. **对称性权重**: 0.1-1.0之间调整

### 数据增强

可以考虑:
- 旋转数据增强(旋转整个椭圆集合)
- 添加噪声
- 随机删除部分椭圆

### 模型改进方向

1. **图神经网络**: 考虑椭圆之间的空间关系
2. **Transformer**: 使用完整的Transformer架构
3. **物理引导**: 加入更多物理约束
4. **多任务学习**: 同时预测其他物理量

## 示例结果

训练后的模型通常可以达到:
- R² > 0.95
- MAPE < 5%

具体性能取决于数据质量和模型配置。

## 故障排除

### 内存不足
- 减小batch_size
- 减小模型隐藏层维度
- 设置num_workers=0

### 训练不稳定
- 降低学习率
- 增加weight_decay
- 检查数据归一化

### 过拟合
- 增加Dropout比例
- 增加weight_decay
- 减少模型复杂度
- 使用更多训练数据

## 引用

如果使用本代码,请引用Deep Sets原始论文:

```
@inproceedings{zaheer2017deep,
  title={Deep sets},
  author={Zaheer, Manzil and Kottur, Satwik and Ravanbakhsh, Siamak and Poczos, Barnabas and Salakhutdinov, Russ R and Smola, Alexander J},
  booktitle={Advances in neural information processing systems},
  pages={3391--3401},
  year={2017}
}
```

## 许可证

请参考项目LICENSE文件。

## 联系方式

如有问题,请提交Issue或联系项目维护者。
