# Deep Sets 模型使用指南

## 快速开始 (3步)

### 第1步: 安装依赖

```bash
pip install -r requirements.txt
```

### 第2步: 测试模型

运行测试脚本确保一切正常:

```bash
python test_model.py
```

如果看到 "✓ All tests passed successfully!" 说明环境配置正确。

### 第3步: 开始训练

**方式1: 使用默认配置**
```bash
python train.py
```

**方式2: 使用预设配置**
```bash
# 训练基础模型
python train.py --config config_basic.json

# 训练注意力模型
python train.py --config config_attention.json
```

训练完成后,模型和结果会保存在 `outputs/` 目录下。

---

## 详细使用说明

### 一、训练模型

#### 1.1 查看训练进度

训练时会显示进度条和实时loss:

```
Epoch 1/200
Training: 100%|██████████| 10/10 [00:05<00:00,  2.00it/s, loss=0.1234, mse=0.0987]
Validating: 100%|██████████| 3/3 [00:01<00:00,  3.00it/s]
Train Loss: 0.1234, Val Loss: 0.0987
Val RMSE: 0.0543, Val R²: 0.9234
```

#### 1.2 使用TensorBoard监控

在新的终端窗口中运行:

```bash
tensorboard --logdir outputs/deep_sets_basic/tensorboard
```

然后在浏览器中打开 http://localhost:6006 查看:
- 训练/验证损失曲线
- RMSE和R²指标
- 学习率变化

#### 1.3 训练输出说明

训练完成后,在 `outputs/deep_sets_basic/` 目录下会生成:

```
outputs/deep_sets_basic/
├── checkpoints/
│   ├── best_model.pt              # 最佳模型
│   ├── checkpoint_epoch_20.pt     # 定期检查点
│   └── checkpoint_epoch_40.pt
├── tensorboard/                   # TensorBoard日志
├── config.json                    # 训练配置
├── test_results.json              # 测试集结果
├── best_predictions.png           # 验证集预测图
└── test_predictions.png           # 测试集预测图
```

---

### 二、模型预测

#### 2.1 批量预测所有样本

```bash
python predict.py \
  --checkpoint outputs/deep_sets_basic/checkpoints/best_model.pt \
  --mode batch \
  --output_csv predictions.csv \
  --output_dir prediction_results
```

这会生成:
- `predictions.csv`: 所有样本的预测结果
- `prediction_results/predictions_scatter.png`: 预测vs真实值散点图
- `prediction_results/error_distribution.png`: 误差分布直方图

#### 2.2 预测单个样本

```bash
python predict.py \
  --checkpoint outputs/deep_sets_basic/checkpoints/best_model.pt \
  --mode single \
  --json_file generated_samples_bat/json_files/sample_0aecc93d.json
```

输出示例:
```
Sample ID: sample_0aecc93d
Number of ellipses: 137
Volume fraction (phi): 0.3515

Predicted conductivity matrix:
  k_xx = 5.2145
  k_xy = -5.1892
  k_yx = -5.1421
  k_yy = 5.2734

True conductivity matrix:
  k_xx = 5.2095
  k_xy = -5.1874
  k_yx = -5.1387
  k_yy = 5.2700

Absolute error:
  Δk_xx = 0.0050 (0.10%)
  Δk_xy = 0.0018 (0.03%)
  Δk_yx = 0.0034 (0.07%)
  Δk_yy = 0.0034 (0.06%)

Mean absolute error: 0.0034
Mean relative error: 0.07%
```

---

### 三、配置文件说明

创建自定义配置文件 `my_config.json`:

```json
{
  "batch_size": 32,              // 增大批次大小
  "learning_rate": 0.0005,       // 降低学习率
  "num_epochs": 300,             // 延长训练时间
  "encoder_hidden_dims": [128, 256, 512],  // 更大的模型
  "symmetry_weight": 0.5,        // 增强对称性约束
  "output_dir": "./outputs/my_experiment"
}
```

然后运行:
```bash
python train.py --config my_config.json
```

**重要参数**:

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `batch_size` | 批次大小 | 16-32 |
| `learning_rate` | 学习率 | 0.0001-0.001 |
| `encoder_hidden_dims` | 编码器维度 | [64,128,256] |
| `aggregation` | 聚合方式 | "mean_max" |
| `symmetry_weight` | 对称性权重 | 0.1-1.0 |
| `early_stopping_patience` | 早停耐心 | 20-50 |

---

### 四、常见问题

#### Q1: 内存不足 (Out of Memory)

**解决方案**:
1. 减小batch_size: `"batch_size": 8`
2. 减小模型尺寸: `"encoder_hidden_dims": [32, 64, 128]`
3. 在CPU上训练(较慢): 设置 `CUDA_VISIBLE_DEVICES=""`

#### Q2: 训练很慢

**解决方案**:
1. 确保使用GPU: 检查是否安装了CUDA版本的PyTorch
2. 增大batch_size: `"batch_size": 32`
3. 减少num_workers(Windows): `"num_workers": 0`

#### Q3: 验证loss不下降

**解决方案**:
1. 降低学习率: `"learning_rate": 0.0001`
2. 检查数据是否正确加载
3. 尝试不同的aggregation方式
4. 增加模型容量

#### Q4: 过拟合(训练loss低,验证loss高)

**解决方案**:
1. 增加weight_decay: `"weight_decay": 0.0001`
2. 增加更多训练数据
3. 减小模型复杂度
4. 使用更强的Dropout

---

### 五、评估模型性能

#### 5.1 查看测试结果

```bash
cat outputs/deep_sets_basic/test_results.json
```

示例输出:
```json
{
  "test_loss": 0.0234,
  "test_mse": 0.0198,
  "test_rmse": 0.1407,
  "test_r2": 0.9567,
  "test_symmetry": 0.0012
}
```

#### 5.2 性能指标解释

- **R² > 0.95**: 优秀 ✓
- **R² > 0.90**: 良好
- **R² > 0.80**: 可接受
- **R² < 0.80**: 需要改进

- **MAPE < 2%**: 优秀 ✓
- **MAPE < 5%**: 良好
- **MAPE < 10%**: 可接受

---

### 六、高级用法

#### 6.1 比较不同模型

训练多个模型:
```bash
python train.py --config config_basic.json
python train.py --config config_attention.json
```

比较结果:
```python
import json

with open('outputs/deep_sets_basic/test_results.json') as f:
    basic_results = json.load(f)

with open('outputs/deep_sets_attention/test_results.json') as f:
    attention_results = json.load(f)

print(f"Basic R²: {basic_results['test_r2']:.4f}")
print(f"Attention R²: {attention_results['test_r2']:.4f}")
```

#### 6.2 自定义损失函数

编辑 `train.py` 中的 `ConductivityLoss` 类,添加新的约束项。

#### 6.3 数据分析

```python
import pandas as pd
import matplotlib.pyplot as plt

# 加载预测结果
df = pd.read_csv('predictions.csv')

# 分析误差与体积分数的关系
plt.scatter(df['phi'], df['error_k_xx'])
plt.xlabel('Volume Fraction (phi)')
plt.ylabel('Error in k_xx')
plt.show()
```

---

### 七、实验建议

#### 7.1 第一次训练

使用默认配置先跑一个baseline:
```bash
python train.py
```

等待训练完成,检查结果。

#### 7.2 超参数调优

依次调整以下参数:
1. 学习率: [0.0001, 0.0005, 0.001, 0.005]
2. 批次大小: [8, 16, 32]
3. 模型深度: 浅 → 中 → 深
4. 聚合方式: mean, max, mean_max

#### 7.3 模型对比

- 基础Deep Sets vs 注意力Deep Sets
- 包含全局特征 vs 不包含全局特征
- 不同对称性权重的影响

---

## 预期结果

在200个样本的数据集上,经过适当训练,模型应该达到:

- **R²**: 0.92 - 0.97
- **RMSE**: 0.10 - 0.20
- **MAPE**: 2% - 5%
- **训练时间**: 2-10分钟(取决于硬件)

---

## 下一步

1. ✓ 完成基础训练
2. ✓ 分析预测结果
3. 尝试不同配置
4. 优化模型性能
5. 应用到新数据

---

## 获取帮助

遇到问题时:
1. 运行 `python test_model.py` 检查环境
2. 查看 `README_DeepSets.md` 详细文档
3. 检查TensorBoard训练曲线
4. 提交Issue到GitHub仓库

祝训练顺利! 🚀
