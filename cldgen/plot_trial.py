import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 提取的核心绘图函数 (与原代码完全一致)
# ==========================================

def plot_predictions(preds, targets, save_path, component_names=['k_xx', 'k_xy', 'k_yx', 'k_yy']):
    """
    绘制预测vs真实值
    (代码源自 train.py，保持完全一致的视觉风格)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, component_names)):
        ax.scatter(targets[:, i], preds[:, i], alpha=0.5, s=20)
        
        # 绘制y=x参考线
        min_val = min(targets[:, i].min(), preds[:, i].min())
        max_val = max(targets[:, i].max(), preds[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_xlabel(f'True {name}', fontsize=12)
        ax.set_ylabel(f'Predicted {name}', fontsize=12)
        ax.set_title(f'{name} Prediction', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 计算R²
        # 防止分母为0的极微小保护
        denominator = np.sum((targets[:, i] - np.mean(targets[:, i])) ** 2)
        if denominator < 1e-9: denominator = 1e-9
            
        r2 = 1 - np.sum((preds[:, i] - targets[:, i]) ** 2) / denominator
        
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                verticalalignment='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {save_path}")

# ==========================================
# 2. 仿真数据生成器
# ==========================================

def generate_synthetic_results(num_samples=40, quality='excellent'):
    """
    生成模拟的真实值和预测值
    
    Args:
        quality: 
            'excellent' (很少噪声, R2 > 0.95)
            'good'      (少量噪声, R2 ~ 0.85)
            'average'   (中等噪声, R2 ~ 0.60)
            'poor'      (大量噪声, R2 ~ 0.30)
            'very_poor' (几乎随机或仅预测均值, R2 ~ 0 或负数)
    """
    np.random.seed(42) # 固定随机种子以保证基准数据一致
    
    # 1. 生成物理上合理的真实值 (Targets)
    # k_xx, k_yy 通常为正值 (例如 2.0 到 10.0)
    t_xx = np.random.uniform(-0.4, 0.6, num_samples)
    t_yy = np.random.uniform(-0.4, 0.8, num_samples)
    
    # k_xy, k_yx 通常较小且对称 (例如 -2.0 到 2.0)
    t_xy = np.random.uniform(-0.6, 0.6, num_samples)
    t_yx = np.random.uniform(-0.5, 0.6, num_samples)
    
    targets = np.stack([t_xx, t_xy, t_yx, t_yy], axis=1)
    
    # 2. 根据质量设定噪声参数
    if quality == 'excellent':
        noise_std = 0.15
        bias_factor = 1.0  # 无缩放偏差
        bias_shift = 0.0
    elif quality == 'good':
        noise_std = 0.2
        bias_factor = 0.98
        bias_shift = 0.02
    elif quality == 'average':
        noise_std = 1.5
        bias_factor = 0.9
        bias_shift = 0.2
    elif quality == 'poor':
        noise_std = 2.8
        bias_factor = 0.7  # 模型开始欠拟合，斜率偏离
        bias_shift = 1.0
    elif quality == 'very_poor':
        noise_std = 5.0
        bias_factor = 0.2  # 模型几乎失效
        bias_shift = 3.0
    else:
        raise ValueError("Unknown quality level")

    # 3. 生成预测值 (Preds = Targets * scale + noise + shift)
    # 使用不同的随机种子生成噪声，避免与Targets完全线性相关
    # rng = np.random.default_rng(seed=123 + len(quality))
    rng = np.random.default_rng()
    noise = rng.normal(0, noise_std, targets.shape)
    
    preds = targets * bias_factor + noise + bias_shift
    
    # 针对 'very_poor' 的特殊处理：模拟只学会了平均值的情况，但这会使图像变成水平线
    # 这里我们保留上面的强噪声模式，看起来更像训练发散或没收敛
    
    return preds, targets

# ==========================================
# 3. 主程序：生成5张示意图
# ==========================================

if __name__ == "__main__":
    output_dir = "./simulation_results"
    
    # 定义5个等级
    scenarios = [
        ("result_1_excellent.png", "excellent"),  # 很好
        ("result_2_good.png",      "good"),       # 较好
        ("result_3_average.png",   "average"),    # 一般
        ("result_4_poor.png",      "poor"),       # 较差
        ("result_5_very_poor.png", "very_poor")   # 很差
    ]
    
    print("开始生成示意图...")
    
    for filename, quality in scenarios:
        # 生成数据
        preds, targets = generate_synthetic_results(num_samples=40, quality=quality)
        
        # 绘图 save path
        save_path = os.path.join(output_dir, filename)
        
        # 调用提取的绘图函数
        plot_predictions(preds, targets, save_path)
        
    print(f"\n全部完成！结果已保存在 {output_dir} 文件夹下。")
