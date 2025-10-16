"""
预测脚本
使用训练好的模型进行预测
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from deep_sets_model import DeepSetsModel, DeepSetsWithAttention
from data_loader import EllipseCompositeDataset
from train import MaskedDeepSetsModel


def load_model(checkpoint_path, device='cpu'):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 检查点文件路径
        device: 设备
    
    Returns:
        model, config, dataset_params
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # 创建模型
    if config['model_type'] == 'basic':
        base_model = DeepSetsModel(
            ellipse_feature_dim=5,
            encoder_hidden_dims=config['encoder_hidden_dims'],
            decoder_hidden_dims=config['decoder_hidden_dims'],
            aggregation=config['aggregation'],
            output_dim=4,
            include_global_features=config['include_global_features']
        )
    elif config['model_type'] == 'attention':
        base_model = DeepSetsWithAttention(
            ellipse_feature_dim=5,
            encoder_hidden_dims=config['encoder_hidden_dims'],
            decoder_hidden_dims=config['decoder_hidden_dims'],
            output_dim=4,
            include_global_features=config['include_global_features'],
            attention_heads=config['attention_heads']
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    model = MaskedDeepSetsModel(base_model).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint.get('val_loss', 'N/A')}")
    if 'val_r2' in checkpoint:
        print(f"Val R²: {checkpoint['val_r2']:.4f}")
    
    return model, config


def predict_single_sample(model, sample_data, dataset, device='cpu'):
    """
    预测单个样本
    
    Args:
        model: 训练好的模型
        sample_data: 样本JSON数据
        dataset: 数据集对象(用于归一化参数)
        device: 设备
    
    Returns:
        predicted_k_matrix: 预测的热导率矩阵 [k_xx, k_xy, k_yx, k_yy]
    """
    meta = sample_data['meta']
    
    # 提取椭圆特征
    ellipse_features = []
    for ellipse in sample_data['ellipses']:
        features = [
            ellipse['x'],
            ellipse['y'],
            ellipse['a'],
            ellipse['b'],
            ellipse['theta_deg']
        ]
        ellipse_features.append(features)
    
    ellipse_features = np.array(ellipse_features, dtype=np.float32)
    num_ellipses = len(ellipse_features)
    
    # 归一化
    if dataset.normalize:
        ellipse_features = (ellipse_features - dataset.ellipse_mean) / dataset.ellipse_std
    
    # Padding
    max_num_ellipses = dataset.max_num_ellipses
    if num_ellipses < max_num_ellipses:
        padding = np.zeros((max_num_ellipses - num_ellipses, 5), dtype=np.float32)
        ellipse_features = np.vstack([ellipse_features, padding])
    
    # Mask
    mask = np.zeros(max_num_ellipses, dtype=np.float32)
    mask[:num_ellipses] = 1.0
    
    # 全局特征
    global_features = np.array([
        meta['phi'],
        meta['Lx'],
        meta['Ly'],
        meta['km'],
        meta['ki']
    ], dtype=np.float32)
    
    if dataset.normalize:
        global_features = (global_features - dataset.global_mean) / dataset.global_std
    
    # 转换为tensor并添加batch维度
    ellipse_features = torch.from_numpy(ellipse_features).unsqueeze(0).to(device)
    global_features = torch.from_numpy(global_features).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        pred = model(ellipse_features, global_features, mask)
        pred = pred.cpu().numpy()[0]
    
    # 反归一化
    if dataset.normalize:
        pred = dataset.denormalize_output(pred)
    
    return pred


def predict_from_json(checkpoint_path, json_path, csv_file, json_dir):
    """
    从JSON文件预测热导率矩阵
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model, config = load_model(checkpoint_path, device)
    
    # 创建数据集以获取归一化参数
    dataset = EllipseCompositeDataset(json_dir, csv_file, normalize=True)
    
    # 加载样本数据
    with open(json_path, 'r') as f:
        sample_data = json.load(f)
    
    # 预测
    pred_k = predict_single_sample(model, sample_data, dataset, device)
    
    # 如果CSV中有真实值,进行比较
    sample_id = sample_data['meta']['sample_id']
    df = pd.read_csv(csv_file)
    true_row = df[df['sample_id'] == sample_id]
    
    print(f"\nSample ID: {sample_id}")
    print(f"Number of ellipses: {len(sample_data['ellipses'])}")
    print(f"Volume fraction (phi): {sample_data['meta']['phi']:.4f}")
    print("\nPredicted conductivity matrix:")
    print(f"  k_xx = {pred_k[0]:.4f}")
    print(f"  k_xy = {pred_k[1]:.4f}")
    print(f"  k_yx = {pred_k[2]:.4f}")
    print(f"  k_yy = {pred_k[3]:.4f}")
    
    if not true_row.empty:
        true_k = true_row[['k_xx', 'k_xy', 'k_yx', 'k_yy']].values[0]
        print("\nTrue conductivity matrix:")
        print(f"  k_xx = {true_k[0]:.4f}")
        print(f"  k_xy = {true_k[1]:.4f}")
        print(f"  k_yx = {true_k[2]:.4f}")
        print(f"  k_yy = {true_k[3]:.4f}")
        
        error = np.abs(pred_k - true_k)
        relative_error = error / (np.abs(true_k) + 1e-8) * 100
        
        print("\nAbsolute error:")
        print(f"  Δk_xx = {error[0]:.4f} ({relative_error[0]:.2f}%)")
        print(f"  Δk_xy = {error[1]:.4f} ({relative_error[1]:.2f}%)")
        print(f"  Δk_yx = {error[2]:.4f} ({relative_error[2]:.2f}%)")
        print(f"  Δk_yy = {error[3]:.4f} ({relative_error[3]:.2f}%)")
        
        print(f"\nMean absolute error: {np.mean(error):.4f}")
        print(f"Mean relative error: {np.mean(relative_error):.2f}%")
    
    return pred_k


def batch_predict(checkpoint_path, json_dir, csv_file, output_csv=None):
    """
    批量预测所有样本
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model, config = load_model(checkpoint_path, device)
    
    # 创建数据集
    dataset = EllipseCompositeDataset(json_dir, csv_file, normalize=True)
    
    # 预测所有样本
    results = []
    
    print(f"\nPredicting {len(dataset)} samples...")
    for i in range(len(dataset)):
        sample_data = dataset.samples[i]
        pred_k = predict_single_sample(model, sample_data, dataset, device)
        
        sample_id = sample_data['meta']['sample_id']
        results.append({
            'sample_id': sample_id,
            'pred_k_xx': pred_k[0],
            'pred_k_xy': pred_k[1],
            'pred_k_yx': pred_k[2],
            'pred_k_yy': pred_k[3]
        })
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(dataset)} samples")
    
    # 创建DataFrame
    pred_df = pd.DataFrame(results)
    
    # 合并真实值
    true_df = pd.read_csv(csv_file)
    merged_df = pred_df.merge(true_df, on='sample_id', how='left')
    
    # 计算误差
    for component in ['k_xx', 'k_xy', 'k_yx', 'k_yy']:
        merged_df[f'error_{component}'] = merged_df[f'pred_{component}'] - merged_df[component]
        merged_df[f'rel_error_{component}'] = (
            np.abs(merged_df[f'error_{component}']) / (np.abs(merged_df[component]) + 1e-8) * 100
        )
    
    # 计算整体统计
    print("\n" + "="*60)
    print("Overall prediction statistics:")
    print("="*60)
    
    for component in ['k_xx', 'k_xy', 'k_yx', 'k_yy']:
        mae = np.mean(np.abs(merged_df[f'error_{component}']))
        rmse = np.sqrt(np.mean(merged_df[f'error_{component}']**2))
        mape = np.mean(merged_df[f'rel_error_{component}'])
        
        # 计算R²
        true_vals = merged_df[component].values
        pred_vals = merged_df[f'pred_{component}'].values
        r2 = 1 - np.sum((pred_vals - true_vals)**2) / np.sum((true_vals - np.mean(true_vals))**2)
        
        print(f"\n{component}:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²:   {r2:.4f}")
    
    # 保存结果
    if output_csv:
        merged_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
    
    return merged_df


def visualize_predictions(predictions_df, output_dir):
    """
    可视化预测结果
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    components = ['k_xx', 'k_xy', 'k_yx', 'k_yy']
    
    # 预测vs真实值散点图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, (ax, comp) in enumerate(zip(axes, components)):
        true_vals = predictions_df[comp].values
        pred_vals = predictions_df[f'pred_{comp}'].values
        
        ax.scatter(true_vals, pred_vals, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        
        # y=x线
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        # 计算R²
        r2 = 1 - np.sum((pred_vals - true_vals)**2) / np.sum((true_vals - np.mean(true_vals))**2)
        rmse = np.sqrt(np.mean((pred_vals - true_vals)**2))
        
        ax.set_xlabel(f'True {comp}', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'Predicted {comp}', fontsize=13, fontweight='bold')
        ax.set_title(f'{comp} Predictions', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # 添加统计信息
        textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 误差分布直方图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, (ax, comp) in enumerate(zip(axes, components)):
        errors = predictions_df[f'error_{comp}'].values
        
        ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
        ax.axvline(np.mean(errors), color='g', linestyle='-', linewidth=2, label=f'Mean = {np.mean(errors):.4f}')
        
        ax.set_xlabel(f'Prediction error ({comp})', fontsize=13, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
        ax.set_title(f'{comp} Error Distribution', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # 添加统计信息
        std = np.std(errors)
        textstr = f'Std = {std:.4f}\nMAE = {np.mean(np.abs(errors)):.4f}'
        ax.text(0.75, 0.95, textstr, transform=ax.transAxes,
                verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict conductivity using trained Deep Sets model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['single', 'batch'], default='batch',
                        help='Prediction mode: single sample or batch')
    parser.add_argument('--json_file', type=str, help='Path to single JSON file (for single mode)')
    parser.add_argument('--json_dir', type=str, default='./generated_samples_bat/json_files',
                        help='Directory containing JSON files')
    parser.add_argument('--csv_file', type=str, default='./effective_conductivity_results.csv',
                        help='Path to CSV file with true values')
    parser.add_argument('--output_csv', type=str, default='./predictions_output.csv',
                        help='Path to save predictions (for batch mode)')
    parser.add_argument('--output_dir', type=str, default='./prediction_results',
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.json_file:
            raise ValueError("--json_file is required for single mode")
        predict_from_json(args.checkpoint, args.json_file, args.csv_file, args.json_dir)
    
    elif args.mode == 'batch':
        predictions_df = batch_predict(args.checkpoint, args.json_dir, args.csv_file, args.output_csv)
        visualize_predictions(predictions_df, args.output_dir)
