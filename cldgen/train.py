"""
训练脚本
用于训练Deep Sets模型预测热导率矩阵
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import argparse

from deep_sets_model import DeepSetsModel, DeepSetsWithAttention
from data_loader import create_dataloaders


class ConductivityLoss(nn.Module):
    """
    自定义损失函数,考虑热导率矩阵的对称性和物理约束
    """
    def __init__(self, 
                 mse_weight=1.0,
                 symmetry_weight=0.1,
                 positive_weight=0.0):
        super(ConductivityLoss, self).__init__()
        self.mse_weight = mse_weight
        self.symmetry_weight = symmetry_weight
        self.positive_weight = positive_weight
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: (batch_size, 4) [k_xx, k_xy, k_yx, k_yy]
            target: (batch_size, 4)
        """
        # MSE损失
        mse_loss = self.mse(pred, target)
        
        # 对称性损失 (k_xy应该约等于k_yx)
        k_xy = pred[:, 1]
        k_yx = pred[:, 2]
        symmetry_loss = torch.mean((k_xy - k_yx) ** 2)
        
        # 正定性约束 (对角元素应该为正)
        k_xx = pred[:, 0]
        k_yy = pred[:, 3]
        positive_loss = torch.mean(torch.relu(-k_xx)) + torch.mean(torch.relu(-k_yy))
        
        # 总损失
        total_loss = (self.mse_weight * mse_loss + 
                     self.symmetry_weight * symmetry_loss +
                     self.positive_weight * positive_loss)
        
        return total_loss, {
            'mse': mse_loss.item(),
            'symmetry': symmetry_loss.item(),
            'positive': positive_loss.item()
        }


class MaskedDeepSetsModel(nn.Module):
    """
    支持mask的Deep Sets模型包装器
    """
    def __init__(self, base_model):
        super(MaskedDeepSetsModel, self).__init__()
        self.base_model = base_model
    
    def forward(self, ellipse_features, global_features, mask):
        """
        Args:
            ellipse_features: (batch_size, max_num_ellipses, 5)
            global_features: (batch_size, 5)
            mask: (batch_size, max_num_ellipses) 1表示有效,0表示padding
        """
        # 将mask应用到椭圆特征
        mask_expanded = mask.unsqueeze(-1)  # (batch_size, max_num_ellipses, 1)
        masked_features = ellipse_features * mask_expanded
        
        # 调用基础模型
        output = self.base_model(masked_features, global_features)
        
        return output


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_mse = 0
    total_symmetry = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        ellipse_features = batch['ellipse_features'].to(device).float()
        global_features = batch['global_features'].to(device).float()
        mask = batch['mask'].to(device).float()
        k_matrix = batch['k_matrix'].to(device).float()
        
        # 前向传播
        optimizer.zero_grad()
        pred = model(ellipse_features, global_features, mask)
        
        # 计算损失
        loss, loss_dict = criterion(pred, k_matrix)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        total_mse += loss_dict['mse']
        total_symmetry += loss_dict['symmetry']
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mse': f"{loss_dict['mse']:.4f}"
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_symmetry = total_symmetry / len(dataloader)
    
    return avg_loss, avg_mse, avg_symmetry


def validate(model, dataloader, criterion, device, dataset=None):
    """验证模型"""
    model.eval()
    total_loss = 0
    total_mse = 0
    total_symmetry = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            ellipse_features = batch['ellipse_features'].to(device).float()
            global_features = batch['global_features'].to(device).float()
            mask = batch['mask'].to(device).float()
            k_matrix = batch['k_matrix'].to(device).float()
            
            # 前向传播
            pred = model(ellipse_features, global_features, mask)
            
            # 计算损失
            loss, loss_dict = criterion(pred, k_matrix)
            
            total_loss += loss.item()
            total_mse += loss_dict['mse']
            total_symmetry += loss_dict['symmetry']
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(k_matrix.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_symmetry = total_symmetry / len(dataloader)
    
    # 计算额外的评估指标
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # 反归一化
    if dataset is not None:
        all_preds = dataset.denormalize_output(all_preds)
        all_targets = dataset.denormalize_output(all_targets)
    
    # 计算RMSE和R²
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    r2 = 1 - np.sum((all_preds - all_targets) ** 2) / np.sum((all_targets - np.mean(all_targets, axis=0)) ** 2)
    
    return avg_loss, avg_mse, avg_symmetry, rmse, r2, all_preds, all_targets


def plot_predictions(preds, targets, save_path, component_names=['k_xx', 'k_xy', 'k_yx', 'k_yy']):
    """绘制预测vs真实值"""
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
        r2 = 1 - np.sum((preds[:, i] - targets[:, i]) ** 2) / np.sum((targets[:, i] - np.mean(targets[:, i])) ** 2)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                verticalalignment='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def train_model(config):
    """完整的训练流程"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        json_dir=config['json_dir'],
        csv_file=config['csv_file'],
        batch_size=config['batch_size'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio'],
        num_workers=config['num_workers'],
        random_seed=config['random_seed']
    )
    
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
    
    # 打印模型参数数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # 创建损失函数和优化器
    criterion = ConductivityLoss(
        mse_weight=config['mse_weight'],
        symmetry_weight=config['symmetry_weight'],
        positive_weight=config['positive_weight']
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # TensorBoard
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # 训练
        train_loss, train_mse, train_symmetry = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_mse, val_symmetry, val_rmse, val_r2, val_preds, val_targets = validate(
            model, val_loader, criterion, device, dataset
        )
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MSE/train', train_mse, epoch)
        writer.add_scalar('MSE/val', val_mse, epoch)
        writer.add_scalar('RMSE/val', val_rmse, epoch)
        writer.add_scalar('R2/val', val_r2, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val RMSE: {val_rmse:.4f}, Val R²: {val_r2:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_r2,
                'config': config
            }, checkpoint_dir / 'best_model.pt')
            
            # 绘制预测图
            plot_predictions(val_preds, val_targets, output_dir / 'best_predictions.png')
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # 定期保存检查点
        if (epoch + 1) % config['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    # 在测试集上评估最佳模型
    print("\n" + "="*50)
    print("Evaluating best model on test set...")
    
    # 加载最佳模型
    checkpoint = torch.load(checkpoint_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_mse, test_symmetry, test_rmse, test_r2, test_preds, test_targets = validate(
        model, test_loader, criterion, device, dataset
    )
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # 绘制测试集预测图
    plot_predictions(test_preds, test_targets, output_dir / 'test_predictions.png')
    
    # 保存测试结果
    test_results = {
        'test_loss': test_loss,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_symmetry': test_symmetry
    }
    
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    writer.close()
    print("\nTraining completed!")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Deep Sets model for conductivity prediction')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    args = parser.parse_args()
    
    # 默认配置
    config = {
        # 数据相关
        'json_dir': './generated_samples_bat/json_files',
        'csv_file': './effective_conductivity_results.csv',
        'batch_size': 16,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'num_workers': 0,
        'random_seed': 42,
        
        # 模型相关
        'model_type': 'basic',  # 'basic' or 'attention'
        'encoder_hidden_dims': [64, 128, 256],
        'decoder_hidden_dims': [256, 128, 64],
        'aggregation': 'mean_max',
        'include_global_features': True,
        'attention_heads': 4,
        
        # 训练相关
        'num_epochs': 200,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'mse_weight': 1.0,
        'symmetry_weight': 0.1,
        'positive_weight': 0.0,
        'early_stopping_patience': 30,
        'save_interval': 20,
        
        # 输出相关
        'output_dir': './outputs/deep_sets_basic'
    }
    
    # 如果提供了配置文件,覆盖默认配置
    if args.config:
        with open(args.config, 'r') as f:
            user_config = json.load(f)
            config.update(user_config)
    
    # 开始训练
    train_model(config)
