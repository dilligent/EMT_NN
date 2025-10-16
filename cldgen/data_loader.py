"""
数据加载和预处理模块
用于加载JSON文件和CSV文件,准备训练数据
"""

import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class EllipseCompositeDataset(Dataset):
    """
    椭圆复合材料数据集
    """
    def __init__(self, 
                 json_dir,
                 csv_file,
                 max_num_ellipses=None,
                 normalize=True):
        """
        Args:
            json_dir: JSON文件所在目录
            csv_file: CSV文件路径,包含热导率数据
            max_num_ellipses: 最大椭圆数量(用于padding)
            normalize: 是否归一化特征
        """
        self.json_dir = Path(json_dir)
        self.normalize = normalize
        
        # 读取CSV文件
        self.conductivity_df = pd.read_csv(csv_file)
        self.sample_ids = self.conductivity_df['sample_id'].tolist()
        
        # 加载所有样本数据
        self.samples = []
        self.valid_indices = []
        
        for idx, sample_id in enumerate(self.sample_ids):
            json_path = self.json_dir / f"{sample_id}.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    self.samples.append(data)
                    self.valid_indices.append(idx)
            else:
                print(f"Warning: {json_path} not found, skipping...")
        
        # 过滤CSV数据,只保留有效样本
        self.conductivity_df = self.conductivity_df.iloc[self.valid_indices].reset_index(drop=True)
        
        # 确定最大椭圆数量
        if max_num_ellipses is None:
            self.max_num_ellipses = max(len(sample['ellipses']) for sample in self.samples)
        else:
            self.max_num_ellipses = max_num_ellipses
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"Max number of ellipses: {self.max_num_ellipses}")
        
        # 计算归一化参数
        if self.normalize:
            self._compute_normalization_params()
    
    def _compute_normalization_params(self):
        """计算归一化参数"""
        all_ellipse_features = []
        all_global_features = []
        
        for sample in self.samples:
            # 椭圆特征
            for ellipse in sample['ellipses']:
                all_ellipse_features.append([
                    ellipse['x'],
                    ellipse['y'],
                    ellipse['a'],
                    ellipse['b'],
                    ellipse['theta_deg']
                ])
            
            # 全局特征
            meta = sample['meta']
            all_global_features.append([
                meta['phi'],
                meta['Lx'],
                meta['Ly'],
                meta['km'],
                meta['ki']
            ])
        
        # 椭圆特征归一化参数
        self.ellipse_mean = np.mean(all_ellipse_features, axis=0)
        self.ellipse_std = np.std(all_ellipse_features, axis=0) + 1e-8
        
        # 全局特征归一化参数
        self.global_mean = np.mean(all_global_features, axis=0)
        self.global_std = np.std(all_global_features, axis=0) + 1e-8
        
        # 输出归一化参数 (热导率矩阵)
        k_values = self.conductivity_df[['k_xx', 'k_xy', 'k_yx', 'k_yy']].values
        self.k_mean = np.mean(k_values, axis=0)
        self.k_std = np.std(k_values, axis=0) + 1e-8
        
        print(f"Ellipse features - Mean: {self.ellipse_mean}, Std: {self.ellipse_std}")
        print(f"Global features - Mean: {self.global_mean}, Std: {self.global_std}")
        print(f"Conductivity - Mean: {self.k_mean}, Std: {self.k_std}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        返回一个样本
        """
        sample = self.samples[idx]
        meta = sample['meta']
        
        # 提取椭圆特征
        ellipse_features = []
        for ellipse in sample['ellipses']:
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
        
        # 归一化椭圆特征
        if self.normalize:
            ellipse_features = (ellipse_features - self.ellipse_mean) / self.ellipse_std
        
        # Padding到最大椭圆数量
        if num_ellipses < self.max_num_ellipses:
            padding = np.zeros((self.max_num_ellipses - num_ellipses, 5), dtype=np.float32)
            ellipse_features = np.vstack([ellipse_features, padding])
        
        # 创建mask标记有效椭圆
        mask = np.zeros(self.max_num_ellipses, dtype=np.float32)
        mask[:num_ellipses] = 1.0
        
        # 全局特征
        global_features = np.array([
            meta['phi'],
            meta['Lx'],
            meta['Ly'],
            meta['km'],
            meta['ki']
        ], dtype=np.float32)
        
        if self.normalize:
            global_features = (global_features - self.global_mean) / self.global_std
        
        # 目标热导率矩阵
        row = self.conductivity_df.iloc[idx]
        k_matrix = np.array([
            row['k_xx'],
            row['k_xy'],
            row['k_yx'],
            row['k_yy']
        ], dtype=np.float32)
        
        if self.normalize:
            k_matrix = (k_matrix - self.k_mean) / self.k_std
        
        return {
            'ellipse_features': torch.from_numpy(ellipse_features),
            'global_features': torch.from_numpy(global_features),
            'mask': torch.from_numpy(mask),
            'k_matrix': torch.from_numpy(k_matrix),
            'sample_id': meta['sample_id'],
            'num_ellipses': num_ellipses
        }
    
    def denormalize_output(self, k_matrix_normalized):
        """
        反归一化预测的热导率矩阵
        
        Args:
            k_matrix_normalized: (batch_size, 4) 或 (4,) 归一化的热导率矩阵
        Returns:
            k_matrix: 反归一化后的热导率矩阵
        """
        if not self.normalize:
            return k_matrix_normalized
        
        if isinstance(k_matrix_normalized, torch.Tensor):
            k_matrix_normalized = k_matrix_normalized.cpu().numpy()
        
        return k_matrix_normalized * self.k_std + self.k_mean


def create_dataloaders(json_dir, 
                       csv_file, 
                       batch_size=16,
                       train_ratio=0.8,
                       val_ratio=0.1,
                       test_ratio=0.1,
                       num_workers=0,
                       random_seed=42):
    """
    创建训练、验证和测试数据加载器
    
    Args:
        json_dir: JSON文件目录
        csv_file: CSV文件路径
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        num_workers: 数据加载器的工作进程数
        random_seed: 随机种子
    
    Returns:
        train_loader, val_loader, test_loader, dataset
    """
    # 创建完整数据集
    dataset = EllipseCompositeDataset(json_dir, csv_file, normalize=True)
    
    # 划分数据集
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # 设置随机种子
    torch.manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    return train_loader, val_loader, test_loader, dataset


if __name__ == "__main__":
    # 测试数据加载器
    json_dir = "./generated_samples_bat/json_files"
    csv_file = "./effective_conductivity_results.csv"
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        json_dir=json_dir,
        csv_file=csv_file,
        batch_size=4,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    # 测试一个批次
    print("\nTesting data loader...")
    for batch in train_loader:
        print(f"Ellipse features shape: {batch['ellipse_features'].shape}")
        print(f"Global features shape: {batch['global_features'].shape}")
        print(f"Mask shape: {batch['mask'].shape}")
        print(f"K matrix shape: {batch['k_matrix'].shape}")
        print(f"Sample IDs: {batch['sample_id'][:3]}")
        print(f"Number of ellipses: {batch['num_ellipses'][:3]}")
        
        # 测试反归一化
        k_denorm = dataset.denormalize_output(batch['k_matrix'][:1])
        print(f"Denormalized k_matrix: {k_denorm}")
        break
