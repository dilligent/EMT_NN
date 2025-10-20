import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path

# ==================== 数据集类 ====================
class EllipseDataset(Dataset):
    """从JSON和CSV文件加载椭圆结构和热导率数据"""
    
    def __init__(self, json_dir, csv_file, device='cuda'):
        self.device = device
        self.data = []
        
        # 读取CSV文件获取热导率
        df = pd.read_csv(csv_file)
        self.conductivity_dict = {}
        for _, row in df.iterrows():
            sample_id = row['sample_id']
            k_matrix = np.array([
                [row['k_xx'], row['k_xy']],
                [row['k_yx'], row['k_yy']]
            ], dtype=np.float32)
            self.conductivity_dict[sample_id] = k_matrix
        
        # 读取JSON文件获取椭圆数据
        json_files = sorted(Path(json_dir).glob('*.json'))
        
        for json_file in json_files:
            sample_id = json_file.stem
            
            if sample_id not in self.conductivity_dict:
                continue
                
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # 提取椭圆特征
            ellipse_features = []
            for ellipse in data['ellipses']:
                features = [
                    ellipse['x'],
                    ellipse['y'],
                    ellipse['a'],
                    ellipse['b'],
                    np.cos(np.radians(ellipse['theta_deg'])),
                    np.sin(np.radians(ellipse['theta_deg']))
                ]
                ellipse_features.append(features)
            
            ellipse_features = np.array(ellipse_features, dtype=np.float32)
            conductivity = self.conductivity_dict[sample_id]
            phi = data['meta']['phi']
            
            self.data.append({
                'sample_id': sample_id,
                'ellipse_features': ellipse_features,
                'conductivity': conductivity,
                'phi': phi
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'sample_id': item['sample_id'],
            'ellipse_features': torch.from_numpy(item['ellipse_features']),
            'conductivity': torch.from_numpy(item['conductivity']),
            'phi': torch.tensor(item['phi'], dtype=torch.float32)
        }


# ==================== Sets Transformer 模型 ====================
class SetTransformer(nn.Module):
    """Sets Transformer模型"""
    
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=4, num_heads=4):
        super(SetTransformer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, ellipse_features, phi):
        """
        Args:
            ellipse_features: (batch_size, num_ellipses, 6)
            phi: (batch_size,)
        
        Returns:
            conductivity: (batch_size, 4)
        """
        batch_size, num_ellipses, _ = ellipse_features.shape
        
        # 编码
        encoded = self.encoder(ellipse_features)
        
        # 进行自注意力
        attention_output, _ = self.attention(encoded, encoded, encoded)
        
        # 解码
        aggregated = torch.mean(attention_output, dim=1)  # 聚合
        phi_expanded = phi.unsqueeze(1)
        combined = torch.cat([aggregated, phi_expanded], dim=1)
        conductivity = self.decoder(combined)
        
        return conductivity


# ==================== Early Stopping ====================
class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=15, min_delta=1e-4, path='best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.path = path
        self.best_epoch = 0
    
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
        
        return self.counter >= self.patience


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        ellipse_features = batch['ellipse_features'].to(device)
        conductivity = batch['conductivity'].to(device)
        phi = batch['phi'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        predictions = model(ellipse_features, phi)
        
        # 计算损失
        loss = criterion(predictions, conductivity)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            ellipse_features = batch['ellipse_features'].to(device)
            conductivity = batch['conductivity'].to(device)
            phi = batch['phi'].to(device)
            
            predictions = model(ellipse_features, phi)
            loss = criterion(predictions, conductivity)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train_model(model, train_loader, val_loader, epochs=200, learning_rate=1e-3, 
                device='cuda', patience=15):
    """完整的训练循环，包含早停机制"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    early_stopping = EarlyStopping(patience=patience, path='best_sets_transformer_model.pt')
    
    train_losses = []
    val_losses = []
    
    print(f"开始训练 (设备: {device})")
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f}")
        
        # 早停检查
        if early_stopping(val_loss, epoch):
            print(f"\n早停触发 (Epoch {epoch+1})")
            print(f"最佳验证损失: {early_stopping.best_loss:.6f} (Epoch {early_stopping.best_epoch+1})")
            break
    
    # 加载最佳模型
    if os.path.exists(early_stopping.path):
        model.load_state_dict(torch.load(early_stopping.path))
        print(f"已加载最佳模型")
    
    return model, train_losses, val_losses