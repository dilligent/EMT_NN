import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
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


# ==================== Deep Sets 模型 ====================
class DeepSetsEncoder(nn.Module):
    """Deep Sets编码器 - 处理集合中的单个元素"""
    
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=128):
        super(DeepSetsEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)


class DeepSetsAggregator(nn.Module):
    """Deep Sets聚合器 - 对编码后的特征进行聚合"""
    
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=256):
        super(DeepSetsAggregator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # input_dim*2: cat(mean, max)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, encoded_features, mask=None):
        # encoded_features: (batch_size, num_ellipses, encoded_dim)
        
        if mask is not None:
            # mask: (batch_size, num_ellipses)
            mask = mask.to(encoded_features.device)
            mask_expanded = mask.unsqueeze(-1).float()
            
            # Masked Mean
            # 1. Zero out padded positions
            masked_features = encoded_features * mask_expanded
            # 2. Sum valid features
            sum_features = torch.sum(masked_features, dim=1)
            # 3. Count valid items
            count = torch.sum(mask_expanded, dim=1)
            count = torch.clamp(count, min=1.0) # Avoid div by zero
            mean_pooled = sum_features / count
            
            # Masked Max
            # Replace padding with very small number so they don't affect max
            features_for_max = encoded_features.clone()
            features_for_max[~mask] = -1e9
            max_pooled, _ = torch.max(features_for_max, dim=1)
        else:
            mean_pooled = torch.mean(encoded_features, dim=1)
            max_pooled, _ = torch.max(encoded_features, dim=1)
        
        aggregated = torch.cat([mean_pooled, max_pooled], dim=1)
        return self.mlp(aggregated)


class EllipseToThermalDeepSets(nn.Module):
    """完整的Deep Sets模型：椭圆结构 -> 热导率矩阵"""
    
    def __init__(self, encoder_hidden=128, encoder_output=128, 
                 aggregator_hidden=256, aggregator_output=256):
        super(EllipseToThermalDeepSets, self).__init__()
        
        # 编码器
        self.encoder = DeepSetsEncoder(
            input_dim=6,
            hidden_dim=encoder_hidden,
            output_dim=encoder_output
        )
        
        # 聚合器
        self.aggregator = DeepSetsAggregator(
            input_dim=encoder_output,
            hidden_dim=aggregator_hidden,
            output_dim=aggregator_output
        )
        
        # 解码器 - 输出4个热导率分量 (k_xx, k_xy, k_yx, k_yy)
        self.decoder = nn.Sequential(
            nn.Linear(aggregator_output + 1, aggregator_hidden),  # +1 for phi
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(aggregator_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # k_xx, k_xy, k_yx, k_yy
        )
    
    def forward(self, ellipse_features, phi, mask=None):
        """
        Args:
            ellipse_features: (batch_size, num_ellipses, 6)
            phi: (batch_size,)
            mask: (batch_size, num_ellipses) - Optional
        
        Returns:
            conductivity: (batch_size, 4)
        """
        # 编码每个椭圆
        batch_size, num_ellipses, _ = ellipse_features.shape
        ellipse_flat = ellipse_features.reshape(-1, 6)
        encoded = self.encoder(ellipse_flat)
        encoded = encoded.reshape(batch_size, num_ellipses, -1)
        
        # 聚合
        aggregated = self.aggregator(encoded, mask)
        
        # 解码
        phi_expanded = phi.unsqueeze(1)
        combined = torch.cat([aggregated, phi_expanded], dim=1)
        conductivity = self.decoder(combined)
        
        return conductivity


# ==================== 培训函数 ====================
class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=50, min_delta=1e-4, path='best_model.pt'):
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


def collate_fn(batch):
    """自定义collate函数，处理变长序列"""
    sample_ids = [item['sample_id'] for item in batch]
    conductivity = torch.stack([item['conductivity'] for item in batch])
    phi = torch.stack([item['phi'] for item in batch])
    
    # 处理变长的 ellipse_features
    features_list = [item['ellipse_features'] for item in batch]
    
    # 填充序列 (batch, max_len, feat)
    padded_features = torch.nn.utils.rnn.pad_sequence(features_list, batch_first=True, padding_value=0.0)
    
    # 创建掩码 (batch, max_len) - 真实数据为True，填充为False
    lengths = torch.tensor([len(f) for f in features_list])
    max_len = padded_features.shape[1]
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    
    return {
        'sample_id': sample_ids,
        'ellipse_features': padded_features,
        'conductivity': conductivity,
        'phi': phi,
        'mask': mask
    }


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        ellipse_features = batch['ellipse_features'].to(device)
        conductivity = batch['conductivity'].to(device)
        phi = batch['phi'].to(device)
        mask = batch['mask'].to(device)
        
        # 将conductivity矩阵展平为向量 (batch_size, 4)
        batch_size = conductivity.shape[0]
        conductivity_vec = conductivity.reshape(batch_size, 4)
        
        # 前向传播
        optimizer.zero_grad()
        predictions = model(ellipse_features, phi, mask)
        
        # 计算损失
        loss = criterion(predictions, conductivity_vec)
        
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
            mask = batch['mask'].to(device)
            
            batch_size = conductivity.shape[0]
            conductivity_vec = conductivity.reshape(batch_size, 4)
            
            predictions = model(ellipse_features, phi, mask)
            loss = criterion(predictions, conductivity_vec)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train_model(model, train_loader, val_loader, epochs=200, learning_rate=1e-3, 
                device='cuda', patience=50):
    """完整的训练循环，包含早停机制"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # 修复：删除了 verbose=True 参数，因为它在较新版本的 PyTorch 中会导致 TypeError
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )
    
    early_stopping = EarlyStopping(patience=patience, path='best_deepsets_model.pt')
    
    train_losses = []
    val_losses = []
    
    print(f"开始训练 (设备: {device})")
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print("-" * 60)
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 手动检查并打印学习率变化（替代 verbose=True 的功能）
        last_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if current_lr != last_lr:
            print(f"Epoch {epoch+1}: 学习率已调整为 {current_lr:.6f}")
        
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


# ==================== 主程序 ====================
def main():
    # 设置参数
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {DEVICE}")
    
    JSON_DIR = './generated_samples_bat/json_files'  # 修改为实际的JSON文件目录
    CSV_FILE = './effective_conductivity_results.csv'  # 修改为实际的CSV文件路径
    BATCH_SIZE = 32
    EPOCHS = 500
    LEARNING_RATE = 1e-3
    PATIENCE = 100
    
    # 创建数据集
    print("加载数据...")
    dataset = EllipseDataset(JSON_DIR, CSV_FILE, device=DEVICE)
    print(f"总数据量: {len(dataset)}")
    
    # 分割训练集和验证集 (80-20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn  # 添加自定义collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=collate_fn  # 添加自定义collate_fn
    )
    
    # 创建模型
    model = EllipseToThermalDeepSets(
        encoder_hidden=128,
        encoder_output=128,
        aggregator_hidden=256,
        aggregator_output=256
    ).to(DEVICE)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")
    
    # 训练
    print("\n开始训练...\n")
    model, train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        patience=PATIENCE
    )
    
    # 保存模型
    torch.save(model.state_dict(), 'deepsets_model_final.pt')
    print("\n模型已保存为 'deepsets_model_final.pt'")
    
    # 测试
    print("\n" + "="*60)
    print("测试模型性能")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(val_loader))
        ellipse_features = test_batch['ellipse_features'].to(DEVICE)
        conductivity_true = test_batch['conductivity'].to(DEVICE)
        phi = test_batch['phi'].to(DEVICE)
        mask = test_batch['mask'].to(DEVICE)
        
        batch_size = conductivity_true.shape[0]
        conductivity_vec_true = conductivity_true.reshape(batch_size, 4)
        
        predictions = model(ellipse_features, phi, mask)
        
        # 计算平均绝对误差 (MAE)
        mae = torch.mean(torch.abs(predictions - conductivity_vec_true)).item()
        print(f"验证集MAE: {mae:.6f}")
        
        # 显示几个示例
        print("\n示例预测 (前5个样本):")
        print("  k_xx (真实 -> 预测)")
        for i in range(min(5, batch_size)):
            true_val = conductivity_vec_true[i, 0].item()
            pred_val = predictions[i, 0].item()
            error = abs(true_val - pred_val)
            print(f"  {true_val:.6f} -> {pred_val:.6f} (误差: {error:.6f})")
    
    return model


if __name__ == "__main__":
    model = main()
