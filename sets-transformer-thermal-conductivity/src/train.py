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
from models.sets_transformer import SetsTransformer  # Assuming SetsTransformer is defined in sets_transformer.py

class EllipseDataset(Dataset):
    def __init__(self, json_dir, csv_file, device='cuda'):
        self.device = device
        self.data = []
        
        df = pd.read_csv(csv_file)
        self.conductivity_dict = {}
        for _, row in df.iterrows():
            sample_id = row['sample_id']
            k_matrix = np.array([
                [row['k_xx'], row['k_xy']],
                [row['k_yx'], row['k_yy']]
            ], dtype=np.float32)
            self.conductivity_dict[sample_id] = k_matrix
        
        json_files = sorted(Path(json_dir).glob('*.json'))
        
        for json_file in json_files:
            sample_id = json_file.stem
            
            if sample_id not in self.conductivity_dict:
                continue
                
            with open(json_file, 'r') as f:
                data = json.load(f)
            
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

class EarlyStopping:
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
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        ellipse_features = batch['ellipse_features'].to(device)
        conductivity = batch['conductivity'].to(device)
        phi = batch['phi'].to(device)
        
        batch_size = conductivity.shape[0]
        conductivity_vec = conductivity.reshape(batch_size, 4)
        
        optimizer.zero_grad()
        predictions = model(ellipse_features, phi)
        
        loss = criterion(predictions, conductivity_vec)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            ellipse_features = batch['ellipse_features'].to(device)
            conductivity = batch['conductivity'].to(device)
            phi = batch['phi'].to(device)
            
            batch_size = conductivity.shape[0]
            conductivity_vec = conductivity.reshape(batch_size, 4)
            
            predictions = model(ellipse_features, phi)
            loss = criterion(predictions, conductivity_vec)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train_model(model, train_loader, val_loader, epochs=200, learning_rate=1e-3, 
                device='cuda', patience=15):
    
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
        
        if early_stopping(val_loss, epoch):
            print(f"\n早停触发 (Epoch {epoch+1})")
            print(f"最佳验证损失: {early_stopping.best_loss:.6f} (Epoch {early_stopping.best_epoch+1})")
            break
    
    if os.path.exists(early_stopping.path):
        model.load_state_dict(torch.load(early_stopping.path))
        print(f"已加载最佳模型")
    
    return model, train_losses, val_losses

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {DEVICE}")
    
    JSON_DIR = './data/json_files'  
    CSV_FILE = './data/effective_conductivity_results.csv'
    BATCH_SIZE = 16
    EPOCHS = 200
    LEARNING_RATE = 1e-3
    PATIENCE = 15
    
    print("加载数据...")
    dataset = EllipseDataset(JSON_DIR, CSV_FILE, device=DEVICE)
    print(f"总数据量: {len(dataset)}")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    
    model = SetsTransformer().to(DEVICE)  # Assuming SetsTransformer is defined in sets_transformer.py
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")
    
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
    
    torch.save(model.state_dict(), 'sets_transformer_model_final.pt')
    print("\n模型已保存为 'sets_transformer_model_final.pt'")

if __name__ == "__main__":
    model = main()