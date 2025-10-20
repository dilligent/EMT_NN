import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

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