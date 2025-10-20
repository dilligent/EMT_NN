import json
import pandas as pd
import numpy as np
import os
from pathlib import Path

def preprocess_data(json_dir, csv_file, output_file):
    """加载和处理数据集，保存为预处理后的格式"""
    
    # 读取CSV文件获取热导率
    df = pd.read_csv(csv_file)
    conductivity_dict = {}
    for _, row in df.iterrows():
        sample_id = row['sample_id']
        k_matrix = np.array([
            [row['k_xx'], row['k_xy']],
            [row['k_yx'], row['k_yy']]
        ], dtype=np.float32)
        conductivity_dict[sample_id] = k_matrix
    
    # 读取JSON文件获取椭圆数据
    json_files = sorted(Path(json_dir).glob('*.json'))
    processed_data = []
    
    for json_file in json_files:
        sample_id = json_file.stem
        
        if sample_id not in conductivity_dict:
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
        conductivity = conductivity_dict[sample_id]
        phi = data['meta']['phi']
        
        processed_data.append({
            'sample_id': sample_id,
            'ellipse_features': ellipse_features.tolist(),
            'conductivity': conductivity.tolist(),
            'phi': phi
        })
    
    # 保存预处理后的数据
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    JSON_DIR = './data/json_files'  # 修改为实际的JSON文件目录
    CSV_FILE = './data/effective_conductivity_results.csv'
    OUTPUT_FILE = './data/processed_data.json'  # 输出文件路径
    
    preprocess_data(JSON_DIR, CSV_FILE, OUTPUT_FILE)
    print(f"数据预处理完成，结果保存在 {OUTPUT_FILE}")