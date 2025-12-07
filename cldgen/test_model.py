"""
快速测试脚本
验证模型和数据加载是否正常工作
"""

import torch
import sys
from pathlib import Path

print("="*60)
print("Deep Sets Model - Quick Test")
print("="*60)

# 测试导入
print("\n1. Testing imports...")
try:
    from deep_sets_model import DeepSetsModel, DeepSetsWithAttention
    from data_loader import create_dataloaders
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# 测试数据加载
print("\n2. Testing data loader...")
try:
    json_dir = "../generated_samples_bat/json_files"
    csv_file = "../effective_conductivity_results.csv"
    
    if not Path(json_dir).exists():
        print(f"✗ Directory not found: {json_dir}")
        sys.exit(1)
    
    if not Path(csv_file).exists():
        print(f"✗ File not found: {csv_file}")
        sys.exit(1)
    
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        json_dir=json_dir,
        csv_file=csv_file,
        batch_size=4,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        num_workers=0,
        random_seed=42
    )
    
    print(f"✓ Data loaded successfully")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Max ellipses: {dataset.max_num_ellipses}")
    
except Exception as e:
    print(f"✗ Data loading error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试批次数据
print("\n3. Testing batch data...")
try:
    batch = next(iter(train_loader))
    print(f"✓ Batch loaded successfully")
    print(f"  Ellipse features shape: {batch['ellipse_features'].shape}")
    print(f"  Global features shape: {batch['global_features'].shape}")
    print(f"  Mask shape: {batch['mask'].shape}")
    print(f"  K matrix shape: {batch['k_matrix'].shape}")
    
except Exception as e:
    print(f"✗ Batch loading error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试基础模型
print("\n4. Testing basic Deep Sets model...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    model = DeepSetsModel(
        ellipse_feature_dim=5,
        encoder_hidden_dims=[64, 128, 256],
        decoder_hidden_dims=[256, 128, 64],
        aggregation='mean_max',
        output_dim=4,
        include_global_features=True
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Basic model created successfully")
    print(f"  Total parameters: {num_params:,}")
    
    # 前向传播测试
    # Fix: Explicitly cast to float32 to avoid Double vs Float errors
    ellipse_features = batch['ellipse_features'].to(device).float()
    global_features = batch['global_features'].to(device).float()
    
    with torch.no_grad():
        output = model(ellipse_features, global_features)
    
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {output.shape}")
    print(f"  Output sample: {output[0].cpu().numpy()}")
    
except Exception as e:
    print(f"✗ Basic model error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试注意力模型
print("\n5. Testing attention-based model...")
try:
    attention_model = DeepSetsWithAttention(
        ellipse_feature_dim=5,
        encoder_hidden_dims=[64, 128, 256],
        decoder_hidden_dims=[256, 128, 64],
        output_dim=4,
        include_global_features=True,
        attention_heads=4
    ).to(device)
    
    num_params_att = sum(p.numel() for p in attention_model.parameters())
    print(f"✓ Attention model created successfully")
    print(f"  Total parameters: {num_params_att:,}")
    
    # 前向传播测试
    with torch.no_grad():
        output_att = attention_model(ellipse_features, global_features)
    
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {output_att.shape}")
    
except Exception as e:
    print(f"✗ Attention model error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试mask功能
print("\n6. Testing masked model...")
try:
    from train import MaskedDeepSetsModel
    
    masked_model = MaskedDeepSetsModel(model).to(device)
    # Fix: Explicitly cast mask to float32
    mask = batch['mask'].to(device).float()
    
    with torch.no_grad():
        output_masked = masked_model(ellipse_features, global_features, mask)
    
    print(f"✓ Masked model works correctly")
    print(f"  Output shape: {output_masked.shape}")
    
except Exception as e:
    print(f"✗ Masked model error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试损失函数
print("\n7. Testing loss function...")
try:
    from train import ConductivityLoss
    
    criterion = ConductivityLoss(mse_weight=1.0, symmetry_weight=0.1, positive_weight=0.0)
    # Fix: Explicitly cast target to float32
    k_matrix = batch['k_matrix'].to(device).float()
    
    loss, loss_dict = criterion(output_masked, k_matrix)
    
    print(f"✓ Loss computation successful")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  MSE: {loss_dict['mse']:.4f}")
    print(f"  Symmetry: {loss_dict['symmetry']:.4f}")
    print(f"  Positive: {loss_dict['positive']:.4f}")
    
except Exception as e:
    print(f"✗ Loss function error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试反归一化
print("\n8. Testing denormalization...")
try:
    k_denorm = dataset.denormalize_output(output_masked.cpu())
    print(f"✓ Denormalization successful")
    print(f"  Denormalized shape: {k_denorm.shape}")
    print(f"  Denormalized sample: {k_denorm[0]}")
    
except Exception as e:
    print(f"✗ Denormalization error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ All tests passed successfully!")
print("="*60)
print("\nYou can now run training with:")
print("  python train.py")
print("\nOr with custom config:")
print("  python train.py --config config_basic.json")
print("  python train.py --config config_attention.json")
print("="*60)
