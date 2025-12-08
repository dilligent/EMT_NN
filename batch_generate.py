import random
import numpy as np
from pathlib import Path
import time

# 从你现有的脚本中导入必要的类和函数
# 确保 generate_elliptic_composites.py 在同一目录下
from generate_elliptic_composites import (
    SampleConfig,
    generate_sample,
    save_sample_as_json,
    append_summary_csv
)

def run_batch_generation():
    # ================= 配置区域 =================
    TOTAL_SAMPLES = 100          # 要生成的样本总数
    OUTPUT_DIR = Path("dataset_v1") # 输出 JSON 的文件夹
    SUMMARY_CSV = Path("dataset_v1/summary.csv") # 汇总 CSV 路径
    
    # 容器尺寸范围 (Lx, Ly)
    LX_RANGE = (0.8, 1.2)
    LY_RANGE = (0.8, 1.2)
    
    # 目标体积分数范围 (例如 0.1 到 0.4)
    PHI_RANGE = (0.1, 0.45)
    
    # 椭圆半长轴/半短轴的基础范围 (会在此基础上微调)
    BASE_A_RANGE = (0.02, 0.06)
    BASE_B_RANGE = (0.01, 0.04)
    
    # 其他固定参数
    KM = 1.0
    KI = 10.0
    GMIN = 0.005
    BOUNDARY_MARGIN = 0.02
    # ===========================================

    print(f"Start generating {TOTAL_SAMPLES} samples...")
    print(f"Output directory: {OUTPUT_DIR}")

    success_count = 0
    
    for i in range(TOTAL_SAMPLES):
        # 1. 随机化容器尺寸
        lx = round(random.uniform(*LX_RANGE), 3)
        ly = round(random.uniform(*LY_RANGE), 3)
        
        # 2. 随机化目标体积分数
        target_phi = round(random.uniform(*PHI_RANGE), 3)
        
        # 3. 随机化椭圆尺寸范围 (可选：增加数据的多样性)
        # 这里做一个简单的扰动，让不同样本的椭圆大小分布略有不同
        scale_factor = random.uniform(0.8, 1.2) 
        a_min = BASE_A_RANGE[0] * scale_factor
        a_max = BASE_A_RANGE[1] * scale_factor
        b_min = BASE_B_RANGE[0] * scale_factor
        b_max = BASE_B_RANGE[1] * scale_factor
        
        # 确保 a >= b 的逻辑在 Config 中处理，这里只需传入范围
        
        # 4. 生成随机种子
        current_seed = random.randint(0, 10**9)
        
        # 构造配置对象
        cfg = SampleConfig(
            Lx=lx,
            Ly=ly,
            km=KM,
            ki=KI,
            phi_target=target_phi, # 使用体积分数控制而不是固定数量 N
            N=None,
            gmin=GMIN,
            boundary_margin=BOUNDARY_MARGIN,
            a_range=(a_min, a_max),
            b_range=(b_min, b_max),
            theta_range_deg=(0.0, 180.0),
            seed=current_seed,
            ensure_a_ge_b=True,
            ellipse_resolution=64,
            max_trials=100000, # 防止死循环
            id_prefix=f"batch_{i}"
        )

        try:
            # 调用生成函数
            # verbose=False 减少控制台输出，只打印进度
            result = generate_sample(cfg, verbose=False)
            
            # 保存结果
            save_sample_as_json(result, OUTPUT_DIR)
            append_summary_csv(result, SUMMARY_CSV)
            
            success_count += 1
            
            # 打印进度条风格的日志
            print(f"[{i+1}/{TOTAL_SAMPLES}] ID: {result.sample_id} | "
                  f"Size: {lx:.2f}x{ly:.2f} | "
                  f"Phi: {result.phi:.3f}/{target_phi:.3f} | "
                  f"N: {len(result.ellipses)}")
                  
        except Exception as e:
            print(f"[{i+1}/{TOTAL_SAMPLES}] Failed to generate sample: {e}")

    print(f"\nBatch generation finished. {success_count}/{TOTAL_SAMPLES} successful.")
    print(f"Summary saved to: {SUMMARY_CSV}")

if __name__ == "__main__":
    run_batch_generation()