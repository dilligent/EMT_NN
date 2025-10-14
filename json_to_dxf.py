import json
import math
import ezdxf
from ezdxf.enums import TextEntityAlignment
from pathlib import Path
import argparse

def convert_json_to_dxf(json_path: Path, dxf_path: Path):
    """
    读取 generate_elliptic_composites.py 生成的JSON文件，
    并将其中的几何信息转换为DXF格式。
    """
    # 1. 读取JSON数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    meta = data['meta']
    ellipses = data['ellipses']
    Lx = meta['Lx']
    Ly = meta['Ly']

    # 2. 创建一个新的DXF文档
    doc = ezdxf.new()
    msp = doc.modelspace()

    # 3. 绘制矩形边界
    # 将矩形放在一个单独的图层，方便在COMSOL中选择
    doc.layers.add(name="BOUNDARY", color=1) # 颜色1是红色
    points = [(0, 0), (Lx, 0), (Lx, Ly), (0, Ly), (0, 0)]
    msp.add_lwpolyline(points, close=False, dxfattribs={"layer": "BOUNDARY"})
    
    # 添加尺寸标注文本 (可选，但有助于验证)
    msp.add_text(f"Lx={Lx}, Ly={Ly}", 
                 dxfattribs={'style': 'OpenSans', 'height': Lx/50}
    ).set_placement((Lx/2, -Ly/20), align=TextEntityAlignment.TOP_CENTER)


    # 4. 绘制所有椭圆
    doc.layers.add(name="ELLIPSES", color=5) # 颜色5是蓝色
    for i, ell_param in enumerate(ellipses):
        x = ell_param['x']
        y = ell_param['y']
        a = ell_param['a']
        b = ell_param['b']
        theta_deg = ell_param['theta_deg']
        
        # ezdxf.add_ellipse 需要中心点、主轴矢量和短轴/长轴比
        center = (x, y)
        ratio = b / a # ezdxf 使用的是短轴与长轴的比例
        
        # 计算主轴矢量 (一个从中心点指向椭圆长轴顶点的矢量)
        theta_rad = math.radians(theta_deg)
        major_axis_vector = (a * math.cos(theta_rad), a * math.sin(theta_rad))
        
        msp.add_ellipse(
            center=center,
            major_axis=major_axis_vector,
            ratio=ratio,
            dxfattribs={"layer": "ELLIPSES"}
        )

    # 5. 保存DXF文件
    try:
        doc.saveas(dxf_path)
        print(f"成功将 '{json_path.name}' 转换为 '{dxf_path.name}'")
    except IOError:
        print(f"无法保存DXF文件: '{dxf_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ellipse JSON data to DXF format for COMSOL.")
    parser.add_argument("json_file", type=str, help="Input JSON file path from generate_elliptic_composites.py")
    args = parser.parse_args()

    json_p = Path(args.json_file)
    if not json_p.exists():
        print(f"错误: 文件不存在 {json_p}")
    else:
        # 将输出的dxf文件与json文件放在同一目录，并使用相同的文件名
        dxf_p = json_p.with_suffix('.dxf')
        convert_json_to_dxf(json_p, dxf_p)

