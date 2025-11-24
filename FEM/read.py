# Python读取 VTU 并定位三角形
# - 思路：读出 V、F 和 K 分量；用重心坐标测试找到包含该点的三角形索引 i；返回该三角形的 K11[i], K12[i], K22[i] 组成的 2×2 矩阵。

import numpy as np
import meshio

def point_in_tri(p, a, b, c, eps=1e-12):
    v0 = c - a; v1 = b - a; v2 = p - a
    den = v0[0]*v1[1] - v1[0]*v0[1]
    if abs(den) < eps:
        return False, (np.nan, np.nan, np.nan)
    u = (v2[0]*v1[1] - v1[0]*v2[1]) / den
    v = (v0[0]*v2[1] - v2[0]*v0[1]) / den
    w = 1.0 - u - v
    inside = (u >= -eps) and (v >= -eps) and (w >= -eps)
    return inside, (u, v, w)

def find_face_index(V, F, p):
    # 线性扫描简单版；如网格很大可换成空间索引加速
    for i, (i0, i1, i2) in enumerate(F):
        inside, _ = point_in_tri(p, V[i0], V[i1], V[i2])
        if inside:
            return i
    return None

def load_K_at_point(vtu_path, x, y):
    m = meshio.read(vtu_path)
    V = m.points[:, :2]
    # 提取三角形单元（VTU 里 cells/cell_data 可能有多种类型，选择 triangle）
    tri_id = None
    for k, cells in enumerate(m.cells):
        if cells.type == "triangle":
            tri_id = k; break
    if tri_id is None:
        raise RuntimeError("No triangle cells found in VTU.")
    F = m.cells[tri_id].data

    # 对应的 cell_data 取同一个索引
    detJ = m.cell_data["detJ"][tri_id]
    K11 = m.cell_data["K11"][tri_id]
    K12 = m.cell_data["K12"][tri_id]
    K22 = m.cell_data["K22"][tri_id]

    p = np.array([x, y], dtype=float)
    i = find_face_index(V, F, p)
    if i is None:
        return None, None, "Point is outside the mesh or inside the hole."
    if not np.isfinite(detJ[i]) or detJ[i] <= 0:
        warn = "Triangle has tiny/non-positive detJ; K may be invalid here."
    else:
        warn = None
    K = np.array([[K11[i], K12[i]],
                  [K12[i], K22[i]]], dtype=float)
    return K, i, warn

# 使用示例
K, face_idx, warn = load_K_at_point("out_annulus.vtu", x=0.8, y=-0.3)
print("face =", face_idx, "\nK =\n", K, "\nwarn:", warn)

# 或者在同一脚本运行结束时，直接用内存里的数组
# - 运行完 main 后，内存中已有：
#   - V, F：网格
#   - Kdata["K"] = (K11, K12, K22)
# - 用同样的 find_face_index 查到三角形索引 i，然后：
#   - K = [[K11[i], K12[i]], [K12[i], K22[i]]]

# 提示
# - 在洞内或网格外的点，会返回 None；在 detJ 很小或非正的三角形上，K 可能为 NaN 或数值不可靠。
# - 如果你想得到“连续”的 K(x,y) 而不是分片常数，可以把每个三角形的 K 平均到顶点再线性插值；但严格的变换热学公式是基于连续映射，离散实现常用分片常数即可用于 FEM。需要的话我可以给出“顶点平均+插值”的示例。