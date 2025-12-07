import numpy as np
import igl
import scipy.sparse
import meshio
import matplotlib.pyplot as plt

def polygon_area(pts):
    # Shoelace formula; pts: (m,2) closed or open
    x = pts[:,0]; y = pts[:,1]
    x2 = np.roll(x, -1); y2 = np.roll(y, -1)
    return 0.5*np.sum(x*y2 - x2*y)

def build_polylines(n_outer=200, n_inner=120, seed=0):
    # 构造“任意”外/内边界（可替换成你的多边形）
    rng = np.random.default_rng(seed)
    t0 = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    t1 = np.linspace(0, 2*np.pi, n_inner, endpoint=False)

    # 外边界：略带起伏的超椭圆
    R0 = 1.0 + 0.08*np.sin(5*t0) + 0.05*np.cos(3*t0+0.3)
    a0, b0 = 1.4, 1.0
    x0 = a0*R0*np.cos(t0)
    y0 = b0*R0*np.sin(t0)
    outer = np.c_[x0, y0]

    # 内边界：旋转椭圆+花瓣扰动
    R1 = 0.35 + 0.05*np.sin(7*t1+0.5) + 0.03*np.cos(4*t1-0.2)
    a1, b1 = 0.55, 0.35
    phi = 0.4
    x1r = a1*R1*np.cos(t1)
    y1r = b1*R1*np.sin(t1)
    x1 = x1r*np.cos(phi) - y1r*np.sin(phi) + 0.0
    y1 = x1r*np.sin(phi) + y1r*np.cos(phi) + 0.0
    inner = np.c_[x1, y1]

    return outer, inner

def triangulate_annulus(outer, inner, max_area_fraction=2e-3):
    # 约束三角剖分（libigl::triangle）
    # P: 所有点；E: 线段；H: 孔内任一点；flags: 质量+最大面积
    n0, n1 = len(outer), len(inner)
    P = np.vstack([outer, inner])
    # 外边界段
    E0 = np.c_[np.arange(0,n0), np.roll(np.arange(0,n0), -1)]
    # 内边界段（索引需要平移）
    E1 = np.c_[n0 + np.arange(0,n1), n0 + np.roll(np.arange(0,n1), -1)]
    E = np.vstack([E0, E1]).astype(np.int32)
    # 内孔里随便取个点
    H = np.array([inner.mean(axis=0)])
    # 最大面积估计：用外包框面积 * 比例
    bb = np.array([P.min(axis=0), P.max(axis=0)])
    bbox_area = (bb[1,0]-bb[0,0])*(bb[1,1]-bb[0,1])
    max_area = bbox_area * max_area_fraction
    flags = f"q30a{max_area:.6e}"  # 最小角度约束+最大单元面积

    V, F = igl.triangulate(P, E, H, flags)
    F = F.astype(np.int32)
    return V, F

def split_boundary_loops(V, F):
    # 找出两条边界回路，并区分内外
    loops = igl.boundary_loop(F)  # Python 下返回 list[np.ndarray]
    if len(loops) != 2:
        raise RuntimeError(f"Expected 2 boundary loops, got {len(loops)}")
    # 按面积大小区分外（大）内（小）
    areas = [abs(polygon_area(V[idx])) for idx in loops]
    outer_idx = loops[int(np.argmax(areas))]
    inner_idx = loops[int(np.argmin(areas))]
    return outer_idx.astype(np.int32), inner_idx.astype(np.int32)

def harmonic_map_to_annulus(V, F, outer_idx, inner_idx, epsilon=0.05):
    # 将两条边界分别映射到半径 1 和 epsilon 的圆
    bc_outer = igl.map_vertices_to_circle(V, outer_idx)   # (k,2) 在单位圆
    bc_inner = igl.map_vertices_to_circle(V, inner_idx)   # (m,2) 在单位圆
    bc_inner *= float(epsilon)

    b = np.concatenate([outer_idx, inner_idx], axis=0)
    bc = np.vstack([bc_outer, bc_inner]).astype(V.dtype)

    # 求解一次 k=1 的 harmonic 参数化：G(x) = (ξ,η)
    # UV: 每个顶点对应的 (ξ,η)
    k = 1
    UV = igl.harmonic(V, F, b, bc, k)
    return UV, b, bc

def per_face_jacobians_and_K(V, F, UV, k0=1.0, det_tol=1e-12):
    # 使用离散梯度算子计算每个三角形上的 J_G 和 K
    n = V.shape[0]
    V3 = np.c_[V, np.zeros((n,1))]  # 提升到 3D 以适配 igl.grad
    G = igl.grad(V3, F)             # (#F*3) x #V 稀疏矩阵

    xi  = UV[:,0]
    eta = UV[:,1]
    dxi  = G @ xi    # (#F*3,)
    deta = G @ eta

    m = F.shape[0]
    dxi  = dxi.reshape((m,3))
    deta = deta.reshape((m,3))
    # 仅用 x,y 分量（z 分量应为 0）
    dxi2  = dxi[:, :2]   # (m,2) -> [∂ξ/∂x, ∂ξ/∂y]
    deta2 = deta[:, :2]  # (m,2) -> [∂η/∂x, ∂η/∂y]

    J11 = dxi2[:,0]; J12 = dxi2[:,1]
    J21 = deta2[:,0]; J22 = deta2[:,1]
    detJ = J11*J22 - J12*J21

    # 检查与数值防护
    flipped = np.where(detJ <= det_tol)[0]
    if len(flipped) > 0:
        print(f"Warning: {len(flipped)} triangles have non-positive or tiny det(J). "
              f"Consider refining mesh or increasing epsilon.")

    # 计算 K = k0 detJ J^{-1} J^{-T}，逐单元
    # 先算 J^{-1}
    invJ11 =  J22 / detJ
    invJ12 = -J12 / detJ
    invJ21 = -J21 / detJ
    invJ22 =  J11 / detJ

    # invJ * invJ^T
    S11 = invJ11*invJ11 + invJ12*invJ12
    S12 = invJ11*invJ21 + invJ12*invJ22
    S22 = invJ21*invJ21 + invJ22*invJ22

    K11 = k0 * detJ * S11
    K12 = k0 * detJ * S12
    K22 = k0 * detJ * S22

    # 条件数（各向异性指标）：K 的本征值比
    # 对 2x2 SPD，trace ± sqrt(discriminant) 可闭式计算
    trK = K11 + K22
    detK = K11*K22 - K12*K12
    disc = np.maximum(trK*trK - 4.0*detK, 0.0)
    lmax = 0.5*(trK + np.sqrt(disc))
    lmin = 0.5*(trK - np.sqrt(disc))
    condK = np.where(lmin>0, lmax/lmin, np.inf)

    return {
        "J": (J11, J12, J21, J22),
        "detJ": detJ,
        "K": (K11, K12, K22),
        "condK": condK
    }

def export_vtu(V, F, Kdata, filename="out_annulus.vtu"):
    # 导出到 VTK/VTU，便于在 ParaView 中可视化张量或标量
    cells = [("triangle", F)]
    cell_data = {
        "detJ": [Kdata["detJ"]],
        "condK": [Kdata["condK"]],
        "K11": [Kdata["K"][0]],
        "K12": [Kdata["K"][1]],
        "K22": [Kdata["K"][2]],
    }
    mesh = meshio.Mesh(points=np.c_[V, np.zeros((V.shape[0],1))],
                       cells=cells, cell_data=cell_data)
    meshio.write(filename, mesh)
    print(f"VTU written: {filename}")

def main():
    # 参数
    epsilon = 0.05   # 虚拟环形内半径（外半径=1）
    k0 = 1.0         # 虚拟域等效导热率
    # 1) 构造或导入多边形边界
    outer, inner = build_polylines(n_outer=220, n_inner=140, seed=2)
    # 2) 约束三角剖分
    V, F = triangulate_annulus(outer, inner, max_area_fraction=1.5e-3)
    print(f"Mesh: V={len(V)}, F={len(F)}")
    # 3) 找边界回路并区分内外
    outer_idx, inner_idx = split_boundary_loops(V, F)
    # 4) Harmonic 参数化到同心圆环
    UV, b, bc = harmonic_map_to_annulus(V, F, outer_idx, inner_idx, epsilon=epsilon)
    # 5) 计算每单元雅可比与等效 K
    Kdata = per_face_jacobians_and_K(V, F, UV, k0=k0, det_tol=1e-12)
    # 6) 导出
    export_vtu(V, F, Kdata, filename="out_annulus.vtu")

    # 可选：快速绘图
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].triplot(V[:,0], V[:,1], F, lw=0.3, color='gray')
    ax[0].plot(V[outer_idx,0], V[outer_idx,1], 'r.', ms=2)
    ax[0].plot(V[inner_idx,0], V[inner_idx,1], 'b.', ms=2)
    ax[0].set_aspect('equal'); ax[0].set_title('Physical domain mesh')
    ax[1].triplot(UV[:,0], UV[:,1], F, lw=0.3, color='gray')
    ax[1].set_aspect('equal'); ax[1].set_title('Harmonic map UV (to annulus)')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()