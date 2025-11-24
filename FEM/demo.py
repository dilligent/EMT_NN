# -*- coding: utf-8 -*-
"""
harmonic_cloak_pyigl_demo.py

用 pyigl/libigl 实现二维任意环形域的 harmonic map 参数化，
并基于变换热学公式生成每个三角形上的等效导热张量 K。

依赖
- pyigl (libigl 的 Python 绑定)
- numpy, scipy
- meshio（可选：导出 VTU 以在 ParaView 可视化）
- matplotlib（可选：快速预览）

安装
- pip install pyigl meshio matplotlib scipy

运行
- python harmonic_cloak_pyigl_demo.py
- 可通过命令行参数调整 epsilon 等参数：python harmonic_cloak_pyigl_demo.py --epsilon 0.05
"""

import argparse
import numpy as np
import igl
import scipy.sparse
import meshio
import matplotlib.pyplot as plt


def polygon_area(pts):
    # Shoelace formula; pts: (m,2) closed or open
    x = pts[:, 0]
    y = pts[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    return 0.5 * np.sum(x * y2 - x2 * y)


def build_polylines(n_outer=200, n_inner=120, seed=0):
    """
    构造“任意”外/内边界（可替换为你的多边形点集）
    返回:
      outer: (N0,2)
      inner: (N1,2)
    """
    rng = np.random.default_rng(seed)
    t0 = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    t1 = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)

    # 外边界：略带起伏的超椭圆
    R0 = 1.0 + 0.08 * np.sin(5 * t0) + 0.05 * np.cos(3 * t0 + 0.3)
    a0, b0 = 1.4, 1.0
    x0 = a0 * R0 * np.cos(t0)
    y0 = b0 * R0 * np.sin(t0)
    outer = np.c_[x0, y0]

    # 内边界：旋转椭圆+花瓣扰动
    R1 = 0.35 + 0.05 * np.sin(7 * t1 + 0.5) + 0.03 * np.cos(4 * t1 - 0.2)
    a1, b1 = 0.55, 0.35
    phi = 0.4
    x1r = a1 * R1 * np.cos(t1)
    y1r = b1 * R1 * np.sin(t1)
    x1 = x1r * np.cos(phi) - y1r * np.sin(phi) + 0.0
    y1 = x1r * np.sin(phi) + y1r * np.cos(phi) + 0.0
    inner = np.c_[x1, y1]

    return outer, inner


def triangulate_annulus(outer, inner, max_area_fraction=2e-3):
    """
    用 libigl.triangulate 做约束三角剖分
    输入:
      outer, inner: 两条闭合多边形（按顺序的点）
      max_area_fraction: 最大单元面积相对外包框的比例
    返回:
      V: (#V,2), F: (#F,3)
    """
    n0, n1 = len(outer), len(inner)
    P = np.vstack([outer, inner])
    # 外边界段
    E0 = np.c_[np.arange(0, n0), np.roll(np.arange(0, n0), -1)]
    # 内边界段（索引需要平移）
    E1 = np.c_[n0 + np.arange(0, n1), n0 + np.roll(np.arange(0, n1), -1)]
    E = np.vstack([E0, E1]).astype(np.int32)
    # 内孔里随便取个点作为 Hole
    H = np.array([inner.mean(axis=0)])
    # 最大面积估计：用外包框面积 * 比例
    bb = np.array([P.min(axis=0), P.max(axis=0)])
    bbox_area = (bb[1, 0] - bb[0, 0]) * (bb[1, 1] - bb[0, 1])
    max_area = bbox_area * max_area_fraction
    flags = f"q30a{max_area:.6e}"  # 最小角度约束+最大单元面积

    V, F = igl.triangulate(P, E, H, flags)
    F = F.astype(np.int32)
    return V, F


def split_boundary_loops(V, F):
    """
    找出两条边界回路，并区分内外（按面积大小）
    返回:
      outer_idx, inner_idx: 顶点索引数组
    """
    loops = igl.boundary_loop(F)  # list[np.ndarray], 每条为一圈的顶点索引
    if len(loops) != 2:
        raise RuntimeError(f"Expected 2 boundary loops, got {len(loops)}")
    areas = [abs(polygon_area(V[idx])) for idx in loops]
    outer_idx = loops[int(np.argmax(areas))]
    inner_idx = loops[int(np.argmin(areas))]
    return outer_idx.astype(np.int32), inner_idx.astype(np.int32)


def harmonic_map_to_annulus(V, F, outer_idx, inner_idx, epsilon=0.05):
    """
    将两条边界分别映射到半径 1 和 epsilon 的圆，并解 harmonic 参数化
    返回:
      UV: (#V,2) 映射坐标 (ξ,η)
      b: 已知（边界）顶点索引
      bc: 对应的边界坐标
    """
    bc_outer = igl.map_vertices_to_circle(V, outer_idx)  # (k,2) 在单位圆
    bc_inner = igl.map_vertices_to_circle(V, inner_idx)  # (m,2) 在单位圆
    bc_inner *= float(epsilon)

    b = np.concatenate([outer_idx, inner_idx], axis=0)
    bc = np.vstack([bc_outer, bc_inner]).astype(V.dtype)

    k = 1  # harmonic 次数
    UV = igl.harmonic(V, F, b, bc, k)
    return UV, b, bc


def per_face_jacobians_and_K(V, F, UV, k0=1.0, det_tol=1e-12):
    """
    使用离散梯度算子计算每个三角形上的 J_G 和等效导热张量 K
    返回:
      dict 包含 detJ、K 的分量和各向异性条件数 condK
    """
    n = V.shape[0]
    V3 = np.c_[V, np.zeros((n, 1))]  # 提升到 3D 以适配 igl.grad
    G = igl.grad(V3, F)              # (#F*3) x #V 稀疏矩阵（SciPy）

    xi = UV[:, 0]
    eta = UV[:, 1]
    dxi = G @ xi     # (#F*3,)
    deta = G @ eta

    m = F.shape[0]
    dxi = dxi.reshape((m, 3))
    deta = deta.reshape((m, 3))
    # 仅用 x,y 分量（z 分量应为 0）
    dxi2 = dxi[:, :2]    # [∂ξ/∂x, ∂ξ/∂y]
    deta2 = deta[:, :2]  # [∂η/∂x, ∂η/∂y]

    J11 = dxi2[:, 0]
    J12 = dxi2[:, 1]
    J21 = deta2[:, 0]
    J22 = deta2[:, 1]
    detJ = J11 * J22 - J12 * J21

    # 数值防护
    flipped = np.where(detJ <= det_tol)[0]
    if len(flipped) > 0:
        print(
            f"Warning: {len(flipped)} triangles have non-positive or tiny det(J). "
            f"Consider refining mesh or increasing epsilon."
        )

    # 计算 K = k0 detJ J^{-1} J^{-T}，逐单元
    invJ11 = J22 / detJ
    invJ12 = -J12 / detJ
    invJ21 = -J21 / detJ
    invJ22 = J11 / detJ

    # invJ * invJ^T
    S11 = invJ11 * invJ11 + invJ12 * invJ12
    S12 = invJ11 * invJ21 + invJ12 * invJ22
    S22 = invJ21 * invJ21 + invJ22 * invJ22

    K11 = k0 * detJ * S11
    K12 = k0 * detJ * S12
    K22 = k0 * detJ * S22

    # 条件数（各向异性指标）：K 的本征值比
    trK = K11 + K22
    detK = K11 * K22 - K12 * K12
    disc = np.maximum(trK * trK - 4.0 * detK, 0.0)
    lmax = 0.5 * (trK + np.sqrt(disc))
    lmin = 0.5 * (trK - np.sqrt(disc))
    condK = np.where(lmin > 0, lmax / lmin, np.inf)

    return {
        "J": (J11, J12, J21, J22),
        "detJ": detJ,
        "K": (K11, K12, K22),
        "condK": condK,
    }


def export_vtu(V, F, Kdata, filename="out_annulus.vtu"):
    """
    导出到 VTK/VTU，便于在 ParaView 中可视化张量或标量
    """
    cells = [("triangle", F)]
    mesh = meshio.Mesh(
        points=np.c_[V, np.zeros((V.shape[0], 1))],
        cells=cells,
        cell_data={
            "detJ": [Kdata["detJ"]],
            "condK": [Kdata["condK"]],
            "K11": [Kdata["K"][0]],
            "K12": [Kdata["K"][1]],
            "K22": [Kdata["K"][2]],
        },
    )
    meshio.write(filename, mesh)
    print(f"VTU written: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Harmonic map cloak demo using pyigl/libigl")
    parser.add_argument("--epsilon", type=float, default=0.05, help="虚拟环形内半径（外半径=1）")
    parser.add_argument("--k0", type=float, default=1.0, help="虚拟域标量导热率")
    parser.add_argument("--n-outer", type=int, default=220, help="外边界采样点数")
    parser.add_argument("--n-inner", type=int, default=140, help="内边界采样点数")
    parser.add_argument("--seed", type=int, default=2, help="随机种子（仅影响示例边界形状）")
    parser.add_argument(
        "--max-area-frac",
        type=float,
        default=1.5e-3,
        help="最大单元面积占外包框面积的比例",
    )
    parser.add_argument("--out", type=str, default="out_annulus.vtu", help="输出 VTU 文件名")
    parser.add_argument("--no-plot", action="store_true", help="不显示 matplotlib 预览图")
    args = parser.parse_args()

    # 1) 构造或导入多边形边界（此处使用示例）
    outer, inner = build_polylines(args.n_outer, args.n_inner, args.seed)

    # 2) 约束三角剖分
    V, F = triangulate_annulus(outer, inner, max_area_fraction=args.max_area_frac)
    print(f"Mesh: V={len(V)}, F={len(F)}")

    # 3) 找边界回路并区分内外
    outer_idx, inner_idx = split_boundary_loops(V, F)

    # 4) Harmonic 参数化到同心圆环
    UV, b, bc = harmonic_map_to_annulus(V, F, outer_idx, inner_idx, epsilon=args.epsilon)

    # 5) 计算每单元雅可比与等效 K
    Kdata = per_face_jacobians_and_K(V, F, UV, k0=args.k0, det_tol=1e-12)

    # 6) 导出
    export_vtu(V, F, Kdata, filename=args.out)

    # 7) 可选：快速绘图
    if not args.no_plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].triplot(V[:, 0], V[:, 1], F, lw=0.3, color="gray")
        ax[0].plot(V[outer_idx, 0], V[outer_idx, 1], "r.", ms=2, label="outer")
        ax[0].plot(V[inner_idx, 0], V[inner_idx, 1], "b.", ms=2, label="inner")
        ax[0].set_aspect("equal")
        ax[0].set_title("Physical domain mesh")
        ax[0].legend(loc="best", fontsize=8)

        ax[1].triplot(UV[:, 0], UV[:, 1], F, lw=0.3, color="gray")
        ax[1].set_aspect("equal")
        ax[1].set_title("Harmonic map UV (to annulus)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()