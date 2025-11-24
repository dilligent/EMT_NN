# -*- coding: utf-8 -*-
"""
demo.py
Harmonic map + 变换热学导出任意环形隐身斗篷张量 K（pyigl 版）
- 约束三角剖分：优先 igl.copyleft.triangle.triangulate，退回 Python 包 'triangle'
- 边界回路：纯 numpy 从三角形拓扑重建，避免 pyigl 接口差异

依赖:
  pip install pyigl numpy scipy meshio matplotlib
  可选: pip install triangle   # 若你的 pyigl 不含 Triangle 后端
"""

import argparse
import numpy as np
import igl
import scipy.sparse
import meshio
import matplotlib.pyplot as plt


def polygon_area(pts):
    x = pts[:, 0]; y = pts[:, 1]
    x2 = np.roll(x, -1); y2 = np.roll(y, -1)
    return 0.5 * np.sum(x * y2 - x2 * y)


def build_polylines(n_outer=100, n_inner=30, seed=0):
    t0 = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    t1 = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)

    # # 外边界：略带起伏的超椭圆
    # R0 = 1.0 + 0.08 * np.sin(5 * t0) + 0.05 * np.cos(3 * t0 + 0.3)
    # a0, b0 = 1.4, 1.0
    # x0 = a0 * R0 * np.cos(t0)
    # y0 = b0 * R0 * np.sin(t0)
    # outer = np.c_[x0, y0]

    # # 内边界：旋转椭圆+花瓣扰动
    # R1 = 0.35 + 0.05 * np.sin(7 * t1 + 0.5) + 0.03 * np.cos(4 * t1 - 0.2)
    # a1, b1 = 0.55, 0.35
    # phi = 0.4
    # x1r = a1 * R1 * np.cos(t1)
    # y1r = b1 * R1 * np.sin(t1)
    # x1 = x1r * np.cos(phi) - y1r * np.sin(phi)
    # y1 = x1r * np.sin(phi) + y1r * np.cos(phi)
    # inner = np.c_[x1, y1]

    # # 外边界：旋转椭圆+花瓣扰动
    # R0 = 1.0 + 0.08 * np.sin(7 * t0 + 0.5) + 0.05 * np.cos(4 * t0 - 0.2)
    # a0, b0 = 1.4, 1.0
    # phi = 0.4
    # x0r = a0 * R0 * np.cos(t0)
    # y0r = b0 * R0 * np.sin(t0)
    # x0 = x0r * np.cos(phi) - y0r * np.sin(phi)
    # y0 = x0r * np.sin(phi) + y0r * np.cos(phi)
    # outer = np.c_[x0, y0]

    # # 内边界：略带起伏的超椭圆
    # R1 = 0.35 + 0.05 * np.sin(5 * t1) + 0.03 * np.cos(3 * t1 + 0.3)
    # a1, b1 = 0.55, 0.35
    # x1 = a1 * R1 * np.cos(t1)
    # y1 = b1 * R1 * np.sin(t1)
    # inner = np.c_[x1, y1]

    # 外边界：大圆
    R0 = 1.0
    x0 = R0 * np.cos(t0)
    y0 = R0 * np.sin(t0)
    outer = np.c_[x0, y0]

    # 内边界：略带起伏的超椭圆
    R1 = 0.35 + 0.05 * np.sin(5 * t1) + 0.03 * np.cos(3 * t1 + 0.3)
    a1, b1 = 0.55, 0.35
    x1 = a1 * R1 * np.cos(t1)
    y1 = b1 * R1 * np.sin(t1)
    inner = np.c_[x1, y1]

    return outer, inner

def triangulate_annulus(outer, inner, max_area_fraction=5e-4):
    """
    约束三角剖分：优先 igl.copyleft.triangle，失败则用 pip 的 triangle 包
    返回:
      V: (#V,2), F: (#F,3)
    """
    n0, n1 = len(outer), len(inner)
    P = np.vstack([outer, inner])

    # 段
    E0 = np.c_[np.arange(0, n0), np.roll(np.arange(0, n0), -1)]
    E1 = np.c_[n0 + np.arange(0, n1), n0 + np.roll(np.arange(0, n1), -1)]
    E = np.vstack([E0, E1]).astype(np.int32)

    # 孔内一点
    H = np.array([inner.mean(axis=0)])

    # 最大面积
    bb = np.array([P.min(axis=0), P.max(axis=0)])
    bbox_area = (bb[1, 0] - bb[0, 0]) * (bb[1, 1] - bb[0, 1])
    max_area = bbox_area * max_area_fraction

    # 1) 尝试 igl.copyleft.triangle.triangulate
    has_igl_triangle = (
        hasattr(igl, "copyleft")
        and hasattr(igl.copyleft, "triangle")
        and hasattr(igl.copyleft.triangle, "triangulate")
    )
    if has_igl_triangle:
        flags = f"q30a{max_area:.6e}"
        V, F = igl.copyleft.triangle.triangulate(P, E, H, flags)
        return V, F.astype(np.int32)

    # 2) 退回到 Python 'triangle' 包
    try:
        import triangle as tr
    except ImportError:
        raise RuntimeError(
            "pyigl 不包含 igl.copyleft.triangle.triangulate，且未安装 Python 包 'triangle'.\n"
            "请执行: pip install triangle\n"
            "或从源码编译 pyigl 并启用 copyleft/triangle 支持。"
        )
    A = {"vertices": P, "segments": E, "holes": H}
    B = tr.triangulate(A, f"pq30a{max_area:.6e}")
    V = B["vertices"].astype(float)
    F = B["triangles"].astype(np.int32)
    return V, F


def boundary_loops_from_faces(F):
    """
    仅用三角形面 F 构建边界边并重建所有边界回路。
    返回: list[np.ndarray]，每条回路是按顺序排列的一串顶点索引。
    """
    F = F.astype(np.int64, copy=False)
    # 所有三角形的无向边（排序后便于计数）
    e01 = np.sort(F[:, [0, 1]], axis=1)
    e12 = np.sort(F[:, [1, 2]], axis=1)
    e20 = np.sort(F[:, [2, 0]], axis=1)
    E_all = np.vstack([e01, e12, e20])
    # 唯一边与出现次数
    E_uniq, counts = np.unique(E_all, axis=0, return_counts=True)
    E_bnd = E_uniq[counts == 1]  # 边界边（只出现一次）
    if E_bnd.size == 0:
        raise RuntimeError("No boundary edges found. Mesh appears closed.")
    # 无向邻接（环形边界上每个顶点度应为 2）
    neighbors = {}
    for a, b in E_bnd:
        a = int(a); b = int(b)
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    # 重建所有闭合回路
    visited = set()
    loops = []
    for start in list(neighbors.keys()):
        if start in visited:
            continue
        loop = []
        prev = None
        curr = start
        while True:
            loop.append(curr)
            visited.add(curr)
            nbs = neighbors[curr]
            # 选择不是上一个顶点的那个邻居
            if prev is None:
                nxt = nbs[0]
            else:
                if len(nbs) == 1:
                    nxt = nbs[0]
                else:
                    nxt = nbs[0] if nbs[1] == prev else nbs[1]
            prev, curr = curr, nxt
            if curr == start:
                break
            # 防止异常死循环
            if len(loop) > 10 * len(neighbors):
                raise RuntimeError("Loop reconstruction seems stuck. Check boundary edges.")
        loops.append(np.array(loop, dtype=np.int32))
    return loops


def split_boundary_loops(V, F):
    loops = boundary_loops_from_faces(F)
    if len(loops) != 2:
        raise RuntimeError(f"Expected 2 boundary loops (annulus), got {len(loops)}")
    # 先按面积大小区分内外
    areas_signed = [polygon_area(V[idx]) for idx in loops]
    areas_abs = [abs(a) for a in areas_signed]
    outer_idx = loops[int(np.argmax(areas_abs))]
    inner_idx = loops[int(np.argmin(areas_abs))]
    # 方向统一：都改为逆时针（CCW => signed area > 0）
    if polygon_area(V[outer_idx]) < 0:
        outer_idx = outer_idx[::-1].copy()
    if polygon_area(V[inner_idx]) < 0:
        inner_idx = inner_idx[::-1].copy()
    return outer_idx, inner_idx

def harmonic_map_to_annulus(V, F, outer_idx, inner_idx, epsilon=0.2):
    bc_outer = igl.map_vertices_to_circle(V, outer_idx)
    bc_inner = igl.map_vertices_to_circle(V, inner_idx) * float(epsilon)
    b = np.concatenate([outer_idx, inner_idx], axis=0)
    bc = np.vstack([bc_outer, bc_inner]).astype(V.dtype)
    # UV = igl.harmonic(V, F, b, bc, 2)
    UV = igl.harmonic(V, F, b, bc, 1)
    return UV, b, bc


def per_face_jacobians_and_K(V, F, UV, k0=1.0, det_tol=1e-12):
    n = V.shape[0]
    V3 = np.c_[V, np.zeros((n, 1))]
    G = igl.grad(V3, F)
    xi = UV[:, 0]; eta = UV[:, 1]
    dxi = (G @ xi).reshape((-1, 3))[:, :2]
    deta = (G @ eta).reshape((-1, 3))[:, :2]
    J11, J12 = dxi[:, 0], dxi[:, 1]
    J21, J22 = deta[:, 0], deta[:, 1]
    detJ = J11 * J22 - J12 * J21

    mask = detJ > det_tol
    if (~mask).any():
        print(f"Warning: {(~mask).sum()} triangles have tiny/non-positive det(J).")

    # 先全部设为 NaN，再填有效项，避免除零告警
    invJ11 = np.full_like(detJ, np.nan)
    invJ12 = np.full_like(detJ, np.nan)
    invJ21 = np.full_like(detJ, np.nan)
    invJ22 = np.full_like(detJ, np.nan)
    invJ11[mask] =  J22[mask] / detJ[mask]
    invJ12[mask] = -J12[mask] / detJ[mask]
    invJ21[mask] = -J21[mask] / detJ[mask]
    invJ22[mask] =  J11[mask] / detJ[mask]

    S11 = invJ11 * invJ11 + invJ12 * invJ12
    S12 = invJ11 * invJ21 + invJ12 * invJ22
    S22 = invJ21 * invJ21 + invJ22 * invJ22

    K11 = k0 * detJ * S11
    K12 = k0 * detJ * S12
    K22 = k0 * detJ * S22

    trK = K11 + K22
    detK = K11 * K22 - K12 * K12
    disc = np.maximum(trK * trK - 4.0 * detK, 0.0)
    lmax = 0.5 * (trK + np.sqrt(disc))
    lmin = 0.5 * (trK - np.sqrt(disc))
    condK = np.where(lmin > 0, lmax / lmin, np.inf)

    return {"J": (J11, J12, J21, J22), "detJ": detJ, "K": (K11, K12, K22), "condK": condK}


def export_vtu(V, F, Kdata, filename="out_annulus.vtu"):
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
    parser = argparse.ArgumentParser(description="Harmonic-map cloak demo (pyigl)")
    parser.add_argument("--epsilon", type=float, default=0.2, help="参数域内半径（外半径=1）")
    parser.add_argument("--k0", type=float, default=1.0, help="虚拟域等效导热率")
    parser.add_argument("--n-outer", type=int, default=100, help="外边界采样点数")
    parser.add_argument("--n-inner", type=int, default=30, help="内边界采样点数")
    parser.add_argument("--seed", type=int, default=42, help="示例边界形状随机种子")
    parser.add_argument("--max-area-frac", type=float, default=5e-4, help="最大单元面积/外包框面积")
    parser.add_argument("--out", type=str, default="out_annulus.vtu", help="输出 VTU 文件名")
    parser.add_argument("--no-plot", action="store_true", help="不显示 matplotlib 预览图")
    args = parser.parse_args()

    outer, inner = build_polylines(args.n_outer, args.n_inner, args.seed)
    V, F = triangulate_annulus(outer, inner, max_area_fraction=args.max_area_frac)
    print(f"Mesh: V={len(V)}, F={len(F)}")

    outer_idx, inner_idx = split_boundary_loops(V, F)
    UV, b, bc = harmonic_map_to_annulus(V, F, outer_idx, inner_idx, epsilon=args.epsilon)

    Kdata = per_face_jacobians_and_K(V, F, UV, k0=args.k0, det_tol=1e-12)
    export_vtu(V, F, Kdata, filename=args.out)

    if not args.no_plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].triplot(V[:, 0], V[:, 1], F, lw=0.3, color="gray")
        ax[0].plot(V[outer_idx, 0], V[outer_idx, 1], "r.", ms=1, label="outer")
        ax[0].plot(V[inner_idx, 0], V[inner_idx, 1], "b.", ms=1, label="inner")
        ax[0].set_aspect("equal"); ax[0].set_title("Physical domain mesh"); ax[0].legend(fontsize=8)
        ax[1].triplot(UV[:, 0], UV[:, 1], F, lw=0.3, color="gray")
        ax[1].set_aspect("equal"); ax[1].set_title("Harmonic map UV (to annulus)")
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()