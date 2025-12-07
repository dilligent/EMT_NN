# -*- coding: utf-8 -*-
"""
demo.py
Harmonic map + 变换热学导出任意环形隐身斗篷张量 K（pyigl 版）
- 优先使用 igl.copyleft.triangle.triangulate
- 若不可用，自动使用 Python 包 'triangle' 做 PSLG 约束三角剖分

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


def build_polylines(n_outer=200, n_inner=120, seed=0):
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
    x1 = x1r * np.cos(phi) - y1r * np.sin(phi)
    y1 = x1r * np.sin(phi) + y1r * np.cos(phi)
    inner = np.c_[x1, y1]
    return outer, inner


def triangulate_annulus(outer, inner, max_area_fraction=2e-3):
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
    # 'p' PSLG, 'q30' 质量, 'a' 最大面积
    B = tr.triangulate(A, f"pq30a{max_area:.6e}")
    V = B["vertices"].astype(float)
    F = B["triangles"].astype(np.int32)
    return V, F


def split_boundary_loops(V, F):
    loops = igl.boundary_loop(F)
    # pyigl 新版返回 list[np.ndarray]；旧版可能返回一个 2D 数组
    if isinstance(loops, list):
        loop_list = loops
    else:
        loop_list = [loops[i, np.where(loops[i] != -1)[0]] for i in range(loops.shape[0])]
    if len(loop_list) != 2:
        raise RuntimeError(f"Expected 2 boundary loops, got {len(loop_list)}")
    areas = [abs(polygon_area(V[idx])) for idx in loop_list]
    outer_idx = loop_list[int(np.argmax(areas))].astype(np.int32)
    inner_idx = loop_list[int(np.argmin(areas))].astype(np.int32)
    return outer_idx, inner_idx


def harmonic_map_to_annulus(V, F, outer_idx, inner_idx, epsilon=0.05):
    bc_outer = igl.map_vertices_to_circle(V, outer_idx)
    bc_inner = igl.map_vertices_to_circle(V, inner_idx) * float(epsilon)
    b = np.concatenate([outer_idx, inner_idx], axis=0)
    bc = np.vstack([bc_outer, bc_inner]).astype(V.dtype)
    UV = igl.harmonic(V, F, b, bc, 1)
    return UV, b, bc


def per_face_jacobians_and_K(V, F, UV, k0=1.0, det_tol=1e-12):
    n = V.shape[0]
    V3 = np.c_[V, np.zeros((n, 1))]
    G = igl.grad(V3, F)  # (#F*3) x #V

    xi = UV[:, 0]; eta = UV[:, 1]
    dxi = (G @ xi).reshape((-1, 3))[:, :2]
    deta = (G @ eta).reshape((-1, 3))[:, :2]

    J11, J12 = dxi[:, 0], dxi[:, 1]
    J21, J22 = deta[:, 0], deta[:, 1]
    detJ = J11 * J22 - J12 * J21

    flipped = np.where(detJ <= det_tol)[0]
    if len(flipped) > 0:
        print(f"Warning: {len(flipped)} triangles have tiny/non-positive det(J). Refine mesh or increase epsilon.")

    invJ11 =  J22 / detJ
    invJ12 = -J12 / detJ
    invJ21 = -J21 / detJ
    invJ22 =  J11 / detJ

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
    parser.add_argument("--epsilon", type=float, default=0.05, help="参数域内半径（外半径=1）")
    parser.add_argument("--k0", type=float, default=1.0, help="虚拟域等效导热率")
    parser.add_argument("--n-outer", type=int, default=220, help="外边界采样点数")
    parser.add_argument("--n-inner", type=int, default=140, help="内边界采样点数")
    parser.add_argument("--seed", type=int, default=2, help="示例边界形状随机种子")
    parser.add_argument("--max-area-frac", type=float, default=1.5e-3, help="最大单元面积/外包框面积")
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
        ax[0].plot(V[outer_idx, 0], V[outer_idx, 1], "r.", ms=2, label="outer")
        ax[0].plot(V[inner_idx, 0], V[inner_idx, 1], "b.", ms=2, label="inner")
        ax[0].set_aspect("equal"); ax[0].set_title("Physical domain mesh"); ax[0].legend(fontsize=8)
        ax[1].triplot(UV[:, 0], UV[:, 1], F, lw=0.3, color="gray")
        ax[1].set_aspect("equal"); ax[1].set_title("Harmonic map UV (to annulus)")
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()