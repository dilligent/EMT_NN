# -*- coding: utf-8 -*-
"""
demo3holes.py
Harmonic map + 变换热学：具有 3 个空腔（3 条内边界）的 2D 隐身斗篷张量 K

依赖:
  pip install pyigl numpy scipy meshio matplotlib
  可选: pip install triangle   # 若 pyigl 没有 copyleft.triangle
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


def build_polylines(n_outer=120, n_inner=40, seed=0):
    """
    生成 1 个外边界 + 3 个内边界。
    外边界：单位圆
    内边界1：略带起伏的椭圆，中心在 (-0.4, 0)
    内边界2：略带起伏的椭圆，中心在 ( 0.4, 0.25)
    内边界3：略带起伏的椭圆，中心在 ( 0.3,-0.35)
    """
    rng = np.random.default_rng(seed)

    t0 = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    t1 = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)

    # 外边界：大圆
    R0 = 1.0
    x0 = R0 * np.cos(t0)
    y0 = R0 * np.sin(t0)
    outer = np.c_[x0, y0]

    def make_inner(center, a=0.35, b=0.25, amp1=0.05, amp2=0.03, f1=5, f2=3,
                   phase1=0.3, phase2=0.8):
        R = 0.35 + amp1 * np.sin(f1 * t1 + phase1) + amp2 * np.cos(f2 * t1 + phase2)
        x = center[0] + a * R * np.cos(t1)
        y = center[1] + b * R * np.sin(t1)
        return np.c_[x, y]

    inner1 = make_inner(center=(-0.40, 0.00))
    inner2 = make_inner(center=( 0.40, 0.25))
    inner3 = make_inner(center=( 0.30,-0.35))

    inners = [inner1, inner2, inner3]
    return outer, inners


def triangulate_multi_hole(outer, inners, max_area_fraction=5e-4):
    """
    约束三角剖分：支持 1 个外边界 + 任意多个内边界。
    """
    n0 = len(outer)
    n_inners = [len(b) for b in inners]

    all_pts = [outer] + inners
    P = np.vstack(all_pts)

    E_list = []
    offset = 0
    idx0 = np.arange(offset, offset + n0)
    E0 = np.c_[idx0, np.roll(idx0, -1)]
    E_list.append(E0)
    offset += n0
    for ni in n_inners:
        idx = np.arange(offset, offset + ni)
        Ei = np.c_[idx, np.roll(idx, -1)]
        E_list.append(Ei)
        offset += ni
    E = np.vstack(E_list).astype(np.int32)

    H = np.array([b.mean(axis=0) for b in inners], dtype=float)

    bb = np.array([P.min(axis=0), P.max(axis=0)])
    bbox_area = (bb[1, 0] - bb[0, 0]) * (bb[1, 1] - bb[0, 1])
    max_area = bbox_area * max_area_fraction

    has_igl_triangle = (
        hasattr(igl, "copyleft")
        and hasattr(igl.copyleft, "triangle")
        and hasattr(igl.copyleft.triangle, "triangulate")
    )
    if has_igl_triangle:
        flags = f"q30a{max_area:.6e}"
        V, F = igl.copyleft.triangle.triangulate(P, E, H, flags)
        return V, F.astype(np.int32)

    try:
        import triangle as tr
    except ImportError:
        raise RuntimeError(
            "pyigl 不包含 igl.copyleft.triangle.triangulate，且未安装 Python 包 'triangle'."
        )
    A = {"vertices": P, "segments": E, "holes": H}
    B = tr.triangulate(A, f"pq30a{max_area:.6e}")
    V = B["vertices"].astype(float)
    F = B["triangles"].astype(np.int32)
    return V, F


def boundary_loops_from_faces(F):
    F = F.astype(np.int64, copy=False)
    e01 = np.sort(F[:, [0, 1]], axis=1)
    e12 = np.sort(F[:, [1, 2]], axis=1)
    e20 = np.sort(F[:, [2, 0]], axis=1)
    E_all = np.vstack([e01, e12, e20])
    E_uniq, counts = np.unique(E_all, axis=0, return_counts=True)
    E_bnd = E_uniq[counts == 1]
    if E_bnd.size == 0:
        raise RuntimeError("No boundary edges found. Mesh appears closed.")

    neighbors = {}
    for a, b in E_bnd:
        a = int(a); b = int(b)
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)

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
            if len(loop) > 10 * len(neighbors):
                raise RuntimeError("Loop reconstruction seems stuck.")
        loops.append(np.array(loop, dtype=np.int32))
    return loops


def split_boundary_loops_multi(V, F, expected_holes=3):
    """
    根据面积把边界回路分成 1 个 outer + 多个 inners。
    """
    loops = boundary_loops_from_faces(F)
    if len(loops) != expected_holes + 1:
        raise RuntimeError(
            f"Expected {expected_holes+1} boundary loops, got {len(loops)}"
        )
    areas = [polygon_area(V[idx]) for idx in loops]
    abs_areas = np.abs(areas)
    outer_id = int(np.argmax(abs_areas))
    outer_idx = loops[outer_id]
    inner_idxs = [loops[i] for i in range(len(loops)) if i != outer_id]

    if polygon_area(V[outer_idx]) < 0:
        outer_idx = outer_idx[::-1].copy()
    for j in range(len(inner_idxs)):
        if polygon_area(V[inner_idxs[j]]) < 0:
            inner_idxs[j] = inner_idxs[j][::-1].copy()
    return outer_idx, inner_idxs


def harmonic_map_to_multi_annulus(V, F, outer_idx, inner_idxs,
                                  epsilons=(0.15, 0.18, 0.22)):
    """
    多连通域的调和映射：
      - outer_idx -> 半径 1 的圆（以原点为中心）
      - 第 i 个 inner -> 以“该内边界原几何中心”为圆心、半径 epsilons[i] 的小圆
    这样每个空腔都缩到自己原来的中心附近，而不是全部挤到原点。
    """
    m = len(inner_idxs)
    if len(epsilons) < m:
        raise ValueError("epsilons 数量不足以匹配所有内边界")

    # 外边界映射到单位圆（圆心在原点）
    bc_outer = igl.map_vertices_to_circle(V, outer_idx)

    bc_inners = []
    for i, loop in enumerate(inner_idxs):
        # 1) 源坐标中的几何中心（在物理域）
        center = V[loop].mean(axis=0)
        # 2) 在以该中心为原点的局部坐标系中做“单位圆”参数化
        #    相当于先平移到局部坐标，再 map_vertices_to_circle
        local_V = V.copy()
        local_V[:, 0] -= center[0]
        local_V[:, 1] -= center[1]
        # 仅对当前 loop 的顶点做 map
        tmp = igl.map_vertices_to_circle(local_V, loop) * float(epsilons[i])
        # 3) 再平移回全局坐标，以 center 为圆心
        tmp[:, 0] += center[0]
        tmp[:, 1] += center[1]
        bc_inners.append(tmp)

    b = np.concatenate([outer_idx] + inner_idxs, axis=0)
    bc = np.vstack([bc_outer] + bc_inners).astype(V.dtype)

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

    return {"J": (J11, J12, J21, J22), "detJ": detJ,
            "K": (K11, K12, K22), "condK": condK}


def export_vtu(V, F, Kdata, filename="out_3holes.vtu"):
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
    parser = argparse.ArgumentParser(description="Harmonic-map cloak demo with 3 holes (pyigl)")
    parser.add_argument("--epsilon1", type=float, default=0.15, help="第1个空腔在参数域中的半径")
    parser.add_argument("--epsilon2", type=float, default=0.18, help="第2个空腔在参数域中的半径")
    parser.add_argument("--epsilon3", type=float, default=0.22, help="第3个空腔在参数域中的半径")
    parser.add_argument("--k0", type=float, default=1.0, help="虚拟域等效导热率")
    parser.add_argument("--n-outer", type=int, default=120, help="外边界采样点数")
    parser.add_argument("--n-inner", type=int, default=40, help="每个内边界采样点数")
    parser.add_argument("--seed", type=int, default=42, help="示例边界形状随机种子")
    parser.add_argument("--max-area-frac", type=float, default=5e-4, help="最大单元面积/外包框面积")
    parser.add_argument("--out", type=str, default="out_3holes.vtu", help="输出 VTU 文件名")
    parser.add_argument("--no-plot", action="store_true", help="不显示 matplotlib 预览图")
    args = parser.parse_args()

    outer, inners = build_polylines(args.n_outer, args.n_inner, args.seed)
    V, F = triangulate_multi_hole(outer, inners, max_area_fraction=args.max_area_frac)
    print(f"Mesh: V={len(V)}, F={len(F)}")

    outer_idx, inner_idxs = split_boundary_loops_multi(V, F, expected_holes=3)

    UV, b, bc = harmonic_map_to_multi_annulus(
        V, F, outer_idx, inner_idxs,
        epsilons=(args.epsilon1, args.epsilon2, args.epsilon3)
    )

    Kdata = per_face_jacobians_and_K(V, F, UV, k0=args.k0, det_tol=1e-12)
    export_vtu(V, F, Kdata, filename=args.out)

    if not args.no_plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].triplot(V[:, 0], V[:, 1], F, lw=0.3, color="gray")
        ax[0].plot(V[outer_idx, 0], V[outer_idx, 1], "r.", ms=1, label="outer")
        colors = ["b", "g", "m"]
        for i, loop in enumerate(inner_idxs):
            ax[0].plot(V[loop, 0], V[loop, 1], colors[i % len(colors)] + ".", ms=1,
                       label=f"inner{i+1}")
        ax[0].set_aspect("equal"); ax[0].set_title("Physical domain mesh")
        ax[0].legend(fontsize=8)

        ax[1].triplot(UV[:, 0], UV[:, 1], F, lw=0.3, color="gray")
        ax[1].set_aspect("equal"); ax[1].set_title("Harmonic map UV (3-hole domain)")
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()