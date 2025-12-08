# -*- coding: utf-8 -*-
# filepath: d:\FDU\热学习\EMT_NN\FEM\demo3.py
"""
demo3.py
显式极坐标 + 线性径向插值的映射 + 变换热学导出任意环形隐身斗篷张量 K（pyigl 版）

- 不再解调和映射 PDE，直接用极坐标构造从物理域到参数环域 [epsilon, 1] 的显式映射。
- 对当前“外圆 + 任意内边界”的情形，Jacobian 更稳定，避免大面积 detJ≈0 或翻折。

依赖:
  pip install pyigl numpy scipy meshio matplotlib
  可选: pip install triangle   # 若你的 pyigl 不含 Triangle 后端
"""

import argparse
import numpy as np
import igl
import meshio
import matplotlib.pyplot as plt


def polygon_area(pts):
    x = pts[:, 0]; y = pts[:, 1]
    x2 = np.roll(x, -1); y2 = np.roll(y, -1)
    return 0.5 * np.sum(x * y2 - x2 * y)


def build_polylines(n_outer=100, n_inner=30, seed=0):
    """
    生成外圆 + 内部起伏超椭圆的物理域边界
    """
    t0 = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    t1 = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)

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


# ---------- 显式极坐标映射的关键部分 ----------

def build_inner_radius_function(inner, n_samples=720):
    """
    给定内边界点 inner (N,2)，构造一个函数 R_in(theta)，
    方法：预先在 [0, 2π) 上取若干角度采样，沿每条射线求与多边形的交点半径最大值。
    为简单起见，这里采用“投影 + 插值”的近似实现：对 inner 的每个点求角度和半径，
    并按角度排序后线性插值。
    对你当前这种“单调包围原点的闭合曲线”是足够的。
    """
    x, y = inner[:, 0], inner[:, 1]
    theta = np.arctan2(y, x)  # [-pi, pi]
    r = np.sqrt(x * x + y * y)

    # 把角度映射到 [0, 2π)
    theta = np.mod(theta, 2 * np.pi)

    # 按角度排序
    order = np.argsort(theta)
    theta_sorted = theta[order]
    r_sorted = r[order]

    # 保证首尾闭合：在开头加一个 theta_sorted[-1]-2π，在末尾加一个 theta_sorted[0]+2π
    theta_ext = np.concatenate([
        theta_sorted[-1:] - 2 * np.pi,
        theta_sorted,
        theta_sorted[:1] + 2 * np.pi
    ])
    r_ext = np.concatenate([r_sorted[-1:], r_sorted, r_sorted[:1]])

    def R_in(query_theta):
        # query_theta 可以是标量或数组，先映射到 [0, 2π)
        qt = np.mod(query_theta, 2 * np.pi)
        # 用 np.interp 在扩展的 [theta_ext, r_ext] 上插值
        return np.interp(qt, theta_ext, r_ext)

    return R_in


def explicit_polar_map(V, outer, inner, epsilon=0.2):
    """
    显式极坐标映射:
      - 外边界 outer 视为半径 Rout(θ)（当前为常数 1）；
      - 内边界 inner 通过 R_in(theta) 计算各角度的半径；
      - 任一点 (x,y): r = sqrt(x^2+y^2), θ = atan2(y,x)
        t = (r - R_in(θ)) / (Rout(θ) - R_in(θ))
        r̃ = ε + t * (1 - ε)
        ξ = r̃ cosθ, η = r̃ sinθ
    返回:
      UV: (nV, 2)
    """
    # 外边界是圆：Rout 恒为 1
    R_out = 1.0

    # 构造内半径函数
    R_in = build_inner_radius_function(inner)

    X, Y = V[:, 0], V[:, 1]
    theta = np.arctan2(Y, X)
    r = np.sqrt(X * X + Y * Y)

    Rin = R_in(theta)            # 每个点对应方向上的内边界半径
    Rout = R_out * np.ones_like(Rin)

    # 安全处理：若数值上有 r < Rin，强制拉回到 Rin
    r_clamped = np.maximum(r, Rin)

    # 线性插值参数
    denom = Rout - Rin
    # 防止极小间隙导致除零
    denom_safe = np.where(denom > 1e-8, denom, 1e-8)
    t = (r_clamped - Rin) / denom_safe
    t = np.clip(t, 0.0, 1.0)

    r_tilde = epsilon + t * (1.0 - epsilon)

    UV = np.column_stack([r_tilde * np.cos(theta),
                          r_tilde * np.sin(theta)])
    return UV


# ---------- 变换热学 K 的计算（与原来相同） ----------

def per_face_jacobians_and_K(V, F, UV, k0=1.0, det_tol=1e-12):
    """
    按变换介质公式计算每个三角形的雅可比与等效导热张量 K。
    为避免大面积 NaN：
      - 对 detJ 做下限裁剪；
      - 对最小特征值做下限裁剪。
    """
    n = V.shape[0]
    V3 = np.c_[V, np.zeros((n, 1))]
    G = igl.grad(V3, F)
    xi = UV[:, 0]; eta = UV[:, 1]
    dxi = (G @ xi).reshape((-1, 3))[:, :2]
    deta = (G @ eta).reshape((-1, 3))[:, :2]

    J11, J12 = dxi[:, 0], dxi[:, 1]
    J21, J22 = deta[:, 0], deta[:, 1]
    detJ = J11 * J22 - J12 * J21

    # 输出 detJ 统计信息
    print("detJ stats:",
          "min =", float(np.nanmin(detJ)),
          "max =", float(np.nanmax(detJ)),
          "mean =", float(np.nanmean(detJ)))
    print("detJ <= 0 count:", int((detJ <= 0).sum()))
    print("detJ < det_tol count:", int((detJ < det_tol).sum()))

    # 为数值稳定，对 detJ 做裁剪（避免过小或非正）
    detJ_clipped = np.copy(detJ)
    bad = detJ_clipped <= det_tol
    if bad.any():
        print(f"Warning: {bad.sum()} triangles have tiny/non-positive det(J). "
              f"Clipping them to det_tol = {det_tol}.")
        detJ_clipped[bad] = det_tol

    invJ11 = J22 / detJ_clipped
    invJ12 = -J12 / detJ_clipped
    invJ21 = -J21 / detJ_clipped
    invJ22 = J11 / detJ_clipped

    # S = J^{-1} (J^{-1})^T
    S11 = invJ11 * invJ11 + invJ12 * invJ12
    S12 = invJ11 * invJ21 + invJ12 * invJ22
    S22 = invJ21 * invJ21 + invJ22 * invJ22

    # K = k0 * detJ * S （这里用裁剪后的 detJ_clipped）
    K11 = k0 * detJ_clipped * S11
    K12 = k0 * detJ_clipped * S12
    K22 = k0 * detJ_clipped * S22

    # 特征值与条件数，加入下限裁剪
    trK = K11 + K22
    detK = K11 * K22 - K12 * K12
    disc = np.maximum(trK * trK - 4.0 * detK, 0.0)
    lmax = 0.5 * (trK + np.sqrt(disc))
    lmin = 0.5 * (trK - np.sqrt(disc))

    eps_eig = 1e-12
    lmin_clipped = np.where(lmin > eps_eig, lmin, eps_eig)
    condK = lmax / lmin_clipped

    return {
        "J": (J11, J12, J21, J22),
        "detJ": detJ_clipped,
        "K": (K11, K12, K22),
        "condK": condK,
    }


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
    parser = argparse.ArgumentParser(description="Explicit-polar-map cloak demo (pyigl)")
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

    # 直接用显式映射构造 UV（不再解调和映射）
    UV = explicit_polar_map(V, outer, inner, epsilon=args.epsilon)

    Kdata = per_face_jacobians_and_K(V, F, UV, k0=args.k0, det_tol=1e-12)
    export_vtu(V, F, Kdata, filename=args.out)

    if not args.no_plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].triplot(V[:, 0], V[:, 1], F, lw=0.3, color="gray")
        ax[0].plot(outer[:, 0], outer[:, 1], "r-", lw=1, label="outer")
        ax[0].plot(inner[:, 0], inner[:, 1], "b-", lw=1, label="inner")
        ax[0].set_aspect("equal")
        ax[0].set_title("Physical domain mesh")
        ax[0].legend(fontsize=8)

        ax[1].triplot(UV[:, 0], UV[:, 1], F, lw=0.3, color="gray")
        ax[1].set_aspect("equal")
        ax[1].set_title("Explicit polar map UV (to annulus)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()