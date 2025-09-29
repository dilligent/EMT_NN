import json
import csv
import uuid
import math
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Point, box, Polygon
from shapely import affinity


@dataclass
class EllipseParam:
    x: float
    y: float
    a: float   # semimajor (不强制 a>=b，但建议)
    b: float   # semiminor
    theta_deg: float  # rotation angle in degrees


@dataclass
class SampleConfig:
    Lx: float = 1.0
    Ly: float = 1.0
    km: float = 1.0     # 基底热导率
    ki: float = 10.0    # 椭圆热导率（单样品中相同）
    N: Optional[int] = 30  # 椭圆数，和 phi_target 二选一
    phi_target: Optional[float] = None  # 目标体积分数（0~1），可空
    phi_tol: float = 1e-3  # 体积分数到达目标后的容差
    a_range: Tuple[float, float] = (0.02, 0.06)  # 半长轴范围
    b_range: Tuple[float, float] = (0.01, 0.04)  # 半短轴范围
    theta_range_deg: Tuple[float, float] = (0.0, 180.0) # 角度范围（度）
    gmin: float = 0.002  # 最小间隙
    max_trials: int = 200000  # 全局最大尝试次数（拒绝采样保护）
    per_ellipse_max_tries: int = 5000  # 每个椭圆最大尝试次数
    ensure_a_ge_b: bool = True         # 若为 True，交换以确保 a>=b
    seed: Optional[int] = 42
    ellipse_resolution: int = 64       # 圆近似的每1/4圆分段数（越大越光滑）
    id_prefix: str = "sample"          # 样品ID前缀


@dataclass
class SampleResult:
    sample_id: str
    config: SampleConfig
    ellipses: List[EllipseParam]
    phi: float  # 实际体积分数


def make_ellipse_polygon(x: float, y: float, a: float, b: float, theta_deg: float,
                         ellipse_resolution: int) -> Polygon:
    """
    使用 shapely 由单位圆经缩放、旋转、平移构造椭圆多边形。
    ellipse_resolution: 每1/4圆段的分段数，越大越光滑
    """
    circle = Point(0.0, 0.0).buffer(1.0, resolution=ellipse_resolution)  # 约 4*resolution 边
    ell = affinity.scale(circle, a, b, origin=(0, 0))
    if theta_deg != 0.0:
        ell = affinity.rotate(ell, theta_deg, origin=(0, 0), use_radians=False)
    ell = affinity.translate(ell, xoff=x, yoff=y)
    return ell


def try_place_one(rect: Polygon,
                  placed_buffers: List[Polygon],
                  cfg: SampleConfig) -> Optional[EllipseParam]:
    """
    随机尝试放置一个椭圆：采样 a,b,theta,x,y，构造椭圆，
    通过 buffer(gmin/2) 后要求：
      - 缓冲后的椭圆完全在矩形内
      - 与已放置的所有“缓冲后椭圆”不相交
    成功返回参数，失败返回 None。
    """
    for _ in range(cfg.per_ellipse_max_tries):
        a = random.uniform(*cfg.a_range)
        b = random.uniform(*cfg.b_range)
        if cfg.ensure_a_ge_b and a < b:
            a, b = b, a
        theta = random.uniform(*cfg.theta_range_deg)

        # 中心点先在整个矩形内均匀采样
        x = random.uniform(0.0, cfg.Lx)
        y = random.uniform(0.0, cfg.Ly)

        ell = make_ellipse_polygon(x, y, a, b, theta, cfg.ellipse_resolution)
        # 用 buffer(gmin/2) 来同时保证椭圆间距与到边界距离
        ell_buf = ell.buffer(cfg.gmin / 2.0, resolution=cfg.ellipse_resolution)

        if not ell_buf.within(rect):
            continue

        ok = True
        for eb in placed_buffers:
            if ell_buf.intersects(eb):
                ok = False
                break
        if ok:
            return EllipseParam(x=x, y=y, a=a, b=b, theta_deg=theta)
    return None


def generate_sample(cfg: SampleConfig, verbose: bool = True) -> SampleResult:
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    rect = box(0.0, 0.0, cfg.Lx, cfg.Ly)
    placed_params: List[EllipseParam] = []
    placed_buffers: List[Polygon] = []

    trials = 0

    def current_phi(n_ellipses: int, areas_sum: float) -> float:
        return areas_sum / (cfg.Lx * cfg.Ly)

    # 目标控制
    want_by_count = cfg.N is not None
    want_by_phi = (cfg.phi_target is not None)

    if not want_by_count and not want_by_phi:
        raise ValueError("必须设置 N 或 phi_target 至少一个。")

    areas_sum = 0.0
    A_rect = cfg.Lx * cfg.Ly

    # 按 N 放置（若给了 N）
    if want_by_count:
        while len(placed_params) < int(cfg.N):
            if trials >= cfg.max_trials:
                break
            trials += 1
            ep = try_place_one(rect, placed_buffers, cfg)
            if ep is None:
                continue
            placed_params.append(ep)
            ell_buf = make_ellipse_polygon(ep.x, ep.y, ep.a, ep.b, ep.theta_deg, cfg.ellipse_resolution) \
                      .buffer(cfg.gmin / 2.0, resolution=cfg.ellipse_resolution)
            placed_buffers.append(ell_buf)
            areas_sum += math.pi * ep.a * ep.b
            if verbose and len(placed_params) % 10 == 0:
                print(f"Placed {len(placed_params)}/{cfg.N} ... trials={trials}")
    # 按 phi_target 放置（若给了 phi）
    if want_by_phi:
        target = float(cfg.phi_target)
        while current_phi(len(placed_params), areas_sum) < target:
            if trials >= cfg.max_trials:
                break
            trials += 1
            ep = try_place_one(rect, placed_buffers, cfg)
            if ep is None:
                continue
            placed_params.append(ep)
            ell_buf = make_ellipse_polygon(ep.x, ep.y, ep.a, ep.b, ep.theta_deg, cfg.ellipse_resolution) \
                      .buffer(cfg.gmin / 2.0, resolution=cfg.ellipse_resolution)
            placed_buffers.append(ell_buf)
            areas_sum += math.pi * ep.a * ep.b
            if verbose and len(placed_params) % 10 == 0:
                print(f"Placed {len(placed_params)} ... phi={areas_sum / A_rect:.4f}, trials={trials}")

        # 可选：尝试微调/回退超出过多的情况，这里简化为接受当前结果

    phi = areas_sum / A_rect
    sample_id = f"{cfg.id_prefix}_{uuid.uuid4().hex[:8]}"

    if verbose:
        print(f"Done. Placed {len(placed_params)} ellipses, phi={phi:.6f}, trials={trials}")

    return SampleResult(sample_id=sample_id, config=cfg, ellipses=placed_params, phi=phi)


def save_sample_as_json(sample: SampleResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "meta": {
            "sample_id": sample.sample_id,
            "seed": sample.config.seed,
            "Lx": sample.config.Lx,
            "Ly": sample.config.Ly,
            "km": sample.config.km,
            "ki": sample.config.ki,
            "gmin": sample.config.gmin,
            "N": len(sample.ellipses),
            "phi": sample.phi,
            "a_range": list(sample.config.a_range),
            "b_range": list(sample.config.b_range),
            "theta_range_deg": list(sample.config.theta_range_deg),
            "ellipse_resolution": sample.config.ellipse_resolution,
        },
        "ellipses": [
            {
                "x": ep.x,
                "y": ep.y,
                "a": ep.a,
                "b": ep.b,
                "theta_deg": ep.theta_deg
            } for ep in sample.ellipses
        ]
    }
    json_path = out_dir / f"{sample.sample_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return json_path


def append_summary_csv(sample: SampleResult, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "sample_id", "seed", "Lx", "Ly", "km", "ki",
                "N", "phi", "gmin",
                "a_min", "a_max", "b_min", "b_max",
                "theta_min_deg", "theta_max_deg"
            ])
        cfg = sample.config
        writer.writerow([
            sample.sample_id, cfg.seed, cfg.Lx, cfg.Ly, cfg.km, cfg.ki,
            len(sample.ellipses), sample.phi, cfg.gmin,
            cfg.a_range[0], cfg.a_range[1], cfg.b_range[0], cfg.b_range[1],
            cfg.theta_range_deg[0], cfg.theta_range_deg[1]
        ])


def parse_args():
    p = argparse.ArgumentParser(description="Generate random non-overlapping rotated ellipses in a rectangle.")
    p.add_argument("--Lx", type=float, default=1.0)
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--km", type=float, default=1.0)
    p.add_argument("--ki", type=float, default=10.0)
    p.add_argument("--N", type=int, default=None, help="Number of ellipses to place.")
    p.add_argument("--phi_target", type=float, default=None, help="Target area fraction (0~1).")
    p.add_argument("--gmin", type=float, default=0.002)
    p.add_argument("--a_min", type=float, default=0.02)
    p.add_argument("--a_max", type=float, default=0.06)
    p.add_argument("--b_min", type=float, default=0.01)
    p.add_argument("--b_max", type=float, default=0.04)
    p.add_argument("--theta_min", type=float, default=0.0)
    p.add_argument("--theta_max", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ensure_a_ge_b", action="store_true")
    p.add_argument("--ellipse_resolution", type=int, default=64)
    p.add_argument("--max_trials", type=int, default=200000)
    p.add_argument("--per_ellipse_max_tries", type=int, default=5000)
    p.add_argument("--id_prefix", type=str, default="sample")
    p.add_argument("--out_dir", type=str, default="out_json")
    p.add_argument("--summary_csv", type=str, default="samples_summary.csv")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = SampleConfig(
        Lx=args.Lx, Ly=args.Ly,
        km=args.km, ki=args.ki,
        N=args.N,
        phi_target=args.phi_target,
        gmin=args.gmin,
        a_range=(args.a_min, args.a_max),
        b_range=(args.b_min, args.b_max),
        theta_range_deg=(args.theta_min, args.theta_max),
        seed=args.seed,
        ensure_a_ge_b=args.ensure_a_ge_b,
        ellipse_resolution=args.ellipse_resolution,
        max_trials=args.max_trials,
        per_ellipse_max_tries=args.per_ellipse_max_tries,
        id_prefix=args.id_prefix
    )

    sample = generate_sample(cfg, verbose=args.verbose)
    out_dir = Path(args.out_dir)
    json_path = save_sample_as_json(sample, out_dir)
    append_summary_csv(sample, Path(args.summary_csv))

    if args.verbose:
        print(f"Saved JSON: {json_path}")
        print(f"Appended summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
