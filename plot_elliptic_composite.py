import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse


def load_sample(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 兼容性处理
    meta = data.get("meta", {})
    ellipses = data.get("ellipses", [])
    Lx = float(meta.get("Lx", 1.0))
    Ly = float(meta.get("Ly", 1.0))
    sample_id = meta.get("sample_id", Path(json_path).stem)
    km = meta.get("km", None)
    ki = meta.get("ki", None)
    phi = meta.get("phi", None)
    return {
        "meta": meta,
        "ellipses": ellipses,
        "Lx": Lx,
        "Ly": Ly,
        "sample_id": sample_id,
        "km": km,
        "ki": ki,
        "phi": phi,
    }


def draw_sample(ax, sample, edgecolor="C0", facecolor="C1", alpha=0.6, lw=0.8,
                rect_edgecolor="black", rect_lw=1.2, draw_rect=True):
    Lx, Ly = sample["Lx"], sample["Ly"]
    # 矩形边界
    if draw_rect:
        rect = Rectangle((0, 0), Lx, Ly, fill=False, edgecolor=rect_edgecolor, linewidth=rect_lw)
        ax.add_patch(rect)

    # 椭圆
    for e in sample["ellipses"]:
        x = float(e["x"])
        y = float(e["y"])
        a = float(e["a"])
        b = float(e["b"])
        theta_deg = float(e.get("theta_deg", 0.0))
        el = Ellipse((x, y), width=2*a, height=2*b, angle=theta_deg,
                     facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, alpha=alpha)
        ax.add_patch(el)

    # 轴范围与比例
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect("equal", adjustable="box")


def main():
    parser = argparse.ArgumentParser(description="Plot ellipse composite from JSON.")
    parser.add_argument("json", type=str, help="Path to the sample JSON file.")
    parser.add_argument("--save", type=str, default=None, help="Path to save the figure (e.g., out.png).")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI when saving.")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively.")
    parser.add_argument("--no_axes", action="store_true", help="Hide axes.")
    parser.add_argument("--alpha", type=float, default=0.6, help="Ellipse face alpha.")
    parser.add_argument("--lw", type=float, default=0.8, help="Ellipse edge linewidth.")
    parser.add_argument("--edgecolor", type=str, default="C0", help="Ellipse edge color.")
    parser.add_argument("--facecolor", type=str, default="C1", help="Ellipse face color.")
    args = parser.parse_args()

    sample = load_sample(Path(args.json))
    fig, ax = plt.subplots(figsize=(6, 6))

    draw_sample(
        ax, sample,
        edgecolor=args.edgecolor,
        facecolor=args.facecolor,
        alpha=args.alpha,
        lw=args.lw
    )

    meta = sample["meta"]
    N = len(sample["ellipses"])
    phi = sample["phi"]
    title_parts = [f"{sample['sample_id']}  N={N}"]
    if phi is not None:
        title_parts.append(f"phi={phi:.3f}")
    if sample["km"] is not None and sample["ki"] is not None:
        title_parts.append(f"km={sample['km']}, ki={sample['ki']}")
    ax.set_title(" | ".join(title_parts))

    if args.no_axes:
        ax.axis("off")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.tight_layout()

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure to: {out_path}")

    if args.show or not args.save:
        plt.show()


if __name__ == "__main__":
    main()
