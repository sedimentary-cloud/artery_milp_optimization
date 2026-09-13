"""Stage 1 max-band 目标的周期扫描示例。

复用 ``objective_templates_comparison.py`` 里的 3.1 场景与配置：

- 干线、路口、信号方案和约束来自 ``build_comparison_arterial()``；
- 目标使用 3.1 的 ``global_sum``，即最大化所有全局 band 的带宽和；
- 求解器只使用 Stage 1（``SegmentedBandSolver``），不运行 Stage 2；
- 遍历公共周期 C = 40, 42, ..., 180 秒，记录 3.1 口径的带层得分并绘制折线图。

输出：

- ``cycle_sweep_iterative_maxband_score.png``：周期 -> band_score - intersection_loss 折线图；
- ``cycle_sweep_iterative_maxband_score.csv``：每个周期的详细结果。
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artery_milp.models import (Arterial, Intersection, SignalConstraint,
                                SignalLoss, SignalPlan)
from artery_milp.solvers.stage1 import SegmentedBandSolver
from artery_milp.examples.objective_templates_comparison import (
    build_comparison_arterial,
    build_margin,
    objective_global_sum,
)

OUTPUT_DIR = Path(__file__).resolve().parent
CYCLE_START = 40.0
CYCLE_STOP = 180.0
CYCLE_STEP = 2.0


def _validate_cycle_pair(reference_cycle: float, target_cycle: float) -> tuple[float, float]:
    reference = float(reference_cycle)
    target = float(target_cycle)
    if reference <= 0:
        raise ValueError("reference_cycle 必须为正")
    if target <= 0:
        raise ValueError("target_cycle 必须为正")
    return reference, target


def _rebase_ratio(value: float, reference_cycle: float, target_cycle: float) -> float:
    """保持绝对秒数不变，把参考周期占比换算为目标周期占比。"""
    reference, target = _validate_cycle_pair(reference_cycle, target_cycle)
    return float(value) * reference / target


def _rebase_optional_ratio(
    value: float | None,
    reference_cycle: float,
    target_cycle: float,
) -> float | None:
    if value is None:
        return None
    return _rebase_ratio(value, reference_cycle, target_cycle)


def _rebase_term_bounds(
    metadata: dict[str, object],
    reference_cycle: float,
    target_cycle: float,
) -> dict[str, object]:
    out = dict(metadata)
    raw = out.get("term_bounds")
    if not isinstance(raw, dict):
        return out
    rebased: dict[str, tuple[float, float]] = {}
    for term, bounds in raw.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"term_bounds[{term!r}] 不是二元上下界")
        lower = min(max(_rebase_ratio(float(bounds[0]), reference_cycle, target_cycle), 0.0), 1.0)
        upper = min(max(_rebase_ratio(float(bounds[1]), reference_cycle, target_cycle), 0.0), 1.0)
        if lower > upper:
            raise ValueError(
                f"term_bounds[{term!r}] 换算并裁剪后下界大于上界: {(lower, upper)}"
            )
        rebased[str(term)] = (lower, upper)
    out["term_bounds"] = rebased
    return out


def _rebase_plan(plan: SignalPlan, reference_cycle: float, target_cycle: float) -> SignalPlan:
    """只重建约束/损失/term_bounds，绿灯窗口仍保持周期占比。"""
    return SignalPlan(
        name=plan.name,
        up_segments=list(plan.up_segments),
        down_segments=list(plan.down_segments),
        signal_constraints=[
            replace(
                constraint,
                rhs=_rebase_ratio(constraint.rhs, reference_cycle, target_cycle),
            )
            for constraint in plan.signal_constraints
        ],
        signal_losses=[
            replace(
                loss,
                lower_threshold=_rebase_optional_ratio(
                    loss.lower_threshold, reference_cycle, target_cycle
                ),
                upper_threshold=_rebase_optional_ratio(
                    loss.upper_threshold, reference_cycle, target_cycle
                ),
            )
            for loss in plan.signal_losses
        ],
        metadata=_rebase_term_bounds(plan.metadata, reference_cycle, target_cycle),
    )


def _rebase_arterial(base: Arterial, target_cycle: float) -> Arterial:
    """按 90s 参考周期重建成目标周期的 Arterial。"""
    reference_cycle = float(base.cycle)
    _validate_cycle_pair(reference_cycle, target_cycle)
    new_intersections: dict[str, Intersection] = {}
    for name, inter in base.intersections.items():
        new_plans: list[SignalPlan] = []
        for plan in inter.plans:
            try:
                new_plan = _rebase_plan(plan, reference_cycle, target_cycle)
            except ValueError:
                # 目标周期下无法满足自身硬约束的候选方案直接丢弃。
                continue
            new_plans.append(new_plan)
        if not new_plans:
            raise ValueError(f"路口 {name} 在 cycle={target_cycle} 下没有可用方案")
        new_intersections[name] = Intersection(name=inter.name, plans=new_plans)
    return Arterial(
        cycle=float(target_cycle),
        intersections=new_intersections,
        segments=dict(base.segments),
        order=list(base.order),
    )


def build_cycle_arterial(cycle: float) -> Arterial:
    """复用 3.1 场景，并把 90s 下的约束/损失阈值换算到目标周期。"""
    base = build_comparison_arterial()
    return _rebase_arterial(base, cycle)


def solve_one_cycle(cycle: float) -> dict[str, object]:
    """在指定周期下只运行 Stage 1 max-band 求解。"""
    try:
        arterial = build_cycle_arterial(cycle)
        solver = SegmentedBandSolver(
            config=objective_global_sum(),
            max_loops=3,
            up_global_output=True,
            down_global_output=True,
            margin=build_margin(),
        )
        sol = solver.solve(arterial, band_loss_weight=0.5)
    except Exception as exc:  # noqa: BLE001 - 示例需要把失败周期也记录进 CSV
        return {
            "cycle": float(cycle),
            "status": f"failed: {type(exc).__name__}: {exc}",
            "band_score": math.nan,
            "band_objective": math.nan,
            "band_loss": math.nan,
            "intersection_loss": math.nan,
            "up_bandwidth": math.nan,
            "down_bandwidth": math.nan,
        }

    if not str(sol.status).startswith("optimal"):
        return {
            "cycle": float(cycle),
            "status": str(sol.status),
            "band_score": math.nan,
            "band_objective": math.nan,
            "band_loss": math.nan,
            "intersection_loss": math.nan,
            "up_bandwidth": math.nan,
            "down_bandwidth": math.nan,
        }

    up_total = sum(sol.multi_bandwidths.get("up", {}).values())
    down_total = sum(sol.multi_bandwidths.get("down", {}).values())
    return {
        "cycle": float(cycle),
        "status": sol.status,
        "band_score": float(sol.band_score),
        "band_objective": float(sol.band_objective),
        "band_loss": float(sol.band_loss),
        "intersection_loss": float(sol.intersection_loss),
        "up_bandwidth": float(up_total),
        "down_bandwidth": float(down_total),
    }


def save_csv(rows: list[dict[str, object]]) -> Path:
    path = OUTPUT_DIR / "cycle_sweep_iterative_maxband_score.csv"
    fieldnames = [
        "cycle",
        "status",
        "band_score",
        "band_objective",
        "band_loss",
        "intersection_loss",
        "up_bandwidth",
        "down_bandwidth",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_curve(rows: list[dict[str, object]]) -> Path:
    cycles = [float(row["cycle"]) for row in rows]
    band_scores = [float(row["band_score"]) for row in rows]
    inter_losses = [float(row["intersection_loss"]) for row in rows]
    net_scores = [
        band - loss
        for band, loss in zip(band_scores, inter_losses)
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # 主曲线：band_score - intersection_loss。
    ax.plot(
        cycles,
        net_scores,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        color="#1f77b4",
        label="band_score - intersection_loss",
    )
    # 两条细虚线作为组成项参考。
    ax.plot(
        cycles,
        band_scores,
        linestyle=":",
        linewidth=1.0,
        color="#2ca02c",
        alpha=0.65,
        label="band_score",
    )
    ax.plot(
        cycles,
        inter_losses,
        linestyle=":",
        linewidth=1.0,
        color="#d62728",
        alpha=0.65,
        label="intersection_loss",
    )

    valid = [
        (cycle, net)
        for cycle, net in zip(cycles, net_scores)
        if not math.isnan(net)
    ]
    if valid:
        best_cycle, best_net = max(valid, key=lambda item: item[1])
        ax.scatter([best_cycle], [best_net], color="#d62728", s=55, zorder=5)
        ax.annotate(
            f"best: C={best_cycle:g}s, net={best_net:.2f}",
            xy=(best_cycle, best_net),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9,
            color="#d62728",
        )

    ax.set_xlabel("Cycle C (s)")
    ax.set_ylabel("band_score - intersection_loss")
    ax.set_title("Stage 1: band_score - intersection_loss vs cycle")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    path = OUTPUT_DIR / "cycle_sweep_iterative_maxband_score.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    cycles = []
    c = CYCLE_START
    while c <= CYCLE_STOP + 1e-9:
        cycles.append(round(c, 6))
        c += CYCLE_STEP

    rows: list[dict[str, object]] = []
    print("开始周期扫描：Stage 1 max-band (3.1 global_sum)")
    for cycle in cycles:
        row = solve_one_cycle(cycle)
        rows.append(row)
        score = row["band_score"]
        score_text = "nan" if isinstance(score, float) and math.isnan(score) else f"{float(score):.6f}"
        status = str(row["status"])
        print(f"  C={cycle:5.1f}s  score={score_text:>12s}  status={status}")

    csv_path = save_csv(rows)
    png_path = plot_curve(rows)
    print(f"CSV 已保存到 {csv_path}")
    print(f"折线图已保存到 {png_path}")


if __name__ == "__main__":
    main()
