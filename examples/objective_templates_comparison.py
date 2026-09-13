"""ObjectiveConfig 模板对比实验。

基于 ``examples/window_band_ranges_case.py`` 里的干线案例，比较：

1. 单阶段 SegmentedBandSolver
2. 双阶段 TwoStageSolver
3. 迭代两阶段 IterativeTwoStageSolver

三种优化目标模板：

1.1 最大全局带宽和
1.2 均衡全局带宽
1.3 单向优化（下行权重小）

共输出 10 张图：

1.1, 1.2, 1.3
2.1, 2.2, 2.3
3.1, 3.2, 3.3
4.1

其中 4.1 对比：
- 单阶段 1.1 的单点；
- 迭代 3.1 稳定解作为 prior 时的 Pareto 前沿。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artery_milp import plot_pareto_frontier, plot_time_space
from artery_milp.models import SignalLoss
from artery_milp.solution import Solution
from artery_milp.solvers.core import (BandMarginConfig, ObjectiveConfig,
                                      build_objective_config)
from artery_milp.solvers.pipeline import (BandObjectiveConfig,
                                          EpsilonConstraintRunner,
                                          IterativeTwoStageSolver,
                                          TwoStageConfig,
                                          TwoStageSolver)
from artery_milp.solvers.stage1 import SegmentedBandSolver
from artery_milp.examples.window_band_ranges_case import (
    build_margin,
    build_window_range_case,
)

OUTPUT_DIR = Path(__file__).resolve().parent


def build_comparison_arterial():
    """返回对比实验使用的干线。

    直接复用 ``window_band_ranges_case.py`` 的案例，但额外给 I4 baseline
    增加两个软损失，用于在 ε-约束扫描中产生更多中间损失水平。
    这样 4.1 的 Pareto 前沿不会退化成只有两个端点。
    """
    arterial = build_window_range_case()
    i4_plan = arterial.intersections["I4"].plans[0]
    i4_plan.signal_losses.extend([
        SignalLoss(
            terms={"up.1.start": 1.0},
            lower_threshold=0.39,
            upper_threshold=0.43,
            lower_slope=1.0,
            upper_slope=1.0,
            name="I4 up.1.start 软区间（对比实验用）",
        ),
        SignalLoss(
            terms={"down.1.start": 1.0},
            lower_threshold=0.19,
            upper_threshold=0.23,
            lower_slope=1.0,
            upper_slope=1.0,
            name="I4 down.1.start 软区间（对比实验用）",
        ),
    ])
    return arterial


def objective_global_sum() -> ObjectiveConfig:
    """模板 1.1：最大全局带宽和。"""
    return build_objective_config(
        mode="global",
        objective_mode="sum",
        up_weight=1.0,
        down_weight=1.0,
    )


def objective_global_balanced() -> ObjectiveConfig:
    """模板 1.2：均衡全局带宽。"""
    return build_objective_config(
        mode="global",
        objective_mode="balanced",
    )


def objective_oneway_down_low() -> ObjectiveConfig:
    """模板 1.3：单向优化，下行方向权重设小。"""
    return build_objective_config(
        mode="oneway",
        up_weight=1.0,
        window_weights={2: 0.05, 3: 0.02},
        n_intersections=4,
        normalize_window_weights=False,
    )


def build_two_stage_config(mode: str, objective: ObjectiveConfig,
                           margin: BandMarginConfig) -> TwoStageConfig:
    """构造统一的 TwoStageConfig。"""
    return TwoStageConfig(
        band=BandObjectiveConfig(
            mode=mode,
            objective=objective,
            band_loss_weight=0.5,
        ),
        margin=margin,
        max_loops=3,
    )


def solve_single_stage(arterial, margin: BandMarginConfig,
                       mode: str, objective: ObjectiveConfig) -> Solution:
    """单阶段：只运行 SegmentedBandSolver。"""
    solver = SegmentedBandSolver(
        config=objective,
        up_global_output=True,
        down_global_output=(mode == "global"),
        margin=margin,
    )
    return solver.solve(arterial, band_loss_weight=0.5)


def solve_two_stage(arterial, margin: BandMarginConfig,
                    mode: str, objective: ObjectiveConfig) -> Solution:
    """双阶段：TwoStageSolver。"""
    config = build_two_stage_config(mode, objective, margin)
    return TwoStageSolver(config=config).solve(arterial)


def solve_iterative(arterial, margin: BandMarginConfig,
                    mode: str, objective: ObjectiveConfig) -> Solution:
    """迭代两阶段：直到窗口分配稳定。"""
    config = build_two_stage_config(mode, objective, margin)
    solver = IterativeTwoStageSolver(config=config, max_iterations=10)
    return solver.solve(arterial)


def save_time_space(arterial, solution: Solution, filename: str, title: str) -> Path:
    """保存一张时空图。"""
    path = OUTPUT_DIR / filename
    plot_time_space(
        arterial,
        solution,
        save_path=str(path),
        notes=[title],
    )
    return path


def global_sum_metric(solution: Solution) -> float:
    """和 EpsilonConstraintRunner(metric='sum') 一致的口径。"""
    bu = next(iter(solution.bandwidth_up.values()), 0.0)
    bd = next(iter(solution.bandwidth_down.values()), 0.0)
    return float(bu + bd)


def main() -> None:
    # 清理旧的对比图，保证每次只留下本次 10 张图和对应顺序。
    for old_path in OUTPUT_DIR.glob("compare_*.png"):
        old_path.unlink()

    arterial = build_comparison_arterial()
    margin = build_margin()

    objective_specs = [
        ("global_sum", "global", objective_global_sum, "Global Sum"),
        ("global_balanced", "global", objective_global_balanced, "Global Balanced"),
        ("oneway_down_low", "oneway", objective_oneway_down_low, "One-way (Down Low)"),
    ]

    stage1_results: dict[str, Solution] = {}
    stage2_results: dict[str, Solution] = {}
    iterative_results: dict[str, Solution] = {}

    # ---------------- 第 1 组：单阶段 ----------------
    for idx, (name, mode, objective_builder, title) in enumerate(objective_specs, start=1):
        objective = objective_builder()
        solution = solve_single_stage(arterial, margin, mode, objective)
        stage1_results[name] = solution
        save_time_space(
            arterial,
            solution,
            f"compare_1_{idx}_stage1_{name}.png",
            f"1.{idx} {title} / Stage 1",
        )

    # ---------------- 第 2 组：双阶段 ----------------
    for idx, (name, mode, objective_builder, title) in enumerate(objective_specs, start=1):
        objective = objective_builder()
        solution = solve_two_stage(arterial, margin, mode, objective)
        stage2_results[name] = solution
        save_time_space(
            arterial,
            solution,
            f"compare_2_{idx}_two_stage_{name}.png",
            f"2.{idx} {title} / TwoStageSolver",
        )

    # ---------------- 第 3 组：迭代两阶段 ----------------
    for idx, (name, mode, objective_builder, title) in enumerate(objective_specs, start=1):
        objective = objective_builder()
        solution = solve_iterative(arterial, margin, mode, objective)
        iterative_results[name] = solution
        save_time_space(
            arterial,
            solution,
            f"compare_3_{idx}_iterative_{name}.png",
            f"3.{idx} {title} / IterativeTwoStageSolver",
        )

    # ---------------- 第 4 组：Pareto 对比 ----------------
    # 使用 1.1 最大全局带宽和目标下的单阶段解，与 3.1 迭代稳定解作为 prior
    # 的 ε-约束 Pareto 前沿对比。
    global_sum_objective = objective_global_sum()
    config_4 = build_two_stage_config("global", global_sum_objective, margin)

    runner = EpsilonConstraintRunner(
        config=config_4,
        n_points=20,
        metric="sum",
    )
    prior_for_pareto = iterative_results["global_sum"]  # 3.1 迭代稳定解
    frontier = runner.run(arterial, prior_for_pareto)
    print(f"4.1 Pareto 前沿点数: {len(frontier)}")

    ax = plot_pareto_frontier(
        frontier,
        save_path=None,
        title="4.1 Pareto: Global Sum (Iterative prior) vs Stage 1 point",
        annotate=True,
        knee_point=runner.knee_point(),
    )

    stage1_point_x = global_sum_metric(stage1_results["global_sum"])
    stage1_point_y = stage1_results["global_sum"].intersection_loss
    ax.scatter(
        [stage1_point_y],
        [stage1_point_x],
        color="#d62728",
        marker="*",
        s=140,
        zorder=5,
        label="Stage 1 global sum",
    )
    ax.annotate(
        "1.1 single-stage",
        (stage1_point_y, stage1_point_x),
        textcoords="offset points",
        xytext=(8, -12),
        fontsize=9,
        color="#d62728",
    )
    ax.legend(loc="best")

    figure_path = OUTPUT_DIR / "compare_4_1_pareto_global_sum.png"
    ax.figure.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)

    print("对比实验完成，图片顺序：")
    for filename in [
        "compare_1_1_stage1_global_sum.png",
        "compare_1_2_stage1_global_balanced.png",
        "compare_1_3_stage1_oneway_down_low.png",
        "compare_2_1_two_stage_global_sum.png",
        "compare_2_2_two_stage_global_balanced.png",
        "compare_2_3_two_stage_oneway_down_low.png",
        "compare_3_1_iterative_global_sum.png",
        "compare_3_2_iterative_global_balanced.png",
        "compare_3_3_iterative_oneway_down_low.png",
        "compare_4_1_pareto_global_sum.png",
    ]:
        print(f"  {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
