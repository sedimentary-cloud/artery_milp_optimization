"""局部绿波带时间范围示例。

这个示例展示：
- Stage 1 求解后自动回填 `Solution.window_band_ranges`；
- 如何读取内部相邻路口窗口带（非全局带）的时间范围；
- 如何给单个路口配置多个可选 `SignalPlan`；
- 如何在同一个 `SignalPlan` 内配置多个 `up/down segments`；
- 如何把路口内约束/软损失直接绑定到某个 `SignalPlan`；
- 如何给方案配置第二阶段可调范围 `metadata["term_bounds"]`；
- 如何配置硬/软边距 `BandMarginConfig`；
- 如何在 Stage 1 之后继续运行 Stage 2，查看 term_bounds 带来的端点微调；
- 如何把该结果直接导出为 JSON，供绘图或外部系统消费。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artery_milp import plot_time_space
from artery_milp.models import (Arterial, GreenWindow, Intersection, Segment,
                                SignalConstraint, SignalLoss, SignalPlan)
from artery_milp.solvers.core import BandMarginConfig, ObjectiveConfig, SumGroup
from artery_milp.solvers.stage1 import SegmentedBandSolver
from artery_milp.solvers.stage2 import FullFlexiblePhaseTuneSolver

CYCLE = 90.0


def make_windows(values: list[tuple[float, float]]) -> list[GreenWindow]:
    """函数名：make_windows；参数：比例区间列表；返回值：GreenWindow 列表；异常：ValueError。"""
    return [GreenWindow(start, end) for start, end in values]


def build_window_range_case() -> Arterial:
    """函数名：build_window_range_case；参数：无；返回值：Arterial；异常：ValueError。"""
    plans = {
        "I1": [
            SignalPlan(
                name="baseline",
                up_segments=make_windows([(0.05, 0.64)]),
                down_segments=make_windows([(0.38, 0.76)]),
            ),
        ],
        "I2": [
            SignalPlan(
                name="baseline",
                up_segments=make_windows([(0.17, 0.36)]),
                down_segments=make_windows([(0.45, 0.64)]),
            ),
            SignalPlan(
                name="split_priority",
                up_segments=make_windows([(0.15, 0.29), (0.31, 0.92)]),
                down_segments=make_windows([(0.23, 0.66), (0.68, 0.78)]),
                signal_constraints=[
                    SignalConstraint(
                        terms={"up.2.start": 1.0, "up.1.end": -1.0},
                        sense=">=",
                        rhs=0.02,
                        name="上行两段之间至少保留 0.02 周期间隔",
                    ),
                    SignalConstraint(
                        terms={"down.2.start": 1.0, "down.1.end": -1.0},
                        sense=">=",
                        rhs=0.02,
                        name="下行两段之间至少保留 0.02 周期间隔",
                    ),
                ],
                signal_losses=[
                    SignalLoss(
                        terms={"down.1.start": 1.0},
                        lower_threshold=0.28,
                        upper_threshold=0.34,
                        lower_slope=1.0,
                        upper_slope=1.0,
                        name="希望 split_priority 的下行首段不要贴住加宽后的左边缘",
                    ),
                ],
                # Stage 2 只会在这些范围内微调对应端点；未列出的端点保持名义值。
                metadata={
                    "term_bounds": {
                        "up.1.start":   (0.13, 0.17),
                        "up.1.end":     (0.27, 0.31),
                        "up.2.start":   (0.30, 0.33),
                        "down.1.start": (0.28, 0.34),
                        "down.1.end":   (0.58, 0.64),
                        "down.2.start": (0.66, 0.72),
                    }
                },
            ),
        ],
        "I3": [
            SignalPlan(
                name="baseline",
                up_segments=make_windows([(0.29, 0.68)]),
                down_segments=make_windows([(0.33, 0.82)]),
            ),
            SignalPlan(
                name="split_balanced",
                up_segments=make_windows([(0.27, 0.40), (0.42, 0.55)]),
                down_segments=make_windows([(0.31, 0.56), (0.58, 0.68)]),
                signal_constraints=[
                    SignalConstraint(
                        terms={"down.1.end": 1.0, "up.1.start": -1.0},
                        sense=">=",
                        rhs=0.04,
                        name="下行首段结束需晚于上行首段开始至少 0.04 周期",
                    ),
                    SignalConstraint(
                        terms={"down.2.start": 1.0, "down.1.end": -1.0},
                        sense=">=",
                        rhs=0.02,
                        name="下行两段之间至少保留 0.02 周期间隔",
                    ),
                ],
                signal_losses=[
                    SignalLoss(
                        terms={"up.2.end": 1.0},
                        lower_threshold=0.51,
                        upper_threshold=0.55,
                        lower_slope=1.0,
                        upper_slope=1.0,
                        name="希望 split_balanced 的上行第二段结束时刻落在窗口内",
                    ),
                ],
            ),
        ],
        "I4": [
            SignalPlan(
                name="baseline",
                up_segments=make_windows([(0.41, 0.80)]),
                down_segments=make_windows([(0.21, 0.90)]),
            ),
        ],
    }
    return Arterial(
        cycle=CYCLE,
        intersections={
            "I1": Intersection("I1", plans=plans["I1"]),
            "I2": Intersection("I2", plans=plans["I2"]),
            "I3": Intersection("I3", plans=plans["I3"]),
            "I4": Intersection("I4", plans=plans["I4"]),
        },
        segments={
            "S12": Segment("S12", length_up=190.0, length_down=188.0, speed_up=14.0, speed_down=14.0),
            "S23": Segment("S23", length_up=205.0, length_down=200.0, speed_up=14.0, speed_down=14.0),
            "S34": Segment("S34", length_up=185.0, length_down=180.0, speed_up=14.0, speed_down=14.0),
        },
        order=["I1", "S12", "I2", "S23", "I3", "S34", "I4"],
    )


def build_objective() -> ObjectiveConfig:
    """函数名：build_objective；参数：无；返回值：ObjectiveConfig；异常：ValueError。"""
    return ObjectiveConfig(sum_groups=[SumGroup({
        "up.global": 0.6,
        "down.seg2": 0.4,
        "down.win3@I2-I4": 1.6,
    })])


def build_margin() -> BandMarginConfig:
    """硬边距保证不贴边，软边距进一步追求居中。"""
    return BandMarginConfig(
        hard_margin_up=0.01,
        hard_margin_down=0.01,
        soft_margin_up=0.03,
        soft_margin_down=0.03,
        penalty_up=1.0,
        penalty_down=1.0,
    )


def main() -> None:
    """函数名：main；参数：无；返回值：无；异常：RuntimeError。"""
    arterial = build_window_range_case()
    margin = build_margin()
    solver = SegmentedBandSolver(
        config=build_objective(),
        up_global_output=True,
        down_global_output=False,
        margin=margin,
    )
    solution = solver.solve(arterial)
    if solution.status != "optimal":
        raise RuntimeError(f"unexpected solver status: {solution.status}")

    focus_key = "down.win3@I2-I4"
    focus_ranges = solution.window_band_ranges.get(focus_key, [])
    if not focus_ranges:
        raise RuntimeError(f"{focus_key} was not generated")

    output_dir = Path(__file__).resolve().parent
    json_path = output_dir / "window_band_ranges_case_output.json"
    image_path = output_dir / "window_band_ranges_case_time_space.png"
    json_path.write_text(
        json.dumps(solution.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_time_space(
        arterial,
        solution,
        save_path=str(image_path),
        notes=[f"Focus: {focus_key}"],
    )

    print("=" * 80)
    print("局部绿波带时间范围示例")
    print(f"求解状态: {solution.status}")
    print(f"选中的方案: {solution.plan_choices}")
    print(f"band_objective={solution.band_objective:.3f}")
    print(f"重点窗口带: {focus_key}")
    for idx, item in enumerate(focus_ranges, start=1):
        print(f"  实例 {idx}: bandwidth={item['bandwidth']:.2f}s, time=[{item['time_min']:.2f}, {item['time_max']:.2f}]")
        for name, band_range in item["intersection_ranges"].items():
            print(
                f"    {name}: start={band_range['start']:.2f}s, "
                f"end={band_range['end']:.2f}s"
            )
    print(f"JSON 已保存到 {json_path}")
    print(f"时空图已保存到 {image_path}")

    # ------------------------------------------------------------------
    # Stage 2：读取 SignalPlan.metadata["term_bounds"]，在给定范围内微调段端点。
    # Stage 1 使用固定窗口；只有 Stage 2 才会真正执行这里的上下界。
    # ------------------------------------------------------------------
    tuner = FullFlexiblePhaseTuneSolver(
        config=build_objective(),
        max_loops=3,
        margin=margin,
    )
    tuned = tuner.solve(arterial, prior=solution, band_loss_weight=0.5)
    if tuned.status != "optimal":
        raise RuntimeError(f"unexpected stage2 status: {tuned.status}")

    tuned_json_path = output_dir / "window_band_ranges_case_stage2_output.json"
    tuned_json_path.write_text(
        json.dumps(tuned.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("-" * 80)
    print("Stage 2 端点微调（term_bounds 生效）")
    print(f"求解状态: {tuned.status}")
    print(f"选中的方案: {tuned.plan_choices}")

    plan_name = tuned.plan_choices.get("I2")
    plan = arterial.intersections["I2"].plan_by_name(plan_name)
    bounds = plan.metadata.get("term_bounds", {})
    if bounds:
        for term in sorted(bounds):
            new_seconds = tuned.segment_times.get("I2", {}).get(term)
            if new_seconds is None:
                continue
            lower, upper = bounds[term]
            print(
                f"  {term}: 名义 {plan.term_value(term) * CYCLE:.2f}s -> "
                f"调整后 {new_seconds:.2f}s, "
                f"允许范围 [{lower * CYCLE:.2f}, {upper * CYCLE:.2f}]s"
            )
    else:
        print(f"  I2 选中方案 {plan_name!r} 没有配置 term_bounds")

    print(
        f"  band_objective={tuned.band_objective:.3f}, "
        f"band_loss={tuned.band_loss:.3f}, "
        f"band_score={tuned.band_score:.3f}, "
        f"intersection_loss={tuned.intersection_loss:.3f}"
    )
    print(f"Stage 2 JSON 已保存到 {tuned_json_path}")


if __name__ == "__main__":
    main()
