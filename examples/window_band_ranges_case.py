"""局部绿波带时间范围示例。

这个示例按顺序演示一个完整的绿波带优化流程：

1. 构造一条 4 路口干线
   - 每个路口可以有一个或多个候选 `SignalPlan`；
   - 每个方案里，上行/下行都可以有 1 段或多段绿灯窗口；
   - 方案内部可以带硬约束 `SignalConstraint` 和软损失 `SignalLoss`；
   - 方案可以给 Stage 2 配置端点可调范围 `metadata["term_bounds"]`。

2. 配置目标 `ObjectiveConfig`
   - `up.global`：所有上行全局 band；
   - `down.seg2`：下行第二个局部路段对应的 2 路口窗口带；
   - `down.win3@I2-I4`：从 I2 到 I4 的 3 路口局部窗口带。

3. 配置边距 `BandMarginConfig`
   - 硬边距：绿波带不能贴住绿灯窗口边缘；
   - 软边距：Stage 2 进一步希望带子居中，否则产生 `band_loss`。

4. 运行 Stage 1：`SegmentedBandSolver`
   - 选择每个路口的信号方案；
   - 为每条 band 在每个路口自由选择绿灯窗口；
   - 在固定窗口下优化带宽；
   - 求解后自动回填所有方向、所有长度的 `window_band_ranges`。

5. 运行 Stage 2：`FullFlexiblePhaseTuneSolver`
   - 固定 Stage 1 选出的方案和窗口分配；
   - 只在 `term_bounds` 范围内微调绿灯窗口端点；
   - 同时考虑 `SignalConstraint` / `SignalLoss`、硬/软边距。

6. 运行迭代两阶段：`IterativeTwoStageSolver`
   - Stage 2 调完端点后，把新窗口写回 Stage 1；
   - 再跑 Stage 1，直到窗口分配稳定或进入循环。

7. 导出结果
   - JSON：给外部系统读取；
   - 时空图 PNG：Stage 1、Stage 2、迭代结果各一张。
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
from artery_milp.solvers.pipeline import (BandObjectiveConfig,
                                          IterativeTwoStageSolver,
                                          TwoStageConfig)
from artery_milp.solvers.stage1 import SegmentedBandSolver
from artery_milp.solvers.stage2 import FullFlexiblePhaseTuneSolver

CYCLE = 90.0


def make_windows(values: list[tuple[float, float]]) -> list[GreenWindow]:
    """把 ``[(start, end), ...]`` 转成 ``GreenWindow`` 列表。

    这里的 start/end 都是“占周期的比例”：
    - 0.10 表示周期 10% 的位置；
    - 0.30 表示周期 30% 的位置。

    例如 CYCLE=90s 时：
    - (0.10, 0.30) -> 第 9s 到第 27s 的绿灯窗口。
    """
    return [GreenWindow(start, end) for start, end in values]


def build_window_range_case() -> Arterial:
    """构造这个示例使用的 4 路口干线。

    结构是：

        I1 --S12-- I2 --S23-- I3 --S34-- I4

    上行方向是 I1 -> I2 -> I3 -> I4；
    下行方向是 I4 -> I3 -> I2 -> I1。

    每个路口可以有多个候选方案；
    每个方案里，上行/下行又可以有 1 段或多段绿灯窗口。
    """
    # plans 是“路口名 -> 候选方案列表”。
    # 每个路口至少要有 1 个方案；方案里上下行都要有绿灯窗口。
    plans = {
        # I1 只有一个方案 baseline。
        # 上行绿灯窗口在周期 5%~64%，下行在 38%~76%。
        "I1": [
            SignalPlan(
                name="baseline",
                up_segments=make_windows([(0.05, 0.64)]),
                down_segments=make_windows([(0.38, 0.76)]),
            ),
        ],
        # I2 有两个候选方案：
        # - baseline：上下行各 1 段绿灯；
        # - split_priority：上下行各 2 段绿灯，并带方案内部约束/软损失/Stage 2 可调范围。
        "I2": [
            SignalPlan(
                name="baseline",
                up_segments=make_windows([(0.17, 0.36)]),
                down_segments=make_windows([(0.45, 0.64)]),
            ),
            # split_priority 把上下行都拆成两段。
            # 多段之间的间隔可以表达“黄灯/全红/损失时间”等物理含义；
            # 这里用 SignalConstraint 明确要求两段之间至少留 0.02 个周期。
            SignalPlan(
                name="split_priority",
                up_segments=make_windows([(0.15, 0.29), (0.31, 0.92)]),
                down_segments=make_windows([(0.23, 0.66), (0.68, 0.78)]),
                # SignalConstraint 是“方案内部硬约束”。
                # up.2.start - up.1.end >= 0.02 表示：
                # 第 2 段上行绿灯开始，至少比第 1 段结束晚 0.02 个周期。
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
                # SignalLoss 是“方案内部软损失”。
                # 它希望 down.1.start 落在 0.28~0.34 之间；
                # 太早或太晚都会产生 intersection_loss。
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
                    # 每个区间以名义值为中心，把可调宽度扩大到原来的 2 倍。
                    "term_bounds": {
                        "up.1.start":   (0.11, 0.19),
                        "up.1.end":     (0.25, 0.33),
                        "up.2.start":   (0.285, 0.345),
                        "down.1.start": (0.25, 0.37),
                        "down.1.end":   (0.55, 0.67),
                        "down.2.start": (0.63, 0.75),
                    }
                },
            ),
        ],
        # I3 也有两个候选方案：
        # - baseline：上下行各 1 段；
        # - split_balanced：上下行各 2 段，并带方案内部约束/软损失。
        "I3": [
            SignalPlan(
                name="baseline",
                up_segments=make_windows([(0.29, 0.68)]),
                down_segments=make_windows([(0.33, 0.82)]),
            ),
            # split_balanced 同样是多段方案。
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
        # I4 只有一个 baseline 方案。
        "I4": [
            SignalPlan(
                name="baseline",
                up_segments=make_windows([(0.41, 0.80)]),
                down_segments=make_windows([(0.21, 0.90)]),
            ),
        ],
    }
    # 把上面的方案组装成一个 Arterial：
    # - cycle：公共信号周期，90s；
    # - intersections：路口名 -> Intersection；
    # - segments：物理路段，给出上下行长度和速度；
    # - order：路口/路段交替顺序，例如 I1, S12, I2, S23, ...
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
    """配置 Stage 1 / Stage 2 共用的绿波带目标。

    这里使用一个 SumGroup，也就是加权和：

    - up.global=0.6：
        奖励所有上行全局 band；
    - down.seg2=0.4：
        奖励下行第 2 个局部路段对应的 2 路口窗口带；
        在 4 路口干线里，seg2 对应 I3-I4 这一段。
    - down.win3@I2-I4=1.6：
        奖励从 I2 到 I4 的 3 路口局部窗口带；
        这是本示例重点观察的局部带。

    权重越大，求解器越优先把带宽分配给对应的 band。
    """
    return ObjectiveConfig(sum_groups=[SumGroup({
        "up.global": 0.6,
        "down.seg2": 0.4,
        "down.win3@I2-I4": 1.6,
    })])


def build_margin() -> BandMarginConfig:
    """配置硬边距和软边距。

    - hard_margin_*：
        缩小有效绿灯窗口，绿波带不能贴住窗口边缘；
        Stage 1 和 Stage 2 都生效。
    - soft_margin_*：
        Stage 2 额外希望带子离边缘更远；
        如果不够远，会产生 band_loss。
    - penalty_*：
        soft margin 损失的惩罚权重。

    当前配置：
    - 硬边距 1% 周期；
    - 软边距 3% 周期；
    - 上下行惩罚都是 1.0。
    """
    return BandMarginConfig(
        hard_margin_up=0.01,
        hard_margin_down=0.01,
        soft_margin_up=0.03,
        soft_margin_down=0.03,
        penalty_up=1.0,
        penalty_down=1.0,
    )


def main() -> None:
    """完整示例入口：Stage 1 -> Stage 2 -> 迭代两阶段。

    整体顺序：

    1. 构造 4 路口干线；
    2. 配置目标、边距；
    3. 跑 Stage 1，得到方案选择 + 固定窗口带宽 + 所有局部窗口带；
    4. 导出 Stage 1 JSON / 时空图；
    5. 跑 Stage 2，固定 Stage 1 选择，只微调端点；
    6. 导出 Stage 2 JSON / 时空图；
    7. 跑迭代两阶段，直到窗口分配稳定或进入循环；
    8. 导出迭代 JSON / 时空图，并打印每轮结果。
    """
    # 第 1 步：构造干线和边距配置。
    arterial = build_window_range_case()
    margin = build_margin()
    # 第 2 步：创建 Stage 1 求解器。
    # - config：目标函数；
    # - up_global_output=True：上行带宽摘要按“所有 band 求和”输出；
    # - down_global_output=False：下行摘要仍按逐路段宽度输出；
    # - margin：Stage 1 只使用其中的硬边距。
    solver = SegmentedBandSolver(
        config=build_objective(),
        up_global_output=True,
        down_global_output=False,
        margin=margin,
    )
    # 第 3 步：运行 Stage 1。
    # Stage 1 会：
    # 1) 选择每个路口的方案；
    # 2) 为每条 band 在每个路口选择绿灯窗口；
    # 3) 在固定窗口下最大化目标带宽；
    # 4) 求解后自动补齐所有方向、所有长度的 window_band_ranges。
    solution = solver.solve(arterial)
    if solution.status != "optimal":
        raise RuntimeError(f"unexpected solver status: {solution.status}")

    # 第 4 步：读取重点局部窗口带 down.win3@I2-I4。
    # window_band_ranges[key] 里每个实例包含：
    # - direction / band_no / bandwidth
    # - intersections
    # - time_min / time_max
    # - intersection_ranges：每个路口的起止时间
    focus_key = "down.win3@I2-I4"
    focus_ranges = solution.window_band_ranges.get(focus_key, [])
    if not focus_ranges:
        raise RuntimeError(f"{focus_key} was not generated")

    # 第 5 步：导出 Stage 1 结果。
    # to_dict() 会把 Solution 变成普通 dict，方便 json.dumps。
    output_dir = Path(__file__).resolve().parent
    json_path = output_dir / "window_band_ranges_case_output.json"
    image_path = output_dir / "window_band_ranges_case_time_space.png"
    json_path.write_text(
        json.dumps(solution.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # 画 Stage 1 时空图：
    # - 红条表示红灯；
    # - 绿色条表示绿灯窗口；
    # - 带子会根据 multi_band_starts / window_band_ranges 自动绘制。
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
    # Stage 2：固定 Stage 1 的选择，只微调绿灯窗口端点。
    #
    # Stage 1 使用的是固定绿灯窗口；
    # Stage 2 会读取 SignalPlan.metadata["term_bounds"]，
    # 在上/下界内移动端点 x，同时考虑：
    # - SignalConstraint 硬约束；
    # - SignalLoss 软损失；
    # - hard_margin / soft_margin。
    # ------------------------------------------------------------------
    tuner = FullFlexiblePhaseTuneSolver(
        config=build_objective(),
        max_loops=3,
        margin=margin,
    )
    # prior=solution：把 Stage 1 的选择（方案、窗口、band）固定下来。
    # band_loss_weight=0.5：软边距损失在最终得分里的权重。
    tuned = tuner.solve(arterial, prior=solution, band_loss_weight=0.5)
    if tuned.status != "optimal":
        raise RuntimeError(f"unexpected stage2 status: {tuned.status}")

    tuned_json_path = output_dir / "window_band_ranges_case_stage2_output.json"
    tuned_json_path.write_text(
        json.dumps(tuned.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tuned_image_path = output_dir / "window_band_ranges_case_stage2_time_space.png"
    # Stage 2 时空图：可以看到端点被微调后，绿波带位置和宽度发生变化。
    plot_time_space(
        arterial,
        tuned,
        save_path=str(tuned_image_path),
        notes=["Stage 2: term_bounds tuned endpoints"],
    )

    print("-" * 80)
    print("Stage 2 端点微调（term_bounds 生效）")
    print(f"求解状态: {tuned.status}")
    print(f"选中的方案: {tuned.plan_choices}")

    plan_name = tuned.plan_choices.get("I2")
    plan = arterial.intersections["I2"].plan_by_name(plan_name)
    # 打印 Stage 2 实际的“名义值 -> 调整后值”，验证 term_bounds 是否生效。
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
    print(f"Stage 2 时空图已保存到 {tuned_image_path}")

    # ------------------------------------------------------------------
    # 迭代两阶段：把 Stage 2 调好的窗口端点写回 Stage 1，再重新跑 Stage 1。
    #
    # 停止规则：
    # - 窗口分配连续两轮相同 -> iterative_converged；
    # - 窗口分配进入循环 -> 返回循环第一个解；
    # - 达到 max_iterations -> 返回最后解；
    # - 每轮 Stage 2 必须 optimal，否则抛 RuntimeError。
    # ------------------------------------------------------------------
    iterative_config = TwoStageConfig(
        band=BandObjectiveConfig(
            mode="global",
            objective=build_objective(),
            band_loss_weight=0.5,
        ),
        margin=margin,
        max_loops=3,
    )
    # max_iterations=10：最多迭代 10 轮；通常几轮就会稳定。
    iterative_solver = IterativeTwoStageSolver(
        config=iterative_config,
        max_iterations=10,
    )
    iterative = iterative_solver.solve(arterial)
    if not iterative.status.startswith("optimal"):
        raise RuntimeError(f"unexpected iterative status: {iterative.status}")

    # 导出迭代最终结果，并绘制最终稳定窗口分配下的时空图。
    iterative_json_path = output_dir / "window_band_ranges_case_iterative_output.json"
    iterative_json_path.write_text(
        json.dumps(iterative.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    iterative_image_path = output_dir / "window_band_ranges_case_iterative_time_space.png"
    plot_time_space(
        arterial,
        iterative,
        save_path=str(iterative_image_path),
        notes=["Iterative two-stage: stable window assignment"],
    )

    print("-" * 80)
    print("迭代两阶段求解（IterativeTwoStageSolver）")
    print(f"最终状态: {iterative.status}")
    print(f"迭代轮数: {len(iterative_solver.history)}")
    print(f"选中的方案: {iterative.plan_choices}")
    print(
        f"  band_objective={iterative.band_objective:.3f}, "
        f"band_loss={iterative.band_loss:.3f}, "
        f"band_score={iterative.band_score:.3f}, "
        f"intersection_loss={iterative.intersection_loss:.3f}"
    )
    # history 保存每一轮成功的 Stage 2 结果，便于观察是否稳定。
    print("每轮 band_score / plan_choices:")
    for idx, item in enumerate(iterative_solver.history, start=1):
        print(
            f"  第 {idx} 轮: band_score={item.band_score:.3f}, "
            f"plan_choices={item.plan_choices}"
        )
    print(f"迭代 JSON 已保存到 {iterative_json_path}")
    print(f"迭代时空图已保存到 {iterative_image_path}")


if __name__ == "__main__":
    main()
