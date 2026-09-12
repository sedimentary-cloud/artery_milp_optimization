"""6 路口一般化案例：单阶段与双阶段 sum / balance / one-way 对比。

保留 6 路口多相位、相位不对称、路段不对称、非等速配置。
共 6 个案例，每个案例只输出一张时空图：

    1.1 单阶段 sum
    1.2 单阶段 balance
    1.3 单阶段 one-way

    2.1 双阶段 sum
    2.2 双阶段 balance
    2.3 双阶段 one-way
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from greenwave import (Arterial, GreenWindow, Intersection, Phase, Segment,
                       SignalPlan, plot_time_space)
from greenwave.solvers import (AlignmentLossBuilder, ConstraintBuilder,
                               EpsilonConstraintRunner, CompositeBandSolver,
                               LinearSpec, OneWayPrioritySolver,
                               PhaseLossBuilder, PhaseLossSpec,
                               PhaseTuneSolver, TwoStageSolver)

C = 90.0


def make_plan(name, phases, up_phase, down_phase, lost_time):
    """根据相位初始时长生成 Stage 1 需要的固定绿灯窗。"""
    starts = {}
    acc = 0.0
    for ph in phases:
        starts[ph.name] = acc
        acc += ph.green

    def dur(p):
        return next(ph.green for ph in phases if ph.name == p)

    us = starts[up_phase]
    ds = starts[down_phase]
    return SignalPlan(
        name=name,
        up_windows=[GreenWindow(us / C, (us + dur(up_phase)) / C)],
        down_windows=[GreenWindow(ds / C, (ds + dur(down_phase)) / C)],
        phases=phases,
        up_phase=up_phase,
        down_phase=down_phase,
        lost_time=lost_time,
    )


plans = [
    make_plan("I1",
              [Phase("P1", 35.0, 20.0, 50.0),
               Phase("P2", 40.0, 25.0, 55.0)],
              "P1", "P2", 15.0),
    make_plan("I2",
              [Phase("P1", 30.0, 15.0, 45.0),
               Phase("P2", 35.0, 20.0, 50.0),
               Phase("P3", 15.0, 10.0, 25.0)],
              "P1", "P2", 10.0),
    make_plan("I3",
              [Phase("P1", 40.0, 20.0, 55.0),
               Phase("P2", 35.0, 20.0, 50.0)],
              "P1", "P2", 15.0),
    make_plan("I4",
              [Phase("P1", 30.0, 15.0, 45.0),
               Phase("P2", 35.0, 20.0, 50.0),
               Phase("P3", 10.0, 5.0, 20.0)],
              "P1", "P2", 15.0),
    make_plan("I5",
              [Phase("P1", 45.0, 25.0, 60.0),
               Phase("P2", 40.0, 25.0, 55.0)],
              "P1", "P2", 5.0),
    make_plan("I6",
              [Phase("P1", 35.0, 20.0, 50.0),
               Phase("P2", 40.0, 25.0, 55.0)],
              "P1", "P2", 15.0),
]

names = ["I1", "I2", "I3", "I4", "I5", "I6"]
arterial = Arterial(
    cycle=C,
    intersections={n: Intersection(n, plans=[plans[i]])
                   for i, n in enumerate(names)},
    segments={
        "seg1": Segment("seg1", 1089.0, 1012.0, 11.0, 11.0),
        "seg2": Segment("seg2", 913.0, 903.0, 11.0, 10.5),
        "seg3": Segment("seg3", 1176.0, 1116.0, 12.0, 12.0),
        "seg4": Segment("seg4", 924.0, 935.0, 11.0, 11.0),
        "seg5": Segment("seg5", 1034.0, 1023.0, 11.0, 11.0),
    },
    order=["I1", "seg1", "I2", "seg2", "I3", "seg3",
           "I4", "seg4", "I5", "seg5", "I6"],
)


def show(tag, sol, save_path, notes=None):
    print(f"[{tag}] status={sol.status}, objective={sol.objective:.2f}")
    print(f"  b_up={sol.bandwidth_up.get('seg1', 0):.2f}  "
          f"b_down={sol.bandwidth_down.get('seg1', 0):.2f}  "
          f"band_loss={sol.band_loss:.2f}  "
          f"intersection_loss={sol.intersection_loss:.2f}")
    if sol.phase_times:
        print(f"  phase_times={sol.phase_times}")
    plot_time_space(arterial, sol, save_path=save_path, notes=notes)
    print(f"  时空图已保存到 {save_path}")




# ======================================================================
# 1.1 单阶段 sum：
#     只做 Stage 1，方案/窗口固定，优化 max b_up + b_down。
# ======================================================================
print("=" * 70)
print("1.1 单阶段 sum（CompositeBandSolver）")
s_1_1 = CompositeBandSolver(
    down_weight=1.0,
    objective_mode="sum",
).solve(arterial)
show("1.1 stage1 sum", s_1_1, "case_1_1_stage1_sum.png",
     notes=["Stage 1 only",
            "Objective: max b_up + b_down"])


# ======================================================================
# 1.2 单阶段 balance：
#     只做 Stage 1，优化总带宽 + eps * min(b_up, b_down)。
# ======================================================================
print("=" * 70)
print("1.2 单阶段 balance（CompositeBandSolver, balanced_composite）")
s_1_2 = CompositeBandSolver(
    down_weight=1.0,
    objective_mode="balanced_composite",
    balance_eps=0.1,
).solve(arterial)
show("1.2 stage1 balance", s_1_2, "case_1_2_stage1_balance.png",
     notes=["Stage 1 only",
            "Objective: max b_up + b_down + eps * B_bal",
            "B_bal = min(b_up, b_down)"])


# ======================================================================
# 1.3 单阶段 one-way：
#     只做 Stage 1，上行全局带 + 下行分段/窗口带加权。
# ======================================================================
print("=" * 70)
print("1.3 单阶段 one-way（OneWayPrioritySolver）")
s_1_3 = OneWayPrioritySolver(
    up_weight=1.0,
    window_weights={2: 1.0, 3: 0.5},
    n_intersections=len(names),
).solve(arterial)
show("1.3 stage1 one-way", s_1_3, "case_1_3_stage1_oneway.png",
     notes=["Stage 1 only",
            "Objective: up-priority + down window bands",
            "window_weights: win2=1.0, win3=0.5"])


# ======================================================================
# 2.1 双阶段 sum：
#     Stage 1 选方案/固定窗口，Stage 2 锁方案后优化相位 g 和带宽。
# ======================================================================
print("=" * 70)
print("2.1 双阶段 sum（TwoStageSolver）")
s_2_1 = TwoStageSolver(
    CompositeBandSolver(down_weight=1.0, objective_mode="sum"),
    mode="global",
    down_weight=1.0,
    objective_mode="sum",
).solve(arterial)
show("2.1 two-stage sum", s_2_1, "case_2_1_stage2_sum.png",
     notes=["Two-stage",
            "Stage 1: max b_up + b_down",
            "Stage 2: phase durations optimized"])


# ======================================================================
# 2.2 双阶段 balance：
#     Stage 1 用 balanced_composite 选方案；
#     Stage 2 继续用同一目标优化相位 g。
# ======================================================================
print("=" * 70)
print("2.2 双阶段 balance（TwoStageSolver, balanced_composite）")
s_2_2 = TwoStageSolver(
    CompositeBandSolver(
        down_weight=1.0,
        objective_mode="balanced_composite",
        balance_eps=0.1,
    ),
    mode="global",
    down_weight=1.0,
    objective_mode="balanced_composite",
    balance_eps=0.1,
).solve(arterial)
show("2.2 two-stage balance", s_2_2, "case_2_2_stage2_balance.png",
     notes=["Two-stage",
            "Objective: b_up + b_down + eps * B_bal",
            "Stage 2: phase durations optimized"])


# ======================================================================
# 2.3 双阶段 one-way：
#     Stage 1 选方案/窗口，Stage 2 优化相位 g 后保持 one-way 目标。
# ======================================================================
print("=" * 70)
print("2.3 双阶段 one-way（TwoStageSolver, oneway）")
s_2_3 = TwoStageSolver(
    OneWayPrioritySolver(
        up_weight=1.0,
        window_weights={2: 1.0, 3: 0.5},
        n_intersections=len(names),
    ),
    mode="oneway",
    up_weight=1.0,
    window_weights={2: 1.0, 3: 0.5},
).solve(arterial)
show("2.3 two-stage one-way", s_2_3, "case_2_3_stage2_oneway.png",
     notes=["Two-stage",
            "Objective: up-priority + down window bands",
            "window_weights: win2=1.0, win3=0.5",
            "Stage 2: phase durations optimized"])
