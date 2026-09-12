"""6 路口一般化案例：单阶段与双阶段 sum / balance / one-way 对比。

保留 6 路口多相位、相位不对称、路段不对称、非等速配置。
共 6 个案例，每个案例只输出一张时空图：

    1.1 单阶段 sum
    1.2 单阶段 balance
    1.3 单阶段 one-way

    2.1 双阶段 sum
    2.2 双阶段 balance
    2.3 双阶段 one-way

    3.1 双阶段 sum + 硬约束：某相位不能太短
    3.2 双阶段 sum + 增大 alignment loss 权重
    3.3 双阶段 sum + 硬约束：两个相位之和不能过大

    4.1 加入 intersection loss 后，对比单阶段与双阶段的 Pareto 前沿
        本案例只输出 Pareto 图，不输出时空图
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from greenwave import (Arterial, GreenWindow, Intersection, Phase, Segment,
                       SignalPlan, plot_time_space)
from greenwave.solvers import (AlignmentLossBuilder, ConstraintBuilder,
                               CompositeBandSolver, LinearSpec,
                               OneWayPrioritySolver, PhaseLossBuilder,
                               PhaseLossSpec, PhaseTuneSolver, TwoStageSolver,
                               BandObjectiveConfig, IntersectionLossConfig,
                               TwoStageConfig, composite_config,
                               oneway_config)

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
cfg_2_1 = TwoStageConfig(
    band=BandObjectiveConfig(
        mode="global",
        objective=composite_config(down_weight=1.0, objective_mode="sum"),
    ),
)
s_2_1 = TwoStageSolver(cfg_2_1).solve(arterial)
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
cfg_2_2 = TwoStageConfig(
    band=BandObjectiveConfig(
        mode="global",
        objective=composite_config(
            down_weight=1.0,
            objective_mode="balanced_composite",
            balance_eps=0.1,
        ),
    ),
)
s_2_2 = TwoStageSolver(cfg_2_2).solve(arterial)
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
cfg_2_3 = TwoStageConfig(
    band=BandObjectiveConfig(
        mode="oneway",
        objective=oneway_config(
            up_weight=1.0,
            window_weights={2: 1.0, 3: 0.5},
            n_intersections=len(names),
        ),
    ),
)
s_2_3 = TwoStageSolver(cfg_2_3).solve(arterial)
show("2.3 two-stage one-way", s_2_3, "case_2_3_stage2_oneway.png",
     notes=["Two-stage",
            "Objective: up-priority + down window bands",
            "window_weights: win2=1.0, win3=0.5",
            "Stage 2: phase durations optimized"])


# ======================================================================
# 3.1 双阶段 sum + 硬约束：选一个绿灯相位时间不能太短
#     约束：I2.P1 >= 35
# ======================================================================
print("=" * 70)
print("3.1 双阶段 sum + 硬约束：I2.P1 >= 35")
constraint_3_1 = ConstraintBuilder([
    LinearSpec({"I2.P1": 1.0}, sense=">=", rhs=35.0),
])
cfg_3_1 = TwoStageConfig(
    band=BandObjectiveConfig(
        mode="global",
        objective=composite_config(down_weight=1.0, objective_mode="sum"),
    ),
    intersection=IntersectionLossConfig(
        constraint_builder=constraint_3_1,
    ),
)
s_3_1 = TwoStageSolver(cfg_3_1).solve(arterial)
show("3.1 two-stage sum + min green", s_3_1,
     "case_3_1_stage2_sum_min_green.png",
     notes=["Two-stage sum",
            "Hard constraint: I2.P1 >= 35",
            "Prevent one green phase from being too short"])


# ======================================================================
# 3.2 双阶段 sum + 增大 alignment loss 权重
#     对齐目标：由上游累计行驶时间推出的带中心
# ======================================================================
up_center_target = {}
acc = 0.0
up_center_target[names[0]] = 0.0
for idx, seg in enumerate(arterial.segment_order):
    acc += seg.travel_time_up
    up_center_target[names[idx + 1]] = acc % C

alignment_builder_3_2 = AlignmentLossBuilder(
    targets_up=up_center_target,
    tolerance=5.0,
    weight_up=1.0,
)
print("=" * 70)
print("3.2 双阶段 sum + alignment loss 权重=10")
cfg_3_2 = TwoStageConfig(
    band=BandObjectiveConfig(
        mode="global",
        objective=composite_config(down_weight=1.0, objective_mode="sum"),
        alignment_builder=alignment_builder_3_2,
        band_loss_weight=10.0,
    ),
)
s_3_2 = TwoStageSolver(cfg_3_2).solve(arterial)
show("3.2 two-stage sum + alignment weight", s_3_2,
     "case_3_2_stage2_sum_align_weight.png",
     notes=["Two-stage sum",
            "Alignment loss weight = 10",
            "Band center pulled toward arrival target"])


# ======================================================================
# 3.3 双阶段 sum + 硬约束：两个相位之和不能过大
#     约束：I2.P1 + I2.P2 <= 70
# ======================================================================
print("=" * 70)
print("3.3 双阶段 sum + 硬约束：I2.P1 + I2.P2 <= 70")
constraint_3_3 = ConstraintBuilder([
    LinearSpec({"I2.P1": 1.0, "I2.P2": 1.0}, sense="<=", rhs=70.0),
])
cfg_3_3 = TwoStageConfig(
    band=BandObjectiveConfig(
        mode="global",
        objective=composite_config(down_weight=1.0, objective_mode="sum"),
    ),
    intersection=IntersectionLossConfig(
        constraint_builder=constraint_3_3,
    ),
)
s_3_3 = TwoStageSolver(cfg_3_3).solve(arterial)
show("3.3 two-stage sum + max phase sum", s_3_3,
     "case_3_3_stage2_sum_max_phase_sum.png",
     notes=["Two-stage sum",
            "Hard constraint: I2.P1 + I2.P2 <= 70",
            "Limit sum of two green phases"])


# ======================================================================
# 4.1 加入 intersection loss 后：
#     对比单阶段固定相位点与双阶段 Pareto 前沿。
#     本案例只绘制 Pareto 图，不输出时空图。
# ======================================================================
print("=" * 70)
print("4.1 单阶段 vs 双阶段 Pareto：band_objective vs intersection_loss")

loss_builder_4_1 = PhaseLossBuilder([
    PhaseLossSpec("P1", threshold=35.0, slope=2.0, intersection="I2"),
    PhaseLossSpec("P2", threshold=40.0, slope=1.0, intersection="I4"),
    PhaseLossSpec("P1", threshold=50.0, slope=2.5, intersection="I5"),
    PhaseLossSpec("P2", threshold=45.0, slope=3.0, intersection="I1"),
])

# 单阶段固定相位：不调 g，只优化带宽和带前沿。
s4_stage1 = PhaseTuneSolver(
    mode="global",
    down_weight=1.0,
    tunable_intersections=set(),
).solve(
    arterial,
    prior=s_1_1,
    loss_builder=loss_builder_4_1,
    objective="bandwidth",
)

# 双阶段：扫描 intersection_loss 上界，最大化 band_objective。
tuner_4_1 = PhaseTuneSolver(mode="global", down_weight=1.0)
s4_hi = tuner_4_1.solve(
    arterial,
    prior=s_1_1,
    loss_builder=loss_builder_4_1,
    objective="bandwidth",
)
s4_lo = tuner_4_1.solve(
    arterial,
    prior=s_1_1,
    loss_builder=loss_builder_4_1,
    objective="loss",
)

stage2_points_4_1 = []
if s4_hi.status == "optimal" and s4_lo.status == "optimal":
    L_hi = float(s4_hi.intersection_loss)
    L_lo = float(s4_lo.intersection_loss)
    n_points = 10
    if n_points <= 1 or abs(L_hi - L_lo) < 1e-9:
        eps_values = [L_lo]
    else:
        step = (L_hi - L_lo) / (n_points - 1)
        eps_values = [L_lo + i * step for i in range(n_points)]

    for eps in eps_values:
        s = tuner_4_1.solve(
            arterial,
            prior=s_1_1,
            loss_builder=loss_builder_4_1,
            max_intersection_loss=eps,
            objective="bandwidth",
        )
        if s.status == "optimal":
            stage2_points_4_1.append((
                float(s.intersection_loss),
                float(s.band_objective),
                float(eps),
                s,
            ))


def _nondominated_4_1(points):
    """非支配：intersection_loss 越小、band_objective 越大越好。"""
    front = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            no_worse = q[0] <= p[0] + 1e-9 and q[1] >= p[1] - 1e-9
            strictly_better = q[0] < p[0] - 1e-9 or q[1] > p[1] + 1e-9
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(p)
    unique = {}
    for p in front:
        key = (round(p[0], 6), round(p[1], 6))
        unique[key] = p
    return sorted(unique.values(), key=lambda t: (t[0], -t[1]))


front_4_1 = _nondominated_4_1(stage2_points_4_1)

print("  单阶段固定相位点：")
if s4_stage1.status != "optimal":
    print("    infeasible")
else:
    print("    intersection_loss   band_objective")
    print(f"    {s4_stage1.intersection_loss:>17.2f}   "
          f"{s4_stage1.band_objective:>13.2f}")

print("  双阶段 Pareto 点：")
print("    eps      intersection_loss   band_objective")
for loss, obj, eps, _ in front_4_1:
    print(f"    {eps:>5.2f}    {loss:>17.2f}   {obj:>13.2f}")

fig, ax = plt.subplots(figsize=(9, 5.5))
if s4_stage1.status == "optimal":
    ax.scatter(
        [s4_stage1.intersection_loss],
        [s4_stage1.band_objective],
        marker="*", s=260, color="#d95f02",
        label="Stage 1 only (fixed phases)", zorder=4,
    )
    ax.annotate(
        "Stage 1 only",
        (s4_stage1.intersection_loss, s4_stage1.band_objective),
        textcoords="offset points", xytext=(8, 8),
        fontsize=9, color="#d95f02",
    )

if front_4_1:
    xs = [p[0] for p in front_4_1]
    ys = [p[1] for p in front_4_1]
    ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=7,
            color="#1b9e77", label="Two-stage Pareto front", zorder=3)
    for loss, obj, eps, _ in front_4_1:
        ax.annotate(f"eps={eps:.1f}", (loss, obj),
                    textcoords="offset points", xytext=(6, -12),
                    fontsize=8, color="#1b9e77")

ax.set_xlabel("Intersection loss")
ax.set_ylabel("Band objective")
ax.set_title("Case 4.1: Stage 1 only vs Two-stage Pareto Front")
ax.grid(alpha=0.3)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
fig.savefig("case_4_1_intersection_loss_pareto.png", dpi=150)
plt.close(fig)

print()
print("Case 4.1 Pareto 图已保存到 "
      "case_4_1_intersection_loss_pareto.png")
