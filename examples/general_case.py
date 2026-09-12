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


def make_plan(name, phases, up_phase=None, down_phase=None, lost_time=0.0,
              phase_lost_times=None, terminal_lost_time=None):
    """根据相位初始时长生成 Stage 1 需要的固定绿灯窗。

    方向绑定优先级：
        显式 up_phase / down_phase；
        否则从 Phase.serves 中解析单一服务相位。
    相位间损失 phase_lost_times 加到指定相位绿灯末尾，
    作为下一个相位起点之前的常数间隔。
    """
    phase_lost_times = dict(phase_lost_times or {})
    if terminal_lost_time is not None:
        lost_time = terminal_lost_time

    starts = {}
    acc = 0.0
    for ph in phases:
        starts[ph.name] = acc
        acc += ph.green + float(phase_lost_times.get(ph.name, 0.0))

    def dur(p):
        return next(ph.green for ph in phases if ph.name == p)

    def resolve_direction_names(direction, explicit):
        if explicit is not None:
            return [explicit]
        served = [ph.name for ph in phases if direction in ph.serves]
        if not served:
            raise ValueError(
                f"方案 {name} 无法解析 {direction} 方向相位；"
                f"请提供 {direction}_phase 或设置 Phase.serves"
            )
        return served

    up_names = resolve_direction_names("up", up_phase)
    down_names = resolve_direction_names("down", down_phase)
    up_windows = [
        GreenWindow(starts[name_] / C,
                    (starts[name_] + dur(name_)) / C)
        for name_ in up_names
    ]
    down_windows = [
        GreenWindow(starts[name_] / C,
                    (starts[name_] + dur(name_)) / C)
        for name_ in down_names
    ]
    return SignalPlan(
        name=name,
        up_windows=up_windows,
        down_windows=down_windows,
        phases=phases,
        # 如果调用方显式给了 up_phase/down_phase，则保留；
        # 否则保留 None，让第二阶段通过 Phase.serves 解析方向绑定。
        up_phase=up_phase,
        down_phase=down_phase,
        lost_time=lost_time,
        phase_lost_times=phase_lost_times,
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
    window_weights={2: 0.2, 3: 0.1},
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
            window_weights={2: 0.2, 3: 0.1},
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

# 相位 hinge loss：同时包含“不能太短”和“不能过长”两类。
# 斜率保持不同，便于观察不同相位对 intersection_loss 的贡献差异。
loss_builder_4_1 = PhaseLossBuilder([
    PhaseLossSpec("P1", threshold=35.0, slope=2.0,
                  upper_threshold=45.0, upper_slope=1.2,
                  intersection="I2"),
    PhaseLossSpec("P2", threshold=40.0, slope=1.0,
                  upper_threshold=55.0, upper_slope=2.0,
                  intersection="I4"),
    PhaseLossSpec("P1", threshold=50.0, slope=2.5,
                  upper_threshold=58.0, upper_slope=1.5,
                  intersection="I5"),
    PhaseLossSpec("P2", threshold=45.0, slope=3.0,
                  upper_threshold=56.0, upper_slope=1.0,
                  intersection="I1"),
])

# 额外约束类型：
#   1) 硬等式：I1.P1 + I1.P2 = 75（初始相位满足，但第二阶段必须保持）
#   2) 硬上界：I3.P2 <= 45
#   3) 硬下界：I6.P1 >= 30
#   4) 软下界：I2.P1 + I2.P2 >= 70，缺口罚款 1.5
#   5) 软上界：I4.P1 + I4.P2 <= 60，超出罚款 2.0
#   6) 软下界：I5.P1 >= 50，缺口罚款 1.2
constraint_builder_4_1 = ConstraintBuilder([
    LinearSpec({"I1.P1": 1.0, "I1.P2": 1.0}, sense="=", rhs=75.0),
    LinearSpec({"I3.P2": 1.0}, sense="<=", rhs=45.0),
    LinearSpec({"I6.P1": 1.0}, sense=">=", rhs=30.0),
    LinearSpec({"I2.P1": 1.0, "I2.P2": 1.0}, sense=">=",
               rhs=70.0, soft=True, penalty=1.5),
    LinearSpec({"I4.P1": 1.0, "I4.P2": 1.0}, sense="<=",
               rhs=60.0, soft=True, penalty=2.0),
    LinearSpec({"I5.P1": 1.0}, sense=">=",
               rhs=50.0, soft=True, penalty=1.2),
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
    constraint_builder=constraint_builder_4_1,
    objective="bandwidth",
)

# Case 4.1 只额外输出一张单阶段时空图。
if s4_stage1.status == "optimal":
    show(
        "4.1 stage1 only",
        s4_stage1,
        "case_4_1_stage1_time_space.png",
        notes=[
            "Case 4.1: Stage 1 only",
            "Phases fixed",
            "Only band objective optimized",
        ],
    )

# 双阶段：扫描 intersection_loss 上界，最大化 band_objective。
tuner_4_1 = PhaseTuneSolver(mode="global", down_weight=1.0)
s4_hi = tuner_4_1.solve(
    arterial,
    prior=s_1_1,
    loss_builder=loss_builder_4_1,
    constraint_builder=constraint_builder_4_1,
    objective="bandwidth",
)
s4_lo = tuner_4_1.solve(
    arterial,
    prior=s_1_1,
    loss_builder=loss_builder_4_1,
    constraint_builder=constraint_builder_4_1,
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
            constraint_builder=constraint_builder_4_1,
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
print("    idx   eps      intersection_loss   band_objective")
for idx, (loss, obj, eps, _) in enumerate(front_4_1, start=1):
    print(f"    {idx:>3}   {eps:>5.2f}    {loss:>17.2f}   {obj:>13.2f}")

fig, ax = plt.subplots(figsize=(9, 5.5))
if s4_stage1.status == "optimal":
    ax.scatter(
        [s4_stage1.intersection_loss],
        [s4_stage1.band_objective],
        marker="*", s=260, color="#d95f02",
        label="Stage 1 only (fixed phases)", zorder=4,
    )
    ax.annotate(
        "S1",
        (s4_stage1.intersection_loss, s4_stage1.band_objective),
        textcoords="offset points", xytext=(8, 8),
        fontsize=11, fontweight="bold", color="#d95f02",
    )

if front_4_1:
    xs = [p[0] for p in front_4_1]
    ys = [p[1] for p in front_4_1]
    ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=7,
            color="#1b9e77", label="Two-stage Pareto front", zorder=3)
    for idx, (loss, obj, eps, _) in enumerate(front_4_1, start=1):
        ax.annotate(f"{idx}", (loss, obj),
                    textcoords="offset points", xytext=(7, -12),
                    fontsize=10, fontweight="bold", color="#1b9e77")

ax.set_xlabel("Intersection loss")
ax.set_ylabel("Band objective")
ax.set_title("Case 4.1: Stage 1 only vs Two-stage Pareto Front")
ax.grid(alpha=0.3)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
fig.savefig("case_4_1_intersection_loss_pareto.png", dpi=150)
plt.close(fig)

# 为每一个双阶段 Pareto 点输出一张时空图。
for idx, (loss, obj, eps, sol) in enumerate(front_4_1, start=1):
    png = f"case_4_1_stage2_pareto_ts_{idx:02d}.png"
    plot_time_space(
        arterial,
        sol,
        save_path=png,
        notes=[
            f"Case 4.1 Stage 2 Pareto point {idx}",
            f"eps={eps:.2f}s, intersection_loss={loss:.2f}s, "
            f"band_objective={obj:.2f}",
        ],
    )
    print(f"  双阶段 Pareto 时空图已保存到 {png}")

print()
print("Case 4.1 Pareto 图已保存到 "
      "case_4_1_intersection_loss_pareto.png")
print("Case 4.1 单阶段时空图已保存到 "
      "case_4_1_stage1_time_space.png")


print("=" * 70)
print("5.1 Phase.serves：同一相位同时服务上下行")

# 小型 3 路口测试网，仍使用公共周期 C=90。
phases_5_1 = [
    Phase("P1", 30.0, 10.0, 50.0, serves=("up", "down")),
    Phase("P2", 35.0, 10.0, 55.0, serves=()),
]
plan_5_1 = make_plan(
    "双向直行",
    phases_5_1,
    lost_time=25.0,  # 30 + 35 + 25 = 90
)

art_5_1 = Arterial(
    cycle=C,
    intersections={
        n: Intersection(n, plans=[plan_5_1])
        for n in ["X1", "X2", "X3"]
    },
    segments={
        "xs1": Segment("xs1", 180.0, 180.0, 12.0, 12.0),
        "xs2": Segment("xs2", 180.0, 180.0, 12.0, 12.0),
    },
    order=["X1", "xs1", "X2", "xs2", "X3"],
)

s_5_1_prior = CompositeBandSolver(down_weight=1.0).solve(art_5_1)
s_5_1 = PhaseTuneSolver(mode="global", down_weight=1.0).solve(
    art_5_1,
    prior=s_5_1_prior,
)
print(f"  status={s_5_1.status}, objective={s_5_1.objective:.2f}")
print(f"  phase_times={s_5_1.phase_times}")
plot_time_space(
    art_5_1,
    s_5_1,
    save_path="case_5_1_phase_serves_both.png",
    notes=[
        "Phase.serves = ('up', 'down')",
        "Same phase serves both directions",
    ],
)
print("  时空图已保存到 case_5_1_phase_serves_both.png")


print("=" * 70)
print("5.2 相位末尾损失：lost 指定加到某个相位末尾")

phases_5_2 = [
    Phase("P1", 30.0, 10.0, 50.0, serves=("up",)),
    Phase("P2", 25.0, 10.0, 50.0, serves=("down",)),
    Phase("P3", 10.0, 5.0, 20.0, serves=()),
]
plan_5_2 = make_plan(
    "带末尾损失",
    phases_5_2,
    up_phase="P1",
    down_phase="P2",
    phase_lost_times={"P1": 5.0},  # P1 绿灯后先损失 5s，再进入 P2
    lost_time=20.0,                # 周期尾部损失
)
# 周期检查：30 + 25 + 10 + 5 + 20 = 90
assert abs(phases_5_2[0].green + phases_5_2[1].green
           + phases_5_2[2].green
           + plan_5_2.total_lost_time() - C) < 1e-9

art_5_2 = Arterial(
    cycle=C,
    intersections={
        n: Intersection(n, plans=[plan_5_2])
        for n in ["Y1", "Y2", "Y3"]
    },
    segments={
        "ys1": Segment("ys1", 180.0, 180.0, 12.0, 12.0),
        "ys2": Segment("ys2", 180.0, 180.0, 12.0, 12.0),
    },
    order=["Y1", "ys1", "Y2", "ys2", "Y3"],
)

s_5_2_prior = CompositeBandSolver(down_weight=1.0).solve(art_5_2)
s_5_2 = PhaseTuneSolver(mode="global", down_weight=1.0).solve(
    art_5_2,
    prior=s_5_2_prior,
)
print(f"  status={s_5_2.status}, objective={s_5_2.objective:.2f}")
print(f"  phase_times={s_5_2.phase_times}")
print(f"  phase_lost_times={plan_5_2.phase_lost_times}, "
      f"terminal_lost={plan_5_2.lost_time}")
plot_time_space(
    art_5_2,
    s_5_2,
    save_path="case_5_2_phase_end_lost.png",
    notes=[
        "phase_lost_times={'P1': 5.0}",
        "Loss is inserted after P1 before P2",
    ],
)
print("  时空图已保存到 case_5_2_phase_end_lost.png")


print("=" * 70)
print("5.3 多窗口选择：一个方向由多个相位服务")

phases_5_3 = [
    Phase("P1", 20.0, 5.0, 40.0, serves=("up",)),
    Phase("P2", 30.0, 10.0, 50.0, serves=("up", "down")),
    Phase("P3", 20.0, 5.0, 40.0, serves=("down",)),
]
plan_5_3 = make_plan(
    "多窗口",
    phases_5_3,
    lost_time=20.0,  # 20 + 30 + 20 + 20 = 90
)
# 上行候选窗口：P1、P2
# 下行候选窗口：P2、P3
assert len(plan_5_3.up_windows) == 2
assert len(plan_5_3.down_windows) == 2

art_5_3 = Arterial(
    cycle=C,
    intersections={
        n: Intersection(n, plans=[plan_5_3])
        for n in ["Z1", "Z2", "Z3"]
    },
    segments={
        "zs1": Segment("zs1", 180.0, 180.0, 12.0, 12.0),
        "zs2": Segment("zs2", 180.0, 180.0, 12.0, 12.0),
    },
    order=["Z1", "zs1", "Z2", "zs2", "Z3"],
)

s_5_3_prior = CompositeBandSolver(down_weight=1.0).solve(art_5_3)
s_5_3 = PhaseTuneSolver(mode="global", down_weight=1.0).solve(
    art_5_3,
    prior=s_5_3_prior,
)
print(f"  status={s_5_3.status}, objective={s_5_3.objective:.2f}")
print(f"  phase_times={s_5_3.phase_times}")
print("  第二阶段选中的窗口：")
for name_, choice in s_5_3.window_choices.items():
    print(f"    {name_}: up_phase={choice.get('up_phase')}, "
          f"down_phase={choice.get('down_phase')}, "
          f"up_idx={choice.get('up_window')}, "
          f"down_idx={choice.get('down_window')}")
plot_time_space(
    art_5_3,
    s_5_3,
    save_path="case_5_3_multi_window_choice.png",
    notes=[
        "Multiple serving phases per direction",
        "0-1 window selection in stage 2",
    ],
)
print("  时空图已保存到 case_5_3_multi_window_choice.png")
