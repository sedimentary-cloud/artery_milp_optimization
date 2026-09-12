"""6 路口一般化案例：多相位 / 相位不对称 / 单路口 hinge 损失 / 路段不对称 / 不等速。

用于比较不同优化目标与约束的结果差异。每个场景都会保存时空图，
帕累托前沿单独保存一张图。
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


print("=" * 70)
print("1) 求和目标（CompositeBandSolver）")
s_sum = CompositeBandSolver(down_weight=1.0).solve(arterial)
show("sum", s_sum, "case_01_sum.png",
     notes=["Objective: max b_up + b_down",
            "No hinge loss / no extra constraint"])

print("=" * 70)
print("2) 均衡目标（CompositeBandSolver, balanced_composite）")
s_bal = CompositeBandSolver(objective_mode="balanced_composite",
                           balance_eps=0.1).solve(arterial)
show("balanced", s_bal, "case_02_balanced.png",
     notes=["Objective: max b_up + b_down + eps*B_bal",
            "B_bal = min(b_up, b_down)"])

print("=" * 70)
print("3) 上行优先 + 下行窗口加权（OneWayPrioritySolver）")
s_one = OneWayPrioritySolver(up_weight=1.0,
                             window_weights={2: 1.0, 3: 0.5},
                             n_intersections=len(names)).solve(arterial)
show("one-way", s_one, "case_03_oneway.png",
     notes=["Objective: up-priority + down window bands",
            "window_weights: win2=1.0, win3=0.5"])

print("=" * 70)
print("4) 两阶段相位优化（TwoStageSolver）")
s_two = TwoStageSolver(CompositeBandSolver(down_weight=1.0),
                       mode="global", down_weight=1.0).solve(arterial)
show("two-stage", s_two, "case_04_two_stage.png",
     notes=["Stage 2: phase durations optimized",
            "No hinge loss / no extra constraint"])

print("=" * 70)
print("5) 相位 hinge 损失（I2.P1<35, I4.P2<40, I5.P1<50, I1.P2<45）")
loss_builder = PhaseLossBuilder([
    PhaseLossSpec("P1", threshold=35.0, slope=2.0, intersection="I2"),
    PhaseLossSpec("P2", threshold=40.0, slope=2.0, intersection="I4"),
    PhaseLossSpec("P1", threshold=50.0, slope=2.0, intersection="I5"),
    PhaseLossSpec("P2", threshold=45.0, slope=2.0, intersection="I1"),
])
s_loss = PhaseTuneSolver(mode="global", down_weight=1.0).solve(
    arterial, prior=s_sum, loss_builder=loss_builder, objective="loss")
show("loss", s_loss, "case_05_loss.png",
     notes=["Min total hinge loss",
            "I2.P1<35, I4.P2<40, I5.P1<50, I1.P2<45"])

print("=" * 70)
print("6) 硬约束：I2.P1 + I2.P2 >= 60")
hard = ConstraintBuilder([
    LinearSpec({"I2.P1": 1.0, "I2.P2": 1.0}, sense=">=", rhs=60.0)
])
s_hard = PhaseTuneSolver(mode="global", down_weight=1.0).solve(
    arterial, prior=s_sum, constraint_builder=hard)
show("hard", s_hard, "case_06_hard.png",
     notes=["Hard constraint: I2.P1 + I2.P2 >= 60"])

print("=" * 70)
print("7) 软约束：I5.P1 + I5.P2 >= 90，缺口罚款 3.0")
soft = ConstraintBuilder([
    LinearSpec({"I5.P1": 1.0, "I5.P2": 1.0}, sense=">=", rhs=90.0,
               soft=True, penalty=3.0)
])
s_soft = PhaseTuneSolver(mode="global", down_weight=1.0).solve(
    arterial, prior=s_sum, constraint_builder=soft)
show("soft", s_soft, "case_07_soft.png",
     notes=["Soft constraint: I5.P1 + I5.P2 >= 90",
            "penalty = 3.0 / s"])

print("=" * 70)
print("8) 帕累托前沿（hinge loss + 最大带宽目标）")
runner = EpsilonConstraintRunner(mode="global",
                                 loss_builder=loss_builder,
                                 n_points=10,
                                 down_weight=1.0,
                                 objective_mode="sum",
                                 metric="sum")
frontier = runner.run(arterial, prior=s_sum)

print("  idx  eps      sum_bandwidth   intersection_loss   time-space png")
for idx, (eps, b, inter_loss, sol) in enumerate(frontier, start=1):
    png = f"case_08_pareto_ts_{idx:02d}.png"
    print(f"  {idx:>3}  {eps:6.2f}   {b:6.2f}    {inter_loss:6.2f}   {png}")
    notes = [f"Pareto point {idx}: eps={eps:.2f}s, "
             f"intersection_loss={inter_loss:.2f}s",
             "Objective = b_up + b_down",
             "Hinge losses: I2.P1<35, I4.P2<40, I5.P1<50, I1.P2<45"]
    plot_time_space(arterial, sol, save_path=png, notes=notes)

if frontier:
    bandwidths = [b for _, b, _, _ in frontier]
    losses = [loss for _, _, loss, _ in frontier]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(bandwidths, losses, marker="o", linewidth=2,
            label="Pareto frontier", zorder=3)
    ax.scatter([bandwidths[0]], [losses[0]], marker="s", s=80,
               color="green", label="Min loss", zorder=4)
    ax.scatter([bandwidths[-1]], [losses[-1]], marker="s", s=80,
               color="red", label="Max bandwidth", zorder=4)
    # 在帕累托前沿上标记序号，序号对应上面保存的时空图
    for idx, (eps, b, loss, _) in enumerate(frontier, start=1):
        ax.annotate(str(idx), (b, loss),
                    textcoords="offset points", xytext=(7, 7),
                    fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", edgecolor="gray", alpha=0.8),
                    zorder=6)
    knee = runner.knee_point()
    if knee is not None:
        ax.scatter([knee[0]], [knee[1]], marker="*", s=220,
                   color="black", label="Knee point", zorder=5)
    ax.set_xlabel("Sum bandwidth b_up + b_down (s)")
    ax.set_ylabel("Intersection loss (s)")
    ax.set_title("Pareto Frontier: Sum Bandwidth vs Intersection Loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig("case_08_pareto_frontier.png", dpi=150)
    print()
    print("帕累托前沿图已保存到 case_08_pareto_frontier.png")


print("=" * 70)
print("9) one-way 帕累托前沿：window_weights={2:0.1, 3:0.05}")
s_one_low = OneWayPrioritySolver(
    up_weight=1.0,
    window_weights={2: 0.1, 3: 0.05},
    n_intersections=len(names),
).solve(arterial)
show("one-way-low-window-weight", s_one_low, "case_09_oneway_stage1.png",
     notes=["OneWayPrioritySolver",
            "window_weights: win2=0.10, win3=0.05"])

runner_one = EpsilonConstraintRunner(
    mode="oneway",
    loss_builder=loss_builder,
    n_points=10,
    up_weight=1.0,
    window_weights={2: 0.1, 3: 0.05},
    metric="objective",
)
frontier_one = runner_one.run(arterial, prior=s_one_low)

print("  idx  eps      objective   intersection_loss   time-space png")
for idx, (eps, obj, inter_loss, sol) in enumerate(frontier_one, start=1):
    png = f"case_09_oneway_pareto_ts_{idx:02d}.png"
    print(f"  {idx:>3}  {eps:6.2f}   {obj:6.2f}    {inter_loss:6.2f}   {png}")
    notes = [
        f"One-way Pareto point {idx}: eps={eps:.2f}s, "
        f"intersection_loss={inter_loss:.2f}s",
        "window_weights: win2=0.10, win3=0.05",
        "Hinge losses: I2.P1<35, I4.P2<40, I5.P1<50, I1.P2<45",
    ]
    plot_time_space(arterial, sol, save_path=png, notes=notes)

if frontier_one:
    objectives = [obj for _, obj, _, _ in frontier_one]
    losses = [loss for _, _, loss, _ in frontier_one]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(objectives, losses, marker="o", linewidth=2,
            label="One-way Pareto frontier", zorder=3)
    ax.scatter([objectives[0]], [losses[0]], marker="s", s=80,
               color="green", label="Min loss", zorder=4)
    ax.scatter([objectives[-1]], [losses[-1]], marker="s", s=80,
               color="red", label="Max objective", zorder=4)
    for idx, (eps, obj, loss, _) in enumerate(frontier_one, start=1):
        ax.annotate(str(idx), (obj, loss),
                    textcoords="offset points", xytext=(7, 7),
                    fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", edgecolor="gray", alpha=0.8),
                    zorder=6)
    knee = runner_one.knee_point()
    if knee is not None:
        ax.scatter([knee[0]], [knee[1]], marker="*", s=220,
                   color="black", label="Knee point", zorder=5)
    ax.set_xlabel("One-way objective (s)")
    ax.set_ylabel("Intersection loss (s)")
    ax.set_title("One-way Pareto Frontier: Objective vs Intersection Loss "
                 "(win2=0.10, win3=0.05)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig("case_09_oneway_pareto_frontier.png", dpi=150)
    print()
    print("one-way 帕累托前沿图已保存到 case_09_oneway_pareto_frontier.png")


print("=" * 70)
print("10) 带边/带中心对齐损失（纯线性 hinge）")
# 目标中心：由上游累计行驶时间推出，再折算到 [0, C)
up_center_target = {}
acc = 0.0
up_center_target[names[0]] = 0.0
for idx, seg in enumerate(arterial.segment_order):
    acc += seg.travel_time_up
    up_center_target[names[idx + 1]] = acc % C

tolerance = 5.0
alignment_builder = AlignmentLossBuilder(
    targets_up=up_center_target,
    tolerance=tolerance,
    weight_up=1.0,
)

# 无对齐损失的基准解
s_no_align = PhaseTuneSolver(mode="global", down_weight=1.0).solve(
    arterial, prior=s_sum
)
print("  无对齐损失 vs 有对齐损失 对比：")
print(f"  {'路口':<4}{'目标中心':>10}{'无对齐中心':>12}{'对齐中心':>12}")
for n in names:
    a = up_center_target[n]
    c_no = s_no_align.band_start_up[n] + 0.5 * s_no_align.bandwidth_up["seg1"]
    c_no = c_no % C
    print(f"  {n:<4}{a:>10.2f}{c_no:>12.2f}")

plot_time_space(
    arterial, s_no_align,
    save_path="case_10_no_alignment.png",
    notes=["No alignment loss",
           "Band may sit at the edge of green windows"],
)

s_align = PhaseTuneSolver(mode="global", down_weight=1.0).solve(
    arterial,
    prior=s_sum,
    alignment_builder=alignment_builder,
    band_loss_weight=10.0,  # 绿波带层对齐损失以加权和进入 band_score
    objective="bandwidth",
)

print("  有对齐损失优化后各路口带中心：")
print(f"  {'路口':<4}{'目标中心':>10}{'对齐中心':>12}")
for n in names:
    a = up_center_target[n]
    c_al = s_align.band_start_up[n] + 0.5 * s_align.bandwidth_up["seg1"]
    c_al = c_al % C
    print(f"  {n:<4}{a:>10.2f}{c_al:>12.2f}")

plot_time_space(
    arterial, s_align,
    save_path="case_10_alignment_loss.png",
    notes=["Alignment loss in band_score (weight=10)",
           "Band center is pulled toward arrival peak"],
)
print()
print("有对齐损失时空图已保存到 case_10_alignment_loss.png")


print("=" * 70)
print("11) 仅一阶段 vs 两阶段 Pareto：band_score vs intersection_loss")
# 本案例只输出 Pareto 图，不输出时空图。
#
# 两个目标：
#   第一目标（绿波带层）：band_score，越大越好
#   第二目标（交叉口层）：intersection_loss，越小越好
#
# 仅在 g 可调时，第二阶段才能改变 intersection_loss。
# 因此“仅一阶段”在固定相位下只能得到单个可行点；
# “两阶段”则通过调 g 得到 intersection_loss 与 band_score 的 Pareto 前沿。

case11_band_loss_weight = 1.0
case11_n_points = 10

# 仅一阶段：相位固定，只优化 t/b/B/alignment。
s1_fixed = PhaseTuneSolver(
    mode="global",
    down_weight=1.0,
    tunable_intersections=set(),  # 所有路口相位固定
).solve(
    arterial,
    prior=s_sum,
    loss_builder=loss_builder,
    alignment_builder=alignment_builder,
    band_loss_weight=case11_band_loss_weight,
    objective="bandwidth",
)

if s1_fixed.status == "optimal":
    stage1_point = (float(s1_fixed.intersection_loss),
                    float(s1_fixed.band_score),
                    "Stage 1 only",
                    s1_fixed)
else:
    stage1_point = None

# 两阶段：扫描 intersection_loss 上界，最大化 band_score。
tuner_case11 = PhaseTuneSolver(mode="global", down_weight=1.0)

s_hi = tuner_case11.solve(
    arterial,
    prior=s_sum,
    loss_builder=loss_builder,
    alignment_builder=alignment_builder,
    band_loss_weight=case11_band_loss_weight,
    objective="bandwidth",
)
s_lo = tuner_case11.solve(
    arterial,
    prior=s_sum,
    loss_builder=loss_builder,
    alignment_builder=alignment_builder,
    band_loss_weight=case11_band_loss_weight,
    objective="loss",
)

stage2_points = []
if s_hi.status == "optimal" and s_lo.status == "optimal":
    L_hi = float(s_hi.intersection_loss)
    L_lo = float(s_lo.intersection_loss)
    if case11_n_points <= 1 or abs(L_hi - L_lo) < 1e-9:
        eps_values = [L_lo]
    else:
        step = (L_hi - L_lo) / (case11_n_points - 1)
        eps_values = [L_lo + i * step for i in range(case11_n_points)]

    for eps in eps_values:
        s = tuner_case11.solve(
            arterial,
            prior=s_sum,
            loss_builder=loss_builder,
            alignment_builder=alignment_builder,
            band_loss_weight=case11_band_loss_weight,
            max_intersection_loss=eps,
            objective="bandwidth",
        )
        if s.status == "optimal":
            stage2_points.append((float(s.intersection_loss),
                                  float(s.band_score),
                                  float(eps),
                                  s))


def _nondominated_case11(points):
    """非支配：intersection_loss 越小、band_score 越大越好。"""
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

front_stage2 = _nondominated_case11(stage2_points)

print("  仅一阶段固定相位点：")
if stage1_point is None:
    print("    infeasible")
else:
    print("    intersection_loss   band_score")
    print(f"    {stage1_point[0]:>17.2f}   {stage1_point[1]:>10.2f}")

print("  两阶段 Pareto 点：")
print("    eps      intersection_loss   band_score")
for loss, score, eps, _ in front_stage2:
    print(f"    {eps:>5.2f}    {loss:>17.2f}   {score:>10.2f}")

fig, ax = plt.subplots(figsize=(9, 5.5))
if stage1_point is not None:
    ax.scatter([stage1_point[0]], [stage1_point[1]],
               marker="*", s=260, color="#d95f02",
               label="Stage 1 only (fixed phases)", zorder=4)
    ax.annotate("Stage 1 only", (stage1_point[0], stage1_point[1]),
                textcoords="offset points", xytext=(8, 8),
                fontsize=9, color="#d95f02")

if front_stage2:
    xs = [p[0] for p in front_stage2]
    ys = [p[1] for p in front_stage2]
    ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=7,
            color="#1b9e77", label="Two-stage Pareto front", zorder=3)
    for loss, score, eps, _ in front_stage2:
        ax.annotate(f"eps={eps:.1f}", (loss, score),
                    textcoords="offset points", xytext=(6, -12),
                    fontsize=8, color="#1b9e77")

ax.set_xlabel("Intersection loss")
ax.set_ylabel("Band score")
ax.set_title("Case 11: Stage 1 only vs Two-stage Pareto Front")
ax.grid(alpha=0.3)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
fig.savefig("case_11_stage1_vs_stage2_pareto.png", dpi=150)
plt.close(fig)

print()
print("仅一阶段 vs 两阶段 Pareto 图已保存到 "
      "case_11_stage1_vs_stage2_pareto.png")
