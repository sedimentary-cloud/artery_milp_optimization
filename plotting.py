"""时空图绘制。

坐标约定：横轴 = 时间（秒），纵轴 = 沿干线的距离（米，向上递增）。
- 每个路口在其纵坐标处画水平红条（红灯时段，按周期重复）；
- 上行带：左下到右上的斜带；下行带：左上到右下的斜带。
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import MultipleLocator

plt.rcParams["axes.unicode_minus"] = False

from .models import Arterial, GreenWindow
from .solution import Solution

# 全局绿波带 / 局部绿波带 / 窗口短绿波带的图层样式，互不干扰
GLOBAL_BAND_ALPHA = 0.5
GLOBAL_BAND_ZORDER = 3
GLOBAL_UP_COLOR = "#4AC52E"
GLOBAL_DOWN_COLOR = "#3077e2"

WINDOW_BAND_ALPHA = 0.05
WINDOW_BAND_ZORDER = 1.1
# 兼容旧名字：没有方向信息的窗口 key 默认按下行处理。
WINDOW_BAND_COLOR = GLOBAL_DOWN_COLOR
WINDOW_UP_COLOR = GLOBAL_UP_COLOR
WINDOW_DOWN_COLOR = GLOBAL_DOWN_COLOR
# 主绿波带使用斑点纹理，避免只靠颜色区分。
# 上行用密集小点，下行用圆斑，保持上下行可区分。
GLOBAL_UP_HATCH = "..."
GLOBAL_DOWN_HATCH = "ooo"
BAND_HATCH_LINEWIDTH = 0.25


def _band_facecolor(color: str, alpha: float) -> tuple[float, float, float, float]:
    """带子面颜色：保留指定透明度，但让 hatch 边线保持不透明。"""
    r, g, b, _ = to_rgba(color)
    return (r, g, b, alpha)


def _window_facecolor(color: str) -> tuple[float, float, float, float]:
    """窗口带面颜色：低透明度纯填充，不使用 hatch。"""
    return _band_facecolor(color, WINDOW_BAND_ALPHA)


def _positions(arterial: Arterial) -> list[float]:
    """各路口的累计里程（米）。

    约定：距离轴统一采用上行方向里程。上/下行路段不等距时，
    差异体现在下行带的斜率上（下行带跨越同样的空间距离，
    但使用下行行驶时间），而不是另建一套下行里程坐标。
    """
    pos = [0.0]
    for seg in arterial.segment_order:
        pos.append(pos[-1] + seg.length_up)
    return pos


def _time_max(arterial: Arterial, solution: Solution | None = None) -> float:
    """时间轴右边界（秒）：保证一条完整绿波带能呈现在图上。

    时间轴从 0 开始，右边界取：
        max(上行全程时间, 下行全程时间) + 最大带宽 + 一个周期 C
    额外加一个周期，是为了让“起点接近周期末端”的带子也能完整落入
    [0, t_max] 区间内。绘制灯条和带子时会从负周期开始画，
    负时间部分由 xlim=(0, t_max) 自动截断。
    """
    t_up = sum(s.travel_time_up for s in arterial.segment_order)
    t_dn = sum(s.travel_time_down for s in arterial.segment_order)

    band_max = 0.0
    if solution is not None:
        band_values = (
            list(solution.bandwidth_up.values())
            + list(solution.bandwidth_down.values())
            + list(solution.window_bands.values())
        )
        if band_values:
            band_max = max(band_values)

    return max(t_up, t_dn) + band_max + arterial.cycle


def plot_time_space(arterial: Arterial,
                    solution: Solution | None = None,
                    n_cycles: int | None = None,
                    max_band_window: int = 5,
                    save_path: str | None = None,
                    ax: plt.Axes | None = None,
                    notes: list[str] | None = None) -> plt.Axes:
    """绘制干线时空图。

    args:
        arterial: 干线对象；
        solution: 解（None 则只画红条和路口）；
        n_cycles: 最少展示的周期数；None 表示按车速、总距离和带宽自动确定右边界，
            保证至少一条绿波带完整呈现；
        max_band_window: 绘制短绿波带的最大窗口（相邻路口组的大小），
            即两两路口、三个一组……直到该值；默认 5；
        save_path: 若给定则保存图片；
        ax: 复用已有坐标轴；
        notes: 可选的图内说明文字，用于标记损失/约束信息；每项一行。
    """
    close_after_save = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    C = arterial.cycle
    ints = arterial.intersection_order
    pos = _positions(arterial)

    # 最终显示区间从 0 开始；右边界动态计算。
    # 绘制周期仍从负值开始（k_min = -1），负时间部分会被 xlim 截断。
    t_min = 0.0
    t_max = _time_max(arterial, solution)
    if n_cycles is not None:
        t_max = max(t_max, n_cycles * C)
    k_min = math.floor(t_min / C) - 1
    k_max = math.ceil(t_max / C) + 1

    # --- 灯条（上/下行分别绘制：路口位置上半条为上行，下半条为下行；
    #     绿色 = 绿灯窗口，红/橙色 = 红灯段，按周期重复） ---
    h = _bar_height(arterial)
    # 时间轴最终显示为 [0, 2*t_half]；把路口标签放在轴内左侧，
    # 并加白色背景，避免和红绿灯条、绿波带混在一起看不清。
    label_x = t_min + 0.01 * (t_max - t_min)
    for i, inter in enumerate(ints):
        # 优先使用求解器选中的方案；没有解或没有方案选择信息时退回第一个方案。
        plan = inter.plans[0]
        if solution is not None and inter.name in solution.plan_choices:
            try:
                plan = inter.plan_by_name(solution.plan_choices[inter.name])
            except KeyError:
                plan = inter.plans[0]
        win_up = max(plan.up_windows, key=lambda w: w.width)
        win_dn = max(plan.down_windows, key=lambda w: w.width)

        # 如果求解器返回了选中的窗口，优先使用选中的窗口而不是最宽窗口
        if solution is not None and inter.name in solution.window_choices:
            wc = solution.window_choices[inter.name]
            try:
                up_idx = int(wc.get("up_window", -1))
                dn_idx = int(wc.get("down_window", -1))
                if 0 <= up_idx < len(plan.up_windows):
                    win_up = plan.up_windows[up_idx]
                if 0 <= dn_idx < len(plan.down_windows):
                    win_dn = plan.down_windows[dn_idx]
            except (TypeError, ValueError, KeyError):
                pass

        # 第二阶段相位优化后，用 phase_times 反算选中方案的绿灯窗
        if (solution is not None
                and inter.name in solution.phase_times
                and plan.phases):
            pt = solution.phase_times[inter.name]
            starts: dict[str, float] = {}
            acc = 0.0
            for ph in plan.phases:
                starts[ph.name] = acc
                acc += float(pt.get(ph.name, ph.green))
            if plan.up_phase in starts and plan.up_phase in pt:
                us = starts[plan.up_phase]
                win_up = GreenWindow(us / C, (us + float(pt[plan.up_phase])) / C)
            if plan.down_phase in starts and plan.down_phase in pt:
                ds = starts[plan.down_phase]
                win_dn = GreenWindow(ds / C, (ds + float(pt[plan.down_phase])) / C)
        # 需要绘制的绿灯窗口集合。
        # 有 phase_times 时，使用反算出的单窗口；否则绘制方案里的全部绿灯窗口。
        if (solution is not None
                and inter.name in solution.phase_times
                and plan.phases):
            up_windows_to_draw = [win_up]
            down_windows_to_draw = [win_dn]
        else:
            up_windows_to_draw = list(plan.up_windows)
            down_windows_to_draw = list(plan.down_windows)

        for windows, yc, red_color in ((up_windows_to_draw, pos[i] + h / 2, "red"),
                                       (down_windows_to_draw, pos[i] - h / 2, "darkorange")):
            for k in range(k_min, k_max + 1):
                # 先铺满红灯
                ax.add_patch(Rectangle((k * C, yc - h / 2), C, h,
                                       facecolor=red_color, alpha=1.0,
                                       edgecolor="none", zorder=5))
                # 再叠加所有绿灯窗口
                for win in windows:
                    ax.add_patch(Rectangle((win.start * C + k * C, yc - h / 2),
                                           (win.end - win.start) * C, h,
                                           facecolor="limegreen", alpha=1.0,
                                           edgecolor="none", zorder=5))
        # 每个路口中心画一条水平细黑线（在灯条上方，路口编号下方）
        ax.axhline(pos[i], color="black", linewidth=0.8, zorder=6)

        # 在灯条附近标注“路口.相位”，文字小、半透明白底。
        phase_labels = []
        if plan.up_phase:
            phase_labels.append((f"{inter.name}.{plan.up_phase}",
                                 pos[i] + h / 2))
        if plan.down_phase:
            phase_labels.append((f"{inter.name}.{plan.down_phase}",
                                 pos[i] - h / 2))
        for phase_text, phase_y in phase_labels:
            ax.text(label_x, phase_y, phase_text,
                    ha="left", va="center", fontsize=6.5,
                    color="black", zorder=21, clip_on=True,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.65, boxstyle="round,pad=0.08"))

        ax.text(label_x, pos[i], inter.name,
                ha="left", va="center", fontsize=11, fontweight="bold",
                zorder=20, clip_on=True,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          boxstyle="round,pad=0.2"))

    # 求解器通过 Solution.band_up_style / band_down_style 告诉绘图：
    # global 画成全局绿波带颜色；local 画成浅色局部带。
    up_style = "global"
    down_style = "global"
    has_window_bands = False
    if solution is not None:
        up_style = getattr(solution, "band_up_style", "global")
        down_style = getattr(solution, "band_down_style", "global")
        has_window_bands = bool(solution.window_bands)

    # 图例
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="limegreen", alpha=1.0, label="Green Window"),
        Patch(facecolor="red", alpha=1.0, label="Red (Up)"),
        Patch(facecolor="darkorange", alpha=1.0, label="Red (Down)"),
    ]
    if up_style == "global":
        handles.append(Patch(
            facecolor=_band_facecolor(GLOBAL_UP_COLOR, GLOBAL_BAND_ALPHA),
            edgecolor=GLOBAL_UP_COLOR,
            hatch=GLOBAL_UP_HATCH,
            linewidth=BAND_HATCH_LINEWIDTH,
            label="Up Band",
        ))
    if down_style == "global":
        handles.append(Patch(
            facecolor=_band_facecolor(GLOBAL_DOWN_COLOR, GLOBAL_BAND_ALPHA),
            edgecolor=GLOBAL_DOWN_COLOR,
            hatch=GLOBAL_DOWN_HATCH,
            linewidth=BAND_HATCH_LINEWIDTH,
            label="Down Band",
        ))
    if has_window_bands:
        handles.append(Patch(
            facecolor=_window_facecolor(WINDOW_UP_COLOR),
            edgecolor="none",
            label="Window Bands (Up)",
        ))
        handles.append(Patch(
            facecolor=_window_facecolor(WINDOW_DOWN_COLOR),
            edgecolor="none",
            label="Window Bands (Down)",
        ))
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    # --- 绿波带 ---
    if solution is not None and solution.band_start_up:
        names = [v.name for v in ints]
        segs = arterial.segment_order

        # 带前沿的绝对时刻：沿行驶方向按行驶时间递推（不再 mod 周期）
        tU = [solution.band_start_up[names[0]]]
        for i, seg in enumerate(segs):
            tU.append(tU[-1] + seg.travel_time_up)
        tD = [0.0] * len(names)
        tD[-1] = solution.band_start_down[names[-1]]
        for i in range(len(segs) - 1, -1, -1):
            tD[i] = tD[i + 1] + segs[i].travel_time_down

        # 绿波带从更负的周期开始绘制，让画面左侧也有带子覆盖。
        # 注意：这里只影响绿波带，xlim 和灯条仍使用原来的 k_min/k_max。
        band_values = (
            list(solution.bandwidth_up.values())
            + list(solution.bandwidth_down.values())
            + list(solution.window_bands.values())
        )
        max_band_width = max(band_values, default=0.0)
        max_travel_time = max(
            sum(seg.travel_time_up for seg in segs),
            sum(seg.travel_time_down for seg in segs),
        )
        band_k_min = k_min - math.ceil((max_travel_time + max_band_width) / C)

        # ============================================================
        # 图层 1：全局绿波带
        # ============================================================
        for k in range(band_k_min, k_max + 1):
            shift = k * C
            for i, seg in enumerate(segs):
                bu = solution.bandwidth_up.get(seg.name, 0.0)
                bd = solution.bandwidth_down.get(seg.name, 0.0)
                if up_style == "global" and bu > 0:
                    ax.add_patch(_quad(
                        pos[i], pos[i + 1],
                        tU[i] + shift, tU[i + 1] + shift, bu,
                        facecolor=_band_facecolor(GLOBAL_UP_COLOR, GLOBAL_BAND_ALPHA),
                        edgecolor=GLOBAL_UP_COLOR,
                        linewidth=BAND_HATCH_LINEWIDTH,
                        hatch=GLOBAL_UP_HATCH,
                        zorder=GLOBAL_BAND_ZORDER,
                    ))
                if down_style == "global" and bd > 0:
                    ax.add_patch(_quad(
                        pos[i], pos[i + 1],
                        tD[i] + shift, tD[i + 1] + shift, bd,
                        facecolor=_band_facecolor(GLOBAL_DOWN_COLOR, GLOBAL_BAND_ALPHA),
                        edgecolor=GLOBAL_DOWN_COLOR,
                        linewidth=BAND_HATCH_LINEWIDTH,
                        hatch=GLOBAL_DOWN_HATCH,
                        zorder=GLOBAL_BAND_ZORDER,
                    ))

        # ============================================================
        # 图层 2：窗口绿波带（win2 两两路口、win3 三个一组……）
        # ============================================================
        if has_window_bands:
            # 按方向 + 窗口大小分组；up 用 tU，down 用 tD。
            grouped: dict[str, dict[int, list[tuple[int, float]]]] = {
                "up": {},
                "down": {},
            }
            for key, bw in solution.window_bands.items():
                direction, k, j = _parse_window_key(key, names)
                if (direction in grouped and k is not None
                        and k <= max_band_window and bw > 0):
                    grouped[direction].setdefault(k, []).append((j, bw))

            for direction in ("up", "down"):
                t_series = tU if direction == "up" else tD
                color = WINDOW_UP_COLOR if direction == "up" else WINDOW_DOWN_COLOR
                for k in sorted(grouped[direction], reverse=True):
                    for j, bw in grouped[direction][k]:
                        ts = [t_series[j + i] for i in range(k)]
                        ps = [pos[j + i] for i in range(k)]
                        for kk in range(band_k_min, k_max + 1):
                            t_shifted = [t + kk * C for t in ts]
                            ax.add_patch(_poly_band(
                                ps, t_shifted, bw,
                                facecolor=_window_facecolor(color),
                                edgecolor="none",
                                zorder=WINDOW_BAND_ZORDER,
                            ))

    ax.set_xlim(t_min, t_max)
    ax.set_ylim(-0.05 * pos[-1], 1.05 * pos[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title(f"Time-Space Diagram (Cycle {C:.0f}s)")
    ax.grid(alpha=0.3, zorder=0)

    # x 轴主刻度按周期 C 对齐，副刻度按 C/4 细分，保证主副网格线对齐
    ax.xaxis.set_major_locator(MultipleLocator(C))
    ax.xaxis.set_minor_locator(MultipleLocator(C / 4.0))
    ax.grid(which="minor", axis="x", color="lightgray",
            linewidth=0.5, alpha=0.6, zorder=0)

    if notes:
        note_text = "\n".join(notes)
        ax.text(0.02, 0.98, note_text,
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=8,
                zorder=25,
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="white", edgecolor="black", alpha=0.85))

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
        if close_after_save:
            plt.close(ax.figure)
    return ax


def _bar_height(arterial: Arterial) -> float:
    total = sum(s.length_up for s in arterial.segment_order)
    return max(total * 0.012, 1.0)


def _quad(x0: float, x1: float, t0: float, t1: float, b: float,
          **kw) -> Polygon:
    """单个路段的带子四边形（x=时间, y=距离；带宽 b 为时间方向宽度）。"""
    zorder = kw.pop("zorder", 3)
    verts = [(t0, x0), (t1, x1), (t1 + b, x1), (t0 + b, x0)]
    kw.setdefault("edgecolor", "none")
    return Polygon(verts, closed=True, zorder=zorder, **kw)


def _poly_band(pos: list[float], t: list[float], b: float, **kw) -> Polygon:
    """跨多个路段的带子多边形（x=时间, y=距离），前沿经过各中间路口。"""
    verts = (list(zip(t, pos))
             + list(zip([v + b for v in t[::-1]], pos[::-1])))
    kw.setdefault("edgecolor", "none")
    return Polygon(verts, closed=True, **kw)


def _parse_window_key(key: str, names: list[str]) -> tuple[str | None, int | None, int | None]:
    """解析窗口带 key。

    新格式：
        "up.win3@I1-I3"   -> ("up", 3, 0)
        "down.win2@I1-I2" -> ("down", 2, 0)

    兼容旧格式：
        "win3@I1-I3"      -> ("down", 3, 0)

    解析失败返回 (None, None, None)。
    """
    try:
        prefix, rng = key.split("@")
        # 新格式带方向前缀；旧格式没有方向，默认按下行处理。
        if "." in prefix:
            direction, k_str = prefix.split(".", 1)
        else:
            direction, k_str = "down", prefix
        if direction not in ("up", "down"):
            return None, None, None
        k = int(k_str.replace("win", ""))
        first = rng.split("-")[0]
        return direction, k, names.index(first)
    except (ValueError, IndexError):
        return None, None, None
