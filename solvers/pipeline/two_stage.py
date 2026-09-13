"""两阶段求解与 ε-约束帕累托扫描。

当前主语义已经统一到“段级输入”：

- Stage 1：方案/窗口选择 + 带宽目标；
- Stage 2：锁定方案后，优化段级端点、业务约束与带宽；
- intersection_loss：段级软损失 + 软 LinearSpec slack；
- band_loss：kind="band" 的通用带层软损失。

推荐入口：

    config = TwoStageConfig(...)
    solution = TwoStageSolver(config).solve(arterial)
"""

from __future__ import annotations

import numpy as np

from ...solution import Solution
from ..core.base import Solver
from ..stage1.segmented_band import SegmentedBandSolver
from ..stage2.phase_tune import FullFlexiblePhaseTuneSolver
from .config import TwoStageConfig


class TwoStageSolver(Solver):
    """两阶段编排器：Stage 1 选方案，Stage 2 优化段级端点。"""

    name = "two-stage"

    def __init__(self, config: TwoStageConfig) -> None:
        """函数名：__init__；参数：TwoStageConfig；返回值：无；异常：无。"""
        self.config = config

    def solve(self, arterial) -> Solution:
        """函数名：solve；参数：arterial；返回值：Solution；异常：ValueError。"""
        cfg = self.config
        cfg.validate()
        mode = cfg.band.mode

        stage1 = SegmentedBandSolver(
            config=cfg.band.objective,
            max_loops=cfg.max_loops,
            up_global_output=True,
            down_global_output=(mode == "global"),
            margin=cfg.margin,
        )
        s1 = stage1.solve(
            arterial,
            band_loss_weight=cfg.band.band_loss_weight,
        )

        try:
            stage2 = FullFlexiblePhaseTuneSolver(
                config=cfg.band.objective,
                max_loops=cfg.max_loops,
                up_global_output=True,
                down_global_output=(mode == "global"),
                margin=cfg.margin,
            )
            s2 = stage2.solve(
                arterial,
                prior=s1,
                loss_builder=cfg.intersection.loss_builder,
                constraint_builder=cfg.intersection.constraint_builder,
                band_loss_weight=cfg.band.band_loss_weight,
                objective="bandwidth",
                tunable_intersections=cfg.intersection.tunable_intersections,
            )
            if s2.status == "optimal":
                return s2
        except Exception:
            pass

        s1.status = f"{s1.status}|stage1_fallback"
        return s1


class EpsilonConstraintRunner:
    """ε-约束法扫描段级带宽-损失帕累托前沿。"""

    def __init__(
        self,
        config: TwoStageConfig,
        n_points: int = 5,
        metric: str | None = None,
    ) -> None:
        """函数名：__init__；参数：TwoStageConfig、点数、metric；返回值：无；异常：无。"""
        config.validate()
        self.config = config
        self.mode = config.band.mode
        self.loss_builder = config.intersection.loss_builder
        self.constraint_builder = config.intersection.constraint_builder
        self.band_loss_weight = config.band.band_loss_weight
        self.max_loops = config.max_loops
        self.tunable_intersections = config.intersection.tunable_intersections
        self.n_points = n_points
        self.metric = metric or "band_score"
        self.frontier: list[tuple[float, float, float, Solution]] = []

    def _make_tuner(self) -> FullFlexiblePhaseTuneSolver:
        """函数名：_make_tuner；参数：无；返回值：FullFlexiblePhaseTuneSolver；异常：无。"""
        return FullFlexiblePhaseTuneSolver(
            config=self.config.band.objective,
            max_loops=self.max_loops,
            up_global_output=True,
            down_global_output=(self.mode == "global"),
            margin=self.config.margin,
        )

    def _bandwidth(self, sol: Solution) -> float:
        """函数名：_bandwidth；参数：sol；返回值：x 轴带宽口径；异常：无。"""
        if self.metric == "band_score":
            return float(sol.band_score)
        if self.mode == "global":
            bu = next(iter(sol.bandwidth_up.values()), 0.0)
            bd = next(iter(sol.bandwidth_down.values()), 0.0)
            if self.metric == "balanced":
                return float(min(bu, bd))
            if self.metric == "objective":
                return float(sol.band_score)
            return float(bu + bd)
        if self.metric == "objective":
            return float(sol.band_score)
        return float(sol.objective)

    def _solve_tuner(self, arterial, prior, **kwargs) -> Solution:
        """函数名：_solve_tuner；参数：arterial、prior、求解参数；返回值：Solution；异常：无。"""
        return self._make_tuner().solve(
            arterial,
            prior=prior,
            loss_builder=self.loss_builder,
            constraint_builder=self.constraint_builder,
            band_loss_weight=self.band_loss_weight,
            tunable_intersections=self.tunable_intersections,
            **kwargs,
        )

    def run(self, arterial, prior: Solution) -> list[tuple[float, float, float, Solution]]:
        """函数名：run；参数：arterial、prior；返回值：前沿点列表；异常：无。"""
        s_hi = self._solve_tuner(arterial, prior, objective="bandwidth")
        if s_hi.status != "optimal":
            return []
        l_hi = s_hi.intersection_loss

        s_lo = self._solve_tuner(arterial, prior, objective="loss")
        if s_lo.status != "optimal":
            return []
        l_lo = s_lo.intersection_loss

        eps_values = list(np.linspace(l_lo, l_hi, max(self.n_points, 2)))
        frontier: list[tuple[float, float, float, Solution]] = []
        for eps in eps_values:
            sol = self._solve_tuner(
                arterial,
                prior,
                max_intersection_loss=float(eps),
                objective="bandwidth",
            )
            if sol.status == "optimal":
                frontier.append((float(eps), self._bandwidth(sol), sol.intersection_loss, sol))

        deduped: list[tuple[float, float, float, Solution]] = []
        for point in frontier:
            if deduped:
                prev = deduped[-1]
                same_bandwidth = abs(prev[1] - point[1]) <= 1e-8
                same_loss = abs(prev[2] - point[2]) <= 1e-8
                if same_bandwidth and same_loss:
                    continue
            deduped.append(point)

        # ε-约束采样得到的点不一定都互不支配：
        # 例如某个 eps 点可能和另一个点带宽相同但损失更大，
        # 或带宽更小且损失更大。这里做一次真正的 Pareto 过滤。
        filtered = self._pareto_filter(deduped)
        self.frontier = filtered
        return filtered

    @staticmethod
    def _pareto_filter(
        points: list[tuple[float, float, float, Solution]],
    ) -> list[tuple[float, float, float, Solution]]:
        """去掉被支配点。

        点结构：(eps, bandwidth_metric, intersection_loss, solution)。
        目标口径：bandwidth_metric 越大越好，intersection_loss 越小越好。
        """
        kept: list[tuple[float, float, float, Solution]] = []
        for i, p in enumerate(points):
            p_bw, p_loss = float(p[1]), float(p[2])
            dominated = False
            for j, q in enumerate(points):
                if i == j:
                    continue
                q_bw, q_loss = float(q[1]), float(q[2])
                better_or_equal = (
                    q_bw >= p_bw - 1e-9 and q_loss <= p_loss + 1e-9
                )
                strictly_better = (
                    q_bw > p_bw + 1e-9 or q_loss < p_loss - 1e-9
                )
                if better_or_equal and strictly_better:
                    dominated = True
                    break
            if not dominated:
                kept.append(p)
        return kept

    def knee_point(self) -> tuple[float, float, Solution] | None:
        """函数名：knee_point；参数：无；返回值：启发式拐点；异常：无。"""
        if len(self.frontier) < 3:
            return None
        p0 = (self._bandwidth(self.frontier[0][3]), self.frontier[0][2])
        p1 = (self._bandwidth(self.frontier[-1][3]), self.frontier[-1][2])
        best = None
        best_d = -1.0
        for _eps, bandwidth, loss, sol in self.frontier:
            distance = abs(
                (p1[0] - p0[0]) * (p0[1] - loss)
                - (p0[0] - bandwidth) * (p1[1] - p0[1])
            )
            if distance > best_d:
                best_d = distance
                best = (bandwidth, loss, sol)
        return best
