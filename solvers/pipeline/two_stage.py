"""两阶段求解与 ε-约束帕累托扫描。

当前主语义已经统一到“段级输入”：
- Stage 1：方案/窗口选择 + 带宽目标；
- Stage 2：锁定方案后，优化段级端点、业务约束与带宽；
- intersection_loss：段级软损失 + 软 LinearSpec slack；
- band_loss：AlignmentLossBuilder 等绿波带层软损失。
"""

from __future__ import annotations

import numpy as np

from ...solution import Solution
from ..core.base import Solver
from ..core.objective import composite_config, oneway_config
from ..stage1.segmented_band import SegmentedBandSolver
from ..stage2.phase_tune import (FullFlexiblePhaseTuneSolver,
                                 PhaseTuneSolver)
from .config import TwoStageConfig


class TwoStageSolver(Solver):
    """两阶段编排器：Stage 1 选方案，Stage 2 优化段级端点。"""

    name = "two-stage"

    def __init__(
        self,
        stage1: Solver | TwoStageConfig | None = None,
        mode: str = "global",
        loss_builder=None,
        constraint_builder=None,
        alignment_builder=None,
        band_loss_weight: float = 0.0,
        config: TwoStageConfig | None = None,
        **tune_kwargs,
    ) -> None:
        """函数名：__init__；参数：stage1、mode 等；返回值：无；异常：无。"""
        if isinstance(stage1, TwoStageConfig) and config is None:
            config = stage1
            stage1 = None
        self.config = config
        self.stage1 = stage1
        self.mode = mode
        self.loss_builder = loss_builder
        self.constraint_builder = constraint_builder
        self.alignment_builder = alignment_builder
        self.band_loss_weight = band_loss_weight
        self.tune_kwargs = tune_kwargs

    def _solve_with_config(self, arterial) -> Solution:
        """函数名：_solve_with_config；参数：arterial；返回值：Solution；异常：无。"""
        cfg = self.config
        cfg.validate()
        mode = cfg.band.mode

        stage1 = SegmentedBandSolver(
            config=cfg.band.objective,
            max_loops=cfg.max_loops,
            up_style="global",
            down_style=("global" if mode == "global" else "local"),
            up_global_output=True,
            down_global_output=(mode == "global"),
        )
        s1 = stage1.solve(
            arterial,
            alignment_builder=cfg.band.alignment_builder,
            band_loss_weight=cfg.band.band_loss_weight,
        )

        try:
            stage2 = FullFlexiblePhaseTuneSolver(
                config=cfg.band.objective,
                max_loops=cfg.max_loops,
                up_global_output=True,
                down_global_output=(mode == "global"),
            )
            s2 = stage2.solve(
                arterial,
                prior=s1,
                loss_builder=cfg.intersection.loss_builder,
                constraint_builder=cfg.intersection.constraint_builder,
                alignment_builder=cfg.band.alignment_builder,
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

    def solve(self, arterial) -> Solution:
        """函数名：solve；参数：arterial；返回值：Solution；异常：ValueError。"""
        if self.config is not None:
            return self._solve_with_config(arterial)
        if self.stage1 is None:
            if self.mode == "global":
                objective_mode = self.tune_kwargs.get("objective_mode", "sum")
                balance_eps = self.tune_kwargs.get("balance_eps", 0.1)
                balance_terms = self.tune_kwargs.get("balance_terms", ("up", "down"))
                objective = composite_config(
                    up_weight=self.tune_kwargs.get("up_weight", 1.0),
                    down_weight=self.tune_kwargs.get("down_weight", 1.0),
                    objective_mode=objective_mode,
                    balance_eps=balance_eps,
                    balance_terms=balance_terms,
                )
            elif self.mode == "oneway":
                objective = oneway_config(
                    up_weight=self.tune_kwargs.get("up_weight", 1.0),
                    window_weights=self.tune_kwargs.get("window_weights"),
                    n_intersections=len(arterial.intersection_order),
                    normalize_window_weights=self.tune_kwargs.get(
                        "normalize_window_weights", True
                    ),
                )
            else:
                raise ValueError(f"unknown mode: {self.mode}")
            self.stage1 = SegmentedBandSolver(
                config=objective,
                max_loops=self.tune_kwargs.get("max_loops", 3),
                up_style="global",
                down_style=("global" if self.mode == "global" else "local"),
                up_global_output=True,
                down_global_output=(self.mode == "global"),
            )

        import inspect

        stage1_params = inspect.signature(self.stage1.solve).parameters
        stage1_kwargs = {}
        if self.alignment_builder is not None and "alignment_builder" in stage1_params:
            stage1_kwargs["alignment_builder"] = self.alignment_builder
        if self.band_loss_weight and "band_loss_weight" in stage1_params:
            stage1_kwargs["band_loss_weight"] = self.band_loss_weight
            
        s1 = self.stage1.solve(arterial, **stage1_kwargs)

        try:
            tuner = PhaseTuneSolver(mode=self.mode, **self.tune_kwargs)
            s2 = tuner.solve(
                arterial,
                prior=s1,
                loss_builder=self.loss_builder,
                constraint_builder=self.constraint_builder,
                alignment_builder=self.alignment_builder,
                band_loss_weight=self.band_loss_weight,
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
        mode: str = "global",
        loss_builder=None,
        constraint_builder=None,
        alignment_builder=None,
        n_points: int = 5,
        down_weight: float = 1.0,
        up_weight: float = 1.0,
        window_weights: dict[int, float] | None = None,
        max_loops: int = 3,
        metric: str | None = None,
        objective_mode: str = "sum",
        balance_eps: float = 0.1,
        balance_terms: tuple[str, ...] = ("up", "down"),
        band_loss_weight: float = 0.0,
        config: TwoStageConfig | None = None,
    ) -> None:
        """函数名：__init__；参数：mode、loss_builder 等；返回值：无；异常：无。"""
        if isinstance(mode, TwoStageConfig) and config is None:
            config = mode
            mode = config.band.mode

        if config is not None:
            config.validate()
            self.mode = config.band.mode
            self.loss_builder = config.intersection.loss_builder
            self.constraint_builder = config.intersection.constraint_builder
            self.alignment_builder = config.band.alignment_builder
            self.band_loss_weight = config.band.band_loss_weight
            self.max_loops = config.max_loops
            self._objective_config = config.band.objective
            if metric is None:
                metric = "band_score"
        else:
            self.mode = mode
            self.loss_builder = loss_builder
            self.constraint_builder = constraint_builder
            self.alignment_builder = alignment_builder
            self.band_loss_weight = band_loss_weight
            self.max_loops = max_loops
            self._objective_config = None

        self.n_points = n_points
        self.down_weight = down_weight
        self.up_weight = up_weight
        self.window_weights = window_weights
        if metric is None:
            if band_loss_weight != 0.0:
                metric = "band_score"
            elif objective_mode == "balanced":
                metric = "balanced"
            elif objective_mode == "balanced_composite":
                metric = "objective"
            else:
                metric = "sum"
        self.metric = metric
        self.objective_mode = objective_mode
        self.balance_eps = balance_eps
        self.balance_terms = tuple(balance_terms)
        self.frontier: list[tuple[float, float, float, Solution]] = []

    def _make_tuner(self) -> PhaseTuneSolver:
        """函数名：_make_tuner；参数：无；返回值：PhaseTuneSolver；异常：无。"""
        if self._objective_config is not None:
            return PhaseTuneSolver(
                mode=self.mode,
                max_loops=self.max_loops,
                objective_config=self._objective_config,
            )
        return PhaseTuneSolver(
            mode=self.mode,
            down_weight=self.down_weight,
            up_weight=self.up_weight,
            window_weights=self.window_weights,
            max_loops=self.max_loops,
            objective_mode=self.objective_mode,
            balance_eps=self.balance_eps,
            balance_terms=self.balance_terms,
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

    def run(self, arterial, prior: Solution) -> list[tuple[float, float, float, Solution]]:
        """函数名：run；参数：arterial、prior；返回值：前沿点列表；异常：无。"""
        tuner = self._make_tuner()

        s_hi = tuner.solve(
            arterial,
            prior=prior,
            loss_builder=self.loss_builder,
            constraint_builder=self.constraint_builder,
            alignment_builder=self.alignment_builder,
            band_loss_weight=self.band_loss_weight,
            objective="bandwidth",
        )
        if s_hi.status != "optimal":
            return []
        l_hi = s_hi.intersection_loss

        s_lo = tuner.solve(
            arterial,
            prior=prior,
            loss_builder=self.loss_builder,
            constraint_builder=self.constraint_builder,
            alignment_builder=self.alignment_builder,
            band_loss_weight=self.band_loss_weight,
            objective="loss",
        )
        if s_lo.status != "optimal":
            return []
        l_lo = s_lo.intersection_loss

        eps_values = list(np.linspace(l_lo, l_hi, max(self.n_points, 2)))
        frontier: list[tuple[float, float, float, Solution]] = []
        for eps in eps_values:
            sol = tuner.solve(
                arterial,
                prior=prior,
                loss_builder=self.loss_builder,
                constraint_builder=self.constraint_builder,
                alignment_builder=self.alignment_builder,
                band_loss_weight=self.band_loss_weight,
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

        self.frontier = deduped
        return deduped

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
