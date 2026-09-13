"""迭代式两阶段求解：不断把 Stage 2 调好的端点写回 Stage 1。

与 ``TwoStageSolver`` 的区别：
- ``TwoStageSolver`` 只跑一次 Stage 1 + Stage 2；
- ``IterativeTwoStageSolver`` 会把 Stage 2 微调后的绿灯端点写回
  ``Arterial``，再重新跑 Stage 1 的窗口分配，直到窗口分配稳定。

收敛判据同时检查窗口分配和 Stage 2 实际得分：
- 窗口分配重复且 Stage 2 得分不再显著提高，才判定收敛/震荡；
- 如果重复状态下得分仍在显著提高，则继续迭代；
- 连续同一状态得分不再提高 -> ``iterative_converged``；
- 回到历史状态且得分未超过该状态历史最佳 -> ``iterative_cycle``；
- 每轮 Stage 2 都必须返回 optimal，否则抛出 ``RuntimeError``。
"""

from __future__ import annotations

from ...models import Arterial, GreenWindow, Intersection, SignalPlan
from ...solution import Solution
from ..core.base import Solver
from ..stage1.segmented_band import SegmentedBandSolver
from ..stage2.phase_tune import FullFlexiblePhaseTuneSolver
from .config import TwoStageConfig


class IterativeTwoStageSolver(Solver):
    """两阶段固定点迭代求解器。"""

    name = "iterative-two-stage"

    def __init__(
        self,
        config: TwoStageConfig,
        max_iterations: int = 10,
        score_tol: float = 1e-8,
    ) -> None:
        """函数名：__init__；参数：TwoStageConfig、最大迭代次数、得分容差；返回值：无；异常：ValueError。"""
        config.validate()
        if int(max_iterations) <= 0:
            raise ValueError("max_iterations 必须为正")
        if float(score_tol) < 0:
            raise ValueError("score_tol 不能为负")
        self.config = config
        self.max_iterations = int(max_iterations)
        self.score_tol = float(score_tol)
        self.history: list[Solution] = []

    def solve(self, arterial: Arterial) -> Solution:
        """函数名：solve；参数：arterial；返回值：Solution；异常：RuntimeError/ValueError。"""
        cfg = self.config
        mode = cfg.band.mode

        stage1 = SegmentedBandSolver(
            config=cfg.band.objective,
            max_loops=cfg.max_loops,
            up_global_output=True,
            down_global_output=(mode == "global"),
            margin=cfg.margin,
        )
        stage2 = FullFlexiblePhaseTuneSolver(
            config=cfg.band.objective,
            max_loops=cfg.max_loops,
            up_global_output=True,
            down_global_output=(mode == "global"),
            margin=cfg.margin,
        )

        current_arterial = arterial
        history: list[Solution] = []
        # 每个窗口分配状态对应的历史最佳 Stage 2 得分。
        state_best_score: dict[tuple, float] = {}
        previous_state: tuple | None = None
        best_solution: Solution | None = None
        best_score = float("-inf")

        for iteration in range(1, self.max_iterations + 1):
            s1 = stage1.solve(
                current_arterial,
                band_loss_weight=cfg.band.band_loss_weight,
            )
            if s1.status != "optimal":
                s1.status = f"{s1.status}|iterative_stage1_failed"
                s1.solver_msg = (
                    f"{s1.solver_msg}; iterative stopped at Stage 1 "
                    f"iteration {iteration}"
                )
                self.history = history
                return s1

            s2 = stage2.solve(
                current_arterial,
                prior=s1,
                loss_builder=cfg.intersection.loss_builder,
                constraint_builder=cfg.intersection.constraint_builder,
                band_loss_weight=cfg.band.band_loss_weight,
                objective="bandwidth",
                tunable_intersections=cfg.intersection.tunable_intersections,
            )
            if s2.status != "optimal":
                raise RuntimeError(
                    f"Stage 2 failed at iteration {iteration}: {s2.status}; "
                    f"{s2.solver_msg}"
                )

            history.append(s2)
            state = self._window_assignment_state(s2)
            score = float(s2.band_score)

            # 记录全局最佳解，避免最终返回得分回落的解。
            if best_solution is None or self._score_improved(score, best_score):
                best_solution = s2
                best_score = score

            previous_best = state_best_score.get(state)
            if previous_best is None:
                state_best_score[state] = score
                previous_state = state
                current_arterial = self._clone_arterial_with_tuned_windows(
                    current_arterial, s2
                )
                continue

            # 窗口分配重复，但 Stage 2 得分仍在显著提高：继续迭代。
            if self._score_improved(score, previous_best):
                state_best_score[state] = score
                previous_state = state
                current_arterial = self._clone_arterial_with_tuned_windows(
                    current_arterial, s2
                )
                continue

            # 窗口分配重复且得分不再提高：
            # - 连续同一状态：收敛；
            # - 回到更早的历史状态：震荡。
            if previous_state == state:
                result = s2
                if self._score_improved(best_score, score):
                    result = best_solution
                result.status = f"{result.status}|iterative_converged"
                result.solver_msg = (
                    f"{result.solver_msg}; iterative converged at iteration "
                    f"{iteration}: window assignment repeated and Stage 2 "
                    "score did not improve"
                )
                self.history = history
                return result

            predecessor = history[-2] if len(history) >= 2 else s2
            result = predecessor
            if (
                best_solution is not None
                and self._score_improved(best_score, float(predecessor.band_score))
            ):
                result = best_solution
            result.status = f"{result.status}|iterative_cycle"
            result.solver_msg = (
                f"{result.solver_msg}; oscillation cycle detected at iteration "
                f"{iteration}: window assignment repeated and Stage 2 score "
                "did not improve"
            )
            self.history = history
            return result

        last = best_solution if best_solution is not None else history[-1]
        last.status = f"{last.status}|iterative_max_iter"
        last.solver_msg = (
            f"{last.solver_msg}; iterative reached max_iterations="
            f"{self.max_iterations}, returned best-scoring solution"
        )
        self.history = history
        return last

    def _score_improved(self, score: float, reference: float) -> bool:
        """判断 Stage 2 得分是否相对 reference 有显著提高。"""
        return float(score) > float(reference) + self.score_tol

    @staticmethod
    def _window_assignment_state(solution: Solution) -> tuple:
        """函数名：_window_assignment_state；参数：solution；返回值：可哈希窗口分配状态。"""
        global_items: list[tuple] = []
        for direction in sorted(solution.band_window_choices.keys()):
            band_map = solution.band_window_choices.get(direction, {})
            for band_no in sorted(band_map.keys()):
                intersection_map = band_map[band_no]
                for intersection in sorted(intersection_map.keys()):
                    choice = intersection_map[intersection] or {}
                    global_items.append((
                        direction,
                        int(band_no),
                        intersection,
                        choice.get("plan"),
                        int(choice.get("window", 0)) if choice.get("window") is not None else None,
                    ))

        local_items: list[tuple] = []
        for key in sorted(solution.local_band_window_choices.keys()):
            band_map = solution.local_band_window_choices.get(key, {})
            for band_no in sorted(band_map.keys()):
                intersection_map = band_map[band_no]
                for intersection in sorted(intersection_map.keys()):
                    choice = intersection_map[intersection] or {}
                    local_items.append((
                        key,
                        int(band_no),
                        intersection,
                        choice.get("plan"),
                        int(choice.get("window", 0)) if choice.get("window") is not None else None,
                    ))

        return (tuple(global_items), tuple(local_items))

    @classmethod
    def _clone_arterial_with_tuned_windows(cls,
                                           arterial: Arterial,
                                           solution: Solution) -> Arterial:
        """函数名：_clone_arterial_with_tuned_windows；参数：arterial、solution；返回值：新 Arterial。"""
        new_intersections: dict[str, Intersection] = {}
        for inter in arterial.intersection_order:
            selected_name = solution.plan_choices.get(inter.name)
            new_plans: list[SignalPlan] = []
            for plan in inter.plans:
                if plan.name != selected_name:
                    new_plans.append(plan)
                    continue
                times = solution.segment_times.get(inter.name, {})
                new_plans.append(
                    cls._clone_plan_with_tuned_windows(plan, times, arterial.cycle)
                )
            new_intersections[inter.name] = Intersection(inter.name, new_plans)

        return Arterial(
            cycle=arterial.cycle,
            intersections=new_intersections,
            segments=dict(arterial.segments),
            order=list(arterial.order),
        )

    @staticmethod
    def _clone_plan_with_tuned_windows(plan: SignalPlan,
                                       times: dict[str, float],
                                       cycle: float) -> SignalPlan:
        """函数名：_clone_plan_with_tuned_windows；参数：plan、times、cycle；返回值：新 SignalPlan。"""
        if not times:
            return plan

        def build_segments(direction: str, original: list[GreenWindow]) -> list[GreenWindow]:
            result: list[GreenWindow] = []
            for idx, old in enumerate(original, start=1):
                start_key = f"{direction}.{idx}.start"
                end_key = f"{direction}.{idx}.end"
                if start_key not in times or end_key not in times:
                    result.append(old)
                    continue
                start = float(times[start_key]) / cycle
                end = float(times[end_key]) / cycle
                if not (0.0 <= start < end <= 1.0):
                    result.append(old)
                else:
                    result.append(GreenWindow(start, end))
            return result

        up_segments = build_segments("up", plan.up_segments)
        down_segments = build_segments("down", plan.down_segments)

        try:
            return SignalPlan(
                name=plan.name,
                up_segments=up_segments,
                down_segments=down_segments,
                signal_constraints=list(plan.signal_constraints),
                signal_losses=list(plan.signal_losses),
                metadata=dict(plan.metadata),
            )
        except ValueError:
            # 微调后的窗口不能通过 SignalPlan 校验时，保持原方案。
            return plan


__all__ = ["IterativeTwoStageSolver"]
