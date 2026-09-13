"""IterativeTwoStageSolver 的收敛/震荡/失败语义测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artery_milp.models import (Arterial, GreenWindow, Intersection, Segment,
                                SignalPlan)
from artery_milp.solution import Solution
from artery_milp.solvers.core import (BandMarginConfig, ObjectiveConfig,
                                      SumGroup)
from artery_milp.solvers.pipeline import (BandObjectiveConfig,
                                          IterativeTwoStageSolver,
                                          TwoStageConfig)
from artery_milp.solvers.stage1 import SegmentedBandSolver
from artery_milp.solvers.stage2 import FullFlexiblePhaseTuneSolver
from artery_milp.tests.test_band_margin import OBJECTIVE as MARGIN_OBJECTIVE
from artery_milp.tests.test_band_margin import make_arterial as make_margin_arterial

CYCLE = 90.0


def windows(values):
    return [GreenWindow(*item) for item in values]


def make_arterial() -> Arterial:
    return Arterial(
        cycle=CYCLE,
        intersections={
            f"I{i}": Intersection(f"I{i}", [SignalPlan(
                f"p{i}",
                up_segments=windows([(0.00, 0.50)]),
                down_segments=windows([(0.50, 0.90)]),
            )])
            for i in range(1, 4)
        },
        segments={
            "S12": Segment("S12", 140.0, 140.0, 14.0, 14.0),
            "S23": Segment("S23", 140.0, 140.0, 14.0, 14.0),
        },
        order=["I1", "S12", "I2", "S23", "I3"],
    )


def fake_solution(window_i2: int, status: str = "optimal") -> Solution:
    sol = Solution(cycle=CYCLE, status=status)
    sol.plan_choices = {"I1": "p1", "I2": "p2", "I3": "p3"}
    sol.band_window_choices = {
        "up": {
            1: {
                "I1": {"plan": "p1", "window": 1},
                "I2": {"plan": "p2", "window": window_i2},
                "I3": {"plan": "p3", "window": 1},
            }
        },
        "down": {
            1: {
                "I1": {"plan": "p1", "window": 1},
                "I2": {"plan": "p2", "window": 1},
                "I3": {"plan": "p3", "window": 1},
            }
        },
    }
    sol.local_band_window_choices = {}
    sol.segment_times = {}
    return sol


class IterativePipelineLogicTests(unittest.TestCase):
    def make_config(self) -> TwoStageConfig:
        objective = ObjectiveConfig(sum_groups=[SumGroup({"up.global": 1.0})])
        return TwoStageConfig(
            band=BandObjectiveConfig(mode="global", objective=objective),
            max_loops=3,
        )

    def test_consecutive_same_window_assignment_converges(self):
        first = fake_solution(window_i2=1)
        second = fake_solution(window_i2=1)
        with patch.object(SegmentedBandSolver, "solve", return_value=first), \
             patch.object(FullFlexiblePhaseTuneSolver, "solve", side_effect=[first, second]):
            solver = IterativeTwoStageSolver(self.make_config(), max_iterations=5)
            result = solver.solve(make_arterial())

        self.assertIs(result, second)
        self.assertIn("iterative_converged", result.status)
        self.assertEqual(len(solver.history), 2)

    def test_window_assignment_cycle_returns_cycle_first_solution(self):
        first = fake_solution(window_i2=1)
        second = fake_solution(window_i2=2)
        third = fake_solution(window_i2=1)
        with patch.object(SegmentedBandSolver, "solve", return_value=first), \
             patch.object(FullFlexiblePhaseTuneSolver, "solve", side_effect=[first, second, third]):
            solver = IterativeTwoStageSolver(self.make_config(), max_iterations=5)
            result = solver.solve(make_arterial())

        self.assertIs(result, first)
        self.assertIn("iterative_cycle", result.status)
        self.assertEqual(len(solver.history), 3)

    def test_stage2_failure_raises(self):
        first = fake_solution(window_i2=1)
        failed = fake_solution(window_i2=2, status="infeasible")
        with patch.object(SegmentedBandSolver, "solve", return_value=first), \
             patch.object(FullFlexiblePhaseTuneSolver, "solve", side_effect=[failed]):
            solver = IterativeTwoStageSolver(self.make_config(), max_iterations=5)
            with self.assertRaises(RuntimeError):
                solver.solve(make_arterial())

    def test_real_iterative_pipeline_runs_to_stable(self):
        cfg = TwoStageConfig(
            band=BandObjectiveConfig(mode="global", objective=MARGIN_OBJECTIVE),
            margin=BandMarginConfig(
                hard_margin_up=0.01,
                hard_margin_down=0.01,
                soft_margin_up=0.02,
                soft_margin_down=0.02,
            ),
        )
        solver = IterativeTwoStageSolver(cfg, max_iterations=5)
        solution = solver.solve(make_margin_arterial())

        self.assertIn("iterative", solution.status)
        self.assertGreaterEqual(len(solver.history), 1)



if __name__ == "__main__":
    unittest.main()
