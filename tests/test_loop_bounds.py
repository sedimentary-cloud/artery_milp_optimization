"""自适应 max_loops 的回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artery_milp.models import (Arterial, GreenWindow, Intersection, Segment,
                                SignalPlan)
from artery_milp.solvers.core import ObjectiveConfig, SumGroup
from artery_milp.solvers.core.base import compute_effective_max_loops
from artery_milp.solvers.stage1 import SegmentedBandSolver

CYCLE = 90.0


def windows(values):
    return [GreenWindow(*item) for item in values]


def make_wide_arterial(travel_time: float = 200.0) -> Arterial:
    length = travel_time * 14.0
    return Arterial(
        cycle=CYCLE,
        intersections={
            "I1": Intersection("I1", [SignalPlan(
                "p1",
                up_segments=windows([(0.00, 0.90)]),
                down_segments=windows([(0.00, 0.90)]),
            )]),
            "I2": Intersection("I2", [SignalPlan(
                "p2",
                up_segments=windows([(0.00, 0.90)]),
                down_segments=windows([(0.00, 0.90)]),
            )]),
        },
        segments={
            "S12": Segment("S12", length, length, 14.0, 14.0),
        },
        order=["I1", "S12", "I2"],
    )


class AdaptiveMaxLoopsTests(unittest.TestCase):
    def test_user_value_is_lower_bound(self):
        arterial = make_wide_arterial(travel_time=200.0)
        # ceil(200 / 90) + 1 = 4
        self.assertEqual(compute_effective_max_loops(arterial, 1), 4)
        self.assertEqual(compute_effective_max_loops(arterial, 7), 7)

    def test_stage1_uses_auto_expanded_loop_bound(self):
        arterial = make_wide_arterial(travel_time=200.0)
        objective = ObjectiveConfig(
            sum_groups=[SumGroup({"up.global": 1.0, "down.global": 1.0})]
        )
        # 用户只给 max_loops=1；如果没有自适应，200s 路段在 90s 周期下
        # 需要 m=-2 才能传播，应该会 infeasible。
        solution = SegmentedBandSolver(
            config=objective, max_bands=1, max_loops=1
        ).solve(arterial)
        self.assertEqual(solution.status, "optimal")
        self.assertGreater(solution.band_objective, 0.0)


if __name__ == "__main__":
    unittest.main()
