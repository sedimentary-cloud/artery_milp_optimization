"""Stage 1 结果自动计算 plan 级 SignalLoss 的回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artery_milp.models import (Arterial, GreenWindow, Intersection, Segment,
                                SignalLoss, SignalPlan)
from artery_milp.solvers.core import ObjectiveConfig, SumGroup
from artery_milp.solvers.stage1 import SegmentedBandSolver

CYCLE = 90.0


def windows(values):
    return [GreenWindow(*item) for item in values]


class Stage1IntersectionLossTests(unittest.TestCase):
    def test_selected_plan_signal_loss_is_reported(self):
        arterial = Arterial(
            cycle=CYCLE,
            intersections={
                "I1": Intersection("I1", [SignalPlan(
                    "p1",
                    up_segments=windows([(0.00, 0.50)]),
                    down_segments=windows([(0.38, 0.86)]),
                    signal_losses=[SignalLoss(
                        terms={"down.1.start": 1.0},
                        upper_threshold=0.10,
                        upper_slope=2.9,
                        name="down.1.start 不要晚于 0.10",
                    )],
                )]),
                "I2": Intersection("I2", [SignalPlan(
                    "p2",
                    up_segments=windows([(0.20, 0.70)]),
                    down_segments=windows([(0.10, 0.60)]),
                )]),
            },
            segments={
                "S12": Segment("S12", 100.0, 100.0, 10.0, 10.0),
            },
            order=["I1", "S12", "I2"],
        )
        objective = ObjectiveConfig(sum_groups=[SumGroup({"up.global": 1.0})])

        sol = SegmentedBandSolver(
            config=objective, max_bands=1, max_loops=3
        ).solve(arterial)

        self.assertEqual(sol.status, "optimal")
        expected = (0.38 - 0.10) * CYCLE * 2.9
        self.assertAlmostEqual(sol.intersection_loss, expected, places=6)


if __name__ == "__main__":
    unittest.main()
