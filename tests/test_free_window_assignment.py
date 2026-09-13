"""自由窗口分配与同窗多带的回归测试。"""

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
from artery_milp.solvers.stage1 import SegmentedBandSolver
from artery_milp.solvers.stage2 import FullFlexiblePhaseTuneSolver

CYCLE = 100.0


def windows(values):
    return [GreenWindow(*item) for item in values]


def make_free_window_arterial() -> Arterial:
    """I2 有两个上行窗口；最优 band 需要在 I2 选第二个窗口。"""
    return Arterial(
        cycle=CYCLE,
        intersections={
            "I1": Intersection("I1", [SignalPlan(
                "p1",
                up_segments=windows([(0.00, 0.20)]),
                down_segments=windows([(0.00, 0.99)]),
            )]),
            "I2": Intersection("I2", [SignalPlan(
                "p2",
                up_segments=windows([(0.10, 0.20), (0.30, 0.50)]),
                down_segments=windows([(0.00, 0.99)]),
            )]),
            "I3": Intersection("I3", [SignalPlan(
                "p3",
                up_segments=windows([(0.45, 0.65)]),
                down_segments=windows([(0.00, 0.99)]),
            )]),
        },
        segments={
            "S12": Segment("S12", 25 * 14, 25 * 14, 14, 14),
            "S23": Segment("S23", 25 * 14, 25 * 14, 14, 14),
        },
        order=["I1", "S12", "I2", "S23", "I3"],
    )


def make_same_window_arterial() -> Arterial:
    """两个路口各只有一个上行窗口，用于验证同窗多带不会复制刷目标。"""
    return Arterial(
        cycle=CYCLE,
        intersections={
            "I1": Intersection("I1", [SignalPlan(
                "p1",
                up_segments=windows([(0.00, 0.20)]),
                down_segments=windows([(0.00, 0.99)]),
            )]),
            "I2": Intersection("I2", [SignalPlan(
                "p2",
                up_segments=windows([(0.20, 0.40)]),
                down_segments=windows([(0.00, 0.99)]),
            )]),
        },
        segments={
            "S12": Segment("S12", 20 * 14, 20 * 14, 14, 14),
        },
        order=["I1", "S12", "I2"],
    )


def make_wide_window_arterial() -> Arterial:
    """宽窗口案例：保证所有方向、所有长度的局部带都能自动回填。"""
    return Arterial(
        cycle=CYCLE,
        intersections={
            f"I{i}": Intersection(
                f"I{i}",
                [SignalPlan(
                    f"p{i}",
                    up_segments=windows([(0.00, 0.90)]),
                    down_segments=windows([(0.00, 0.90)]),
                )],
            )
            for i in range(1, 4)
        },
        segments={
            "S12": Segment("S12", 10 * 14, 10 * 14, 14, 14),
            "S23": Segment("S23", 10 * 14, 10 * 14, 14, 14),
        },
        order=["I1", "S12", "I2", "S23", "I3"],
    )


class FreeWindowAssignmentTests(unittest.TestCase):
    def test_global_band_can_choose_different_window_per_intersection(self):
        arterial = make_free_window_arterial()
        objective = ObjectiveConfig(sum_groups=[SumGroup({"up.global": 1.0})])

        stage1 = SegmentedBandSolver(config=objective, max_bands=1).solve(arterial)
        self.assertEqual(stage1.status, "optimal")
        self.assertEqual(
            stage1.band_window_choices["up"][1]["I2"]["window"],
            2,
        )

        stage2 = FullFlexiblePhaseTuneSolver(config=objective, max_bands=1).solve(
            arterial, prior=stage1
        )
        self.assertEqual(stage2.status, "optimal")
        self.assertEqual(
            stage2.band_window_choices["up"][1]["I2"]["window"],
            2,
        )

    def test_local_window_band_can_choose_different_window(self):
        arterial = make_free_window_arterial()
        objective = ObjectiveConfig(sum_groups=[SumGroup({"up.win2@I1-I2": 1.0})])

        stage1 = SegmentedBandSolver(config=objective, max_bands=1).solve(arterial)
        self.assertEqual(stage1.status, "optimal")
        key = "up.win2@I1-I2"
        self.assertIn(key, stage1.local_band_window_choices)
        self.assertEqual(
            stage1.local_band_window_choices[key][1]["I2"]["window"],
            2,
        )
        self.assertIn(key, stage1.window_band_ranges)

        stage2 = FullFlexiblePhaseTuneSolver(config=objective, max_bands=1).solve(
            arterial, prior=stage1
        )
        self.assertEqual(stage2.status, "optimal")
        self.assertEqual(
            stage2.local_band_window_choices[key][1]["I2"]["window"],
            2,
        )

    def test_same_window_multi_band_is_ordered_not_inflated(self):
        arterial = make_same_window_arterial()
        objective = ObjectiveConfig(sum_groups=[SumGroup({"up.global": 1.0})])

        stage1 = SegmentedBandSolver(
            config=objective, max_bands=2, band_gap=1.0
        ).solve(arterial)
        self.assertEqual(stage1.status, "optimal")
        # 同方向两条 band 都在同一个 20s 窗口内，顺序约束应保证带宽总和不膨胀。
        self.assertLessEqual(
            sum(stage1.multi_bandwidths["up"].values()),
            20.0 + 1e-6,
        )
        self.assertGreater(stage1.band_objective, 0.0)
        for band_no in (1, 2):
            self.assertEqual(
                stage1.band_window_choices["up"][band_no]["I1"]["window"],
                1,
            )
        self.assertIn(1, stage1.band_order_choices["up"])
        self.assertIn(2, stage1.band_order_choices["up"][1])

        stage2 = FullFlexiblePhaseTuneSolver(
            config=objective, max_bands=2, band_gap=1.0
        ).solve(arterial, prior=stage1)
        self.assertEqual(stage2.status, "optimal")
        self.assertLessEqual(
            sum(stage2.multi_bandwidths["up"].values()),
            20.0 + 1e-6,
        )

    def test_all_direction_length_window_bands_are_autofilled(self):
        arterial = make_wide_window_arterial()
        objective = ObjectiveConfig(sum_groups=[SumGroup({"up.global": 1.0})])

        stage1 = SegmentedBandSolver(config=objective, max_bands=1).solve(arterial)
        self.assertEqual(stage1.status, "optimal")
        expected = {
            "up.win2@I1-I2",
            "up.win2@I2-I3",
            "up.win3@I1-I3",
            "down.win2@I1-I2",
            "down.win2@I2-I3",
            "down.win3@I1-I3",
        }
        self.assertTrue(expected.issubset(set(stage1.window_band_ranges)))
        self.assertTrue(expected.issubset(set(stage1.window_bands)))
        for direction in ("up", "down"):
            for key in expected:
                if key.startswith(direction):
                    self.assertIn(key, stage1.multi_window_bands[direction][1])

        stage2 = FullFlexiblePhaseTuneSolver(config=objective, max_bands=1).solve(
            arterial, prior=stage1
        )
        self.assertEqual(stage2.status, "optimal")
        self.assertTrue(expected.issubset(set(stage2.window_band_ranges)))



if __name__ == "__main__":
    unittest.main()
