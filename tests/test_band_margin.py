"""硬/软边距 BandMarginConfig 的回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artery_milp.models import (Arterial, GreenWindow, Intersection, Segment,
                                SignalPlan)
from artery_milp.solvers.core import (BandMarginConfig, ObjectiveConfig,
                                      SumGroup)
from artery_milp.solvers.stage1 import SegmentedBandSolver
from artery_milp.solvers.stage2 import FullFlexiblePhaseTuneSolver

CYCLE = 90.0


def windows(values):
    return [GreenWindow(*item) for item in values]


def make_arterial() -> Arterial:
    return Arterial(
        cycle=CYCLE,
        intersections={
            "I1": Intersection("I1", [SignalPlan(
                "p1",
                up_segments=windows([(0.05, 0.25)]),
                down_segments=windows([(0.55, 0.75)]),
            )]),
            "I2": Intersection("I2", [SignalPlan(
                "p2",
                up_segments=windows([(0.20, 0.40)]),
                down_segments=windows([(0.40, 0.60)]),
            )]),
            "I3": Intersection("I3", [SignalPlan(
                "p3",
                up_segments=windows([(0.35, 0.55)]),
                down_segments=windows([(0.25, 0.45)]),
            )]),
        },
        segments={
            "S12": Segment("S12", 200, 200, 14, 14),
            "S23": Segment("S23", 200, 200, 14, 14),
        },
        order=["I1", "S12", "I2", "S23", "I3"],
    )


OBJECTIVE = ObjectiveConfig(sum_groups=[SumGroup({"up.global": 1.0, "down.global": 1.0})])


class BandMarginConfigTests(unittest.TestCase):
    def test_validate_rejects_negative_hard_margin(self):
        with self.assertRaises(ValueError):
            BandMarginConfig(hard_margin_up=-0.01).validate()

    def test_validate_rejects_soft_less_than_hard(self):
        with self.assertRaises(ValueError):
            BandMarginConfig(
                hard_margin_up=0.05, soft_margin_up=0.01
            ).validate()

    def test_valid_config_passes(self):
        BandMarginConfig(
            hard_margin_up=0.01,
            hard_margin_down=0.01,
            soft_margin_up=0.03,
            soft_margin_down=0.03,
        ).validate()


class BandMarginSolverTests(unittest.TestCase):
    def test_hard_margin_reduces_bandwidth(self):
        base = SegmentedBandSolver(config=OBJECTIVE).solve(make_arterial())
        margin = BandMarginConfig(
            hard_margin_up=0.03,
            hard_margin_down=0.03,
        )
        robust = SegmentedBandSolver(config=OBJECTIVE, margin=margin).solve(make_arterial())
        self.assertEqual(base.status, "optimal")
        self.assertEqual(robust.status, "optimal")
        self.assertLessEqual(
            robust.total_bandwidth + 1e-6,
            base.total_bandwidth,
        )

    def test_soft_margin_produces_band_loss(self):
        margin = BandMarginConfig(
            hard_margin_up=0.01,
            hard_margin_down=0.01,
            soft_margin_up=0.05,
            soft_margin_down=0.05,
            penalty_up=1.0,
            penalty_down=1.0,
        )
        prior = SegmentedBandSolver(config=OBJECTIVE, margin=margin).solve(make_arterial())
        tuned = FullFlexiblePhaseTuneSolver(config=OBJECTIVE, margin=margin).solve(
            make_arterial(), prior=prior, band_loss_weight=0.5
        )
        self.assertEqual(tuned.status, "optimal")
        self.assertGreater(tuned.band_loss, 0.0)
        self.assertAlmostEqual(
            tuned.band_score,
            tuned.band_objective - 0.5 * tuned.band_loss,
            places=6,
        )

    def test_soft_equals_hard_gives_zero_soft_loss(self):
        margin = BandMarginConfig(
            hard_margin_up=0.05,
            hard_margin_down=0.05,
            soft_margin_up=0.05,
            soft_margin_down=0.05,
        )
        prior = SegmentedBandSolver(config=OBJECTIVE, margin=margin).solve(make_arterial())
        tuned = FullFlexiblePhaseTuneSolver(config=OBJECTIVE, margin=margin).solve(
            make_arterial(), prior=prior, band_loss_weight=0.5
        )
        self.assertEqual(tuned.status, "optimal")
        # 硬边距已经保证 actual margin >= hard = soft，因此软损失应为 0。
        self.assertAlmostEqual(tuned.band_loss, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
