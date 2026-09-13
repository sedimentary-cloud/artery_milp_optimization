"""AlignmentLossBuilder 简化后的回归测试。

现在它只处理上下行全局带：
    up:   tU_i + 0.5 * b_up
    down: tD_i + 0.5 * b_down
未写目标的路口跳过；不再生成逐路段 bD_/bU_ 对齐项。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artery_milp.solvers.builders import AlignmentLossBuilder


class AlignmentLossBuilderTests(unittest.TestCase):
    def test_only_configured_intersections_get_specs(self):
        builder = AlignmentLossBuilder(
            targets_up={"I2": 0.50},
            targets_down={"I2": 0.70},
            tolerance=0.05,
            weight_up=1.0,
            weight_down=2.0,
        )
        specs = builder.to_linear_specs(["I1", "I2", "I3"])

        self.assertEqual(len(specs), 4)
        self.assertEqual(
            {spec.name for spec in specs},
            {
                "align.up.I2.upper",
                "align.up.I2.lower",
                "align.down.I2.upper",
                "align.down.I2.lower",
            },
        )
        up_specs = [s for s in specs if s.name.startswith("align.up")]
        down_specs = [s for s in specs if s.name.startswith("align.down")]
        for spec in up_specs:
            self.assertEqual(spec.terms, {"tU_I2": 1.0, "b_up": 0.5})
            self.assertTrue(spec.soft)
            self.assertEqual(spec.penalty, 1.0)
        for spec in down_specs:
            self.assertEqual(spec.terms, {"tD_I2": 1.0, "b_down": 0.5})
            self.assertTrue(spec.soft)
            self.assertEqual(spec.penalty, 2.0)

    def test_no_local_segment_terms_are_generated(self):
        builder = AlignmentLossBuilder(
            targets_up={"I1": 0.10, "I2": 0.20},
            targets_down={"I1": 0.30, "I2": 0.40},
        )
        specs = builder.to_linear_specs(["I1", "I2"])
        for spec in specs:
            joined = " ".join(spec.terms)
            self.assertNotIn("bD_", joined)
            self.assertNotIn("bU_", joined)
            self.assertNotIn("win", joined)

    def test_upper_and_lower_tolerance(self):
        builder = AlignmentLossBuilder(
            targets_up={"I1": 0.50},
            tolerance=0.05,
        )
        specs = builder.to_linear_specs(["I1"])
        self.assertEqual(len(specs), 2)
        upper = next(s for s in specs if s.sense == "<=")
        lower = next(s for s in specs if s.sense == ">=")
        self.assertAlmostEqual(upper.rhs, 0.55)
        self.assertAlmostEqual(lower.rhs, 0.45)


if __name__ == "__main__":
    unittest.main()
