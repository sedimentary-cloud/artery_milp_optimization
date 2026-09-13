"""build_objective_config 的回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artery_milp.solvers.core import (BalanceGroup, ObjectiveConfig, SumGroup,
                                      build_objective_config,
                                      composite_config, oneway_config)


class BuildObjectiveConfigTests(unittest.TestCase):
    def test_passthrough_objective_config(self):
        obj = ObjectiveConfig(sum_groups=[SumGroup({"up.global": 1.0})])
        self.assertIs(build_objective_config(objective_config=obj), obj)

    def test_global_sum(self):
        cfg = build_objective_config(
            mode="global", up_weight=2.0, down_weight=3.0, objective_mode="sum"
        )
        self.assertEqual(len(cfg.sum_groups), 1)
        self.assertEqual(cfg.sum_groups[0].terms, {"up.global": 2.0, "down.global": 3.0})

    def test_global_balanced(self):
        cfg = build_objective_config(mode="global", objective_mode="balanced")
        self.assertEqual(len(cfg.balance_groups), 1)
        self.assertEqual(cfg.balance_groups[0].members, ["up.global", "down.global"])
        self.assertEqual(cfg.balance_groups[0].weight, 1.0)

    def test_global_balanced_composite(self):
        cfg = build_objective_config(
            mode="global",
            up_weight=1.0,
            down_weight=1.0,
            objective_mode="balanced_composite",
            balance_eps=0.25,
        )
        self.assertEqual(len(cfg.sum_groups), 1)
        self.assertEqual(len(cfg.balance_groups), 1)
        self.assertEqual(cfg.balance_groups[0].weight, 0.25)

    def test_oneway_contains_down_segments(self):
        cfg = build_objective_config(
            mode="oneway",
            up_weight=1.0,
            window_weights={2: 1.0},
            n_intersections=4,
        )
        terms = cfg.sum_groups[0].terms
        self.assertIn("up.global", terms)
        self.assertIn("down.seg1", terms)
        self.assertIn("down.seg2", terms)
        self.assertIn("down.seg3", terms)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            build_objective_config(mode="does-not-exist")


if __name__ == "__main__":
    unittest.main()
