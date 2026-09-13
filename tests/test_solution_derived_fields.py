"""Solution 派生摘要字段的回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artery_milp.solution import Solution


class SolutionDerivedFieldTests(unittest.TestCase):
    def test_band_start_fields_follow_lowest_multi_band_number(self):
        sol = Solution(cycle=90.0)
        sol.multi_band_starts = {
            "up": {
                2: {"I1": 20.0, "I2": 30.0},
                1: {"I1": 5.0, "I2": 15.0},
            },
            "down": {},
        }

        self.assertEqual(sol.band_start_up, {"I1": 5.0, "I2": 15.0})
        self.assertEqual(sol.band_start_down, {})

    def test_window_bands_aggregates_across_band_numbers(self):
        sol = Solution(cycle=90.0)
        sol.multi_window_bands = {
            "up": {
                1: {"up.win2@I1-I2": 4.0},
                2: {"up.win2@I1-I2": 3.0},
            },
            "down": {
                1: {"down.win3@I1-I3": 2.5},
            },
        }

        self.assertEqual(
            sol.window_bands,
            {
                "up.win2@I1-I2": 7.0,
                "down.win3@I1-I3": 2.5,
            },
        )

    def test_band_score_is_derived_from_objective_loss_and_weight(self):
        sol = Solution(cycle=90.0)
        sol.band_objective = 12.0
        sol.band_loss = 4.0
        sol.band_loss_weight = 0.5

        self.assertAlmostEqual(sol.band_score, 10.0)

    def test_to_dict_keeps_legacy_summary_keys(self):
        sol = Solution(cycle=90.0)
        sol.multi_band_starts = {"up": {1: {"I1": 2.0, "I2": 4.0}}}
        sol.multi_window_bands = {"up": {1: {"up.win2@I1-I2": 3.0}}}
        sol.band_objective = 10.0
        sol.band_loss = 2.0
        sol.band_loss_weight = 0.25

        data = sol.to_dict()
        for key in (
            "band_start_up",
            "band_start_down",
            "window_bands",
            "band_score",
            "band_loss_weight",
        ):
            self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main()
