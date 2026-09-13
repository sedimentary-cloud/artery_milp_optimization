"""EpsilonConstraintRunner Pareto 过滤测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artery_milp.solution import Solution
from artery_milp.solvers.pipeline import EpsilonConstraintRunner


class ParetoFilterTests(unittest.TestCase):
    def test_dominated_points_are_removed(self):
        points = [
            (0.0, 30.0, 0.0, Solution(cycle=90.0)),   # 最优：低损失高带宽
            (0.3, 30.5, 0.3, Solution(cycle=90.0)),   # 更优：带宽更高、损失略高但可接受？不，被下面点支配？
            (0.3, 29.0, 0.9, Solution(cycle=90.0)),   # 被第一个和第二个支配
            (0.6, 28.0, 0.0, Solution(cycle=90.0)),   # 被 (30.0, 0.0) 支配
        ]
        filtered = EpsilonConstraintRunner._pareto_filter(points)
        filtered_values = {(float(p[1]), float(p[2])) for p in filtered}
        self.assertNotIn((29.0, 0.9), filtered_values)
        self.assertIn((30.0, 0.0), filtered_values)
        self.assertIn((30.5, 0.3), filtered_values)
        self.assertEqual(len(filtered), 2)


if __name__ == "__main__":
    unittest.main()
