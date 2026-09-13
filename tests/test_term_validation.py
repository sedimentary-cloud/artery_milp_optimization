"""统一 term 校验层的回归测试。

运行方式（在仓库目录下）::

    ./conda-envs/artery_milp/bin/python -m unittest discover -s tests -v

或::

    PYTHONPATH=/home/qktx python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# tests/ -> artery_milp/ -> 项目根目录（/home/qktx）
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artery_milp.models import (Arterial, GreenWindow, Intersection, Segment,
                                SignalPlan)
from artery_milp.solvers.builders import (ConstraintBuilder, LinearSpec,
                                          SegmentLossBuilder, SegmentLossSpec,
                                          TermValidationContext,
                                          TermValidationError,
                                          common_endpoint_terms)
from artery_milp.solvers.core import ObjectiveConfig, SumGroup
from artery_milp.solvers.stage1 import SegmentedBandSolver
from artery_milp.solvers.stage2 import FullFlexiblePhaseTuneSolver

CYCLE = 90.0


def windows(values):
    return [GreenWindow(*item) for item in values]


def one_plan_arterial() -> Arterial:
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


def multi_plan_arterial() -> Arterial:
    arterial = one_plan_arterial()
    short = SignalPlan("short", up_segments=windows([(0.20, 0.40)]),
                       down_segments=windows([(0.40, 0.60)]))
    long_ = SignalPlan("long", up_segments=windows([(0.15, 0.29), (0.31, 0.42)]),
                       down_segments=windows([(0.43, 0.66), (0.68, 0.78)]))
    arterial.intersections["I2"].plans = [short, long_]
    return arterial


OBJECTIVE = ObjectiveConfig(sum_groups=[SumGroup({"up.global": 1.0, "down.global": 1.0})])


class DirectValidationTests(unittest.TestCase):
    def make_context(self):
        return TermValidationContext(
            intersection_names=["I1", "I2", "I3"],
            segment_names=["S12", "S23"],
            active_directions={"up", "down"},
            active_segment_numbers={"up": {1, 2}, "down": {1, 2}},
            balance_group_count=1,
            endpoint_terms_by_intersection={
                "I1": {"up.1.start", "up.1.end", "down.1.start", "down.1.end"},
                "I2": {"up.1.start", "up.1.end", "down.1.start", "down.1.end"},
                "I3": {"up.1.start", "up.1.end", "down.1.start", "down.1.end"},
            },
            plan_names_by_intersection={"I1": {"p1"}, "I2": {"p2"}, "I3": {"p3"}},
        )

    def assert_term_error(self, term):
        with self.assertRaises(TermValidationError):
            self.make_context().validate_specs(
                [(LinearSpec(terms={term: 1.0}, sense=">=", rhs=0.0), "intersection")]
            )

    def test_valid_endpoint_and_specials(self):
        ctx = self.make_context()
        for term in ("I1.up.1.start", "I3.down.1.end", "b_up", "b_down",
                     "B_bal", "tU_I2", "tD_I3", "bU_S12", "bD_S23"):
            with self.subTest(term=term):
                ctx.parse_term(term)

    def test_missing_endpoint_segment(self):
        self.assert_term_error("I1.up.3.start")

    def test_unknown_intersection(self):
        self.assert_term_error("I9.up.1.start")

    def test_typo_endpoint(self):
        self.assert_term_error("I1.up.1.foo")

    def test_typo_direction(self):
        self.assert_term_error("I1.xx.1.start")

    def test_unknown_special(self):
        self.assert_term_error("bQ_up")

    def test_missing_prefix(self):
        self.assert_term_error("up.1.start")

    def test_partial_invalid_terms_raise(self):
        ctx = self.make_context()
        with self.assertRaises(TermValidationError):
            ctx.validate_specs([(
                LinearSpec(terms={"I1.up.1.start": 1.0, "I9.up.1.start": -1.0},
                           sense="<=", rhs=-1.0),
                "intersection",
            )])

    def test_plan_tags_unknown_plan(self):
        ctx = self.make_context()
        with self.assertRaises(TermValidationError):
            ctx.validate_specs([(
                LinearSpec(terms={"I1.up.1.start": 1.0}, sense=">=", rhs=0.0,
                           plan_tags={"I1": "does-not-exist"}),
                "intersection",
            )])

    def test_common_endpoint_terms(self):
        arterial = multi_plan_arterial()
        common = common_endpoint_terms(arterial.intersections["I2"].plans)
        self.assertIn("up.1.start", common)
        self.assertNotIn("up.2.start", common)


class SolverValidationIntegrationTests(unittest.TestCase):
    def test_stage1_invalid_endpoint_raises_clear_error(self):
        with self.assertRaises(TermValidationError):
            SegmentedBandSolver(config=OBJECTIVE).solve(
                one_plan_arterial(),
                constraint_builder=ConstraintBuilder([
                    LinearSpec(terms={"I1.up.3.start": 1.0}, sense=">=", rhs=0.0)
                ]),
            )

    def test_stage1_valid_cross_intersection_constraint_still_solves(self):
        solution = SegmentedBandSolver(config=OBJECTIVE).solve(
            one_plan_arterial(),
            constraint_builder=ConstraintBuilder([
                LinearSpec(terms={"I1.up.1.start": 1.0, "I2.up.1.start": -1.0},
                           sense="<=", rhs=1.0)
            ]),
        )
        self.assertEqual(solution.status, "optimal")

    def test_stage2_invalid_endpoint_raises_clear_error(self):
        prior = SegmentedBandSolver(config=OBJECTIVE).solve(one_plan_arterial())
        with self.assertRaises(TermValidationError):
            FullFlexiblePhaseTuneSolver(
                config=OBJECTIVE, max_loops=3
            ).solve(
                one_plan_arterial(),
                prior=prior,
                loss_builder=SegmentLossBuilder([
                    SegmentLossSpec(terms={"I1.up.3.start": 1.0},
                                    lower_threshold=0.9, lower_slope=1.0)
                ]),
            )

    def test_stage1_partial_plan_endpoint_raises(self):
        # I2 的两个候选方案中，只有 long 有 up.2.start
        with self.assertRaises(TermValidationError):
            SegmentedBandSolver(config=OBJECTIVE).solve(
                multi_plan_arterial(),
                constraint_builder=ConstraintBuilder([
                    LinearSpec(terms={"I2.up.2.start": 1.0}, sense=">=", rhs=0.0)
                ]),
            )


if __name__ == "__main__":
    unittest.main()
