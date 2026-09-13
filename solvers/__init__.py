"""求解器公共入口。

推荐直接使用新分层模块：
- `solvers.core`
- `solvers.builders`
- `solvers.stage1`
- `solvers.stage2`
- `solvers.pipeline`

"""

from .builders import (AlignmentLossBuilder, ConstraintBuilder,
                       LinearSpec, SegmentLossBuilder, SegmentLossSpec,
                       TermValidationContext, TermValidationError,
                       WindowExpr, direction_phase_name,
                       phase_start_times, window_exprs)
from .core import (BalanceGroup, BandKey, ObjectiveConfig, Solver,
                   SumGroup, composite_config, oneway_config,
                   parse_band_key)
from .pipeline import (BandObjectiveConfig, EpsilonConstraintRunner,
                       IntersectionLossConfig, TwoStageConfig, TwoStageSolver)
from .stage1 import SegmentedBandObjective, SegmentedBandSolver
from .stage2 import (FlexiblePhaseTuneSolver, FullFlexiblePhaseTuneSolver,
                     PhaseTuneSolver)

__all__ = [
    "Solver",
    "TermValidationContext",
    "TermValidationError",
    "BandKey",
    "SumGroup",
    "BalanceGroup",
    "ObjectiveConfig",
    "parse_band_key",
    "WindowExpr",
    "direction_phase_name",
    "phase_start_times",
    "window_exprs",
    "SegmentLossBuilder",
    "SegmentLossSpec",
    "ConstraintBuilder",
    "LinearSpec",
    "AlignmentLossBuilder",
    "SegmentedBandObjective",
    "SegmentedBandSolver",
    "composite_config",
    "oneway_config",
    "FullFlexiblePhaseTuneSolver",
    "FlexiblePhaseTuneSolver",
    "PhaseTuneSolver",
    "BandObjectiveConfig",
    "IntersectionLossConfig",
    "TwoStageConfig",
    "TwoStageSolver",
    "EpsilonConstraintRunner",
]
