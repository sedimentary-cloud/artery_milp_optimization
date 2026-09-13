"""求解器公共入口。

推荐直接使用新分层模块：
- `solvers.core`
- `solvers.builders`
- `solvers.stage1`
- `solvers.stage2`
- `solvers.pipeline`

"""

from .builders import (ConstraintBuilder, LinearSpec,
                       SegmentLossBuilder, SegmentLossSpec,
                       TermValidationContext, TermValidationError,
                       WindowExpr, window_exprs)
from .core import (BalanceGroup, BandKey, BandMarginConfig,
                   ObjectiveConfig, Solver, SumGroup,
                   build_objective_config, composite_config,
                   oneway_config, parse_band_key)
from .pipeline import (BandObjectiveConfig, EpsilonConstraintRunner,
                       IntersectionLossConfig, TwoStageConfig, TwoStageSolver)
from .stage1 import SegmentedBandSolver
from .stage2 import FullFlexiblePhaseTuneSolver

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
    "window_exprs",
    "SegmentLossBuilder",
    "SegmentLossSpec",
    "ConstraintBuilder",
    "LinearSpec",
    "SegmentedBandSolver",
    "composite_config",
    "oneway_config",
    "build_objective_config",
    "BandMarginConfig",
    "FullFlexiblePhaseTuneSolver",
    "BandObjectiveConfig",
    "IntersectionLossConfig",
    "TwoStageConfig",
    "TwoStageSolver",
    "EpsilonConstraintRunner",
]
