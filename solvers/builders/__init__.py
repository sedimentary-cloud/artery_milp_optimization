"""段级表达式、约束与损失构建器。"""

from .signal_constraints import (AlignmentLossBuilder, ConstraintBuilder,
                                 LinearSpec, PhaseLossBuilder,
                                 SegmentLossBuilder, SegmentLossSpec,
                                 WindowExpr, direction_phase_name,
                                 phase_start_times, window_exprs)
from .term_validation import (EndpointRef, SpecialRef, TermValidationContext,
                              TermValidationError, common_endpoint_terms,
                              union_endpoint_terms)

__all__ = [
    "AlignmentLossBuilder",
    "ConstraintBuilder",
    "LinearSpec",
    "PhaseLossBuilder",
    "SegmentLossBuilder",
    "SegmentLossSpec",
    "WindowExpr",
    "direction_phase_name",
    "phase_start_times",
    "window_exprs",
    "EndpointRef",
    "SpecialRef",
    "TermValidationContext",
    "TermValidationError",
    "common_endpoint_terms",
    "union_endpoint_terms",
]
