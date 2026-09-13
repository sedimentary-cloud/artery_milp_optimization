"""段级表达式、约束与损失构建器。"""

from .signal_constraints import (ConstraintBuilder, LinearSpec,
                                 SegmentLossBuilder, SegmentLossSpec,
                                 WindowExpr, window_exprs)
from .term_validation import (EndpointRef, SpecialRef, TermValidationContext,
                              TermValidationError, common_endpoint_terms,
                              union_endpoint_terms)

__all__ = [
    "ConstraintBuilder",
    "LinearSpec",
    "SegmentLossBuilder",
    "SegmentLossSpec",
    "WindowExpr",
    "window_exprs",
    "EndpointRef",
    "SpecialRef",
    "TermValidationContext",
    "TermValidationError",
    "common_endpoint_terms",
    "union_endpoint_terms",
]
