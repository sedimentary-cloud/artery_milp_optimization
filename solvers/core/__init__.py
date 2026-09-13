"""求解器核心构件。"""

from .base import Solver, compute_effective_max_loops
from .margin import BandMarginConfig
from .band_lattice import (fill_solution_multi_window_bands,
                           fill_solution_window_band_ranges)
from .objective import (BalanceGroup, BandKey, ObjectiveConfig, SumGroup,
                        build_objective_config, composite_config,
                        oneway_config, parse_band_key)

__all__ = [
    "Solver",
    "compute_effective_max_loops",
    "fill_solution_multi_window_bands",
    "fill_solution_window_band_ranges",
    "BandKey",
    "SumGroup",
    "BalanceGroup",
    "ObjectiveConfig",
    "parse_band_key",
    "BandMarginConfig",
    "composite_config",
    "oneway_config",
    "build_objective_config",
]
