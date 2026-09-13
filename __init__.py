"""greenwave: 干线绿波协调优化（建模骨架）。"""

from .models import (Arterial, GreenWindow, Intersection, Segment,
                     SignalConstraint, SignalPlan)
from .plotting import plot_pareto_frontier, plot_time_space
from .solution import Solution

__all__ = [
    "Arterial",
    "GreenWindow",
    "Intersection",
    "Segment",
    "SignalConstraint",
    "SignalPlan",
    "Solution",
    "plot_pareto_frontier",
    "plot_time_space",
]

__version__ = "0.1.0"
