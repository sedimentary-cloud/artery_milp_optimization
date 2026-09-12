"""两阶段求解的统一配置对象。

这个文件把之前散落在 TwoStageSolver / CompositeBandSolver /
OneWayPrioritySolver / PhaseTuneSolver 参数里的配置收敛到 dataclass：

    TwoStageConfig
    ├── BandObjectiveConfig
    │   ├── mode
    │   ├── objective: ObjectiveConfig
    │   ├── alignment_builder
    │   └── band_loss_weight
    └── IntersectionLossConfig
        ├── loss_builder
        └── constraint_builder

其中：
    band_score = band_objective - band_loss_weight * band_loss
    intersection_loss = phase hinge + 软 LinearSpec slack
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .objective_config import ObjectiveConfig
from .phase import (AlignmentLossBuilder, ConstraintBuilder,
                    PhaseLossBuilder)


@dataclass
class BandObjectiveConfig:
    """绿波带层目标配置。Stage 1 和 Stage 2 共用。

    Attributes:
        mode: "global" 或 "oneway"。
        objective: ObjectiveConfig，定义 SumGroup / BalanceGroup。
        alignment_builder: 绿波带对齐损失。
        band_loss_weight: alignment loss 的权重 λ。
    """

    mode: str = "global"
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    alignment_builder: AlignmentLossBuilder | None = None
    band_loss_weight: float = 0.0

    def validate(self) -> None:
        if self.mode not in ("global", "oneway"):
            raise ValueError(f"unknown mode: {self.mode}")
        if self.band_loss_weight < 0:
            raise ValueError("band_loss_weight 不能为负")


@dataclass
class IntersectionLossConfig:
    """交叉口/相位层损失配置。只在 Stage 2 使用。

    Attributes:
        loss_builder: PhaseLossBuilder，相位 hinge loss。
        constraint_builder: ConstraintBuilder，硬/软 LinearSpec。
    """

    loss_builder: PhaseLossBuilder | None = None
    constraint_builder: ConstraintBuilder | None = None


@dataclass
class TwoStageConfig:
    """完整两阶段配置。

    Attributes:
        band: 绿波带层目标配置。
        intersection: 交叉口层损失配置。
        max_loops: mU/mD 整数圈数上界。
    """

    band: BandObjectiveConfig
    intersection: IntersectionLossConfig = field(
        default_factory=IntersectionLossConfig
    )
    max_loops: int = 3

    def validate(self) -> None:
        self.band.validate()
        if self.max_loops < 0:
            raise ValueError("max_loops 不能为负")


__all__ = [
    "BandObjectiveConfig",
    "IntersectionLossConfig",
    "TwoStageConfig",
]
