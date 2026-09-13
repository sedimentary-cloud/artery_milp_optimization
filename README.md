# artery_milp

`artery_milp` 是一个**干线绿波协调优化原型**。

它的核心建模思想是：不再用“相位语义”描述问题，而是把每个路口拆成 `up` / `down` 两个方向，每个方向在一个周期内可以有多段绿灯窗口。求解器只关心这些窗口，以及绿波带如何穿过它们。

技术栈：

- MILP 求解：`scipy.optimize.milp`（内部使用 HiGHS）；
- 数据模型：标准库 `dataclasses`；
- 可视化：`matplotlib`。

本文按“用户实际使用这个项目的顺序”组织：

1. 定义路口、路段、干线；
2. 定义单个交叉口的信号、约束与损失；
3. 定义绿波带优化目标与带层损失；
4. 代码如何自动检查输入配置；
5. 两阶段 solver 分别做什么；
6. 输出格式；
7. 案例。

---

## 0. 环境、目录与导入

### 0.1 运行环境

依赖见 `requirements.txt`：

```text
numpy>=1.20.0
scipy>=1.9.0       # 必须包含 milp 求解器
matplotlib>=3.5.0  # 时空图与 Pareto 前沿
```

仓库内自带一个可用的 conda 环境：

```bash
./conda-envs/artery_milp/bin/python --version
# Python 3.11.x
```

推荐直接用这个解释器运行示例和测试。

### 0.2 目录结构

```text
artery_milp/
  README.md
  requirements.txt
  __init__.py
  models.py                      # 输入数据模型
  solution.py                    # 统一输出结构 Solution
  plotting.py                    # 时空图、Pareto 前沿
  examples/
    window_band_ranges_case.py   # 可运行示例
  tests/
    test_term_validation.py      # 输入校验回归测试
  solvers/
    core/                        # 目标 DSL、窗口带格、抽象基类
    builders/                    # 段级表达式、约束/损失构建器、term 校验
    stage1/                      # 第一阶段：方案选择 + 带宽优化
    stage2/                      # 第二阶段：段端点微调 + 约束/损失
    pipeline/                    # 两阶段编排、TwoStageConfig、Pareto 扫描
```

### 0.3 推荐导入方式

```python
# 数据模型
from artery_milp.models import (
    Arterial, GreenWindow, Intersection, Segment, SignalPlan,
    SignalConstraint, SignalLoss,
)

# 目标 DSL
from artery_milp.solvers.core import (
    ObjectiveConfig, SumGroup, BalanceGroup,
    build_objective_config, composite_config, oneway_config,
)

# 输入校验错误
from artery_milp.solvers import TermValidationError

# 第一阶段
from artery_milp.solvers.stage1 import SegmentedBandSolver

# 第二阶段
from artery_milp.solvers.stage2 import FullFlexiblePhaseTuneSolver

# 两阶段编排 + 配置
from artery_milp.solvers.pipeline import (
    TwoStageConfig, BandObjectiveConfig, IntersectionLossConfig,
    TwoStageSolver, EpsilonConstraintRunner,
)

# 高级：跨路口约束 / 带层损失
from artery_milp.solvers.builders import (
    ConstraintBuilder, LinearSpec,
    SegmentLossBuilder, SegmentLossSpec,
)
```

---

## 1. 定义路口、路段与干线

### 1.1 最小骨架

```python
from artery_milp.models import Arterial, GreenWindow, Intersection, Segment, SignalPlan

CYCLE = 90.0


def w(*pairs):
    """把 (start, end) 比例对转成 GreenWindow 列表。"""
    return [GreenWindow(start, end) for start, end in pairs]


i1 = Intersection("I1", [SignalPlan(
    name="base",
    up_segments=w((0.05, 0.24)),
    down_segments=w((0.58, 0.76)),
)])

i2 = Intersection("I2", [SignalPlan(
    name="base",
    up_segments=w((0.17, 0.36)),
    down_segments=w((0.45, 0.64)),
)])

i3 = Intersection("I3", [SignalPlan(
    name="base",
    up_segments=w((0.29, 0.48)),
    down_segments=w((0.33, 0.52)),
)])

arterial = Arterial(
    cycle=CYCLE,
    intersections={"I1": i1, "I2": i2, "I3": i3},
    segments={
        "S12": Segment("S12", length_up=190.0, length_down=188.0,
                       speed_up=14.0, speed_down=14.0),
        "S23": Segment("S23", length_up=205.0, length_down=200.0,
                       speed_up=14.0, speed_down=14.0),
    },
    order=["I1", "S12", "I2", "S23", "I3"],
)
```

### 1.2 路段 `Segment`

```python
Segment(
    name="S12",
    length_up=190.0,     # 上行长度（米）
    length_down=188.0,   # 下行长度（米）
    speed_up=14.0,       # 上行车速（米/秒）
    speed_down=14.0,     # 下行车速（米/秒）
)
```

`Segment` 会自动提供：

- `travel_time_up = length_up / speed_up`；
- `travel_time_down = length_down / speed_down`。

这两个行驶时间用于绿波带传播约束：

```text
t_{i+1} - t_i - cycle * m = travel_time
```

其中 `m` 是整数圈数。

### 1.3 路口 `Intersection` 与信控方案 `SignalPlan`

- `Intersection(name, plans)`：一个路口，持有若干候选方案 `SignalPlan`。
- `SignalPlan(name, up_segments, down_segments, ...)`：一个候选方案，定义该路口上下行的绿灯窗口，以及本方案内部的约束/软损失。

```python
SignalPlan(
    name="base",
    up_segments=w((0.05, 0.24)),     # 上行绿灯窗口（周期比例）
    down_segments=w((0.58, 0.76)),   # 下行绿灯窗口
    signal_constraints=[...],        # 可选：本方案内部硬约束
    signal_losses=[...],             # 可选：本方案内部软损失
    metadata={...},                  # 可选：第二阶段可调端点范围
)
```

一个方向可以有多段，例如：

```python
up_segments=w((0.15, 0.29), (0.31, 0.42))
down_segments=w((0.43, 0.66), (0.68, 0.78))
```

规则：

- 每段必须满足 `0 <= start < end <= 1`；
- 同一方向的多段必须按时间排序，且不能重叠；
- 段号从 1 开始，即 `up.1.start`、`up.2.end` 等。

### 1.4 干线 `Arterial`

```python
Arterial(
    cycle=90.0,
    intersections={...},   # 路口名 -> Intersection
    segments={...},        # 路段名 -> Segment
    order=["I1", "S12", "I2", "S23", "I3"],
)
```

`order` 是路口与路段交替出现的名称列表：

- 长度必须是奇数；
- 必须以路口开始、以路口结束；
- 相邻两个路口之间恰好一个路段。

即 `order = [I1, S12, I2, S23, I3, ...]`。

### 1.5 时间、方向与 term 命名约定

- **所有对外时间比例量都用“占周期比例”**：`0.10` 表示周期的 10%，不是 10 秒。
- 求解器内部统一乘以 `cycle` 转成秒域。
- `up` = 上行，`down` = 下行。
- `Segment.travel_time_*` 是秒。

term 两类写法：

| 场景 | 写法 | 例子 |
| :--- | :--- | :--- |
| `SignalConstraint` / `SignalLoss`（路口内部） | 局部写法 | `up.1.start`、`down.2.end` |
| `LinearSpec` / `SegmentLossSpec`（跨路口/外部） | 完整写法 | `I1.up.1.start`、`I3.down.2.end` |

局部写法只在 `SignalPlan` 内部有意义；跨路口的构建器必须写全路口名。

---

## 2. 定义单个交叉口的信号、约束与损失

### 2.0 先理清关系：Intersection、SignalPlan、SignalConstraint、SignalLoss

第 1 章定义了干线的骨架：`Intersection` 和 `Segment` 交替挂在 `Arterial.order` 上。进入第 2 章之前，先把“单个交叉口”这一层的对象关系理顺。

一个路口不是只有一套信号配时，而是可以持有多个候选方案；每个候选方案又包含“绿灯窗口”和“方案内部的约束/损失”两部分。层次关系如下：

```text
Arterial
└── intersections
    └── Intersection "I2"                     # 一个路口
        └── plans
            ├── SignalPlan "base"              # 候选方案 A
            │   ├── up_segments:   [GreenWindow, GreenWindow, ...]    # 上行绿灯窗口
            │   ├── down_segments: [GreenWindow, GreenWindow, ...]    # 下行绿灯窗口
            │   ├── signal_constraints: [SignalConstraint, ...]       # 方案内部硬约束
            │   ├── signal_losses:      [SignalLoss, ...]             # 方案内部软损失
            │   └── metadata["term_bounds"]                           # Stage 2 可调范围
            └── SignalPlan "split"             # 候选方案 B
                └── ...
```

用一句话概括各自角色：

| 对象 | 角色 |
| :--- | :--- |
| `Intersection` | 干线上的一个节点，持有若干候选 `SignalPlan` |
| `SignalPlan` | 该路口的一种候选信控方案，核心是上下行两个 `GreenWindow` 列表 |
| `SignalConstraint` | 挂在某个 `SignalPlan` 上的硬约束，描述该方案内部的段相对关系；构造方案时先校验名义值，Stage 2 再进入 MILP |
| `SignalLoss` | 挂在某个 `SignalPlan` 上的软损失，描述该方案内部的“期望区间”；只进入 Stage 2 |

它们之间的关系可以记成：

```text
Intersection  1 ── *  SignalPlan
SignalPlan    1 ── *  GreenWindow        （up_segments / down_segments）
SignalPlan    1 ── *  SignalConstraint   （硬约束）
SignalPlan    1 ── *  SignalLoss         （软损失）
```

`SignalConstraint` 和 `SignalLoss` 里的 `terms` 并不是随便写的字符串，而是在指向“某个 `SignalPlan` 里的某一段绿灯窗口的起点或终点”。要读懂这些字符串，只需要理解下面这个格式：

```text
{direction}.{segment_no}.{endpoint}
```

| 部分 | 取值 | 含义 |
| :--- | :--- | :--- |
| `direction` | `up` / `down` | 上行 / 下行 |
| `segment_no` | 从 1 开始的整数 | 该方向 `up_segments` / `down_segments` 列表里的第几段 |
| `endpoint` | `start` / `end` | 这一段的起点 / 终点 |

换句话说，`up.2.start` 就是“这个 `SignalPlan` 的 `up_segments[1].start`”；它描述的是上行第 2 段绿灯窗口的起点，而不是“第 2 个路口”或“第 2 个相位”。

下面用一个两段绿灯窗口的例子把对应关系展开：

```python
SignalPlan(
    name="split",
    up_segments=[
        GreenWindow(0.15, 0.29),   # up.1
        GreenWindow(0.31, 0.42),   # up.2
    ],
    down_segments=[
        GreenWindow(0.43, 0.66),   # down.1
        GreenWindow(0.68, 0.78),   # down.2
    ],
)
```

对应的 term 就是：

| term | 含义 | 当前值（周期比例） |
| :--- | :--- | :--- |
| `up.1.start` | 上行第 1 段起点 | `0.15` |
| `up.1.end` | 上行第 1 段终点 | `0.29` |
| `up.2.start` | 上行第 2 段起点 | `0.31` |
| `up.2.end` | 上行第 2 段终点 | `0.42` |
| `down.1.start` | 下行第 1 段起点 | `0.43` |
| `down.1.end` | 下行第 1 段终点 | `0.66` |
| `down.2.start` | 下行第 2 段起点 | `0.68` |
| `down.2.end` | 下行第 2 段终点 | `0.78` |

所以：

```text
up.2.start
= 上行第 2 段绿灯窗口的起点
= up_segments[1].start
= 0.31（周期比例）
```

几个容易混淆的点：

- **`up.2.start` 不是“第 2 个路口”**，而是“当前这个方案的 `up_segments` 列表里的第 2 段”。路口由所在的 `SignalPlan` 决定；在 `SignalConstraint` / `SignalLoss` 里不需要写路口名。
- **段号从 1 开始**，对应 Python 列表下标 `segment_no - 1`。
- **值都是周期比例**，不是秒。例如 `up.2.start = 0.31` 表示周期的 31%；`cycle=90` 时，求解器内部会换算成 `0.31 * 90 = 27.9` 秒。
- **`rhs` / 阈值也是周期比例**。例如 `rhs=0.02` 表示 0.02 个周期，`cycle=90` 时等于 1.8 秒。
- **如果方案里没有这一段**，`up.3.start` 就是非法 term。`SignalPlan` 构造、Stage 1/Stage 2 的输入校验都会报错；不要靠“写一个不存在的段号”来表达禁用。
- **跨路口的外部约束要写完整 term**：`SignalConstraint` 用 `up.1.start`，外部 `LinearSpec` 必须写成 `I1.up.1.start`（路口名.方向.段号.端点）。

把语法放回两个实际场景里，读起来就直观了。

第一个是 `SignalConstraint`：

```python
SignalConstraint(
    terms={"up.2.start": 1.0, "up.1.end": -1.0},
    sense=">=",
    rhs=0.02,
)
```

读作：

```text
1.0 * up.2.start - 1.0 * up.1.end >= 0.02
= 上行第 2 段起点 - 上行第 1 段终点 >= 0.02 个周期
= 两段之间至少间隔 0.02 个周期
```

第二个是 `SignalLoss`：

```python
SignalLoss(
    terms={"down.1.start": 1.0},
    lower_threshold=0.42,
    upper_threshold=0.46,
)
```

表示“希望下行第 1 段起点落在 0.42 ~ 0.46 个周期之间”。

理清这层关系后，下面就可以分别看 `SignalPlan` 里的三个部分：绿灯窗口、硬约束、软损失。

### 2.1 绿灯窗口 `GreenWindow`

```python
GreenWindow(start=0.05, end=0.24)
```

- `start`、`end` 都是周期比例；
- 必须满足 `0 <= start < end <= 1`；
- `width = end - start`。

一个路口的一个方向可以有多段：

```python
SignalPlan(
    name="split",
    up_segments=[GreenWindow(0.15, 0.29), GreenWindow(0.31, 0.42)],
    down_segments=[GreenWindow(0.43, 0.66), GreenWindow(0.68, 0.78)],
)
```

### 2.2 路口内部硬约束 `SignalConstraint`

用于描述**同一个路口、同一个方案内部**的段/相位相对关系（段顺序、最小间隔等）。

```python
from artery_milp.models import SignalConstraint

SignalConstraint(
    terms={"up.2.start": 1.0, "up.1.end": -1.0},
    sense=">=",
    rhs=0.02,
    name="上行第 2 段必须晚于第 1 段结束至少 0.02 周期",
)
```

含义：

```text
1.0 * up.2.start - 1.0 * up.1.end >= 0.02
```

要点：

- `terms` 使用**局部 term**：`up.1.start`、`down.2.end`，不带路口名前缀；term 语法详见 2.0；
- `rhs` 使用**周期比例**；
- `sense` 可以是 `"<="`、`">="`、`"="`；
- **会在 `SignalPlan` 构造时按当前窗口值校验**：如果当前窗口已经违反约束，构造 `SignalPlan` 时直接抛 `ValueError`；
- **当前实现中，Stage 1 不会把 `SignalConstraint` 作为优化约束**，它只负责方案选择 + 带宽优化；`SignalConstraint` 会在 Stage 2 随选中方案加载并进入 MILP。若 Stage 2 在 `term_bounds` 内无法满足它，该点会 infeasible，`TwoStageSolver` 会回退到 Stage 1。

### 2.3 路口内部软损失 `SignalLoss`

用于描述“希望某个表达式尽量落在某个区间内”，偏离时产生惩罚。

```python
from artery_milp.models import SignalLoss

SignalLoss(
    terms={"down.1.start": 1.0},
    lower_threshold=0.42,
    upper_threshold=0.46,
    lower_slope=1.5,
    upper_slope=1.5,
    name="下行第 1 段起点尽量落在 0.42~0.46",
)
```

数学含义：

```text
损失 = lower_slope * max(0, lower_threshold - expr)
     + upper_slope * max(0, expr - upper_threshold)
```

要点：

- `lower_threshold / upper_threshold` 都是周期比例；
- 可以只写一侧：只给 `lower_threshold` 或只给 `upper_threshold`；
- `terms` 同样使用局部写法，语法见 2.0；
- 会随 `SignalPlan` 一起加载，只在 Stage 2 进入优化。

### 2.4 第二阶段可调端点 `metadata["term_bounds"]`

Stage 1 的绿灯窗口是固定的；Stage 2 可以在给定范围内微调段端点。

```python
SignalPlan(
    name="base",
    up_segments=w((0.17, 0.36)),
    down_segments=w((0.45, 0.64)),
    metadata={
        "term_bounds": {
            "up.1.start":   (0.15, 0.19),
            "up.1.end":     (0.34, 0.38),
            "down.1.start": (0.43, 0.47),
            "down.1.end":   (0.62, 0.66),
        }
    },
)
```

要点：

- 上下界都是周期比例；
- 没写进 `term_bounds` 的端点默认**固定为当前值**；
- `term_bounds` 只在 Stage 2 生效；Stage 1 始终使用固定窗口；
- 完整运行效果可参考 7.2 节的 `examples/window_band_ranges_case.py`，它会在 Stage 2 打印“名义值 -> 调整后值”的对比。

### 2.5 一个路口多个候选方案

`Intersection.plans` 可以放多个 `SignalPlan`：

```python
i2 = Intersection("I2", [
    SignalPlan("base",  up_segments=w((0.17, 0.36)), down_segments=w((0.45, 0.64))),
    SignalPlan("split",
               up_segments=w((0.15, 0.29), (0.31, 0.42)),
               down_segments=w((0.43, 0.66), (0.68, 0.78)),
               signal_constraints=[...],
               signal_losses=[...]),
])
```

- Stage 1：通过二元变量自动选择方案；
- Stage 2：沿用 Stage 1 选中的方案，只调该方案的段端点。

注意：Stage 1 只对“所有候选方案共同拥有”的段号建模。例如上一个路口 `base` 只有 1 段、`split` 有 2 段，那么 Stage 1 只会对第 1 段做多段传播；第 2 段的约束/损失要到 Stage 2 选中 `split` 后才生效。

### 2.6 选型建议：路口内部 vs 跨路口

| 场景 | 推荐工具 | 绑定位置 |
| :--- | :--- | :--- |
| 单个路口内部的段顺序、最小间隔 | `SignalConstraint` | `SignalPlan.signal_constraints` |
| 单个路口内部的期望区间/软目标 | `SignalLoss` | `SignalPlan.signal_losses` |
| 跨路口协调、带宽层规则、临时实验规则 | `LinearSpec` / `SegmentLossSpec` | `ConstraintBuilder` / `SegmentLossBuilder` |

**不要把单个路口内部的规则外置成 `LinearSpec`。** 原因：

1. 不同候选方案的段数/端点可能不同，外置后很难随方案自动切换；
2. `SignalConstraint` / `SignalLoss` 随方案自动加载，Stage 2 会直接按选中方案施加；而外部 `LinearSpec` 需要自己处理多方案差异；
3. 外部 term 如果引用了某个候选方案不存在的端点，Stage 1 会在统一校验阶段直接报错；
4. Stage 1 本身不建模 `SignalConstraint` / `SignalLoss`，它们的实际效果发生在 Stage 2；把路口内部规则放在方案里语义最清晰。

跨路口的 `LinearSpec` / `SegmentLossSpec` 用法见第 3.6 节。

---

## 3. 定义绿波带优化目标与带层损失

> 说明：早期的 `AlignmentLossBuilder`（带中心绝对时间对齐）已移除。
> 新语义（绿波带边缘到绿灯区间边缘的距离控制）计划按“Stage 1 硬边距 + Stage 2 软边距”重新设计；当前版本暂不提供该功能。

### 3.1 带标识 `BandKey` 语法

绿波带按“方向 + 窗口大小 + 起点”描述：

| 写法 | 含义 | 窗口大小 |
| :--- | :--- | :--- |
| `up.global` | 上行整条干线全局带 | `k = n` |
| `down.global` | 下行整条干线全局带 | `k = n` |
| `up.seg1` | 上行第 1 条物理路段（`I1-I2`） | `k = 2` |
| `down.seg2` | 下行第 2 条物理路段 | `k = 2` |
| `up.win3@I1-I3` | 上行从 `I1` 到 `I3` 的三路口局部带 | `k = 3` |
| `down.win4@I2-I5` | 下行从 `I2` 到 `I5` 的四路口局部带 | `k = 4` |

重要语义：

- `*.global`：整条干线共享同一条传播轨迹；
- `*.segX` / `*.winK@...`：该子走廊**独立求解**，可以有自己的传播时刻和带宽；
- 例如 `up.win2@I3-I4`、`up.win3@I2-I4`、`up.win4@I1-I4` 各自独立，不共享主带轨迹。

### 3.2 加权和 `SumGroup`

```python
from artery_milp.solvers.core import ObjectiveConfig, SumGroup

cfg = ObjectiveConfig(
    sum_groups=[
        SumGroup({
            "up.global": 1.0,
            "down.global": 1.0,
        })
    ]
)
```

数学含义：

```text
max 1.0 * B_up_global + 1.0 * B_down_global
```

常见配法：

```python
# 上行优先
SumGroup({"up.global": 2.0, "down.global": 1.0})

# 强调瓶颈路段
SumGroup({"up.seg1": 1.0, "up.seg2": 1.0, "down.seg2": 2.0})

# 强调内部局部绿波
SumGroup({
    "up.global": 0.5,
    "down.win3@I2-I4": 1.5,
    "down.seg2": 1.0,
})
```

### 3.3 均衡组 `BalanceGroup`

`BalanceGroup` 的语义是“最大化组内最小值”：

```python
from artery_milp.solvers.core import BalanceGroup

BalanceGroup(["up.global", "down.global"], weight=1.0, eps=0.01)
```

数学含义：

```text
B_group <= up.global
B_group <= down.global
目标 += weight * B_group
目标 += weight * eps * (up.global + down.global)
```

- `weight`：均衡组在总目标中的权重，越大越优先补短板；
- `eps`：托底项权重，在均衡达标后继续推动组内总量；
- 组内成员的窗口大小 `k` 必须相同；
- 同一个带不能出现在多个 `BalanceGroup`。

### 3.4 复合目标 `ObjectiveConfig`

```python
from artery_milp.solvers.core import ObjectiveConfig, SumGroup, BalanceGroup

objective = ObjectiveConfig(
    sum_groups=[
        SumGroup({
            "up.global": 1.0,
            "down.global": 1.0,
        })
    ],
    balance_groups=[
        BalanceGroup(["up.global", "down.global"], weight=0.2, eps=0.01)
    ],
)
```

语义：

```text
max (up.global + down.global) + 0.2 * min(up.global, down.global)
    + 0.2 * 0.01 * (up.global + down.global)
```

常用模板：

```python
# A. 双向总带宽
ObjectiveConfig(sum_groups=[SumGroup({"up.global": 1.0, "down.global": 1.0})])

# B. 双向均衡
ObjectiveConfig(balance_groups=[
    BalanceGroup(["up.global", "down.global"], weight=1.0, eps=0.0)
])

# C. 总量 + 均衡托底
ObjectiveConfig(
    sum_groups=[SumGroup({"up.global": 1.0, "down.global": 1.0})],
    balance_groups=[BalanceGroup(["up.global", "down.global"], weight=0.1, eps=0.01)],
)

# D. 强调内部局部绿波
ObjectiveConfig(sum_groups=[SumGroup({
    "up.global": 0.4,
    "down.win3@I2-I4": 1.4,
    "down.seg2": 1.0,
})])
```

`ObjectiveConfig.validate(n_intersections)` 会检查：

- 同一个带不能在多个 `SumGroup` 中重复；
- `BalanceGroup` 不能为空；
- `BalanceGroup.weight > 0`；
- `BalanceGroup` 组内成员的 `k` 必须相同；
- 同一个带不能出现在多个 `BalanceGroup` 中。

### 3.5 高级：外部约束与软损失

以下工具适合跨路口、带宽层、临时策略，**不建议**用来描述单个路口内部关系。

#### `LinearSpec` + `ConstraintBuilder`

```python
from artery_milp.solvers.builders import ConstraintBuilder, LinearSpec

hard = ConstraintBuilder([
    LinearSpec(
        terms={"I1.up.1.start": 1.0, "I2.up.1.start": -1.0},
        sense="<=",
        rhs=8.0 / 90.0,
        soft=False,
        name="I1/I2 上行首段起点间隔",
    )
])
```

- term 必须写完整的 `{Int}.{dir}.{idx}.{start|end}`；
- `rhs` 是周期比例，内部乘 `cycle`；
- `soft=False` 是硬约束，`soft=True` 是软约束，`penalty` 控制惩罚权重。

#### `SegmentLossSpec` + `SegmentLossBuilder`

```python
from artery_milp.solvers.builders import SegmentLossBuilder, SegmentLossSpec

soft = SegmentLossBuilder([
    SegmentLossSpec(
        terms={"I2.down.1.start": 1.0},
        lower_threshold=0.42,
        upper_threshold=0.48,
        lower_slope=1.0,
        upper_slope=1.0,
        name="I2 下行首段起点期望区间",
        kind="intersection",   # 记入 intersection_loss
    )
])
```

- `lower_threshold / upper_threshold` 是周期比例；
- `kind="band"` 的损失记入 `band_loss`，`kind="intersection"` 记入 `intersection_loss`；
- 支持完整 term 和特殊变量 `b_up`、`b_down`、`tU_I2`、`tD_I2`、`bU_S12`、`bD_S12`、`B_bal`。

#### `plan_tags` 条件约束

```python
LinearSpec(
    terms={"I2.up.2.start": 1.0, "I2.up.1.end": -1.0},
    sense=">=",
    rhs=0.02,
    plan_tags={"I2": "split"},
)
```

- 语义：只有 `I2` 选中 `"split"` 方案时，这条约束才生效；
- `plan_tags` 引用的路口和方案必须存在；
- Stage 1 的 `plan_tags` 不能用来“规避某个候选方案缺少段端点”的情况，因为 Stage 1 的端点解析仍会遍历所有候选方案。此类问题请拆成不同求解调用，或写进方案内部。

---

## 4. 代码如何自动检查输入配置

这个项目在三个层次做校验。

### 4.1 模型构造期校验

发生在创建数据对象时，立即抛 `ValueError`：

| 对象 | 校验内容 |
| :--- | :--- |
| `GreenWindow` | `0 <= start < end <= 1` |
| `SignalPlan` | 同方向多段按时间排序、不重叠 |
| `SignalConstraint` | `sense` 合法；term 必须存在于本方案；当前窗口值必须满足该约束 |
| `SignalLoss` | 至少一个阈值；斜率非负；term 必须存在于本方案 |
| `Arterial` | `cycle > 0`；`order` 奇数长度、路口/路段交替且都已注册 |

### 4.2 目标配置校验 `ObjectiveConfig.validate(n)`

在 solver 内部建模前调用，检查：

- `SumGroup` 中是否有重复带；
- `BalanceGroup` 是否为空、权重是否为正；
- 组内成员窗口大小 `k` 是否一致；
- 同一个带是否被多个 `BalanceGroup` 引用。

### 4.3 统一 term 校验 `TermValidationContext`

这是最近新增的一层，专门解决“外部约束/损失 term 写错后静默丢项、静默截短、或抛 `IndexError`”的问题。

```python
from artery_milp.solvers.builders import (
    TermValidationContext, TermValidationError,
)
```

所有通过以下构建器传入的 `LinearSpec` / `SegmentLossSpec` 都会在组装 MILP 之前被校验：

- `ConstraintBuilder`
- `SegmentLossBuilder`

校验规则：

| 项目 | 规则 |
| :--- | :--- |
| 端点 term | 必须是完整 `{Int}.{dir}.{idx}.{start\|end}` |
| `dir` | 只能是 `up` / `down` |
| `endpoint` | 只能是 `start` / `end` |
| `idx` | 从 1 开始，且对应段必须存在 |
| Stage 1 端点可用性 | 必须被该路口的**所有候选方案**共同定义 |
| Stage 2 端点可用性 | 必须被**选中方案**定义 |
| `b_up` / `b_down` | 对应方向必须有全走廊带实例 |
| `tU_I2` / `tD_I2` | 路口必须存在，且对应方向有带实例 |
| `bU_S12` / `bD_S12` | 物理路段必须存在，且对应方向有带实例 |
| `B_bal` | 必须至少有一个 `BalanceGroup` |
| `plan_tags` | 引用的路口和方案名必须存在 |
| 部分解析 | 不允许；任一 term 非法，整条 spec 拒绝 |

校验失败会一次性汇总所有错误：

```text
TermValidationError: 约束/损失 term 校验失败:
- [intersection:spec#0('I1.up.3.start')] term 'I1.up.3.start' 非法:
  路口 I1 上不存在可用端点 'up.3.start'
  （Stage 1 要求所有候选方案共同定义；Stage 2 要求选中方案定义）。
  当前可用端点示例: ['down.1.end', 'down.1.start', 'up.1.end', 'up.1.start']
```

### 4.4 常见错误对照表

| 输入 | 行为 |
| :--- | :--- |
| `"I1.up.3.start"`（没有第 3 段） | `TermValidationError` |
| `"I9.up.1.start"`（路口不存在） | `TermValidationError` |
| `"up.1.start"`（外部 spec 忘写路口前缀） | `TermValidationError` |
| `"I1.up.1.foo"`（后缀拼错） | `TermValidationError` |
| `"I1.xx.1.start"`（方向拼错） | `TermValidationError` |
| `"bQ_up"`（未知特殊变量） | `TermValidationError` |
| `"bD_S99"`（路段不存在） | `TermValidationError` |
| `"B_bal"`（没有 `BalanceGroup`） | `TermValidationError` |
| `plan_tags={"I2": "不存在的方案"}` | `TermValidationError` |
| 部分 term 非法 | 整条 spec 拒绝，不截短 |

### 4.5 运行校验测试

```bash
cd /home/qktx/artery_milp
./conda-envs/artery_milp/bin/python -m unittest discover -s tests -v
```

---

## 5. 两阶段 solver

### 5.1 Stage 1：`SegmentedBandSolver`

Stage 1 做两件事：

1. 在路口多个候选方案中选择一个方案；
2. 在固定绿灯窗口下，优化各方向、各段号的绿波带宽度和传播轨迹。

```python
from artery_milp.solvers.stage1 import SegmentedBandSolver

solver = SegmentedBandSolver(
    config=objective,       # ObjectiveConfig
    max_segments=3,         # 最多建模到第几段
    max_loops=3,            # 整数圈数范围 [-max_loops, max_loops]
    up_global_output=False,
    down_global_output=False,
)
solution = solver.solve(
    arterial,
    band_loss_weight=0.0,       # 带层损失权重（kind="band"）
    constraint_builder=None,    # 可选，跨路口约束
    loss_builder=None,          # 可选，跨路口软损失
)
```

建模要点：

- 对每个路口候选方案引入二元变量，每组恰好选 1 个；
- 对每个“方向-段号”实例建立传播时刻 `t`、整数圈数 `m`、逐路段带宽 `width`、全走廊带宽 `global`；
- 只对“所有候选方案共同拥有”的段号建模；
- 目标由 `ObjectiveConfig` 决定；

`SegmentedBandSolver` 现在必须显式传入 `ObjectiveConfig`。如果只是想在旧的“权重”形式上快速构造目标，可以用：

```python
from artery_milp.solvers.core import build_objective_config
from artery_milp.solvers.stage1 import SegmentedBandSolver

config = build_objective_config(
    mode="global",
    up_weight=1.0,
    down_weight=1.0,
    objective_mode="balanced_composite",  # sum / balanced / balanced_composite
    balance_eps=0.20,
)
solver = SegmentedBandSolver(config=config)
```

注意：旧的 `SegmentedBandObjective` 已移除；`build_objective_config` 只负责把 mode/权重翻译成 `ObjectiveConfig`，不参与求解。

### 5.2 Stage 2：`FullFlexiblePhaseTuneSolver`

Stage 2 做三件事：

1. 锁定 Stage 1 选中的方案；
2. 在 `metadata["term_bounds"]` 范围内微调段端点；
3. 同时优化带宽、`SignalConstraint` / `SignalLoss`、外部 `LinearSpec` / `SegmentLossSpec`。

```python
from artery_milp.solvers.stage2 import FullFlexiblePhaseTuneSolver

tuner = FullFlexiblePhaseTuneSolver(
    config=objective,
    max_loops=3,
    up_global_output=True,
    down_global_output=False,
)
solution = tuner.solve(
    arterial,
    prior=stage1_solution,      # 必须提供 Stage 1 的解
    loss_builder=loss_builder,          # 可选，SegmentLossBuilder
    constraint_builder=constraint_builder,  # 可选，ConstraintBuilder
    band_loss_weight=0.1,                # 带层损失权重（kind="band"）
    objective="bandwidth",      # bandwidth / loss
    tunable_intersections=None, # None 表示所有路口都可调；否则只调集合内路口
)
```

要点：

- `prior.plan_choices` 决定每个路口使用哪个方案；
- `term_bounds` 决定哪些端点可调、范围多少；不在其中的端点固定；
- `FullFlexiblePhaseTuneSolver` 本身不再接收 `mode`；`mode` 只保留在 `TwoStageConfig.band.mode` 等上层配置，用于选择预设目标和输出口径；
- `up_global_output` / `down_global_output` 控制输出口径；
- `tunable_intersections` 可以进一步限制哪些路口允许调；
- `objective="bandwidth"` 主优化带宽；
- `objective="loss"` 主优化交叉口损失（用于 ε-约束扫描）；
- `max_intersection_loss` 可以给交叉口损失加上界。

如果你只有 mode/权重而没有现成的 `ObjectiveConfig`，可以先在外部构造：

```python
from artery_milp.solvers.core import build_objective_config

objective = build_objective_config(
    mode="global",          # global / oneway
    up_weight=1.0,
    down_weight=1.0,
    objective_mode="sum",
)
```

然后把 `objective` 传给 `FullFlexiblePhaseTuneSolver`。

### 5.3 两阶段编排：`TwoStageSolver`

推荐直接用 `TwoStageSolver`，它把 Stage 1 和 Stage 2 串起来，并在 Stage 2 失败时回退到 Stage 1。

```python
from artery_milp.solvers.pipeline import (
    TwoStageConfig, BandObjectiveConfig, IntersectionLossConfig, TwoStageSolver,
)

config = TwoStageConfig(
    band=BandObjectiveConfig(
        mode="global",
        objective=objective,
        band_loss_weight=0.1,
    ),
    intersection=IntersectionLossConfig(
        loss_builder=loss_builder,
        constraint_builder=constraint_builder,
        tunable_intersections=None,
    ),
    max_loops=3,
)

solution = TwoStageSolver(config=config).solve(arterial)
```

流程：

```text
TwoStageSolver
  ├─ Stage 1: SegmentedBandSolver
  │    输出 s1（方案选择 + 固定窗口带宽）
  └─ Stage 2: FullFlexiblePhaseTuneSolver
       输入 s1 作为 prior
       在 term_bounds 内调端点
       加载 SignalConstraint / SignalLoss
       加载外部 ConstraintBuilder / SegmentLossBuilder
       成功 -> 返回 s2
       失败 -> 返回 s1，status 追加 "|stage1_fallback"
```

### 5.4 Pareto 扫描：`EpsilonConstraintRunner`

用于扫描“交叉口损失 vs 绿波带目标”的前沿。

```python
from artery_milp.solvers.pipeline import EpsilonConstraintRunner

runner = EpsilonConstraintRunner(
    config=config,
    n_points=5,
    metric="band_score",   # band_score / sum / balanced / objective
)

prior = TwoStageSolver(config=config).solve(arterial)
frontier = runner.run(arterial, prior)
knee = runner.knee_point()
```

`frontier` 中每个元素是：

```text
(eps, corridor_metric, intersection_loss, solution)
```

可用 `plotting.plot_pareto_frontier(frontier, knee_point=knee)` 绘图。

### 5.5 solver 入口速查

| 目标 | 入口 |
| :--- | :--- |
| 只做 Stage 1 | `SegmentedBandSolver` |
| 只做 Stage 2（已有 prior） | `FullFlexiblePhaseTuneSolver` |
| 完整两阶段 | `TwoStageSolver(config=TwoStageConfig(...))` |
| 带宽-损失 Pareto | `EpsilonConstraintRunner` |

---

## 6. 输出格式 `Solution`

所有 solver 统一返回 `artery_milp.solution.Solution`。

### 6.1 字段总览

```python
@dataclass
class Solution:
    cycle: float

    # 选择与时序
    plan_choices: dict[str, str]
    window_choices: dict[str, dict[str, int | str]]
    segment_times: dict[str, dict[str, float]]

    # 带宽结果
    bandwidth_up: dict[str, float]
    bandwidth_down: dict[str, float]
    band_up_style: str
    band_down_style: str
    band_start_up: dict[str, float]
    band_start_down: dict[str, float]

    # 多段/窗口带
    multi_bandwidths: dict[str, dict[int, float]]
    multi_band_starts: dict[str, dict[int, dict[str, float]]]
    multi_window_bands: dict[str, dict[int, dict[str, float]]]
    window_bands: dict[str, float]
    window_band_ranges: dict[str, list[dict[str, object]]]

    # 目标与损失
    band_objective: float
    band_loss: float
    band_score: float
    intersection_loss: float
    objective: float          # solver 原始目标
    status: str
    solver_msg: str
```

### 6.2 选择与时序

| 字段 | 读法 |
| :--- | :--- |
| `status` | 先看这个。`optimal` 最可信；`infeasible` 表示无可行解；带 `|stage1_fallback` 表示 Stage 2 失败，结果是 Stage 1 |
| `plan_choices` | `路口名 -> 方案名`，Stage 1 选中的方案；Stage 2 沿用 |
| `window_choices` | `路口名 -> {"plan": ..., "up_window": 0, ...}`，记录窗口/段号选择 |
| `segment_times` | Stage 2 最重要字段：`路口名 -> term -> 秒`，例如 `{"I2": {"up.1.start": 9.0, ...}}`；Stage 1 通常为空 |

### 6.3 带宽结果

| 字段 | 含义 |
| :--- | :--- |
| `bandwidth_up` / `bandwidth_down` | 兼容口径摘要：`物理路段名 -> 带宽秒`。全局口径下各路段同值；local 口径下逐路段不同 |
| `band_up_style` / `band_down_style` | 结果口径标记；当前两个主 solver 固定输出 `multi` |
| `band_start_up` / `band_start_down` | 聚合带在各路口的到达时刻（秒，`mod cycle`） |
| `multi_bandwidths` | `方向 -> 段号 -> 全走廊带宽秒`，多段模型最原始结果 |
| `multi_band_starts` | `方向 -> 段号 -> 路口 -> 到达时刻秒`，适合解包轨迹/画图 |
| `multi_window_bands` | `方向 -> 段号 -> 窗口 key -> 带宽秒`，每个段号对应的独立局部最优带 |
| `window_bands` | `窗口 key -> 带宽秒`，跨段号聚合后的局部窗口带 |
| `window_band_ranges` | `窗口 key -> [实例...]`，每个实例有显式时间范围，最适合导出和绘图 |

`window_band_ranges` 的实例结构：

```python
{
    "direction": "down",
    "segment_no": 1,
    "bandwidth": 13.36,
    "intersections": ["I2", "I3", "I4"],
    "time_min": 18.90,
    "time_max": 59.40,
    "intersection_ranges": {
        "I2": {"start": 46.04, "end": 59.40},
        "I3": {"start": 31.76, "end": 45.11},
        "I4": {"start": 18.90, "end": 32.26},
    },
}
```

### 6.4 目标与损失

| 字段 | 含义 |
| :--- | :--- |
| `band_objective` | 带层收益，`SumGroup + BalanceGroup` 部分 |
| `band_loss` | 带层软损失，主要来自 `kind="band"` 的 `SegmentLossSpec` |
| `band_score` | `band_objective - band_loss_weight * band_loss`，带层真实得分 |
| `intersection_loss` | 交叉口层软损失：`kind="intersection"` 的损失 + 软 `LinearSpec` slack |
| `objective` | solver 原始目标值，不同 solver/模式语义不同 |

### 6.5 序列化与绘图

```python
data = solution.to_dict()   # 普通 dict，可 json.dumps

from artery_milp import plot_time_space, plot_pareto_frontier

plot_time_space(
    arterial,
    solution,
    save_path="time_space.png",
    notes=["example notes"],
)

plot_pareto_frontier(
    frontier,
    save_path="pareto.png",
    knee_point=knee,
)
```

`plot_time_space` 会自动画：

- 各路口的红条信号；
- 全局带/多段带；
- 局部窗口带；
- 按传播方向解包后的绝对时刻轨迹。

---

## 7. 案例

### 7.1 完整可运行示例

下面这个例子包含：

- 3 个路口、2 个路段；
- `I2` 有 2 段上行/下行，并带内部 `SignalConstraint`、`SignalLoss` 和 `term_bounds`；
- 双向全局带 + 均衡目标；
- Stage 1 + 两阶段求解。

```python
from artery_milp.models import (
    Arterial, GreenWindow, Intersection, Segment,
    SignalConstraint, SignalLoss, SignalPlan,
)
from artery_milp.solvers.core import BalanceGroup, ObjectiveConfig, SumGroup
from artery_milp.solvers.pipeline import (
    BandObjectiveConfig, IntersectionLossConfig, TwoStageConfig, TwoStageSolver,
)
from artery_milp.solvers.stage1 import SegmentedBandSolver

CYCLE = 90.0


def w(*pairs):
    return [GreenWindow(start, end) for start, end in pairs]


i1 = Intersection("I1", [SignalPlan(
    name="base",
    up_segments=w((0.05, 0.24)),
    down_segments=w((0.58, 0.76)),
)])

i2 = Intersection("I2", [SignalPlan(
    name="split",
    up_segments=w((0.15, 0.29), (0.31, 0.42)),
    down_segments=w((0.43, 0.66), (0.68, 0.78)),
    signal_constraints=[
        SignalConstraint(
            terms={"up.2.start": 1.0, "up.1.end": -1.0},
            sense=">=",
            rhs=0.02,
            name="上行两段间隔",
        ),
    ],
    signal_losses=[
        SignalLoss(
            terms={"down.1.start": 1.0},
            lower_threshold=0.42,
            upper_threshold=0.46,
            lower_slope=1.5,
            upper_slope=1.5,
            name="下行首段起点靠近周期中部",
        ),
    ],
    metadata={"term_bounds": {
        "up.1.start": (0.10, 0.20),
        "up.1.end": (0.26, 0.36),
        "down.1.start": (0.38, 0.48),
        "down.1.end": (0.58, 0.68),
    }},
)])

i3 = Intersection("I3", [SignalPlan(
    name="base",
    up_segments=w((0.29, 0.48)),
    down_segments=w((0.33, 0.52)),
)])

arterial = Arterial(
    cycle=CYCLE,
    intersections={"I1": i1, "I2": i2, "I3": i3},
    segments={
        "S12": Segment("S12", 190.0, 188.0, 14.0, 14.0),
        "S23": Segment("S23", 205.0, 200.0, 14.0, 14.0),
    },
    order=["I1", "S12", "I2", "S23", "I3"],
)

objective = ObjectiveConfig(
    sum_groups=[SumGroup({"up.global": 1.0, "down.global": 1.0})],
    balance_groups=[
        BalanceGroup(["up.global", "down.global"], weight=0.2, eps=0.01)
    ],
)

# ---- Stage 1 ----
stage1 = SegmentedBandSolver(config=objective, max_segments=2, max_loops=3)
sol1 = stage1.solve(arterial)
print("Stage 1:", sol1.status, sol1.plan_choices)

# ---- 两阶段 ----
config = TwoStageConfig(
    band=BandObjectiveConfig(
        mode="global",
        objective=objective,
    ),
    intersection=IntersectionLossConfig(
        tunable_intersections=None,   # 所有路口都可调
    ),
    max_loops=3,
)

sol2 = TwoStageSolver(config=config).solve(arterial)
print("Stage 2:", sol2.status)
print("plan_choices:", sol2.plan_choices)
print("segment_times I2:", sol2.segment_times.get("I2"))
print("band_objective:", sol2.band_objective)
print("intersection_loss:", sol2.intersection_loss)
print("multi_bandwidths:", sol2.multi_bandwidths)
```

运行：

```bash
cd /home/qktx/artery_milp
PYTHONPATH=/home/qktx ./conda-envs/artery_milp/bin/python your_script.py
```

### 7.2 运行仓库自带示例

`examples/window_band_ranges_case.py` 展示了：

- 4 路口、3 路段；
- 单个 `SignalPlan` 内多段；
- 一个路口多个候选方案；
- 方案内部 `SignalConstraint` / `SignalLoss`；
- 给 `split_priority` 配置第二阶段可调范围 `metadata["term_bounds"]`；
- Stage 1 求解后输出重点窗口带 `down.win3@I2-I4`，并打印 `band_objective`；
- Stage 1 之后继续运行 Stage 2，打印 `term_bounds` 带来的端点微调；
- 输出 JSON 与时空图。

```bash
cd /home/qktx/artery_milp
./conda-envs/artery_milp/bin/python examples/window_band_ranges_case.py
```

> 当前示例不再包含 `AlignmentLossBuilder`；带层损失功能暂不存在。

Stage 1 会输出：

```text
================================================================================
局部绿波带时间范围示例
求解状态: optimal
选中的方案: {'I1': 'baseline', 'I2': 'split_priority', 'I3': 'baseline', 'I4': 'baseline'}
band_objective=32.354
重点窗口带: down.win3@I2-I4
  实例 1: bandwidth=13.36s, time=[18.90, 59.40]
    I2: start=46.04s, end=59.40s
    I3: start=31.76s, end=45.11s
    I4: start=18.90s, end=32.26s
```

随后 Stage 2 会读取 `split_priority` 的 `term_bounds`，在允许范围内微调端点，并打印对比：

```text
--------------------------------------------------------------------------------
Stage 2 端点微调（term_bounds 生效）
求解状态: optimal
选中的方案: {'I1': 'baseline', 'I2': 'split_priority', 'I3': 'baseline', 'I4': 'baseline'}
  down.1.end: 名义 59.40s -> 调整后 61.20s, 允许范围 [57.60, 61.20]s
  down.1.start: 名义 38.70s -> 调整后 36.90s, 允许范围 [36.90, 40.50]s
  down.2.start: 名义 61.20s -> 调整后 63.00s, 允许范围 [59.40, 63.00]s
  up.1.end: 名义 26.10s -> 调整后 27.90s, 允许范围 [24.30, 27.90]s
  up.1.start: 名义 13.50s -> 调整后 11.70s, 允许范围 [11.70, 15.30]s
  up.2.start: 名义 27.90s -> 调整后 29.70s, 允许范围 [27.00, 29.70]s
  band_objective=35.751, intersection_loss=1.350
```

并生成：

- `examples/window_band_ranges_case_output.json`：Stage 1 结果；
- `examples/window_band_ranges_case_time_space.png`：Stage 1 时空图；
- `examples/window_band_ranges_case_stage2_output.json`：Stage 2 微调后的结果。

### 7.3 常用配置模板

**模板 1：固定分段输入**

```python
SignalPlan(
    name="baseline",
    up_segments=[GreenWindow(0.05, 0.20), GreenWindow(0.38, 0.54)],
    down_segments=[GreenWindow(0.31, 0.47), GreenWindow(0.67, 0.82)],
)
```

**模板 2：Stage 1 目标（权重形式）**

```python
build_objective_config(
    mode="global",
    up_weight=1.0,
    down_weight=1.0,
    objective_mode="balanced_composite",
    balance_eps=0.20,
)
```

**模板 3：Stage 2 目标配置**

```python
ObjectiveConfig(
    sum_groups=[SumGroup({"up.global": 1.0, "down.global": 1.0})],
    balance_groups=[BalanceGroup(["up.global", "down.global"], weight=0.2)],
)
```

**模板 4：跨路口硬约束**

```python
ConstraintBuilder([
    LinearSpec(
        terms={"I1.up.1.start": 1.0, "I2.up.1.start": -1.0},
        sense="<=",
        rhs=8.0 / 90.0,
        soft=False,
    )
])
```

**模板 5：跨路口软损失**

```python
SegmentLossBuilder([
    SegmentLossSpec(
        terms={"I2.down.1.start": 1.0},
        lower_threshold=0.42,
        upper_threshold=0.48,
        lower_slope=1.0,
        upper_slope=1.0,
        name="I2 下行首段期望区间",
    )
])
```

---

## 附录 A：term 速查

| term | 含义 | 可用位置 |
| :--- | :--- | :--- |
| `up.1.start` / `up.1.end` | 当前路口上行第 1 段起止 | `SignalConstraint` / `SignalLoss` |
| `down.2.start` / `down.2.end` | 当前路口下行第 2 段起止 | `SignalConstraint` / `SignalLoss` |
| `I2.up.1.start` | `I2` 上行第 1 段起点 | `LinearSpec` / `SegmentLossSpec` |
| `I3.down.2.end` | `I3` 下行第 2 段终点 | `LinearSpec` / `SegmentLossSpec` |
| `b_up` / `b_down` | 上/下行全走廊带宽 | 目标/高级约束/损失 |
| `tU_I2` / `tD_I2` | 上/下行带前沿到达 `I2` 的时刻 | 高级约束/损失 |
| `bU_S12` / `bD_S12` | 上/下行在物理路段 `S12` 上的带宽 | 高级约束/损失 |
| `B_bal` | 首个 `BalanceGroup` 的组内最小值变量 | 高级约束/损失 |

## 附录 B：常见问题

**Q：为什么 `LinearSpec` 报 `TermValidationError`，但我觉得 term 没写错？**

逐项检查：

- 是不是写成了局部 term `up.1.start`，但 `LinearSpec` 需要完整 `I1.up.1.start`；
- 该段号是否在某些候选方案中不存在（Stage 1 要求所有候选方案都有）；
- 该端点是否在 Stage 2 选中方案中存在；
- `plan_tags` 引用的方案是否存在。

**Q：为什么 Stage 1 多段模型只用了第 1 段？**

Stage 1 只对“所有候选方案共同拥有”的段号建模。如果某个候选方案只有 1 段，那么第 2 段不会进入 Stage 1；它会在 Stage 2 选中该方案后生效。

**Q：`window_bands` 和 `multi_bandwidths` 有什么区别？**

- `multi_bandwidths`：全走廊共享传播轨迹的带宽，按方向、段号组织；
- `window_bands` / `window_band_ranges`：子走廊独立求解得到的局部最优带，可以有自己的传播时刻。

**Q：为什么 `intersection_loss` 和我手算的软损失不一致？**

`intersection_loss` 会汇总：

- `SignalLoss`（选中方案内部）；
- `kind="intersection"` 的 `SegmentLossSpec`；
- 软 `LinearSpec` 的 slack 违反量。

`kind="band"` 的损失不计入 `intersection_loss`，而计入 `band_loss`。

## 附录 C：当前边界与开发约定

- `composite_config` / `oneway_config` / `build_objective_config` 位于 `solvers/core/objective.py`；
- Stage 2 只保留 `FullFlexiblePhaseTuneSolver`，不再有 `FlexiblePhaseTuneSolver` / `PhaseTuneSolver` 包装层；
- Stage 1 只保留 `SegmentedBandSolver`，必须显式传入 `ObjectiveConfig`，旧的 `SegmentedBandObjective` 已移除；
- 统一 term 校验当前覆盖 `ConstraintBuilder` / `SegmentLossBuilder` 生成的 `LinearSpec`；`SignalPlan` 内部的 `SignalConstraint` / `SignalLoss` 在模型构造期校验；
- `AlignmentLossBuilder` 已删除，当前版本不提供对齐损失；未来计划按“Stage 1 硬边距 + Stage 2 软边距”重新引入；
- `ObjectiveConfig` 里的 band key（如 `up.seg5`）目前由 `parse_band_key` 解析，但尚未做完整的段号边界校验，非法段号可能在后续建模阶段报错；
- 仓库里可能残留 Windows 的 `*:Zone.Identifier` 文件，可安全删除。
