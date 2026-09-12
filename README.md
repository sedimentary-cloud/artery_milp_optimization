# GreenWave: 干线绿波协调优化

本项目用于干线信号协调与绿波带宽优化，核心是把不同方向、不同窗口大小的绿波带统一成“窗口带格”，再用可配置的目标函数求解。

## 1. 目录结构

```text
.
├── models.py                     # 数据模型：Arterial / Intersection / Segment / SignalPlan / GreenWindow / Phase
├── solution.py                   # 求解结果 Solution
├── plotting.py                   # 时空图绘制
├── solvers/
│   ├── base.py                   # Solver 抽象基类
│   ├── milp.py                   # 旧求解器与 legacy 实现
│   ├── phase.py                  # 相位窗口表达式、LinearSpec、PhaseLoss、AlignmentLoss
│   ├── staged.py                 # 两阶段/ε-约束/Pareto 扫描
│   ├── objective_config.py       # ObjectiveConfig / SumGroup / BalanceGroup
│   ├── band_model.py             # 双向窗口带格 BandModel
│   ├── flexible_band_solver.py   # FlexibleBandSolver + 旧求解器工厂
│   └── flexible_phase_solver.py  # 相位微调 + BandModel/ObjectiveConfig
├── examples/
│   ├── general_case.py           # 6 路口综合案例（多场景/多目标/多约束）
│   └── ...
└── docs/
    └── phase_band_migration.md
```

## 2. 数据模型

### 2.1 GreenWindow

```python
GreenWindow(start, end)
```

表示占周期比例的一段绿灯窗口：

```text
start, end ∈ [0, 1]
0 <= start < end <= 1
```

例如 `GreenWindow(0.1, 0.3)` 表示绿灯从 `0.1C` 到 `0.3C`。

### 2.2 SignalPlan

一个路口的一种信控方案，包含：

- `up_windows`: 上行绿灯窗口列表；
- `down_windows`: 下行绿灯窗口列表；
- `phases`: 可选相位结构；
- `up_phase` / `down_phase`: 上/下行绑定到哪个相位；
- `lost_time`: 周期损失时间（秒）。

### 2.3 Segment

相邻路口之间的路段：

```python
Segment(length_up, length_down, speed_up, speed_down)
```

行驶时间：

```text
tau_up_i   = length_up / speed_up
tau_down_i = length_down / speed_down
```

### 2.4 Arterial

一条干线：

```python
Arterial(cycle, intersections, segments, order)
```

`order` 形如：

```text
["I1", "seg1", "I2", "seg2", ...]
```

## 3. 核心抽象：窗口带格

### 3.1 基础段带宽

对每个方向 `d ∈ {up, down}`，每个路段 `i`：

```text
b[d, i] = 该方向第 i 段的基础带宽（秒）
```

### 3.2 窗口带

窗口带定义为：

```text
B[d, k, j] = 方向 d，窗口大小 k，起点路口 j 的窗口带宽
```

约束：

```text
B[d, k, j] <= b[d, i]     for i = j ... j+k-1
```

含义：窗口带的宽度不能超过窗口内任意一个路段的带宽。

- `k = 2`：分段带；
- `k = n`：全局带；
- 其他 `k`：小绿波带。

变量数量：

```text
2 * Σ_{k=2}^{n} (n-k+1) = O(n²)
```

10 个路口时每方向 45 个窗口变量，规模很小。

## 4. 数学模型

### 4.1 变量

| 变量 | 含义 |
|---|---|
| `b_up_i` / `b_down_i` | 上行/下行第 i 段基础带宽 |
| `tU_i` / `tD_i` | 上行/下行带前沿到达路口 i 的时刻（秒，mod C） |
| `mU_i` / `mD_i` | 整数圈数修正量 |
| `B[d,k,j]` | 窗口带宽度 |
| `delta` | 方案/窗口联合选择 0-1 变量 |
| `g_{i,p}` | 路口 i 第 p 个相位的时长（秒） |

### 4.2 带前沿传递

上行：

```text
tU_{i+1} = tU_i + tau_up_i + C * mU_i
```

下行：

```text
tD_i = tD_{i+1} + tau_down_i + C * mD_i
```

`m` 是整数圈数，用于把时间折算回周期内。

### 4.3 绿灯窗约束

对每个路段两端路口窗口：

```text
t_i >= start * C
t_{i+1} + b_i <= end * C
t_i + b_i <= end_i * C
```

如果方案/窗口未定，边界写成 0-1 变量的线性组合：

```text
t_i - Σ δ * start * C >= 0
t_i + b_i - Σ δ * end * C <= 0
```

### 4.4 窗口带约束

```text
B[d,k,j] <= b[d,i]     i = j ... j+k-1
```

### 4.5 相位时长约束

```text
min_green_p <= g_{i,p} <= max_green_p
Σ_p g_{i,p} + lost_time = C
```

相位决定的绿灯窗：

```text
up_start_i = Σ_{p before up_phase} g_{i,p}
up_end_i   = up_start_i + g_{i, up_phase}
```

## 5. 目标函数

### 5.1 SumGroup

任意带、任意方向、任意权重求和：

```text
Σ w_key * band_key
```

### 5.2 BalanceGroup

组内取 min：

```text
B_group <= band_member    for each member
目标中加入 w_group * B_group
```

默认带 ε 托底：

```text
+ w_group * eps * Σ band_member
```

例如双向均衡：

```python
BalanceGroup(["up.global", "down.global"], weight=1.0)
```

表示 `max min(b_up_global, b_down_global)`。

## 6. 求解器

### 6.1 CompositeBandSolver

等价于：

```python
FlexibleBandSolver(
    composite_config(
        up_weight=1.0,
        down_weight=1.0,
        objective_mode="sum",   # 或 balanced / balanced_composite
    )
)
```

### 6.2 OneWayPrioritySolver

等价于：

```python
FlexibleBandSolver(
    oneway_config(
        up_weight=1.0,
        window_weights={2: 1.0, 3: 0.5},
        n_intersections=6,
    )
)
```

### 6.3 FlexibleBandSolver

核心通用求解器：

- 双向分段带；
- 窗口带格；
- 方案/窗口联合选择；
- `ObjectiveConfig` 目标。

求解结束后会从最终带宽结果后处理回填：

- 两个方向；
- `k=2..5` 的窗口绿波带；
- key 形如 `up.win3@I1-I3` / `down.win3@I1-I3`。

回填只写入 `Solution.window_bands`，不修改优化变量或目标值。

### 6.4 PhaseTuneSolver

完整走 `FlexiblePhaseTuneSolver`：

```text
相位变量 g -> 绿灯窗
绿灯窗 + t + b -> 基础段带宽
b -> BandModel 窗口带格 B

band_score      = band_objective - λ * band_loss
intersection_loss = 相位 hinge + 软 LinearSpec slack
```

两类目标明确区分：

- 绿波带层：`band_objective`、`band_loss`、`band_score`；
- 交叉口/相位层：`intersection_loss`。

`alignment_builder` 属于绿波带层损失，通过 `band_loss_weight`
以加权和形式进入 `band_score`，不再混进 `intersection_loss`。

### 6.5 EpsilonConstraintRunner

ε-约束法扫描帕累托前沿：

```text
max 主目标
s.t. total_loss <= eps
```

## 7. 损失项

### 7.1 Phase hinge loss

```python
PhaseLossSpec("P1", threshold=30, slope=2)
```

表示：

```text
loss = slope * max(0, threshold - g)
```

过大惩罚：

```python
upper_threshold=50, upper_slope=3
```

```text
loss += upper_slope * max(0, g - upper_threshold)
```

该损失进入交叉口层：

```text
intersection_loss
```

### 7.2 LinearSpec

线性硬/软约束：

```python
LinearSpec({"I2.P1": 1.0, "I2.P2": 1.0}, sense=">=", rhs=60.0)
```

硬约束直接进入 MILP；
软约束自动创建 slack，进入交叉口层：

```text
intersection_loss += penalty * slack
```

### 7.3 AlignmentLossBuilder

带中心对齐：

```text
center = t_i + b / 2
loss = max(0, |center - a_i| - tolerance)
```

该损失属于绿波带层，进入 `band_loss`：

```text
band_score = band_objective - band_loss_weight * band_loss
```

它可以同时用在第一阶段和第二阶段。

## 8. 运行示例

```bash
cd /home/qktx/artery_milp
/home/qktx/artery_milp/conda-envs/artery_milp/bin/python examples/general_case.py
```

`general_case.py` 会生成：

- 求和目标时空图；
- 均衡目标时空图；
- one-way 目标时空图；
- 两阶段相位优化时空图；
- hinge 损失、硬约束、软约束时空图；
- 带宽-损失帕累托前沿；
- one-way 帕累托前沿；
- 对齐损失帕累托前沿。
