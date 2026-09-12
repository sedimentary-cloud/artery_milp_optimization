# PhaseTuneSolver -> BandModel + ObjectiveConfig 迁移计划

## 目标
让 PhaseTuneSolver 复用：
- `BandModel` 的窗口带格 `B[d,k,j] <= b[d,i]`
- `ObjectiveConfig` 的 sum / balance 目标

## 结构
1. 新增 `solvers/flexible_phase_solver.py`
   - 锁方案：从 prior.plan_choices 读取
   - 相位变量：g_{i,p}
   - 窗口表达式：`window_exprs(plan, C)`
   - 基础段带宽：`b[d,i]` 由两端窗口表达式约束
   - 带格：`B[d,k,j] <= b[d,i]`
   - 目标：`ObjectiveConfig` 解析
   - 第二目标：hinge 损失 / LinearSpec / AlignmentLossBuilder

2. `PhaseTuneSolver` 退化为工厂
   - 保持现有构造参数
   - 内部构造 `FlexiblePhaseTuneSolver`

3. 接入
   - `TwoStageSolver`
   - `EpsilonConstraintRunner`

## 回归
- `general_case.py` 场景 4~10 结果保持一致
- 新增 stage2 新旧对比测试
