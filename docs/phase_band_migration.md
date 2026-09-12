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

## 当前状态

- `FullFlexiblePhaseTuneSolver` 已实现相位变量、BandModel、ObjectiveConfig、
  loss_builder、constraint_builder、alignment_builder、max_loss。
- `PhaseTuneSolver` 已成为薄包装器，所有模式/损失/约束都走新路径，
  不再引用 `_LegacyPhaseTuneSolver`。
- `general_case.py` 场景 1~10 全部通过。
- `_LegacyPhaseTuneSolver` 暂留作历史参考；后续可删除。
