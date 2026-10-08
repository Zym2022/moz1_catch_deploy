# 源码出处与版本钉定

部署日期：2026-09-28。

## 移植来源

源仓库：`~/workspace/MozBoxer`，分支 `feat/moz1-swept-geometry-catch`。

| 部署文件 | 源文件 | 源状态 | 改动 |
| --- | --- | --- | --- |
| `src/moz1_catch/core/prediction.py` | `MozBoxer/catching/prediction.py` | commit `f603888`（2026-09-29 提交；含 200/160 ms 窗口与竖直经验修正） | 无（逐字拷贝，已 diff 核验一致） |
| `src/moz1_catch/core/one_shot.py` | `MozBoxer/catching/one_shot.py` | commit `3a2b5c0`（2026-10-08 提交；两段式轴不对称退让 + 接触速度连续修复，见下） | 两处 import 改为 `moz1_catch.core.*`；`CATCH_BOX_DIMENSIONS_M` → `BOX_DIMENSIONS_M` |
| `src/moz1_catch/core/geometry.py` | `tasks/direct/mozboxer/motions/dataset_schema.py` 的 `PALM_CENTER_OFFSETS_BODY_M`、`PALM_NORMAL_AXES_BODY`；`tasks/direct/mozboxer/palm_coating.py` 的 `CENTERS_M`、`RADIUS_M`、`CORE_CENTERS_M`、`CORE_SIZE_M`；`catching/box_asset.py` 的尺寸/质量 | 同上 | 抽取为独立常数模块；2026-10-01 补充刚芯常数供离线回放，未修改规划行为 |
| `test/test_prediction.py` | `test/test_catching_prediction.py` | 同上 | 仅 import 改写 |
| `test/test_one_shot.py` | `test_catching_one_shot.py` | 同 `3a2b5c0` | 仅 import 改写（含新增的偏侧回中/keep-out/连续性契约测试） |
| `src/moz1_catch/mocap/replay_source.py` 的加载/分段规则 | `scripts/analyze_moz1_box_mocap.py` 的 `load_csv`、`flight_interval`；`scripts/validate_moz1_box_prediction.py` 的 `observations` 降采样与重定心 | 同上 | 移植为类；savolg 参数与规则不变 |

- `7231c7d`（2026-09-28，掌几何标定与动捕适配）：URDF 逐字一致、`one_shot.py`
  仅两行 import 差异、INIT_DEG 与 `[robot.posture]` 一致。
- `f603888`（2026-09-29，36 条记录与预测精化）：`prediction.py` 已同步为该提交
  逐字版本（窗口 200/160 ms + `vertical_forecast_gain_per_m`）；`one_shot.py` 在该
  提交中未变。36 条 CSV 全量同步至 `data/box_flying_csv/`；分析系先验与 k 更新为
  34 条重标定值（见 `mocap_prediction_vertical_refinement_2026-09-29.md`）。
- `3a2b5c0`（2026-10-08，退让改收势）：`one_shot.py` 同步为该提交版本（仅两行
  import + 常量名差异，已 diff 核验）。行为变化：X/Y 短缓冲（`cushion_distance`
  0.08 m，原 `retreat_distance` 0.20 m 全向行程取消）+ Z 保留 0.20 m 支撑滑行
  （`cushion_distance_z`）+ 仅横向收势到 |x| ≤ `settle_center_x`=0.12 m；删除
  `retreat_velocity` 的面法向投影（接触瞬间参考速度阶跃修复）；Z 下限
  `min_retreat_center_z`=0.75 m 仅防离谱下坠。`catch.toml` 覆盖示例与 runtime 日志
  字段（`settle_displacement_m`、`settle_start_s`）同步更新。1024 例配对仿真：
  成功率与基线无差异（core 507:507 零翻转，wide 255:256 p=1.0），强偏置终态
  |x| P95 0.233→0.120 m、越中线 81→0 例；详见 MozBoxer 侧
  `one_shot_retreat_recenter_results_2026-10-08.md`。**实机参数如需回退旧行为，
  不能再用 `retreat_distance` 覆盖（已更名），旧行为无逐字段等价组合。**

- [x] MozBoxer 源提交已回填：`7231c7d`、`f603888`、`3a2b5c0`（最新同步点）

## 数据出处

`data/meshes/`：从 MozBoxer 提交 `f603888ad7677c5c86f47809dcc105575c9152a0` 的
`source/MozBoxer/assets/moz1/meshes/` 逐字拷贝当前部署 URDF 引用的 20 个 STL。
2026-10-01 纳入 Git，供开发机与机器人主机独立生成离线回放；新实物日志与
生成网页仍在被 Git 忽略的 `output/`，不随仓库同步。

`data/box_flying_csv/1..5.csv`：`MozBoxer/catching/box_flying_data/csv/` 的逐字拷贝
（动捕导出，原生 200 Hz，含手持段）。五条记录的 SHA256 见 MozBoxer 侧
`output/mocap_prediction_20260928/03_final_validation/calibration.json`。

## 已钉定的标定常数

- `T_DG`：2026-09-27 手工标定（文档
  `catching/docs/mocap_box_geometry_calibration_2026-09-27.md`）。
- 分析系加速度先验 `(-0.011349, -0.386894, -8.791483) m/s²` 与竖直修正增益
  `k=0.0682745 m⁻¹`：36 条记录研究、34 条可评价记录重标定（文档
  `mocap_prediction_vertical_refinement_2026-09-29.md`，2026-09-29）。部署时先验经
  `rotate_prior`（用等效外参 T_base_torso @ T_FM）换算到 base_link 系；k 沿竖直
  轴作用，在 base_link（+Z 竖直）可直接沿用。
- `LATE_COMMIT_SETTINGS`：配对鲁棒性实验选定（文档
  `one_shot_normal_priority_results_2026-09-26.md` 与后续提交）。

## 运行时推导的机器人常数（2026-09-28 本包新增，后改为加载时推导）

`config/robot.toml [robot.posture]` 的准备姿态关节角是唯一事实源；`load_config`
经 `src/moz1_catch/kinematics.py` 的 URDF FK 在加载时推导 T_base_torso、
T_tcp_palm（左右，掌目标系→法兰）与 base_link 系等待位姿。
`scripts/compute_palm_frames.py` 退为验证工具（--check-base 复核夹具残差）。推导输入：

| 输入 | 来源 |
| --- | --- |
| `data/moz1_boxer.urdf` | `MozBoxer/assets/moz1/moz1_boxer.urdf` 逐字拷贝（32 KB） |
| 准备姿态关节角 INIT_DEG | `MozBoxer/catching/simulation.py`（2026-09-28 工作树，含初始姿态修改；现为 robot.toml 默认值） |
| 掌目标偏置 | 本包 `core/geometry.py`（−60 mm 修订） |

交叉验证：FK 与 Isaac 夹具参考值（`catching_box_301_305_510g_20260927` nominal
run 的起始掌位姿）旋转残差 0.002°，位置残差每手 10.05 mm、方向沿 ±X——对应
`7231c7d` 中的准备姿态外移 seeding（夹具值是该项修订前记录；同提交还含掌目标
点/球阵修订，已含在本包 geometry 常数中）。

T_tcp_palm 语义（2026-09-28 勘误）：规划位姿是"掌目标系"（切平面中心原点 +
hand-link 轴向），非 hand-link 原点；常量存 ^掌目标系 T_flange = Trans(−偏置) ·
^hand T_flange，`palm_targets_to_tcp` 按存储方向直接使用。方向回归测试
（随机旋转下与 hand 系组合等价）与切平面自检测试均已入测试集。

规划参考系为 **base_link**（与仿真一致，规划代码零改动）。硬件侧 torso_flange
只在两个换算边界出现：入口 `T_BG = T_base_torso @ T_FM @ T_MD @ T_DG`
（`calib.FrameChain`），出口 `T_torso_tcp = inv(T_base_torso) @ T_base_hand @
T_hand_tcp`（`palm_targets_to_tcp`）。T_base_torso 为锁定 legwaist 准备姿态下
的 URDF FK 常量（纯平移 `(0, +0.0236, +1.2024) m`，旋转恒等），已填入
`robot.toml`；T_FM 为手眼标定占位（待标，目标 torso_flange）。

## 明确未移植

`simulation.py`、`qp.py`、`population.py` 及整个 Isaac Lab 依赖树。实物执行链由
机器人笛卡尔控制器替代；仿真仓库继续作为调参对照。仿真侧 QP 的法向优先
capture_priority 时序在实物上没有对应物，属已知行为差异（见
`docs/deployment_plan.md` 第 8 节）。
