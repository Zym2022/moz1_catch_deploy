# MozBoxer 一次规划接箱：仿真到实物部署迁移设计

日期：2026-09-28。状态：已实现并通过 36 项测试与五条实物记录的端到端回放。
本文是该部署包的设计文档：模块划分、数据流、坐标与时间约定、配置策略、
以及每个"预留接口"需要补什么。阅读顺序：本文 → README.md（迁移步骤）→ 代码。

## 1. 迁移范围

从 MozBoxer（分支 `feat/moz1-swept-geometry-catch`，含 2026-09-28 未提交工作树修改，
详见 PROVENANCE.md）只迁移三层：

| 层 | 文件 | 说明 |
| --- | --- | --- |
| 预测 | `core/prediction.py` | 位姿窗口 + 加速度先验的自由飞行估计，逐字拷贝 |
| 规划 | `core/one_shot.py` | 一次规划与 `CatchPlan.target(t)`，仅改两行 import |
| 几何 | `core/geometry.py` | 箱体尺寸、掌目标点、法向、12 球阵列，从 tasks 树抽取为常数 |

**不迁移**：`simulation.py`（Isaac 场景）、`qp.py`（关节速度 QP——实物笛卡尔控制器
替换的正是这段执行链）、`population.py`。仿真仓库保持原样，作为调参对照回放器。

依赖只有 numpy / scipy（+3.10 的 tomli）。核心层不 import 任何 I/O、ROS 或 Isaac；
`rclpy` 只在 `robot/ros2_sink.py` 内懒加载，因此除真实发布外的所有功能（测试、回放、
干跑）可在无 ROS 的机器上运行。

## 2. 目录结构

```
moz1_catch_deploy/
├── pyproject.toml            # uv 项目；uv_build 后端，src 布局
├── docs/deployment_plan.md   # 本文
├── PROVENANCE.md             # 源码出处与版本钉定
├── config/
│   ├── catch.toml            # 任务几何（base_link 系，与仿真一致）、预测、规划覆盖、布防、安全
│   ├── frames.toml           # T_FM/T_DG 外参、单位、四元数约定、时钟偏置
│   ├── robot.toml            # 准备姿态关节角（唯一事实源；其余由加载时 FK 推导）+ 可选实测覆盖
│   ├── interfaces.toml       # UDP/ROS2 接口占位、指令周期
│   └── profiles/             # live / bench / replay 深覆盖层
├── src/moz1_catch/
│   ├── core/                 # 移植的纯算法层（见上表）
│   ├── calib.py              # 入口链（T_base_torso·T_FM·T_MD·T_DG）、时钟、先验换算
│   ├── kinematics.py         # URDF FK：关节角 → 等待位姿/T_base_torso/T_tcp_palm（加载时推导）
│   ├── config.py             # TOML 加载、profile 合并、交叉校验、FK 推导
│   ├── mocap/
│   │   ├── source.py         # Observation 模型 + BoxObservationSource 协议
│   │   ├── udp_source.py     # UDP 监听 + 可插拔报文解析器（厂商格式占位）
│   │   └── replay_source.py  # CSV 回放（分析基变换、因果降采样、重定心、实时步进）
│   ├── robot/
│   │   ├── sink.py           # HandTargets + CartesianCommandSink 协议 + 掌目标系→法兰换算
│   │   ├── ros2_sink.py      # rclpy 发布器（话题/消息类型占位，单点适配）
│   │   └── mock_sink.py      # 内存记录 sink
│   ├── safety.py             # 工作空间盒 + 指令步长钳位（最后防线）
│   ├── executor.py           # hold/execute/reject/done 四相指令流
│   ├── runtime.py            # WAIT→ARMED→FLIGHT→EXECUTING 状态机
│   └── trace.py              # 每次尝试的 npz + meta.json（键名与仿真 trace 对齐）
├── scripts/
│   ├── run_catch.py          # 实物入口：UDP + ROS2
│   ├── dry_run_replay.py     # 阶段2干跑：CSV 回放 + mock sink
│   ├── check_frames.py       # 外参校验、静止箱端到端核对、先验换算打印
│   ├── compute_palm_frames.py# 验证工具：配置推导值 + Isaac 夹具比对（--check-base）
│   └── fake_mocap_sender.py  # 以 prototype_json 向 UDP 回放 CSV（先做逆链变换）
├── test/                     # 移植的 2 个测试文件 + 5 个新测试文件
└── data/                     # 五条实物记录 + moz1_boxer.urdf（FK 用）
```

## 3. 数据流与时间轴（单线程状态机）

```
UDP/CSV ──Observation──▶ runtime 状态机 ──Commit──▶ estimate_box_flight
 (base_link 系,          │                            │
  宿主单调钟)             │                            ▼
                          │                        plan_catch(重定时)
                          ▼                            │
                  executor.tick ──plan.target(t)──▶ safety 钳位 ──▶ base→torso→法兰 ──▶ sink
                                                                  （ROS2 topic / mock）
                          └────────── trace.npz + meta.json（每次尝试）
```

- **单时钟**：全程 `time.perf_counter()`（宿主单调钟）。动捕时间戳经
  `frames.toml` 的 `clock_offset_s` 常数偏置映射到该钟。回放源以首帧锚定映射。
- **内部坐标系恒为 base_link，与仿真完全一致**：任务几何、等待位姿、工作空间、
  回放数据的数值都不变，规划代码不感知 torso_flange。硬件侧的 torso_flange
  只出现在两个显式换算边界，各自一个来源单一的常量：
  - **入口（动捕）**：手眼标定得到 T_FM（mocap 全局系 → torso_flange）；
    观测经 `T_BG = T_base_torso @ T_FM @ T_MD @ T_DG` 一步换到 base_link
    （`calib.FrameChain`，两个常量分别校验、可分别 debug）。
  - **出口（控制器）**：掌目标系位姿经 `T_torso_flange = inv(T_base_torso)
    @ T_base_掌目标系 @ T_palm_tcp` 换到"法兰在 torso_flange"的位姿再发布
    （`palm_targets_to_tcp` 单点完成）。
  - T_base_torso（^base T_torso）与等待位姿都由 `robot.toml [robot.posture]`
    的准备姿态关节角在配置加载时经 URDF FK 推导（锁定 legwaist 准备姿态下
    为纯平移 `(0, +0.0236, +1.2024) m`、旋转恒等）——**关节角是唯一事实源，
    改姿态不存在漏改**。
- **延迟补偿走规划器原生机制**：提交时
  `execution_delay_s = (规划开始 − 最新观测时刻) + command_latency_s`，
  `planning_started_s = perf_counter()`，`control_dt_s = 指令周期`。
  `plan_catch` 内部按实测计算耗时对齐控制节拍并重建早段五次多项式，
  **保持绝对接触时刻不变**（移植测试覆盖该契约）。执行起点 =
  `观测时刻 + plan.execution_delay_s`。

## 4. 功能模块与预留接口

### 4.1 动捕输入（mocap/）
- `BoxObservationSource` 协议：`next(timeout_s) -> Observation | None`。
- **UDP 解析器预留**：`udp_source.py` 的 `PARSERS` 注册表。配置默认
  `parser = "TODO_REPLACE_ME"` 会拒绝启动。实现厂商格式时在 `PARSERS` 注册一个
  返回 `(device_t, 位置, xyzw 四元数, tracking_state, rigid_body_id)` 的函数即可，
  其余（滤波、坐标变换、超时）不动。`prototype_json` 是文档化的台架测试格式。
- 跟踪状态有效值集合在 `frames.toml`（CSV 惯例为 8；UDP API 需确认）。
- 单位与四元数顺序在 `frames.toml` 声明，加载时校验（仅支持 xyzw；wxyz 需在解析器内转换）。

### 4.2 机器人输出（robot/）
- `CartesianCommandSink` 协议：`send(HandTargets, t_host)`。**控制器只收位姿**：
  规划器输出的速度仅进日志。
- **ROS2 适配点集中在 `ros2_sink.py`**：`interfaces.toml` 填入真实
  `cartesian_topic`、`message_type`、`message_layout` 后，仅需改
  `build_cartesian_message()` 一个函数适配真实消息（当前占位为 PoseArray
  左右手布局）。话题名与消息类型含 "TODO" 时拒绝启动。
- **掌→法兰换算**：规划器输出的是**掌目标系**的位姿——自定义系：原点在 12 球
  阵列前切平面中心（`geometry.py` 有切平面关系的自检测试），轴向为 hand-link
  轴（安装关节给定的自定义朝向）。`T_tcp_palm`（^掌目标系 T_flange）由
  加载时 FK 从 URDF 固定关节与切平面偏置推导，`palm_targets_to_tcp` 按存储
  方向直接组合 `T_torso_base @ T_base_palm @ T_palm_tcp`；前提是控制器控制点
  就是 URDF 的 `left_flange`/`right_flange`，上机前与厂家定义核对，若不同在
  `[robot.mounting_overrides]` 填实测矩阵覆盖。
- **掌位姿反馈预留**：`robot.toml` `feedback.palm_state_topic`；为空时提交时刻的
  双掌位姿用推导的等待位姿近似（机器人一直保持等待位姿，近似误差=跟踪误差）。

### 4.3 状态机（runtime.py）
`WAIT → ARMED → FLIGHT → EXECUTING/REJECTED → DONE`，外加 `FAULT`。
- 布防：箱体在释放区域内准静止 `arming.hold_s` → ARMED；丢失跟踪或出区域回 WAIT。
- 释放检测：速度 ≥ `release_speed_threshold_mps`（存在手持上摆阶段，检测早于真实
  出手约 0.15 s；估计器窗口因此只覆盖纯飞行段，`release_settle_s` 再排除 25 ms）。
  WAIT 中直接检测到飞行也允许（录制里准静止段常不足 0.5 s）。
- 提交：**仅由新到达的有效观测**越过 `commit_plane_y_m` 触发（不用插值、不用旧帧），
  且距释放检测 ≥ `min_commit_delay_s`。超时 `commit_timeout_s` 拒接。
- 提交后：冻结观测，估计+规划一次（ValueError → 拒接并记录原因）；继续排空动捕
  仅写日志，供事后残差分析。
- 任何桥接异常进 `FAULT`：保持最后指令并停机，绝不因软件异常甩臂。

### 4.4 指令流（executor.py）
- `hold`：以指令周期发布等待位姿（WAIT/ARMED/FLIGHT 全程）。
- `execute`：`plan.target(min(t−执行起点, contact+stop))`，超出后 `done` 保持终位姿
  （接住后双手停在收箱位，由操作员接管）。
- `reject`：五次多项式回blend 到等待位姿（0.25 s，可配）。
- 指令周期 `command_period_s` 与 `command_latency_s` 是**必须实测**的两个数
  （见 README 迁移步骤第 4 步）。

### 4.5 安全钳位（safety.py）
规划器已有自己的速度/加速度预算；本层只拦配置与标定错误：工作空间盒投影 +
相邻指令步长上限。**干净的一次尝试不应触发任何钳位**；每次触发都进 trace。

### 4.6 记录（trace.py）
每次尝试一个目录：`trace.npz`（键名与仿真 trace 对齐：`target_palm_position`、
`observation_*`、`catch_decision`、`contact_time_s` 等）+ `meta.json`（配置与外参
文件 SHA256、准备姿态关节角与 URDF 指纹、catch/prediction 设置全量、事件日志）。
仿真侧分析脚本可基本直接复用。

## 5. 配置策略（什么变动暴露在哪）

| 变动频率 | 内容 | 位置 |
| --- | --- | --- |
| 每次调参 | 提交/接触平面、预测窗口与先验、CatchSettings 任意字段覆盖 | `catch.toml` |
| 标定后一次 | T_FM、T_DG、单位、时钟偏置 | `frames.toml`（版本号随之外参一起更新） |
| 准备姿态/机械改动后 | 只改 `[robot.posture]` 关节角；等待位姿、T_base_torso、T_tcp_palm 加载时 FK 推导 | `robot.toml` |
| 硬件与 URDF 不符时 | `[robot.mounting_overrides]` 实测矩阵覆盖推导值 | `robot.toml` |
| 接口确认后 | 话题、消息类型、UDP 端口/解析器、指令周期 | `interfaces.toml` |
| 实验分层 | live / bench / replay 场景差异 | `profiles/*.toml`（深合并覆盖） |

`[planning.overrides]` 直接透传 `CatchSettings` 字段名（未知字段报错），
默认值为 `LATE_COMMIT_SETTINGS`（配对实验基线：plane −0.58、掌速 3 m/s、
加速度 48 m/s²、close 80 ms、retreat 0.20 m 等）。

## 6. 标定与上机前置条件（TODO 清单）

1. **手眼外参 T_FM（mocap 全局系 → torso_flange）**：按
   `base_link_mocap_hand_eye_plan_2026-09-28.md` 的 Park 流程采集求解，机器人侧
   位姿输入必须用以 torso_flange 为父系的 SDK 末端位姿（这样解出的 X 直接是
   T_FM；计划文档虽以 base_link 推导，方程对任意固定机器人参考系等价）。
   结果写入 `frames.toml`；底盘搬动或动捕重置后重标。入口换算到 base_link 由
   固定常量 T_base_torso 完成，不需要额外操作。
2. **箱体刚体 T_DG**：已标定（2026-09-27），已写入默认配置；重建刚体后重核。
3. **时钟偏置**：挥臂相关法测动捕设备钟 → 宿主钟偏置，写入 `frames.toml`；
   运行期漂移以 `max_observation_age_s`（30 ms）兜底拒旧帧。
4. **加速度先验换系**：T_FM 定好后运行 `scripts/check_frames.py`，其打印的
   base_link 先验（经等效外参 T_base_torso @ T_FM 换算）覆盖
   `catch.toml [prediction]`。分析系先验与实物先验是同一物理量在不同坐标系的
   分量，换算已内置，不要手抄分析系数值。
   `check_frames.py --chain-check` 的抛物线不变性测试（含非平凡 T_base_torso）
   专门防旋转方向/符号错误。
5. **静止箱端到端核对**：箱摆已知位置，`check_frames.py --box-pose ...` 输出
   base_link 下箱心与八角点，与卷尺/机器人示教位姿比对（建议 ≤10 mm）。
6. **等待位姿、T_base_torso、T_tcp_palm**：由 `robot.toml [robot.posture]`
   的准备姿态关节角（默认即仿真 INIT_DEG）在配置加载时经 URDF FK 自动推导
   ——关节角是唯一事实源，改姿态不存在漏改（敏感性有测试锁定）。推导值与
   Isaac 夹具交叉验证（旋转 0.002°；位置差 10.05 mm 恰为 9-28 源提交 7231c7d 的准备姿态外移量），
   `compute_palm_frames.py --check-base` 随时复核。待办只剩核对控制器控制点
   确为 `left_flange`/`right_flange`；若不同，在 `[robot.mounting_overrides]`
   填实测矩阵覆盖推导值。
7. **接口确认**：笛卡尔话题名/消息类型/布局；动捕 UDP 报文格式与端口；
   控制器可接收的指令频率（填 `command_period_s`）。
8. **延迟实测**：指令流首条到位姿起动的端到端延迟，填 `command_latency_s`；
   用动捕末端刚体记录轨迹跟踪滞后，若稳定滞后 τ，可整体前移时间轴补偿。

## 7. 分阶段上机

- **阶段0（任意机器）**：`uv sync && uv run pytest`；36 项测试含五条记录回放
  （决策与 2026-09-28 冻结研究一致：2 号接受，1/3/4/5 拒绝）。
- **阶段1**：完成第 6 节标定清单；`check_frames.py` 全绿。
- **阶段2（干跑）**：`dry_run_replay.py` 走 CSV 全链路（真实节奏、mock sink）；
  台架上 `run_catch.py --profile bench` + `fake_mocap_sender.py`（sender 先做
  逆链变换，bench 走真实输入链）验证 UDP 路径。
- **阶段3（低速实接）**：降低释放高度与初速、垫软地面、监护；每次投掷留档。
- **阶段4（数据驱动调参）**：从 trace 的预测残差与跟踪误差分别归因；
  `lateral_contact_bias_gain` 保持 0，直到实测左右接触时刻差可供标定。

## 8. 已知边界

- 预测精度 ≠ 接触成功：五条记录的 9–13 mm 位置误差是回放指标；
  2 号记录存在单侧边缘接触风险，上机首阶段以"双侧接触窗口按预测出现"为成功判据。
- 接触后的耦合动力学没有任何实物模型；QP 的法向优先保护在实物上由控制器
  刚度行为替代，行为差异未知，阶段3 从低速开始的原因即此。
- `commit_timeout_s = 0.55` 从"检测到运动"起算（含手持上摆约 0.15 s），
  与仿真从指令释放起算的 0.40 s 语义不同；若后续改进释放检测（如竖直速度峰值
  规则），可收回 0.40。
- 仿真执行链的 capture_priority 权重时序没有对应物；实物由位姿流 + 控制器内环实现。
