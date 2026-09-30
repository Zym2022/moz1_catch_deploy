# moz1_catch_deploy — MozBoxer 一次规划接箱的实物部署包

把 MozBoxer 仿真中验证过的一次规划接箱（预测 + 规划）迁移到机器人主机上运行：
监听动捕 UDP 刚体位姿，箱体穿越提交平面后冻结一次规划，按控制周期向 ROS2 笛卡尔
话题发布双掌位姿目标。设计文档见 [docs/deployment_plan.md](docs/deployment_plan.md)，
源码出处与版本钉定见 [PROVENANCE.md](PROVENANCE.md)。

核心算法层（`src/moz1_catch/core/`）是 MozBoxer 的逐字移植，仅依赖 numpy/scipy；
桥接代码（动捕输入、机器人输出、状态机、指令流）全部为本包新写，与算法层解耦。
当前状态：**68 项测试全部通过**，含五条实物自由飞行记录的端到端回放
（决策与 2026-09-28 冻结研究一致：2 号接受，1/3/4/5 拒绝）。

## 快速开始（开发机，无 ROS）

```bash
uv sync                      # 建 .venv 并安装（含 dev 测试组）
uv run pytest -q             # 68 项测试，约 12 s（回放实时步进）
uv run python scripts/dry_run_replay.py --csv data/box_flying_csv/2.csv
uv run python scripts/check_frames.py --chain-check
uv run python scripts/compute_palm_frames.py --check-base   # URDF FK + 夹具交叉验证
```

`dry_run_replay.py` 以真实节奏把一条记录喂给完整运行时（布防、释放检测、提交、
规划、指令流），用内存 mock sink 记录全部指令；产物在 `output/attempt_*/`
（`trace.npz` + `meta.json`，键名与仿真 trace 对齐，可直接用既有分析脚本）。

## 迁移步骤（机器人主机）

按顺序执行；每一步都有明确的完成判据。详细背景见设计文档第 6、7 节。

1. **拷贝与建环境**
   ```bash
   # 机器人主机上（有 ROS2 的环境）：
   cd ~/workspace && rsync -av <开发机>:workspace/moz1_catch_deploy/ moz1_catch_deploy/
   cd moz1_catch_deploy
   uv venv --system-site-packages      # 关键：叠加系统 rclpy
   uv pip install -e . && uv pip install pytest
   .venv/bin/python -m pytest -q       # 判据：68 passed（此步不需要 ROS）
   source /opt/ros/humble/setup.bash && source ~/ros_pkg/movax_interface/install/setup.bash
   .venv/bin/python -c "import rclpy, mc_core_interface.msg"   # 判据：无 ModuleNotFoundError
   ```
   开发机上日常用 `uv sync`（隔离环境）即可；只有需要 rclpy 的实机运行用
   上面的 system-site-packages 方式。`uv.lock` 随仓库走，两端锁同一版本。
   **切勿在机器人主机上运行 `uv sync`**：它会把 `.venv` 重建为隔离环境，
   系统 rclpy 随即不可见（报 `ModuleNotFoundError: No module named
   'rclpy'`）；已误跑的话 `rm -rf .venv` 后按上面三行重建。若 rclpy 仍
   找不到，另查两点：当前终端是否 source 过 ROS（**每个新终端都要重新
   source**）；命令是否带了 `PYTHONPATH=...` 前缀——它整体覆盖 source
   得到的包路径（本包是可编辑安装，任何 PYTHONPATH 前缀都不要加）。
2. **填接口占位**（`config/interfaces.toml`）：笛卡尔话题名、消息类型与布局；
   动捕 UDP 端口与报文解析器（在 `src/moz1_catch/mocap/udp_source.py` 的
   `PARSERS` 注册一个解析函数，参考 `prototype_json`）；实测控制器指令频率填
   `command_period_s`。所有含 "TODO" 的占位未填时程序拒绝启动，这是故意的。
3. **完成标定并写入配置**：
   - T_FM 手眼外参（mocap 全局系 → **torso_flange**）。按 MozBoxer 的
     `base_link_mocap_hand_eye_plan_2026-09-28.md` 流程采集求解，注意标定的
     机器人侧位姿输入必须用 SDK 输出的、以 torso_flange 为父系的末端位姿，
     这样解出的 X 直接就是 T_FM → `config/frames.toml`。底盘搬动或动捕重置
     后重标；legwaist 保持准备姿态锁定（base_link 与 torso_flange 之间是
     固定常量，管线不需要它）。
   - 静止箱端到端核对：`check_frames.py --box-pose "x y z qx qy qz qw"` 输出
     base_link 下箱心与八角点，与卷尺/示教位姿比对（建议 ≤10 mm）。
4. **换先验、测延迟**：
   - `check_frames.py` 打印的 base_link 系先验（经等效外参
     T_base_torso @ T_FM 换算）覆盖 `catch.toml [prediction]`
     （分析系先验与实物先验是同一物理量在不同坐标系的分量表达，换算公式
     已内置，不要手抄分析系数值）；同时把竖直修正增益
     `vertical_forecast_gain_per_m` 设为标定值 0.0682745（2026-09-29 研究，
     沿竖直轴作用，base_link +Z 即竖直，直接沿用；置 0 停用）；
   - 实测"规划完成 → 手臂实际起动"的端到端延迟填 `command_latency_s`；
   - 用动捕末端刚体记录一次空载接箱轨迹的跟踪滞后 τ；若稳定，在指令流里
     前移时间轴 `t + τ` 补偿（当前默认 0）。
5. **核对机械几何、台架干跑**：
   - 准备姿态关节角是唯一事实源（`robot.toml [robot.posture]`，默认即仿真
     INIT_DEG）；等待位姿、T_base_torso、T_tcp_palm 全部在配置加载时由
     URDF 正运动学自动推导，改姿态只改关节角，不存在漏改。推导值与 Isaac
     夹具交叉验证过（旋转 0.002°，位置差恰为 9-28 源提交 7231c7d 的准备姿态外移修订），
     `compute_palm_frames.py --check-base` 随时可复核。
   - 上机前确认一件事：控制器控制的点确实是 URDF 的 `left_flange`/
     `right_flange`；若不同，在 `[robot.mounting_overrides]` 填实测矩阵覆盖。
   - 两个终端验证 socket 路径——`run_catch.py --profile bench` 与
     `fake_mocap_sender.py --csv data/box_flying_csv/2.csv`（sender 会先做
     逆链变换，bench 走的是真实输入链）；确认日志里 `catch_decision`、
     `planning_time_ms`、指令步长合理（参考值：规划 <15 ms，
     120 Hz（8.3 ms 周期）下单步 ≤14 mm）。
6. **低速实接**：降低释放高度/初速、软地面、专人监护；每次投掷的
   `output/attempt_*/` 留档。2026-09-30 起 live 尝试同步记录 `/joint_states`
   实测反馈（FK 成 base_link 掌位姿 + 指令-实测跟踪摘要入 trace），
   `check_replay_tracking.py` 可直接分析任意一次尝试。首次判据是"双侧接触窗口按预测出现"，不是"接住"。

## 注意事项（容易踩的坑）

- **内部坐标系恒为 base_link，torso_flange 只在两个边界出现**：规划、任务几何、
  等待位姿、安全盒、回放数据的数值与仿真完全一致，规划代码零改动。动捕入口
  一步换算 `T_BG = T_base_torso @ T_FM @ T_MD @ T_DG`（`calib.FrameChain`，
  标定值 T_FM 与固定常量 T_base_torso 分开保存、分开 debug）；指令出口在
  `palm_targets_to_tcp` 单点完成 base→torso→法兰的换算后发布。T_base_torso
  是锁定 legwaist 准备姿态下的 URDF FK 常量（当前纯平移
  `(0, +0.0236, +1.2024) m`，旋转恒等）；改 legwaist 锁定角或准备姿态后用
  `compute_palm_frames.py` 重算回填，入口出口同时生效。
- **坐标系只有两个入口**：观测在 `FrameChain` 处一次性转入 base_link，指令在
  `palm_targets_to_tcp` 处一次性转出。改外参只改 `frames.toml`；
  `check_frames.py --chain-check` 的抛物线不变性测试（含非平凡 T_base_torso）
  专防旋转方向/符号装反——先验方向错一个符号等于重力反向，是最危险的
  单一故障，换 T_FM 后必跑。
- **回放必须实时步进**：规划器的重定时假设观测钟与墙钟同速。曾实现的
  "--fast" 快进模式会导致观测年龄为负、retiming 崩坏，已移除；测试多花的
  几秒是买正确性的。
- **`commit_timeout_s = 0.55` 的语义**：从"检测到运动"起算，真实投掷含手持上摆
  ~0.15 s；仿真的 0.40 s 是从指令释放起算。若改进释放检测（竖直速度峰值规则）
  可收回 0.40。
- **控制器只收位姿**：规划器输出的速度不进控制器，只进日志；接触后行为由
  控制器刚度决定，仿真 QP 的法向优先保护没有实物对应物——这是低速起步的原因。
- **掌→法兰变换按"掌目标系"标定（勿改成 hand-link 系）**：规划器输出的位姿是
  自定义的掌目标系——原点在 12 球阵列前切平面中心（左 hand 系 y=−50 mm、
  右 y=+40 mm 处的切平面，2026-09-28 修订后球阵沿 Y 移了 ±10 mm），轴向为
  hand-link 轴。`robot.toml` 的 `T_tcp_palm` 存 **^掌目标系 T_flange**（左平移
  (−0.12, +0.06, 0)、右 (−0.12, −0.06, 0) 加 90° 安装旋转），由
  `compute_palm_frames.py` 从 URDF 固定关节与切平面偏置计算；
  `palm_targets_to_tcp` 按存储方向直接使用（不再取逆）。上机前核对控制器
  控制点确为 URDF 的 `left_flange`/`right_flange`；改准备姿态后重跑脚本回填。
- **干净尝试不应触发 safety 钳位**：trace 里出现 `command_clamped` 非空，
  先查配置/标定，不要带病运行。
- **提交后不重新规划**：一次规划是设计前提；提交后动捕仍被记录但只用于
  事后残差分析。任何桥接异常进入 FAULT 并保持最后指令。
- **版本纪律**：算法层与 MozBoxer 源树按 PROVENANCE.md 钉定；MozBoxer 侧
  工作树提交后回填 sha。改动核心层时两侧同步，避免仿真对照失真。

## 常用调参入口（`config/catch.toml`）

- 提交平面 `mission.commit_plane_y_m`（默认 −1.05；越晚提交预测越准、
  留给执行的越少）；接触平面 `planning.overrides.plane_y`（基线 −0.58）。
- 预测窗口/先验/正则：`[prediction]`；掌速与加速度预算、闭合节奏、回退距离：
  `[planning.overrides]`（任意 `CatchSettings` 字段，未知字段报错）。
- 准备姿态（等待位姿、legwaist 常量、掌→法兰安装常量的共同来源）：
  `robot.toml [robot.posture]` 的关节角，改完即全量生效（加载时 FK 推导）。
- 实验分层用 `config/profiles/*.toml`（深合并覆盖基线），不要复制整份配置。

## 目录速览

```
config/          四个 TOML + profiles（全部可调参数在此；内部一律 base_link 系，与仿真一致）
src/moz1_catch/
  core/          移植的预测与规划（勿改语义；改动需与 MozBoxer 同步）
  calib.py       入口变换链 T_BG = T_base_torso·T_FM·T_MD·T_DG、时钟、先验换算
  feedback.py    /joint_states 反馈记录、FK 实测掌位姿、指令-实物跟踪滞后分析
  mocap/         UDP 监听（解析器占位）与 CSV 回放
  kinematics.py  URDF FK：关节角 → 等待位姿/T_base_torso/T_tcp_palm（加载时推导）
  robot/         ROS2 发布（单点适配）、掌→法兰换算与 mock sink
  runtime.py     状态机；executor.py 指令流；safety.py 钳位；trace.py 记录
scripts/         run_catch / start_catch / move_to_ready / replay_sim_plan / check_replay_tracking / dry_run_replay / check_frames / compute_palm_frames / fake_mocap_sender
test/            68 项测试（含仿真规划回放 sim_plan 的 8 项、反馈记录/跟踪分析与绘图 7 项）
data/            五条实物自由飞行记录 + moz1_boxer.urdf（FK 用）+ sim_plans/（冻结的仿真规划，回放用）
docs/            deployment_plan.md（设计文档）、sim_plan_replay_guide.md（仿真规划实物回放操作指南）
output/          每次尝试的 trace（git 忽略）
```
