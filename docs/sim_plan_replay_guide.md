# 仿真规划轨迹实物回放操作指南

**目的**：把仿真中一次真实接箱规划的结果（`data/sim_plans/*.npz` 里的
`target_palm_*` 规划目标流，非仿真执行记录 `actual_palm_*`）在**无箱**条件下
按原时间轴回放给真实机器人，验证部署包执行输出链路的正确性：帧系换算
（base_link → torso_flange 法兰位姿）、`mx_mix_command` 消息构造、120 Hz
实时节拍、安全钳位、URDF/FK 推导的等待位姿。

**验证不到的部分**（由其它手段覆盖）：mocap UDP 实时输入与 commit 时机
（`dry_run_replay.py` / bench profile）、真实接箱动力学（后续带箱实验）。

**原理**：规划结果是连续时间函数（`CatchPlan`，可任意频率采样）；仿真 trace
以 1 kHz 密集采样存档。回放脚本在每个指令时刻（120 Hz，控制器标准频率，
2026-09-29 与控制器开发人员确认）对 1 ms 采样做线性插值 + 姿态 Slerp 取值，
经安全钳位与掌→法兰换算后发布——与实况接箱走**完全相同**的执行代码路径。

---

## 1. 前置条件

- [ ] 代码已同步到机器人主机，且 `.venv/bin/python -m pytest -q` 全绿
      （环境按 README「迁移步骤」第 1 步：`uv venv --system-site-packages`）。
- [ ] 机器人主机 shell 已 source ROS 与 movax_interface，并 export 域号
      （**每个新开的终端都要重新执行一遍**，source 结果不跨终端）：
      ```bash
      source /opt/ros/humble/setup.bash
      source ~/ros_pkg/movax_interface/install/setup.bash
      export ROS_DOMAIN_ID=33
      ```
- [ ] source 后自检 rclpy 与消息包在本仓库 venv 内可见（rclpy 的路径由
      setup.bash 写入 PYTHONPATH，不 source 就不可见）：
      ```bash
      .venv/bin/python -c "import rclpy, mc_core_interface.msg; print('ros bridge OK')"
      ```
      报 `ModuleNotFoundError: No module named 'rclpy'` 见第 4 节排查表
      第一行。注意命令**不要带任何 `PYTHONPATH=...` 前缀**：前缀会整体
      覆盖 source 得到的 ROS 包路径（本包为可编辑安装，不需要前缀）。
- [ ] **LegWaist 已锁定在准备姿态** `[0, 60, -90, 30, 0, 0]°`（±1°）。
      脚本不控制腰腿、只检查；不在位会拒绝启动。
- [ ] 机械臂上电、控制器运行中；**首次运行加 `--enable-outer-ctrl`**
      （未使能外部控制时 mc_core 会忽略 mix 指令，表现为"发了但不动"）。
- [ ] 双手工作空间内无人、无箱、无障碍物；末端活动区域净空
      （轨迹范围：x ±0.21 m、y −0.63 → −0.42 m、z 0.98 → 1.22 m 附近，
      含撤退段）。
- [ ] 有人在机器人旁边值守，可随时按急停。

## 2. 操作步骤（在仓库根目录执行，逐级推进）

```bash
cd ~/workspace/moz1_catch_deploy
```

**第 0 步 · 空跑（mock sink，不碰硬件，可先在开发机做）**

```bash
.venv/bin/python scripts/replay_sim_plan.py \
    data/sim_plans/final_nominal_120hz_200ms.npz --dry --hold-s 0.2 --no-hold-final
```

判据（应逐项满足）：

| 输出行 | 期望值 |
| --- | --- |
| `decision` | `accept` |
| `wait pose check` | `0.0 mm / 0.00 deg` 量级（>10 mm / >5° 会告警） |
| `commands (period p50 ...)` | p50 ≈ 8.3 ms（120 Hz），p95 与 p50 差 <0.5 ms |
| `clamp violations` | `0` |
| `streamed speed` | ≈ 1.42 m/s（1× 时的指令隐含峰值） |

**第 1 步 · 只看移动计划（不动机器人）**

```bash
.venv/bin/python scripts/replay_sim_plan.py \
    data/sim_plans/final_nominal_120hz_200ms.npz --dry-run-approach
```

打印当前/目标关节角、增量、时长（峰值关节速度 ≤0.25 rad/s，时长 ≥4 s），
确认合理后进入下一步。

**第 2 步 · 半速实机首跑（推荐）**

```bash
.venv/bin/python scripts/replay_sim_plan.py \
    data/sim_plans/final_nominal_120hz_200ms.npz --speed-scale 0.5 --enable-outer-ctrl
```

流程：关节空间慢速移到准备姿态（约 4 s+）→ 到达校验（±1°，不达标自动中止）
→ hold 等待位姿 1 s → **半速**回放约 2.2 s → 保持终位姿。

观察要点：双手是否先平稳合拢到等待位姿；回放段运动方向是否为"向前迎箱
→ 两侧夹合 → 上提撤退"的形状；有无卡顿、抖动、明显滞后。结束按 Ctrl+C
退出 hold（控制器保持最后位姿）。

**第 3 步 · 全速回放**

```bash
.venv/bin/python scripts/replay_sim_plan.py \
    data/sim_plans/final_nominal_120hz_200ms.npz --enable-outer-ctrl
```

与第 2 步相同，速度 ×1（指令隐含峰值掌速 1.43 m/s，安全钳位上限 3.5 m/s，
规划预算 3.0 m/s）。重点观察控制器对 120 Hz 位姿流的跟踪：末端实际速度能
否跟上、滞后是否稳定——这是本次回放要回答的核心问题之一。

**常用开关**：`--skip-approach`（机器人已在准备姿态，跳过移动阶段）；
`--hold-s <秒>`（回放前 hold 时长，默认 1 s）；`--no-hold-final`（回放完
直接退出不保持）；`--speed-scale <比例>`（时间缩放，路径形状不变）。

## 3. 结果检查

每次运行落盘 `output/attempt_<时间戳>_replay/`：

- `trace.npz`：与仿真 trace 键名对齐（`target_palm_position` 为实际发送的
  掌目标、`command_tcp_position` 为换算后的法兰指令、`command_clamped`
  为钳位标记、`command_phase` 为 hold/execute/done 阶段）。
- `meta.json`：配置快照、关节角、本次回放的元信息（源文件、speed scale 等）。

**与仿真参考对比**（量化判据：位置 ≤0.1 mm、姿态 ≤0.01° 量级）：

```bash
.venv/bin/python - <<'EOF'
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from moz1_catch.sim_plan import load_sim_trace_plan

_, info = load_sim_trace_plan("data/sim_plans/final_nominal_120hz_200ms.npz")
i0 = info["plan_start_index"]  # 加载器自动检测的规划起点采样索引
attempt = sorted(Path("output").glob("attempt_*_replay"))[-1]
rep = np.load(attempt / "trace.npz", allow_pickle=True)
sim = np.load("data/sim_plans/final_nominal_120hz_200ms.npz")
t_plan, streamed = rep["command_t_plan_s"], rep["target_palm_position"]
sim_pos, sim_quat = sim["target_palm_position"], sim["target_palm_rotation_xyzw"]
err_p, err_r = [], []
for k in np.where(rep["command_phase"] == "execute")[0]:
    idx = i0 + t_plan[k] * 1000.0
    lo, frac = int(np.floor(idx)), idx - int(np.floor(idx))
    ref = (1-frac)*sim_pos[lo] + frac*sim_pos[min(lo+1, len(sim_pos)-1)]
    err_p.append(np.abs(streamed[k]-ref).max())
    r = Rotation.from_quat(sim_quat[lo])
    err_r.append((Rotation.from_quat(rep["target_palm_rotation_xyzw"][k]) * r.inv()).magnitude())
print(f"max position error: {max(err_p)*1000:.3f} mm")
print(f"max rotation error: {np.degrees(np.array(err_r).max()):.3f} deg")
print(f"clamp violations  : {sum(len(c) for c in rep['command_clamped'])}")
EOF
```

## 4. 故障排查

| 现象 | 原因与处理 |
| --- | --- |
| `ModuleNotFoundError: No module named 'rclpy'` | 按序排查：① 当前终端没 source ROS——rclpy 的路径由 `setup.bash` 写入 PYTHONPATH，每个新终端都要重新 source（见前置条件自检）；② 命令带了 `PYTHONPATH=src` 之类前缀——整体覆盖了 source 得到的路径，去掉前缀重跑（本包可编辑安装，任何前缀都不要加）；③ `.venv` 是隔离环境（在机器人主机跑过 `uv sync`）——`rm -rf .venv` 后按 README 迁移步骤第 1 步用 `uv venv --system-site-packages` 重建。 |
| `wait pose check` 超 10 mm / 5° | 帧系或 URDF 与仿真不一致（T_base_torso、T_tcp_palm、关节角被改过）。跑 `compute_palm_frames.py --check-base` 复核，先解决再上机。 |
| `clamp violations` 非零 | 轨迹超出工作区盒或单步超速：先确认没有手改 `[safety]`/`[execution]`，再看是否加载了非 accept/异常的 trace。干净尝试必须为 0。 |
| 机器人完全不动 | 外部控制未使能（加 `--enable-outer-ctrl`）；或 shell 没 source ROS/movax_interface；或 `ROS_DOMAIN_ID` 不是 33。 |
| 卡在 `waiting for /joint_states` | 域号不对或反馈话题不通；确认 `ros2 topic echo /joint_states` 有输出。 |
| phase A 中途中止（tracking error） | 关节跟踪偏差超 0.2 rad——降低 `--max-joint-speed`、检查是否有机械干涉。 |
| 回放段末端明显滞后/跟不上 | 控制器跟踪能力问题（这正是要观察的）：记录现象，回退半速；必要时与控制器开发人员确认 120 Hz 位姿流的带宽。 |
| `cannot locate the plan segment start` | 加载的 npz 不是标准 trace（缺关键键或时间轴异常），用仿真侧重新导出。 |

## 5. 安全备忘

- 无箱回放的 follow/撤退段是**开环**的（无接触力反馈），行为不代表带箱表现。
- Ctrl+C 在任意阶段都是安全的：移动阶段立即停发（控制器保持最后指令）；
  笛卡尔阶段保持最后位姿（FAULT 纪律），机械臂不会回弹。
- 每次实机运行前确认 `output/` 上一次的 trace 无 `clamp violations` 再提速。
- 换用其它仿真 trace：任何 accept 的 catch trace npz 放进 `data/sim_plans/`
  即可（加载器自动识别三种时间约定；建议先 `--dry` 看 `wait pose check`）。
