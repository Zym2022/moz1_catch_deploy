# 实物日志三维回放

生成后的页面位于 `output/real_robot_replay/index.html`。
直接用近期 Chrome、Edge 或 Firefox 打开 HTML；数据、URDF 网格与渲染库全部内嵌，离线可用。
网页与实验日志均不纳入 Git。

## 生成

在部署仓库根目录执行：

```bash
.venv/bin/python scripts/replay_real_robot.py --self-check
```

默认读取本仓库 `output/attempt_*/`，网格读取本仓库 `data/meshes/`。
输出 `output/real_robot_replay/index.html`。也可以指定新的日志根目录或单条试次：

```bash
.venv/bin/python scripts/replay_real_robot.py /path/to/attempt_or_log_directory \
  --output output/my_replay.html --self-check
```

依赖仅为项目已有的 NumPy 和 SciPy。导出过程不启动 ROS，不重新规划，也不发布机器人指令。
导出时检查日志 URDF 哈希；`--self-check` 逐帧核对反馈关节 FK 与日志掌位姿。
页面加载时也会检查浏览器 FK 的位置与四元数，避免两端坐标实现不一致。

## 开发机与机器人主机通过 GitHub 同步

两台机器使用同一个部署仓库。Git 同步回放脚本、HTML 模板、渲染库、URDF、
`data/meshes/` 中的 20 个机器人网格，以及 12 球＋刚芯常数。
不需要安装或克隆 MozBoxer 仿真项目，不需要 ROS 即可生成和查看回放。

在要查看实验的那台机器上，进入部署仓库：

```bash
git pull --ff-only
.venv/bin/python scripts/replay_real_robot.py output/attempt_日期_时间_accept \
  --output output/real_robot_replay/index.html --self-check
xdg-open output/real_robot_replay/index.html
```

需要查看本机所有试次时，省略 `output/attempt_日期_时间_accept` 参数即可。
首次在新电脑上准备 Python 环境：

```bash
uv venv
uv pip install -e .
```

已有环境的机器人主机直接使用现有 `.venv`。日志保留在实验所在机器的 `output/`；
生成 HTML 也保存在 `output/`，两者均被 `.gitignore` 排除，`git pull` 不会带来另一台机器的实验记录。
只需要把另一个机器上的某条实验日志带过来时，再手动复制该试次目录。

## 查看新的实物实验

每次实验保留整个 `attempt_日期_时间_accept` 或 `attempt_日期_时间_reject` 目录，
包含同一次实验的 `trace.npz` 和 `meta.json`。日志根目录直接包含这些试次目录。
实机默认保存到启动仓库的 `output/`，实际位置见终端的 `catch_trace_dir=...`；
如果实验在另一台机器上，把这些完整目录复制到本机后再生成。

如果数据保存在原来的 MozBoxer `real_robot_log` 目录，显式传入该目录：

```bash
.venv/bin/python scripts/replay_real_robot.py \
  ~/workspace/MozBoxer/source/MozBoxer/MozBoxer/catching/box_flying_data/real_robot_log \
  --self-check
```

随后刷新或重新打开 `output/real_robot_replay/index.html`，在右上角选择新试次。
**HTML 是导出时的数据快照，不会自动读取后来新增的日志。**

如果数据保存在本仓库默认的 `output/`：

```bash
.venv/bin/python scripts/replay_real_robot.py output \
  --output output/real_robot_replay/latest.html --self-check
xdg-open output/real_robot_replay/latest.html
```

如果只看一次实验：

```bash
.venv/bin/python scripts/replay_real_robot.py /path/to/attempt_日期_时间_accept \
  --output output/real_robot_replay/one_attempt.html --self-check
```

新实验必须有 `feedback_joint_left_rad`、`feedback_joint_right_rad`、`feedback_t_s`
才能显示实际双臂动作；当前 live 记录器已经保存这些字段。
若更换了 URDF，用 `--urdf /path/to/the_recorded_robot.urdf` 指向对应版本，
脚本会核对其哈希，避免使用不同关节模型重建姿态。

## 查看方法

1. 在右上角选择试次，先看 `204044_accept`；点击“计划接触”再选“接触特写”。
2. 黄色实体箱为动捕箱体，紫色线框为提交时冻结的自由飞行预测。
3. 机器人双臂由实测关节角还原。蓝色为左掌，粉色为右掌；每掌显示 **12 个半径 30 mm 的软层球＋1 个 140 × 20 × 70 mm 灰色刚芯**，线框手掌是实际下发的目标。
4. 拖动时间轴，或按“上一 / 下一反馈帧”。帧按钮定位原始反馈时间；默认关节插值便于观察，关掉即可保持前一帧角度。
5. 下方图可选择掌 X/Y/Z、掌间距、位置误差，以及箱心动捕 / 预测对照与残差；实测曲线上的点为原始样本。点击图也可以定位时间。右图可查看 7 个关节的双臂角度。
6. “完整记录”包含投掷前的准备过程。整体、正面、侧面、俯视和鼠标旋转用于查看箱面与掌面的朝向。

## 海绵与时间

- 回放按仿真 `palm_coating.py` 的运行时几何替换部署 URDF 中旧的 6 球显示。左 / 右球阵与刚芯中心在 hand-link 系分别为 `(0.100, -0.020, 0)` / `(0.100, 0.010, 0)`；球阵沿 X 为 4 列、沿 Z 为 3 行。
- 12 球软层与刚芯属于设计几何，不是实物海绵的精确测量。刚芯与球阵均附着于 hand-link；指令模型换算到掌目标系后与实际模型保持同一几何。
- **额外海绵示意默认关闭**；需要时可启用，按用户估计设为每侧 **25 mm**、覆盖 **180 × 120 mm**，尺寸均可调整。
- 海绵从 URDF 的掌目标切平面向内法向延伸。不模拟压缩，尺寸未标定，不能据图中穿插判断碰撞或接住。
- 图中 **37.8 cm** 是用户提供的“实物可能已夹紧”的参考。不能再用它与裸模型的 30.1 cm 比较，直接认定实物没有夹紧。
- 时间零点为冻结轨迹的调度起点；reject 为提交观测。不是机器人已物理起动的时刻。
- 关节反馈时间为主机接收时间，约 20 Hz。页面显示插值所用的原始采样区间。
- “反馈时间偏移”默认 0，负值使反馈提前；手动偏移只供比较，不能据此认定具体控制器或机械滞后。

## 能还原到什么程度

晚间 4 条 accept 可以还原双臂关节与箱体姿态；下午 `170015_accept` 未记录关节反馈，页面只显示固定准备姿态并提示缺失。
腿腰没有实测反馈，使用每条 `meta.json` 保存的固定姿态。拒接记录在决策后结束，不能补出后续真实运动。

日志没有规划关节角：规划结果用掌目标轨迹显示，不能把 IK 生成的关节姿态冒充当时规划结果。
冻结箱预测在受力后不再符合自由飞行条件。法向偏角比较的是计划选定箱侧面，并非测得的实际碰撞点法向。
没有接触力、同步视频和控制器内部参考，回放不能补出这些缺失信号。
