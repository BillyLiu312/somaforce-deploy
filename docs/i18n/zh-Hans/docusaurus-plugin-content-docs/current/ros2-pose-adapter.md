---
title: ROS2 位姿适配器
sidebar_position: 3
---

# ROS2 位姿适配器

HDMI suitcase policy 不直接订阅 ROS2，而是读取旧的 ZMQ 位姿 ABI：

```text
pelvis   -> tcp://127.0.0.1:5555
suitcase -> tcp://127.0.0.1:5561
payload  -> float32 [x, y, z, qw, qx, qy, qz]
```

在和 policy 相同的电脑上运行适配器：

```bash
source /opt/ros/humble/setup.bash
source /home/irmv/catkin_vr/install/setup.bash
export PYTHONPATH=/opt/ros/humble/local/lib/python3.10/dist-packages:${PYTHONPATH:-}
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  .venv/bin/python scripts/ros2_pose_to_zmq.py
```

标定后，如果需要让 ZMQ 位姿表达在标定坐标系中，可传入生成的共同世界变换：

```bash
.venv/bin/python scripts/ros2_pose_to_zmq.py \
  --transform-json calibration/suitcase_pose.json
```

VRPN/NOKOV 通常发布毫米单位的位置。适配器默认使用
`--position-scale 0.001`；只有确认上游已经发布米时才使用
`--position-scale 1.0`。ROS 四元数顺序是 `x,y,z,w`，而仓库使用
`w,x,y,z`；适配器会完成重排和归一化。

连接 policy 前先检查源 topic：

```bash
ros2 topic type /suitcase/pose
ros2 topic type /robot_g1/pose
ros2 topic hz /suitcase/pose
ros2 topic hz /robot_g1/pose
ros2 topic echo --once /suitcase/pose
ros2 topic echo --once /robot_g1/pose
```

不依赖 ROS 或真机验证转换：

```bash
.venv/bin/python scripts/ros2_pose_to_zmq.py --self-test
```

带 stale watchdog 的 bring-up 命令：

```bash
.venv/bin/python scripts/ros2_pose_to_zmq.py \
  --stale-timeout 0.25 --exit-on-stale
```

## 标定

1. 保证机器人和 suitcase 两个刚体都在动捕视野内并静止，记录
   `/suitcase/pose` 和 `/robot_g1/pose` 10--20 秒。
2. 确认 `frame_id`、单位、四元数约定和更新频率。房间尺度坐标如果约为
   `200`，通常表示毫米；进入 policy 前必须变成 `0.200` 米。
3. 测量 G1 pelvis 刚体的 marker cluster 原点相对 pelvis 参考点的固定偏移。
   policy 中的 `pelvis` 必须代表 pelvis 参考点，而不是 marker cluster 原点。
   这个固定偏移应在动捕刚体模型或适配器中明确处理。
4. 把 suitcase 放到训练动作的初始相对位置：大约在 pelvis 前方
   `0.532 m`，横向偏差接近零，yaw 差约 `2 度`。策略抓取目标相对
   suitcase 的局部偏移是 `(-0.10,+0.18,0.25)` 和
   `(-0.10,-0.18,0.25)` 米。
5. 检查 ROS 数据计算出的相对位姿，而不是只看两个绝对位姿。将 suitcase
   沿一个轴移动 10 cm，确认相对位置沿预期轴变化 10 cm；旋转 90 度，确认
   heading 向量也按预期旋转。
6. 不要在 policy 内静默交换坐标轴。如果动捕世界坐标轴不同，应对 suitcase
   和 pelvis 同时施加一个有文档记录的刚体变换，然后重新做相对位姿检查。
7. 发送电机命令前，让适配器和 ZMQ 位姿可视化运行至少一分钟，确认两路数据
   有限、没有超过 stale timeout，且没有 rejected message。

现场标定助手会在机器人和 suitcase 静止时采集配对 ROS 样本：

```bash
.venv/bin/python scripts/calibrate_ros2_pose.py \
  --samples 120 \
  --relative-position 0.532 0.0 -0.793 \
  --relative-yaw-deg 0 \
  --output calibration/suitcase_pose.json
```

默认目标含义是：pelvis 为标定原点，suitcase root 在前方 0.532 m、横向偏差为
0，两个 yaw 基本平行。现场尺寸不同就替换这些值。解算器会输出共同的
`world_from_mocap` 矩阵和残差；两刚体相对残差过大表示标定失败，不能把矩阵直接接受。

如果 marker 是任意粘贴的，可以提供通过测量或 CAD 得到的
marker frame 到 policy frame 的固定矩阵：

```json
{
  "pelvis": [[1,0,0,0.02], [0,1,0,0], [0,0,1,0.10], [0,0,0,1]],
  "suitcase": [[1,0,0,-0.10], [0,1,0,0], [0,0,1,0.20], [0,0,0,1]]
}
```

矩阵约定为 `T_marker_policy`，表示 policy frame 的原点和轴在 marker frame
中的表达。如果测量得到的是实体坐标中的 marker 位姿 `T_policy_marker`，放入
文件前先对矩阵求逆。下面的工具可以根据实测米制位置和 XYZ 欧拉角自动求逆：

```bash
.venv/bin/python scripts/make_marker_to_policy.py \
  --pelvis-marker-position <px> <py> <pz> \
  --pelvis-marker-rpy-deg <roll> <pitch> <yaw> \
  --suitcase-marker-position <sx> <sy> <sz> \
  --suitcase-marker-rpy-deg <sroll> <spitch> <syaw> \
  --output calibration/marker_to_policy.json
```

位置单位为米，表示 marker 原点在真实实体 policy frame 中的坐标；RPY 表示
marker 轴在同一个实体坐标系中的方向。如果 suitcase policy 三轴与动捕
world 三轴一致，则这里直接填写当前 ROS marker 姿态对应的 `T_policy_marker`
角度，不要先取反。工具会对完整旋转求逆，生成 adapter 使用的
`T_marker_policy`；不要先把 RPY 取反再让工具求逆，否则会反两次。然后运行：

如果不使用工具而直接手写矩阵，才使用逆旋转：
`R_marker_policy = R_policy_marker.T`，或对四元数取共轭。对 XYZ Euler 的
三个分量逐一取负一般不等于完整旋转的逆，因为逆运算会反转乘法顺序。

```bash
.venv/bin/python scripts/calibrate_ros2_pose.py \
  --marker-to-policy-json calibration/marker_to_policy.json \
  --output calibration/suitcase_pose.json
```

生成文件会同时记录两个矩阵和共同世界变换。适配器实际应用的变换为：
`T_calibration_mocap * T_mocap_marker * T_marker_policy`。

适配器只负责位姿传输。G1 关节状态仍由
`scripts/g1/real_bridge.py` 通过 Unitree DDS 提供，位姿流不能替代
low-state 输入。

## 只使用 marker 的 MuJoCo viewer

如果不想依赖 Unitree 状态，可以使用实时、无动力学的 viewer：

```json
{
  "marker_from_torso": [[...], [...], [...], [0, 0, 0, 1]],
  "marker_from_suitcase": [[...], [...], [...], [0, 0, 0, 1]]
}
```

矩阵分别是 `T_marker_torso` 和 `T_marker_suitcase`，把 policy frame 坐标转换
到对应 marker frame。运行：

```bash
MUJOCO_GL=egl .venv/bin/python scripts/render_live_marker_policy_mujoco.py \
  --torso-topic /robot_g1/pose \
  --suitcase-topic /suitcase/pose \
  --calibration calibration/marker_policy.json
```

viewer 冻结所有关节，只应用两个 marker 变换并调用 `mj_forward`。它不依赖
`rt/lowstate`、腰部 FK、DDS 或电机命令。

## 由 torso 反算 pelvis

如果低位 pelvis 刚体难以稳定追踪，可以使用 torso marker 和 ZMQ low-state
中的实时腰部角度：

```bash
.venv/bin/python scripts/ros2_pose_to_zmq.py --suitcase-only
.venv/bin/python scripts/ros2_torso_to_pelvis.py \
  --torso-topic /robot_torso/pose \
  --torso-from-marker-json calibration/torso_from_marker.json \
  --stale-timeout 0.25 --exit-on-stale
```

JSON 通常保存 `T_torso_marker`，即 marker 在 torso frame 中的位姿。运行脚本
也接受更适合直接测量的 `marker_from_torso`，即 torso 原点在 marker 坐标中的
表达；如果提供该字段，脚本会在内部求逆。
solver 会
在内部求逆，从 low-state 读取 `waist_yaw`、`waist_roll`、`waist_pitch`，按
MJCF 精确 FK 求 pelvis，并发布到 `5555`。MuJoCo 和真机输入共用同一个
`TorsoToPelvisFK` 核心；不要把实时腰角替换成固定初始角度。

torso frame 原点是 G1 腰部 `waist_pitch_joint` 转轴的中心，不是胸部外壳几何
中心。按照 MJCF，在三个腰角为 0 时，它相对 pelvis 是
`[-0.0039635, 0, 0.044] m`。在真实机器人上应找到腰部水平 pitch 轴，用该
转轴中心作为 torso 原点。torso mesh 从这个点向上延伸，视觉/几何中心不能
作为 policy frame。

## Suitcase 真机 bring-up

确认 G1 网卡、VRPN 服务和标定后，使用按依赖顺序启动的脚本：

```bash
# 有界只读检查；不会创建 rt/lowcmd 或 MotionSwitcher。
bash scripts/run_suitcase_hardware.sh --check-only

# 交互式 zero/hold/init 控制；释放模式前必须手动输入 ARM。
bash scripts/run_suitcase_hardware.sh --armed
```

G1 网卡名不同时增加 `--interface <name>`。脚本会先启动 VRPN，等待两路
tracker topic，再启动 G1 bridge 和 pose relay，最后检查 `5555`、`5561`、
`5590`。armed 启动时，bridge 会保持未武装，直到本地控制器产生一帧带新鲜
时间戳的 hold command，之后才允许释放机器人模式。

armed 提示符只接受：

- `z`：使用 HDMI PD gains 的零策略/current-follow target；
- `h`：锁定并保持当前关节位置；
- `i`：用 10 秒平滑插值到 suitcase motion 第 0 帧姿态；
- `q`：回到 zero 模式并停止整套进程。

基础 `--armed` 启动器不加载 suitcase ONNX，也不存在 policy 模式。

下一阶段需要显式启用 nominal shadow：

```bash
bash scripts/run_suitcase_hardware.sh --armed --nominal-shadow
```

先输入 `i`；启动器会等待 10 秒初始化并在完成时明确提示，然后再输入 `s`。
输入 `s` 前，把 suitcase 放到 motion 首帧关系。脚本会检查 pelvis-yaw frame
中的完整 XY 相对向量（不只是距离）、约 `-0.793 m` 的相对高度和 `1.95 deg`
的相对 yaw；默认 pelvis-yaw-frame XY 容差为 `0.20 m`，超差时直接拒绝运行。
safe controller 会继续保持
初始化姿态，同时使用官方 HDMI observation/history 实现，以 50 Hz 运行 student
ONNX 472 步。shadow 进程自身不会绑定或写入 low-command 端口。它会在当前
`outputs/suitcase_hardware/<timestamp>/` 目录记录三组 ONNX 输入、nominal action
与 target、机器人状态、校正位姿、推理时延、OOD ratio、关节限位余量和数据
新鲜度。增加 policy 控制模式前，必须先审查生成的
`nominal_shadow_*.summary.json`。

## 受保护的 100% nominal 实机运行

nominal authority 提升到 100% 后仍通过 guard 运行：

```bash
bash scripts/run_suitcase_hardware.sh --armed --nominal-apply
```

HDMI 进程只在本机 `5594` 端口发布 proposal，不能占用或写入 low-command
端口 `5591`；唯一命令控制器使用 policy-faithful direct 模式：nominal
`q_target` 按元素原样写入 G1 low command，沿用上游 HDMI gains，并保持
velocity target 和 feed-forward torque 为 0。不再做 authority blend、关节目标
位置裁剪或 target slew limit。这是必要的，因为 HDMI 输出的是虚拟 PD setpoint；
策略需要边界力矩时，setpoint 可以有意超出机械关节范围。proposal 非有限或超时、
low-state stale、sequence 逆序，或观测关节速度超过 `12 rad/s` 时，仍会切换为
当前关节位置 hold。`180 deg` 设置实际关闭了绝对倾角 abort。

操作顺序：

1. 机器人进入调试模式，布置物理防摔保护，操作员全程手持宇树遥控器，并清空
   机器人运动范围内的人员和障碍物。第一次 pilot 不要把 suitcase 固定在机器人
   手上，也不要实际提起重物。
2. 运行上述命令，在第一次确认时输入 `ARM`。
3. 输入 `i`，全程观察 10 秒的 motion 第 0 帧初始化。平衡、脚底或关节表现异常
   时立即输入 `h`，或使用遥控器急停。这个姿态与之前 YAML default init 的部分
   关节最大相差约 `0.48 rad`。
4. 将 suitcase 放到已验证的第 0 帧相对位置，然后输入 `p`。runner 会先检查
   low-state、两路校正位姿、init 误差（`<=0.50 rad`）和物体位置，随后才给出
   第二道确认。
5. 输入 `FULL` 后，100% policy 开始闭环稳定控制，但 motion reference 固定在
   frame 0。确认机器人已经依靠策略自身站稳，再完全解除安全绳；这个过程不会
   reset policy。如果此阶段 marker 被遮挡，运行不会立即退出：最多 30 秒内
   runner 会维持最后一帧 corrected pose，在冻结的 motion frame 0 上继续 policy
   inference 和 history 更新，并提示 `MARKER OCCLUDED`；corrected pelvis 或
   suitcase pose stale 时会拒绝 `GO`。新鲜位姿恢复后会提示 `MARKER RESTORED`，
   不 reset policy，直接切回实时 pose 输入。
6. 安全绳完全离开运动范围后输入 `GO`，此时才开始推进 472 帧 reference，约持续
   9.5 秒。稳定等待和动作执行期间都可输入 `h` 并回车来停止 policy control 并
   进入 hold。如果运动不稳，不要等待终端输入，直接使用实体遥控器。
   正式动作中仅 pelvis marker 丢失时允许最多 2 秒 grace：inference 和 reference
   会使用最后一帧 pelvis pose 继续推进，使深弯动作有机会穿过遮挡区；pelvis
   恢复后自动切回实时数据。suitcase 遮挡只有在 reference contact 已生效且 Vicon
   确实测到箱子抬升至少 5 cm 后才会启用 fallback。runner 会锚定最后一次实测的
   pelvis-to-suitcase 变换，使用 reference 中的相对运动继续推进，同时通过实时
   pelvis 跟随机器人在 world 中的运动。连续 60 ms 没有 suitcase 新样本就开始
   预测，早于通用的 250 ms stale gate，避免观测先冻结多个 policy frame 再跳变。
   终端提示
   `SUITCASE MARKER OCCLUDED AFTER CONFIRMED LIFT` 后，inference 会一直运行到
   Vicon 恢复或 motion 结束。重新识别时直接切回 corrected live pose，并记录位置
   和姿态重捕获误差。实际抬升得到确认之前 suitcase stale 仍会终止 motion，从而
   避免抓取失败时把仍在地面的箱子错误地绑定到机器人。
7. 正常完成后启动器会自动发送 `h`。确认机器人静止且已有支撑后，再输入 `q`。

policy 异常退出或 safe controller 报告 abort 时，启动器会 fail closed。raw policy
记录位于 `nominal_apply_*.npz` 和 `nominal_apply_*.summary.json`；唯一命令控制器
看到的 raw target 与实际 applied target 位于 `nominal_pilot_applied.jsonl`；其余
进程日志位于同一 `outputs/suitcase_hardware/<timestamp>/` 目录。进程正常退出只
能证明传输和异常 gate 完成运行；再次运行前仍需检查平衡、接触、target tracking
和 applied log。
inference NPZ 额外记录 `phase`（`stabilize`、`stabilize_pose_grace`、
`motion_pose_grace`、`motion_object_fallback` 或 `motion`）以及各 grace counter。
`suitcase_pose_source` 标记 `live`、`held_last` 和 `reference_attached` 输入；
`suitcase_pose_measured` 保留最后一次 raw corrected measurement，重捕获误差字段
用于审计 pose 替换效果。
