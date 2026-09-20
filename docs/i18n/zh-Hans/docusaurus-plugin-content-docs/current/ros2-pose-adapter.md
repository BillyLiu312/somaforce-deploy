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

## 多刚体 marker 冗余

runtime 可以用多个独立动捕刚体共同确定 torso 或 suitcase 位姿。NOKOV/VRPN
给每个刚体定义的原点和方向不同，因此每一路都必须标定自己的固定
`T_marker_target`；除非三个 marker frame 在物理上完全相同，否则不能共用矩阵。

先测量每个 marker 在目标实体 frame 中的位姿 `T_target_marker`：位置单位为米，
姿态使用目标 frame 下的 XYZ RPY 角度。重复 source 参数生成 v2 标定：

```bash
.venv/bin/python scripts/make_marker_to_policy.py \
  --torso-source robot1 /robot1/pose <x> <y> <z> <roll> <pitch> <yaw> \
  --torso-source robot2 /robot2/pose <x> <y> <z> <roll> <pitch> <yaw> \
  --torso-source robot3 /robot3/pose <x> <y> <z> <roll> <pitch> <yaw> \
  --suitcase-source suitcase1 /suitcase1/pose <x> <y> <z> <roll> <pitch> <yaw> \
  --suitcase-source suitcase2 /suitcase2/pose <x> <y> <z> <roll> <pitch> <yaw> \
  --suitcase-source suitcase3 /suitcase3/pose <x> <y> <z> <roll> <pitch> <yaw> \
  --output calibration/marker_policy_v2.json
```

文件会把每个 topic 及其 `marker_from_target` 矩阵写入
`marker_sources.torso` 或 `marker_sources.suitcase`。运行时每个新鲜 marker
都独立计算
`T_world_target = T_world_mocap * T_mocap_marker * T_marker_target`。

如果每个对象已经有一路 marker 的旧变换可信，可以通过静止时同步采样自动解算
另外几路矩阵，不必逐个手工测量：

```bash
.venv/bin/python scripts/calibrate_redundant_markers.py \
  --anchor-calibration calibration/marker_policy.json \
  --torso-anchor robot1 --suitcase-anchor suitcase1 \
  --torso-source robot1 /robot1/pose \
  --torso-source robot2 /robot2/pose \
  --torso-source robot3 /robot3/pose \
  --suitcase-source suitcase1 /suitcase1/pose \
  --suitcase-source suitcase2 /suitcase2/pose \
  --suitcase-source suitcase3 /suitcase3/pose \
  --samples 240 --output calibration/marker_policy_v2.json
```

标定期间每个对象上的三路刚体必须同时可见。工具会记录固定相对位姿残差；如果
数据不支持刚性固定假设，会把输出标记为 invalid。最终 target frame 仍由 anchor
的旧变换决定，所以 anchor 必须是已经验证过的旧 marker。

只有一路可见时即可继续输出。选定的一致 source 集合对位置取均值、对 SO(3) 旋转
取平均；三路都有效时输出三 marker 均值。三路同时可见时会排除单个位置或角度
离群源。如果多路新鲜数据互相冲突且无法形成一致集合，则停止发布新位姿，最终由
现有 stale watchdog fail closed，而不是任意选择一个错误坐标。参与同一次融合的
数据还必须落在默认 50 ms 同步窗口内，因此被遮挡 marker 会在完整 stale timeout
之前停止影响输出。

选定 source 集合变化时，runtime 在切换首帧保持上一个输出，并在 0.25 秒内逐渐
消除旧均值到新均值之间的局部 SE(3) 对齐偏移。新均值在过渡期间仍跟随实时运动，
因此避免位置或姿态阶跃，同时不会对正常运动施加永久低通延迟。通过
`--marker-source-switch-blend-s` 修改过渡时间。默认一致性阈值为 8 cm 和 12 度；
只能根据实测残差，通过
`--marker-position-consensus-m` 和 `--marker-orientation-consensus-deg` 调整。

suitcase 实机启动器把 `suitcase4` 设为优先源。只要该源仍在 50 ms 同步窗口内，
就直接使用它的标定位姿；该源不可用时，其余 suitcase source 回退到相同的
一致性筛选和均值融合，但实机专用位置阈值放宽为 12 cm，姿态阈值仍为 12 度。
pelvis 融合继续使用默认 8 cm 位置阈值，并且不设置优先源。

如果 suitcase marker set 发生形变或重新安装，可以保留可信的 `suitcase1` 变换作为
anchor，用静止同步样本重算 `suitcase2/3`：

```bash
ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1 \
.venv/bin/python scripts/calibrate_redundant_markers.py \
  --role suitcase \
  --anchor-calibration calibration/marker_policy.json \
  --suitcase-anchor suitcase1 \
  --suitcase-source suitcase1 /suitcase1/pose \
  --suitcase-source suitcase2 /suitcase2/pose \
  --suitcase-source suitcase3 /suitcase3/pose \
  --samples 240 --output /tmp/suitcase_marker_calibration.json
```

新增延长杆 marker 时，应分别用每一路现有可信 marker 作为 anchor 独立求解。只有
每次拟合都通过、并且各 anchor 得到的 `marker_from_target` 彼此一致时才接受。
当前部署的 `suitcase4` 使用 suitcase1、2、3 三份独立解的位置均值和 SO(3) 均值，
记录的交叉验证阈值为 30 mm 和 5 度。fusion source 列表由 calibration 数据驱动；
把 `/suitcase4/pose` 加入 `marker_sources.suitcase` 后，relay 重启时会自动把它纳入
均值、consensus、outlier 排除和 source-switch 缓冲。

两个 relay 使用同一份 v2 文件：

```bash
.venv/bin/python scripts/ros2_pose_to_zmq.py \
  --suitcase-only --transform-json calibration/marker_policy_v2.json
.venv/bin/python scripts/ros2_torso_to_pelvis.py \
  --torso-from-marker-json calibration/marker_policy_v2.json
```

真机启动器会从文件读取 source topic，并分别等待任意一路 torso 与任意一路
suitcase marker。旧 v1 单 marker 文件继续兼容。bring-up 时使用
`bash scripts/run_suitcase_hardware.sh --calibration calibration/marker_policy_v2.json --check-only`
选择 v2 文件。

### Robot marker set 快速标定

实机启动器目前不再加载独立的 `marker_frame_corrections.json` 层。Robot marker
发生变化时，直接把结果标定到主 calibration 文件的 `marker_sources.torso`，避免
runtime 叠加一层参考可能不稳定的 correction。

推荐先由 controller 建立机器人侧的 frame-0 姿态，再把实测 suitcase pose 作为
现场锚点：

```bash
bash scripts/run_suitcase_hardware.sh --armed
# 输入 ARM，再输入 i；init 完成后把 suitcase 摆到 frame 0，然后输入 c
```

G1 保持在 frame-0 init hold 时，`c` 会读取实际融合后的 suitcase pose 和三路
原始 robot marker pose。标定器从
`assets/mujoco/reference/hdmi_suitcase/motion.npz` 第 0 帧取得完整的 suitcase 到
`torso_link` SE(3) 参考，并计算：

```text
T_marker_torso = inverse(T_world_marker_measured)
    * T_world_suitcase_measured
    * T_suitcase_torso_frame0
```

因此使用的是 suitcase 在动捕世界中的实际位置，不会把 robot marker 对齐到写死的
世界坐标。`robot1`、`robot2`、`robot3` 必须全部持续可见，并分别达到完整采样数；
suitcase 锚点至少需要一路有效 marker。更新成功后启动器
会退出，确保下一次运行重新加载新矩阵。如果 G1 已由外部保持在精确 frame-0 姿态，
仍可使用独立的 `--quick-calibrate-robot-markers` 模式；该模式只启动 VRPN，不创建
G1 bridge 或 low-command publisher。

每一路 robot marker-to-torso 变换都使用上述公式和该 marker 的实测 pose 独立求解。
不再使用旧 marker 间相对关系，也不推算缺失 marker。三路中任何一路缺失或不稳定，
整次标定都会失败且不修改文件。成功时原子更新 `calibration/marker_policy.json`，
并先生成带 UTC 时间戳的
`marker_policy.json.before-robot-marker-<UTC>` 备份。标定后先运行 `--check-only`
检查多 marker 一致性，再考虑 armed。

在 armed 交互启动阶段，融合 pelvis 缺失不会直接终止启动器，因为错误的 robot
marker 标定必须保留现场恢复入口。启动器会让机器人保持在安全 hold，并允许依次执行
`i` 和 `c` 完成 frame-0 重新标定。这个例外只适用于首次恢复检查：
`--check-only`、`p` 前置检查和运行时 watchdog 在策略控制前仍严格要求实时 pelvis 流。

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

# 只启动 VRPN 的 frame-0 robot marker 标定；更新选定 calibration。
bash scripts/run_suitcase_hardware.sh --quick-calibrate-robot-markers

# 交互式 zero/hold/init 控制；释放模式前必须手动输入 ARM。
bash scripts/run_suitcase_hardware.sh --armed
```

G1 网卡名不同时增加 `--interface <name>`。脚本会先启动 VRPN，等待两路
tracker topic，再启动 G1 bridge 和 pose relay，最后检查 `5555`、`5561`、
`5590`。armed 启动时，bridge 会保持未武装，直到本地控制器产生一帧带新鲜
时间戳的 hold command，之后才允许释放机器人模式。普通只读和 armed 路径只使用
选定 calibration 文件里的变换；独立 marker correction 文件已经 mute。
armed 交互会话中，即使所有 marker 暂时丢失，pose relay 进程也会保持运行，但不会
发布 stale pose；只有实时一致性结果恢复后才重新发布。每次 `s` 和 `p` 都会重新执行
pelvis/suitcase/low-state 前置检查，policy runner 还会独立执行 250 ms age gate，因此
保持 relay 存活不会放宽 fail-closed 行为。

armed 提示符只接受：

- `z`：使用 HDMI PD gains 的零策略/current-follow target；
- `h`：锁定并保持当前关节位置；
- `i`：用 10 秒平滑插值到 suitcase motion 第 0 帧姿态；
- `c`：用实测 suitcase pose 标定 robot marker，然后退出；
- `q`：回到 zero 模式并停止整套进程。

基础 `--armed` 启动器不加载 suitcase ONNX，也不存在 policy 模式。

下一阶段需要显式启用 nominal shadow：

```bash
bash scripts/run_suitcase_hardware.sh --armed --nominal-shadow
```

先输入 `i`；启动器会等待 10 秒初始化并在完成时明确提示，然后再输入 `s`。加载
ONNX 前，`s` 会执行与 apply 路径相同的一秒 pelvis/suitcase/low-state 前置检查。
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
low-state stale、sequence 逆序，或观测关节速度超过 `18 rad/s` 时，仍会切换为
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
   `pelvis`、`suitcase`、`low_state` 三路数据的一秒预检，再检查新鲜度、init
   误差（`<=0.50 rad`）和物体位置，随后才给出第二道确认。拒绝时终端会直接打印
   具体缺失或 stale 的数据流、pose age，或测量误差与限值，并给出完整日志路径。
5. 输入 `FULL` 后，100% policy 开始闭环稳定控制，但 motion reference 固定在
   frame 0。确认机器人已经依靠策略自身站稳，再完全解除安全绳；这个过程不会
   reset policy。多 marker 是唯一允许的遮挡处理：单个 source 丢失时，只要同一
   对象还有其他已标定 source 能维持有效融合位姿，就可以继续。如果任一对象所有
   source 都 stale，或新鲜 source 无法形成一致集合，relay 会停止发布；超过统一
   stale timeout 后 proposal 进程退出，唯一命令控制器切换到 hold。runner 不会使用
   held pose、绑定 pelvis 的 pose 或 reference 预测 pose 继续 inference。
6. 安全绳完全离开运动范围后输入 `GO`，此时才开始推进 472 帧 reference，约持续
   9.5 秒。稳定等待和动作执行期间都可输入 `h` 并回车来停止 policy control 并
   进入 hold。如果运动不稳，不要等待终端输入，直接使用实体遥控器。动作期间同样
   只接受实时多 marker 融合结果；不再存在 pelvis grace period，也不再存在 suitcase
   attachment fallback。
7. 正常完成后启动器会自动发送 `h`。确认机器人静止且已有支撑后，再输入 `q`。

policy 异常退出或 safe controller 报告 abort 时，启动器会 fail closed。raw policy
记录位于 `nominal_apply_*.npz` 和 `nominal_apply_*.summary.json`；唯一命令控制器
看到的 raw target 与实际 applied target 位于 `nominal_pilot_applied.jsonl`；其余
进程日志位于同一 `outputs/suitcase_hardware/<timestamp>/` 目录。进程正常退出只
能证明传输和异常 gate 完成运行；再次运行前仍需检查平衡、接触、target tracking
和 applied log。
apply 行会增量提交到相邻的 `*.recording/` 目录。policy 进程中断时，launcher 会从
已提交 chunks 重建可读的 partial NPZ，不会暴露半写入的 ZIP archive。
inference NPZ 只记录实时融合得到的 `pelvis_pose` 和 `suitcase_pose`，summary 明确
写入 `pose_source=multi_marker_live_only`；phase 只可能是 `stabilize`、`motion`
或 `frozen`。

需要同步腕部 F/T 输入和论文验证记录时，增加 `--ft` 和
`--record-dir <本地目录>`。`--ft` 会启动 HPS adapter，并默认使用 Cross residual
shadow。执行 `i` 后保留配置中的双手、移除其他手腕外载并保持静止，输入 `t` 并用
`TARE` 确认；F/T preflight
通过前不能启动 `s` 或 `p`。每次尝试会在 `<本地目录>/<run-id>/` 生成按 policy tick
对齐的 NPZ。只记录数据时使用 `--residual-mode off`；只有检查 shadow 结果后，才可
配合 `--nominal-apply` 显式选择 `c1`/`c2`。完整标定步骤和字段说明见
[HPS 腕部 F/T adapter](./hps-ft-adapter.md)。
