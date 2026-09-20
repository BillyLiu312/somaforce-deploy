# HPS 腕部 F/T adapter

deploy adapter 负责标定、机器人运动学、坐标变换、接触估计和 Cross token
构造；sensor SDK 只负责无损采集与 raw SI 单位数据传输。

## Sensor 传输合同

每个传感器提供一路可重连 TCP stream。每个测量值是一行 UTF-8 JSON：

```json
{"schema":"hps6axis.wrench.v1","monotonic_ns":123456789,"sequence":42,"device_id":18174,"status":0,"fx":1.0,"fy":2.0,"fz":3.0,"mx":0.1,"my":0.2,"mz":0.3}
```

- 完整传感器帧解码后立即使用 SDK 主机的 `CLOCK_MONOTONIC` 记录
  `monotonic_ns`。
- `sequence` 在每次 server 进程启动时从 0 开始，每解码一个传感器帧严格加 1。
- force 单位为 N，moment 单位为 Nm，坐标系为传感器原生坐标系。
- `device_id` 必须来自传感器返回值，不能使用 USB 路径代替。HPS 传感器可能返回
  相同 ID，因此左右身份仍以 9000/9001 endpoint 为准。
- `status=0` 表示有效。异常、CRC 错误、不完整或超时帧不能作为有效测量发布。
- 慢 TCP client 不能阻塞采集或其他 client；应丢弃旧 outbound sample，而不是延迟采集。

deploy client 会自动重连。sequence 仅允许在 TCP 重连后重新从 0 开始。旧版
`timestamp_ms` 消息只能通过 `--allow-legacy-sensor-protocol` 用于台架测试，不能
用于 residual 真机 pilot。

## 标定合同

复制 `configs/ft/hps_g1.example.yaml`，填写两侧 device ID 和实测参数后，将
`valid` 设为 `true`。`wrist_from_sensor` 表示 `T_wrist_sensor`：将 sensor
坐标变换到 `left_wrist_yaw_link` 或 `right_wrist_yaw_link`。其中 translation
是从 wrist body 原点指向 sensor 原点、在 wrist frame 表达的向量。

adapter 依次执行 bias 扣除、`measurement_sign`、通过 `r x F` 将力矩平移到
wrist 原点、旋转到 pelvis-yaw frame，并分别使用 100 N 和 10 Nm 归一化。
当前 G1 实测配置已为两侧分别设置手部质量 `0.115 kg`、sensor frame 中 CoM
`[0, 0, 0.076] m`。
residual 真机 pilot 前必须完成并验证重力补偿，或证明上游已移除安装后末端负载
wrench；单一姿态下的 tare 不能覆盖不同手臂姿态。

14 维 token 顺序固定为：

```text
[Fx, Fy, Fz, Mx, My, Mz]：base-yaw frame，已归一化
[wrist linear velocity xyz, wrist angular velocity xyz]：base-yaw frame
[contact_probability, quality]
```

device ID/status 错误、sample stale、超量程或 G1 state stale 时，对应侧的
`quality` 独立置 0。adapter 在所有这些情况下仍以 50 Hz 发布，使传感器故障通过
contact gate 关闭 residual，而不暂停 nominal inference。
当 adapter 进程退出或 frame 超过 100 ms 时，residual receiver 也会生成同样的
零质量 token，而不是阻塞 inference。

## 每次运行的重力感知 bias 标定

SDK 应保持 raw 输出，正常启动时不要反复写入持久化硬件零点。先由安全控制器移动到
固定标定姿态；保持配置中的双手已安装，移除其他所有手腕外载并保持静止，再启动：

```bash
python scripts/run_hps_ft_adapter.py \
  --calibration calibration/hps_g1.yaml \
  --tare-on-start --tare-samples 200 \
  --tare-output outputs/hps_ft_runtime_bias.json
```

未指定 trigger file 时，adapter 会等待操作员按 Enter。自动编排时可传入
`--tare-trigger-file <path>`，只在机器人已到达标定姿态并静止后创建该文件。
adapter 只采集不同的 sensor frame，检查两侧 device ID 与 status；默认 force
标准差超过 1 N 或 moment 标准差超过 0.1 Nm 就拒绝本次标定。启用重力补偿后，
adapter 会读取当前 G1 腕部姿态，在两侧 sensor frame 中预测配置的手部重力 wrench，
并从静态 raw 均值中扣除它；结果是电子 bias，而不是当前姿态下的整体清零值。
进程内 JSON 使用 `somaforce_ft_runtime_bias_v2`，记录 raw 标准差、预测重力 wrench、
质量、CoM 和所得 bias，同时保存计算时使用的 low-state tick、base quaternion、
关节姿态和 wrist rotation。

已知质量/CoM 后，一个姿态可以估计 bias，但不能验证质量/CoM 本身。仍需在多个差异
足够大的腕部姿态下验证；未参与拟合的姿态中，无接触补偿 wrench 应继续接近零。

当前 G1 实测配置设置了 `require_runtime_tare: true`；未提供
`--tare-on-start` 时 adapter 会拒绝从该配置启动。

## 运行

先启动两路 raw SDK server 和现有 G1 low-state bridge。SDK launcher 的参数依次为
左手 ttyUSB 编号、右手 ttyUSB 编号和可选监听地址；脚本固定左手发布到 9000，
右手发布到 9001：

```bash
cd /home/irmv/Workspace/Somaforce/HPS_6axis_SDK
./scripts/start_dual_servers.sh 1 0 0.0.0.0

cd /home/irmv/Workspace/Somaforce/somaforce-deploy
python scripts/run_hps_ft_adapter.py \
  --calibration calibration/hps_g1.yaml \
  --validate-only

python scripts/run_hps_ft_adapter.py \
  --calibration calibration/hps_g1.yaml \
  --left-port 9000 --right-port 9001 \
  --tare-on-start --tare-output outputs/hps_ft_runtime_bias.json

python scripts/check_hps_ft_stream.py \
  --duration 60 --require-both-valid \
  --output outputs/hps_ft_acceptance.npz
```

SDK launcher 的通用形式为：

```text
start_dual_servers.sh <left-ttyUSB-number> <right-ttyUSB-number> [bind-address]
```

adapter 从 5590 读取 low-state，可选从 5555 读取 corrected pelvis pose 以补充
base 平移速度，并在 5580 发布 `ResidualFTFrame`。不使用动捕做 F/T 台架验证时
可增加 `--no-pelvis`；此时 wrist twist 不包含 base translation。

`ResidualFTFrame` v2 同时携带 adapter 的 realtime/monotonic 发布时间，以及左右
两侧的 raw sensor-frame wrench、device/status、SDK source time、raw sequence 和
deploy 主机 TCP 接收 monotonic 时间，并记录坐标变换使用的 G1 kinematics 时间；旧帧
仍可解码。新鲜度判断和 policy 对齐统一使用 deploy 主机的 `CLOCK_MONOTONIC`，
realtime 只用于关联独立日志。只有 SDK 与 deploy adapter 确实共享时钟域时，SDK
source timestamp 才能直接与 deploy 时间相减。
每个 50 Hz policy tick 只选取不超过 100 ms 的最新 F/T frame；超时则输入零质量
token，不会静默插值 wrench。记录中保留 sample age 和有符号的
sample-to-kinematics skew，离线分析可据此使用更严格的同步筛选，而不需要根据数组
行号猜测时间关系。

## Suitcase policy 集成与记录

先启动两路 SDK server，再启动完整实机栈。`--ft` 默认把 Cross residual 设为
`shadow`：residual 会参与推理和记录，但机器人仍接收 nominal action：

```bash
bash scripts/run_suitcase_hardware.sh \
  --armed --nominal-shadow --ft \
  --record-dir /data/somaforce/suitcase_trials
```

输入 `ARM` 后，先用 `i` 到 motion frame 0，保持配置中的双手已安装，移除其他所有
手腕外载，再输入 `t` 并用 `TARE` 确认。左右传感器未通过 rate、age 和 quality 检查前，启动器会拒绝 `s` 或
`p`。`--residual-mode off` 可只记录 F/T 而不计算 Cross；`c1`/`c2` 会实际施加有界
residual，因此必须与 `--nominal-apply` 一起使用，并且只能在检查 shadow 记录后启用。
除非已经把独立验证过的持久电子 bias 写回配置，否则应保留每次运行的重力感知 bias
标定。

每次 policy 尝试会在 `<record-dir>/<run-id>/` 写一个压缩 NPZ。每一行对应一个 policy
tick，包含 realtime/monotonic 双时钟、low-state tick 与各数据流 age、关节位置/速度/
力矩、IMU 姿态与角速度、实时 pelvis/suitcase pose、完整 reference 关节与 body
目标状态、nominal/applied target、全部 nominal observation 与 OOD ratio、F/T token/
wrench/timestamp/sequence、Cross 输入与 history、residual 各级 safety 输出，以及 loop/
inference 时延。metadata 保存 joint/body 顺序、policy hash、完整 F/T calibration
及其 hash、归一化、history 顺序和时钟合同。
如果 watchdog 或操作员中止运行，已完成的行仍会写入 NPZ，并由同名
`.partial.json` 标记记录不完整。运行过程中 runner 会先把原子 chunks 写入
`<record>.recording/manifest.json`；即使进程在最终压缩阶段被终止，launcher 也能从
这些 chunks 重建 NPZ。正常完成时会先确认 controller 已进入 hold，再最终化 NPZ，
避免 proposal watchdog 在保存阶段破坏 archive。

## 启用 residual authority 前的标定步骤

1. 两只传感器充分预热至热稳定，并固定左右设备映射。当前两只设备的 `device_id`
   相同，因此 udev 路径与 9000/9001 映射本身就是标定的一部分。
2. 在无外载条件下，用已知力和力臂分别验证六轴正负方向；检查 N/Nm 单位、符号、
   轴间串扰、status、饱和以及 `measurement_sign`。
3. 分别测量两侧 `T_wrist_sensor`，包括 proper rotation 和 sensor 原点相对
   `wrist_yaw_link` 的位置。用已知作用点加载，联合核对变换后的 force 与 `r x F`
   moment。
4. 预热后估计电子 bias。shadow 试验每次都应在 frame 0 静止姿态执行重力感知 bias
   标定，并保存本次 JSON。
5. 启用 `c1/c2` 前，在至少六个充分分离、无外部接触的手臂姿态下采样。验证并在必要
   时修正实测的 `0.115 kg` 质量与 sensor-frame CoM `[0, 0, 0.076] m`；只有 held-out
   姿态的残余 force/moment 足够小时才算通过。单姿态结果不满足此要求。
6. 用重复静态无接触/已知接触试验确定噪声、漂移、过载、接触阈值和 100 ms freshness
   上限。除非 Cross 训练合同改变，否则保持 100 N/10 Nm 归一化不变。
7. 先完成至少一分钟双流验收，再做 nominal + Cross shadow。授权 residual 前检查
   轴向/符号曲线、F/T age 与 quality 覆盖率、接触时序、target tracking、实测关节
   力矩、loop overrun 和 residual safety clipping。

启动器会强制执行该边界。当前实测文件声明
`validation_scope: runtime_tare_and_residual_shadow`，因此会拒绝 `c1/c2`。完成多姿态
验证后，把 scope 改为 `residual_authority`。两侧 deploy-side gravity compensation
现已启用，应保持 `upstream_distal_load_compensated: false`。每次运行的重力感知 bias
标定与该补偿兼容，并应继续启用以跟踪温漂和电子零偏变化。

## 验收边界

SDK 侧满足以下条件视为交付：

1. 双端口连续运行 1 分钟，无 malformed JSON、重复/倒序 sequence、非有限值或采集停顿。
2. 拔掉任一 USB adapter 不影响另一侧；重新插入后通过新的 TCP connection 恢复，
   deploy adapter 不需要重启。
3. 在目标采样率下，接收间隔 p99 小于两个 sensor period，deploy 输入端不存在超过
   100 ms 的有效帧。
4. 六个已知正负轴向载荷的 raw axis 与符号正确；deploy 变换前单位确认为 N 和 Nm。
5. `status != 0`、CRC failure、timeout 和 overload 均可观测，且不会被静默发布为
   `status=0`。

当前两只传感器都返回 `device_id=18174`，因此无法通过 payload 检测 ttyUSB 参数
是否左右互换。必须保持 `ttyUSB1 -> left/9000`、`ttyUSB0 -> right/9001` 的启动
映射，并在启用 residual authority 前增加持久化 udev symlink。

deploy adapter 的验收条件是：单元测试通过；任一侧 stale/断线后 100 ms 内
`quality=0`，同时 5580 仍保持 50 Hz；已知载荷能复现预期 base-yaw force 和
`r x F` moment；F/T 整体断线不会造成 nominal proposal gap。
