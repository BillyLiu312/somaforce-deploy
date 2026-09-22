# SomaForce 部署

本仓库基于 HDMI `sim2real` commit
`d9e1f700667bc75d8d2eeb5ef74bb2a066600612`，复用官方 G1 I/O、ONNX/TensorRT
推理边界、motion backend 和 MuJoCo sim2sim 环路。

## 运行拓扑

```text
HDMI student -> normalized a_nom[23]；Sonic G1 nominal -> q_target[29]
    -> Cross 力觉 residual -> contact gain 与 authority
    -> safety/watchdog -> Unitree G1 或 MuJoCo
```

离线 replay、MuJoCo 和真机使用同一个 `DeploymentStack` 与 action history，只替换
RobotIO 实现。

## 模式

- `hdmi_student_baseline`：只运行 HDMI student。
- `hdmi_student_residual`：主要 zero-shot 力适应路径。
- `sonic_baseline`：不依赖 HDMI object-state 输入的 Sonic。
- `sonic_residual`：保留该契约；当前 Sonic G1 导出输出 29 个关节目标，因此
  在明确验证 23 维 adapter 之前只支持 nominal。
- `hdmi_student_residual_shadow` 和 `sonic_residual_shadow`：计算 residual，
  但只发送 nominal action。

## 模型绑定

原生 HDMI student 可以导出为单一 deterministic action graph，也可以导出两个 graph：

```text
adapt_ema(policy[249], command[356], object[10]) -> priv_pred[256]
actor_adapt(command[356], policy[249], priv_pred[256]) -> action[23]
```

`HDMIStudentTwoStageNominal` 强制检查上述 shape。Residual 输入为
`wrist_tokens[1,2,16,14]`、`proprio[1,64]`、`a_nom_history[1,23,3]` 和
`previous_a_total[1,23]`，输出 normalized `delta_a[1,23]`。F/T 标定、坐标变换、
接触门控、authority ramp 和物理 action scaling 保持在 ONNX 图外。

Sonic reference 转换必须固定 50 Hz、root-yaw 对齐和 future-step 语义。
push-door 真机脚本已经收敛为 Sonic-only：它把原生 HDMI body/joint motion 转为
any4hdmi qpos tree，并发布受安全控制器保护的 29 维 G1 proposal。所有真机模式
都必须启用并同步记录双腕 F/T，但 F/T 不会进入 Sonic ONNX。仅凭 shape 相等不能证明 Sonic
residual 兼容。

## Push-door-hand 真机入口

真机脚本是 `scripts/run_push_door_hand_hardware.sh`。它沿用 suitcase 脚本的
F/T、G1 bridge、hold/init、shadow 和 guarded-pilot 流程，但不会启动 ROS/VRPN
服务或 marker relay。Sonic 使用 G1 proprioception 与 573 帧 HDMI reference；
F/T adapter 使用 low-state 运动学与 `--no-pelvis`。policy/F-T 记录每 25 帧原子
提交一个 chunk；普通异常和终止信号会自动生成 partial NPZ，`SIGKILL` 或掉电后
可运行 `scripts/finalize_chunked_record.py --output <record>.npz` 恢复。G1/F-T
read-only preflight 已完成实机验证，shadow/apply 仍需分阶段验收。

默认 pilot 保留 safe controller 的 joint-limit clipping 与 `0.08 rad/tick`
target slew limit；`--direct-policy-targets` 仍以注释形式保留，后续验收通过后可恢复。
raw Sonic target 超出 joint limit 超过 `0.05 rad` 或单 tick 跳变超过 `0.50 rad`
时会在发布前 fail closed；controller 出现 `pilot_abort:*` 时 launcher 会立即停止
runner。运行时 F/T 无效只输出 warning，并以 quality-zero 数据保存。原始 HDMI
NPZ 不会被修改；生成的部署 cache 默认使用 2 倍时间插值和 9 帧
Savitzky-Golay 平滑。

## Artifact 校验

checkpoint、私有 F/T 标定和传感器 SDK 不放入 Git。以
`configs/tasks/manifest.example.json` 为模板，替换所有 SHA-256 后运行：

```bash
python scripts/verify_artifacts.py artifacts/<task>/manifest.json
```

缺少必需文件、路径穿越和 hash 不匹配都会 fail closed。

## 部署推进

```text
offline replay -> MuJoCo sim2sim -> nominal 真机
    -> residual shadow -> C0 parity -> low-authority pilot
```

MuJoCo 用于验证导出 parity、history/reset、joint mapping、timing、合成 F/T token 和
safety 行为，不能替代 Isaac C1/C2/C3 交互评估。
